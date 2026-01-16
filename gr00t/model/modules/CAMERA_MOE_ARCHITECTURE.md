# Camera MoE (Mixture of Experts) 아키텍처 문서

PI05 VLA 모델의 멀티 카메라 입력 처리를 위한 MoE 구현 설명서

## 📋 목차

1. [전체 구조 개요](#전체-구조-개요)
2. [라우터 (CameraRouter) 구현](#라우터-camerarouter-구현)
3. [Camera MoE (CameraMoE) 구현](#camera-moe-cameramoe-구현)
4. [Pi0 모델에서의 통합](#pi0-모델에서의-통합)
5. [설계 철학](#설계-철학)

---

## 전체 구조 개요

이 코드는 멀티 카메라 입력을 위한 **라우터 기반 게이팅(Router-based Gating)** 방식의 MoE를 구현합니다. 전통적인 expert network가 아니라, **router가 각 카메라 feature에 gating weight를 적용**하는 방식입니다.

### 주요 컴포넌트

```
Input Cameras (cam1, cam2, cam3)
         ↓
    ViT Encoding
         ↓
    CameraRouter → routing_weights [batch, 2]
         ↓
    Soft Gating (w ⊙ F)
         ↓
   Feature Fusion
         ↓
    Output Tokens
```

### 카메라 매핑

- **cam1**: L515/External camera (외부 카메라)
- **cam2**: Wrist camera (손목 카메라, base)
- **cam3**: Thermal/Right_Wrist camera (열화상 카메라)

---

## 라우터 (CameraRouter) 구현

**파일**: `camera_router.py`

### 핵심 설계

라우터는 다양한 입력 소스로부터 카메라 선택 결정을 학습합니다.

#### 입력 타입 (RouterInputType)

```python
class RouterInputType(str, Enum):
    PROMPT = "prompt"                    # cam2 + prompt
    STATE = "state"                      # cam2 + state
    PROMPT_STATE = "prompt_state"        # cam2 + prompt + state
```

### 주요 구성요소

#### 1. Learnable Attention Pooling

단순 평균 풀링 대신 학습 가능한 어텐션을 사용하여 prompt의 중요한 단어를 식별합니다.

```python
def _attention_pool(self, tokens):
    """
    Prompt에서 카메라 선택에 중요한 단어를 학습
    예: "apple"이나 "plate" 같은 키워드에 집중
    """
    # Query 학습
    query = self.prompt_attention_query(...)
    
    # Attention 계산: Q·K^T / sqrt(d)
    attention_scores = jnp.sum(query * tokens, axis=-1) / scale
    attention_weights = softmax(attention_scores)
    
    # Weighted sum
    pooled = jnp.sum(tokens * attention_weights[:, :, None], axis=1)
    return pooled
```

**장점**:
- 태스크와 관련된 prompt 키워드에 집중
- 단순 평균 풀링보다 정보 손실 감소

#### 2. Router MLP 구조

```
Input Features → Linear1(512) → ReLU → Linear2(256) → ReLU → Linear3(2) → Logits
```

**입력 차원 계산**:
- `PROMPT`: `embed_dim + embed_dim` (cam2 + prompt)
- `STATE`: `embed_dim + state_dim` (cam2 + state)
- `PROMPT_STATE`: `embed_dim + embed_dim + state_dim` (cam2 + prompt + state)

#### 3. Routing Weight 계산

```python
def __call__(self, cam1_tokens, cam2_tokens, prompt_tokens, state, train, rng):
    # 1. 설정된 input_type에 따라 features 수집
    features = []
    if needs_cam2:
        cam2_pooled = jnp.mean(cam2_tokens, axis=1)
        features.append(cam2_pooled)
    if needs_prompt:
        prompt_pooled = self._attention_pool(prompt_tokens)  # Learnable pooling
        features.append(prompt_pooled)
    if needs_state:
        features.append(state)
    
    # 2. Feature concatenation
    combined_features = jnp.concatenate(features, axis=-1)
    
    # 3. MLP를 통한 logits 계산
    x = nnx.relu(self.router_linear1(combined_features))
    x = nnx.relu(self.router_linear2(x))
    logits = self.router_linear3(x)  # [batch, num_experts=2]
    
    # 4. Temperature scaling
    logits = logits / self.temperature
    
    # 5. Softmax 또는 Gumbel-Softmax
    if train and self.use_gumbel:
        routing_weights = self._gumbel_softmax(logits, rng, temperature=self.gumbel_temp)
    else:
        routing_weights = jax.nn.softmax(logits, axis=-1)
    
    return routing_weights  # [batch, 2] where [:, 0]=L515, [:, 1]=Thermal
```

#### 4. Focal Loss for Routing

Hard example에 집중하는 Focal Loss를 사용하여 router를 학습합니다.

```python
def compute_routing_loss(self, routing_weights, target_camera_idx, alpha=0.25, gamma=2.0):
    """
    Focal Loss = -α * (1 - p_t)^γ * log(p_t)
    
    Args:
        routing_weights: [batch, 2] 예측된 카메라 gating weights
        target_camera_idx: [batch] ground truth (0=L515, 1=Thermal)
        alpha: 0.25 (positive/negative example 균형)
        gamma: 2.0 (focusing parameter, 클수록 hard example에 집중)
    """
    # Target camera probability
    p_t = jnp.take_along_axis(routing_weights, target_camera_idx[:, None], axis=1)
    p_t = jnp.clip(p_t, 1e-10, 1.0 - 1e-10)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = jnp.power(1.0 - p_t, gamma)
    
    # Focal Loss
    loss = -alpha * focal_weight * jnp.log(p_t)
    return jnp.mean(loss)
```

**Focal Loss의 장점**:
- Well-classified example은 down-weight (쉬운 예제 무시)
- Misclassified example에 집중 (어려운 예제 강조)
- `gamma=0`이면 standard cross-entropy로 환원

---

## Camera MoE (CameraMoE) 구현

**파일**: `camera_router.py`

### 핵심: Soft Gating with Learnable Scaling

전통적인 expert network를 사용하지 않고, **직접 feature gating**을 수행합니다.

### 수학적 정의

```
F_gated = γ ⊙ (w ⊙ F_original)
```

**변수 설명**:
- `F_original`: ViT에서 나온 원본 카메라 features `[batch, seq_len, embed_dim]`
- `w`: Router가 예측한 gating weight (스칼라, 0-1 범위)
- `γ`: 학습 가능한 scale parameter (채널별 벡터 `[embed_dim]`)
- `⊙`: Element-wise multiplication

### 구현 단계

#### Step 1: Router 호출

```python
routing_weights = self.router(
    cam1_tokens=cam1_tokens,      # L515 features [batch, seq_len, embed_dim]
    cam2_tokens=cam2_tokens,      # Wrist features (base)
    prompt_tokens=prompt_tokens,  # Prompt embeddings
    state=state,                  # Robot state vector
    train=train,
    rng=rng,
)
# 출력: [batch, 2] where [:, 0]=w_L515, [:, 1]=w_Thermal
```

#### Step 2: Soft Gating 적용

```python
# Routing weights 추출
w_cam1 = routing_weights[:, 0:1, None]  # [batch, 1, 1]
w_cam3 = routing_weights[:, 1:2, None]  # [batch, 1, 1]

if self.use_learnable_scales:
    # Two-step gating with learnable scaling
    
    # Step 1: Router gating (w ⊙ F)
    cam1_weighted = cam1_tokens * w_cam1  # [batch, seq_len, embed_dim]
    cam3_weighted = cam3_tokens * w_cam3
    
    # Step 2: Learnable magnitude compensation (γ ⊙ (w ⊙ F))
    cam1_gated = cam1_weighted * self.scale_cam1.value  # [batch, seq_len, embed_dim]
    cam3_gated = cam3_weighted * self.scale_cam3.value
else:
    # 단순 gating (구버전 호환용)
    cam1_gated = cam1_tokens * w_cam1
    cam3_gated = cam3_tokens * w_cam3
```

**Learnable Scaling의 역할**:
- Gating으로 인한 magnitude 감소를 보상
- 각 채널별로 학습 가능한 scale 적용
- 초기값은 1.0 (identity transformation)

#### Step 3: Feature Fusion

```python
# 모든 카메라 feature 결합
concatenated = jnp.concatenate([
    cam2_tokens,   # Wrist (base, always included)
    cam1_gated,    # L515 (gated)
    cam3_gated     # Thermal (gated)
], axis=-1)
# Shape: [batch, seq_len, 3 * embed_dim]

# Projection layer로 원래 차원으로 복원
output_tokens = self.projection(concatenated)
# Shape: [batch, seq_len, embed_dim]

return output_tokens, routing_weights
```

### 전체 Forward Pass

```python
def __call__(
    self,
    cam2_tokens,      # Wrist camera (base) [batch, seq_len, embed_dim]
    cam1_tokens,      # L515/External camera [batch, seq_len, embed_dim]
    cam3_tokens,      # Thermal camera [batch, seq_len, embed_dim]
    prompt_tokens,    # Prompt [batch, prompt_len, embed_dim]
    state,            # Robot state [batch, state_dim]
    train,
    rng,
) -> tuple[output_tokens, routing_weights]:
    """
    Returns:
        output_tokens: Fused camera features [batch, seq_len, embed_dim]
        routing_weights: Camera gating weights [batch, 2]
    """
    # 1. Router: 카메라별 gating weight 계산
    routing_weights = self.router(...)
    
    # 2. Missing camera 처리 (zero padding)
    if cam1_tokens is None: cam1_tokens = jnp.zeros_like(cam2_tokens)
    if cam3_tokens is None: cam3_tokens = jnp.zeros_like(cam2_tokens)
    
    # 3. Soft gating 적용
    w_cam1 = routing_weights[:, 0:1, None]
    w_cam3 = routing_weights[:, 1:2, None]
    cam1_gated = cam1_tokens * w_cam1 * self.scale_cam1
    cam3_gated = cam3_tokens * w_cam3 * self.scale_cam3
    
    # 4. Feature fusion
    concatenated = jnp.concatenate([cam2_tokens, cam1_gated, cam3_gated], axis=-1)
    output_tokens = self.projection(concatenated)
    
    return output_tokens, routing_weights
```

### Routing Loss 계산

```python
def compute_routing_loss(self, routing_weights, cam1_activate, cam3_activate):
    """
    Ground truth labels를 기반으로 routing loss 계산
    
    Args:
        routing_weights: [batch, 2] 예측된 gating weights
        cam1_activate: [batch] binary (1=L515 활성화)
        cam3_activate: [batch] binary (1=Thermal 활성화)
    """
    # Label을 target camera index로 변환
    target_camera_idx = jnp.where(
        cam1_activate == 1,
        0,  # L515
        jnp.where(cam3_activate == 1, 1, 0)  # Thermal or default L515
    )
    
    return self.router.compute_routing_loss(routing_weights, target_camera_idx)
```

---

## Pi0 모델에서의 통합

**파일**: `pi0.py`

### 초기화

```python
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # ... PaliGemma, ViT 초기화 ...
        
        # Camera MoE 초기화
        if config.use_camera_moe:
            from openpi.models.camera_router import CameraMoE, CameraRouterConfig
            
            router_config = CameraRouterConfig(
                embed_dim=paligemma_config.width,  # 2048
                state_dim=config.state_dim,         # 32
                router_input_type=config.camera_router_input_type,  # "prompt", "state", etc.
                use_attention_pooling=config.camera_router_use_attention_pooling,
                use_learnable_scales=config.camera_router_use_learnable_scales,
                num_experts=2,  # cam2 vs cam3 선택
                router_hidden_dim=config.camera_router_hidden_dim,
                router_temperature=config.camera_router_temperature,
                use_gumbel_softmax=config.camera_router_use_gumbel,
                gumbel_temperature=config.camera_router_gumbel_temp,
            )
            self.camera_moe = CameraMoE(router_config, rngs)
            self.routing_loss_weight = config.camera_routing_loss_weight
```

### 사용 예시

```python
def embed_prefix(self, obs, rng, router_weights=None):
    """Prefix embedding with Camera MoE"""
    
    if self.use_camera_moe and self.camera_moe is not None:
        # 1. 각 카메라별로 ViT encoding
        cam1_tokens = None
        cam2_tokens = None
        cam3_tokens = None
        
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            
            # 카메라 이름에서 cam1/cam2/cam3 매핑
            if "base" in name or name == "image":
                cam1_tokens = image_tokens
            elif "wrist" in name and "right" not in name:
                cam2_tokens = image_tokens
            elif "right_wrist" in name or "thermal" in name:
                cam3_tokens = image_tokens
        
        # 2. Prompt embedding
        prompt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        
        # 3. Camera MoE를 통한 multi-camera fusion
        fused_tokens, routing_weights = self.camera_moe(
            cam2_tokens,          # Base camera (wrist)
            cam1_tokens,          # L515 (external)
            cam3_tokens,          # Thermal (right wrist)
            prompt_tokens=prompt_tokens,
            state=obs.state,      # Robot state
            train=not self.deterministic,
            rng=rng,
        )
        
        # 4. Fused tokens를 token list에 추가
        tokens.append(fused_tokens)
        tokens.append(prompt_tokens)
        
        return tokens, routing_weights
```

### Configuration 예시

```python
# pi0_config.py
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # Camera Router/MoE configuration
    use_camera_moe: bool = False
    camera_router_hidden_dim: int = 512
    camera_router_temperature: float = 1.0
    camera_router_use_gumbel: bool = False
    camera_router_gumbel_temp: float = 1.0
    camera_routing_loss_weight: float = 0.1
    camera_router_input_type: str = "prompt"  # "prompt", "state", "prompt_state"
    camera_router_use_attention_pooling: bool = False
    camera_router_use_learnable_scales: bool = False
```

---

## 설계 철학

### 1. Expert Network를 사용하지 않는 이유

**전통적인 MoE**:
```
Router → Expert1, Expert2, ... → Weighted combination
```

**본 구현 (Direct Gating)**:
```
Router → Gating weights → Direct feature scaling
```

**이유**:
- ViT features가 이미 충분히 semantic함
- Expert network 추가는 불필요한 복잡도 증가
- 직접 gating이 더 효율적이고 해석 가능함

### 2. Soft Gating의 장점

**Hard Gating (discrete selection)**:
```
if w_L515 > w_Thermal:
    output = cam1_features  # L515만 사용
else:
    output = cam3_features  # Thermal만 사용
```

**Soft Gating (continuous weighting)**:
```
output = 0.7 * cam1_features + 0.3 * cam3_features
```

**장점**:
1. **Train-test 일관성**: Training과 inference에서 동일한 동작
2. **부드러운 fusion**: 카메라 간 smooth transition
3. **정보 보존**: 모든 카메라 정보를 일부라도 활용
4. **Gradient flow**: 모든 카메라에 대해 gradient 전파 가능

### 3. Learnable Scaling의 필요성

**문제**: Gating으로 인한 magnitude 감소

```python
# w < 1.0이면 feature magnitude 감소
F_gated = w * F_original  # magnitude가 줄어듦
```

**해결**: Learnable scale parameter

```python
# γ로 magnitude 보상
F_gated = γ * w * F_original
# γ는 학습을 통해 최적 scale 찾음
```

**효과**:
- Feature magnitude 유지
- 다운스트림 layer에 안정적인 입력 제공
- 성능 향상

### 4. Focal Loss 사용 이유

**Standard Cross-Entropy**:
```
Loss = -log(p_t)
# 모든 example을 동등하게 처리
```

**Focal Loss**:
```
Loss = -(1 - p_t)^γ * log(p_t)
# Hard example에 집중
```

**효과**:
- Well-classified example (p_t ≈ 1) → Loss ≈ 0
- Misclassified example (p_t ≈ 0) → Loss 크게 증가
- Router가 어려운 case 학습에 집중

### 5. 카메라 선택 전략

**Base Camera (cam2 - Wrist)**:
- 항상 포함 (gating 없음)
- 손목 위치에서의 관점 제공
- 로봇의 end-effector 근처 정보

**Auxiliary Cameras (cam1, cam3)**:
- Router가 선택적으로 가중치 부여
- cam1 (L515): 외부 관점, 넓은 시야
- cam3 (Thermal): 온도 정보, 특수 센서

**선택 기준**:
- Prompt 내용 (예: "hot", "cold" → Thermal 선호)
- Robot state (특정 자세에서 특정 카메라 유용)
- Task context

---

## 사용 예시

### Training

```python
# Forward pass
fused_tokens, routing_weights = model.camera_moe(
    cam2_tokens, cam1_tokens, cam3_tokens,
    prompt_tokens=prompt_tokens,
    state=robot_state,
    train=True,
    rng=rng_key,
)

# Compute losses
action_loss = compute_action_loss(...)

# Routing loss (ground truth camera labels 필요)
routing_loss = model.camera_moe.compute_routing_loss(
    routing_weights,
    cam1_activate=ground_truth_cam1,  # [batch] binary
    cam3_activate=ground_truth_cam3,  # [batch] binary
)

# Total loss
total_loss = action_loss + 0.1 * routing_loss
```

### Inference

```python
# Router가 자동으로 카메라 선택
fused_tokens, routing_weights = model.camera_moe(
    cam2_tokens, cam1_tokens, cam3_tokens,
    prompt_tokens=prompt_tokens,
    state=robot_state,
    train=False,
)

# routing_weights 확인
print(f"L515 weight: {routing_weights[0, 0]:.3f}")
print(f"Thermal weight: {routing_weights[0, 1]:.3f}")
```

---

## 파라미터 튜닝 가이드

### Router Configuration

```python
CameraRouterConfig(
    embed_dim=2048,              # PaliGemma의 embedding dimension
    state_dim=32,                # Robot state vector size
    router_input_type="prompt",  # Router가 사용할 입력 선택
    use_attention_pooling=True,  # Prompt attention pooling 사용 (권장)
    use_learnable_scales=True,   # Learnable scaling 사용 (권장)
    num_experts=2,               # 카메라 개수
    router_hidden_dim=512,       # Router MLP hidden dimension
    router_temperature=1.0,      # Softmax temperature (낮을수록 discrete)
    use_gumbel_softmax=False,    # Gumbel-Softmax 사용 여부
    gumbel_temperature=1.0,      # Gumbel temperature
)
```

**튜닝 팁**:
- `router_temperature`: 0.5-2.0 범위에서 실험
  - 낮음 → 더 discrete한 선택
  - 높음 → 더 smooth한 blending
- `router_hidden_dim`: 256-1024
  - 작음 → 빠르지만 표현력 낮음
  - 큼 → 느리지만 표현력 높음
- `use_attention_pooling=True`: 대부분의 경우 성능 향상
- `use_learnable_scales=True`: 성능 향상에 필수

### Training Hyperparameters

```python
camera_routing_loss_weight=0.1  # Routing loss 가중치 (0.05-0.2 범위)
```

**조정 기준**:
- 너무 낮음 → Router가 학습되지 않음
- 너무 높음 → Action prediction 성능 저하
- 시작: 0.1, 필요시 조정

---

## 참고 자료

- **Original MoE Paper**: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection"
- **Gumbel-Softmax**: Jang et al., "Categorical Reparameterization with Gumbel-Softmax"
- **PaliGemma**: Google Research, "PaliGemma: A versatile 3B VLM for transfer"

---

## FAQ

### Q1: 왜 3개 카메라를 모두 fusion하나요?

**A**: cam2 (wrist)를 base로 사용하고, cam1 (L515)과 cam3 (Thermal)을 선택적으로 추가합니다. 이렇게 하면:
- Wrist 관점은 항상 유지
- 추가 정보는 필요할 때만 활용
- 불필요한 정보로 인한 noise 감소

### Q2: Soft gating vs Hard gating 중 어느 것이 좋나요?

**A**: **Soft gating 권장**
- Train-test consistency 보장
- 모든 카메라 정보 활용 가능
- Gradient가 모든 카메라에 흐름
- Production에서 더 안정적

### Q3: Expert network 없이 어떻게 작동하나요?

**A**: ViT features가 이미 충분히 semantic하므로:
- Direct gating만으로 효과적
- Expert network는 불필요한 복잡도
- 더 빠르고 해석 가능

### Q4: Learnable scaling이 정말 필요한가요?

**A**: **네, 권장합니다**
- Gating으로 인한 magnitude 감소 보상
- 실험 결과 성능 향상 확인
- 계산 비용은 미미함 (channel-wise multiplication)

### Q5: Router input type을 어떻게 선택하나요?

**A**:
- `"prompt"`: Task description이 카메라 선택에 중요한 경우
- `"state"`: Robot 자세/위치가 카메라 선택에 중요한 경우
- `"prompt_state"`: 둘 다 중요한 경우 (권장)

---

## 버전 호환성

### Backward Compatibility

구버전 checkpoint 로딩을 위한 옵션:

```python
# 옵션 1: Attention pooling 없이 (구버전)
use_attention_pooling=False  # Mean pooling 사용

# 옵션 2: Learnable scales 없이 (구버전)
use_learnable_scales=False   # Simple gating만 사용
```

### Migration Guide

구버전에서 신버전으로 마이그레이션:

```python
# 1. 먼저 기존 설정으로 로드
config = Pi0Config(
    use_camera_moe=True,
    camera_router_use_attention_pooling=False,
    camera_router_use_learnable_scales=False,
)

# 2. Checkpoint 로드
model = config.create(rng_key)
model = load_checkpoint(model, old_checkpoint_path)

# 3. 새로운 기능 활성화 후 fine-tuning
config = Pi0Config(
    use_camera_moe=True,
    camera_router_use_attention_pooling=True,   # ✓
    camera_router_use_learnable_scales=True,    # ✓
)
model = config.create(rng_key)
# 이전 weight 로드 후 새로운 파라미터만 학습
```

---

**Last Updated**: January 14, 2026  
**Maintained by**: OpenPI Team
