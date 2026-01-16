# Camera MoE Integration for GR00T N1d6

PI05 VLA 모델의 멀티 카메라 라우터를 GR00T N1d6에 적용한 구현 문서입니다.

## 📋 목차

1. [개요](#개요)
2. [카메라 구성](#카메라-구성)
3. [아키텍처](#아키텍처)
4. [사용 방법](#사용-방법)
5. [설정 옵션](#설정-옵션)
6. [데이터 형식](#데이터-형식)
7. [예제](#예제)

---

## 개요

GR00T N1d6 모델에 Camera MoE (Mixture of Experts)를 통합하여 멀티 카메라 입력을 효과적으로 처리합니다.

### 주요 특징

- **Router 기반 Gating**: 각 카메라에 대한 가중치를 학습하여 적응적 fusion
- **Soft Gating**: 모든 카메라 정보를 부드럽게 결합
- **Learnable Scaling**: Gating으로 인한 magnitude 감소를 보상
- **Focal Loss**: Hard example에 집중하는 routing loss

---

## 카메라 구성

GR00T N1d6의 카메라 설정:

```
cam1: L515/External camera (외부 카메라, base - 항상 포함)
cam2: Left Wrist camera (왼손 카메라, gated)
cam3: Right Wrist camera (오른손 카메라, gated)
```

### PI05와의 차이점

| 항목 | PI05 | GR00T N1d6 |
|------|------|------------|
| Base Camera | cam2 (Wrist) | cam1 (L515/External) |
| Auxiliary Camera 1 | cam1 (L515) | cam2 (Left Wrist) |
| Auxiliary Camera 2 | cam3 (Thermal) | cam3 (Right Wrist) |
| Router Input | cam2 + prompt + state | cam1 + prompt + state |

---

## 아키텍처

### 전체 구조

```
Input: cam1, cam2, cam3 (multi-camera images)
         ↓
    Eagle Backbone (각 카메라별로 처리)
         ↓
    Camera Features: [batch, seq_len, embed_dim]
         ↓
    CameraRouter(cam1_features, prompt, state) → routing_weights [batch, 2]
         ↓
    Soft Gating: w ⊙ F
         ↓
    Learnable Scaling: γ ⊙ (w ⊙ F)
         ↓
    Feature Fusion: concat(cam1, cam2_gated, cam3_gated) → Projection
         ↓
    Fused Features → Action Head
```

### 수학적 정의

**Soft Gating with Learnable Scaling**:
```
F_gated = γ ⊙ (w ⊙ F_original)
```

- `F_original`: Eagle backbone에서 나온 카메라 features
- `w`: Router가 예측한 gating weight (0-1 범위의 스칼라)
- `γ`: 학습 가능한 scale parameter (채널별 벡터)

**Routing Loss (Focal Loss)**:
```
Loss = -α * (1 - p_t)^γ * log(p_t)
```

- `p_t`: 정답 카메라에 대한 예측 확률
- `α`: 0.25 (positive/negative example 균형)
- `γ`: 2.0 (focusing parameter)

---

## 사용 방법

### 1. Config 설정

`gr00t/configs/model/gr00t_n1d6.py` 또는 YAML config:

```python
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

config = Gr00tN1d6Config(
    # 기존 설정...
    model_name="nvidia/Eagle-Block2A-2B-v2",
    
    # Camera MoE 설정
    use_camera_moe=True,  # Camera MoE 활성화
    camera_router_hidden_dim=512,  # Router MLP hidden dimension
    camera_router_temperature=1.0,  # Softmax temperature
    camera_router_use_gumbel=False,  # Gumbel-Softmax 사용 여부
    camera_router_gumbel_temp=1.0,  # Gumbel temperature
    camera_routing_loss_weight=0.1,  # Routing loss 가중치
    camera_router_use_attention_pooling=True,  # Attention pooling 사용 (권장)
    camera_router_use_learnable_scales=True,  # Learnable scaling 사용 (권장)
)
```

### 2. 모델 초기화

```python
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

model = Gr00tN1d6(config)

# Camera MoE가 초기화되었는지 확인
if model.use_camera_moe:
    print(f"Camera MoE enabled: {model.camera_moe}")
else:
    print("Camera MoE disabled - using single camera mode")
```

### 3. Training

```python
# Multi-camera 입력 준비
inputs = {
    # Camera 1 (Base - L515/External)
    "cam1_pixel_values": cam1_images,  # [batch, channels, height, width]
    "cam1_input_ids": cam1_text_ids,
    "cam1_attention_mask": cam1_attn_mask,
    
    # Camera 2 (Left Wrist)
    "cam2_pixel_values": cam2_images,
    "cam2_input_ids": cam2_text_ids,
    "cam2_attention_mask": cam2_attn_mask,
    
    # Camera 3 (Right Wrist)
    "cam3_pixel_values": cam3_images,
    "cam3_input_ids": cam3_text_ids,
    "cam3_attention_mask": cam3_attn_mask,
    
    # Action inputs
    "state": robot_state,  # [batch, state_dim]
    "action": actions,  # [batch, action_horizon, action_dim]
    "embodiment_id": embodiment_ids,  # [batch]
    "action_mask": action_mask,  # [batch, action_horizon, action_dim]
    
    # Ground truth camera labels (routing loss를 위해)
    "cam2_activate": cam2_labels,  # [batch] binary (1=cam2 should be active)
    "cam3_activate": cam3_labels,  # [batch] binary (1=cam3 should be active)
}

# Forward pass
outputs = model(inputs)

# Loss 확인
total_loss = outputs["loss"]  # action_loss + routing_loss_weight * routing_loss
action_loss = outputs.get("action_loss")
routing_loss = outputs.get("routing_loss")
routing_weights = outputs.get("routing_weights")  # [batch, 2]

print(f"Total Loss: {total_loss.item():.4f}")
print(f"Action Loss: {action_loss.mean().item():.4f}")
if routing_loss is not None:
    print(f"Routing Loss: {routing_loss.item():.4f}")
if routing_weights is not None:
    print(f"Routing Weights (cam2, cam3): {routing_weights[0]}")
```

### 4. Inference

```python
# Multi-camera 입력 (ground truth labels 불필요)
inputs = {
    "cam1_pixel_values": cam1_images,
    "cam1_input_ids": cam1_text_ids,
    "cam1_attention_mask": cam1_attn_mask,
    
    "cam2_pixel_values": cam2_images,
    "cam2_input_ids": cam2_text_ids,
    "cam2_attention_mask": cam2_attn_mask,
    
    "cam3_pixel_values": cam3_images,
    "cam3_input_ids": cam3_text_ids,
    "cam3_attention_mask": cam3_attn_mask,
    
    "state": robot_state,
    "embodiment_id": embodiment_ids,
}

# Generate actions
with torch.no_grad():
    outputs = model.get_action(inputs)

actions = outputs["action"]  # [batch, action_horizon, action_dim]
routing_weights = outputs.get("routing_weights")  # [batch, 2]

print(f"Predicted Actions: {actions[0, 0]}")  # First timestep
if routing_weights is not None:
    print(f"Camera Weights - Left Wrist: {routing_weights[0, 0]:.3f}, Right Wrist: {routing_weights[0, 1]:.3f}")
```

---

## 설정 옵션

### Camera Router Config

```python
@dataclass
class Gr00tN1d6Config:
    # Camera MoE 활성화
    use_camera_moe: bool = False
    
    # Router Architecture
    camera_router_hidden_dim: int = 512
    # Router MLP hidden dimension (256-1024 권장)
    
    camera_router_temperature: float = 1.0
    # Softmax temperature (0.5-2.0)
    # - 낮음: 더 discrete한 선택
    # - 높음: 더 smooth한 blending
    
    # Gumbel-Softmax (선택적)
    camera_router_use_gumbel: bool = False
    # Training 중 discrete sampling을 위해 사용
    
    camera_router_gumbel_temp: float = 1.0
    # Gumbel-Softmax temperature
    
    # Loss Weight
    camera_routing_loss_weight: float = 0.1
    # Routing loss 가중치 (0.05-0.2 권장)
    # - 너무 낮음: Router가 학습되지 않음
    # - 너무 높음: Action prediction 성능 저하
    
    # Advanced Features
    camera_router_use_attention_pooling: bool = True
    # Prompt에 대한 learnable attention pooling (권장)
    
    camera_router_use_learnable_scales: bool = True
    # Gated features에 대한 learnable scaling (권장)
```

### 튜닝 가이드

| 파라미터 | 권장 범위 | 설명 |
|---------|----------|------|
| `camera_router_hidden_dim` | 256-1024 | Router MLP 크기 |
| `camera_router_temperature` | 0.5-2.0 | 작을수록 discrete, 클수록 smooth |
| `camera_routing_loss_weight` | 0.05-0.2 | Routing loss 가중치 |
| `camera_router_use_attention_pooling` | True (권장) | Prompt attention pooling |
| `camera_router_use_learnable_scales` | True (권장) | Learnable magnitude compensation |

---

## 데이터 형식

### Multi-Camera Input Format

각 카메라별로 독립적인 입력을 제공:

```python
# Camera prefix: cam1_, cam2_, cam3_
inputs = {
    # Camera 1 (Base - always required)
    "cam1_pixel_values": torch.Tensor,  # [batch, C, H, W]
    "cam1_input_ids": torch.LongTensor,  # [batch, seq_len]
    "cam1_attention_mask": torch.LongTensor,  # [batch, seq_len]
    
    # Camera 2 (Left Wrist - optional, will be zero-padded if missing)
    "cam2_pixel_values": torch.Tensor,
    "cam2_input_ids": torch.LongTensor,
    "cam2_attention_mask": torch.LongTensor,
    
    # Camera 3 (Right Wrist - optional, will be zero-padded if missing)
    "cam3_pixel_values": torch.Tensor,
    "cam3_input_ids": torch.LongTensor,
    "cam3_attention_mask": torch.LongTensor,
    
    # State and Action
    "state": torch.Tensor,  # [batch, state_dim]
    "action": torch.Tensor,  # [batch, action_horizon, action_dim]
    "embodiment_id": torch.LongTensor,  # [batch]
    "action_mask": torch.Tensor,  # [batch, action_horizon, action_dim]
    
    # Ground Truth Labels (for training only)
    "cam2_activate": torch.LongTensor,  # [batch] - binary (0 or 1)
    "cam3_activate": torch.LongTensor,  # [batch] - binary (0 or 1)
}
```

### Ground Truth Labels

Routing loss를 계산하기 위해 각 샘플에 대해 어떤 카메라가 활성화되어야 하는지 레이블 필요:

```python
# 예시: Left Wrist 카메라가 중요한 경우
cam2_activate = torch.tensor([1, 1, 0, 1])  # batch=4, cam2 active for samples 0,1,3
cam3_activate = torch.tensor([0, 0, 1, 0])  # cam3 active for sample 2
```

**레이블 생성 가이드**:
- Task나 object location에 따라 결정
- 예: "pick up object with left hand" → cam2_activate=1
- 예: "manipulate with right hand" → cam3_activate=1
- 둘 다 0인 경우: Router가 자동으로 선택 (default cam2)

---

## 예제

### Example 1: 기본 사용 (단일 카메라)

Camera MoE를 비활성화하고 기존 방식으로 사용:

```python
config = Gr00tN1d6Config(
    use_camera_moe=False,  # 비활성화
    # 다른 설정...
)

model = Gr00tN1d6(config)

# 기존 입력 형식 그대로 사용
inputs = {
    "pixel_values": images,
    "input_ids": text_ids,
    "attention_mask": attn_mask,
    "state": state,
    "action": action,
    "embodiment_id": embodiment_id,
}

outputs = model(inputs)
```

### Example 2: Multi-Camera Training

```python
config = Gr00tN1d6Config(
    use_camera_moe=True,
    camera_router_hidden_dim=512,
    camera_routing_loss_weight=0.1,
    camera_router_use_attention_pooling=True,
    camera_router_use_learnable_scales=True,
)

model = Gr00tN1d6(config)

# Training loop
for batch in dataloader:
    inputs = {
        "cam1_pixel_values": batch["cam1_images"],
        "cam1_input_ids": batch["cam1_text"],
        "cam1_attention_mask": batch["cam1_mask"],
        
        "cam2_pixel_values": batch["cam2_images"],
        "cam2_input_ids": batch["cam2_text"],
        "cam2_attention_mask": batch["cam2_mask"],
        
        "cam3_pixel_values": batch["cam3_images"],
        "cam3_input_ids": batch["cam3_text"],
        "cam3_attention_mask": batch["cam3_mask"],
        
        "state": batch["state"],
        "action": batch["action"],
        "embodiment_id": batch["embodiment_id"],
        "action_mask": batch["action_mask"],
        
        "cam2_activate": batch["cam2_label"],
        "cam3_activate": batch["cam3_label"],
    }
    
    outputs = model(inputs)
    loss = outputs["loss"]
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Log routing weights
    if "routing_weights" in outputs:
        routing_weights = outputs["routing_weights"]
        print(f"Routing: Cam2={routing_weights[:, 0].mean():.3f}, Cam3={routing_weights[:, 1].mean():.3f}")
```

### Example 3: Inference with Router Analysis

```python
model.eval()

with torch.no_grad():
    outputs = model.get_action(inputs)
    
    actions = outputs["action"]
    routing_weights = outputs["routing_weights"]
    
    # Analyze router decisions
    cam2_weight = routing_weights[:, 0]
    cam3_weight = routing_weights[:, 1]
    
    print(f"Left Wrist weight: {cam2_weight.mean():.3f} ± {cam2_weight.std():.3f}")
    print(f"Right Wrist weight: {cam3_weight.mean():.3f} ± {cam3_weight.std():.3f}")
    
    # Identify which camera is more important
    dominant_camera = torch.argmax(routing_weights, dim=-1)
    print(f"Dominant camera: {'Left Wrist' if dominant_camera[0] == 0 else 'Right Wrist'}")
```

---

## Backward Compatibility

Camera MoE를 비활성화하면 기존 단일 카메라 모드로 작동:

```python
# Old code - still works
config = Gr00tN1d6Config(use_camera_moe=False)
model = Gr00tN1d6(config)

# Use single camera input as before
outputs = model(single_camera_inputs)
```

---

## Troubleshooting

### Issue 1: "Camera MoE enabled but no multi-camera inputs found"

**원인**: Camera MoE가 활성화되었지만 입력에 `cam1_*`, `cam2_*`, `cam3_*` prefix가 없음

**해결**:
1. 입력 데이터에 camera prefix 추가
2. 또는 `use_camera_moe=False`로 설정

### Issue 2: Routing loss가 감소하지 않음

**원인**: 
- Ground truth labels가 부정확
- `camera_routing_loss_weight`가 너무 작음

**해결**:
1. `cam2_activate`, `cam3_activate` 레이블 확인
2. `camera_routing_loss_weight`를 0.1 → 0.2로 증가

### Issue 3: Action prediction 성능이 저하됨

**원인**: Routing loss가 너무 강함

**해결**:
- `camera_routing_loss_weight`를 0.1 → 0.05로 감소

---

## FAQ

### Q1: 2개의 카메라만 있는 경우는?

**A**: 없는 카메라는 자동으로 zero-padding됩니다. 예를 들어, cam2만 있고 cam3가 없으면:

```python
inputs = {
    "cam1_pixel_values": cam1_images,
    "cam1_input_ids": cam1_text,
    "cam1_attention_mask": cam1_mask,
    
    "cam2_pixel_values": cam2_images,
    "cam2_input_ids": cam2_text,
    "cam2_attention_mask": cam2_mask,
    
    # cam3는 생략 - 자동으로 zero padding
}
```

### Q2: Ground truth labels 없이 training 가능한가요?

**A**: 가능하지만 권장하지 않습니다. Labels 없이는 routing loss가 계산되지 않고, router가 학습되지 않습니다. 대신:

1. Pseudo-labels 사용 (예: heuristic 기반)
2. Unsupervised routing (구현 필요)

### Q3: 어떤 카메라를 base(cam1)로 선택해야 하나요?

**A**: 
- **항상 사용 가능한 카메라**: 모든 데이터에 존재
- **넓은 시야각**: 전체 scene을 볼 수 있는 카메라
- **안정적인 위치**: 로봇 움직임에 영향받지 않는 고정 카메라

GR00T의 경우 L515/External이 이 조건을 만족합니다.

---

## References

- Original PI05 Camera MoE Documentation: `CAMERA_MOE_ARCHITECTURE.md`
- GR00T N1.6 Paper: [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection"

---

**Last Updated**: January 15, 2026  
**Author**: GR00T Team
