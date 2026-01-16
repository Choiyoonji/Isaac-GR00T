# Camera MoE Implementation for GR00T N1d6

PI05 VLA 모델의 멀티 카메라 라우터를 GR00T N1d6에 성공적으로 적용했습니다.

## 📦 구현 파일

### 1. Camera Router 모듈
- **위치**: `gr00t/model/modules/camera_router.py`
- **내용**:
  - `CameraRouterConfig`: Router 설정
  - `CameraRouter`: 라우터 네트워크 (cam1 + prompt + state → routing weights)
  - `CameraMoE`: Multi-camera fusion with soft gating

### 2. Config 업데이트
- **위치**: `gr00t/configs/model/gr00t_n1d6.py`
- **추가된 설정**:
  ```python
  use_camera_moe: bool = False
  camera_router_hidden_dim: int = 512
  camera_router_temperature: float = 1.0
  camera_router_use_gumbel: bool = False
  camera_router_gumbel_temp: float = 1.0
  camera_routing_loss_weight: float = 0.1
  camera_router_use_attention_pooling: bool = True
  camera_router_use_learnable_scales: bool = True
  ```

### 3. Model 통합
- **위치**: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`
- **변경사항**:
  - `Gr00tN1d6.__init__()`: Camera MoE 초기화
  - `forward()`: Multi-camera 지원 및 routing loss 계산
  - `get_action()`: Inference에서 multi-camera 지원
  - `_forward_with_camera_moe()`: Multi-camera backbone 처리

### 4. 문서
- **사용 가이드**: `gr00t/model/modules/CAMERA_MOE_GR00T_N1D6.md`
- **테스트 스크립트**: `test_camera_moe_integration.py`

---

## 🔧 주요 차이점: PI05 vs GR00T

| 항목 | PI05 | GR00T N1d6 |
|------|------|------------|
| Base Camera | cam2 (Wrist) | cam1 (L515/External) |
| Auxiliary 1 | cam1 (L515) | cam2 (Left Wrist) |
| Auxiliary 2 | cam3 (Thermal) | cam3 (Right Wrist) |
| Router Input | cam2 + prompt + state | cam1 + prompt + state |
| Backbone | PaliGemma | Eagle-Block2A-2B-v2 |

---

## 🚀 빠른 시작

### 1. Config 설정

```python
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

config = Gr00tN1d6Config(
    model_name="nvidia/Eagle-Block2A-2B-v2",
    use_camera_moe=True,  # Camera MoE 활성화
    camera_router_hidden_dim=512,
    camera_routing_loss_weight=0.1,
    camera_router_use_attention_pooling=True,
    camera_router_use_learnable_scales=True,
)
```

### 2. 모델 생성

```python
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

model = Gr00tN1d6(config)
```

### 3. Multi-Camera 입력 형식

```python
inputs = {
    # Camera 1 (Base - L515/External, 필수)
    "cam1_pixel_values": cam1_images,
    "cam1_input_ids": cam1_text,
    "cam1_attention_mask": cam1_mask,
    
    # Camera 2 (Left Wrist, 선택)
    "cam2_pixel_values": cam2_images,
    "cam2_input_ids": cam2_text,
    "cam2_attention_mask": cam2_mask,
    
    # Camera 3 (Right Wrist, 선택)
    "cam3_pixel_values": cam3_images,
    "cam3_input_ids": cam3_text,
    "cam3_attention_mask": cam3_mask,
    
    # State and Action
    "state": robot_state,
    "action": actions,
    "embodiment_id": embodiment_ids,
    "action_mask": action_mask,
    
    # Ground truth labels (training only)
    "cam2_activate": cam2_labels,  # [batch] binary
    "cam3_activate": cam3_labels,  # [batch] binary
}
```

### 4. Training

```python
outputs = model(inputs)

total_loss = outputs["loss"]  # action_loss + routing_loss
routing_weights = outputs["routing_weights"]  # [batch, 2]

print(f"Routing: Cam2={routing_weights[:, 0].mean():.3f}, "
      f"Cam3={routing_weights[:, 1].mean():.3f}")
```

### 5. Inference

```python
with torch.no_grad():
    outputs = model.get_action(inputs)
    actions = outputs["action"]
    routing_weights = outputs["routing_weights"]
```

---

## ✅ 테스트 결과

```bash
cd /home/choiyj/Isaac-GR00T
source .venv/bin/activate
python test_camera_moe_integration.py
```

**결과**:
```
✓ All tests passed successfully!

Summary:
  1. CameraRouterConfig ✓
  2. CameraMoE module ✓
  3. GR00T N1d6 integration ✓
  4. Backward compatibility ✓
  5. Multi-camera input format ✓
```

---

## 🔄 Backward Compatibility

Camera MoE를 비활성화하면 기존 코드가 그대로 작동합니다:

```python
config = Gr00tN1d6Config(
    use_camera_moe=False,  # 기본값
)

model = Gr00tN1d6(config)

# 기존 단일 카메라 입력 사용
outputs = model(single_camera_inputs)
```

---

## 📚 추가 문서

- **상세 사용 가이드**: [CAMERA_MOE_GR00T_N1D6.md](gr00t/model/modules/CAMERA_MOE_GR00T_N1D6.md)
- **원본 PI05 문서**: [CAMERA_MOE_ARCHITECTURE.md](gr00t/model/modules/CAMERA_MOE_ARCHITECTURE.md)

---

## 🎯 다음 단계

1. **Pretrained Eagle weights 로드** 후 full model test
2. **Real data로 training** 및 routing loss 모니터링
3. **Multi-camera task에서 evaluation**
4. **Hyperparameter tuning** (router_hidden_dim, routing_loss_weight 등)

---

## 📝 구현 상세

### Router Architecture

```
Input: cam1_features (pooled) + prompt (attention-pooled) + state
  ↓
Linear(embed_dim*2 + state_dim → 512) + ReLU
  ↓
Linear(512 → 256) + ReLU
  ↓
Linear(256 → 2) → Logits
  ↓
Softmax / Gumbel-Softmax → Routing Weights [batch, 2]
```

### Feature Fusion

```
cam1_tokens (always included)
cam2_gated = γ_cam2 ⊙ (w_cam2 ⊙ cam2_tokens)
cam3_gated = γ_cam3 ⊙ (w_cam3 ⊙ cam3_tokens)
  ↓
concat([cam1_tokens, cam2_gated, cam3_gated], dim=-1)
  ↓
Linear(3*embed_dim → embed_dim) → Fused Features
```

### Routing Loss (Focal Loss)

```python
p_t = routing_weights[target_camera]
focal_weight = (1 - p_t)^gamma
loss = -alpha * focal_weight * log(p_t)
```

**파라미터**:
- `alpha = 0.25`: Positive/negative balance
- `gamma = 2.0`: Focusing parameter (hard examples)

---

## 🐛 Troubleshooting

### Q: "Camera MoE enabled but no multi-camera inputs found"
**A**: 입력 데이터에 `cam1_*`, `cam2_*`, `cam3_*` prefix 추가 또는 `use_camera_moe=False`

### Q: Routing loss가 감소하지 않음
**A**: 
1. Ground truth labels (`cam2_activate`, `cam3_activate`) 확인
2. `camera_routing_loss_weight` 증가 (0.1 → 0.2)

### Q: Action prediction 성능 저하
**A**: `camera_routing_loss_weight` 감소 (0.1 → 0.05)

---

**구현 완료**: 2026년 1월 15일  
**테스트**: ✅ 모든 테스트 통과  
**문서**: ✅ 완성
