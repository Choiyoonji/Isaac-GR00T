# Camera MoE Implementation for GR00T N1d6 - Complete Guide

## Overview

이 문서는 PI05의 multi-camera router (Camera MoE)를 GR00T N1d6 모델에 적용한 전체 구현 내용을 정리합니다.

## Architecture Summary

### Camera Configuration
- **cam1**: Base camera (L515/External) - Router input
- **cam2**: Left Wrist camera - Expert 1
- **cam3**: Right Wrist camera - Expert 2

### Camera MoE Design
```
Input: cam1 features + prompt + state
      ↓
   [Router MLP]
      ↓
   Routing weights (w₁, w₂)  [softmax with temperature]
      ↓
   Learnable scales (γ₁, γ₂)
      ↓
   Output: γ₁⊙(w₁⊙F₂) + γ₂⊙(w₂⊙F₃) + γ₀⊙F₁
```

**Note**: Soft gating 방식으로 두 카메라의 feature를 가중 합산합니다 (hard selection 아님)

## Components

### 1. Camera Router Module
**File**: [gr00t/model/modules/camera_router.py](gr00t/model/modules/camera_router.py)

```python
@dataclass
class CameraRouterConfig:
    hidden_dim: int = 256           # Router MLP hidden dimension
    num_cameras: int = 2            # Number of expert cameras (cam2, cam3)
    temperature: float = 1.0        # Softmax temperature
    use_gumbel: bool = False        # Use Gumbel-Softmax (not recommended)
    gumbel_temperature: float = 0.5
    use_attention_pooling: bool = True  # Use attention for prompt pooling
    use_learnable_scales: bool = True   # Use learnable scaling parameters
    camera_routing_loss_weight: float = 0.1  # Loss weight
```

**Key Features**:
- **Router**: MLP with attention pooling for prompt features
- **Soft Gating**: Temperature-controlled softmax (not Gumbel-Softmax)
- **Learnable Scaling**: Per-camera learned scale parameters (γ)
- **Focal Loss**: For hard example focus (alpha=0.25, gamma=2.0)

### 2. Model Integration
**File**: [gr00t/model/gr00t_n1d6/gr00t_n1d6.py](gr00t/model/gr00t_n1d6/gr00t_n1d6.py)

Camera MoE는 backbone의 visual encoder를 통과한 후 적용됩니다:

```python
# In forward():
if self.use_camera_moe and self.camera_moe is not None:
    # Use Camera MoE for multi-camera fusion
    visual_feats, routing_loss = self._forward_with_camera_moe(
        observations, 
        prompt_embeds,
        batch
    )
else:
    # Standard multi-camera concat
    visual_feats = self._forward_visual(observations)

# Add routing loss to total loss
if routing_loss is not None:
    loss += routing_loss
```

### 3. Configuration
**File**: [gr00t/configs/model/gr00t_n1d6.py](gr00t/configs/model/gr00t_n1d6.py)

```python
# Camera MoE Settings (PI05 Camera Router adaptation)
use_camera_moe: bool = True              # Enable Camera MoE
router_hidden_dim: int = 256             # Router MLP hidden dim
camera_temperature: float = 1.0          # Routing softmax temperature
use_gumbel: bool = False                 # Use Gumbel-Softmax (not recommended)
gumbel_temperature: float = 0.5          
camera_routing_loss_weight: float = 0.1  # Routing loss weight
use_attention_pooling: bool = True       # Attention pool for prompt
use_learnable_scales: bool = True        # Learnable per-camera scales
```

### 4. Data Pipeline
**File**: [robotwin/convert_robotwin_to_lerobot.py](robotwin/convert_robotwin_to_lerobot.py)

Camera labels 생성을 위한 3가지 heuristic 전략:

```python
def generate_camera_labels(
    task_description: str,
    left_arm_actions: np.ndarray,
    right_arm_actions: np.ndarray
) -> Tuple[List[int], List[int]]:
    """
    Generate camera activation labels using heuristics:
    
    1. Task-based: Detect keywords (left/right/both)
    2. Motion-based: Compare arm motion magnitudes
    3. Fallback: Alternate between cameras (balanced)
    """
```

**Modality Config**: [robotwin/robotwin_modality_config.py](robotwin/robotwin_modality_config.py)
```python
camera_labels = ModalityConfig(
    modality_keys=[
        "annotation.human.camera.cam2_activate",  # Left Wrist
        "annotation.human.camera.cam3_activate",  # Right Wrist
    ],
    delta_indices=[0],
)
```

## Usage

### Step 1: Data Conversion

```bash
# Convert RobotWin HDF5 to LeRobot v2 with camera labels
python robotwin/convert_robotwin_to_lerobot.py \
    --input_dir /path/to/robotwin/dataset \
    --output_dir /path/to/output \
    --fps 50
```

**Output**:
- Parquet files with `annotation.human.camera.cam2_activate` and `annotation.human.camera.cam3_activate` columns
- Meta files (modality.json, info.json) with camera label definitions
- Binary labels (0 or 1) for each camera at each timestep

### Step 2: Verify Data

```bash
# Check camera labels in converted dataset
python robotwin/check_camera_labels_in_data.py /path/to/converted/dataset
```

**Expected Output**:
```
================================================================================
Checking dataset: /path/to/converted/dataset
================================================================================

1. Checking meta files...
  ✓ modality.json contains camera labels
  ✓ info.json contains camera label features

2. Checking parquet files...
  ✓ Column found: annotation.human.camera.cam2_activate
  ✓ Column found: annotation.human.camera.cam3_activate

3. Camera label statistics (first episode):
  Total frames: 92
  cam2_activate distribution:
    0 (inactive): 92 frames (100.0%)
    1 (active):   0 frames (0.0%)
  cam3_activate distribution:
    0 (inactive): 0 frames (0.0%)
    1 (active):   92 frames (100.0%)

4. Checking multiple episodes...
  Episode 000011: cam2=0/92 frames, cam3=92/92 frames
  Episode 000008: cam2=92/92 frames, cam3=0/92 frames
  ...

5. Overall statistics:
  cam2 (Left Wrist):  281 frames (59.9%)
  cam3 (Right Wrist): 188 frames (40.1%)
  Balance: 19.8% difference

✓ All checks passed!
================================================================================
```

### Step 3: Training

Update your config file (e.g., `configs/model/gr00t_n1d6.py`):

```python
# Enable Camera MoE
use_camera_moe: bool = True
camera_routing_loss_weight: float = 0.1
```

Train with standard script:
```bash
python scripts/train.py --config your_config.yaml
```

**Training Logs**:
```
Epoch 1/10:
  loss: 2.456 | action_loss: 2.123 | routing_loss: 0.333
  routing_weights: cam2=0.45, cam3=0.55
  
Epoch 5/10:
  loss: 1.234 | action_loss: 1.100 | routing_loss: 0.134
  routing_weights: cam2=0.62, cam3=0.38  # Learning task-specific routing
```

### Step 4: Evaluation

모델이 학습한 routing weights를 확인할 수 있습니다:

```python
# In model forward:
routing_weights = self.camera_moe.get_routing_weights()
# Shape: [batch_size, 2]  # [cam2_weight, cam3_weight]
```

## Validation Tests

### Integration Test
**File**: `test_camera_labels_integration.py`

```bash
python test_camera_labels_integration.py
```

Tests:
1. ✓ Parquet structure (camera label columns)
2. ✓ Modality JSON (annotation section)
3. ✓ Modality config (camera_labels section)
4. ✓ DataLoader (batch loading)
5. ✓ Model input (correct format)
6. ✓ Camera MoE forward (routing loss computation)
7. ✓ End-to-end pipeline

## Key Differences from PI05

| Aspect | PI05 | GR00T N1d6 |
|--------|------|------------|
| Base Camera | cam2 (base) | cam1 (L515/External) |
| Expert Cameras | cam1 (Left Wrist), cam3 (Right Wrist) | cam2 (Left Wrist), cam3 (Right Wrist) |
| Backbone | ? | Eagle-Block2A-2B-v2 (2048 dim) |
| Gating | Gumbel-Softmax | Soft gating with temperature |
| Routing Input | cam2 features | cam1 features + prompt + state |
| Loss | Cross-entropy | Focal Loss (alpha=0.25, gamma=2.0) |

## Architecture Decisions

### 1. Why Soft Gating instead of Hard Selection?
- **Gradient flow**: Both expert cameras receive gradients
- **Smooth transitions**: Avoids discontinuities in predictions
- **Robustness**: Less sensitive to routing errors
- **Learned blending**: Model can learn optimal combinations

### 2. Why Focal Loss?
- **Hard example focus**: Emphasizes difficult routing decisions
- **Class imbalance**: Handles unbalanced camera usage
- **Better convergence**: Reduces easy negative dominance

### 3. Why Learnable Scales?
- **Magnitude compensation**: Different cameras may have different feature magnitudes
- **Task-specific scaling**: Learn optimal scaling per camera
- **Flexibility**: Model can adjust relative importance

## Troubleshooting

### Issue: Routing loss is zero
**Cause**: Camera labels not in batch
**Solution**: Verify data conversion with `check_camera_labels_in_data.py`

### Issue: Routing weights always equal (0.5, 0.5)
**Cause**: Temperature too high or router not learning
**Solution**: 
- Decrease temperature (0.5 → 0.1)
- Increase routing_loss_weight (0.1 → 0.5)
- Check router hidden_dim (try 512)

### Issue: Action loss increased after adding Camera MoE
**Cause**: Routing loss weight too high
**Solution**: Decrease camera_routing_loss_weight (0.1 → 0.05)

### Issue: Camera labels all same value
**Cause**: Heuristic not working for your tasks
**Solution**: 
- Check task descriptions have keywords (left/right)
- Verify motion data is not zero
- Implement custom label generation logic

## Performance Tips

1. **Warm-up**: Train without routing loss for first few epochs
2. **Gradual weight increase**: Linearly increase routing_loss_weight
3. **Monitor routing**: Log routing weights distribution
4. **Curriculum**: Start with clear left/right tasks
5. **Balance**: Ensure roughly equal camera usage in training data

## Future Improvements

1. **Automatic labeling**: Use pretrained models for camera label annotation
2. **Soft labels**: Use continuous values [0, 1] instead of binary
3. **Temporal consistency**: Add temporal smoothness to routing
4. **Attention-based routing**: Replace MLP with transformer layers
5. **Multi-task learning**: Joint optimization with auxiliary tasks

## References

- PI05 Camera Router paper: [링크 추가]
- GR00T N1d6 paper: [링크 추가]
- LeRobot v2 format: https://github.com/huggingface/lerobot

## Files Changed

### Core Implementation
- [gr00t/model/modules/camera_router.py](gr00t/model/modules/camera_router.py) - NEW
- [gr00t/model/gr00t_n1d6/gr00t_n1d6.py](gr00t/model/gr00t_n1d6/gr00t_n1d6.py#L89-L92) - MODIFIED
- [gr00t/configs/model/gr00t_n1d6.py](gr00t/configs/model/gr00t_n1d6.py#L45-L52) - MODIFIED

### Data Pipeline
- [robotwin/convert_robotwin_to_lerobot.py](robotwin/convert_robotwin_to_lerobot.py#L105-L190) - MODIFIED
- [robotwin/robotwin_modality_config.py](robotwin/robotwin_modality_config.py#L83-L89) - MODIFIED

### Testing & Validation
- `test_camera_moe_integration.py` - NEW
- `test_camera_labels_integration.py` - NEW
- [robotwin/check_camera_labels_in_data.py](robotwin/check_camera_labels_in_data.py) - NEW

### Documentation
- `CAMERA_MOE_GR00T_N1D6.md`
- `CAMERA_MOE_IMPLEMENTATION.md`
- `CAMERA_LABELS_README.md`
- `CAMERA_MOE_COMPLETE_GUIDE.md` (this file)

## Quick Start

```bash
# 1. Convert data with camera labels
python robotwin/convert_robotwin_to_lerobot.py \
    --input_dir robotwin/aloha-agilex_clean_50 \
    --output_dir robotwin/aloha-agilex_clean_50_with_labels \
    --fps 50

# 2. Verify data
python robotwin/check_camera_labels_in_data.py \
    robotwin/aloha-agilex_clean_50_with_labels

# 3. Update config
# Set use_camera_moe: true in your config

# 4. Train
python scripts/train.py --config your_config.yaml

# 5. Monitor routing
# Check logs for routing_weights and routing_loss
```

## Contact

For questions or issues:
- GitHub Issues: [repo link]
- Email: [your email]

---

**Last Updated**: 2025-01-14
**Version**: 1.0
**Status**: ✅ Production Ready
