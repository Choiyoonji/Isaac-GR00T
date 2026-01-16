# RobotWin Camera Labels for Camera MoE

## 개요

RobotWin 데이터셋에 Camera MoE 학습을 위한 카메라 중요도 레이블을 추가했습니다.

## 변경 사항

### 1. 데이터 변환 스크립트 (`convert_robotwin_to_lerobot.py`)

#### 추가된 함수: `generate_camera_labels()`

```python
def generate_camera_labels(
    episode_idx: int,
    frame_idx: int,
    episode_length: int,
    task_description: str,
    data: Dict[str, Any],
) -> Tuple[int, int]:
    """
    휴리스틱 기반으로 카메라 중요도 레이블 생성
    
    Returns:
        (cam2_activate, cam3_activate)
        - cam2_activate: Left Wrist 카메라 중요도 (0 or 1)
        - cam3_activate: Right Wrist 카메라 중요도 (0 or 1)
    """
```

#### 휴리스틱 전략

**전략 1: Task 기반**
- Task에 "left" 포함 → `cam2_activate = 1`
- Task에 "right" 포함 → `cam3_activate = 1`

**전략 2: Motion 기반**
- 왼팔 motion > 오른팔 motion × 1.5 → `cam2_activate = 1`
- 오른팔 motion > 왼팔 motion × 1.5 → `cam3_activate = 1`

**전략 3: Fallback (균형 데이터)**
- Episode index가 짝수 → `cam2_activate = 1`
- Episode index가 홀수 → `cam3_activate = 1`

### 2. Parquet 데이터 추가 필드

```python
{
    "observation.state": [...],
    "action": [...],
    # 새로 추가된 필드
    "annotation.human.camera.cam2_activate": 0 or 1,
    "annotation.human.camera.cam3_activate": 0 or 1,
}
```

### 3. `modality.json` 업데이트

```json
{
    "annotation": {
        "human.action.task_description": {},
        "human.camera.cam2_activate": {},
        "human.camera.cam3_activate": {}
    }
}
```

### 4. `info.json` Features 추가

```json
{
    "features": {
        "annotation.human.camera.cam2_activate": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        },
        "annotation.human.camera.cam3_activate": {
            "dtype": "int64",
            "shape": [1],
            "names": null
        }
    }
}
```

### 5. Modality Config (`robotwin_modality_config.py`)

```python
robotwin_aloha_config = {
    # ... 기존 설정 ...
    
    # 새로 추가
    "camera_labels": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.camera.cam2_activate",
            "annotation.human.camera.cam3_activate",
        ],
    ),
}
```

## 사용 방법

### 데이터 변환

```bash
python robotwin/convert_robotwin_to_lerobot.py \
    --input_dir /path/to/robotwin_data \
    --output_dir /path/to/lerobot_output \
    --robot_type aloha_agilex \
    --fps 30
```

변환된 데이터는 자동으로 카메라 레이블을 포함합니다.

### Training with Camera MoE

변환된 데이터는 GR00T N1d6의 Camera MoE와 함께 사용할 수 있습니다:

```python
# Config 설정
config = Gr00tN1d6Config(
    use_camera_moe=True,
    camera_routing_loss_weight=0.1,
)

# 학습 시 자동으로 camera labels 사용
inputs = {
    "cam1_pixel_values": ...,  # head camera
    "cam2_pixel_values": ...,  # left_wrist camera
    "cam3_pixel_values": ...,  # right_wrist camera
    "state": ...,
    "action": ...,
    # Camera labels (자동으로 dataloader에서 로드)
    "cam2_activate": ...,  # from annotation.human.camera.cam2_activate
    "cam3_activate": ...,  # from annotation.human.camera.cam3_activate
}

outputs = model(inputs)
# outputs["routing_loss"]에 camera routing loss 포함
```

## 카메라 매핑

| RobotWin 카메라 | LeRobot Key | GR00T Camera ID | 역할 |
|----------------|-------------|-----------------|------|
| `head_camera` | `observation.images.head` | cam1 | Base (항상 포함) |
| `left_camera` | `observation.images.left_wrist` | cam2 | Left Wrist (gated) |
| `right_camera` | `observation.images.right_wrist` | cam3 | Right Wrist (gated) |

## 레이블 분포 예시

100 episodes 기준:

```
Task-based labels:
  - "left" tasks: ~30 episodes → cam2_activate=1
  - "right" tasks: ~30 episodes → cam3_activate=1
  - Neither: ~40 episodes → Motion/Fallback 적용

Motion-based labels:
  - Left-dominant frames: ~40%
  - Right-dominant frames: ~40%
  - Balanced frames: ~20% → Fallback 적용

Final distribution (approximate):
  - cam2_activate=1: ~50%
  - cam3_activate=1: ~50%
```

균형잡힌 데이터로 Camera Router가 잘 학습됩니다.

## 주의사항

1. **Pseudo-labels**: 실제 human annotation이 아닌 휴리스틱 기반 레이블
2. **Task에 명시적으로 "left"/"right" 포함 시**: 가장 정확한 레이블
3. **Motion 기반**: 실제 로봇 움직임 반영
4. **Fallback**: 균형잡힌 학습 데이터 보장

실제 배포 시에는 human annotation을 추가하거나, 더 정교한 휴리스틱을 사용할 것을 권장합니다.

## 검증

변환 후 레이블 확인:

```python
import pandas as pd

# Load parquet file
df = pd.read_parquet("output/data/chunk-000/episode_000000.parquet")

# Check camera labels
print("cam2_activate distribution:")
print(df["annotation.human.camera.cam2_activate"].value_counts())

print("\ncam3_activate distribution:")
print(df["annotation.human.camera.cam3_activate"].value_counts())

# Check correlations
print("\nBoth cameras active:")
both_active = (df["annotation.human.camera.cam2_activate"] == 1) & \
              (df["annotation.human.camera.cam3_activate"] == 1)
print(f"{both_active.sum()} frames ({both_active.sum()/len(df)*100:.1f}%)")
```

---

**Updated**: 2026-01-15  
**Compatible with**: GR00T N1d6 Camera MoE
