# RobotWin to GR00T LeRobot v2 Conversion

This directory contains scripts and configurations for converting RobotWin simulation datasets to the GR00T LeRobot v2 format for training.

## Files

-   **`convert_robotwin_to_lerobot.py`**: Main conversion script that transforms RobotWin HDF5 files to LeRobot v2 format
-   **`robotwin_modality_config.py`**: Modality configuration for training with GR00T
-   **`validate_conversion.py`**: Validation script to ensure conversion correctness
-   **`README.md`**: This file

## RobotWin Data Structure

RobotWin datasets have the following structure:

```
<dataset_name>/├── data/│   ├── episode0.hdf5│   ├── episode1.hdf5│   └── ...├── video/│   ├── episode0.mp4│   ├── episode1.mp4│   └── ...├── instructions/│   ├── episode0.json│   ├── episode1.json│   └── ...├── _traj_data/│   ├── episode0.pkl│   └── ...├── scene_info.json└── seed.txt
```

### HDF5 Data Format

Each episode HDF5 file contains:

-   `joint_action/left_arm`: (N, 6) - Left arm joint positions
-   `joint_action/left_gripper`: (N,) - Left gripper state
-   `joint_action/right_arm`: (N, 6) - Right arm joint positions
-   `joint_action/right_gripper`: (N,) - Right gripper state
-   `joint_action/vector`: (N, 14) - Concatenated action vector
-   `endpose/left_endpose`: (N, 7) - Left end effector pose
-   `endpose/right_endpose`: (N, 7) - Right end effector pose
-   `observation/*/rgb`: Camera observations

## LeRobot v2 Output Format

The converted dataset follows the GR00T LeRobot v2 format:

```
<output_dir>/├── data/│   └── chunk-000/│       ├── episode_000000.parquet│       ├── episode_000001.parquet│       └── ...├── videos/│   └── chunk-000/│       └── observation.images.<camera>/│           ├── episode_000000.mp4│           └── ...└── meta/    ├── info.json    ├── episodes.jsonl    ├── tasks.jsonl    └── modality.json
```

## Usage

### Step 1: Convert RobotWin Dataset to LeRobot v2

```bash
python convert_robotwin_to_lerobot.py     --input_dir aloha-agilex_clean_50     --output_dir aloha-agilex_clean_50_lerobot     --robot_type aloha_agilex     --fps 30     --chunk_size 1000
```

**Arguments:**

-   `--input_dir`: Path to RobotWin dataset directory
-   `--output_dir`: Path to output LeRobot v2 dataset directory
-   `--robot_type`: Robot type name (default: `aloha_agilex`)
-   `--fps`: Frames per second (default: 30)
-   `--chunk_size`: Number of episodes per chunk (default: 1000)
-   `--use_seen_instructions`: Use 'seen' instructions for task descriptions (default: True)

### Step 2: Validate Conversion

```bash
python validate_conversion.py
```

This will check:

-   ✓ Parquet file structure and required columns
-   ✓ Meta files (info.json, episodes.jsonl, tasks.jsonl, modality.json)
-   ✓ Video files and directory structure
-   ✓ Consistency between parquet data and meta files

### Step 3: Train with GR00T

Use the generated dataset with GR00T training scripts:

```bash
# Example finetuning command (adjust paths as needed)python -m gr00t.experiment.launch_finetune     --config configs/finetune_config.py     --dataset_path /path/to/aloha-agilex_clean_50_lerobot     --modality_config_path robotwin/robotwin_modality_config.py     --output_dir /path/to/output
```

## Robot Specification

**ALOHA Dual-Arm Robot (Agilex Chassis)**

-   2 × 6 DOF arms
-   2 × 1 DOF grippers
-   Total: 14 DOF

**Cameras:**

-   `head`: Top-down view
-   `left_wrist`: Left wrist camera
-   `right_wrist`: Right wrist camera

## Data Format Details

### State Vector (14 DOF)

```
[left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
```

### Action Vector (14 DOF)

```
[left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)]
```

### Video Observations

-   Resolution: 240×320 pixels
-   Format: H.264 encoded MP4
-   Color: RGB (converted from HDF5 JPEG streams)

## Modality Configuration

Two configurations are provided in `robotwin_modality_config.py`:

### 1. `robotwin_aloha_config` (Recommended)

-   **Arms**: Relative actions for smoother control
-   **Grippers**: Absolute actions for precise control
-   **State**: Sin/cos embedding for joint angles

### 2. `robotwin_aloha_absolute_config`

-   **All modalities**: Absolute actions
-   Use this if you experience drifting with relative actions

## modality.json

The generated `modality.json` maps state/action indices and video keys:

```json
{    "state": {        "left_arm": {"start": 0, "end": 6},        "left_gripper": {"start": 6, "end": 7},        "right_arm": {"start": 7, "end": 13},        "right_gripper": {"start": 13, "end": 14}    },    "action": {        "left_arm": {"start": 0, "end": 6},        "left_gripper": {"start": 6, "end": 7},        "right_arm": {"start": 7, "end": 13},        "right_gripper": {"start": 13, "end": 14}    },    "video": {        "head": {"original_key": "observation.images.head"},        "left_wrist": {"original_key": "observation.images.left_wrist"},        "right_wrist": {"original_key": "observation.images.right_wrist"}    },    "annotation": {        "human.action.task_description": {}    }}
```

## Expected Output

After successful conversion:

```
Conversion complete!  Total episodes: 50  Total frames: 4778  Total tasks: 50  Output directory: aloha-agilex_clean_50_lerobot
```

Validation output:

```
✅ ALL VALIDATIONS PASSED!   The dataset is correctly formatted according to data_preparation.md📋 Summary:   • Episodes: 50   • Frames: 4778   • Tasks: 50   • Videos: 150   • Robot type: aloha_agilex   • FPS: 30
```

## Troubleshooting

### Video Color Issues

If videos appear with incorrect colors:

-   The script uses `cv2.imdecode` to decode JPEG streams from HDF5
-   Make sure BGR to RGB conversion is correct in `decode_jpeg_frame()`

### Missing Dependencies

Required packages:

```bash
pip install h5py pandas pyarrow opencv-python numpy tqdm
```

## References

-   [GR00T Data Preparation Guide](../getting_started/data_preparation.md)
-   [GR00T Data Config Guide](../getting_started/data_config.md)
-   [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
-   [RobotWin Documentation](https://github.com/BAAI-DCAI/RobotWin)

## Notes

1.  The script uses task descriptions from the `instructions/*.json` files
2.  Video files are copied to the new format (not re-encoded)
3.  Each episode has a unique task instruction
4.  The `annotation.human.validity` column uses task_index 0 ("valid") for all frames

  

  

```bash
export NUM_GPUS=1CUDA_VISIBLE_DEVICES=0 python     gr00t/experiment/launch_finetune.py     --base-model-path nvidia/GR00T-N1.6-3B     --dataset-path ./robotwin/aloha-agilex_clean_50_lerobot     --embodiment-tag NEW_EMBODIMENT     --modality-config-path ./robotwin/robotwin_modality_config.py     --num-gpus $NUM_GPUS     --output-dir ./robotwin/checkpoints     --save-total-limit 5     --save-steps 20000     --max-steps 60000     --use-wandb     --global-batch-size 4     --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08     --dataloader-num-workers 4
```