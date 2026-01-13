#!/usr/bin/env python3
"""
Convert RobotWin dataset to GR00T LeRobot v2 format.

This script converts RobotWin simulation data to the LeRobot v2 format compatible with GR00T.

RobotWin data structure:
    - data/*.hdf5: Episode data (joint_action, endpose, observation)
    - video/*.mp4: Episode videos
    - instructions/*.json: Task instructions (seen/unseen)
    - _traj_data/*.pkl: Trajectory data
    - scene_info.json: Scene information

LeRobot v2 output structure:
    - data/chunk-XXX/episode_XXXXXX.parquet
    - videos/chunk-XXX/observation.images.<view>/episode_XXXXXX.mp4
    - meta/info.json
    - meta/episodes.jsonl
    - meta/tasks.jsonl
    - meta/modality.json

Usage:
    python convert_robotwin_to_lerobot.py --input_dir /path/to/robotwin_data --output_dir /path/to/lerobot_output
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Camera mapping: HDF5 camera name -> LeRobot video key
CAMERA_MAPPING = {
    "head_camera": "head",
    "left_camera": "left_wrist",
    "right_camera": "right_wrist",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RobotWin dataset to LeRobot v2 format")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to RobotWin dataset directory (e.g., aloha-agilex_clean_50)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output LeRobot v2 dataset directory",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="aloha_agilex",
        help="Robot type name (default: aloha_agilex)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of episodes per chunk (default: 1000)",
    )
    parser.add_argument(
        "--use_seen_instructions",
        action="store_true",
        default=True,
        help="Use 'seen' instructions for task description (default: True)",
    )
    return parser.parse_args()


def get_episode_files(input_dir: Path) -> List[int]:
    """Get sorted list of episode indices from data directory."""
    data_dir = input_dir / "data"
    episode_files = list(data_dir.glob("episode*.hdf5"))
    
    episode_indices = []
    for f in episode_files:
        # Extract episode number from filename (e.g., episode0.hdf5 -> 0)
        idx = int(f.stem.replace("episode", ""))
        episode_indices.append(idx)
    
    return sorted(episode_indices)


def load_episode_data(input_dir: Path, episode_idx: int) -> Dict[str, Any]:
    """Load episode data from HDF5 file."""
    hdf5_path = input_dir / "data" / f"episode{episode_idx}.hdf5"
    
    data = {}
    with h5py.File(hdf5_path, "r") as f:
        # Load joint action data
        data["left_arm_action"] = np.array(f["joint_action/left_arm"])
        data["left_gripper_action"] = np.array(f["joint_action/left_gripper"])
        data["right_arm_action"] = np.array(f["joint_action/right_arm"])
        data["right_gripper_action"] = np.array(f["joint_action/right_gripper"])
        data["action_vector"] = np.array(f["joint_action/vector"])
        
        # Load endpose (can be used as state)
        data["left_endpose"] = np.array(f["endpose/left_endpose"])
        data["left_gripper_state"] = np.array(f["endpose/left_gripper"])
        data["right_endpose"] = np.array(f["endpose/right_endpose"])
        data["right_gripper_state"] = np.array(f["endpose/right_gripper"])
        
        # Get episode length
        data["length"] = data["action_vector"].shape[0]
    
    return data


def load_instruction(input_dir: Path, episode_idx: int, use_seen: bool = True) -> str:
    """Load task instruction for an episode."""
    instruction_path = input_dir / "instructions" / f"episode{episode_idx}.json"
    
    if instruction_path.exists():
        with open(instruction_path, "r") as f:
            instructions = json.load(f)
        
        key = "seen" if use_seen else "unseen"
        if key in instructions and len(instructions[key]) > 0:
            # Use the first instruction
            return instructions[key][0]
    
    return "perform the task"


def create_state_vector(data: Dict[str, Any], frame_idx: int) -> np.ndarray:
    """
    Create observation.state vector from episode data.
    
    State format for dual-arm robot (14 DOF):
    - left_arm: 6 DOF
    - left_gripper: 1 DOF
    - right_arm: 6 DOF
    - right_gripper: 1 DOF
    """
    left_arm = data["left_arm_action"][frame_idx]  # 6
    left_gripper = np.array([data["left_gripper_state"][frame_idx]])  # 1
    right_arm = data["right_arm_action"][frame_idx]  # 6
    right_gripper = np.array([data["right_gripper_state"][frame_idx]])  # 1
    
    state = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
    return state.astype(np.float32)


def create_action_vector(data: Dict[str, Any], frame_idx: int) -> np.ndarray:
    """
    Create action vector from episode data.
    
    Action format for dual-arm robot (14 DOF):
    - left_arm: 6 DOF
    - left_gripper: 1 DOF
    - right_arm: 6 DOF
    - right_gripper: 1 DOF
    """
    # Use the concatenated vector directly if it's the right format
    action = data["action_vector"][frame_idx]
    return action.astype(np.float32)


def decode_jpeg_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array (RGB).
    
    RobotWin stores images as bit streams that should be decoded with cv2.imdecode.
    cv2.imdecode returns BGR format, so we convert to RGB.
    """
    # Decode using OpenCV as per RobotWin documentation
    image_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    # Convert BGR to RGB for correct colors
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    return image_bgr


def extract_video_from_hdf5(
    hdf5_path: Path,
    camera_name: str,
    output_video_path: Path,
    fps: int = 30,
) -> Tuple[int, int]:
    """
    Extract video from HDF5 file's camera RGB data.
    
    Args:
        hdf5_path: Path to HDF5 file
        camera_name: Camera name in HDF5 (e.g., "head_camera")
        output_video_path: Output video file path
        fps: Frames per second
        
    Returns:
        Tuple of (height, width) of the video
    """
    with h5py.File(hdf5_path, "r") as f:
        rgb_data = f[f"observation/{camera_name}/rgb"]
        num_frames = rgb_data.shape[0]
        
        # Decode first frame to get dimensions
        first_frame = decode_jpeg_frame(rgb_data[0])
        height, width = first_frame.shape[:2]
        
        # Try using ffmpeg directly with pipe input (more reliable for color)
        try:
            # Start ffmpeg process
            # Input is RGB format after BGR2RGB conversion
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',  # RGB format after conversion
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',  # Read from stdin
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                '-crf', '23',
                str(output_video_path)
            ]
            
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write all frames to ffmpeg stdin
            for i in range(num_frames):
                frame_rgb = decode_jpeg_frame(rgb_data[i])
                process.stdin.write(frame_rgb.tobytes())
            
            process.stdin.close()
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, ffmpeg_cmd)
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to OpenCV if ffmpeg fails or is not installed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            
            for i in range(num_frames):
                frame_rgb = decode_jpeg_frame(rgb_data[i])
                # OpenCV VideoWriter expects BGR format, so convert RGB back to BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
    
    return height, width


def create_modality_json(output_dir: Path):
    """Create modality.json file for GR00T LeRobot format with 3 cameras."""
    modality = {
        "state": {
            "left_arm": {
                "start": 0,
                "end": 6
            },
            "left_gripper": {
                "start": 6,
                "end": 7
            },
            "right_arm": {
                "start": 7,
                "end": 13
            },
            "right_gripper": {
                "start": 13,
                "end": 14
            }
        },
        "action": {
            "left_arm": {
                "start": 0,
                "end": 6
            },
            "left_gripper": {
                "start": 6,
                "end": 7
            },
            "right_arm": {
                "start": 7,
                "end": 13
            },
            "right_gripper": {
                "start": 13,
                "end": 14
            }
        },
        "video": {
            "head": {
                "original_key": "observation.images.head"
            },
            "left_wrist": {
                "original_key": "observation.images.left_wrist"
            },
            "right_wrist": {
                "original_key": "observation.images.right_wrist"
            }
        },
        "annotation": {
            "human.action.task_description": {}
        }
    }
    
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)


def create_info_json(
    output_dir: Path,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    chunk_size: int,
    video_info: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """Create info.json file for LeRobot v2 format with 3 cameras."""
    if video_info is None:
        video_info = {
            "head": (480, 640),
            "left_wrist": (480, 640),
            "right_wrist": (480, 640),
        }
    
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": [
                    "left_arm.joint1", "left_arm.joint2", "left_arm.joint3",
                    "left_arm.joint4", "left_arm.joint5", "left_arm.joint6",
                    "left_gripper",
                    "right_arm.joint1", "right_arm.joint2", "right_arm.joint3",
                    "right_arm.joint4", "right_arm.joint5", "right_arm.joint6",
                    "right_gripper"
                ],
                "shape": [14]
            },
            "observation.state": {
                "dtype": "float32",
                "names": [
                    "left_arm.joint1", "left_arm.joint2", "left_arm.joint3",
                    "left_arm.joint4", "left_arm.joint5", "left_arm.joint6",
                    "left_gripper",
                    "right_arm.joint1", "right_arm.joint2", "right_arm.joint3",
                    "right_arm.joint4", "right_arm.joint5", "right_arm.joint6",
                    "right_gripper"
                ],
                "shape": [14]
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            }
        },
        "total_chunks": (total_episodes // chunk_size) + 1,
        "total_videos": total_episodes * 3  # 3 cameras per episode
    }
    
    # Add video features for each camera
    for video_key, (height, width) in video_info.items():
        info["features"][f"observation.images.{video_key}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False
            }
        }
    
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def convert_robotwin_to_lerobot(
    input_dir: Path,
    output_dir: Path,
    robot_type: str = "aloha_agilex",
    fps: int = 30,
    chunk_size: int = 1000,
    use_seen_instructions: bool = True,
):
    """
    Convert RobotWin dataset to LeRobot v2 format.
    
    Args:
        input_dir: Path to RobotWin dataset directory
        output_dir: Path to output LeRobot v2 dataset directory
        robot_type: Robot type name
        fps: Frames per second
        chunk_size: Number of episodes per chunk
        use_seen_instructions: Use 'seen' instructions for task description
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)
    
    # Get episode indices
    episode_indices = get_episode_files(input_dir)
    total_episodes = len(episode_indices)
    print(f"Found {total_episodes} episodes")
    
    # Collect all unique task descriptions
    task_to_index: Dict[str, int] = {}
    tasks_list: List[Dict[str, Any]] = []
    
    # Add validity task
    task_to_index["valid"] = 0
    tasks_list.append({"task_index": 0, "task": "valid"})
    
    # First pass: collect all task descriptions
    for episode_idx in episode_indices:
        instruction = load_instruction(input_dir, episode_idx, use_seen_instructions)
        if instruction not in task_to_index:
            task_index = len(task_to_index)
            task_to_index[instruction] = task_index
            tasks_list.append({"task_index": task_index, "task": instruction})
    
    print(f"Found {len(tasks_list)} unique tasks")
    
    # Write tasks.jsonl
    with open(output_dir / "meta" / "tasks.jsonl", "w") as f:
        for task in tasks_list:
            f.write(json.dumps(task) + "\n")
    
    # Create video directories for each camera
    video_info: Dict[str, Tuple[int, int]] = {}
    
    # Process episodes
    episodes_list: List[Dict[str, Any]] = []
    global_index = 0
    total_frames = 0
    
    for new_episode_idx, orig_episode_idx in enumerate(tqdm(episode_indices, desc="Converting episodes")):
        # Load episode data
        data = load_episode_data(input_dir, orig_episode_idx)
        episode_length = data["length"]
        
        # Get task instruction
        instruction = load_instruction(input_dir, orig_episode_idx, use_seen_instructions)
        task_index = task_to_index[instruction]
        
        # Determine chunk
        chunk_idx = new_episode_idx // chunk_size
        
        # Create data directory for chunk
        data_chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video directories for each camera and extract videos
        hdf5_path = input_dir / "data" / f"episode{orig_episode_idx}.hdf5"
        
        for hdf5_cam_name, lerobot_cam_name in CAMERA_MAPPING.items():
            video_chunk_dir = output_dir / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{lerobot_cam_name}"
            video_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            dst_video = video_chunk_dir / f"episode_{new_episode_idx:06d}.mp4"
            
            # Extract video from HDF5
            height, width = extract_video_from_hdf5(
                hdf5_path,
                hdf5_cam_name,
                dst_video,
                fps=fps,
            )
            
            # Store video info (from first episode)
            if lerobot_cam_name not in video_info:
                video_info[lerobot_cam_name] = (height, width)
        
        # Create parquet data
        parquet_data = []
        
        for frame_idx in range(episode_length):
            # Create state and action vectors
            state = create_state_vector(data, frame_idx)
            action = create_action_vector(data, min(frame_idx+30, episode_length - 1))  # Use next action for current frame
            
            # Create row
            row = {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "timestamp": float(frame_idx / fps),
                "annotation.human.action.task_description": task_index,
                "annotation.human.validity": 0,  # 0 = valid
                "task_index": task_index,
                "episode_index": new_episode_idx,
                "frame_index": frame_idx,
                "index": global_index,
                "next.reward": 0.0,
                "next.done": frame_idx == episode_length - 1,
            }
            
            parquet_data.append(row)
            global_index += 1
        
        # Save parquet file
        df = pd.DataFrame(parquet_data)
        parquet_path = data_chunk_dir / f"episode_{new_episode_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Add episode info
        episodes_list.append({
            "episode_index": new_episode_idx,
            "tasks": [instruction],
            "length": episode_length,
        })
        
        total_frames += episode_length
    
    # Write episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for episode in episodes_list:
            f.write(json.dumps(episode) + "\n")
    
    # Create modality.json
    create_modality_json(output_dir)
    
    # Create info.json
    create_info_json(
        output_dir,
        robot_type,
        total_episodes,
        total_frames,
        len(tasks_list),  # This includes the "valid" task, so actual count is correct
        fps,
        chunk_size,
        video_info,
    )
    
    print(f"\nConversion complete!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total tasks: {len(tasks_list)}")
    print(f"  Output directory: {output_dir}")


def main():
    args = parse_args()
    
    convert_robotwin_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        robot_type=args.robot_type,
        fps=args.fps,
        chunk_size=args.chunk_size,
        use_seen_instructions=args.use_seen_instructions,
    )


if __name__ == "__main__":
    main()
