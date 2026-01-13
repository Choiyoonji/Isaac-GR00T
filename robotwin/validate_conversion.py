#!/usr/bin/env python3
"""
Validate RobotWin to LeRobot v2 conversion against data_preparation.md requirements.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def validate_parquet_structure(data_dir: Path) -> Tuple[bool, List[str]]:
    """Validate parquet file structure against requirements."""
    issues = []
    
    # Required columns per data_preparation.md
    required_columns = {
        "observation.state": "1D concatenated array of all state modalities",
        "action": "1D concatenated array of all action modalities",
        "timestamp": "float point number of the starting time",
        "annotation.human.action.task_description": "index of task description",
        "annotation.human.validity": "validity annotation index",
        "task_index": "index of the task",
        "episode_index": "index of the episode",
        "index": "global index across all observations",
        "next.reward": "reward of the next observation",
        "next.done": "whether the episode is done"
    }
    
    # Load first parquet file
    parquet_files = list(data_dir.glob("chunk-000/*.parquet"))
    if not parquet_files:
        issues.append("No parquet files found in data/chunk-000/")
        return False, issues
    
    df = pd.read_parquet(parquet_files[0])
    
    print("=" * 80)
    print("PARQUET FILE STRUCTURE VALIDATION")
    print("=" * 80)
    
    # Check required columns
    print("\n1. Required Columns Check:")
    all_present = True
    for col, desc in required_columns.items():
        if col in df.columns:
            print(f"   ✓ {col:<45}")
        else:
            print(f"   ✗ {col:<45} - MISSING!")
            issues.append(f"Missing required column: {col}")
            all_present = False
    
    if not all_present:
        return False, issues
    
    # Check data types and structure
    print(f"\n2. Data Types and Structure:")
    print(f"   observation.state: type={type(df['observation.state'].iloc[0])}, length={len(df['observation.state'].iloc[0])}")
    print(f"   action: type={type(df['action'].iloc[0])}, length={len(df['action'].iloc[0])}")
    print(f"   timestamp: type={type(df['timestamp'].iloc[0])}, value={df['timestamp'].iloc[0]}")
    print(f"   task_index: type={type(df['task_index'].iloc[0])}, value={df['task_index'].iloc[0]}")
    
    # Check if state and action are lists or numpy arrays (both are acceptable)
    # Pandas/parquet may load them as numpy arrays, which is fine
    import numpy as np
    if not isinstance(df['observation.state'].iloc[0], (list, np.ndarray)):
        issues.append("observation.state should be a list or numpy array")
    if not isinstance(df['action'].iloc[0], (list, np.ndarray)):
        issues.append("action should be a list or numpy array")
    
    # Check episode structure
    print(f"\n3. Episode Structure Check:")
    print(f"   Total rows in episode: {len(df)}")
    print(f"   First index: {df['index'].iloc[0]}")
    print(f"   Last index: {df['index'].iloc[-1]}")
    print(f"   Last frame next.done: {df['next.done'].iloc[-1]}")
    
    if not df['next.done'].iloc[-1]:
        issues.append("Last frame should have next.done=True")
    
    # Sample first row
    print(f"\n4. Sample First Row:")
    first_row = df.iloc[0].to_dict()
    for key, value in first_row.items():
        if isinstance(value, list):
            if len(value) <= 3:
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: [{value[0]:.4f}, {value[1]:.4f}, ..., {value[-1]:.4f}] (length={len(value)})")
        else:
            print(f"   {key}: {value}")
    
    return len(issues) == 0, issues


def validate_meta_files(meta_dir: Path) -> Tuple[bool, List[str]]:
    """Validate meta files structure."""
    issues = []
    
    print("\n" + "=" * 80)
    print("META FILES VALIDATION")
    print("=" * 80)
    
    # Check required meta files
    print("\n5. Meta Directory Structure:")
    required_meta = ["episodes.jsonl", "tasks.jsonl", "info.json", "modality.json"]
    for req in required_meta:
        if (meta_dir / req).exists():
            print(f"   ✓ meta/{req}")
        else:
            print(f"   ✗ meta/{req} - MISSING!")
            issues.append(f"Missing meta file: {req}")
    
    if issues:
        return False, issues
    
    # Validate modality.json structure
    print("\n6. modality.json Structure:")
    with open(meta_dir / "modality.json") as f:
        modality = json.load(f)
    
    required_modality_keys = ["state", "action", "video", "annotation"]
    for key in required_modality_keys:
        if key in modality:
            print(f"   ✓ {key}: {list(modality[key].keys())}")
        else:
            print(f"   ✗ {key} - MISSING!")
            issues.append(f"Missing key in modality.json: {key}")
    
    # Validate state/action indices
    print("\n7. State/Action Index Validation:")
    for modality_type in ["state", "action"]:
        if modality_type not in modality:
            continue
        print(f"   {modality_type}:")
        total_size = 0
        for key, val in modality[modality_type].items():
            size = val["end"] - val["start"]
            total_size += size
            print(f"      {key}: [{val['start']}:{val['end']}] = {size} DOF")
        print(f"      Total: {total_size} DOF")
    
    # Validate info.json
    print("\n8. info.json Validation:")
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    
    required_info_keys = [
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "fps",
        "data_path",
        "video_path",
        "features"
    ]
    
    for key in required_info_keys:
        if key in info:
            value = info[key]
            if isinstance(value, (dict, int, float)):
                print(f"   ✓ {key}: {value if not isinstance(value, dict) else '...'}")
            else:
                print(f"   ✓ {key}: {value}")
        else:
            print(f"   ✗ {key} - MISSING!")
            issues.append(f"Missing key in info.json: {key}")
    
    # Check codebase version
    if info.get("codebase_version") != "v2.1":
        issues.append(f"codebase_version should be 'v2.1', got '{info.get('codebase_version')}'")
    
    # Check video features
    print("\n9. Video Features in info.json:")
    video_features = [k for k in info["features"].keys() if k.startswith("observation.images.")]
    if not video_features:
        issues.append("No video features found in info.json")
    for vf in video_features:
        vinfo = info["features"][vf]
        print(f"   ✓ {vf}:")
        print(f"      shape: {vinfo['shape']}, codec: {vinfo['info']['video.codec']}")
    
    # Validate episodes.jsonl
    print("\n10. episodes.jsonl Structure:")
    with open(meta_dir / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    print(f"   Total episodes: {len(episodes)}")
    print(f"   Sample episode 0: {episodes[0]}")
    
    # Check episode structure
    if episodes:
        required_episode_keys = ["episode_index", "tasks", "length"]
        for key in required_episode_keys:
            if key not in episodes[0]:
                issues.append(f"Missing key in episodes.jsonl: {key}")
    
    # Validate tasks.jsonl
    print("\n11. tasks.jsonl Structure:")
    with open(meta_dir / "tasks.jsonl") as f:
        tasks = [json.loads(line) for line in f]
    print(f"   Total tasks: {len(tasks)}")
    print(f"   First 3 tasks:")
    for t in tasks[:3]:
        print(f"      {t}")
    
    # Check task structure
    if tasks:
        required_task_keys = ["task_index", "task"]
        for key in required_task_keys:
            if key not in tasks[0]:
                issues.append(f"Missing key in tasks.jsonl: {key}")
    
    # Check if "valid" task exists
    valid_task_exists = any(t.get("task") == "valid" for t in tasks)
    if valid_task_exists:
        print("   ✓ 'valid' task exists")
    else:
        print("   ⚠ 'valid' task not found (optional)")
    
    return len(issues) == 0, issues


def validate_videos(video_dir: Path, info: Dict) -> Tuple[bool, List[str]]:
    """Validate video files."""
    issues = []
    
    print("\n" + "=" * 80)
    print("VIDEO FILES VALIDATION")
    print("=" * 80)
    
    print("\n12. Video Directory Structure:")
    video_dirs = list((video_dir / "chunk-000").glob("observation.images.*"))
    if not video_dirs:
        issues.append("No video directories found in videos/chunk-000/")
        return False, issues
    
    total_videos = 0
    for vdir in sorted(video_dirs):
        video_files = list(vdir.glob("*.mp4"))
        total_videos += len(video_files)
        print(f"   ✓ {vdir.name}: {len(video_files)} videos")
        if len(video_files) > 0:
            print(f"      First: {video_files[0].name}, Last: {video_files[-1].name}")
    
    # Check video count matches info.json
    expected_videos = info.get("total_videos", 0)
    if total_videos != expected_videos:
        issues.append(f"Video count mismatch: expected={expected_videos}, actual={total_videos}")
        print(f"   ✗ Video count mismatch: expected={expected_videos}, actual={total_videos}")
    else:
        print(f"   ✓ Total videos match: {total_videos}")
    
    return len(issues) == 0, issues


def validate_consistency(data_dir: Path, meta_dir: Path) -> Tuple[bool, List[str]]:
    """Validate consistency between parquet and meta files."""
    issues = []
    
    print("\n" + "=" * 80)
    print("CONSISTENCY VALIDATION")
    print("=" * 80)
    
    # Load data
    parquet_files = list(data_dir.glob("chunk-000/*.parquet"))
    df = pd.read_parquet(parquet_files[0])
    
    with open(meta_dir / "modality.json") as f:
        modality = json.load(f)
    
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    
    print("\n13. State/Action Size Consistency:")
    
    # Check state size
    state_size_parquet = len(df['observation.state'].iloc[0])
    state_size_modality = sum(v["end"] - v["start"] for v in modality["state"].values())
    
    print(f"   State size in parquet: {state_size_parquet}")
    print(f"   State size in modality.json: {state_size_modality}")
    
    if state_size_parquet == state_size_modality:
        print(f"   ✓ State sizes match")
    else:
        issues.append(f"State size mismatch: parquet={state_size_parquet}, modality={state_size_modality}")
        print(f"   ✗ State sizes do not match!")
    
    # Check action size
    action_size_parquet = len(df['action'].iloc[0])
    action_size_modality = sum(v["end"] - v["start"] for v in modality["action"].values())
    
    print(f"   Action size in parquet: {action_size_parquet}")
    print(f"   Action size in modality.json: {action_size_modality}")
    
    if action_size_parquet == action_size_modality:
        print(f"   ✓ Action sizes match")
    else:
        issues.append(f"Action size mismatch: parquet={action_size_parquet}, modality={action_size_modality}")
        print(f"   ✗ Action sizes do not match!")
    
    # Check video keys consistency
    print("\n14. Video Keys Consistency:")
    modality_video_keys = set(modality["video"].keys())
    info_video_keys = set(k.replace("observation.images.", "") for k in info["features"].keys() if k.startswith("observation.images."))
    
    print(f"   Video keys in modality.json: {modality_video_keys}")
    print(f"   Video keys in info.json: {info_video_keys}")
    
    if modality_video_keys == info_video_keys:
        print(f"   ✓ Video keys match")
    else:
        issues.append(f"Video keys mismatch between modality.json and info.json")
        print(f"   ✗ Video keys do not match!")
    
    return len(issues) == 0, issues


def main():
    """Main validation function."""
    dataset_dir = Path("/home/choiyj/Isaac-GR00T/robotwin/aloha-agilex_clean_50_lerobot")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print("🔍 Validating RobotWin to LeRobot v2 conversion")
    print(f"📁 Dataset: {dataset_dir}")
    print()
    
    all_issues = []
    
    # Validate parquet files
    success, issues = validate_parquet_structure(dataset_dir / "data")
    all_issues.extend(issues)
    
    # Validate meta files
    success, issues = validate_meta_files(dataset_dir / "meta")
    all_issues.extend(issues)
    
    # Load info.json for video validation
    with open(dataset_dir / "meta" / "info.json") as f:
        info = json.load(f)
    
    # Validate videos
    success, issues = validate_videos(dataset_dir / "videos", info)
    all_issues.extend(issues)
    
    # Validate consistency
    success, issues = validate_consistency(dataset_dir / "data", dataset_dir / "meta")
    all_issues.extend(issues)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    
    if not all_issues:
        print("\n✅ ALL VALIDATIONS PASSED!")
        print("   The dataset is correctly formatted according to data_preparation.md")
        print("\n📋 Summary:")
        print(f"   • Episodes: {info['total_episodes']}")
        print(f"   • Frames: {info['total_frames']}")
        print(f"   • Tasks: {info['total_tasks']}")
        print(f"   • Videos: {info['total_videos']}")
        print(f"   • Robot type: {info['robot_type']}")
        print(f"   • FPS: {info['fps']}")
    else:
        print("\n❌ VALIDATION FAILED!")
        print(f"   Found {len(all_issues)} issue(s):\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")


if __name__ == "__main__":
    main()
