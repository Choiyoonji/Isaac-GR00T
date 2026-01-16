#!/usr/bin/env python3
"""
Quick validation script for converted RobotWin data with camera labels.

Usage:
    python check_camera_labels_in_data.py /path/to/converted/dataset
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def check_dataset(dataset_path: Path):
    """Check converted dataset for camera labels."""
    dataset_path = Path(dataset_path)
    
    print("="*80)
    print(f"Checking dataset: {dataset_path}")
    print("="*80)
    
    # 1. Check meta files
    print("\n1. Checking meta files...")
    
    meta_dir = dataset_path / "meta"
    assert meta_dir.exists(), f"Meta directory not found: {meta_dir}"
    
    # Check modality.json
    modality_path = meta_dir / "modality.json"
    assert modality_path.exists(), f"modality.json not found: {modality_path}"
    
    with open(modality_path) as f:
        modality = json.load(f)
    
    assert "annotation" in modality, "No 'annotation' in modality.json"
    assert "human.camera.cam2_activate" in modality["annotation"], \
        "Missing human.camera.cam2_activate in modality.json"
    assert "human.camera.cam3_activate" in modality["annotation"], \
        "Missing human.camera.cam3_activate in modality.json"
    
    print("  ✓ modality.json contains camera labels")
    
    # Check info.json
    info_path = meta_dir / "info.json"
    assert info_path.exists(), f"info.json not found: {info_path}"
    
    with open(info_path) as f:
        info = json.load(f)
    
    assert "features" in info, "No 'features' in info.json"
    assert "annotation.human.camera.cam2_activate" in info["features"], \
        "Missing cam2_activate in info.json features"
    assert "annotation.human.camera.cam3_activate" in info["features"], \
        "Missing cam3_activate in info.json features"
    
    print("  ✓ info.json contains camera label features")
    
    # 2. Check parquet files
    print("\n2. Checking parquet files...")
    
    data_dir = dataset_path / "data"
    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    
    # Find first parquet file
    parquet_files = list(data_dir.glob("**/*.parquet"))
    assert len(parquet_files) > 0, "No parquet files found"
    
    first_parquet = parquet_files[0]
    print(f"  Checking: {first_parquet.relative_to(dataset_path)}")
    
    df = pd.read_parquet(first_parquet)
    
    # Check columns
    required_cols = [
        "annotation.human.camera.cam2_activate",
        "annotation.human.camera.cam3_activate"
    ]
    
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
        print(f"  ✓ Column found: {col}")
    
    # Check data statistics
    cam2_activate = df["annotation.human.camera.cam2_activate"]
    cam3_activate = df["annotation.human.camera.cam3_activate"]
    
    print(f"\n3. Camera label statistics (first episode):")
    print(f"  Total frames: {len(df)}")
    print(f"  cam2_activate distribution:")
    print(f"    0 (inactive): {(cam2_activate == 0).sum()} frames ({(cam2_activate == 0).sum()/len(df)*100:.1f}%)")
    print(f"    1 (active):   {(cam2_activate == 1).sum()} frames ({(cam2_activate == 1).sum()/len(df)*100:.1f}%)")
    print(f"  cam3_activate distribution:")
    print(f"    0 (inactive): {(cam3_activate == 0).sum()} frames ({(cam3_activate == 0).sum()/len(df)*100:.1f}%)")
    print(f"    1 (active):   {(cam3_activate == 1).sum()} frames ({(cam3_activate == 1).sum()/len(df)*100:.1f}%)")
    
    # Check mutual exclusivity
    both_active = (cam2_activate == 1) & (cam3_activate == 1)
    both_inactive = (cam2_activate == 0) & (cam3_activate == 0)
    
    print(f"\n  Mutual exclusivity check:")
    print(f"    Both active:   {both_active.sum()} frames ({both_active.sum()/len(df)*100:.1f}%)")
    print(f"    Both inactive: {both_inactive.sum()} frames ({both_inactive.sum()/len(df)*100:.1f}%)")
    print(f"    Exclusive:     {(~both_active & ~both_inactive).sum()} frames ({(~both_active & ~both_inactive).sum()/len(df)*100:.1f}%)")
    
    # 4. Check multiple episodes
    print(f"\n4. Checking multiple episodes...")
    
    total_episodes = min(5, len(parquet_files))
    cam2_counts = []
    cam3_counts = []
    
    for i, parquet_file in enumerate(parquet_files[:total_episodes]):
        df = pd.read_parquet(parquet_file)
        cam2_active = (df["annotation.human.camera.cam2_activate"] == 1).sum()
        cam3_active = (df["annotation.human.camera.cam3_activate"] == 1).sum()
        cam2_counts.append(cam2_active)
        cam3_counts.append(cam3_active)
        
        episode_num = parquet_file.stem.split("_")[-1]
        print(f"  Episode {episode_num}: cam2={cam2_active}/{len(df)} frames, cam3={cam3_active}/{len(df)} frames")
    
    # Overall statistics
    print(f"\n5. Overall statistics (first {total_episodes} episodes):")
    total_cam2 = sum(cam2_counts)
    total_cam3 = sum(cam3_counts)
    total = total_cam2 + total_cam3
    
    if total > 0:
        print(f"  cam2 (Left Wrist):  {total_cam2} frames ({total_cam2/total*100:.1f}%)")
        print(f"  cam3 (Right Wrist): {total_cam3} frames ({total_cam3/total*100:.1f}%)")
        print(f"  Balance: {abs(total_cam2 - total_cam3) / total * 100:.1f}% difference")
    
    print("\n" + "="*80)
    print("✓ All checks passed!")
    print("="*80)
    
    print("\nDataset is ready for training with Camera MoE!")
    print("\nTo train:")
    print("  1. Set use_camera_moe: true in config")
    print("  2. Set camera_routing_loss_weight: 0.1")
    print("  3. Run training script")
    print("\nThe model will automatically use cam2_activate and cam3_activate")
    print("from the dataset for routing loss computation.")


def main():
    parser = argparse.ArgumentParser(
        description="Check converted RobotWin data for camera labels"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to converted dataset directory"
    )
    args = parser.parse_args()
    
    check_dataset(args.dataset_path)


if __name__ == "__main__":
    main()
