#!/usr/bin/env python3
"""
Test script to verify the RobotWin modality configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from robotwin.robotwin_modality_config import (
        robotwin_aloha_config,
        robotwin_aloha_absolute_config,
    )
    print("✅ Successfully imported RobotWin modality configs")
except ImportError as e:
    print(f"❌ Failed to import configs: {e}")
    sys.exit(1)


def validate_config(config, config_name):
    """Validate a modality config structure."""
    print(f"\n{'='*80}")
    print(f"Validating: {config_name}")
    print('='*80)
    
    required_keys = ["video", "state", "action", "language"]
    issues = []
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key: {key}")
        else:
            print(f"✓ {key}: {config[key].modality_keys}")
    
    # Check video config
    if "video" in config:
        video_keys = config["video"].modality_keys
        expected_cameras = {"head", "left_wrist", "right_wrist"}
        if set(video_keys) != expected_cameras:
            issues.append(f"Video keys mismatch. Expected: {expected_cameras}, Got: {set(video_keys)}")
    
    # Check state config
    if "state" in config:
        state_keys = config["state"].modality_keys
        expected_state = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
        if state_keys != expected_state:
            issues.append(f"State keys mismatch. Expected: {expected_state}, Got: {state_keys}")
    
    # Check action config
    if "action" in config:
        action = config["action"]
        action_keys = action.modality_keys
        expected_action = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
        
        if action_keys != expected_action:
            issues.append(f"Action keys mismatch. Expected: {expected_action}, Got: {action_keys}")
        
        # Check action configs
        if action.action_configs:
            if len(action.action_configs) != len(action_keys):
                issues.append(f"Action configs count mismatch: {len(action.action_configs)} vs {len(action_keys)}")
            
            print(f"\n  Action Configurations:")
            for i, (key, ac) in enumerate(zip(action_keys, action.action_configs)):
                print(f"    {i}. {key}:")
                print(f"       - Representation: {ac.rep}")
                print(f"       - Type: {ac.type}")
                print(f"       - Format: {ac.format}")
                if ac.state_key:
                    print(f"       - State Key: {ac.state_key}")
    
    # Check language config
    if "language" in config:
        lang_keys = config["language"].modality_keys
        expected_lang = ["annotation.human.action.task_description"]
        if lang_keys != expected_lang:
            issues.append(f"Language keys mismatch. Expected: {expected_lang}, Got: {lang_keys}")
    
    # Check delta indices
    print(f"\n  Delta Indices:")
    for key in required_keys:
        if key in config:
            indices = config[key].delta_indices
            print(f"    {key}: {indices if len(indices) <= 5 else f'{indices[:3]}...{indices[-1:]} (total: {len(indices)})'}")
    
    # Report issues
    if issues:
        print(f"\n❌ Validation failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ {config_name} is valid!")
        return True


def main():
    """Main validation function."""
    print("🔍 Testing RobotWin Modality Configurations")
    print("="*80)
    
    all_valid = True
    
    # Test main config
    all_valid &= validate_config(robotwin_aloha_config, "robotwin_aloha_config")
    
    # Test absolute config
    all_valid &= validate_config(robotwin_aloha_absolute_config, "robotwin_aloha_absolute_config")
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if all_valid:
        print("\n✅ All configurations are valid!")
        print("\nYou can now use these configs for training:")
        print("  1. robotwin_aloha_config (recommended - relative arms, absolute grippers)")
        print("  2. robotwin_aloha_absolute_config (all absolute actions)")
        return 0
    else:
        print("\n❌ Some configurations have issues!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
