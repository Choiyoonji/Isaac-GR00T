#!/usr/bin/env python3
"""
Modality configuration for RobotWin ALOHA dual-arm robot dataset.

This configuration defines how the RobotWin data should be loaded and processed
for training with GR00T.

Robot: ALOHA dual-arm (Agilex chassis)
- 2 arms with 6 DOF each
- 2 grippers with 1 DOF each
- Total: 14 DOF (6 + 1 + 6 + 1)

Cameras:
- head: Top-down view
- left_wrist: Left wrist camera
- right_wrist: Right wrist camera

Dataset structure matches the modality.json created by convert_robotwin_to_lerobot.py:
{
    "state": {
        "left_arm": {"start": 0, "end": 6},
        "left_gripper": {"start": 6, "end": 7},
        "right_arm": {"start": 7, "end": 13},
        "right_gripper": {"start": 13, "end": 14}
    },
    "action": {
        "left_arm": {"start": 0, "end": 6},
        "left_gripper": {"start": 6, "end": 7},
        "right_arm": {"start": 7, "end": 13},
        "right_gripper": {"start": 13, "end": 14}
    },
    "video": {
        "head": {"original_key": "observation.images.head"},
        "left_wrist": {"original_key": "observation.images.left_wrist"},
        "right_wrist": {"original_key": "observation.images.right_wrist"}
    },
    "annotation": {
        "human.action.task_description": {}
    }
}
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# RobotWin ALOHA dual-arm configuration
robotwin_aloha_config = {
    # Video observations: 3 cameras
    "video": ModalityConfig(
        delta_indices=[0],  # Current frame only
        modality_keys=[
            "head",         # Top-down view camera
            "left_wrist",   # Left wrist camera
            "right_wrist",  # Right wrist camera
        ],
    ),
    
    # State observations: Proprioceptive states (14 DOF)
    "state": ModalityConfig(
        delta_indices=[0],  # Current state
        modality_keys=[
            "left_arm",        # Left arm joint positions (6 DOF)
            "left_gripper",    # Left gripper position (1 DOF)
            "right_arm",       # Right arm joint positions (6 DOF)
            "right_gripper",   # Right gripper position (1 DOF)
        ],
        # Apply sin/cos embedding to arm joint angles (typically in radians)
        # This helps the model better understand cyclic joint angles
        sin_cos_embedding_keys=[
            "left_arm",
            "right_arm",
        ],
    ),
    
    # Action predictions: 16-step prediction horizon (14 DOF)
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # Predict 16 steps into the future
        modality_keys=[
            "left_arm",        # Left arm joint actions (6 DOF)
            "left_gripper",    # Left gripper action (1 DOF)
            "right_arm",       # Right arm joint actions (6 DOF)
            "right_gripper",   # Right gripper action (1 DOF)
        ],
        action_configs=[
            # Left arm: Use relative actions for smoother control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,  # Delta from current state
                type=ActionType.NON_EEF,            # Joint space control
                format=ActionFormat.DEFAULT,         # Standard joint angle format
                state_key="left_arm",               # Reference state for relative actions
            ),
            # Left gripper: Use absolute actions for precise control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # Target position
                type=ActionType.NON_EEF,            # Gripper space control
                format=ActionFormat.DEFAULT,         # Standard gripper format
            ),
            # Right arm: Use relative actions for smoother control
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,  # Delta from current state
                type=ActionType.NON_EEF,            # Joint space control
                format=ActionFormat.DEFAULT,         # Standard joint angle format
                state_key="right_arm",              # Reference state for relative actions
            ),
            # Right gripper: Use absolute actions for precise control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # Target position
                type=ActionType.NON_EEF,            # Gripper space control
                format=ActionFormat.DEFAULT,         # Standard gripper format
            ),
        ],
    ),
    
    # Language annotations: Task descriptions
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",  # Task description annotations
        ],
    ),
}


# Alternative configuration: Absolute actions for all modalities
# Use this if you experience drifting with relative actions
robotwin_aloha_absolute_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head",
            "left_wrist",
            "right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        sin_cos_embedding_keys=[
            "left_arm",
            "right_arm",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            # All absolute actions
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}


# Register only the main configuration (comment out alternative config registration)
# Users can manually register the alternative config if needed
register_modality_config(robotwin_aloha_config)

# Note: robotwin_aloha_absolute_config is available but not registered by default
# To use it, comment out the above line and uncomment the line below:
# register_modality_config(robotwin_aloha_absolute_config)


# Export for easy importing
__all__ = [
    "robotwin_aloha_config",
    "robotwin_aloha_absolute_config",
]
