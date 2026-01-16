#!/bin/bash
set -x -e

# Test finetuning with Camera MoE on ALOHA RobotWin data
export NUM_GPUS=1

torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path robotwin/aloha-agilex_clean_50_labeled \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path robotwin/robotwin_modality_config.py \
    --num_gpus $NUM_GPUS \
    --output_dir robotwin/checkpoints/aloha_camera_moe_test \
    --save_steps 50 \
    --save_total_limit 3 \
    --max_steps 100 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 2 \
    --use_camera_moe \
    --camera_routing_loss_weight 0.1 \
    --use_wandb
