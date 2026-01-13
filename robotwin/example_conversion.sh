#!/bin/bash
# Example script to convert RobotWin dataset and validate
# Usage: bash example_conversion.sh

set -e  # Exit on error

echo "======================================================================"
echo "RobotWin to GR00T LeRobot v2 Conversion Example"
echo "======================================================================"
echo ""

# Configuration
INPUT_DIR="aloha-agilex_clean_50"
OUTPUT_DIR="aloha-agilex_clean_50_lerobot"
ROBOT_TYPE="aloha_agilex"
FPS=30
CHUNK_SIZE=1000

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory '$INPUT_DIR' not found!"
    echo "Please download or specify the correct path to your RobotWin dataset."
    exit 1
fi

echo "📁 Input directory: $INPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🤖 Robot type: $ROBOT_TYPE"
echo "🎬 FPS: $FPS"
echo "📦 Chunk size: $CHUNK_SIZE"
echo ""

# Step 1: Convert dataset
echo "======================================================================"
echo "Step 1: Converting RobotWin dataset to LeRobot v2 format"
echo "======================================================================"
echo ""

python3 convert_robotwin_to_lerobot.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --robot_type "$ROBOT_TYPE" \
    --fps "$FPS" \
    --chunk_size "$CHUNK_SIZE" \
    --use_seen_instructions

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Conversion failed!"
    exit 1
fi

echo ""
echo "✅ Conversion completed successfully!"
echo ""

# Step 2: Validate conversion
echo "======================================================================"
echo "Step 2: Validating conversion"
echo "======================================================================"
echo ""

python3 validate_conversion.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Validation failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ All steps completed successfully!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Review the converted dataset in: $OUTPUT_DIR"
echo "2. Use the modality config: robotwin_modality_config.py"
echo "3. Start training with GR00T:"
echo ""
echo "   python -m gr00t.experiment.launch_finetune \\"
echo "       --config configs/finetune_config.py \\"
echo "       --dataset_path $OUTPUT_DIR \\"
echo "       --modality_config_path robotwin/robotwin_modality_config.py \\"
echo "       --output_dir /path/to/output"
echo ""
