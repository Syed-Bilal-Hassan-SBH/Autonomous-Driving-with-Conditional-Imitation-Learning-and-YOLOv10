#!/bin/bash
# Evaluation script for YOLOv10 Autonomous Driving
# Usage: ./eval.sh [checkpoint] [device]

# Default configuration
CHECKPOINT="${1:-models/checkpoints/best_model.pt}"
DEVICE="${2:-auto}"

echo "=========================================="
echo "YOLOv10 Autonomous Driving - Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Device: $DEVICE"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Run evaluation
python src/main.py \
    --mode eval \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    2>&1 | tee models/logs/eval_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Evaluation completed."
