#!/bin/bash
# Training script for YOLOv10 Autonomous Driving
# Usage: ./train.sh [config] [device]

# Default configuration
CONFIG="${1:-configs/training_config.yaml}"
DEVICE="${2:-auto}"

# Create logs directory
mkdir -p models/logs

echo "=========================================="
echo "YOLOv10 Autonomous Driving - Training"
echo "=========================================="
echo "Configuration: $CONFIG"
echo "Device: $DEVICE"
echo ""

# Run training
python src/main.py \
    --mode train \
    --config "$CONFIG" \
    --device "$DEVICE" \
    2>&1 | tee models/logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training completed. Logs saved to models/logs/"
