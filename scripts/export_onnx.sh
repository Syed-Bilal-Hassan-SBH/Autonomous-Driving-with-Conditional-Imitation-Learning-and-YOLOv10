#!/bin/bash
# ONNX Export and Quantization script
# Usage: ./export_onnx.sh [checkpoint] [output_dir]

# Default configuration
CHECKPOINT="${1:-models/checkpoints/best_model.pt}"
OUTPUT_DIR="${2:-models/weights}"

echo "=========================================="
echo "YOLOv10 Autonomous Driving - ONNX Export"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run export
python src/main.py \
    --mode export \
    --checkpoint "$CHECKPOINT" \
    2>&1 | tee models/logs/export_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Export completed. Models saved to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
