#!/bin/bash
# CPU Benchmark script for YOLOv10 Autonomous Driving
# Usage: ./benchmark_cpu.sh [checkpoint] [batch_sizes] [num_runs]

# Default configuration
CHECKPOINT="${1:-models/checkpoints/best_model.pt}"
BATCH_SIZES="${2:-1,2,4,8,16}"
NUM_RUNS="${3:-100}"

echo "=========================================="
echo "YOLOv10 Autonomous Driving - CPU Benchmark"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Batch Sizes: $BATCH_SIZES"
echo "Number of Runs: $NUM_RUNS"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Create logs directory
mkdir -p models/logs

# Run benchmark
python scripts/benchmark_cpu.py \
    --checkpoint "$CHECKPOINT" \
    --batch-sizes "$BATCH_SIZES" \
    --num-runs "$NUM_RUNS" \
    2>&1 | tee models/logs/benchmark_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Benchmark completed. Results saved to models/logs/"
