#!/bin/bash
# Full experimental pipeline for MU Transformer

set -e

echo "=========================================="
echo "MU TRANSFORMER EXPERIMENTAL PIPELINE"
echo "=========================================="
echo ""

# Check if we're in test mode
TEST_MODE=""
if [ "$1" == "--test" ]; then
    TEST_MODE="--test_mode"
    echo "Running in TEST MODE (minimal data)"
    echo ""
fi

# Download datasets
echo "Step 1: Downloading datasets..."
bash scripts/download_data.sh
echo ""

# Train MU model
echo "Step 2: Training MU Transformer..."
python scripts/train.py \
    --config configs/mu_small.yaml \
    --model mu \
    --seed 42 \
    $TEST_MODE

echo ""

# Train baseline model
echo "Step 3: Training Baseline Transformer..."
python scripts/train.py \
    --config configs/baseline_small.yaml \
    --model baseline \
    --seed 42 \
    $TEST_MODE

echo ""

# Evaluate MU model
echo "Step 4: Evaluating MU Transformer..."
python scripts/evaluate.py \
    --checkpoint results/checkpoints/mu/best_model.pt \
    --task all \
    --output results/mu_results.txt

echo ""

# Evaluate baseline model
echo "Step 5: Evaluating Baseline Transformer..."
python scripts/evaluate.py \
    --checkpoint results/checkpoints/baseline/best_model.pt \
    --task all \
    --output results/baseline_results.txt

echo ""

echo "=========================================="
echo "EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/mu_results.txt"
echo "  - results/baseline_results.txt"
echo "  - results/logs/"
echo ""
echo "To view results, run:"
echo "  cat results/mu_results.txt"
echo "  cat results/baseline_results.txt"
