#!/bin/bash
set -e

export MPLCONFIGDIR=/tmp
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTHONUNBUFFERED=1

PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/envs/qfusion/bin/python"
DATA_DIR="/Volumes/Jonath SSD1/Research/Quantum Fusion/data"

echo "========================================================================"
echo "--- Q-FUSION MULTI-SEED PIPELINE (SEEDS: 101 & 2024) ---"
echo "========================================================================"

# Run Seed 101 (will automatically skip already completed models)
echo ">>> Executing Seed 101 Matrix..."
$PYTHON_BIN src/run_experiment_b.py \
  --data_dir "$DATA_DIR" \
  --modality both \
  --model all \
  --seed 101 \
  --epochs 20 \
  --patience 5 \
  --batch_size 2

# Run Seed 2024
echo ">>> Executing Seed 2024 Matrix..."
$PYTHON_BIN src/run_experiment_b.py \
  --data_dir "$DATA_DIR" \
  --modality both \
  --model all \
  --seed 2024 \
  --epochs 20 \
  --patience 5 \
  --batch_size 2

# Aggregate statistical significance report
echo ">>> Aggregating Multi-Seed Statistics..."
$PYTHON_BIN src/compute_multiseed_stats.py

echo "========================================================================"
echo "--- Multi-Seed Pipeline Execution Complete ---"
echo "========================================================================"
