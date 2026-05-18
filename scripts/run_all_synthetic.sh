#!/usr/bin/env bash
# Run the main synthetic experiments + ablations end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_CONFIG="configs/data/synthetic.yaml"
PROCESSED_DIR="data/processed/synthetic"

python scripts/make_synthetic.py --config "$DATA_CONFIG" --out "$PROCESSED_DIR"

CONFIGS=(
    "configs/experiment/smoke_synthetic.yaml"
    "configs/experiment/single_cell_voltage.yaml"
    "configs/experiment/neuropixels_forecast.yaml"
    "configs/experiment/ablation_no_physics.yaml"
    "configs/experiment/ablation_no_graph.yaml"
    "configs/experiment/ablation_shuffle_graph.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=== $cfg ==="
    python scripts/train.py --config "$cfg"
    LATEST="outputs/runs/latest"
    python scripts/evaluate.py --run-dir "$LATEST" --split test
done

echo "All synthetic experiments complete."
