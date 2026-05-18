#!/usr/bin/env bash
# Smoke test: generate synthetic data, train, evaluate end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/experiment/smoke_synthetic.yaml"
DATA_CONFIG="configs/data/synthetic.yaml"
PROCESSED_DIR="data/processed/synthetic"

python scripts/make_synthetic.py --config "$DATA_CONFIG" --out "$PROCESSED_DIR"
python scripts/train.py --config "$CONFIG"
LATEST="outputs/runs/latest"
python scripts/evaluate.py --run-dir "$LATEST" --split test
echo
echo "Smoke complete. Outputs in $LATEST"
