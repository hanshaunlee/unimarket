# UniMarket

UniMarket is a reproducible research framework for learning mechanistically constrained neural dynamics from public electrophysiology, with a hard separation between *prediction*, *mechanism*, and *interpretation*, and an automatic claim gate that prevents accidental over-claiming.

## What this project does and does NOT claim

**Claims it tries to support, with controls:**

- Predicts future voltage, spike probability, synaptic response, or population spike-counts on held-out time blocks / sessions / cells / protocols.
- Recovers known synthetic parameters and known synthetic graphs above chance under matched and misspecified noise.
- Produces frozen latents whose post-hoc probes beat permuted / random / shuffled-label controls.

**Claims it explicitly refuses to make without evidence:**

- Simulating an entire brain.
- Recovering biological connectivity from extracellular spike data.
- Generalizing across cells / sessions / protocols that were not actually held out.
- Biological meaning for a latent that merely improves reconstruction loss.

Every run produces a `claim_gate.md` enumerating allowed and disallowed claims.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[nwb]"   # Allen / DANDI NWB ingestion via pynwb
pip install -e ".[viz]"   # tensorboard logging
```

Python 3.11+ is required.

## Quickstart with synthetic data

```bash
python scripts/make_synthetic.py --config configs/data/synthetic.yaml --out data/processed/synthetic
python scripts/train.py --config configs/experiment/smoke_synthetic.yaml
python scripts/evaluate.py --run-dir outputs/runs/latest --split test
```

Or, end-to-end:

```bash
bash scripts/run_smoke.sh
```

Run tests:

```bash
pytest -q
```

## Data sources supported

| Source | Adapter status | Notes |
| --- | --- | --- |
| Synthetic LIF circuit | **fully functional** | Ground truth available for parameters and graph. |
| Generic NWB (DANDI) | **functional with local NWB** | `python scripts/scan_nwb.py path/to/file.nwb --out data/processed/nwb` (needs `[nwb]`). |
| Allen Cell Types | **functional with local NWB** | Point `UNIMARKET_ALLEN_CT_DIR` at a directory of `.nwb` files and run `unimarket prepare --config configs/data/allen_cell_types.yaml --out ...`. |
| Allen Synaptic Physiology | **schema_validated_only** | Expects a user-provided manifest CSV (columns documented in `src/unimarket/data/allen_synphys.py`). |
| Allen Neuropixels | **schema_validated_only** | Expects user-provided spike table CSV/Parquet. |

No adapter performs network downloads automatically.

## Commands

```bash
# Synthesize data
python scripts/make_synthetic.py --config configs/data/synthetic.yaml --out data/processed/synthetic

# Inspect an NWB file (requires .[nwb])
python scripts/scan_nwb.py /path/to/file.nwb --out data/processed/nwb

# Prepare any dataset from a data config
python scripts/prepare_dataset.py --config configs/data/allen_cell_types.yaml --out data/processed/allen_cell_types

# Train
python scripts/train.py --config configs/experiment/smoke_synthetic.yaml

# Evaluate
python scripts/evaluate.py --run-dir outputs/runs/latest --split test

# Leakage check (post hoc)
python -m unimarket.cli leakage-check --run-dir outputs/runs/latest

# CLI
unimarket --help
unimarket train --config configs/experiment/smoke_synthetic.yaml
unimarket evaluate --run-dir outputs/runs/latest --split test
unimarket leakage-check --run-dir outputs/runs/latest
unimarket plot --run-dir outputs/runs/latest
```

## Ablations

| Config | What it tests |
| --- | --- |
| `configs/experiment/single_cell_voltage.yaml` | Constrained LIF (full model) on synthetic single-cell voltage. |
| `configs/experiment/ablation_no_physics.yaml` | Unconstrained Neural ODE cell (no biophysics) vs the LIF baseline. |
| `configs/experiment/ablation_no_graph.yaml` | Spike transformer baseline (no graph) for population forecasting. |
| `configs/experiment/ablation_shuffle_graph.yaml` | Graph dynamics with shuffled (broken) edges; should not beat the no-graph baseline. |
| `configs/experiment/ablation_shuffle_stimulus.yaml` | Documents that stimulus identity is forbidden; the leakage check enforces this. |
| `configs/experiment/neuropixels_forecast.yaml` | Population forecasting with a learned graph on synthetic neuropixels-like data. |

## Expected outputs per run

After `train.py` + `evaluate.py`, a run directory contains:

```
outputs/runs/<timestamp>_<run_name>/
  config.json                          # frozen config snapshot
  config_source.yaml                   # copy of source yaml
  best.pt / last.pt                    # checkpoints
  events.jsonl, metrics.csv            # training logs
  normalization_stats.voltage.json     # train-only stats (when applicable)
  normalization_stats.current.json
  train_summary.json
  capacity_report.json
  predictive_metrics_by_split.csv
  predictive_metrics_summary.json
  rollout figures, voltage plots       # outputs/runs/.../figures/
  graph_recovery.json + .csv           # synthetic only
  synthetic_parameter_recovery.csv     # synthetic only
  identifiability_report.{json,md}
  interpretability_controls.csv
  interpretability_report.{json,md}
  leakage_report.json
  claim_gate.{json,md}
  evaluation_summary.json
  report_card.md
```

## Anti-circularity safeguards (hard checks)

Every run enforces:

- Random window-level splits raise unless `allow_leaky_split=true`.
- Normalization stats must be computed on `train` only.
- Checkpoint selection metric must come from `val`, never `test`.
- Forbidden stimulus features (`stimulus_id`, `trial_index`, `block_index`, `global_time`, etc.) raise unless the corresponding `allow_*` flag is true.
- Identity embeddings (`cell_id_embedding`, `pair_id_embedding`, `session_id_embedding`) raise unless explicitly allowed, and are *always* forbidden when the split holds out the same identity dimension.
- Synthetic ground truth is **never** used in the predictive loss.
- For extracellular data without anatomical ground truth, learned edges are reported as "functional predictive dependency", never as "synapse".

See `docs/anti_circularity.md`, `docs/identifiability.md`, `docs/claim_gating.md`, `docs/scientific_contract.md`, and `docs/dataset_limitations.md`.

## Reproducibility checklist

- Configs are versioned via YAML; full snapshot saved to each run dir.
- All randomness routed through `unimarket.utils.seed.set_seed`.
- Synthetic generation is deterministic for a fixed seed (tested).
- Git hash captured per run when available.
- No internet access required for tests.
- Heavy downloads are opt-in only.

## Troubleshooting

- "pynwb not installed": run `pip install -e ".[nwb]"` or skip NWB-based commands.
- Tests fail with `ImportError` for sklearn: it ships in the default `pip install -e .` because scikit-learn is in the runtime dependencies.
- Out-of-disk on synthetic generation: reduce `duration_s` and/or `n_neurons` in `configs/data/synthetic.yaml`.

## License

Apache-2.0.
