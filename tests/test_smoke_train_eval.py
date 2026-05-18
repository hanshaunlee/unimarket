"""End-to-end smoke test: train + evaluate on tiny synthetic data."""

from __future__ import annotations

from pathlib import Path

import yaml

from unimarket.eval.evaluate import evaluate_run
from unimarket.eval.report_cards import write_report_card
from unimarket.train.train import train_from_config


def test_smoke_train_then_evaluate(tmp_path: Path):
    cfg = {
        "run_name": "smoke_unittest",
        "seed": 0,
        "data": {
            "kind": "synthetic",
            "processed_dir": str(tmp_path / "synth"),
            "regenerate": True,
            "split_mode": "time_block",
            "allow_leaky_split": False,
            "model_features": ["history_counts"],
            "synthetic": {
                "n_neurons": 6,
                "duration_s": 0.5,
                "dt_ms": 1.0,
                "seed": 0,
                "use_adaptation": True,
            },
        },
        "window_bins": 6,
        "horizon_bins": 2,
        "model": {
            "name": "graph_latent_dynamics",
            "hidden_dim": 8,
            "graph_mode": "learned",
            "uses": {},
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 1e-3,
            "device": "cpu",
            "log_every_n_steps": 5,
        },
        "selection_split": "val",
        "selection_metric": "loss",
        "allow_stimulus_identity": False,
        "allow_global_time_features": False,
        "allow_trial_index": False,
        "allow_block_index": False,
        "allow_session_id_embedding": False,
    }
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_dir = train_from_config(cfg_path)
    assert (run_dir / "best.pt").exists() or (run_dir / "last.pt").exists()
    summary = evaluate_run(run_dir, splits=["val", "test"])
    write_report_card(run_dir)
    # Required artifacts present.
    for name in [
        "config.json",
        "predictive_metrics_by_split.csv",
        "predictive_metrics_summary.json",
        "leakage_report.json",
        "claim_gate.json",
        "claim_gate.md",
        "capacity_report.json",
        "report_card.md",
        "evaluation_summary.json",
    ]:
        assert (run_dir / name).exists(), f"Missing artifact: {name}"
    # Claim gate must produce some allowed claims (synthetic predictive results).
    assert "allowed" in summary["claim_gate"]
