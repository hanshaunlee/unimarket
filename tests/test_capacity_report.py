"""Capacity report is written after training."""

from pathlib import Path

from unimarket.train.train import train_from_config
from unimarket.utils.io import read_json


def test_capacity_report_after_training(tmp_path: Path):
    # Build a tiny synthetic config inline.
    cfg = {
        "run_name": "capacity_test",
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
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg))
    run_dir = train_from_config(cfg_path)
    cap = read_json(run_dir / "capacity_report.json")
    assert cap["n_trainable_params"] > 0
    assert cap["device"] in {"cpu", "cuda", "mps"}
    assert cap["n_steps"] > 0
