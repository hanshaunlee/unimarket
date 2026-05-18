"""Claim gating behavior tests."""

from __future__ import annotations

from pathlib import Path

from unimarket.eval.claim_gating import compute_claim_gate
from unimarket.utils.io import write_json


def _seed_run(tmp_path: Path, cfg: dict, **extra) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "config.json", cfg)
    for name, payload in extra.items():
        write_json(run / f"{name}.json", payload)
    return run


def test_invalid_when_leakage_fails(tmp_path: Path):
    run = _seed_run(
        tmp_path,
        {"data": {"kind": "synthetic"}, "selection_split": "val", "model": {"name": "lif_residual"}},
        leakage_report={"status": "FAIL", "checks": []},
    )
    gate = compute_claim_gate(run)
    assert gate["status"] == "INVALID"


def test_invalid_when_checkpoint_uses_test(tmp_path: Path):
    run = _seed_run(
        tmp_path,
        {
            "data": {"kind": "synthetic"},
            "selection_split": "test",
            "model": {"name": "lif_residual"},
        },
        leakage_report={"status": "PASS", "checks": []},
    )
    gate = compute_claim_gate(run)
    assert gate["status"] == "INVALID"


def test_valid_with_allowed_claims(tmp_path: Path):
    cfg = {
        "data": {"kind": "synthetic", "split_mode": "time_block"},
        "selection_split": "val",
        "model": {"name": "lif_residual", "uses": {}},
    }
    run = _seed_run(
        tmp_path,
        cfg,
        leakage_report={"status": "PASS", "checks": []},
        predictive_metrics_summary={"val": {"mse": 0.1}, "test": {"mse": 0.2}},
        capacity_report={"n_trainable_params": 100, "wall_clock_s": 1.0, "n_epochs": 1, "n_steps": 10, "device": "cpu"},
    )
    gate = compute_claim_gate(run)
    assert gate["status"] == "VALID"
    assert any("validation split" in c["claim"] for c in gate["allowed"])
    assert any("test split" in c["claim"] for c in gate["allowed"])


def test_synthetic_disallows_real_biology(tmp_path: Path):
    cfg = {
        "data": {"kind": "synthetic", "split_mode": "time_block"},
        "selection_split": "val",
        "model": {"name": "lif_residual", "uses": {}},
    }
    run = _seed_run(tmp_path, cfg, leakage_report={"status": "PASS", "checks": []})
    gate = compute_claim_gate(run)
    assert gate["status"] == "VALID"
    assert any("public biological recordings" in c["claim"] for c in gate["disallowed"])
