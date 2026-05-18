"""Synthetic generation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unimarket.data.synthetic import SyntheticConfig, simulate, write_synthetic


def test_simulate_shapes_and_ground_truth():
    cfg = SyntheticConfig(n_neurons=8, duration_s=1.0, dt_ms=1.0, seed=0)
    out = simulate(cfg)
    T = int(round(cfg.duration_s / (cfg.dt_ms / 1000.0)))
    assert out["voltage"].shape == (T, 8)
    assert out["spikes"].shape == (T, 8)
    assert out["current"].shape == (T, 8)
    assert out["edge_index"].shape[0] == 2
    assert out["edge_weight"].shape[0] == out["edge_index"].shape[1]
    assert out["edge_delay_bins"].shape[0] == out["edge_index"].shape[1]
    assert (out["edge_delay_bins"] >= 1).all()
    # No self-loops in the synthetic graph.
    if out["edge_index"].shape[1] > 0:
        assert (out["edge_index"][0] != out["edge_index"][1]).all()
    # Voltage stays in a sane range (clipped in simulate).
    assert out["voltage"].min() >= -120.0 - 1e-6
    assert out["voltage"].max() <= 60.0 + 1e-6


def test_write_synthetic_produces_manifest(tmp_path: Path):
    cfg = SyntheticConfig(n_neurons=6, duration_s=1.0, dt_ms=1.0, seed=1)
    info = write_synthetic(tmp_path, cfg)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "voltage.npy").exists()
    assert (tmp_path / "spikes.npy").exists()
    assert (tmp_path / "current.npy").exists()
    assert (tmp_path / "edge_index.npy").exists()
    assert (tmp_path / "params.csv").exists()
    assert (tmp_path / "splits.json").exists()
    assert info["N"] == 6
    assert info["T"] > 0
    # Time blocks must be disjoint.
    splits = info["splits"]["time_blocks"]
    assert splits["train"][0][1] == splits["val"][0][0]
    assert splits["val"][0][1] == splits["test"][0][0]


def test_synthetic_is_deterministic():
    cfg = SyntheticConfig(n_neurons=4, duration_s=0.5, dt_ms=1.0, seed=123)
    a = simulate(cfg)
    b = simulate(cfg)
    assert np.allclose(a["voltage"], b["voltage"])
    assert np.array_equal(a["spikes"], b["spikes"])
    assert np.array_equal(a["edge_index"], b["edge_index"])
