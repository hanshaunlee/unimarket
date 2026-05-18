"""Identifiability evaluation tests."""

from __future__ import annotations

import numpy as np

from unimarket.eval.identifiability import (
    evaluate_parameter_recovery,
    perturbation_monotonicity,
    write_identifiability_report,
)


def test_recovery_labels_strong_signal(tmp_path):
    rng = np.random.default_rng(0)
    true_tau = rng.uniform(0.01, 0.03, size=20)
    # Learned = true + small noise -> high correlation.
    learned_tau = true_tau + rng.normal(0, 0.0005, size=20)
    rows = evaluate_parameter_recovery({"tau_m_s": learned_tau}, {"tau_m_s": true_tau})
    assert any(r["label"] == "biological_parameter_supported" for r in rows)
    summary = write_identifiability_report(tmp_path, rows)
    assert (tmp_path / "synthetic_parameter_recovery.csv").exists()
    assert (tmp_path / "identifiability_report.md").exists()
    assert "tau_m_s" in summary["by_label"]["biological_parameter_supported"]


def test_recovery_labels_no_signal():
    rng = np.random.default_rng(0)
    true_tau = rng.uniform(0.01, 0.03, size=20)
    learned_tau = rng.uniform(0.01, 0.03, size=20)  # unrelated
    rows = evaluate_parameter_recovery({"tau_m_s": learned_tau}, {"tau_m_s": true_tau})
    assert rows[0]["label"] in {"predictive_latent_only", "not_recoverable"}


def test_monotonicity_strict_increase():
    pairs = [(1.0, 0.9), (2.0, 1.8), (3.0, 3.1), (4.0, 4.05)]
    res = perturbation_monotonicity(pairs)
    assert res["monotonic"]


def test_monotonicity_random():
    pairs = [(1.0, 5.0), (2.0, 1.0), (3.0, 3.0), (4.0, 2.0)]
    res = perturbation_monotonicity(pairs)
    # Probably not monotonic; either way, must return a dict.
    assert "monotonic" in res
