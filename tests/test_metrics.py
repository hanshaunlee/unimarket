"""Metric correctness tests."""

import numpy as np
import pytest

from unimarket.utils.metrics import (
    binary_auprc,
    binary_auroc,
    f1_at_threshold,
    mae,
    mse,
    pearson_corr,
    poisson_nll,
    rmse,
)


def test_mse_perfect():
    a = np.array([1.0, 2.0, 3.0])
    assert mse(a, a) == 0.0
    assert rmse(a, a) == 0.0
    assert mae(a, a) == 0.0


def test_pearson_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_corr(a, a) == 1.0
    assert pearson_corr(a, -a) == -1.0


def test_pearson_nan_for_constant():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    val = pearson_corr(a, b)
    assert np.isnan(val)


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert binary_auroc(scores, labels) == 1.0


def test_auroc_inverted():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert binary_auroc(scores, labels) == 0.0


def test_auprc_nonneg():
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    ap = binary_auprc(scores, labels)
    assert 0.0 <= ap <= 1.0


def test_f1_at_threshold():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    f1 = f1_at_threshold(scores, labels, 0.5)
    assert 0.0 <= f1 <= 1.0
    assert f1 == pytest.approx(1.0, abs=1e-6)


def test_poisson_nll_nonneg_for_zero_rate_small():
    rate = np.full(4, 0.5)
    counts = np.array([0.0, 1.0, 2.0, 3.0])
    val = poisson_nll(rate, counts)
    assert np.isfinite(val)
