"""Bootstrap / paired-comparison statistics for evaluation reports."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 0
) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    alpha = 1.0 - ci
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "sem": float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "lo": lo,
        "hi": hi,
        "n": int(n),
    }


def paired_metric_delta(
    a: Iterable[float], b: Iterable[float], n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    a_arr = np.asarray(list(a), dtype=np.float64)
    b_arr = np.asarray(list(b), dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        return {"delta_mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_paired": float("nan")}
    diff = a_arr - b_arr
    res = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    # Simple sign test as a robust p-value proxy.
    pos = int((diff > 0).sum())
    n = int(diff.size)
    # Two-sided binomial-style normal approximation.
    p_paired = float(2 * min(pos, n - pos) / max(1, n))
    return {
        "delta_mean": res["mean"],
        "lo": res["lo"],
        "hi": res["hi"],
        "p_paired_signtest_proxy": p_paired,
        "n": n,
    }


def summarize_across_seeds(values_by_seed: dict[int, float]) -> dict[str, float]:
    arr = np.asarray(list(values_by_seed.values()), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n_seeds": 0}
    return {
        "mean": float(arr.mean()),
        "sem": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "n_seeds": int(arr.size),
    }


def format_mean_sem(mean: float, sem: float, precision: int = 4) -> str:
    if math.isnan(mean):
        return "nan"
    return f"{mean:.{precision}f} ± {sem:.{precision}f}"
