"""Preprocessing: normalization (train-only), spike binning, windowing, masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import write_json
from ..utils.validation import assert_normalization_from_train_only


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    computed_on_splits: list[str] = field(default_factory=lambda: ["train"])
    method: str = "zscore"

    def to_meta(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mean_shape": list(self.mean.shape),
            "std_shape": list(self.std.shape),
            "computed_on_splits": self.computed_on_splits,
        }


def fit_norm_stats(
    x: np.ndarray, train_mask_T: np.ndarray | None = None, method: str = "zscore"
) -> NormStats:
    """Fit per-feature normalization stats from training data only.

    ``x`` has shape [T, ...]. ``train_mask_T`` is a 1-D boolean over time of length T
    indicating training timesteps. If None, all timesteps are treated as train.
    """
    x = np.asarray(x, dtype=np.float64)
    if train_mask_T is None:
        train_mask_T = np.ones(x.shape[0], dtype=bool)
    if train_mask_T.sum() == 0:
        raise ValueError("Cannot fit normalization stats: training mask selects 0 timesteps.")
    train = x[train_mask_T]
    if method == "zscore":
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    elif method == "robust":
        med = np.median(train, axis=0)
        mad = np.median(np.abs(train - med), axis=0)
        std = np.where(mad < 1e-6, 1.0, mad * 1.4826)
        mean = med
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return NormStats(mean=mean, std=std, computed_on_splits=["train"], method=method)


def apply_norm(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return (x - stats.mean) / stats.std


def invert_norm(x_norm: np.ndarray, stats: NormStats) -> np.ndarray:
    return x_norm * stats.std + stats.mean


def write_norm_stats(stats: NormStats, path: str | Path) -> None:
    payload = {
        "method": stats.method,
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
        "computed_on_splits": stats.computed_on_splits,
    }
    # This will raise via the validator at load-time if violated.
    assert_normalization_from_train_only({"computed_on_splits": stats.computed_on_splits})
    write_json(path, payload)


def bin_spikes(spike_times_s: np.ndarray, T_s: float, bin_size_s: float) -> np.ndarray:
    """Bin a 1-D array of spike times (seconds) into counts."""
    n_bins = int(np.ceil(T_s / bin_size_s))
    edges = np.arange(n_bins + 1) * bin_size_s
    counts, _ = np.histogram(spike_times_s, bins=edges)
    return counts.astype(np.int64)


def bin_spike_matrix(
    spikes_per_step: np.ndarray, dt_ms: float, bin_size_ms: float
) -> np.ndarray:
    """Bin a [T, N] integer spike matrix into [T_bins, N] counts."""
    T, N = spikes_per_step.shape
    factor = max(1, int(round(bin_size_ms / dt_ms)))
    pad = (-T) % factor
    if pad:
        padded = np.concatenate([spikes_per_step, np.zeros((pad, N), dtype=spikes_per_step.dtype)], axis=0)
    else:
        padded = spikes_per_step
    T2 = padded.shape[0]
    binned = padded.reshape(T2 // factor, factor, N).sum(axis=1)
    return binned.astype(np.int64)


def make_windows(
    x: np.ndarray, window_size: int, horizon: int, stride: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Slice ``x`` of shape [T, ...] into windows.

    Returns (history, future) arrays of shape:
        history: [B, window_size, ...]
        future:  [B, horizon, ...]
    """
    if stride is None:
        stride = max(1, horizon)
    T = x.shape[0]
    starts = list(range(0, T - window_size - horizon + 1, stride))
    if not starts:
        raise ValueError(
            f"Cannot make windows: T={T} window_size={window_size} horizon={horizon}"
        )
    hist = np.stack([x[s : s + window_size] for s in starts], axis=0)
    fut = np.stack([x[s + window_size : s + window_size + horizon] for s in starts], axis=0)
    return hist, fut


def windows_in_time_blocks(
    T: int,
    time_blocks: list[tuple[int, int]],
    window_size: int,
    horizon: int,
    stride: int | None = None,
) -> list[int]:
    """Return window start indices that lie entirely within at least one time block.

    A window ``[s, s+window_size+horizon)`` is admitted if it is fully contained in
    a single block. This prevents windows from straddling split boundaries.
    """
    if stride is None:
        stride = max(1, horizon)
    starts: list[int] = []
    for (a, b) in time_blocks:
        first = a
        last = b - window_size - horizon
        if last < first:
            continue
        for s in range(first, last + 1, stride):
            starts.append(s)
    return sorted(set(starts))
