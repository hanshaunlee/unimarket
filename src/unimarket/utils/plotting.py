"""Matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=120)
    plt.close()


def plot_voltage_pred_vs_true(
    true: np.ndarray, pred: np.ndarray, out_path: str | Path, title: str = "Voltage prediction"
) -> None:
    plt.figure(figsize=(8, 3))
    plt.plot(true, label="true")
    plt.plot(pred, label="pred", linestyle="--")
    plt.xlabel("time step")
    plt.ylabel("voltage (z)")
    plt.title(title)
    plt.legend()
    _save(out_path)


def plot_spike_raster(
    spikes: np.ndarray, out_path: str | Path, title: str = "Spike raster"
) -> None:
    """Plot a binary spike matrix [T, N] as a raster (time on x, neuron on y)."""
    t_idx, n_idx = np.where(spikes > 0)
    plt.figure(figsize=(8, 4))
    plt.scatter(t_idx, n_idx, s=1, c="black")
    plt.xlabel("time bin")
    plt.ylabel("neuron")
    plt.title(title)
    _save(out_path)


def plot_rollout_error(error_by_horizon: Sequence[float], out_path: str | Path) -> None:
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, len(error_by_horizon) + 1), error_by_horizon, marker="o")
    plt.xlabel("rollout horizon (steps)")
    plt.ylabel("MSE")
    plt.title("Rollout error vs horizon")
    _save(out_path)


def plot_adjacency(
    true_adj: np.ndarray, learned_adj: np.ndarray, out_path: str | Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(true_adj, cmap="gray_r", aspect="auto")
    axes[0].set_title("true adjacency")
    axes[1].imshow(learned_adj, cmap="gray_r", aspect="auto")
    axes[1].set_title("learned adjacency")
    for ax in axes:
        ax.set_xlabel("source neuron")
        ax.set_ylabel("target neuron")
    _save(out_path)


def plot_ablation_bars(
    names: list[str], values: list[float], out_path: str | Path, ylabel: str = "metric"
) -> None:
    plt.figure(figsize=(max(4, 1 + len(names) * 0.8), 3))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title("Ablation summary")
    _save(out_path)


def plot_synapse_response(
    true_trace: np.ndarray, pred_trace: np.ndarray, out_path: str | Path
) -> None:
    plt.figure(figsize=(6, 3))
    plt.plot(true_trace, label="true post")
    plt.plot(pred_trace, label="pred post", linestyle="--")
    plt.xlabel("time step")
    plt.ylabel("response (a.u.)")
    plt.legend()
    plt.title("Synapse response prediction")
    _save(out_path)
