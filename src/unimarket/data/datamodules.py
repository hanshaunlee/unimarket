"""PyTorch datasets and dataloaders for UniMarket synthetic data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.io import read_json
from .preprocess import (
    NormStats,
    apply_norm,
    bin_spike_matrix,
    fit_norm_stats,
    windows_in_time_blocks,
    write_norm_stats,
)


@dataclass
class SyntheticBundle:
    voltage: np.ndarray  # [T, N]
    spikes: np.ndarray   # [T, N]
    current: np.ndarray  # [T, N]
    edge_index: np.ndarray  # [2, E]
    edge_weight: np.ndarray  # [E]
    edge_delay_bins: np.ndarray  # [E]
    dt_s: float
    splits: dict[str, list[tuple[int, int]]]
    heldout_neurons: np.ndarray
    seen_neurons: np.ndarray
    window_size: int
    horizon: int
    bin_size_ms: float


def load_synthetic_bundle(processed_dir: str | Path) -> SyntheticBundle:
    d = Path(processed_dir)
    voltage = np.load(d / "voltage.npy")
    spikes = np.load(d / "spikes.npy")
    current = np.load(d / "current.npy")
    edge_index = np.load(d / "edge_index.npy")
    edge_weight = np.load(d / "edge_weight.npy")
    edge_delay = np.load(d / "edge_delay_bins.npy")
    splits_meta = read_json(d / "splits.json")
    return SyntheticBundle(
        voltage=voltage,
        spikes=spikes,
        current=current,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_delay_bins=edge_delay,
        dt_s=float(splits_meta["dt_s"]),
        splits={k: [tuple(b) for b in v] for k, v in splits_meta["time_blocks"].items()},
        heldout_neurons=np.array(splits_meta["heldout_neurons"], dtype=np.int64),
        seen_neurons=np.array(splits_meta["seen_neurons"], dtype=np.int64),
        window_size=int(splits_meta["window_size"]),
        horizon=int(splits_meta["horizon"]),
        bin_size_ms=float(splits_meta["bin_size_ms"]),
    )


class SyntheticVoltageDataset(Dataset):
    """Windowed dataset for single-neuron voltage prediction.

    Each sample provides past voltage and past current; target is next-h voltage.
    """

    def __init__(
        self,
        bundle: SyntheticBundle,
        time_blocks: list[tuple[int, int]],
        norm_voltage: NormStats,
        norm_current: NormStats,
        neuron_idx: int | None = None,
        window_size: int | None = None,
        horizon: int | None = None,
        stride: int | None = None,
    ):
        self.bundle = bundle
        self.window_size = window_size or bundle.window_size
        self.horizon = horizon or bundle.horizon
        self.starts = windows_in_time_blocks(
            T=bundle.voltage.shape[0],
            time_blocks=time_blocks,
            window_size=self.window_size,
            horizon=self.horizon,
            stride=stride,
        )
        self.neuron_idx = neuron_idx
        self.norm_voltage = norm_voltage
        self.norm_current = norm_current

    def __len__(self) -> int:
        if self.neuron_idx is None:
            return len(self.starts) * self.bundle.voltage.shape[1]
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.neuron_idx is None:
            n_neurons = self.bundle.voltage.shape[1]
            s_idx = idx // n_neurons
            n = idx % n_neurons
        else:
            s_idx = idx
            n = self.neuron_idx
        s = self.starts[s_idx]
        ws, h = self.window_size, self.horizon
        v_full = self.bundle.voltage[s : s + ws + h, n]
        i_full = self.bundle.current[s : s + ws + h, n]
        # Per-neuron normalization: use the n-th entry of the per-neuron stats.
        v_mean = self.norm_voltage.mean[n] if np.ndim(self.norm_voltage.mean) > 0 else self.norm_voltage.mean
        v_std = self.norm_voltage.std[n] if np.ndim(self.norm_voltage.std) > 0 else self.norm_voltage.std
        i_mean = self.norm_current.mean[n] if np.ndim(self.norm_current.mean) > 0 else self.norm_current.mean
        i_std = self.norm_current.std[n] if np.ndim(self.norm_current.std) > 0 else self.norm_current.std
        v_norm = (v_full - v_mean) / v_std
        i_norm = (i_full - i_mean) / i_std
        hist_v = v_norm[:ws]
        hist_i = i_norm[:ws]
        target_v = v_norm[ws : ws + h]
        sp = self.bundle.spikes[s + ws : s + ws + h, n].astype(np.float32)
        return {
            "history_voltage": torch.from_numpy(hist_v).float(),
            "history_current": torch.from_numpy(hist_i).float(),
            "target_voltage": torch.from_numpy(target_v).float(),
            "target_spikes": torch.from_numpy(sp).float(),
            "neuron_idx": torch.tensor(int(n), dtype=torch.long),
        }


class SyntheticPopulationDataset(Dataset):
    """Windowed dataset for population spike-count forecasting."""

    def __init__(
        self,
        bundle: SyntheticBundle,
        time_blocks: list[tuple[int, int]],
        bin_size_ms: float | None = None,
        window_bins: int = 20,
        horizon_bins: int = 5,
        stride: int | None = None,
        mask_neurons: np.ndarray | None = None,
    ):
        self.bundle = bundle
        dt_ms = bundle.dt_s * 1000.0
        self.bin_size_ms = bin_size_ms or bundle.bin_size_ms
        binned = bin_spike_matrix(bundle.spikes, dt_ms=dt_ms, bin_size_ms=self.bin_size_ms)
        self.binned = binned.astype(np.float32)
        T_bins = binned.shape[0]
        factor = self.bin_size_ms / dt_ms
        block_bins = []
        for (a, b) in time_blocks:
            block_bins.append((int(np.floor(a / factor)), int(np.floor(b / factor))))
        self.block_bins = block_bins
        self.window_bins = window_bins
        self.horizon_bins = horizon_bins
        self.starts = windows_in_time_blocks(
            T=T_bins,
            time_blocks=block_bins,
            window_size=window_bins,
            horizon=horizon_bins,
            stride=stride,
        )
        self.mask_neurons = (
            np.asarray(mask_neurons, dtype=np.int64) if mask_neurons is not None else None
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.starts[idx]
        w, h = self.window_bins, self.horizon_bins
        hist = self.binned[s : s + w]
        fut = self.binned[s + w : s + w + h]
        item = {
            "history_counts": torch.from_numpy(hist).float(),
            "future_counts": torch.from_numpy(fut).float(),
            "window_start_bin": torch.tensor(int(s), dtype=torch.long),
        }
        if self.mask_neurons is not None:
            mask = np.ones(hist.shape[1], dtype=np.float32)
            mask[self.mask_neurons] = 0.0
            item["mask"] = torch.from_numpy(mask)
        return item


def build_synthetic_voltage_loaders(
    bundle: SyntheticBundle,
    batch_size: int = 32,
    horizon: int | None = None,
    window_size: int | None = None,
    out_norm_path: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build train/val/test loaders for single-neuron voltage prediction."""
    train_blocks = bundle.splits["train"]
    # Fit normalization on TRAIN time only, across all neurons.
    train_mask_T = np.zeros(bundle.voltage.shape[0], dtype=bool)
    for (a, b) in train_blocks:
        train_mask_T[a:b] = True
    # Per-neuron stats (flatten to broadcastable [N])
    norm_v = fit_norm_stats(bundle.voltage, train_mask_T=train_mask_T)
    norm_i = fit_norm_stats(bundle.current, train_mask_T=train_mask_T)
    if out_norm_path is not None:
        write_norm_stats(norm_v, Path(out_norm_path).with_suffix(".voltage.json"))
        write_norm_stats(norm_i, Path(out_norm_path).with_suffix(".current.json"))

    g = torch.Generator()
    g.manual_seed(seed)
    loaders: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        ds = SyntheticVoltageDataset(
            bundle=bundle,
            time_blocks=bundle.splits[split],
            norm_voltage=norm_v,
            norm_current=norm_i,
            window_size=window_size,
            horizon=horizon,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            generator=g if split == "train" else None,
        )
    loaders["norm_voltage"] = norm_v
    loaders["norm_current"] = norm_i
    return loaders


def build_synthetic_population_loaders(
    bundle: SyntheticBundle,
    batch_size: int = 16,
    window_bins: int = 20,
    horizon_bins: int = 5,
    mask_neurons: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = SyntheticPopulationDataset(
            bundle=bundle,
            time_blocks=bundle.splits[split],
            window_bins=window_bins,
            horizon_bins=horizon_bins,
            mask_neurons=mask_neurons,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            generator=g if split == "train" else None,
        )
    return loaders
