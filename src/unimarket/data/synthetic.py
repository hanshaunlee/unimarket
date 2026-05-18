"""Synthetic mechanistic data generator.

Implements a small leaky integrate-and-fire (LIF) circuit with:
- excitatory / inhibitory neuron labels
- random sparse directed coupling with delays
- optional adaptation current
- external injected current
- Gaussian observation noise on voltage
- writes voltage, spikes, current, true graph, true parameters, and a manifest
- writes split metadata via :func:`splits.split_synthetic_by_graph_and_seed`
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import save_npy, write_json
from ..utils.seed import seeded_rng


@dataclass
class SyntheticConfig:
    n_neurons: int = 32
    duration_s: float = 20.0
    dt_ms: float = 1.0
    connection_prob: float = 0.08
    excitatory_fraction: float = 0.8
    noise_std: float = 0.5  # observation noise on voltage (z-units)
    seed: int = 42

    # Neuron biophysics
    tau_m_ms_range: tuple[float, float] = (10.0, 30.0)
    R_m_range: tuple[float, float] = (0.8, 1.2)
    threshold_range: tuple[float, float] = (-50.0, -45.0)
    reset_voltage: float = -65.0
    E_L: float = -65.0
    refractory_ms: float = 3.0

    # Synapse
    weight_exc_range: tuple[float, float] = (0.4, 1.2)
    weight_inh_range: tuple[float, float] = (0.4, 1.5)  # magnitude; will be negated for inhibitory
    delay_ms_range: tuple[float, float] = (1.0, 5.0)

    # Adaptation
    use_adaptation: bool = True
    tau_w_ms: float = 80.0
    a_adapt: float = 0.02
    b_adapt: float = 0.5

    # External input
    stim_amplitude_range: tuple[float, float] = (8.0, 16.0)
    stim_on_prob: float = 0.4

    # Splits
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # Discretization helpers
    window_size: int = 100
    horizon: int = 10
    bin_size_ms: float = 10.0

    # Extras
    held_out_neuron_fraction: float = 0.15  # for held-out-neuron eval

    extra: dict[str, Any] = field(default_factory=dict)


def _sample_in(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(lo, hi, size=n)


def simulate(cfg: SyntheticConfig) -> dict[str, Any]:
    """Run the LIF circuit simulation.

    Returns a dict with arrays: voltage [T,N], spikes [T,N], current [T,N],
    edge_index [2,E], edge_weight [E], edge_delay_bins [E], params_df (DataFrame).
    """
    rng = seeded_rng(cfg.seed)
    dt = cfg.dt_ms / 1000.0  # seconds
    T = int(round(cfg.duration_s / dt))
    N = cfg.n_neurons

    # Cell parameters
    tau_m_ms = _sample_in(rng, N, *cfg.tau_m_ms_range)
    R_m = _sample_in(rng, N, *cfg.R_m_range)
    thresh = _sample_in(rng, N, *cfg.threshold_range)
    n_exc = int(round(cfg.excitatory_fraction * N))
    is_exc = np.zeros(N, dtype=bool)
    exc_idx = rng.choice(N, size=n_exc, replace=False)
    is_exc[exc_idx] = True

    # Random sparse directed graph; no self-loops.
    adj = (rng.random((N, N)) < cfg.connection_prob).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    src_idx, tgt_idx = np.where(adj > 0)
    E = len(src_idx)
    weights = np.zeros(E, dtype=np.float64)
    for k in range(E):
        s = src_idx[k]
        if is_exc[s]:
            weights[k] = rng.uniform(*cfg.weight_exc_range)
        else:
            weights[k] = -rng.uniform(*cfg.weight_inh_range)
    # Delays in bins (at least 1 bin)
    delay_ms = _sample_in(rng, E, *cfg.delay_ms_range)
    delay_bins = np.maximum(1, np.round(delay_ms / cfg.dt_ms).astype(np.int64))
    max_delay = int(delay_bins.max()) if E > 0 else 1

    # External stimulus (per-neuron piecewise current).
    # Construct as a sum of random rectangular pulses to introduce stimulus variability.
    I_ext = np.zeros((T, N), dtype=np.float64)
    block_len = max(50, int(round(0.05 / dt)))  # 50 ms blocks
    n_blocks = (T + block_len - 1) // block_len
    for b in range(n_blocks):
        start = b * block_len
        end = min(T, (b + 1) * block_len)
        for n in range(N):
            if rng.random() < cfg.stim_on_prob:
                amp = rng.uniform(*cfg.stim_amplitude_range) * (1.0 if rng.random() < 0.7 else -1.0)
                I_ext[start:end, n] += amp

    # State arrays.
    V = np.full(N, cfg.E_L, dtype=np.float64)
    w = np.zeros(N, dtype=np.float64)
    refrac_remaining = np.zeros(N, dtype=np.int64)
    voltage = np.zeros((T, N), dtype=np.float64)
    spikes = np.zeros((T, N), dtype=np.int8)
    # Pending synaptic input queue: per-target neuron, indexed by future timestep.
    # Implement as a ring buffer of length max_delay+1.
    syn_buf = np.zeros((max_delay + 1, N), dtype=np.float64)

    tau_m_s = tau_m_ms / 1000.0
    tau_w_s = cfg.tau_w_ms / 1000.0
    refractory_steps = int(round(cfg.refractory_ms / cfg.dt_ms))

    for t in range(T):
        # Pull synaptic current from buffer head.
        I_syn = syn_buf[t % syn_buf.shape[0]].copy()
        syn_buf[t % syn_buf.shape[0]] = 0.0

        # Membrane update for neurons not in refractory period.
        active = refrac_remaining <= 0
        dV = np.zeros(N, dtype=np.float64)
        dV[active] = dt / tau_m_s[active] * (
            -(V[active] - cfg.E_L) + R_m[active] * (I_ext[t, active] + I_syn[active]) - w[active]
        )
        V[active] = V[active] + dV[active]

        # Adaptation update.
        if cfg.use_adaptation:
            w = w + dt / tau_w_s * (cfg.a_adapt * (V - cfg.E_L) - w)

        # Spike detection.
        fired = (V >= thresh) & active
        if fired.any():
            spikes[t, fired] = 1
            V[fired] = cfg.reset_voltage
            refrac_remaining[fired] = refractory_steps
            if cfg.use_adaptation:
                w[fired] = w[fired] + cfg.b_adapt
            # Inject delayed synaptic current into postsynaptic targets.
            fired_neurons = np.where(fired)[0]
            for src in fired_neurons:
                # All outgoing edges of `src`
                mask = src_idx == src
                if not mask.any():
                    continue
                tgts = tgt_idx[mask]
                wts = weights[mask]
                dlys = delay_bins[mask]
                for k_idx, tgt in enumerate(tgts):
                    arrival = (t + int(dlys[k_idx])) % syn_buf.shape[0]
                    syn_buf[arrival, tgt] += float(wts[k_idx])

        refrac_remaining = np.maximum(0, refrac_remaining - 1)

        # Record (add observation noise on voltage only).
        voltage[t] = V + rng.normal(0.0, cfg.noise_std, size=N)

    # Clamp voltage to a sensible range to prevent any pathological blowup.
    voltage = np.clip(voltage, -120.0, 60.0)

    params_df = pd.DataFrame(
        {
            "neuron_id": np.arange(N),
            "is_excitatory": is_exc,
            "tau_m_ms": tau_m_ms,
            "R_m": R_m,
            "threshold": thresh,
            "reset_voltage": np.full(N, cfg.reset_voltage),
            "E_L": np.full(N, cfg.E_L),
            "tau_w_ms": np.full(N, cfg.tau_w_ms if cfg.use_adaptation else float("nan")),
        }
    )

    edge_index = (
        np.stack([src_idx, tgt_idx], axis=0).astype(np.int64)
        if E > 0
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_weight = weights.astype(np.float64)
    edge_delay = delay_bins.astype(np.int64)

    return {
        "voltage": voltage,
        "spikes": spikes,
        "current": I_ext,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_delay_bins": edge_delay,
        "params_df": params_df,
        "dt_s": dt,
        "T": T,
        "N": N,
    }


def write_synthetic(out_dir: str | Path, cfg: SyntheticConfig) -> dict[str, Any]:
    """Generate and write synthetic data to ``out_dir``.

    Returns a dict containing key file paths and the splits.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim = simulate(cfg)

    save_npy(out / "voltage.npy", sim["voltage"])
    save_npy(out / "spikes.npy", sim["spikes"])
    save_npy(out / "current.npy", sim["current"])
    save_npy(out / "edge_index.npy", sim["edge_index"])
    save_npy(out / "edge_weight.npy", sim["edge_weight"])
    save_npy(out / "edge_delay_bins.npy", sim["edge_delay_bins"])
    sim["params_df"].to_csv(out / "params.csv", index=False)

    # Time-block split per the requested fractions; we always split contiguous
    # time blocks (NOT random windows) to avoid leakage.
    T = sim["T"]
    n_train = int(round(cfg.train_fraction * T))
    n_val = int(round(cfg.val_fraction * T))
    train_end = n_train
    val_end = n_train + n_val
    splits = {
        "train": [(0, train_end)],
        "val": [(train_end, val_end)],
        "test": [(val_end, T)],
    }
    # Held-out neuron set for masked-neuron eval.
    rng = seeded_rng(cfg.seed + 1)
    n_holdout = max(1, int(round(cfg.held_out_neuron_fraction * cfg.n_neurons)))
    heldout_neurons = rng.choice(cfg.n_neurons, size=n_holdout, replace=False)
    seen_neurons = np.array(sorted(set(range(cfg.n_neurons)) - set(heldout_neurons.tolist())))

    splits_meta = {
        "split_mode": "time_block",
        "split_strength": "cell_and_protocol_generalization",
        "time_blocks": splits,
        "heldout_neurons": heldout_neurons.tolist(),
        "seen_neurons": seen_neurons.tolist(),
        "dt_s": sim["dt_s"],
        "T": T,
        "N": sim["N"],
        "window_size": cfg.window_size,
        "horizon": cfg.horizon,
        "bin_size_ms": cfg.bin_size_ms,
    }
    write_json(out / "splits.json", splits_meta)

    manifest = {
        "dataset_name": "synthetic_lif_circuit",
        "config": asdict(cfg),
        "files": {
            "voltage": "voltage.npy",
            "spikes": "spikes.npy",
            "current": "current.npy",
            "edge_index": "edge_index.npy",
            "edge_weight": "edge_weight.npy",
            "edge_delay_bins": "edge_delay_bins.npy",
            "params": "params.csv",
            "splits": "splits.json",
        },
        "shapes": {
            "voltage": list(sim["voltage"].shape),
            "spikes": list(sim["spikes"].shape),
            "current": list(sim["current"].shape),
            "edge_index": list(sim["edge_index"].shape),
        },
        "ground_truth_available": True,
        "adapter_status": "fully_functional",
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "out_dir": str(out),
        "manifest_path": str(out / "manifest.json"),
        "splits": splits_meta,
        "N": sim["N"],
        "T": T,
        "edges": int(sim["edge_index"].shape[1]),
    }
