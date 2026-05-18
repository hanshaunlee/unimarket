"""Synthetic parameter recovery driver.

Loads a trained mechanistic model + synthetic ground truth and runs the
identifiability recovery analysis (correlation between learned constrained
parameters and true synthetic parameters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.datamodules import load_synthetic_bundle
from ..models.adaptive_lif import AdaptiveLIFResidualModel
from ..models.lif import LIFResidualModel
from ..train.checkpointing import load_checkpoint
from .identifiability import evaluate_parameter_recovery, write_identifiability_report


def _build_model_for_run(cfg: dict, n_neurons: int, dt_s: float, history_len: int, horizon: int) -> torch.nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    if name == "lif_residual":
        return LIFResidualModel(
            history_len=history_len,
            horizon=horizon,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            residual_scale=model_cfg.get("residual_scale", 0.2),
            dt_s=dt_s,
            learn_per_neuron_params=model_cfg.get("learn_per_neuron_params", True),
            n_neurons=n_neurons,
        )
    if name == "adaptive_lif":
        return AdaptiveLIFResidualModel(
            history_len=history_len,
            horizon=horizon,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            residual_scale=model_cfg.get("residual_scale", 0.2),
            dt_s=dt_s,
            learn_per_neuron_params=model_cfg.get("learn_per_neuron_params", True),
            n_neurons=n_neurons,
        )
    raise ValueError(f"Synthetic recovery currently supports lif_residual / adaptive_lif, not {name}")


def run_synthetic_recovery(run_dir: str | Path) -> dict[str, Any]:
    """Compute parameter recovery for a trained voltage model on synthetic data."""
    from ..utils.io import read_json

    run = Path(run_dir)
    cfg = read_json(run / "config.json")
    processed_dir = Path(cfg["data"]["processed_dir"])
    bundle = load_synthetic_bundle(processed_dir)

    # Mechanistic recovery is only meaningful for LIF-family models.
    model_name = cfg["model"]["name"]
    if model_name not in {"lif_residual", "adaptive_lif"}:
        # Write an empty/null report for non-mechanistic models.
        rows = [
            {"parameter": p, "correlation": float("nan"), "label": "not_tested", "n": 0,
             "note": f"Recovery skipped for model '{model_name}' (no constrained parameters)."}
            for p in ("tau_m_s", "R_m", "threshold")
        ]
        return write_identifiability_report(run, rows, extra={"model_name": model_name})

    history_len = int(cfg.get("window_size", bundle.window_size))
    horizon = int(cfg.get("horizon", bundle.horizon))
    model = _build_model_for_run(cfg, bundle.voltage.shape[1], bundle.dt_s, history_len, horizon)
    ckpt = run / "best.pt"
    if not ckpt.exists():
        ckpt = run / "last.pt"
    load_checkpoint(ckpt, model)
    model.eval()

    extracted = model.extract_constrained_params(n_neurons=bundle.voltage.shape[1])
    # Convert tensors to numpy.
    learned = {k: v.detach().cpu().numpy() for k, v in extracted.items()}

    # Build true parameter vectors from the synthetic params CSV.
    params_df = pd.read_csv(processed_dir / "params.csv")
    true = {
        "tau_m_s": (params_df["tau_m_ms"].to_numpy() * 1e-3),
        "R_m": params_df["R_m"].to_numpy(),
        "threshold": params_df["threshold"].to_numpy(),
    }
    if "tau_w_s" in learned:
        # Single tau_w used during simulation; replicate to length N for comparison.
        N = len(params_df)
        true["tau_w_s"] = np.full(N, float(params_df["tau_w_ms"].iloc[0]) * 1e-3)
    rows = evaluate_parameter_recovery(learned, true)
    return write_identifiability_report(run, rows, extra={"model_name": model_name})
