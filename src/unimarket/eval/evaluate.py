"""End-to-end evaluation driver.

Given a run directory:
1. Loads the model and best checkpoint.
2. Runs predictions on train/val/test loaders.
3. Computes split-wise metrics.
4. Optionally runs graph recovery (synthetic only).
5. Optionally runs identifiability / parameter recovery.
6. Optionally runs interpretability probes.
7. Runs the claim gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..constants import NEGATIVE_RESULTS_STATEMENT
from ..data.datamodules import (
    build_synthetic_population_loaders,
    build_synthetic_voltage_loaders,
    load_synthetic_bundle,
)
from ..train.checkpointing import load_checkpoint
from ..train.train import _build_model
from ..utils.io import read_json, write_csv, write_json
from ..utils.metrics import (
    binary_auprc,
    binary_auroc,
    f1_at_threshold,
    mae,
    mse,
    pearson_corr,
    poisson_nll,
    rmse,
)
from ..utils.plotting import plot_adjacency, plot_rollout_error, plot_voltage_pred_vs_true
from .claim_gating import compute_claim_gate
from .graph_eval import evaluate_graph_recovery
from .interpretability import ProbeSpec, run_interpretability_probes, write_interpretability_report
from .leakage import run_leakage_checks
from .rollout import collect_predictions, rollout_errors_per_horizon
from .statistics import bootstrap_ci
from .synthetic_recovery import run_synthetic_recovery


def _voltage_metrics(arr: dict[str, np.ndarray]) -> dict[str, float]:
    out = {
        "mse": mse(arr["pred"], arr["target"]),
        "rmse": rmse(arr["pred"], arr["target"]),
        "mae": mae(arr["pred"], arr["target"]),
        "corr": pearson_corr(arr["pred"].ravel(), arr["target"].ravel()),
    }
    if "pred_spike_logits" in arr and "target_spikes" in arr:
        sig = 1.0 / (1.0 + np.exp(-arr["pred_spike_logits"]))
        out["spike_auroc"] = binary_auroc(sig.ravel(), arr["target_spikes"].ravel())
        out["spike_auprc"] = binary_auprc(sig.ravel(), arr["target_spikes"].ravel())
        out["spike_f1@0.5"] = f1_at_threshold(sig.ravel(), arr["target_spikes"].ravel(), 0.5)
    rollout = rollout_errors_per_horizon(arr["pred"], arr["target"])
    out["rollout_mse_first"] = rollout[0] if rollout else float("nan")
    out["rollout_mse_last"] = rollout[-1] if rollout else float("nan")
    return out


def _population_metrics(arr: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {
        "poisson_nll": poisson_nll(arr["pred"], arr["target"]),
        "mse": mse(arr["pred"], arr["target"]),
        "corr": pearson_corr(arr["pred"].ravel(), arr["target"].ravel()),
    }
    rollout = rollout_errors_per_horizon(arr["pred"], arr["target"])
    if rollout:
        out["rollout_mse_first"] = rollout[0]
        out["rollout_mse_last"] = rollout[-1]
    if "mask" in arr:
        # Compute masked-neuron MSE: target neurons where mask==0.
        mask = arr["mask"]  # [B, N]
        m = (mask == 0)
        if m.any():
            # broadcast across horizon dim
            pred = arr["pred"]  # [B, H, N]
            tgt = arr["target"]  # [B, H, N]
            m3 = np.broadcast_to(m[:, None, :], pred.shape)
            out["masked_neuron_mse"] = float(((pred[m3] - tgt[m3]) ** 2).mean())
    return out


def _build_loaders_for_eval(cfg: dict, bundle) -> tuple[dict, str]:
    name = cfg["model"]["name"]
    train_cfg = cfg.get("train", {})
    if name in {"lif_residual", "adaptive_lif", "neural_ode", "persistence", "ar", "ridge"}:
        horizon = int(cfg.get("horizon", bundle.horizon))
        window = int(cfg.get("window_size", bundle.window_size))
        loaders = build_synthetic_voltage_loaders(
            bundle, batch_size=int(train_cfg.get("batch_size", 32)),
            horizon=horizon, window_size=window,
        )
        return loaders, "voltage"
    window_bins = int(cfg.get("window_bins", 20))
    horizon_bins = int(cfg.get("horizon_bins", 5))
    loaders = build_synthetic_population_loaders(
        bundle,
        batch_size=int(train_cfg.get("batch_size", 16)),
        window_bins=window_bins,
        horizon_bins=horizon_bins,
        mask_neurons=bundle.heldout_neurons,
    )
    return loaders, "population"


def evaluate_run(
    run_dir: str | Path,
    splits: list[str] | None = None,
    do_graph_recovery: bool = True,
    do_identifiability: bool = True,
    do_interpretability: bool = True,
) -> dict[str, Any]:
    run = Path(run_dir)
    cfg = read_json(run / "config.json")
    if splits is None:
        splits = ["train", "val", "test"]
    bundle = load_synthetic_bundle(Path(cfg["data"]["processed_dir"]))
    loaders, task = _build_loaders_for_eval(cfg, bundle)

    # Build model and load best.
    if task == "voltage":
        horizon = int(cfg.get("horizon", bundle.horizon))
        window = int(cfg.get("window_size", bundle.window_size))
        data_meta = {
            "n_neurons": bundle.voltage.shape[1],
            "history_len": window,
            "horizon": horizon,
            "dt_s": bundle.dt_s,
        }
    else:
        window_bins = int(cfg.get("window_bins", 20))
        horizon_bins = int(cfg.get("horizon_bins", 5))
        data_meta = {
            "n_neurons": bundle.voltage.shape[1],
            "window_bins": window_bins,
            "horizon_bins": horizon_bins,
            "edge_index": bundle.edge_index,
            "edge_weight": bundle.edge_weight,
            "dt_s": bundle.dt_s,
        }
    model, _ = _build_model(cfg["model"], data_meta)
    ckpt = run / "best.pt"
    if not ckpt.exists():
        ckpt = run / "last.pt"
    load_checkpoint(ckpt, model)

    # Predictions per split.
    metrics_per_split: dict[str, dict[str, float]] = {}
    rows_for_csv: list[dict[str, Any]] = []
    for split in splits:
        arr = collect_predictions(model, loaders[split])
        if "target" not in arr:
            metrics_per_split[split] = {}
            continue
        if task == "voltage":
            m = _voltage_metrics(arr)
        else:
            m = _population_metrics(arr)
        metrics_per_split[split] = m
        rows_for_csv.append({"split": split, **m})
        # Save a small qualitative plot for the test split.
        if split == "test":
            try:
                if task == "voltage":
                    plot_voltage_pred_vs_true(
                        arr["target"][0],
                        arr["pred"][0],
                        run / "figures" / "voltage_pred_vs_true.png",
                    )
                rollout = rollout_errors_per_horizon(arr["pred"], arr["target"])
                if rollout:
                    plot_rollout_error(rollout, run / "figures" / "rollout_error.png")
            except Exception:  # pragma: no cover - plotting failures shouldn't kill eval
                pass
    write_csv(run / "predictive_metrics_by_split.csv", rows_for_csv)
    write_json(run / "predictive_metrics_summary.json", metrics_per_split)

    # Graph recovery (synthetic only, when learned graph).
    graph_info: dict[str, Any] = {}
    if do_graph_recovery and cfg["model"]["name"] == "graph_latent_dynamics":
        if cfg["model"].get("graph_mode") == "learned":
            adj = model.learned_adjacency().cpu().numpy()
            graph_info = evaluate_graph_recovery(
                adj, bundle.edge_index, bundle.edge_weight.astype(float)
            )
            write_json(run / "graph_recovery.json", graph_info)
            pd.DataFrame([graph_info]).to_csv(run / "graph_recovery.csv", index=False)
            try:
                from .graph_eval import adjacency_from_edge_index

                true_adj = adjacency_from_edge_index(bundle.edge_index, adj.shape[0])
                plot_adjacency(true_adj, adj, run / "figures" / "adjacency_true_vs_learned.png")
            except Exception:
                pass

    # Identifiability / synthetic parameter recovery.
    ident_info: dict[str, Any] = {}
    if do_identifiability and cfg["model"]["name"] in {"lif_residual", "adaptive_lif"}:
        ident_info = run_synthetic_recovery(run)

    # Interpretability probes (only meaningful on per-neuron parameters).
    interp_info: dict[str, Any] = {}
    if do_interpretability and cfg["model"]["name"] in {"lif_residual", "adaptive_lif"}:
        try:
            extracted = model.extract_constrained_params(n_neurons=bundle.voltage.shape[1])
            latents_arr = np.stack(
                [v.detach().cpu().numpy() for v in extracted.values()], axis=1
            )
            params_df = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "params.csv")
            specs = [
                ProbeSpec(
                    name="tau_m_regression",
                    latents=latents_arr,
                    labels=(params_df["tau_m_ms"].to_numpy() * 1e-3),
                    task="regression",
                ),
                ProbeSpec(
                    name="excitatory_classification",
                    latents=latents_arr,
                    labels=params_df["is_excitatory"].astype(int).to_numpy(),
                    task="classification",
                ),
            ]
            rows = run_interpretability_probes(specs)
            interp_info = write_interpretability_report(run, rows)
        except Exception as e:
            interp_info = {"error": repr(e)}
            write_json(run / "interpretability_report.json", interp_info)

    # Leakage report.
    leak = run_leakage_checks(run)

    # Final claim gate.
    gate = compute_claim_gate(run)

    summary = {
        "run_dir": str(run),
        "metrics_per_split": metrics_per_split,
        "graph_recovery": graph_info,
        "identifiability": ident_info,
        "interpretability": interp_info,
        "leakage": leak,
        "claim_gate": gate,
        "negative_results_statement": NEGATIVE_RESULTS_STATEMENT,
    }
    write_json(run / "evaluation_summary.json", summary)
    return summary
