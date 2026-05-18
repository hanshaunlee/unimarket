"""High-level training entrypoint used by scripts/train.py and the CLI.

Reads an experiment YAML, builds the dataset/loaders, instantiates the model,
runs training, and writes the run directory.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..constants import NEGATIVE_RESULTS_STATEMENT
from ..data.datamodules import (
    build_synthetic_population_loaders,
    build_synthetic_voltage_loaders,
    load_synthetic_bundle,
)
from ..data.synthetic import SyntheticConfig, write_synthetic
from ..models.adaptive_lif import AdaptiveLIFResidualModel
from ..models.baselines import (
    ARVoltage,
    GLMPoissonPopulation,
    MeanRatePopulation,
    PersistenceVoltage,
    RidgeVoltage,
)
from ..models.graph_dynamics import GraphLatentDynamicsModel
from ..models.lif import LIFResidualModel
from ..models.neural_ode import NeuralODECell
from ..models.spike_transformer import SpikeTransformerBaseline
from ..models.synapse_model import SynapseResponseModel
from ..utils.config import load_yaml
from ..utils.io import write_json
from ..utils.paths import make_run_dir
from ..utils.seed import set_seed
from ..utils.validation import (
    assert_checkpoint_not_test,
    assert_no_forbidden_id_embeddings,
    assert_no_forbidden_stimulus_features,
    assert_window_random_split_allowed,
)
from .loops import population_loss_fn, run_training, voltage_loss_fn  # noqa: F401


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _ensure_synthetic_dataset(data_cfg: dict[str, Any]) -> Path:
    out_dir = Path(data_cfg["processed_dir"])
    needs_regen = data_cfg.get("regenerate", False)
    if not (out_dir / "manifest.json").exists() or needs_regen:
        # Build SyntheticConfig honoring overrides.
        sc_kwargs = dict(data_cfg.get("synthetic", {}))
        cfg = SyntheticConfig(**sc_kwargs)
        write_synthetic(out_dir, cfg)
    return out_dir


def _build_model(
    model_cfg: dict[str, Any], data_meta: dict[str, Any]
) -> tuple[torch.nn.Module, str]:
    name = model_cfg["name"]
    if name == "lif_residual":
        m = LIFResidualModel(
            history_len=data_meta["history_len"],
            horizon=data_meta["horizon"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            residual_scale=model_cfg.get("residual_scale", 0.2),
            dt_s=data_meta["dt_s"],
            learn_per_neuron_params=model_cfg.get("learn_per_neuron_params", True),
            n_neurons=data_meta["n_neurons"],
        )
        return m, "voltage"
    if name == "adaptive_lif":
        m = AdaptiveLIFResidualModel(
            history_len=data_meta["history_len"],
            horizon=data_meta["horizon"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            residual_scale=model_cfg.get("residual_scale", 0.2),
            dt_s=data_meta["dt_s"],
            learn_per_neuron_params=model_cfg.get("learn_per_neuron_params", True),
            n_neurons=data_meta["n_neurons"],
        )
        return m, "voltage"
    if name == "neural_ode":
        m = NeuralODECell(
            history_len=data_meta["history_len"],
            horizon=data_meta["horizon"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            dt_s=data_meta["dt_s"],
        )
        return m, "voltage"
    if name == "persistence":
        return PersistenceVoltage(horizon=data_meta["horizon"]), "voltage"
    if name == "ar":
        return ARVoltage(p=model_cfg.get("p", 5), horizon=data_meta["horizon"]), "voltage"
    if name == "ridge":
        return RidgeVoltage(
            history_len=data_meta["history_len"], horizon=data_meta["horizon"]
        ), "voltage"
    if name == "graph_latent_dynamics":
        # Edge index can come from synthetic ground truth (only allowed in synthetic experiments).
        edge_index = None
        edge_attr = None
        gm = model_cfg.get("graph_mode", "none")
        if gm == "given":
            edge_index = torch.from_numpy(data_meta["edge_index"]).long()
            edge_attr = torch.from_numpy(data_meta["edge_weight"]).float().unsqueeze(-1)
        elif gm == "random":
            N = data_meta["n_neurons"]
            rng = np.random.default_rng(model_cfg.get("graph_seed", 0))
            E = max(1, int(0.08 * N * N))
            src = rng.integers(0, N, size=E)
            tgt = rng.integers(0, N, size=E)
            mask = src != tgt
            edge_index = torch.from_numpy(np.stack([src[mask], tgt[mask]], axis=0)).long()
            edge_attr = torch.ones(edge_index.shape[1], 1)
        elif gm == "shuffled":
            true_idx = data_meta["edge_index"]
            rng = np.random.default_rng(model_cfg.get("graph_seed", 0))
            src = rng.permutation(true_idx[0])
            tgt = rng.permutation(true_idx[1])
            edge_index = torch.from_numpy(np.stack([src, tgt], axis=0)).long()
            edge_attr = torch.ones(edge_index.shape[1], 1)
        m = GraphLatentDynamicsModel(
            n_neurons=data_meta["n_neurons"],
            history_len=data_meta["window_bins"],
            horizon=data_meta["horizon_bins"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            graph_mode=gm,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        return m, "population"
    if name == "spike_transformer":
        m = SpikeTransformerBaseline(
            n_neurons=data_meta["n_neurons"],
            history_len=data_meta["window_bins"],
            horizon=data_meta["horizon_bins"],
            hidden_dim=model_cfg.get("hidden_dim", 64),
            n_layers=model_cfg.get("n_layers", 2),
        )
        return m, "population"
    if name == "mean_rate":
        return (
            MeanRatePopulation(n_neurons=data_meta["n_neurons"], horizon=data_meta["horizon_bins"]),
            "population",
        )
    if name == "glm_poisson":
        return (
            GLMPoissonPopulation(
                n_neurons=data_meta["n_neurons"],
                history_len=data_meta["window_bins"],
                horizon=data_meta["horizon_bins"],
            ),
            "population",
        )
    if name == "synapse_response":
        m = SynapseResponseModel(
            max_delay_steps=model_cfg.get("max_delay_steps", 20),
            hidden_dim=model_cfg.get("hidden_dim", 32),
            constrain_amplitude_sign=model_cfg.get("constrain_amplitude_sign", False),
            amplitude_sign=model_cfg.get("amplitude_sign", 1.0),
            residual_scale=model_cfg.get("residual_scale", 0.1),
        )
        return m, "synapse"
    raise ValueError(f"Unknown model name: {name}")


def train_from_config(config_path: str | Path) -> Path:
    """Run training end-to-end from an experiment YAML and return the run dir."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))

    # Resolve dataset.
    data_cfg = cfg["data"]
    dataset_kind = data_cfg.get("kind", "synthetic")
    if dataset_kind != "synthetic":
        raise NotImplementedError(
            f"Training entrypoint currently supports kind='synthetic'; got '{dataset_kind}'. "
            "Use the adapter scripts to prepare real datasets first."
        )

    processed_dir = _ensure_synthetic_dataset(data_cfg)
    bundle = load_synthetic_bundle(processed_dir)

    # Configure leakage/scientific safeguards from config.
    split_mode = data_cfg.get("split_mode", "time_block")
    allow_leaky = bool(data_cfg.get("allow_leaky_split", False))
    assert_window_random_split_allowed(split_mode, allow_leaky)
    feature_set = list(data_cfg.get("model_features", ["history_voltage", "history_current"]))
    assert_no_forbidden_stimulus_features(
        feature_set,
        allow_stimulus_identity=bool(cfg.get("allow_stimulus_identity", False)),
        allow_global_time_features=bool(cfg.get("allow_global_time_features", False)),
        allow_trial_index=bool(cfg.get("allow_trial_index", False)),
        allow_block_index=bool(cfg.get("allow_block_index", False)),
    )
    embedding_uses = cfg.get("model", {}).get("uses", {})
    assert_no_forbidden_id_embeddings(
        embedding_uses,
        allow_cell_id_embedding=bool(cfg.get("allow_cell_id_embedding", False)),
        allow_pair_id_embedding=bool(cfg.get("allow_pair_id_embedding", False)),
        allow_session_id_embedding=bool(cfg.get("allow_session_id_embedding", False)),
        split_mode=split_mode,
    )

    selection_split = cfg.get("selection_split", "val")
    assert_checkpoint_not_test(selection_split)

    # Build data meta for model construction.
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})

    # Run dir.
    run_dir = make_run_dir(cfg.get("run_name", model_cfg["name"]))
    write_json(run_dir / "config.json", cfg)
    git_h = _git_hash()
    if git_h:
        (run_dir / "git_hash.txt").write_text(git_h)
    # Copy the source config for reproducibility.
    try:
        shutil.copy(config_path, run_dir / "config_source.yaml")
    except Exception:
        pass

    # Build loaders + model depending on task.
    if model_cfg["name"] in {
        "lif_residual",
        "adaptive_lif",
        "neural_ode",
        "persistence",
        "ar",
        "ridge",
    }:
        horizon = int(cfg.get("horizon", bundle.horizon))
        window = int(cfg.get("window_size", bundle.window_size))
        loaders = build_synthetic_voltage_loaders(
            bundle, batch_size=int(train_cfg.get("batch_size", 32)),
            horizon=horizon, window_size=window,
            out_norm_path=run_dir / "normalization_stats",
        )
        data_meta = {
            "n_neurons": bundle.voltage.shape[1],
            "history_len": window,
            "horizon": horizon,
            "dt_s": bundle.dt_s,
        }
        model, task_resolved = _build_model(model_cfg, data_meta)
        loss_fn = voltage_loss_fn
    elif model_cfg["name"] in {"graph_latent_dynamics", "spike_transformer", "mean_rate", "glm_poisson"}:
        window_bins = int(cfg.get("window_bins", 20))
        horizon_bins = int(cfg.get("horizon_bins", 5))
        loaders = build_synthetic_population_loaders(
            bundle,
            batch_size=int(train_cfg.get("batch_size", 16)),
            window_bins=window_bins,
            horizon_bins=horizon_bins,
        )
        data_meta = {
            "n_neurons": bundle.voltage.shape[1],
            "window_bins": window_bins,
            "horizon_bins": horizon_bins,
            "edge_index": bundle.edge_index,
            "edge_weight": bundle.edge_weight,
            "dt_s": bundle.dt_s,
        }
        model, task_resolved = _build_model(model_cfg, data_meta)
        loss_fn = population_loss_fn
    else:
        raise ValueError(f"Unknown model task for '{model_cfg['name']}'")

    summary = run_training(
        model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        loss_fn=loss_fn,
        run_dir=run_dir,
        epochs=int(train_cfg.get("epochs", 3)),
        lr=float(train_cfg.get("lr", 1e-3)),
        device=_device(train_cfg.get("device", "auto")),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
        selection_metric=cfg.get("selection_metric", "loss"),
        selection_split=selection_split,
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
    )
    summary["config_path"] = str(config_path)
    summary["task"] = task_resolved
    summary["dataset_kind"] = dataset_kind
    summary["processed_dir"] = str(processed_dir)
    summary["negative_results_statement"] = NEGATIVE_RESULTS_STATEMENT
    write_json(run_dir / "train_summary.json", summary)
    # Capacity report (per safeguard #9).
    write_json(
        run_dir / "capacity_report.json",
        {
            "n_params": summary["n_params"],
            "n_trainable_params": summary["n_trainable_params"],
            "wall_clock_s": summary["wall_clock_s"],
            "n_epochs": summary["n_epochs_run"],
            "n_steps": summary["n_steps"],
            "device": summary["device"],
            "model_name": model_cfg["name"],
        },
    )
    return run_dir
