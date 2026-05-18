"""Generic training loop and per-task loss assembly."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from ..models.losses import (
    poisson_nll,
    rollout_consistency_loss,
    smoothness_loss,
    spike_bce,
    voltage_mse,
)
from ..utils.logging import RunLogger
from ..utils.validation import assert_checkpoint_not_test
from .checkpointing import save_checkpoint


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def voltage_loss_fn(out: dict, residual_weight: float = 0.01) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    if "target" in out:
        losses["voltage_mse"] = voltage_mse(out["pred"], out["target"])
        losses["rollout"] = rollout_consistency_loss(out["pred"], out["target"])
    if "target_spikes" in out and "pred_spike_logits" in out:
        losses["spike_bce"] = spike_bce(out["pred_spike_logits"], out["target_spikes"])
    losses["smoothness"] = smoothness_loss(out["pred"]) * 0.01
    if "aux" in out and "residual_scale" in out["aux"]:
        losses["physics_residual"] = torch.tensor(
            residual_weight * (out["aux"]["residual_scale"] ** 2),
            device=out["pred"].device,
        )
    total = sum(losses.values()) if losses else torch.tensor(0.0)
    losses["loss"] = total
    return losses


def population_loss_fn(out: dict, edge_logits: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    if "target" in out:
        losses["poisson_nll"] = poisson_nll(out["pred"], out["target"])
        losses["rollout"] = rollout_consistency_loss(out["pred"], out["target"]) * 0.1
    if edge_logits is not None:
        losses["graph_sparsity"] = torch.sigmoid(edge_logits).mean() * 1e-3
    total = sum(losses.values()) if losses else torch.tensor(0.0)
    losses["loss"] = total
    return losses


def synapse_loss_fn(out: dict) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    if "target" in out:
        losses["response_mse"] = F.mse_loss(out["pred"], out["target"])
        if out["pred"].shape[-1] > 1:
            dp = out["pred"][..., 1:] - out["pred"][..., :-1]
            dt = out["target"][..., 1:] - out["target"][..., :-1]
            losses["derivative"] = F.mse_loss(dp, dt) * 0.1
    total = sum(losses.values()) if losses else torch.tensor(0.0)
    losses["loss"] = total
    return losses


def run_training(
    model: torch.nn.Module,
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    loss_fn: Callable[[dict], dict[str, torch.Tensor]],
    *,
    run_dir: str | Path,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    grad_clip: float = 1.0,
    early_stopping_patience: int = 5,
    selection_metric: str = "loss",
    selection_split: str = "val",
    log_every_n_steps: int = 50,
) -> dict[str, Any]:
    """Train ``model`` and return a dict with best checkpoint info.

    Selection of the best checkpoint uses ``selection_split`` (must be 'val').
    Raises if ``selection_split == 'test'``.
    """
    assert_checkpoint_not_test(selection_split)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir)
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    bad_epochs = 0
    best_ckpt_path = run_dir / "best.pt"
    last_ckpt_path = run_dir / "last.pt"
    history: list[dict[str, float]] = []
    t0 = time.time()
    step = 0

    n_params = int(sum(p.numel() for p in model.parameters()))
    n_train_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(
        f"Starting training: device={device} epochs={epochs} lr={lr} "
        f"params={n_params} trainable={n_train_params}"
    )

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            step += 1
            batch = _to_device(batch, device)
            out = model(batch)
            losses = loss_fn(out)
            loss = losses["loss"]
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if step % log_every_n_steps == 0:
                logger.log_metrics(step, {k: float(v.detach()) for k, v in losses.items()})

        # Validation pass.
        model.eval()
        val_losses: dict[str, list[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                out = model(batch)
                losses = loss_fn(out)
                for k, v in losses.items():
                    val_losses.setdefault(k, []).append(float(v.detach()))
        val_means = {f"val_{k}": (sum(v) / max(1, len(v))) for k, v in val_losses.items()}
        logger.log_metrics(step, val_means)
        cur = val_means.get(f"val_{selection_metric}", val_means.get("val_loss", float("inf")))
        history.append({"epoch": epoch, "step": step, **val_means})
        logger.info(f"Epoch {epoch}: val_{selection_metric}={cur:.6f}")
        save_checkpoint(last_ckpt_path, model, optimizer, step, extra={"epoch": epoch})
        if cur < best_val:
            best_val = cur
            bad_epochs = 0
            save_checkpoint(best_ckpt_path, model, optimizer, step, extra={"epoch": epoch})
            logger.info(f"  new best: {best_val:.6f}")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    wall = time.time() - t0
    summary = {
        "best_val_metric": best_val,
        "best_ckpt_path": str(best_ckpt_path),
        "last_ckpt_path": str(last_ckpt_path),
        "wall_clock_s": wall,
        "selection_metric": selection_metric,
        "selection_split": selection_split,
        "n_params": n_params,
        "n_trainable_params": n_train_params,
        "n_epochs_run": min(epochs, len(history)),
        "n_steps": step,
        "device": str(device),
        "history": history,
    }
    return summary
