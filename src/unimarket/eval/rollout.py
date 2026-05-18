"""Multi-step rollout evaluation utilities."""

from __future__ import annotations

import numpy as np
import torch

from ..utils.metrics import mse


def rollout_errors_per_horizon(pred: np.ndarray, target: np.ndarray) -> list[float]:
    """Mean squared error broken down by horizon step.

    pred/target have shape [..., H, ...] where the second-to-last axis is the
    horizon. We accept [B, H] or [B, H, N]; aggregate over all but the H axis.
    """
    if pred.ndim < 2 or target.ndim < 2:
        return []
    H = pred.shape[1]
    errs = []
    for h in range(H):
        errs.append(mse(pred[:, h, ...], target[:, h, ...]))
    return errs


def collect_predictions(
    model: torch.nn.Module, loader, device: str | torch.device = "cpu"
) -> dict[str, np.ndarray]:
    """Run model over a loader and return concatenated np arrays."""
    device = torch.device(device)
    model.eval().to(device)
    preds, targets, aux_lists = [], [], []
    spike_logits, spike_targets = [], []
    masks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(batch)
            preds.append(out["pred"].detach().cpu().numpy())
            if "target" in out:
                targets.append(out["target"].detach().cpu().numpy())
            if "pred_spike_logits" in out:
                spike_logits.append(out["pred_spike_logits"].detach().cpu().numpy())
            if "target_spikes" in out:
                spike_targets.append(out["target_spikes"].detach().cpu().numpy())
            if "mask" in out:
                masks.append(out["mask"].detach().cpu().numpy())
            aux_lists.append(out.get("aux", {}))
    result = {"pred": np.concatenate(preds, axis=0) if preds else np.zeros(0)}
    if targets:
        result["target"] = np.concatenate(targets, axis=0)
    if spike_logits:
        result["pred_spike_logits"] = np.concatenate(spike_logits, axis=0)
    if spike_targets:
        result["target_spikes"] = np.concatenate(spike_targets, axis=0)
    if masks:
        result["mask"] = np.concatenate(masks, axis=0)
    return result
