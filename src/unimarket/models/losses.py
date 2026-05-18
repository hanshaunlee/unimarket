"""Loss functions for UniMarket models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..constants import EPS


def voltage_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def spike_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def poisson_nll(rate: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    rate = rate.clamp_min(EPS)
    return (rate - counts * torch.log(rate)).mean()


def negative_binomial_nll(
    rate: torch.Tensor, counts: torch.Tensor, dispersion: torch.Tensor
) -> torch.Tensor:
    """NB NLL with shape-rate parameterization. Pass positive dispersion."""
    r = dispersion.clamp_min(EPS)
    mu = rate.clamp_min(EPS)
    log_prob = (
        torch.lgamma(counts + r)
        - torch.lgamma(r)
        - torch.lgamma(counts + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + counts * (torch.log(mu) - torch.log(r + mu))
    )
    return -log_prob.mean()


def rollout_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize errors that grow with horizon (encourages stable multi-step)."""
    if pred.dim() < 2 or target.dim() < 2:
        return torch.tensor(0.0, device=pred.device)
    H = pred.shape[1]
    weights = torch.linspace(0.5, 1.0, steps=H, device=pred.device)
    diff = (pred - target) ** 2
    # If trailing dims exist, average over them.
    while diff.dim() > 2:
        diff = diff.mean(dim=-1)
    return (diff.mean(dim=0) * weights).mean()


def physics_residual_penalty(residual_scale: float, weight: float = 1.0) -> torch.Tensor:
    """Penalty on residual magnitude; encourages reliance on mechanism."""
    return weight * residual_scale**2 * torch.tensor(1.0)


def graph_sparsity_loss(edge_logits: torch.Tensor) -> torch.Tensor:
    """Encourage learned graph to be sparse."""
    return torch.sigmoid(edge_logits).mean()


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalize jagged trajectories along the time axis (dim=1)."""
    if pred.dim() < 2 or pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    d = pred[:, 1:] - pred[:, :-1]
    return (d**2).mean()


def derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match the time-derivative of trajectories."""
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return F.mse_loss(dp, dt)
