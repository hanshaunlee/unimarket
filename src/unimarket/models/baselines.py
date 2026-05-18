"""Classical baselines: persistence, mean-rate, AR, GLM-Poisson, ridge."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .components import softplus_positive


class PersistenceVoltage(nn.Module):
    """Predict next-h voltage = last-observed voltage."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon
        # Dummy parameter so optimizers don't choke; not used.
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        v = batch["history_voltage"][:, -1:].clone()  # [B, 1]
        pred = v.expand(-1, self.horizon)
        out = {"pred": pred, "aux": {"baseline": "persistence_voltage"}}
        if "target_voltage" in batch:
            out["target"] = batch["target_voltage"]
        return out


class MeanRatePopulation(nn.Module):
    """Predict future counts = train-set mean rate per neuron, broadcast."""

    def __init__(self, n_neurons: int, horizon: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.horizon = horizon
        self.mean = nn.Parameter(torch.zeros(n_neurons), requires_grad=False)
        self._init = False

    def fit(self, history_counts: torch.Tensor) -> None:
        # history_counts: [B, T, N]
        self.mean.data = history_counts.mean(dim=(0, 1)).detach()
        self._init = True

    def forward(self, batch: dict) -> dict:
        hist = batch["history_counts"]
        B = hist.shape[0]
        if not self._init:
            self.fit(hist)
        rate = self.mean.view(1, 1, -1).expand(B, self.horizon, -1).contiguous()
        rate = softplus_positive(rate)  # ensure positive
        out = {"pred": rate, "aux": {"baseline": "mean_rate"}}
        if "future_counts" in batch:
            out["target"] = batch["future_counts"]
        return out


class ARVoltage(nn.Module):
    """Per-sample AR(p) model fit jointly across batch."""

    def __init__(self, p: int, horizon: int):
        super().__init__()
        self.p = p
        self.horizon = horizon
        self.coef = nn.Parameter(torch.ones(p) / p)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        v = batch["history_voltage"]
        B, T = v.shape
        p = self.p
        assert T >= p, f"AR({p}) requires history >= {p}"
        ctx = v[:, -p:].clone()
        preds = []
        for _ in range(self.horizon):
            nxt = (ctx * self.coef).sum(dim=-1) + self.bias
            preds.append(nxt)
            ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(-1)], dim=-1)
        out = {"pred": torch.stack(preds, dim=1), "aux": {"baseline": "AR", "p": p}}
        if "target_voltage" in batch:
            out["target"] = batch["target_voltage"]
        return out


class GLMPoissonPopulation(nn.Module):
    """A Poisson GLM over a history window, per-neuron."""

    def __init__(self, n_neurons: int, history_len: int, horizon: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.history_len = history_len
        self.horizon = horizon
        # Per-neuron linear weights over last history_len bins, summed across neurons.
        self.W = nn.Parameter(torch.zeros(n_neurons, history_len * n_neurons))
        self.b = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, batch: dict) -> dict:
        hist = batch["history_counts"]  # [B, T, N]
        B, T, N = hist.shape
        flat = hist.reshape(B, T * N)
        log_rate = flat @ self.W.t() + self.b  # [B, N]
        rate = softplus_positive(log_rate).unsqueeze(1).expand(-1, self.horizon, -1).contiguous()
        out = {"pred": rate, "aux": {"baseline": "GLM_poisson"}}
        if "future_counts" in batch:
            out["target"] = batch["future_counts"]
        return out


class RidgeVoltage(nn.Module):
    """Linear ridge baseline for voltage prediction (with explicit weight decay loss term)."""

    def __init__(self, history_len: int, horizon: int, weight_decay: float = 1e-3):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(horizon, 2 * history_len))
        self.b = nn.Parameter(torch.zeros(horizon))
        self.weight_decay = weight_decay

    def forward(self, batch: dict) -> dict:
        v = batch["history_voltage"]
        i = batch["history_current"]
        x = torch.cat([v, i], dim=-1)  # [B, 2T]
        pred = x @ self.W.t() + self.b  # [B, H]
        out = {"pred": pred, "aux": {"baseline": "ridge"}}
        if "target_voltage" in batch:
            out["target"] = batch["target_voltage"]
        return out
