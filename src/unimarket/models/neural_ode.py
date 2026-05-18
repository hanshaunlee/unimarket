"""Simple discrete-time neural-ODE-style cell used as an unconstrained baseline.

Implements V_{t+1} = V_t + dt * f_theta(V_t, I_t) where f_theta is an MLP with
no biophysical constraints. Capacity-matchable against LIFResidualModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import MLP


class NeuralODECell(nn.Module):
    def __init__(
        self,
        history_len: int,
        horizon: int,
        hidden_dim: int = 64,
        dt_s: float = 1e-3,
    ):
        super().__init__()
        self.history_len = history_len
        self.horizon = horizon
        self.dt_s = dt_s
        self.ctx = min(8, history_len)
        self.f = MLP(in_dim=self.ctx * 2 + 1, hidden_dim=hidden_dim, out_dim=1, n_layers=3)

    def forward(self, batch: dict) -> dict:
        hist_v = batch["history_voltage"]
        hist_i = batch["history_current"]
        H = self.horizon
        ctx = self.ctx
        ctx_v = hist_v[:, -ctx:].clone()
        ctx_i = hist_i[:, -ctx:].clone()
        V = hist_v[:, -1].clone()
        I_t = hist_i[:, -1].clone()
        preds = []
        for _ in range(H):
            inp = torch.cat([ctx_v, ctx_i, I_t.unsqueeze(-1)], dim=-1)
            dV = self.f(inp).squeeze(-1) * self.dt_s * 1000.0  # MLP outputs rate per ms
            V_next = V + dV
            preds.append(V_next)
            ctx_v = torch.cat([ctx_v[:, 1:], V_next.unsqueeze(-1)], dim=-1)
            ctx_i = torch.cat([ctx_i[:, 1:], I_t.unsqueeze(-1)], dim=-1)
            V = V_next
        out = {"pred": torch.stack(preds, dim=1), "aux": {"unconstrained": True}}
        if "target_voltage" in batch:
            out["target"] = batch["target_voltage"]
        return out
