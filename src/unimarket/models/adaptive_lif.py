"""Adaptive LIF residual model with a learnable adaptation current."""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import MLP, softplus_positive
from .lif import LIFResidualModel


class AdaptiveLIFResidualModel(LIFResidualModel):
    """LIF + adaptation current ``w`` with positive tau_w."""

    def __init__(
        self,
        history_len: int,
        horizon: int,
        hidden_dim: int = 64,
        residual_scale: float = 0.2,
        dt_s: float = 1e-3,
        reset_voltage: float = -65.0,
        E_L: float = -65.0,
        learn_per_neuron_params: bool = False,
        n_neurons: int = 1,
    ):
        super().__init__(
            history_len=history_len,
            horizon=horizon,
            hidden_dim=hidden_dim,
            residual_scale=residual_scale,
            dt_s=dt_s,
            reset_voltage=reset_voltage,
            E_L=E_L,
            learn_per_neuron_params=learn_per_neuron_params,
            n_neurons=n_neurons,
        )
        # Adaptation parameters.
        if learn_per_neuron_params:
            self.tau_w_raw = nn.Parameter(torch.zeros(n_neurons))
            self.a_adapt_raw = nn.Parameter(torch.zeros(n_neurons))
        else:
            self.tau_w_raw = nn.Parameter(torch.zeros(1))
            self.a_adapt_raw = nn.Parameter(torch.zeros(1))

    def tau_w(self, neuron_idx: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.tau_w_raw
        if self.learn_per_neuron_params and neuron_idx is not None:
            raw = raw[neuron_idx]
        return softplus_positive(raw + 4.0) * 1e-3  # seconds; ~80 ms initial

    def a_adapt(self, neuron_idx: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.a_adapt_raw
        if self.learn_per_neuron_params and neuron_idx is not None:
            raw = raw[neuron_idx]
        return softplus_positive(raw - 3.0)  # small positive

    def forward(self, batch: dict) -> dict:
        hist_v = batch["history_voltage"]
        hist_i = batch["history_current"]
        B = hist_v.shape[0]
        H = self.horizon
        neuron_idx = batch.get("neuron_idx")

        tau = self.tau_m(neuron_idx).view(-1) if neuron_idx is not None else self.tau_m()
        R = self.R_m(neuron_idx).view(-1) if neuron_idx is not None else self.R_m()
        thr = self.threshold(neuron_idx).view(-1) if neuron_idx is not None else self.threshold()
        tau_w = self.tau_w(neuron_idx).view(-1) if neuron_idx is not None else self.tau_w()
        a = self.a_adapt(neuron_idx).view(-1) if neuron_idx is not None else self.a_adapt()
        if tau.numel() == 1:
            tau = tau.expand(B)
        if R.numel() == 1:
            R = R.expand(B)
        if thr.numel() == 1:
            thr = thr.expand(B)
        if tau_w.numel() == 1:
            tau_w = tau_w.expand(B)
        if a.numel() == 1:
            a = a.expand(B)

        V = hist_v[:, -1].clone()
        w = torch.zeros_like(V)
        I_t = hist_i[:, -1].clone()
        ctx = self.ctx
        ctx_v = hist_v[:, -ctx:].clone()
        ctx_i = hist_i[:, -ctx:].clone()

        preds, spike_logits = [], []
        for _ in range(H):
            dV = self.dt_s / tau * (-(V - self.E_L_const) + R * I_t - w)
            dw = self.dt_s / tau_w * (a * (V - self.E_L_const) - w)
            ctx_feat = torch.cat([ctx_v, ctx_i], dim=-1)
            residual = self.residual_mlp(ctx_feat).squeeze(-1) * self.residual_scale
            V_next = V + dV + residual
            w = w + dw
            sl = self.spike_head(torch.cat([ctx_feat, V_next.unsqueeze(-1)], dim=-1)).squeeze(-1)
            preds.append(V_next)
            spike_logits.append(sl)
            ctx_v = torch.cat([ctx_v[:, 1:], V_next.unsqueeze(-1)], dim=-1)
            ctx_i = torch.cat([ctx_i[:, 1:], I_t.unsqueeze(-1)], dim=-1)
            V = V_next

        out = {
            "pred": torch.stack(preds, dim=1),
            "pred_spike_logits": torch.stack(spike_logits, dim=1),
            "aux": {
                "tau_m_s": tau.detach(),
                "R_m": R.detach(),
                "threshold": thr.detach(),
                "tau_w_s": tau_w.detach(),
                "a_adapt": a.detach(),
                "residual_scale": float(self.residual_scale),
            },
        }
        if "target_voltage" in batch:
            out["target"] = batch["target_voltage"]
        if "target_spikes" in batch:
            out["target_spikes"] = batch["target_spikes"]
        return out

    def extract_constrained_params(
        self, n_neurons: int | None = None
    ) -> dict[str, torch.Tensor]:
        base = super().extract_constrained_params(n_neurons=n_neurons)
        n = n_neurons if n_neurons is not None else (
            self.n_neurons if self.learn_per_neuron_params else 1
        )
        idx = torch.arange(n, device=self.tau_w_raw.device)
        base["tau_w_s"] = self.tau_w(idx).detach()
        base["a_adapt"] = self.a_adapt(idx).detach()
        return base
