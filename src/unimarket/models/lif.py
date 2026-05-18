"""Mechanistic LIF residual model for single-neuron voltage prediction.

V_{t+1} = V_t + dt/tau_m * (-(V_t - E_L) + R_m * I_t) + residual

Constraints:
- tau_m, R_m parameterized via softplus to ensure positivity
- threshold > reset enforced by parameterization

The residual is intentionally bounded and penalized so that the constrained
dynamical core remains identifiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import MLP, softplus_positive


class LIFResidualModel(nn.Module):
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
        super().__init__()
        self.history_len = history_len
        self.horizon = horizon
        self.dt_s = dt_s
        self.E_L_const = E_L
        self.reset_voltage = reset_voltage
        self.residual_scale = residual_scale
        self.learn_per_neuron_params = learn_per_neuron_params
        self.n_neurons = n_neurons

        if learn_per_neuron_params:
            # Per-neuron raw parameters (softplus'd to positive at use).
            self.tau_m_raw = nn.Parameter(torch.zeros(n_neurons))
            self.R_m_raw = nn.Parameter(torch.zeros(n_neurons))
            self.threshold_raw = nn.Parameter(torch.zeros(n_neurons))
        else:
            self.tau_m_raw = nn.Parameter(torch.zeros(1))
            self.R_m_raw = nn.Parameter(torch.zeros(1))
            self.threshold_raw = nn.Parameter(torch.zeros(1))

        # Residual MLP operates on a fixed-size context (last few V and I).
        ctx = min(8, history_len)
        self.ctx = ctx
        self.residual_mlp = MLP(in_dim=ctx * 2, hidden_dim=hidden_dim, out_dim=1, n_layers=2)
        self.spike_head = MLP(in_dim=ctx * 2 + 1, hidden_dim=hidden_dim, out_dim=1, n_layers=2)

    # ----- Parameter accessors -----

    def tau_m(self, neuron_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Return per-sample tau_m in seconds (after softplus)."""
        raw = self.tau_m_raw
        if self.learn_per_neuron_params and neuron_idx is not None:
            raw = raw[neuron_idx]
        # Center around ~20ms.
        return softplus_positive(raw + 3.0) * 1e-3  # seconds

    def R_m(self, neuron_idx: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.R_m_raw
        if self.learn_per_neuron_params and neuron_idx is not None:
            raw = raw[neuron_idx]
        return softplus_positive(raw + 0.0)  # positive scalar

    def threshold(self, neuron_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Threshold parameterized to be > reset by softplus offset."""
        raw = self.threshold_raw
        if self.learn_per_neuron_params and neuron_idx is not None:
            raw = raw[neuron_idx]
        return self.reset_voltage + softplus_positive(raw + 2.0)  # > reset

    # ----- Forward -----

    def forward(self, batch: dict) -> dict:
        """Predict next-h voltages given history.

        Expected batch keys:
            history_voltage: [B, T]
            history_current: [B, T]
            target_voltage:  [B, H]   (optional, for loss computation)
            target_spikes:   [B, H]   (optional)
            neuron_idx:      [B]      (optional, for per-neuron params)
        """
        hist_v = batch["history_voltage"]  # [B, T]
        hist_i = batch["history_current"]  # [B, T]
        B = hist_v.shape[0]
        H = self.horizon
        neuron_idx = batch.get("neuron_idx")  # [B] or None

        # Pull per-sample params.
        tau = self.tau_m(neuron_idx).view(-1) if neuron_idx is not None else self.tau_m()
        R = self.R_m(neuron_idx).view(-1) if neuron_idx is not None else self.R_m()
        thr = self.threshold(neuron_idx).view(-1) if neuron_idx is not None else self.threshold()
        # Broadcast scalars over batch if needed.
        if tau.numel() == 1:
            tau = tau.expand(B)
        if R.numel() == 1:
            R = R.expand(B)
        if thr.numel() == 1:
            thr = thr.expand(B)

        # Start state from last observed voltage.
        V = hist_v[:, -1].clone()
        preds = []
        spike_logits = []
        # For rollout, we extrapolate current using last-known current (constant approximation).
        I_t = hist_i[:, -1].clone()
        # Residual context: last ``ctx`` history values.
        ctx = self.ctx
        ctx_v = hist_v[:, -ctx:].clone()
        ctx_i = hist_i[:, -ctx:].clone()

        for _ in range(H):
            # Mechanistic update (data is in normalized z-units but the equation shape holds).
            dV = self.dt_s / tau * (-(V - self.E_L_const) + R * I_t)
            ctx_feat = torch.cat([ctx_v, ctx_i], dim=-1)  # [B, 2*ctx]
            residual = self.residual_mlp(ctx_feat).squeeze(-1) * self.residual_scale
            V_next = V + dV + residual
            # Spike head (probability of spike in next bin).
            spike_in = torch.cat([ctx_feat, V_next.unsqueeze(-1)], dim=-1)
            sl = self.spike_head(spike_in).squeeze(-1)
            preds.append(V_next)
            spike_logits.append(sl)
            # Roll context forward.
            ctx_v = torch.cat([ctx_v[:, 1:], V_next.unsqueeze(-1)], dim=-1)
            ctx_i = torch.cat([ctx_i[:, 1:], I_t.unsqueeze(-1)], dim=-1)
            V = V_next

        pred_voltage = torch.stack(preds, dim=1)  # [B, H]
        pred_spike_logits = torch.stack(spike_logits, dim=1)  # [B, H]

        out = {
            "pred": pred_voltage,
            "pred_spike_logits": pred_spike_logits,
            "aux": {
                "tau_m_s": tau.detach(),
                "R_m": R.detach(),
                "threshold": thr.detach(),
                "residual_scale": float(self.residual_scale),
            },
        }
        if "target_voltage" in batch:
            out["target"] = batch["target_voltage"]
        if "target_spikes" in batch:
            out["target_spikes"] = batch["target_spikes"]
        return out

    # Convenience for identifiability / interpretability post hoc:
    def extract_constrained_params(
        self, n_neurons: int | None = None
    ) -> dict[str, torch.Tensor]:
        n = n_neurons if n_neurons is not None else (self.n_neurons if self.learn_per_neuron_params else 1)
        idx = torch.arange(n, device=self.tau_m_raw.device)
        return {
            "tau_m_s": self.tau_m(idx).detach(),
            "R_m": self.R_m(idx).detach(),
            "threshold": self.threshold(idx).detach(),
        }
