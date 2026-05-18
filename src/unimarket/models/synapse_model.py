"""Synapse response model for paired recordings.

Inputs:
  presynaptic event/stimulation trace
Outputs:
  postsynaptic response trace
Constrained interpretable parameters:
  delay (>=0), amplitude, rise_tau (>0), decay_tau (>0), STP proxy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import MLP, softplus_positive


class SynapseResponseModel(nn.Module):
    def __init__(
        self,
        max_delay_steps: int = 20,
        hidden_dim: int = 32,
        constrain_amplitude_sign: bool = False,
        amplitude_sign: float = 1.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.max_delay_steps = max_delay_steps
        self.constrain_amplitude_sign = constrain_amplitude_sign
        self.amplitude_sign = amplitude_sign
        self.residual_scale = residual_scale
        self.delay_raw = nn.Parameter(torch.zeros(1))
        self.amp_raw = nn.Parameter(torch.zeros(1))
        self.rise_raw = nn.Parameter(torch.zeros(1))
        self.decay_raw = nn.Parameter(torch.zeros(1))
        self.stp_raw = nn.Parameter(torch.zeros(1))  # short-term plasticity proxy
        self.residual = MLP(in_dim=8, hidden_dim=hidden_dim, out_dim=1, n_layers=2)

    def delay(self) -> torch.Tensor:
        return softplus_positive(self.delay_raw + 0.5)  # in steps; > 0

    def rise_tau(self) -> torch.Tensor:
        return softplus_positive(self.rise_raw + 1.0)

    def decay_tau(self) -> torch.Tensor:
        return softplus_positive(self.decay_raw + 2.0)

    def amplitude(self) -> torch.Tensor:
        a = self.amp_raw
        if self.constrain_amplitude_sign:
            return self.amplitude_sign * softplus_positive(a)
        return a

    def stp(self) -> torch.Tensor:
        # Bound STP factor between e.g. -0.5 (depression) and +0.5 (facilitation).
        return 0.5 * torch.tanh(self.stp_raw)

    def _double_exp_kernel(self, K: int, device: torch.device) -> torch.Tensor:
        """Causal alpha-like kernel using rise/decay taus, length K."""
        t = torch.arange(K, device=device, dtype=torch.float32)
        rise = self.rise_tau()
        decay = self.decay_tau()
        kernel = torch.exp(-t / decay) - torch.exp(-t / rise)
        # Normalize to peak=1.
        peak = kernel.max().clamp_min(1e-6)
        return kernel / peak

    def forward(self, batch: dict) -> dict:
        """batch keys:
            pre_event: [B, T] event/stim trace
            target_response: [B, T] (optional, for loss)
            pre_history_count: [B, T] (optional, running event count for STP proxy)
        """
        pre = batch["pre_event"]  # [B, T]
        B, T = pre.shape
        device = pre.device
        K = max(8, T // 2)
        kernel = self._double_exp_kernel(K, device=device)  # [K]

        # Apply integer delay via shift; use floor of self.delay() (clamped).
        delay_steps = int(torch.clamp(self.delay(), 0, self.max_delay_steps - 1).item())
        # Shift pre by delay.
        if delay_steps > 0:
            shifted = F.pad(pre, (delay_steps, 0))[:, :T]
        else:
            shifted = pre
        # STP proxy: scale amplitude by (1 + stp * cumulative event count normalized).
        stp_factor = self.stp()
        if "pre_history_count" in batch:
            count_norm = batch["pre_history_count"] / (
                batch["pre_history_count"].max().clamp_min(1.0)
            )
        else:
            count_norm = torch.cumsum(shifted, dim=-1) / T
        amp = self.amplitude() * (1.0 + stp_factor * count_norm)

        # 1-D convolution of (shifted * amp) with the kernel.
        signal = (shifted * amp).unsqueeze(1)  # [B,1,T]
        kk = kernel.flip(0).view(1, 1, -1)
        # Causal padding: only past contributes.
        signal_padded = F.pad(signal, (kernel.shape[0] - 1, 0))
        conv = F.conv1d(signal_padded, kk).squeeze(1)  # [B, T]

        # Residual: small correction from local features of pre & past response.
        ctx = pre[:, -8:]
        if ctx.shape[1] < 8:
            ctx = F.pad(ctx, (8 - ctx.shape[1], 0))
        residual = self.residual(ctx).squeeze(-1).unsqueeze(-1) * self.residual_scale
        pred = conv + residual

        out = {
            "pred": pred,
            "aux": {
                "delay_steps": float(delay_steps),
                "amplitude": float(self.amplitude().detach().cpu().item()),
                "rise_tau": float(self.rise_tau().detach().cpu().item()),
                "decay_tau": float(self.decay_tau().detach().cpu().item()),
                "stp": float(self.stp().detach().cpu().item()),
            },
        }
        if "target_response" in batch:
            out["target"] = batch["target_response"]
        return out
