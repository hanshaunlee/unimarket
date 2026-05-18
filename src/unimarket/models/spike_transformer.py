"""Spike transformer baseline (no graph, no physics).

Uses a small causal transformer over time, applied per-neuron in parallel as a
black-box sequence model. Used to test whether graph + physics actually helps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import sinusoid_time_embedding, softplus_positive


class CausalTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal mask.
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class SpikeTransformerBaseline(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        history_len: int,
        horizon: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.history_len = history_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.proj_in = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(hidden_dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, batch: dict) -> dict:
        hist = batch["history_counts"]  # [B, T, N]
        B, T, N = hist.shape
        device = hist.device
        # Treat each neuron independently as a sequence.
        x = hist.permute(0, 2, 1).reshape(B * N, T, 1)
        x = self.proj_in(x)
        pos = sinusoid_time_embedding(T, self.hidden_dim, device=device).unsqueeze(0)
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        last = x[:, -1, :]  # [B*N, H]
        out_log = self.head(last)  # [B*N, horizon]
        rate = softplus_positive(out_log).reshape(B, N, self.horizon).permute(0, 2, 1)
        out = {
            "pred": rate,
            "aux": {"graph_mode": "none"},
        }
        if "future_counts" in batch:
            out["target"] = batch["future_counts"]
        return out
