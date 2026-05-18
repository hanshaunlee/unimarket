"""Reusable model components."""

from __future__ import annotations

import torch
import torch.nn as nn


def softplus_positive(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Positive parameterization via softplus with numerical safety."""
    return torch.nn.functional.softplus(x, beta=beta) + 1e-6


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(prev, hidden_dim), activation()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMessagePassing(nn.Module):
    """Edge-gated message passing without PyTorch Geometric.

    Computes for each edge j -> i:
        m_e = MLP([h_src, h_tgt, edge_attr]) * gate_e
    and aggregates into target nodes via index_add. Operates on batched
    inputs of shape [B, N, H].
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int = 1):
        super().__init__()
        self.msg = MLP(
            in_dim=2 * hidden_dim + edge_attr_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=2,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.update = MLP(
            in_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=2,
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """h: [B, N, H]; edge_index: [2, E]; edge_attr: [E, A] or None.

        Returns updated node states [B, N, H].
        """
        B, _N, H = h.shape
        src = edge_index[0]  # [E]
        tgt = edge_index[1]  # [E]
        E = src.shape[0]
        if edge_attr is None:
            edge_attr = torch.zeros(E, 1, device=h.device, dtype=h.dtype)
        # Gather source/target per edge.
        h_src = h[:, src, :]  # [B, E, H]
        h_tgt = h[:, tgt, :]  # [B, E, H]
        ea = edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, A]
        cat = torch.cat([h_src, h_tgt, ea], dim=-1)  # [B, E, 2H+A]
        m = self.msg(cat)  # [B, E, H]
        g = self.gate(cat)  # [B, E, 1]
        msgs = m * g
        # Aggregate via index_add into target nodes.
        agg = torch.zeros_like(h)
        # index_add over neuron dimension; do per-batch in a vectorized way.
        # Flatten batch dim for index_add to avoid Python loop:
        idx = tgt.view(1, E, 1).expand(B, E, H)
        agg = agg.scatter_add(dim=1, index=idx, src=msgs)
        out = self.update(torch.cat([h, agg], dim=-1))
        return h + out


class CausalTemporalEncoder(nn.Module):
    """Causal temporal encoder using GRU (default) over [B, T, N, F]."""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.gru(x)
        return out  # [B, T, H]


def sinusoid_time_embedding(T: int, dim: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal time embedding."""
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
    angle_rate = 1.0 / (10000.0 ** (2 * i / dim))
    angles = pos * angle_rate
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        pad = torch.zeros(T, dim - emb.shape[-1], device=device)
        emb = torch.cat([emb, pad], dim=-1)
    return emb[:, :dim]
