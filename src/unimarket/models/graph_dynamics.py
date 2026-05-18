"""Graph-temporal circuit model for population spike-count forecasting.

Inputs:
    history_counts: [B, T, N] non-negative spike counts (float)
Outputs:
    pred_rate: [B, H, N] non-negative rates (after softplus)
    optional learned edge weights when graph_mode == 'learned'

Graph modes:
    - none: no message passing
    - given: use provided edge_index (e.g. synthetic ground truth)
    - random: random edge_index sampled at init
    - distance: not implemented (requires anatomical metadata)
    - train_correlation: caller must compute from train data only
    - learned: learn an N x N gating mask
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import CausalTemporalEncoder, GraphMessagePassing, MLP, softplus_positive


class GraphLatentDynamicsModel(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        history_len: int,
        horizon: int,
        hidden_dim: int = 64,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        graph_mode: str = "none",
        n_temporal_layers: int = 1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.history_len = history_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.graph_mode = graph_mode

        self.input_proj = nn.Linear(1, hidden_dim)
        self.temporal = CausalTemporalEncoder(
            in_dim=hidden_dim, hidden_dim=hidden_dim, n_layers=n_temporal_layers
        )
        self.mp = GraphMessagePassing(hidden_dim=hidden_dim, edge_attr_dim=1)
        self.decoder = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=1, n_layers=2)

        if graph_mode == "learned":
            # Learn an N x N gating logit matrix (no self-loops via mask at use time).
            self.edge_logits = nn.Parameter(torch.zeros(n_neurons, n_neurons))
            self.register_buffer("edge_index", torch.empty(2, 0, dtype=torch.long))
            self.register_buffer("edge_attr", torch.empty(0, 1))
        else:
            if edge_index is None:
                edge_index = torch.empty(2, 0, dtype=torch.long)
            self.register_buffer("edge_index", edge_index.long())
            if edge_attr is None or edge_attr.numel() == 0:
                edge_attr = torch.ones(edge_index.shape[1], 1)
            self.register_buffer("edge_attr", edge_attr.float())

    def _get_edges(self, sparsity: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph_mode == "learned":
            mask = 1 - torch.eye(self.n_neurons, device=self.edge_logits.device)
            gates = torch.sigmoid(self.edge_logits) * mask
            # Top-k threshold per target to keep graph sparse during training stability.
            k = max(1, int(sparsity * self.n_neurons))
            topk_vals, topk_idx = gates.topk(k=k, dim=0)
            edge_index = torch.stack(
                [
                    topk_idx.reshape(-1),
                    torch.arange(self.n_neurons, device=gates.device)
                    .unsqueeze(0)
                    .expand(k, -1)
                    .reshape(-1),
                ],
                dim=0,
            )
            edge_attr = topk_vals.reshape(-1, 1)
            return edge_index, edge_attr, gates
        return self.edge_index, self.edge_attr, torch.zeros(0)

    def forward(self, batch: dict) -> dict:
        hist = batch["history_counts"]  # [B, T, N]
        B, T, N = hist.shape
        device = hist.device
        H = self.horizon

        # Encode as [B, T*N, hidden] grouped per neuron, but simpler: run temporal per neuron.
        x = hist.unsqueeze(-1)  # [B, T, N, 1]
        x = self.input_proj(x)  # [B, T, N, H]
        # Run temporal across time for each neuron independently.
        BNT = x.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        temp_out = self.temporal(BNT)  # [B*N, T, H]
        last = temp_out[:, -1, :].reshape(B, N, self.hidden_dim)  # [B, N, H]

        # Graph message passing.
        edge_index, edge_attr, gates_mat = self._get_edges()
        if edge_index.shape[1] > 0:
            h = self.mp(last, edge_index=edge_index.to(device), edge_attr=edge_attr.to(device))
        else:
            h = last

        # Multi-step forecast: simple non-autoregressive decoder predicting H steps.
        # Repeat h across H and decode.
        rates = []
        cur_h = h
        for _ in range(H):
            log_rate = self.decoder(cur_h).squeeze(-1)  # [B, N]
            rate = softplus_positive(log_rate)
            rates.append(rate)
            # Recurrence: simple residual update via temporal then graph.
            cur_h = cur_h + 0.1 * self.mp(
                cur_h, edge_index=edge_index.to(device), edge_attr=edge_attr.to(device)
            ) if edge_index.shape[1] > 0 else cur_h * 0.95 + 0.05 * h
        pred_rate = torch.stack(rates, dim=1)  # [B, H, N]

        out = {
            "pred": pred_rate,
            "aux": {
                "n_edges": int(edge_index.shape[1]),
                "graph_mode": self.graph_mode,
            },
        }
        if self.graph_mode == "learned":
            out["aux"]["edge_logits"] = self.edge_logits.detach()
            out["aux"]["gates_matrix"] = gates_mat.detach()
        if "future_counts" in batch:
            out["target"] = batch["future_counts"]
        if "mask" in batch:
            out["mask"] = batch["mask"]
        return out

    def learned_adjacency(self) -> torch.Tensor | None:
        if self.graph_mode != "learned":
            return None
        mask = 1 - torch.eye(self.n_neurons, device=self.edge_logits.device)
        return (torch.sigmoid(self.edge_logits) * mask).detach()
