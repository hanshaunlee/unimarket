"""Synthetic graph recovery evaluation."""

from __future__ import annotations

import numpy as np

from ..utils.metrics import binary_auprc, binary_auroc, pearson_corr


def adjacency_from_edge_index(
    edge_index: np.ndarray, n_neurons: int, edge_weight: np.ndarray | None = None
) -> np.ndarray:
    A = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    if edge_index.shape[1] == 0:
        return A
    src, tgt = edge_index[0], edge_index[1]
    if edge_weight is None:
        A[tgt, src] = 1.0
    else:
        A[tgt, src] = edge_weight
    return A


def evaluate_graph_recovery(
    learned_adj: np.ndarray,
    true_edge_index: np.ndarray,
    true_edge_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare a learned adjacency to ground-truth synthetic edges.

    The learned adjacency is expected to be ``[N, N]`` with ``learned_adj[i, j]`` =
    edge gate from j -> i (matching the message-passing convention).
    """
    N = learned_adj.shape[0]
    true_adj = adjacency_from_edge_index(true_edge_index, N, edge_weight=None)
    # Off-diagonal only.
    mask = (1 - np.eye(N)).astype(bool)
    y = true_adj[mask]
    y_bin = (y > 0).astype(np.int64)
    s = learned_adj[mask].astype(np.float64)
    auroc = binary_auroc(s, y_bin)
    auprc = binary_auprc(s, y_bin)
    # Weight correlation if weights available.
    weight_corr = float("nan")
    if true_edge_weight is not None and true_edge_index.shape[1] > 0:
        # Compute learned weight at the true edge positions.
        src, tgt = true_edge_index[0], true_edge_index[1]
        learned_at_edges = learned_adj[tgt, src]
        weight_corr = pearson_corr(learned_at_edges, true_edge_weight)
    return {
        "edge_auroc": float(auroc),
        "edge_auprc": float(auprc),
        "edge_weight_correlation_at_true_edges": float(weight_corr),
        "n_true_edges": int(true_edge_index.shape[1]),
        "n_possible_edges": int(N * (N - 1)),
    }
