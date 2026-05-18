"""Common evaluation metrics."""

from __future__ import annotations

import numpy as np

from ..constants import EPS


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, target)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < EPS:
        return float("nan")
    return float((a * b).sum() / denom)


def poisson_nll(rate: np.ndarray, counts: np.ndarray) -> float:
    rate = np.clip(np.asarray(rate, dtype=np.float64), EPS, None)
    counts = np.asarray(counts, dtype=np.float64)
    return float(np.mean(rate - counts * np.log(rate)))


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC without sklearn (Mann-Whitney U formulation)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i : j + 1]]).mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def binary_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (area under PR curve)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels == 1).astype(np.float64)
    fp = np.cumsum(labels == 0).astype(np.float64)
    n_pos = float((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    precision = tp / np.clip(tp + fp, EPS, None)
    recall = tp / n_pos
    # Average precision: sum over rank of (recall_i - recall_{i-1}) * precision_i
    prev_recall = 0.0
    ap = 0.0
    for p, r in zip(precision.tolist(), recall.tolist()):
        ap += (r - prev_recall) * p
        prev_recall = r
    return float(ap)


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(np.int64)
    pred = (scores >= threshold).astype(np.int64)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return float(2 * precision * recall / (precision + recall + EPS))
