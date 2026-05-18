"""Split definitions and leakage-aware split helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..utils.seed import seeded_rng
from ..utils.validation import assert_no_overlap


def _three_way(items: list, fracs: tuple[float, float, float], rng: np.random.Generator) -> dict:
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def split_by_cell(
    cell_ids: Iterable[str],
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    items = sorted(set(cell_ids))
    rng = seeded_rng(seed)
    out = _three_way(list(items), fracs, rng)
    assert_no_overlap("cell_id", out["train"], out["val"], out["test"])
    return out


def split_by_session(
    session_ids: Iterable[str],
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    items = sorted(set(session_ids))
    rng = seeded_rng(seed)
    out = _three_way(list(items), fracs, rng)
    assert_no_overlap("session_id", out["train"], out["val"], out["test"])
    return out


def split_by_subject(
    subject_ids: Iterable[str],
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    items = sorted(set(subject_ids))
    rng = seeded_rng(seed)
    out = _three_way(list(items), fracs, rng)
    assert_no_overlap("subject_id", out["train"], out["val"], out["test"])
    return out


def split_by_protocol(
    protocols: Iterable[str],
    test_protocol_families: list[str] | None = None,
    val_protocol_families: list[str] | None = None,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Assign protocol families to train/val/test (disjoint by family).

    If ``test_protocol_families`` is provided, those families are placed in test;
    likewise for ``val_protocol_families``. Remaining families go to train.
    Otherwise, a random three-way family-level split is created.
    """
    items = sorted(set(protocols))
    if test_protocol_families is None and val_protocol_families is None:
        rng = seeded_rng(seed)
        out = _three_way(list(items), (0.7, 0.15, 0.15), rng)
        assert_no_overlap("protocol_family", out["train"], out["val"], out["test"])
        return out
    test_protocol_families = test_protocol_families or []
    val_protocol_families = val_protocol_families or []
    train = [p for p in items if p not in test_protocol_families and p not in val_protocol_families]
    out = {"train": train, "val": list(val_protocol_families), "test": list(test_protocol_families)}
    assert_no_overlap("protocol_family", out["train"], out["val"], out["test"])
    return out


def split_by_pair(
    pair_ids: Iterable[str],
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    items = sorted(set(pair_ids))
    rng = seeded_rng(seed)
    out = _three_way(list(items), fracs, rng)
    assert_no_overlap("pair_id", out["train"], out["val"], out["test"])
    return out


def split_by_time_block(
    T: int,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[tuple[int, int]]]:
    """Contiguous time-block split. Never overlap.

    Returns time block lists for each split as ``[(start, end), ...]``.
    """
    n_train = int(round(fracs[0] * T))
    n_val = int(round(fracs[1] * T))
    train_end = n_train
    val_end = n_train + n_val
    return {
        "train": [(0, train_end)],
        "val": [(train_end, val_end)],
        "test": [(val_end, T)],
    }


def split_synthetic_by_graph_and_seed(
    n_graphs: int,
    seed: int = 0,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[int]]:
    """Assign synthetic graphs to splits by graph index."""
    rng = seeded_rng(seed)
    items = list(range(n_graphs))
    out = _three_way(items, fracs, rng)
    assert_no_overlap("graph_id", out["train"], out["val"], out["test"])
    return out


def cell_protocol_split(
    pairs: list[tuple[str, str]],
    mode: str,
    fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[tuple[str, str]]]:
    """Compose cell- and protocol-level splits.

    ``mode`` is one of:
      - seen_cell_heldout_sweep (debug only on sweep IDs, not handled here)
      - seen_cell_heldout_protocol_family
      - heldout_cell_seen_protocol_family
      - heldout_cell_heldout_protocol_family

    Each item in ``pairs`` is ``(cell_id, protocol_family)``.
    """
    cells = sorted({c for c, _ in pairs})
    protos = sorted({p for _, p in pairs})

    if mode == "seen_cell_heldout_protocol_family":
        proto_splits = split_by_protocol(protos, seed=seed)
        out = {k: [] for k in ("train", "val", "test")}
        for c, p in pairs:
            for k in out:
                if p in proto_splits[k]:
                    out[k].append((c, p))
        return out
    if mode == "heldout_cell_seen_protocol_family":
        cell_splits = split_by_cell(cells, fracs=fracs, seed=seed)
        out = {k: [] for k in ("train", "val", "test")}
        for c, p in pairs:
            for k in out:
                if c in cell_splits[k]:
                    out[k].append((c, p))
        return out
    if mode == "heldout_cell_heldout_protocol_family":
        cell_splits = split_by_cell(cells, fracs=fracs, seed=seed)
        proto_splits = split_by_protocol(protos, seed=seed + 1)
        out = {k: [] for k in ("train", "val", "test")}
        for c, p in pairs:
            for k in out:
                if c in cell_splits[k] and p in proto_splits[k]:
                    out[k].append((c, p))
        return out
    raise ValueError(f"Unknown cell_protocol split mode: {mode}")
