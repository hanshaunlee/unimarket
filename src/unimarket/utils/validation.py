"""Hard-fail validation utilities for scientific safeguards.

These functions enforce UniMarket's scientific contract. They raise rather
than warn so that misconfigured runs cannot silently produce invalid claims.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


class CircularityError(ValueError):
    """Raised when a configuration would produce circular or leaky results."""


# ---------- Split / leakage validation ----------

FORBIDDEN_STIMULUS_FEATURES = {
    "stimulus_id",
    "stimulus_label",
    "stimulus_onehot",
    "trial_index",
    "block_index",
    "global_time",
    "absolute_timestamp",
    "presentation_number",
}

FORBIDDEN_ID_EMBEDDINGS = {
    "cell_id_embedding",
    "pair_id_embedding",
    "session_id_embedding",
}


def assert_no_overlap(name: str, train: Iterable, val: Iterable, test: Iterable) -> None:
    """Fail if the same identifier appears in more than one split."""
    s_tr, s_va, s_te = set(train), set(val), set(test)
    tr_va = s_tr & s_va
    tr_te = s_tr & s_te
    va_te = s_va & s_te
    if tr_va or tr_te or va_te:
        raise CircularityError(
            f"Split leakage on '{name}': "
            f"train∩val={sorted(tr_va)[:5]} train∩test={sorted(tr_te)[:5]} "
            f"val∩test={sorted(va_te)[:5]}"
        )


def assert_time_blocks_disjoint(blocks: dict[str, list[tuple[float, float]]]) -> None:
    """Fail if time ranges overlap across splits.

    ``blocks`` is a mapping like ``{"train": [(0,10)], "val": [(10,12)], "test": [(12,14)]}``.
    """
    items = list(blocks.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, intervals_a = items[i]
            b, intervals_b = items[j]
            for (s1, e1) in intervals_a:
                for (s2, e2) in intervals_b:
                    if s1 < e2 and s2 < e1:
                        raise CircularityError(
                            f"Time block overlap between '{a}' and '{b}': "
                            f"({s1},{e1}) vs ({s2},{e2})"
                        )


def assert_window_random_split_allowed(split_mode: str, allow_leaky_split: bool) -> None:
    if split_mode == "window_random" and not allow_leaky_split:
        raise CircularityError(
            "split_mode='window_random' is not allowed for neural time series. "
            "Adjacent windows leak heavily. Set allow_leaky_split=true to override "
            "and accept that no generalization claim can be made."
        )


# ---------- Feature / leakage validation ----------

def assert_no_forbidden_stimulus_features(
    feature_names: Iterable[str],
    *,
    allow_stimulus_identity: bool,
    allow_global_time_features: bool,
    allow_trial_index: bool,
    allow_block_index: bool,
) -> None:
    names = set(feature_names)
    violations: list[str] = []
    for feat in names:
        if feat in {"stimulus_id", "stimulus_label", "stimulus_onehot"} and not allow_stimulus_identity:
            violations.append(feat)
        if feat in {"global_time", "absolute_timestamp"} and not allow_global_time_features:
            violations.append(feat)
        if feat == "trial_index" and not allow_trial_index:
            violations.append(feat)
        if feat == "block_index" and not allow_block_index:
            violations.append(feat)
    if violations:
        raise CircularityError(
            "Stimulus leakage: forbidden features present without explicit allow flag: "
            f"{violations}"
        )


def assert_no_forbidden_id_embeddings(
    model_uses: Mapping[str, bool],
    *,
    allow_cell_id_embedding: bool,
    allow_pair_id_embedding: bool,
    allow_session_id_embedding: bool,
    split_mode: str | None = None,
) -> None:
    violations: list[str] = []
    if model_uses.get("cell_id_embedding") and not allow_cell_id_embedding:
        violations.append("cell_id_embedding")
    if model_uses.get("pair_id_embedding") and not allow_pair_id_embedding:
        violations.append("pair_id_embedding")
    if model_uses.get("session_id_embedding") and not allow_session_id_embedding:
        violations.append("session_id_embedding")
    if violations:
        raise CircularityError(
            "Identity-leakage embedding(s) enabled without explicit allow flag: "
            f"{violations}. Held-out identity claims require these to remain disabled."
        )
    # Per-split additional check: even with allow flags, held-out identity claims
    # cannot use embeddings of held-out identities.
    if split_mode and "heldout_cell" in split_mode and model_uses.get("cell_id_embedding"):
        raise CircularityError(
            "cell_id_embedding=true under a held-out-cell split is forbidden because "
            "the model cannot have an embedding for a never-seen cell."
        )


def assert_normalization_from_train_only(stats_meta: Mapping[str, Any]) -> None:
    used = set(stats_meta.get("computed_on_splits", []))
    if not used:
        raise CircularityError("normalization_stats.json missing computed_on_splits")
    if used != {"train"}:
        raise CircularityError(
            f"Normalization statistics computed on {sorted(used)} but must use only 'train'."
        )


def assert_checkpoint_not_test(selection_metric_split: str) -> None:
    if selection_metric_split.lower() == "test":
        raise CircularityError(
            "Checkpoint cannot be selected by a test-split metric. Use 'val'."
        )
