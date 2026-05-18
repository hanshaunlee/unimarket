"""Leakage / circularity checks. Returns PASS/FAIL JSON, raises on hard failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.io import read_json, write_json
from ..utils.validation import (
    CircularityError,
    assert_checkpoint_not_test,
    assert_no_forbidden_id_embeddings,
    assert_no_forbidden_stimulus_features,
    assert_no_overlap,
    assert_normalization_from_train_only,
    assert_time_blocks_disjoint,
    assert_window_random_split_allowed,
)


def run_leakage_checks(run_dir: str | Path) -> dict[str, Any]:
    """Inspect a run directory and produce a leakage report.

    Returns a dict with per-check entries and a top-level ``status``.
    """
    run = Path(run_dir)
    cfg = read_json(run / "config.json")
    report: dict[str, Any] = {"run_dir": str(run), "checks": [], "status": "PASS"}

    def _check(name: str, fn) -> None:
        try:
            fn()
            report["checks"].append({"name": name, "status": "PASS"})
        except CircularityError as e:
            report["checks"].append({"name": name, "status": "FAIL", "message": str(e)})
            report["status"] = "FAIL"
        except Exception as e:
            report["checks"].append(
                {"name": name, "status": "SKIP", "message": f"{type(e).__name__}: {e}"}
            )

    data_cfg = cfg.get("data", {})

    _check(
        "window_random_split_guard",
        lambda: assert_window_random_split_allowed(
            data_cfg.get("split_mode", "time_block"),
            bool(data_cfg.get("allow_leaky_split", False)),
        ),
    )

    _check(
        "stimulus_feature_guard",
        lambda: assert_no_forbidden_stimulus_features(
            data_cfg.get("model_features", ["history_voltage", "history_current"]),
            allow_stimulus_identity=bool(cfg.get("allow_stimulus_identity", False)),
            allow_global_time_features=bool(cfg.get("allow_global_time_features", False)),
            allow_trial_index=bool(cfg.get("allow_trial_index", False)),
            allow_block_index=bool(cfg.get("allow_block_index", False)),
        ),
    )

    _check(
        "id_embedding_guard",
        lambda: assert_no_forbidden_id_embeddings(
            cfg.get("model", {}).get("uses", {}),
            allow_cell_id_embedding=bool(cfg.get("allow_cell_id_embedding", False)),
            allow_pair_id_embedding=bool(cfg.get("allow_pair_id_embedding", False)),
            allow_session_id_embedding=bool(cfg.get("allow_session_id_embedding", False)),
            split_mode=data_cfg.get("split_mode", "time_block"),
        ),
    )

    _check(
        "checkpoint_not_test",
        lambda: assert_checkpoint_not_test(cfg.get("selection_split", "val")),
    )

    # Normalization stats files (if present).
    norm_paths = [
        run / "normalization_stats.voltage.json",
        run / "normalization_stats.current.json",
    ]
    for p in norm_paths:
        if p.exists():
            stats = read_json(p)
            _check(
                f"normalization_train_only::{p.name}",
                lambda s=stats: assert_normalization_from_train_only(s),
            )

    # Splits file (if present in processed dir).
    processed_dir = (
        Path(cfg.get("data", {}).get("processed_dir", "")) if cfg.get("data") else None
    )
    if processed_dir is not None and (processed_dir / "splits.json").exists():
        splits_meta = read_json(processed_dir / "splits.json")
        tb = splits_meta.get("time_blocks", {})
        _check(
            "time_block_disjoint",
            lambda: assert_time_blocks_disjoint(
                {k: [tuple(b) for b in v] for k, v in tb.items()}
            ),
        )
        # If neuron splits exist, also check no neuron-overlap when used.
        ho = splits_meta.get("heldout_neurons", [])
        seen = splits_meta.get("seen_neurons", [])
        _check(
            "heldout_seen_neurons_disjoint",
            lambda: assert_no_overlap("neuron_id", seen, [], ho),
        )

    write_json(run / "leakage_report.json", report)
    return report
