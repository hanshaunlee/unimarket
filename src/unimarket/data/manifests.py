"""Manifest reading/writing helpers.

A 'manifest' is a CSV index listing per-record metadata for a processed dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.io import write_json


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def read_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def write_splits_json(
    splits: dict[str, list[str]], path: str | Path, split_mode: str, split_strength: str
) -> None:
    write_json(
        path,
        {
            "split_mode": split_mode,
            "split_strength": split_strength,
            "splits": {k: sorted(v) for k, v in splits.items()},
        },
    )


def write_preprocess_report(
    report: dict[str, Any], path: str | Path, normalization_meta: dict[str, Any]
) -> None:
    report = dict(report)
    report["normalization"] = normalization_meta
    write_json(path, report)
