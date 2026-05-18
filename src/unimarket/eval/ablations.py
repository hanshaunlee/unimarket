"""Helpers for running and summarizing ablation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.io import write_json


def collect_ablation_results(
    run_dirs: list[str | Path], names: list[str], metric_key: str = "val_loss"
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, rd in zip(names, run_dirs):
        rd = Path(rd)
        summary_path = rd / "train_summary.json"
        if not summary_path.exists():
            rows.append({"name": name, "run_dir": str(rd), "val_metric": float("nan")})
            continue
        from ..utils.io import read_json

        s = read_json(summary_path)
        best = s.get("best_val_metric")
        rows.append({"name": name, "run_dir": str(rd), "val_metric": best})
    return {"rows": rows, "metric_key": metric_key}


def write_ablation_summary(out_dir: str | Path, summary: dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "ablation_summary.json", summary)
    pd.DataFrame(summary["rows"]).to_csv(out / "ablation_summary.csv", index=False)
    return out / "ablation_summary.json"
