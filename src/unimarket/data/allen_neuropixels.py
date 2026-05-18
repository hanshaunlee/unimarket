"""Allen Visual Coding Neuropixels adapter (schema-validated local import).

Reads a user-provided CSV/Parquet of spike events with columns:
    unit_id, session_id, spike_time_s [, brain_area]
plus optional units metadata and stimulus table. Bins spikes into a [T, N]
count matrix and produces a normalized manifest. Does NOT attempt to download
Allen Neuropixels raw data; full ingestion is ``adapter_status:
schema_validated_only``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import save_npy, write_json
from .manifests import write_manifest

EXPECTED_COLS = {"unit_id", "session_id", "spike_time_s"}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ingest_spike_table(
    spikes_path: str | Path,
    out_dir: str | Path,
    bin_size_ms: float = 10.0,
    stimulus_table_path: str | Path | None = None,
    allow_stimulus_identity: bool = False,
) -> dict[str, Any]:
    src = Path(spikes_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _read_table(src)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Neuropixels spike table missing required columns: {sorted(missing)}. "
            f"Required columns: {sorted(EXPECTED_COLS)}."
        )

    unit_rows: list[dict[str, Any]] = []
    counts_by_session: dict[str, np.ndarray] = {}
    for session_id, sdf in df.groupby("session_id"):
        units = sdf["unit_id"].unique()
        units = np.asarray(sorted(units))
        unit_to_col = {u: i for i, u in enumerate(units.tolist())}
        T_end_s = float(sdf["spike_time_s"].max()) + 1e-3
        bin_s = bin_size_ms / 1000.0
        n_bins = int(np.ceil(T_end_s / bin_s))
        counts = np.zeros((n_bins, len(units)), dtype=np.int32)
        for u, udf in sdf.groupby("unit_id"):
            col = unit_to_col[u]
            idx = (udf["spike_time_s"].to_numpy() / bin_s).astype(int)
            idx = idx[idx < n_bins]
            np.add.at(counts, (idx, col), 1)
        out_path = out / f"session_{session_id}_counts.npy"
        save_npy(out_path, counts)
        counts_by_session[str(session_id)] = counts
        for u in units:
            row = {
                "unit_id": str(u),
                "session_id": str(session_id),
                "spike_times_path": "",
                "split": "train",
            }
            if "brain_area" in sdf.columns:
                ba = sdf[sdf["unit_id"] == u]["brain_area"].iloc[0]
                row["brain_area"] = str(ba)
            unit_rows.append(row)

    write_manifest(unit_rows, out / "manifest_units.csv")

    stim_info: dict[str, Any] = {"loaded": False, "allowed_in_model": bool(allow_stimulus_identity)}
    if stimulus_table_path is not None:
        stim_path = Path(stimulus_table_path)
        if stim_path.exists():
            stim_df = _read_table(stim_path)
            stim_df.to_csv(out / "stimulus_table.csv", index=False)
            stim_info["loaded"] = True
            stim_info["n_rows"] = int(len(stim_df))

    write_json(
        out / "preprocessing_report.json",
        {
            "dataset_name": "allen_neuropixels",
            "n_units": len(unit_rows),
            "n_sessions": len(counts_by_session),
            "bin_size_ms": bin_size_ms,
            "source_table": str(src),
            "stimulus_info": stim_info,
            "adapter_status": "schema_validated_only",
            "notes": (
                "Default forecasting model must NOT receive stimulus identity unless "
                "the experiment explicitly sets allow_stimulus_identity=true."
            ),
        },
    )
    return {
        "n_units": len(unit_rows),
        "n_sessions": len(counts_by_session),
        "out_dir": str(out),
        "adapter_status": "schema_validated_only",
    }
