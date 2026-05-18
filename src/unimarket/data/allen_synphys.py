"""Allen Synaptic Physiology adapter (schema-validated local import).

Expects a user-provided CSV manifest with the following columns:
    pair_id, pre_cell_id, post_cell_id, stimulus_path, response_path,
    sampling_rate_hz, [connection_label]

The adapter validates this schema and produces a normalized
manifest_pairs.csv plus a preprocessing_report.json. It does NOT attempt to
download Allen SynPhys raw data; full ingestion from the upstream raw format is
``adapter_status: schema_validated_only``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.io import write_json
from .manifests import write_manifest
from .schemas import SynapsePairRecord

EXPECTED_COLS = {
    "pair_id",
    "pre_cell_id",
    "post_cell_id",
    "stimulus_path",
    "response_path",
    "sampling_rate_hz",
}


def ingest_from_csv(csv_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    src = Path(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"SynPhys manifest CSV not found: {src}")
    df = pd.read_csv(src)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"SynPhys CSV is missing required columns: {sorted(missing)}. "
            f"Required columns: {sorted(EXPECTED_COLS)}."
        )

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rec = SynapsePairRecord(
            pair_id=str(r["pair_id"]),
            pre_cell_id=str(r["pre_cell_id"]),
            post_cell_id=str(r["post_cell_id"]),
            connection_label=str(r["connection_label"]) if "connection_label" in df.columns else None,
            stimulus_path=str(r["stimulus_path"]),
            response_path=str(r["response_path"]),
            sampling_rate_hz=float(r["sampling_rate_hz"]),
            split="train",
        )
        rows.append(rec.model_dump())

    write_manifest(rows, out / "manifest_pairs.csv")
    write_json(
        out / "preprocessing_report.json",
        {
            "dataset_name": "allen_synphys",
            "n_pairs": len(rows),
            "source_csv": str(src),
            "adapter_status": "schema_validated_only",
            "notes": (
                "This adapter validates a user-supplied manifest CSV. Full SynPhys raw "
                "ingestion is planned future work."
            ),
        },
    )
    return {
        "n_pairs": len(rows),
        "out_dir": str(out),
        "adapter_status": "schema_validated_only",
    }
