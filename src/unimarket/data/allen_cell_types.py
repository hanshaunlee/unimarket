"""Allen Cell Types local NWB adapter.

Reads local NWB files (or a directory of NWB files) and emits SweepRecords plus
extracted stimulus/response arrays. Functional with ``pynwb`` and local files.
Does not initiate any download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import save_npy, sha256_file, write_json
from .manifests import write_manifest


def _require_pynwb():
    try:
        import pynwb  # noqa: F401
        return True
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "pynwb is required for the Allen Cell Types adapter. "
            "Install with `pip install -e .[nwb]`."
        ) from e


def ingest_directory(
    nwb_dir: str | Path, out_dir: str | Path, dataset_name: str = "allen_cell_types"
) -> dict[str, Any]:
    """Ingest a directory of Allen Cell Types NWB files.

    Output structure:
      out_dir/
        cells/<cell_id>/<sweep_id>_stim.npy
        cells/<cell_id>/<sweep_id>_resp.npy
        manifest_cells.csv
        manifest_sweeps.csv
        preprocessing_report.json
    """
    _require_pynwb()
    from pynwb import NWBHDF5IO

    src = Path(nwb_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.nwb"))
    if not files:
        raise FileNotFoundError(f"No .nwb files in {src}")

    cell_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    checksums: list[dict[str, Any]] = []

    for f in files:
        digest = sha256_file(f)
        checksums.append({"file": f.name, "sha256": digest})
        cell_id = f.stem
        cell_dir = out / "cells" / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(f), mode="r", load_namespaces=True) as io:
            nwb = io.read()
            subj = getattr(nwb, "subject", None)
            cell_rows.append(
                {
                    "dataset_name": dataset_name,
                    "cell_id": cell_id,
                    "species": getattr(subj, "species", None) if subj else None,
                    "subject_id": getattr(subj, "subject_id", None) if subj else None,
                    "session_id": getattr(nwb, "session_id", None),
                    "source_file": f.name,
                }
            )
            sweep_idx = 0
            for name, obj in (nwb.acquisition or {}).items():
                try:
                    data = np.asarray(obj.data[:])
                    rate = float(getattr(obj, "rate", 0.0) or 0.0)
                except Exception:
                    continue
                sweep_id = f"{cell_id}_sweep{sweep_idx:04d}"
                # We don't have a guaranteed stim/response pairing for every acquisition,
                # so we emit the trace itself as 'response_path' and leave stimulus_path
                # to be discovered separately for now.
                resp_path = cell_dir / f"{sweep_id}_resp.npy"
                save_npy(resp_path, data)
                sweep_rows.append(
                    {
                        "sweep_id": sweep_id,
                        "cell_id": cell_id,
                        "protocol_name": name,
                        "stimulus_path": "",
                        "response_path": str(resp_path.relative_to(out)),
                        "sampling_rate_hz": rate,
                        "duration_s": float(data.shape[0]) / rate if rate > 0 else float("nan"),
                        "split": "train",
                    }
                )
                sweep_idx += 1

    write_manifest(cell_rows, out / "manifest_cells.csv")
    write_manifest(sweep_rows, out / "manifest_sweeps.csv")
    write_json(
        out / "preprocessing_report.json",
        {
            "dataset_name": dataset_name,
            "n_cells": len(cell_rows),
            "n_sweeps": len(sweep_rows),
            "source_dir": str(src),
            "checksums": checksums,
            "adapter_status": "functional_with_local_nwb",
        },
    )
    return {
        "n_cells": len(cell_rows),
        "n_sweeps": len(sweep_rows),
        "out_dir": str(out),
        "adapter_status": "functional_with_local_nwb",
    }
