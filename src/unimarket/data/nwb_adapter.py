"""Generic NWB scanner / adapter.

Functional when ``pynwb`` is installed and a local ``.nwb`` file is provided.
Without ``pynwb``, calls raise a clear ImportError-derived message rather than
silently doing nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import save_npy, write_json


def _require_pynwb():
    try:
        import pynwb  # noqa: F401
        from pynwb import NWBHDF5IO  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "pynwb is not installed. Install with `pip install -e .[nwb]` to use the NWB adapter."
        ) from e


def scan_nwb(path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Open an NWB file and write a normalized manifest plus extracted arrays.

    Extracts:
      - acquisitions: TimeSeries data (voltage/current-like signals) -> .npy
      - units table: spike_times per unit -> .npy
      - processing modules: names only
    """
    _require_pynwb()
    from pynwb import NWBHDF5IO

    src = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source": str(src),
        "acquisitions": [],
        "processing_modules": [],
        "units": [],
        "adapter_status": "functional_with_local_nwb",
    }

    with NWBHDF5IO(str(src), mode="r", load_namespaces=True) as io:
        nwb = io.read()

        # Acquisitions: extract numeric TimeSeries-like data.
        for name, obj in (nwb.acquisition or {}).items():
            entry: dict[str, Any] = {"name": name, "type": type(obj).__name__}
            try:
                data = np.asarray(obj.data[:])
                rate = float(getattr(obj, "rate", 0.0) or 0.0)
                starting_time = float(getattr(obj, "starting_time", 0.0) or 0.0)
                arr_path = out / f"acq_{name}.npy"
                save_npy(arr_path, data)
                entry.update(
                    {
                        "shape": list(data.shape),
                        "dtype": str(data.dtype),
                        "rate_hz": rate,
                        "starting_time_s": starting_time,
                        "path": str(arr_path.relative_to(out)),
                    }
                )
            except Exception as e:  # pragma: no cover - defensive
                entry["error"] = repr(e)
            summary["acquisitions"].append(entry)

        # Processing modules.
        for name in list((nwb.processing or {}).keys()):
            summary["processing_modules"].append({"name": name})

        # Units table.
        if getattr(nwb, "units", None) is not None and len(nwb.units) > 0:
            for i in range(len(nwb.units)):
                row = nwb.units[i]
                try:
                    spike_times = np.asarray(row["spike_times"].iloc[0])
                except Exception:
                    spike_times = np.array([])
                unit_id = str(int(row.index[0]))
                p = out / f"unit_{unit_id}_spikes.npy"
                save_npy(p, spike_times)
                summary["units"].append(
                    {
                        "unit_id": unit_id,
                        "n_spikes": int(spike_times.size),
                        "spike_times_path": str(p.relative_to(out)),
                    }
                )

    write_json(out / "nwb_summary.json", summary)
    return summary
