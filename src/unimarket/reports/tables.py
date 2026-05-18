"""Result-table writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_table(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
