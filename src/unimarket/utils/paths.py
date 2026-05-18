"""Project paths and run-directory management."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    Resolves by walking up from this file until ``pyproject.toml`` is found.
    Falls back to the current working directory if not found.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def outputs_dir() -> Path:
    d = repo_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = outputs_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(subdir: str = "") -> Path:
    base = repo_root() / "data"
    if subdir:
        d = base / subdir
    else:
        d = base
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_run_dir(name: str | None = None) -> Path:
    """Create and return a fresh run directory under outputs/runs/."""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = name if name else "run"
    run_dir = runs_dir() / f"{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Update a 'latest' symlink for convenience.
    latest = runs_dir() / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(run_dir, latest)
    except OSError:
        # Symlinks may fail on some filesystems; ignore.
        pass
    return run_dir
