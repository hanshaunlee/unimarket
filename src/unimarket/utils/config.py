"""YAML config loading with simple include/merge semantics."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Support an optional `include:` field with a list of relative paths.
    includes = cfg.pop("include", None)
    if includes:
        merged: dict[str, Any] = {}
        for inc in includes:
            inc_path = (p.parent / inc).resolve()
            merged = _deep_merge(merged, load_yaml(inc_path))
        cfg = _deep_merge(merged, cfg)
    return cfg


def save_yaml(path: str | Path, cfg: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Access nested config values via 'a.b.c' style keys."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
