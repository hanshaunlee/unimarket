"""CSV/JSONL/console training loggers (no external dependencies)."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from .io import append_jsonl


def get_logger(name: str = "unimarket", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class RunLogger:
    """Lightweight per-run logger that writes JSONL events and a CSV metrics table."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.metrics_path = self.run_dir / "metrics.csv"
        self._metric_keys: list[str] = []
        self._console = get_logger(f"unimarket.run[{self.run_dir.name}]")

    def info(self, msg: str) -> None:
        self._console.info(msg)
        append_jsonl(self.events_path, {"level": "info", "msg": msg})

    def warn(self, msg: str) -> None:
        self._console.warning(msg)
        append_jsonl(self.events_path, {"level": "warn", "msg": msg})

    def error(self, msg: str) -> None:
        self._console.error(msg)
        append_jsonl(self.events_path, {"level": "error", "msg": msg})

    def log_metrics(self, step: int, metrics: Mapping[str, Any]) -> None:
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        # Maintain a stable column order, extending as new keys appear.
        new_keys = [k for k in row.keys() if k not in self._metric_keys]
        if new_keys:
            self._metric_keys.extend(new_keys)
            self._rewrite_with_new_header()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._metric_keys)
            writer.writerow({k: row.get(k, "") for k in self._metric_keys})

    def _rewrite_with_new_header(self) -> None:
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._metric_keys)
                writer.writeheader()
            return
        existing = list(csv.DictReader(self.metrics_path.open("r", encoding="utf-8")))
        with self.metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._metric_keys)
            writer.writeheader()
            for row in existing:
                writer.writerow({k: row.get(k, "") for k in self._metric_keys})
