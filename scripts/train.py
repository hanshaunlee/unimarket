#!/usr/bin/env python
"""Train a model from a YAML experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

from unimarket.train.train import train_from_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    run_dir = train_from_config(args.config)
    print(f"Training complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
