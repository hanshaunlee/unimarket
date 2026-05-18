#!/usr/bin/env python
"""Prepare a dataset from its data config."""

from __future__ import annotations

import argparse
from pathlib import Path

from unimarket.cli import prepare


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    prepare(config=args.config, out=args.out)


if __name__ == "__main__":
    main()
