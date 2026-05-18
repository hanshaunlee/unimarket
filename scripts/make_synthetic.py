#!/usr/bin/env python
"""Generate synthetic LIF circuit data."""

from __future__ import annotations

import argparse
from pathlib import Path

from unimarket.data.synthetic import SyntheticConfig, write_synthetic
from unimarket.utils.config import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    sc_kwargs = dict(cfg.get("synthetic", {}))
    sc = SyntheticConfig(**sc_kwargs)
    info = write_synthetic(args.out, sc)
    print(f"Generated synthetic data at {info['out_dir']} (N={info['N']}, T={info['T']}, edges={info['edges']})")


if __name__ == "__main__":
    main()
