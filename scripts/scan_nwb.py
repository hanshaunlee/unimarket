#!/usr/bin/env python
"""Scan a local NWB file and write a normalized manifest plus extracted arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from unimarket.data.nwb_adapter import scan_nwb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="Local .nwb file path")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    summary = scan_nwb(args.path, args.out)
    print(f"Scanned {args.path}.")
    print(f"  Acquisitions: {len(summary['acquisitions'])}")
    print(f"  Units: {len(summary['units'])}")
    print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
