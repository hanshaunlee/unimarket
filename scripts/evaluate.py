#!/usr/bin/env python
"""Evaluate a trained run on a split."""

from __future__ import annotations

import argparse
from pathlib import Path

from unimarket.eval.evaluate import evaluate_run
from unimarket.eval.report_cards import write_report_card


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    args = p.parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    summary = evaluate_run(args.run_dir, splits=splits)
    write_report_card(args.run_dir)
    print(f"Eval done. Claim gate: {summary['claim_gate']['status']}; "
          f"allowed={len(summary['claim_gate']['allowed'])}, "
          f"disallowed={len(summary['claim_gate']['disallowed'])}.")


if __name__ == "__main__":
    main()
