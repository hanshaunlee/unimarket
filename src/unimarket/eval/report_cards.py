"""Generate a per-run, human-readable report card."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..constants import NEGATIVE_RESULTS_STATEMENT
from ..utils.io import read_json


def write_report_card(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    cfg = read_json(run / "config.json")
    metrics = read_json(run / "predictive_metrics_summary.json") if (run / "predictive_metrics_summary.json").exists() else {}
    gate = read_json(run / "claim_gate.json") if (run / "claim_gate.json").exists() else {}
    leak = read_json(run / "leakage_report.json") if (run / "leakage_report.json").exists() else {}
    cap = read_json(run / "capacity_report.json") if (run / "capacity_report.json").exists() else {}

    md = [
        f"# Report card: {run.name}",
        "",
        "## What was trained",
        f"- Model: `{cfg.get('model', {}).get('name', '?')}`",
        f"- Dataset kind: `{cfg.get('data', {}).get('kind', '?')}`",
        f"- Split mode: `{cfg.get('data', {}).get('split_mode', 'time_block')}`",
        f"- Selection metric / split: `{cfg.get('selection_metric', 'loss')}` / `{cfg.get('selection_split', 'val')}`",
        "",
        "## What was held out / forbidden",
        f"- Test split: held out; not used for checkpoint / normalization / probe selection.",
        f"- Stimulus identity allowed: `{bool(cfg.get('allow_stimulus_identity', False))}`",
        f"- Global time / trial / block features allowed: "
        f"`{bool(cfg.get('allow_global_time_features', False))}` / "
        f"`{bool(cfg.get('allow_trial_index', False))}` / "
        f"`{bool(cfg.get('allow_block_index', False))}`",
        f"- Cell/Pair/Session ID embeddings allowed: "
        f"`{bool(cfg.get('allow_cell_id_embedding', False))}` / "
        f"`{bool(cfg.get('allow_pair_id_embedding', False))}` / "
        f"`{bool(cfg.get('allow_session_id_embedding', False))}`",
        "",
        "## Metrics",
    ]
    for split, m in metrics.items():
        if not m:
            continue
        md.append(f"### {split}")
        for k, v in m.items():
            try:
                md.append(f"- {k}: {float(v):.6f}")
            except Exception:
                md.append(f"- {k}: {v}")
        md.append("")
    md.append("## Capacity")
    if cap:
        md.append(f"- n_trainable_params: {cap.get('n_trainable_params')}")
        md.append(f"- wall_clock_s: {cap.get('wall_clock_s')}")
        md.append(f"- device: {cap.get('device')}")
        md.append(f"- epochs: {cap.get('n_epochs')} (steps: {cap.get('n_steps')})")
    md.append("")
    md.append("## Leakage")
    md.append(f"- status: `{leak.get('status', '?')}`")
    md.append("")
    md.append("## Claim gate")
    md.append(f"- status: `{gate.get('status', '?')}`")
    md.append(f"- allowed: {len(gate.get('allowed', []))}")
    md.append(f"- disallowed: {len(gate.get('disallowed', []))}")
    md.append("See `claim_gate.md` for full details.")
    md.append("")
    md.append("---")
    md.append(NEGATIVE_RESULTS_STATEMENT)
    out = run / "report_card.md"
    out.write_text("\n".join(md))
    return out
