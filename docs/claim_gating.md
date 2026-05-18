# Claim gating

`src/unimarket/eval/claim_gating.py` produces a per-run `claim_gate.json` and `claim_gate.md`.

## Inputs

- `config.json` — frozen config
- `leakage_report.json` — must pass
- `identifiability_report.json` — for mechanistic claims
- `interpretability_report.json` — for latent claims
- `graph_recovery.json` — for graph claims (synthetic only)
- `predictive_metrics_summary.json` — for predictive claims
- `capacity_report.json` — for capacity disclosure
- `ablation_summary.json` — for "graph helps" / "physics helps" claims (when present)

## Output structure

```jsonc
{
  "status": "VALID" | "INVALID",
  "reasons": [...],
  "allowed": [
    {"claim": "...", "evidence": "...", "caveats": "..."}
  ],
  "disallowed": [
    {"claim": "...", "reason": "...", "how_to_make_valid": "..."}
  ]
}
```

## How decisions are made

The gate is a conservative whitelist: nothing is allowed by default. Claims are added only when explicit evidence is found. A run is `INVALID` if any of the leakage / circularity invariants fail (see `docs/anti_circularity.md`).

Examples of automatic decisions:

- Synthetic-data run → "validated on public biological recordings" is **disallowed**.
- `graph_mode=learned` on synthetic data with `edge_auroc > 0.6` → "recovers synthetic graph above chance" is **allowed** (with caveat that this is synthetic, not biological).
- Identifiability label `predictive_latent_only` → "latent is biologically meaningful" is **disallowed** with the reason "recovery correlation below threshold."
- `selection_split=test` → run is **invalid** entirely.
