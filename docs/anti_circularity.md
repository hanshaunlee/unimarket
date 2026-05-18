# Anti-circularity

UniMarket enforces non-circular evaluation through *hard* checks (raise on failure, not warn). They live in:

- `src/unimarket/utils/validation.py`
- `src/unimarket/eval/leakage.py`

## What can make a run invalid

A run is marked **invalid** by the claim gate if any of these are true:

- test split is used for early stopping, checkpoint selection, normalization, graph construction, threshold selection, hyperparameter selection, or probe selection
- probe labels are used in predictive training
- graph construction uses validation or test activity (unless marked leaky and excluded from graph claims)
- synthetic ground truth is used in the predictive loss
- stimulus identity is indirectly encoded through global block index, trial index, absolute timestamps, etc., when stimulus-ablated
- `cell_id_embedding` is used under a held-out-cell split
- `pair_id_embedding` is used under a held-out-pair split
- `session_id_embedding` is used under a held-out-session split
- normalization statistics are computed on validation or test data
- random window-level splits are used without `allow_leaky_split=true`

## What controls every run carries

- `leakage_report.json` with per-check PASS/FAIL.
- `normalization_stats.*.json` with `computed_on_splits=["train"]`.
- `predictive_metrics_summary.json` with per-split metrics; checkpoint is selected by `val` only.
- `interpretability_controls.csv` containing the real probe alongside three independent negative controls.
- `claim_gate.json` summarizing which claims are allowed / disallowed.

## Cell-embedding rule

Cell embeddings can leak identity. The default is:

```yaml
allow_cell_id_embedding: false
allow_pair_id_embedding: false
allow_session_id_embedding: false
```

For held-out-cell generalization, learned cell embeddings are **always** forbidden because the model cannot have an embedding for a never-seen cell. If a run uses metadata embeddings, mark `metadata_conditioned: true` and accept that the run cannot support unsupervised interpretability claims.

## Stimulus leakage rule

If stimulus labels are withheld, the model must also exclude indirect schedule identifiers: trial index, block index, global time, absolute timestamps, one-hot stimulus rows, deterministic stimulus order features. Allowed: local elapsed time inside the window, neural history.

## Graph semantics rule

For extracellular population data without anatomical ground truth, learned edges are **functional predictive dependencies**, not synapses. UniMarket reports a `graph_claim_type` that distinguishes:

- `synthetic_connectivity` — synthetic data with true graph
- `validated_connectivity` — labels exist (e.g. Allen SynPhys)
- `functional_predictive_dependency` — Neuropixels / generic extracellular
- `invalid_or_leaky` — graph constructed with leakage; excluded from claims
