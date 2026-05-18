# Identifiability

A learned latent parameter is reportable as biologically meaningful **only** when it passes all of the following on synthetic data:

1. Synthetic recovery under matched noise and sampling.
2. Synthetic recovery under misspecified noise.
3. Monotonic perturbation: changing the true parameter changes the recovered latent in the correct direction (rank correlation ≥ threshold).
4. Residual-capacity ablation: the parameter does not disappear or invert as the residual capacity scales.
5. Real-data post-hoc association with an independent held-out electrophysiology feature or metadata field never used in predictive training.

If any of these fail, the parameter is labeled `predictive_latent_only`, not `biological_parameter_supported`.

## Labels

The pipeline labels every parameter as exactly one of:

- `biological_parameter_supported`
- `predictive_latent_only`
- `not_recoverable`
- `not_tested`

The decision lives in `src/unimarket/eval/identifiability.py` and the labels are persisted in `synthetic_parameter_recovery.csv` and `identifiability_report.{json,md}`.

## Why a single correlation isn't enough

A high `corr(learned, true)` can arise from confounded model capacity. The full identifiability protocol (perturbation + residual ablation + independent association) is what distinguishes a constrained parameter from a coincidental fit. The smoke pipeline tests step (1) by default; steps (2)–(5) are exposed as additional perturbation runs and recovery-vs-residual-scale sweeps.
