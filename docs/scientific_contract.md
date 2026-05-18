# Scientific contract

UniMarket separates four ideas. Any claim made by a run must be justified by the matching kind of evidence; the claim gate (`src/unimarket/eval/claim_gating.py`) enforces this automatically.

## 1. Prediction
Future voltage, spike probability, synaptic response, or population spike-count forecasting on a held-out split.

**Evidence required:** `predictive_metrics_summary.json` with a non-empty `val` and/or `test` entry, computed without leakage.

## 2. Mechanism
Constrained dynamical variables inside the model — `tau_m`, `R_m`, `threshold`, adaptation parameters, synaptic delay, decay, weights, graph edges. These are model-internal.

**Evidence required:** the model is one of the mechanistic variants (`lif_residual`, `adaptive_lif`, `synapse_response`, `graph_latent_dynamics`) and the constrained parameters are exposed via `extract_constrained_params`.

## 3. Interpretation
Post-hoc comparison of frozen learned latents to *independent* labels: synthetic ground truth, held-out metadata, or other electrophysiology features that were **never** part of the predictive loss.

**Evidence required:** an `identifiability_report.json` and/or `interpretability_controls.csv` where the real result beats permuted/random/shuffled-label controls; checkpoint was selected by `val` only (never by interpretability metric).

## 4. Claim gating
Automatic decision about which claims are allowed for a given run. Implemented as a conservative whitelist; any claim not explicitly supported by evidence is moved to the `disallowed` list with a reason.

A model may be predictively useful (Prediction passes) while its latents are not biologically interpretable (Interpretation fails). UniMarket reports both.
