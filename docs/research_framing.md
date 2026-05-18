# Research framing

The novelty of UniMarket is not "we used ML on neural data." The combination below is the contribution:

1. **Mechanistically constrained predictive dynamics.** Constrained LIF / adaptive-LIF / synapse-response cores with small, bounded residuals (`residual_scale` is regularized) so that the constrained dynamical variables remain identifiable.
2. **Graph-temporal circuit modeling.** Edge-gated message passing without PyTorch Geometric, evaluated as a *functional* graph by default and as a *connectivity* graph only when synthetic or anatomical ground truth exists.
3. **Cross-dataset adapter design.** Schema-normalized adapters for single-cell, synapse, and population data — synthetic, NWB, Allen Cell Types, Allen SynPhys, Allen Neuropixels.
4. **Strict non-circular validation.** Hard-fail checks for the leakiest configurations: window-random splits on time series, test-split selection, identity embeddings under held-out splits, normalization computed on test.
5. **Synthetic recovery tests with known ground truth.** Parameter recovery and graph recovery against truth, with `not_recoverable` / `predictive_latent_only` / `biological_parameter_supported` labels and perturbation-monotonicity utilities.
6. **Post-hoc independent interpretability.** Frozen-model linear probes trained only on the train split and reported alongside permuted-latent, random-latent, and shuffled-label controls.

UniMarket is positioned as **research infrastructure + benchmark + initial model**, not as a finished brain simulator. Every claim it generates is conservative, falsifiable, and tied to evidence the run actually produced.
