"""Project-wide constants."""

from __future__ import annotations

# Split strength labels used in reports and claim gating.
SPLIT_STRENGTH = {
    "debug_only": "Random window splits; cannot support generalization claims.",
    "weak_within_cell": "Sweep-level split within the same cells.",
    "protocol_generalization": "Held-out protocol families within seen cells.",
    "cell_generalization": "Held-out cells with seen protocol families.",
    "cell_and_protocol_generalization": "Held-out cells AND held-out protocols.",
}

# Graph claim taxonomy.
GRAPH_CLAIM_TYPES = {
    "synthetic_connectivity",
    "validated_connectivity",
    "functional_predictive_dependency",
    "invalid_or_leaky",
}

# Identifiability labels.
IDENTIFIABILITY_LABELS = {
    "biological_parameter_supported",
    "predictive_latent_only",
    "not_recoverable",
    "not_tested",
}

# Adapter status taxonomy.
ADAPTER_STATUS = {
    "fully_functional",
    "functional_with_local_nwb",
    "schema_validated_only",
    "planned",
}

# Default numerical epsilon for log/log-likelihoods.
EPS = 1e-8

# Negative-result language required in reports.
NEGATIVE_RESULTS_STATEMENT = (
    "Negative results are valid outputs of UniMarket. A failed recovery or failed "
    "interpretability test constrains the scientific claim rather than invalidating the software."
)
