"""Claim gating.

Based on a run's config, leakage report, identifiability report, capacity report,
and the dataset type, decide which scientific claims are ALLOWED and which are
DISALLOWED. Writes ``claim_gate.json`` and ``claim_gate.md``.

The gate is a conservative whitelist: a claim is allowed only when explicit
evidence is found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..constants import NEGATIVE_RESULTS_STATEMENT
from ..utils.io import read_json, write_json


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def compute_claim_gate(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    cfg = _safe_read_json(run / "config.json")
    leakage = _safe_read_json(run / "leakage_report.json")
    ident = _safe_read_json(run / "identifiability_report.json")
    cap = _safe_read_json(run / "capacity_report.json")
    interp = _safe_read_json(run / "interpretability_report.json")
    graph_recovery = _safe_read_json(run / "graph_recovery.json")
    metrics = _safe_read_json(run / "predictive_metrics_summary.json")
    ablations = _safe_read_json(run / "ablation_summary.json")

    allowed: list[dict[str, Any]] = []
    disallowed: list[dict[str, Any]] = []
    run_invalid_reasons: list[str] = []

    data_cfg = cfg.get("data", {})
    dataset_kind = data_cfg.get("kind", "synthetic")
    split_mode = data_cfg.get("split_mode", "time_block")
    model_name = cfg.get("model", {}).get("name", "")
    embedding_uses = cfg.get("model", {}).get("uses", {})

    # ---- Hard invalidators ----
    if leakage.get("status") == "FAIL":
        run_invalid_reasons.append("Leakage check failed.")
    if cfg.get("selection_split", "val").lower() == "test":
        run_invalid_reasons.append("Checkpoint was selected by a test-split metric.")
    if data_cfg.get("split_mode") == "window_random" and not data_cfg.get("allow_leaky_split", False):
        run_invalid_reasons.append("window_random split without allow_leaky_split=true.")
    if any(embedding_uses.values()) and not (
        cfg.get("allow_cell_id_embedding")
        or cfg.get("allow_pair_id_embedding")
        or cfg.get("allow_session_id_embedding")
    ):
        run_invalid_reasons.append("Identity embeddings used without explicit allow flags.")

    if run_invalid_reasons:
        gate = {
            "status": "INVALID",
            "reasons": run_invalid_reasons,
            "allowed": [],
            "disallowed": [
                {
                    "claim": "Any scientific claim from this run",
                    "reason": "; ".join(run_invalid_reasons),
                    "how_to_make_valid": "Fix the listed conditions and re-run.",
                }
            ],
            "negative_results_statement": NEGATIVE_RESULTS_STATEMENT,
        }
        write_json(run / "claim_gate.json", gate)
        _write_md(run, gate)
        return gate

    # ---- Predictive claims ----
    # If we have predictive metrics and they exist on val and test:
    if metrics:
        if "val" in metrics:
            allowed.append(
                {
                    "claim": "Predictive performance reported on the validation split.",
                    "evidence": "predictive_metrics_summary.json::val",
                    "caveats": (
                        "Generalization scope depends on the split. Default synthetic split "
                        "is time-block: claims apply to held-out time blocks of the same circuit."
                    ),
                }
            )
        if "test" in metrics:
            allowed.append(
                {
                    "claim": "Predictive performance reported on the held-out test split.",
                    "evidence": "predictive_metrics_summary.json::test",
                    "caveats": (
                        "Test was not used for any model/checkpoint/normalization/probe selection."
                    ),
                }
            )

    # ---- Generalization scope ----
    if split_mode == "time_block":
        allowed.append(
            {
                "claim": "Generalization claim: held-out future time blocks within the same circuit.",
                "evidence": f"split_mode={split_mode}",
                "caveats": "Does NOT cover held-out cells or held-out protocols on real data.",
            }
        )
        disallowed.append(
            {
                "claim": "Generalizes across cells.",
                "reason": "Held-out-cell split was not run.",
                "how_to_make_valid": "Use cell_protocol_split mode heldout_cell_seen_protocol_family.",
            }
        )

    # ---- Mechanistic / identifiability claims ----
    if ident:
        by_label = ident.get("by_label", {})
        bio_params = by_label.get("biological_parameter_supported", [])
        for p in bio_params:
            allowed.append(
                {
                    "claim": f"Latent '{p}' is consistent with synthetic ground truth (predictive_latent → biological_parameter_supported).",
                    "evidence": "identifiability_report.json",
                    "caveats": (
                        "This claim is supported by SYNTHETIC ground truth. It does not by "
                        "itself license a real-biology claim."
                    ),
                }
            )
        for p in by_label.get("predictive_latent_only", []):
            disallowed.append(
                {
                    "claim": f"Latent '{p}' is biologically meaningful.",
                    "reason": "Recovery correlation below threshold; latent is predictive only.",
                    "how_to_make_valid": "Run additional perturbation tests and/or compare to independent ground truth.",
                }
            )
        for p in by_label.get("not_recoverable", []):
            disallowed.append(
                {
                    "claim": f"Latent '{p}' recovers a biological parameter.",
                    "reason": "Recovery test produced NaN or n<2.",
                    "how_to_make_valid": "Run more seeds and ensure parameter is observable in the data.",
                }
            )

    # ---- Graph claims ----
    learned_or_given_graph = cfg.get("model", {}).get("graph_mode", "none")
    if dataset_kind == "synthetic" and learned_or_given_graph == "learned":
        if graph_recovery:
            auroc = graph_recovery.get("edge_auroc", float("nan"))
            if auroc > 0.6:
                allowed.append(
                    {
                        "claim": (
                            f"Learned graph recovers synthetic directed edges above chance "
                            f"(edge AUROC={auroc:.3f})."
                        ),
                        "evidence": "graph_recovery.json",
                        "caveats": (
                            "Recovery shown on synthetic ground truth. Cannot be interpreted as "
                            "recovering biological connectivity on real extracellular data."
                        ),
                    }
                )
            else:
                disallowed.append(
                    {
                        "claim": "Learned graph recovers synthetic connectivity.",
                        "reason": f"edge_auroc={auroc:.3f} <= 0.6 (chance/marginal).",
                        "how_to_make_valid": "Train longer, increase data, or improve graph parameterization.",
                    }
                )
        else:
            disallowed.append(
                {
                    "claim": "Learned graph recovers connectivity.",
                    "reason": "No graph_recovery.json found.",
                    "how_to_make_valid": "Run scripts/evaluate.py with --graph-recovery.",
                }
            )

    if dataset_kind != "synthetic":
        disallowed.append(
            {
                "claim": "Learned graph recovers anatomical/synaptic connectivity.",
                "reason": (
                    "Without synthetic or validated anatomical ground truth, learned edges "
                    "are functional/predictive dependencies, not synapses."
                ),
                "how_to_make_valid": (
                    "Validate against an independent connectivity dataset (e.g. Allen SynPhys)."
                ),
            }
        )

    # ---- Interpretability ----
    if interp:
        rows = interp if isinstance(interp, list) else interp.get("rows", [])
        for r in rows:
            real = r.get("real_val", float("nan"))
            perm = r.get("permuted_latents_val", float("nan"))
            rand = r.get("random_latents_val", float("nan"))
            shuf = r.get("shuffled_labels_val", float("nan"))
            try:
                ok = real > max(perm, rand, shuf) + 0.05
            except TypeError:
                ok = False
            if ok:
                allowed.append(
                    {
                        "claim": f"Probe '{r['probe']}' beats permuted/random/shuffled controls.",
                        "evidence": "interpretability_controls.csv",
                        "caveats": "Held-out probe split; not used for predictive checkpoint selection.",
                    }
                )
            else:
                disallowed.append(
                    {
                        "claim": f"Latents independently predict probe '{r['probe']}'.",
                        "reason": "Real probe did not meaningfully exceed controls.",
                        "how_to_make_valid": "Increase training data or change the latent representation.",
                    }
                )

    # ---- Capacity ----
    if cap:
        allowed.append(
            {
                "claim": (
                    f"Capacity reported: {cap.get('n_trainable_params', '?')} trainable params, "
                    f"{cap.get('wall_clock_s', '?'):.1f}s wall."
                ),
                "evidence": "capacity_report.json",
                "caveats": (
                    "Mechanistic-vs-baseline superiority requires a parameter-matched comparison. "
                    "Marked capacity_controlled=false by default."
                ),
            }
        )

    # ---- Dataset-derived limits ----
    if dataset_kind == "synthetic":
        disallowed.append(
            {
                "claim": "Validated on public biological recordings.",
                "reason": "This run trained on synthetic data only.",
                "how_to_make_valid": "Run scripts/prepare_dataset.py and scripts/train.py against an Allen/DANDI source.",
            }
        )

    # Ablation-derived claim (graph helps or not).
    if ablations:
        rows = ablations.get("rows", [])
        full = next((r for r in rows if r.get("name") == "full"), None)
        no_graph = next((r for r in rows if r.get("name") == "no_graph"), None)
        if full and no_graph and "val_metric" in full and "val_metric" in no_graph:
            if full["val_metric"] < no_graph["val_metric"]:
                allowed.append(
                    {
                        "claim": "Graph component improves predictive forecasting in ablation.",
                        "evidence": "ablation_summary.json",
                        "caveats": "Comparison on validation split. Effect size and CI should be inspected.",
                    }
                )
            else:
                disallowed.append(
                    {
                        "claim": "Graph component improves predictive forecasting.",
                        "reason": "Full model did not beat no-graph ablation on val_metric.",
                        "how_to_make_valid": "Train longer or improve graph parameterization.",
                    }
                )

    gate = {
        "status": "VALID",
        "reasons": [],
        "allowed": allowed,
        "disallowed": disallowed,
        "model_name": model_name,
        "dataset_kind": dataset_kind,
        "split_mode": split_mode,
        "negative_results_statement": NEGATIVE_RESULTS_STATEMENT,
    }
    write_json(run / "claim_gate.json", gate)
    _write_md(run, gate)
    return gate


def _write_md(run_dir: Path, gate: dict[str, Any]) -> None:
    md: list[str] = ["# Claim gate", ""]
    md.append(f"**Status:** `{gate['status']}`")
    if gate["status"] == "INVALID":
        md.append("")
        md.append("Reasons:")
        for r in gate["reasons"]:
            md.append(f"- {r}")
    md.append("")
    md.append("## Allowed claims")
    if not gate["allowed"]:
        md.append("_None._")
    for entry in gate["allowed"]:
        md.append(f"- **{entry['claim']}**")
        md.append(f"  - evidence: `{entry['evidence']}`")
        md.append(f"  - caveats: {entry['caveats']}")
    md.append("")
    md.append("## Disallowed claims")
    if not gate["disallowed"]:
        md.append("_None._")
    for entry in gate["disallowed"]:
        md.append(f"- **{entry['claim']}**")
        md.append(f"  - reason: {entry['reason']}")
        md.append(f"  - how to make valid: {entry['how_to_make_valid']}")
    md.append("")
    md.append("---")
    md.append("")
    md.append(gate.get("negative_results_statement", NEGATIVE_RESULTS_STATEMENT))
    (run_dir / "claim_gate.md").write_text("\n".join(md))
