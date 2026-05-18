"""Parameter identifiability and recovery analysis.

For a trained model with extractable constrained parameters and known synthetic
ground truth, this module:

1. Computes per-parameter correlation between learned and true values.
2. Runs perturbation monotonicity checks by perturbing simulated parameters.
3. Labels each parameter as one of:
   - biological_parameter_supported
   - predictive_latent_only
   - not_recoverable
   - not_tested
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import IDENTIFIABILITY_LABELS
from ..utils.io import write_json
from ..utils.metrics import pearson_corr


def evaluate_parameter_recovery(
    learned: dict[str, np.ndarray],
    true: dict[str, np.ndarray],
    *,
    correlation_threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Per-parameter correlation between learned latent vectors and true ground truth.

    ``learned`` and ``true`` map parameter name -> array of length N.
    Parameters present in only one side are still reported (as not_tested).
    """
    rows: list[dict[str, Any]] = []
    keys = set(learned) | set(true)
    for k in sorted(keys):
        if k not in learned or k not in true:
            rows.append(
                {
                    "parameter": k,
                    "correlation": float("nan"),
                    "label": "not_tested",
                    "n": int(len(true.get(k, learned.get(k, np.array([]))))),
                    "note": "Missing in either learned or true.",
                }
            )
            continue
        a = np.asarray(learned[k]).ravel()
        b = np.asarray(true[k]).ravel()
        n = min(a.size, b.size)
        if n < 2:
            rows.append(
                {"parameter": k, "correlation": float("nan"), "label": "not_recoverable", "n": n}
            )
            continue
        a = a[:n]
        b = b[:n]
        # Filter NaNs
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            rows.append(
                {"parameter": k, "correlation": float("nan"), "label": "not_recoverable", "n": int(mask.sum())}
            )
            continue
        corr = pearson_corr(a[mask], b[mask])
        if np.isnan(corr):
            label = "not_recoverable"
        elif corr >= correlation_threshold:
            label = "biological_parameter_supported"
        else:
            label = "predictive_latent_only"
        rows.append(
            {
                "parameter": k,
                "correlation": float(corr),
                "label": label,
                "n": int(mask.sum()),
            }
        )
    return rows


def perturbation_monotonicity(
    perturb_results: list[tuple[float, float]], correlation_threshold: float = 0.4
) -> dict[str, Any]:
    """Given (true_value, learned_value) pairs across perturbations, test monotonicity.

    Returns a dict with rank correlation and a boolean ``monotonic``.
    """
    if len(perturb_results) < 3:
        return {"rank_correlation": float("nan"), "monotonic": False, "n": len(perturb_results)}
    truth = np.array([t for t, _ in perturb_results])
    learned = np.array([l for _, l in perturb_results])
    # Spearman rank correlation: correlate ranks.
    truth_rank = np.argsort(np.argsort(truth))
    learned_rank = np.argsort(np.argsort(learned))
    rho = pearson_corr(truth_rank.astype(float), learned_rank.astype(float))
    return {
        "rank_correlation": float(rho),
        "monotonic": bool(not np.isnan(rho) and rho >= correlation_threshold),
        "n": len(perturb_results),
    }


def write_identifiability_report(
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run / "synthetic_parameter_recovery.csv", index=False)
    summary = {
        "rows": rows,
        "by_label": {
            label: [r["parameter"] for r in rows if r["label"] == label]
            for label in IDENTIFIABILITY_LABELS
        },
        "extra": extra or {},
    }
    write_json(run / "identifiability_report.json", summary)
    md = ["# Identifiability report", ""]
    md.append("Each learned parameter is compared post hoc to synthetic ground truth.")
    md.append("")
    for r in rows:
        md.append(
            f"- **{r['parameter']}**: correlation={r['correlation']:.3f} → `{r['label']}` (n={r['n']})"
        )
    md.append("")
    md.append("Labels:")
    for label in sorted(IDENTIFIABILITY_LABELS):
        md.append(f"  - `{label}`")
    (run / "identifiability_report.md").write_text("\n".join(md))
    return summary
