"""Non-circular post-hoc interpretability protocol.

The predictive model is frozen. Linear probes are trained ONLY on the train split.
Latent labels (e.g. cell type, tau_m) are never given to the predictive model
during training. Controls:

- random latent: probes with a permutation of latent dimensions
- shuffled labels: probes against label-shuffled targets
- random readout: random feature projection

The result CSV reports probe accuracy / R^2 alongside controls so the reader can
judge whether the latent independently predicts the property.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ..utils.io import write_json
from ..utils.metrics import pearson_corr
from ..utils.seed import seeded_rng


@dataclass
class ProbeSpec:
    name: str
    latents: np.ndarray  # [N, D]
    labels: np.ndarray  # [N]
    task: str  # 'regression' or 'classification'


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    ss_tot = ((target - target.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = ((target - pred) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def _acc(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def _train_eval_probe(
    latents: np.ndarray,
    labels: np.ndarray,
    task: str,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    rng = seeded_rng(seed)
    if task == "regression":
        model = LinearRegression()
        model.fit(latents[train_mask], labels[train_mask])
        pred = model.predict(latents[val_mask])
        return {
            "metric_train": _r2(model.predict(latents[train_mask]), labels[train_mask]),
            "metric_val": _r2(pred, labels[val_mask]),
            "corr_val": pearson_corr(pred, labels[val_mask]),
            "metric_type": "r2",
        }
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)
        if n_classes < 2:
            return {"metric_train": float("nan"), "metric_val": float("nan"), "metric_type": "acc"}
        # If only 1 class in train, can't fit.
        if len(np.unique(y[train_mask])) < 2:
            return {"metric_train": float("nan"), "metric_val": float("nan"), "metric_type": "acc"}
        model = LogisticRegression(max_iter=500)
        model.fit(latents[train_mask], y[train_mask])
        pred_tr = model.predict(latents[train_mask])
        pred_va = model.predict(latents[val_mask])
        return {
            "metric_train": _acc(pred_tr, y[train_mask]),
            "metric_val": _acc(pred_va, y[val_mask]),
            "metric_type": "acc",
        }
    raise ValueError(f"Unknown task: {task}")


def run_interpretability_probes(
    specs: list[ProbeSpec],
    *,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = seeded_rng(seed)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        n = spec.latents.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(round(train_fraction * n))
        tr_mask = np.zeros(n, dtype=bool)
        tr_mask[idx[:cut]] = True
        va_mask = ~tr_mask

        real = _train_eval_probe(spec.latents, spec.labels, spec.task, tr_mask, va_mask, seed=seed)

        # Permutation control: shuffle latents.
        permuted_latents = spec.latents.copy()
        rng2 = seeded_rng(seed + 7)
        for j in range(permuted_latents.shape[1]):
            rng2.shuffle(permuted_latents[:, j])
        perm = _train_eval_probe(permuted_latents, spec.labels, spec.task, tr_mask, va_mask, seed=seed)

        # Random readout: random Gaussian features same shape.
        rng3 = seeded_rng(seed + 11)
        rand_lat = rng3.normal(size=spec.latents.shape)
        rand = _train_eval_probe(rand_lat, spec.labels, spec.task, tr_mask, va_mask, seed=seed)

        # Shuffled labels.
        rng4 = seeded_rng(seed + 13)
        shuf_labels = spec.labels.copy()
        rng4.shuffle(shuf_labels)
        shuf = _train_eval_probe(spec.latents, shuf_labels, spec.task, tr_mask, va_mask, seed=seed)

        rows.append(
            {
                "probe": spec.name,
                "task": spec.task,
                "metric_type": real["metric_type"],
                "real_val": real["metric_val"],
                "permuted_latents_val": perm["metric_val"],
                "random_latents_val": rand["metric_val"],
                "shuffled_labels_val": shuf["metric_val"],
                "real_train": real["metric_train"],
                "n_samples": int(n),
                "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum()),
            }
        )
    return rows


def write_interpretability_report(
    run_dir: str | Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "interpretability_controls.csv", index=False)
    write_json(run / "interpretability_report.json", rows)
    md = [
        "# Interpretability report",
        "",
        "Latent probes were trained on the training split only and evaluated on a "
        "held-out portion. Each probe is reported alongside three independent "
        "negative controls: permuted latents, random latents, and shuffled labels.",
        "",
    ]
    for r in rows:
        md.append(
            f"- **{r['probe']}** ({r['metric_type']}): real={r['real_val']:.3f} | "
            f"perm={r['permuted_latents_val']:.3f} | random={r['random_latents_val']:.3f} | "
            f"shuf={r['shuffled_labels_val']:.3f}"
        )
    md.append("")
    md.append(
        "Interpretability is supported only when the real probe meaningfully exceeds all "
        "three controls."
    )
    (run / "interpretability_report.md").write_text("\n".join(md))
    return {"rows": rows}
