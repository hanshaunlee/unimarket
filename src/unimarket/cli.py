"""UniMarket CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from .data.synthetic import SyntheticConfig, write_synthetic
from .eval.evaluate import evaluate_run
from .eval.leakage import run_leakage_checks
from .eval.report_cards import write_report_card
from .train.train import train_from_config
from .utils.config import load_yaml
from .utils.paths import runs_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("make-synthetic")
def make_synthetic(
    config: Path = typer.Option(..., "--config", "-c", help="Path to data/synthetic.yaml"),
    out: Path = typer.Option(..., "--out", "-o", help="Output processed directory"),
) -> None:
    """Generate synthetic LIF circuit data."""
    cfg = load_yaml(config)
    sc_kwargs = dict(cfg.get("synthetic", {}))
    sc = SyntheticConfig(**sc_kwargs)
    info = write_synthetic(out, sc)
    typer.echo(f"Generated synthetic data at {info['out_dir']} (N={info['N']}, T={info['T']}, edges={info['edges']})")


@app.command("scan-nwb")
def scan_nwb(
    path: Path = typer.Argument(..., help="Path to a local .nwb file"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
) -> None:
    """Scan an NWB file and write a normalized manifest plus arrays."""
    from .data.nwb_adapter import scan_nwb as _scan_nwb

    summary = _scan_nwb(path, out)
    typer.echo(f"Scanned {path} -> {out}. Acquisitions: {len(summary['acquisitions'])}; units: {len(summary['units'])}")


@app.command("prepare")
def prepare(
    config: Path = typer.Option(..., "--config", "-c", help="Path to data config"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
) -> None:
    """Prepare a dataset from its data config (synthetic supported out-of-the-box)."""
    cfg = load_yaml(config)
    kind = cfg.get("kind", "synthetic")
    if kind == "synthetic":
        sc_kwargs = dict(cfg.get("synthetic", {}))
        sc = SyntheticConfig(**sc_kwargs)
        info = write_synthetic(out, sc)
        typer.echo(f"Prepared synthetic dataset at {info['out_dir']}")
    elif kind == "nwb_generic":
        from .data.nwb_adapter import scan_nwb as _scan_nwb

        src = cfg["source"]
        summary = _scan_nwb(src, out)
        typer.echo(f"NWB summary written to {out} (acquisitions: {len(summary['acquisitions'])})")
    elif kind == "allen_cell_types":
        from .data.allen_cell_types import ingest_directory

        info = ingest_directory(cfg["source_dir"], out, dataset_name=cfg.get("dataset_name", "allen_cell_types"))
        typer.echo(f"Allen cell types ingested: {info}")
    elif kind == "allen_synphys":
        from .data.allen_synphys import ingest_from_csv

        info = ingest_from_csv(cfg["source_csv"], out)
        typer.echo(f"Allen SynPhys ingested: {info}")
    elif kind == "allen_neuropixels":
        from .data.allen_neuropixels import ingest_spike_table

        info = ingest_spike_table(
            cfg["spikes_path"],
            out,
            bin_size_ms=float(cfg.get("bin_size_ms", 10.0)),
            stimulus_table_path=cfg.get("stimulus_table_path"),
            allow_stimulus_identity=bool(cfg.get("allow_stimulus_identity", False)),
        )
        typer.echo(f"Allen Neuropixels ingested: {info}")
    else:
        typer.echo(f"Unknown data kind: {kind}", err=True)
        raise typer.Exit(code=2)


@app.command("train")
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
) -> None:
    """Train a model from a YAML experiment config."""
    run_dir = train_from_config(config)
    typer.echo(f"Training complete. Run dir: {run_dir}")


@app.command("evaluate")
def evaluate(
    run_dir: Path = typer.Option(..., "--run-dir", "-r", help="Path to run directory"),
    split: str = typer.Option("test", "--split", "-s"),
) -> None:
    """Evaluate a trained run on a split."""
    summary = evaluate_run(run_dir, splits=[split] if split != "all" else ["train", "val", "test"])
    write_report_card(run_dir)
    typer.echo(f"Evaluation complete. Claim gate: {summary['claim_gate']['status']}")


@app.command("leakage-check")
def leakage_check(
    run_dir: Path = typer.Option(..., "--run-dir", "-r", help="Path to run directory"),
) -> None:
    """Run leakage / circularity checks."""
    report = run_leakage_checks(run_dir)
    typer.echo(f"Leakage check: {report['status']}")
    for c in report["checks"]:
        typer.echo(f"  {c['status']}: {c['name']}")


@app.command("plot")
def plot(
    run_dir: Path = typer.Option(..., "--run-dir", "-r", help="Path to run directory"),
) -> None:
    """Regenerate report card markdown for a run."""
    p = write_report_card(run_dir)
    typer.echo(f"Wrote {p}")


@app.command("runs")
def runs() -> None:
    """List run directories."""
    rd = runs_dir()
    for sub in sorted(rd.iterdir()):
        if sub.is_dir():
            typer.echo(sub.name)


if __name__ == "__main__":
    app()
