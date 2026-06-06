"""Typer CLI: generate / run / score / analyze.

Offline-safe by default: ``run-slice`` uses the mock provider (oracle solver)
unless config points the cheap model at a real provider with an API key in env.
``estimate`` never runs anything — it gates spend via ``budget.guard``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import get_config
from .dsl.ops_spec import render_op_table_markdown
from .runner import budget, grid, run_cell

app = typer.Typer(add_completion=False, help="Program Reconstruction Scaling Study harness")


@app.command("dump-ops")
def dump_ops(out: str = "docs/op_semantics.md") -> None:
    """Regenerate the op semantics table (the CHECKPOINT 1 review doc)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render_op_table_markdown())
    typer.echo(f"wrote {out}")


@app.command("generate")
def generate(
    sample: bool = typer.Option(False, "--sample", help="Write the CHECKPOINT 3 sample of 5 transforms."),
    count: int = 5,
    seed: int = 100,
    out: str = "manifests/samples/checkpoint3_corpusB_sample.md",
) -> None:
    """Generate a SMALL sample of Corpus-B transforms (never the full corpus)."""
    from .corpus import complexity, generator, identifiability

    cfg = get_config()
    if not sample:
        typer.echo("Pass --sample to write the review sample (full-corpus generation is gated at CP7).")
        raise typer.Exit(0)
    lines = [f"# Generated sample of {count} Corpus-B transforms", ""]
    made, s = 0, seed
    while made < count:
        persona = cfg.generators[made % len(cfg.generators)]
        try:
            t = generator.sample_transform(persona, n=3, m=1, target_depth=3, seed=s, transform_id=f"B-SAMPLE-{made:02d}")
        except Exception:
            s += 1
            continue
        rep = identifiability.check(t, n_probes=300)
        lines += [
            f"## {t.id} ({persona.name})  C={complexity.transform_complexity(t):.1f} "
            f"band={complexity.transform_band(t)} identified={rep.passed} {rep.flags}",
            "```python", t.source.rstrip(), "```", "",
        ]
        made += 1
        s += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines))
    typer.echo(f"wrote {out} ({made} transforms)")


@app.command("build-corpus")
def build_corpus_cmd(
    n_b: int = 50, seed: int = 0,
    write: bool = typer.Option(True, "--write/--no-write", help="Persist JSONL + summary."),
) -> None:
    """Assemble 50 Corpus B + 50 Corpus A + 50 twins (offline; no model calls)."""
    from .corpus import build_corpus

    cfg = get_config()
    man = build_corpus.build_full_corpus(n_b=n_b, seed=seed, cfg=cfg)
    s = man.summary()
    typer.echo(
        f"corpus_a={s['corpus_a']} corpus_b={s['corpus_b']} twins={s['twins']} "
        f"| quarantined: A={s['quarantined_a']} B-rejected={s['quarantined_b']} "
        f"twins={s['quarantined_twins']}"
    )
    if write:
        paths = build_corpus.write_corpus(man, cfg=cfg)
        for k, v in paths.items():
            typer.echo(f"  {k}: {v}")


@app.command("run-slice")
def run_slice(
    n: int = 3, m: int = 1, k: int = 100, band: str = "med", noise: str = "none", seed: int = 0,
) -> None:
    """Run ONE cheap cell end-to-end (the vertical slice). Offline by default."""
    cfg = get_config()
    cell = grid.single_cheap_cell(cfg, n=n, m=m, k=k, band=band, noise=noise)
    result = run_cell.run_cell(cell, seed=seed, cfg=cfg)
    typer.echo(result.one_line)
    typer.echo(f"transcript: {result.transcript_path}")


@app.command("estimate")
def estimate(
    full: bool = typer.Option(False, "--full", help="Estimate the full grid cross product (large)."),
    repeats: int = 3,
    i_accept_cost: bool = typer.Option(False, "--i-accept-cost", help="Acknowledge over-ceiling spend."),
) -> None:
    """Estimate spend and gate it against the ceiling. Never runs anything."""
    cfg = get_config()
    cells = grid.enumerate_cells(cfg) if full else [
        grid.single_cheap_cell(cfg, k=k) for k in cfg.grid.k
    ]
    report = budget.estimate(cells, repeats=repeats, cfg=cfg)
    typer.echo(report.render())
    try:
        budget.guard(report, accept_cost=i_accept_cost)
        typer.echo("OK: within ceiling (or cost explicitly accepted).")
    except budget.BudgetRefused as e:
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from None


@app.command("analyze")
def analyze(manifests_dir: str = "manifests") -> None:
    """Load run manifests into a tidy frame and print the cell-level table."""
    from .analysis import aggregate

    outcomes = aggregate.load_manifest_dir(manifests_dir)
    if not outcomes:
        typer.echo(f"no run manifests found under {manifests_dir}/")
        raise typer.Exit(0)
    df = aggregate.outcomes_to_frame(outcomes)
    typer.echo(f"{len(outcomes)} runs")
    typer.echo(aggregate.cell_table(df).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()
