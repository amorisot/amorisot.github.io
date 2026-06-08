"""Typer CLI: generate / run / score / analyze.

Offline-safe by default: ``run-slice`` uses the mock provider (oracle solver)
unless config points the cheap model at a real provider with an API key in env.
``estimate`` never runs anything — it gates spend via ``budget.guard``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import get_api_key, get_config
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
        f"| quarantined_a={s['quarantined_a']} b_rejected={s['quarantined_b']} "
        f"mask_only_twins={s['mask_only_twins']}"
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


@app.command("sweep")
def sweep_cmd(
    name: str = typer.Option("sample", help="sample | dimensionality | semantic | leakage"),
    repeats: int = 1,
    n_transforms: int = 5,
    run: bool = typer.Option(False, "--run", help="Execute the plan (offline oracle by default)."),
    i_accept_cost: bool = typer.Option(False, "--i-accept-cost", help="Acknowledge over-ceiling spend."),
) -> None:
    """Plan a targeted sweep, estimate its cost, and (optionally) run it resumably."""
    from .analysis import aggregate
    from .corpus import build_corpus
    from .runner import sweep

    cfg = get_config()
    loaded = build_corpus.load_persisted(cfg)
    if loaded is None:
        typer.echo("(no persisted corpus found — building it; run `progrecon build-corpus` to cache)")
        man = build_corpus.build_full_corpus(cfg=cfg)
        corpus_a, corpus_b, twins = man.corpus_a, man.corpus_b, man.twins
    else:
        corpus_a, corpus_b, twins = loaded

    mid = cfg.models.cheap.id
    if name == "sample":
        items = sweep.plan_sample_sweep(corpus_b[:n_transforms], "B", ks=cfg.grid.k, repeats=repeats, model_id=mid)
    elif name == "dimensionality":
        items = sweep.plan_dimensionality_sweep(corpus_b[:n_transforms], "B", distractor_levels=[0, 3, 10, 30], k=200, repeats=repeats, model_id=mid)
    elif name == "semantic":
        by_id = {a.id: a for a in corpus_a}
        pairs = [(by_id[t.twin_id], t) for t in twins if t.twin_id in by_id][:n_transforms]
        items = sweep.plan_semantic_sweep(pairs, k=200, repeats=repeats, model_id=mid)
    elif name == "leakage":
        items = sweep.plan_leakage_sweep(corpus_b[:n_transforms], "B", query_budgets=[0, 1, 3], k=200, repeats=repeats, model_id=mid)
    else:
        typer.echo(f"unknown sweep {name!r} (sample | dimensionality | semantic | leakage)")
        raise typer.Exit(2)

    report = sweep.estimate(items, cfg=cfg)
    typer.echo(f"sweep '{name}': {len(items)} runs planned")
    typer.echo(report.render())
    if not run:
        typer.echo("(estimate only — pass --run to execute; offline oracle unless a real provider is configured)")
        raise typer.Exit(0)
    budget.guard(report, accept_cost=i_accept_cost)
    outcomes = sweep.run_sweep(items, cfg=cfg)
    df = aggregate.outcomes_to_frame(outcomes)
    typer.echo(f"{len(outcomes)} runs complete")
    typer.echo(aggregate.cell_table(df).to_string(index=False))


@app.command("experiment")
def experiment(
    ids: str = typer.Option("A-INFO-01,A-RETAIL-01,A-HEALTH-05", help="Comma-separated Corpus-A ids."),
    twins: bool = typer.Option(True, "--twins/--no-twins", help="Also run the matched abstract twins."),
    model: str = typer.Option("", help="Model id (default: config cheap model)."),
    provider: str = typer.Option("anthropic", help="anthropic | mock (mock = offline oracle)."),
    k: int = 128,
    repeats: int = 3,
    test_size: int = 300,
    run: bool = typer.Option(False, "--run", help="Execute (otherwise estimate only)."),
    i_accept_cost: bool = typer.Option(False, "--i-accept-cost"),
) -> None:
    """Run a small curated experiment: chosen semantic programs vs their twins."""
    from .analysis import aggregate
    from .corpus import build_corpus
    from .runner import sweep
    from .runner.model_client import ModelClient

    cfg = get_config()
    model = model or cfg.models.cheap.id
    cfg = cfg.model_copy(update={
        "models": cfg.models.model_copy(update={
            "cheap": cfg.models.cheap.model_copy(update={"id": model, "provider": provider})}),
        "scoring": cfg.scoring.model_copy(update={"test_id_size": test_size, "test_ood_size": test_size}),
    })

    want = [s.strip() for s in ids.split(",") if s.strip()]
    typer.echo(f"preparing {len(want)} program(s){' + twins' if twins else ''}...")
    loaded = build_corpus.load_persisted(cfg)
    if loaded is not None:
        corpus_a, _, twins_all = loaded
    else:
        # Build ONLY what this experiment needs (fast) — not the whole corpus.
        from .corpus import authored
        corpus_a = authored.build_all()
        by_id = {t.id: t for t in corpus_a}
        chosen_now = [by_id[i] for i in want if i in by_id]
        twins_all = [build_corpus.build_twin(a)[0] for a in chosen_now] if twins else []

    items, missing = sweep.plan_experiment(
        corpus_a, twins_all, ids=want, include_twins=twins, k=k, repeats=repeats, model_id=model)
    if missing:
        typer.echo(f"WARNING: unknown ids skipped: {missing}")
    if not items:
        typer.echo("no work items — check --ids")
        raise typer.Exit(2)

    report = sweep.estimate(items, cfg=cfg)
    typer.echo(f"experiment: {len(items)} runs  (model={model}, provider={provider}, k={k}, "
               f"repeats={repeats}, test_size={test_size})")
    typer.echo(report.render())
    if not run:
        typer.echo("(estimate only — add --run to execute)")
        raise typer.Exit(0)

    client = None
    if provider != "mock":
        if not get_api_key("anthropic"):
            typer.echo("ERROR: ANTHROPIC_API_KEY is not set in the environment.")
            raise typer.Exit(2)
        client = ModelClient(cfg)
    budget.guard(report, accept_cost=i_accept_cost)
    outcomes = sweep.run_sweep(items, client=client, cfg=cfg)
    df = aggregate.outcomes_to_frame(outcomes)
    typer.echo(f"\n{len(outcomes)} runs complete")
    typer.echo(aggregate.cell_table(df).to_string(index=False))

    from collections import Counter as _Counter
    breakdown = _Counter(o.outcome for o in outcomes)
    typer.echo(f"\noutcomes: {dict(breakdown)}")
    # If runs errored, surface the first error note from a transcript (don't fail silently).
    if breakdown.get("error"):
        import json as _json
        for o in outcomes:
            if o.outcome != "error":
                continue
            if o.transcript_path and Path(o.transcript_path).exists():
                tr = _json.loads(Path(o.transcript_path).read_text())
                notes = [m.get("content", "") for m in tr if m.get("role") == "harness"]
                if notes:
                    typer.echo(f"first error: {notes[-1][:400]}")
                    break
            elif o.transcript_path.startswith("("):  # inline crash message
                typer.echo(f"first error: {o.transcript_path}")
                break
    if client is not None:
        typer.echo(f"\nactual spend: ${client.total_usd:.4f}")


@app.command("show-data")
def show_data(
    id: str = typer.Option("A-HEALTH-05", "--id", help="Corpus-A program id."),
    k: int = 12,
    seed: int = 0,
    test_size: int = 5,
    semantic: bool = typer.Option(True, "--semantic/--abstract", help="Show real labels, or masked."),
    out: str = typer.Option("", help="If set, write the FULL splits as JSON into this dir."),
) -> None:
    """Reproduce and display a program's train / test_id / test_ood rows.

    Splits are deterministic from (program, seed): to see the exact data a given
    run used, pass that run's seed (the number after '#p' in its manifest name)
    and the same --k / --test-size.
    """
    import json as _json

    from .corpus import authored
    from .data import sampler, serialize
    from .dsl import executor

    cfg = get_config()
    cfg = cfg.model_copy(update={"scoring": cfg.scoring.model_copy(
        update={"test_id_size": test_size, "test_ood_size": test_size})})
    by_id = {t.id: t for t in authored.build_all()}
    if id not in by_id:
        typer.echo(f"unknown id {id!r}. See manifests/samples/corpus_summary.md for the list.")
        raise typer.Exit(2)
    t = by_id[id]
    bundle = sampler.make_dataset(t, k=k, seed=seed, branch_bias=True, cfg=cfg)

    typer.echo(f"=== {t.id}  ({t.domain})  {t.description} ===")
    typer.echo(f"firewall row-id ranges: train {sorted(bundle.train.row_ids)[:1]}.. | "
               f"test_id from {min(bundle.test_id.row_ids)} | test_ood from {min(bundle.test_ood.row_ids)} | "
               f"disjoint={bundle.firewall_ok()}\n")

    typer.echo("--- TRAIN (this is exactly what the model is shown) ---")
    typer.echo(serialize.for_prompt(bundle.train, t.schema, mask_names=not semantic))

    for split in (bundle.test_id, bundle.test_ood):
        typer.echo(f"\n--- {split.kind.upper()} (held out; never shown to the model) ---")
        for ex in split.examples[:test_size]:
            typer.echo(f"  {ex.x}  ->  {ex.y}")

    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        for split in (bundle.train, bundle.test_id, bundle.test_ood):
            (d / f"{t.id}__{split.kind}__seed{seed}.json").write_text(
                _json.dumps([{"row_id": e.row_id, "x": e.x, "y": e.y} for e in split.examples],
                            indent=2, default=lambda o: round(o, 6) if isinstance(o, float) else o))
        # sanity: the ground-truth source that produced these rows
        (d / f"{t.id}__source.py").write_text(t.source)
        typer.echo(f"\nwrote full splits + ground-truth source to {out}/")
        _ = executor  # (kept available for ad-hoc verification)


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
