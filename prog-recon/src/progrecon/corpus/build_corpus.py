"""Assemble the full corpus: 50 Corpus B + 50 Corpus A + 50 abstract twins.

* Corpus A: hand-authored semantic transforms (``authored.load_corpus_a``).
* Corpus B: generated abstract transforms, varying personality/shape/seed,
  each passing identifiability (failures quarantined, not run).
* Twins: one abstract twin per Corpus A program (structurally identical,
  labels masked, ops swapped) — the matched control for Claim 3. Each twin is
  itself re-checked for identifiability; if a swap happens to make it reducible
  we retry with a different seed.

This is pure offline construction — NO model calls. It does not run the grid.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..types import Transform
from . import authored, generator, identifiability, twin
from .complexity import transform_band, transform_complexity

_B_SHAPES = [
    (n, m, d)
    for n in (2, 3, 4, 5)
    for m in (1, 2)
    for d in (2, 3, 4)
]


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_a: list[Transform]
    corpus_b: list[Transform]
    twins: list[Transform]
    quarantined_a: list[str]
    quarantined_b: int  # count of generated-but-rejected B candidates
    quarantined_twins: list[str]

    def summary(self) -> dict[str, int]:
        return {
            "corpus_a": len(self.corpus_a),
            "corpus_b": len(self.corpus_b),
            "twins": len(self.twins),
            "quarantined_a": len(self.quarantined_a),
            "quarantined_b": self.quarantined_b,
            "quarantined_twins": len(self.quarantined_twins),
        }


def build_corpus_b(
    n: int = 50, *, seed: int = 0, cfg: Config | None = None, max_attempts: int = 1200
) -> tuple[list[Transform], int]:
    cfg = cfg or get_config()
    kept: list[Transform] = []
    seen_sources: set[str] = set()
    rejected = 0
    attempt = 0
    idx = 0
    while len(kept) < n and attempt < max_attempts:
        persona = cfg.generators[attempt % len(cfg.generators)]
        bn, bm, bd = _B_SHAPES[attempt % len(_B_SHAPES)]
        attempt += 1
        try:
            t = generator.sample_transform(
                persona, n=bn, m=bm, target_depth=bd, seed=seed + attempt * 2654435761 % (2**31),
                transform_id=f"B-GEN-{idx:03d}",
            )
        except generator.GenerationError:
            rejected += 1
            continue
        if t.source in seen_sources:
            rejected += 1
            continue
        rep = identifiability.check(t, n_probes=300)
        if not rep.passed:
            rejected += 1
            continue
        seen_sources.add(t.source)
        kept.append(t)
        idx += 1
    return kept, rejected


def build_twin(a: Transform, *, max_tries: int = 10) -> tuple[Transform, bool]:
    """Build an identified abstract twin of `a`; (twin, identified?)."""
    last = None
    for s in range(max_tries):
        tw = twin.make_abstract(a, seed=1000 + s, twin_id=f"{a.id}~twin")
        rep = identifiability.check(tw, n_probes=300)
        last = tw
        if rep.passed:
            return tw, True
    assert last is not None
    return last, False


def build_full_corpus(*, n_b: int = 50, seed: int = 0, cfg: Config | None = None) -> CorpusManifest:
    cfg = cfg or get_config()
    corpus_a, quar_a = authored.load_corpus_a(n_probes=400)
    corpus_b, rejected_b = build_corpus_b(n_b, seed=seed, cfg=cfg)

    twins: list[Transform] = []
    quar_twins: list[str] = []
    for a in corpus_a:
        tw, ok = build_twin(a)
        twins.append(tw)
        if not ok:
            quar_twins.append(tw.id)

    return CorpusManifest(
        corpus_a=corpus_a, corpus_b=corpus_b, twins=twins,
        quarantined_a=[tid for tid, _ in quar_a],
        quarantined_b=rejected_b, quarantined_twins=quar_twins,
    )


def render_summary(man: CorpusManifest) -> str:
    s = man.summary()
    lines = [
        "# Corpus summary (50 B + 50 A + 50 twins)",
        "",
        "> Pure offline generation — no model calls, no grid run. Identifiability-",
        "> failing candidates are quarantined (not run).",
        "",
        f"- Corpus A (authored semantic): **{s['corpus_a']}** identified "
        f"({s['quarantined_a']} quarantined)",
        f"- Corpus B (generated abstract): **{s['corpus_b']}** identified "
        f"({s['quarantined_b']} candidates rejected during search)",
        f"- Twins (abstract controls of A): **{s['twins']}** "
        f"({s['quarantined_twins']} not fully identified)",
        "",
    ]
    for title, items in (("Corpus A", man.corpus_a), ("Corpus B", man.corpus_b)):
        lines.append(f"## {title} — id / domain / C(T) / band")
        lines.append("")
        for t in items:
            dom = t.domain or "-"
            lines.append(f"- `{t.id}` {dom}  C={transform_complexity(t):.1f} band={transform_band(t)}")
        lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: Path, transforms: list[Transform]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(t.model_dump_json() for t in transforms) + "\n")


def write_corpus(man: CorpusManifest, *, cfg: Config | None = None) -> dict[str, str]:
    """Persist the full corpus (JSONL, gitignored) + a committed summary md."""
    cfg = cfg or get_config()
    base = Path(cfg.paths.manifests_dir)
    corpus_dir = base / "corpus"
    _write_jsonl(corpus_dir / "corpus_a.jsonl", man.corpus_a)
    _write_jsonl(corpus_dir / "corpus_b.jsonl", man.corpus_b)
    _write_jsonl(corpus_dir / "twins.jsonl", man.twins)
    summary_path = base / "samples" / "corpus_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_summary(man))
    return {
        "corpus_a": str(corpus_dir / "corpus_a.jsonl"),
        "corpus_b": str(corpus_dir / "corpus_b.jsonl"),
        "twins": str(corpus_dir / "twins.jsonl"),
        "summary": str(summary_path),
    }


def load_corpus_jsonl(path: str | Path) -> list[Transform]:
    """Reload a persisted corpus split."""
    p = Path(path)
    return [Transform.model_validate_json(line) for line in p.read_text().splitlines() if line.strip()]
