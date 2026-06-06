"""Identifiability pre-flight checks (protocol §3.8).

A task is *runnable* only if it passes:
1. branch coverage — every arm of every conditional/dispatch node fires under
   P_x within a bounded number of probes (else the data can't pin the program);
2. determinism — the graph and the codegen'd source agree, and re-evaluation is
   stable (ops are pure, so this is a regression guard);
3. distinguishability — a BOUNDED enumerator searches for a strictly *simpler*
   program that reproduces each output on a probe set; if one exists the target
   is under-identified (the data admits a simpler explanation) and is quarantined.

Tasks that fail are quarantined (flagged ``passed=False``), not run. The
enumerator here is deliberately bounded for the vertical slice (copy / affine /
mod_k / categorical-map explanations over single input fields); Phase 7 can
widen it.
"""

from __future__ import annotations

import math
import random

from pydantic import BaseModel, ConfigDict

from ..dsl import executor
from ..dsl.graph import ancestor_nodes
from ..dsl.ops import BRANCH_OPS, branch_arm_taken, branch_arms
from ..types import Schema, Transform


class BranchCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    op: str
    arms: int
    covered: list[int]
    missing: list[int]


class IdentifiabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transform_id: str
    passed: bool
    flags: list[str]
    deterministic: bool
    n_probes: int
    branch_coverage: list[BranchCoverage]
    simpler_equivalent_outputs: list[str]
    constant_outputs: list[str]


def _draw_x(schema: Schema, rng: random.Random) -> dict:
    """Minimal uniform P_x draw (the canonical biased sampler lives in data/)."""
    x: dict = {}
    for f in schema.inputs:
        if f.type == "categorical":
            x[f.name] = rng.choice(f.categories or ["a"])
        elif f.type == "bool":
            x[f.name] = rng.random() < 0.5
        else:
            lo = int(f.low) if f.low is not None else -50
            hi = int(f.high) if f.high is not None else 50
            x[f.name] = rng.randint(lo, max(lo, hi))
    return x


def _approx_eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def _fits_affine(pairs: list[tuple[float, float]]) -> bool:
    """Does an a*x+b explain all (x,y) pairs? Needs >=2 distinct x."""
    distinct_x = {round(x, 9) for x, _ in pairs}
    if len(distinct_x) < 2:
        return False
    (x1, y1), (x2, y2) = pairs[0], next(p for p in pairs if not _approx_eq(p[0], pairs[0][0]))
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return all(_approx_eq(a * x + b, y) for x, y in pairs)


def _fits_mod_k(pairs: list[tuple[int, int]]) -> bool:
    for k in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12):
        if all((isinstance(x, int) and (x % k) == y) for x, y in pairs):
            return True
    return False


def _simpler_explanation(transform: Transform, probes: list[tuple[dict, dict]]) -> list[str]:
    """Outputs that a single-input, single-op program reproduces on all probes,
    while the true output cone has more than one node (=> under-identified)."""
    g = transform.graph
    flagged: list[str] = []
    field_names = [f.name for f in transform.schema.inputs]
    field_type = {f.name: f.type for f in transform.schema.inputs}
    for f in transform.schema.outputs:
        producer = g.output_map[f.name]
        cone = ancestor_nodes(g, {producer})
        if len(cone) <= 1:
            continue  # already a single op; can't be "simpler"
        ys = [y[f.name] for _, y in probes]
        # (a) exact copy of an input field
        if any(all(_approx_eq(x[fn], y) for (x, _), y in zip(probes, ys, strict=True)) for fn in field_names):
            flagged.append(f.name)
            continue
        # (b) single-input affine / mod_k / categorical-map
        found = False
        for fn in field_names:
            xs = [x[fn] for x, _ in probes]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xs) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in ys
            ):
                pairs = list(zip(xs, ys, strict=True))
                if _fits_affine([(float(a), float(b)) for a, b in pairs]):
                    found = True
                    break
                if all(isinstance(a, int) and isinstance(b, int) for a, b in pairs) and _fits_mod_k(pairs):
                    found = True
                    break
            # lookup_table reducibility: only meaningful for a CATEGORICAL input
            # (a lookup over a wide numeric domain is memorization, not a simpler
            # program). The output must be a pure function of this single field.
            if field_type.get(fn) == "categorical":
                mapping: dict = {}
                consistent = True
                for xv, yv in zip(xs, ys, strict=True):
                    if xv in mapping and mapping[xv] != yv:
                        consistent = False
                        break
                    mapping[xv] = yv
                if consistent and len(mapping) < len(xs):  # repeats observed -> real map
                    found = True
                    break
        if found:
            flagged.append(f.name)
    return flagged


def check(transform: Transform, k_min: int = 50, *, seed: int = 0, n_probes: int | None = None) -> IdentifiabilityReport:
    g = transform.graph
    schema = transform.schema
    rng = random.Random(seed if seed else transform.px_seed)
    n = n_probes if n_probes is not None else max(k_min, 200)

    branch_nodes = [nd for nd in g.nodes if nd.op in BRANCH_OPS]
    seen_arms: dict[str, set[int]] = {nd.node_id: set() for nd in branch_nodes}

    deterministic = True
    probes: list[tuple[dict, dict]] = []
    for _ in range(n):
        x = _draw_x(schema, rng)
        y, cache = executor.run_graph_trace(g, x)
        # determinism: source agrees with graph
        try:
            y_src = executor.run_source(transform.source, x)
        except Exception:  # noqa: BLE001
            deterministic = False
            y_src = None
        if y_src is not None and y_src != y:
            deterministic = False
        probes.append((x, y))
        for nd in branch_nodes:
            argv = [cache[i] if i in cache else x[i] for i in nd.inputs]
            seen_arms[nd.node_id].add(branch_arm_taken(nd.op, nd.params, *argv))

    coverage: list[BranchCoverage] = []
    for nd in branch_nodes:
        arms = branch_arms(nd.op, nd.params)
        covered = sorted(seen_arms[nd.node_id])
        missing = [a for a in range(arms) if a not in seen_arms[nd.node_id]]
        coverage.append(BranchCoverage(node_id=nd.node_id, op=nd.op, arms=arms, covered=covered, missing=missing))

    constant_outputs = [
        f.name for f in schema.outputs
        if len({y[f.name] for _, y in probes}) <= 1
    ]
    simpler = _simpler_explanation(transform, probes)

    flags: list[str] = []
    if not deterministic:
        flags.append("nondeterministic")
    if any(bc.missing for bc in coverage):
        flags.append("incomplete_branch_coverage")
    if constant_outputs:
        flags.append("constant_output")
    if simpler:
        flags.append("simpler_equivalent_exists")

    return IdentifiabilityReport(
        transform_id=transform.id,
        passed=not flags,
        flags=flags,
        deterministic=deterministic,
        n_probes=n,
        branch_coverage=coverage,
        simpler_equivalent_outputs=simpler,
        constant_outputs=constant_outputs,
    )


def is_runnable(report: IdentifiabilityReport) -> bool:
    return report.passed
