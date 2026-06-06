"""P_x sampling -> D_k (train), D_test (in-distribution), D_test_OOD.

Firewall (build spec §0.5): train and test draws use INDEPENDENT derived seeds,
and each split gets a DISJOINT row-id range. The harness only ever receives the
train split (and a validation slice of it); the scorer asserts test row-ids are
disjoint from anything the harness saw.

Noise (`eps`/`flip`) is applied to TRAIN ONLY (protocol §3.7). OOD widens
numeric ranges beyond the training support; categorical domains are kept
unchanged so dispatch/lookup ops never see an unseen key.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..dsl import executor
from ..dsl.ops import BRANCH_OPS, branch_arm_taken, branch_arms
from ..types import Example, SampleSet, Schema, Transform

Kind = Literal["train", "test_id", "test_ood"]

# Disjoint row-id ranges per split (the structural backbone of the firewall).
_ROW_OFFSET: dict[str, int] = {"train": 0, "test_id": 10_000_000, "test_ood": 20_000_000}
_KIND_SALT: dict[str, int] = {"train": 1, "test_id": 7919, "test_ood": 104_729}


def _derived_seed(base_seed: int, kind: str) -> int:
    return (base_seed * 1_000_003 + _KIND_SALT[kind]) % (2**31)


def draw_x(schema: Schema, rng: random.Random, *, ood_factor: float | None = None) -> dict:
    """One input row from P_x. `ood_factor` widens numeric ranges for OOD."""
    x: dict = {}
    for f in schema.inputs:
        if f.type == "categorical":
            x[f.name] = rng.choice(f.categories or ["a"])  # domain unchanged for OOD
        elif f.type == "bool":
            x[f.name] = rng.random() < 0.5
        else:
            lo = float(f.low) if f.low is not None else -50.0
            hi = float(f.high) if f.high is not None else 50.0
            if ood_factor is not None and ood_factor > 1.0:
                extra = 0.5 * (hi - lo) * (ood_factor - 1.0)
                lo, hi = lo - extra, hi + extra
            val = rng.uniform(lo, hi)
            x[f.name] = int(round(val)) if f.type == "int" else val
    return x


def _ensure_branch_coverage(
    transform: Transform, examples: list[dict], rng: random.Random, max_tries: int
) -> None:
    """Replace samples as needed so every branch arm is represented (in place)."""
    g = transform.graph
    branch_nodes = [nd for nd in g.nodes if nd.op in BRANCH_OPS]
    if not branch_nodes:
        return
    seen: dict[str, set[int]] = {nd.node_id: set() for nd in branch_nodes}
    for ex in examples:
        _, cache = executor.run_graph_trace(g, ex["x"])
        for nd in branch_nodes:
            argv = [cache[i] if i in cache else ex["x"][i] for i in nd.inputs]
            seen[nd.node_id].add(branch_arm_taken(nd.op, nd.params, *argv))

    replace_idx = 0
    for nd in branch_nodes:
        arms = branch_arms(nd.op, nd.params)
        for arm in range(arms):
            if arm in seen[nd.node_id]:
                continue
            for _ in range(max_tries):
                x = draw_x(transform.schema, rng)
                _, cache = executor.run_graph_trace(g, x)
                argv = [cache[i] if i in cache else x[i] for i in nd.inputs]
                if branch_arm_taken(nd.op, nd.params, *argv) == arm:
                    examples[replace_idx % len(examples)] = {"x": x, "y": executor.run_graph(g, x)}
                    replace_idx += 1
                    seen[nd.node_id].add(arm)
                    break


def _apply_noise(
    examples: list[dict], schema: Schema, noise: str, rng: random.Random, cfg: Config
) -> None:
    """Apply training-label noise in place (train only)."""
    if noise == "none":
        return
    # Output categorical domains are not declared, so infer them from observed
    # labels (the value set the flip can choose an alternative from).
    domains: dict[str, list] = {}
    for f in schema.outputs:
        if f.type in ("categorical", "bool"):
            domains[f.name] = sorted({ex["y"][f.name] for ex in examples}, key=repr)
    for ex in examples:
        y = ex["y"]
        for f in schema.outputs:
            v = y[f.name]
            if noise == "eps" and f.type in ("int", "float") and not isinstance(v, bool):
                sigma = cfg.sampling.noise_eps_sigma * max(1.0, abs(float(v)))
                y[f.name] = float(v) + rng.gauss(0.0, sigma)
            elif noise == "flip" and f.type in ("categorical", "bool"):
                if rng.random() < cfg.sampling.noise_flip_prob:
                    if f.type == "bool":
                        y[f.name] = not v
                    else:
                        alts = [c for c in domains.get(f.name, []) if c != v]
                        if alts:
                            y[f.name] = rng.choice(alts)


def make_samples(
    transform: Transform,
    k: int,
    seed: int,
    kind: Kind,
    *,
    branch_bias: bool = False,
    noise: str = "none",
    cfg: Config | None = None,
) -> SampleSet:
    """Draw k examples for one split, evaluate ground truth, return a SampleSet."""
    cfg = cfg or get_config()
    derived = _derived_seed(seed, kind)
    rng = random.Random(derived)
    ood = cfg.sampling.ood_shift_factor if kind == "test_ood" else None

    examples: list[dict] = []
    for _ in range(k):
        x = draw_x(transform.schema, rng, ood_factor=ood)
        examples.append({"x": x, "y": executor.run_graph(transform.graph, x)})

    if kind == "train" and branch_bias:
        _ensure_branch_coverage(transform, examples, rng, cfg.sampling.max_branch_coverage_tries)
    if kind == "train" and noise != "none":
        _apply_noise(examples, transform.schema, noise, rng, cfg)

    offset = _ROW_OFFSET[kind]
    out = [Example(row_id=offset + i, x=e["x"], y=e["y"]) for i, e in enumerate(examples)]
    return SampleSet(kind=kind, examples=out, seed=derived)


class SampleBundle(BaseModel):
    """Train + both test splits for one task. The harness must touch only train."""

    model_config = ConfigDict(extra="forbid")
    train: SampleSet
    test_id: SampleSet
    test_ood: SampleSet

    def firewall_ok(self) -> bool:
        tr = self.train.row_ids
        return tr.isdisjoint(self.test_id.row_ids) and tr.isdisjoint(self.test_ood.row_ids)


def make_dataset(
    transform: Transform,
    k: int,
    *,
    seed: int,
    branch_bias: bool = True,
    noise: str = "none",
    cfg: Config | None = None,
) -> SampleBundle:
    """Build train (k), test_id, test_ood for a transform with the D_test firewall."""
    cfg = cfg or get_config()
    bundle = SampleBundle(
        train=make_samples(transform, k, seed, "train", branch_bias=branch_bias, noise=noise, cfg=cfg),
        test_id=make_samples(transform, cfg.scoring.test_id_size, seed, "test_id", cfg=cfg),
        test_ood=make_samples(transform, cfg.scoring.test_ood_size, seed, "test_ood", cfg=cfg),
    )
    assert bundle.firewall_ok(), "D_test row-ids overlap train — firewall breach"
    return bundle
