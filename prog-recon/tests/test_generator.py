"""Corpus-B generator tests (CHECKPOINT 3)."""

import random

import pytest

from progrecon.config import load_config
from progrecon.corpus import complexity, generator
from progrecon.dsl import executor, graph

CFG = load_config()


@pytest.mark.parametrize("pers_idx", range(3))
@pytest.mark.parametrize("seed", range(6))
def test_generated_transforms_are_valid_and_roundtrip(pers_idx, seed):
    pers = CFG.generators[pers_idx]
    t = generator.sample_transform(pers, n=3, m=1, target_depth=3, seed=3000 + seed, transform_id=f"B-{pers.name}-{seed}")
    # structurally valid
    assert graph.validate(t.graph, t.schema) == []
    # n/m hit targets (n may shrink to referenced fields; here >=1, <=3)
    assert 1 <= t.schema.n <= 3
    assert t.schema.m == 1
    # every declared input field is referenced by some node (no dead inputs)
    referenced = {i for nd in t.graph.nodes for i in nd.inputs}
    assert all(f.name in referenced for f in t.schema.inputs)
    # codegen source == canonical executor over many inputs
    rng = random.Random(seed)
    for _ in range(100):
        x = {}
        for f in t.schema.inputs:
            x[f.name] = rng.choice(f.categories) if f.type == "categorical" else rng.randint(int(f.low), int(f.high))
        assert executor.run_graph(t.graph, x) == executor.run_source(t.source, x)


def test_complexity_band_assigned():
    pers = CFG.generators[2]
    t = generator.sample_transform(pers, n=4, m=2, target_depth=4, seed=42, transform_id="B-band")
    c = complexity.transform_complexity(t)
    assert c > 0
    assert complexity.transform_band(t) in ("low", "med", "high")


def test_provenance_recorded():
    t = generator.sample_transform(CFG.generators[0], n=2, m=1, target_depth=2, seed=1, transform_id="B-prov")
    assert t.provenance["generator"] == CFG.generators[0].name
    assert t.provenance["seed"] == 1
    assert "target" in t.provenance
