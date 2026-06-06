"""Perturbation tests (CHECKPOINT 3)."""

import random

from progrecon.config import load_config
from progrecon.corpus import generator, perturb
from progrecon.dsl import executor, graph
from progrecon.types import PerturbationSpec

CFG = load_config()


def _base():
    return generator.sample_transform(CFG.generators[1], n=3, m=1, target_depth=3, seed=55, transform_id="B-base")


def _draw(schema, rng):
    x = {}
    for f in schema.inputs:
        x[f.name] = rng.choice(f.categories) if f.type == "categorical" else rng.randint(int(f.low), int(f.high))
    return x


def test_distractors_increase_n_and_are_unused():
    t = _base()
    spec = PerturbationSpec(seed=3, n_distractors=4)
    pt = perturb.apply(t, spec)
    assert pt.complexity.n == t.complexity.n + 4
    referenced = {i for nd in pt.graph.nodes for i in nd.inputs}
    distractors = [f.name for f in pt.schema.inputs if f.name.startswith("distractor_")]
    assert len(distractors) == 4
    assert all(d not in referenced for d in distractors)
    assert graph.validate(pt.graph, pt.schema) == []


def test_distractors_only_preserve_behavior():
    t = _base()
    pt = perturb.apply(t, PerturbationSpec(seed=9, n_distractors=3))
    rng = random.Random(0)
    for _ in range(50):
        x = _draw(t.schema, rng)
        xd = dict(x)
        for f in pt.schema.inputs:
            if f.name.startswith("distractor_"):
                xd[f.name] = rng.choice(f.categories) if f.type == "categorical" else rng.randint(int(f.low), int(f.high))
        assert executor.run_graph(pt.graph, xd) == executor.run_graph(t.graph, x)


def test_permute_only_preserves_behavior():
    t = _base()
    pt = perturb.apply(t, PerturbationSpec(seed=2, permute_fields=True))
    # names unchanged, only order -> identical mapping
    rng = random.Random(1)
    for _ in range(50):
        x = _draw(t.schema, rng)
        assert executor.run_graph(pt.graph, x) == executor.run_graph(t.graph, x)


def test_rename_produces_valid_roundtripping_graph():
    t = _base()
    pt = perturb.apply(t, PerturbationSpec(seed=4, rename_fields=True))
    assert graph.validate(pt.graph, pt.schema) == []
    rng = random.Random(2)
    for _ in range(50):
        x = _draw(pt.schema, rng)
        assert executor.run_graph(pt.graph, x) == executor.run_source(pt.source, x)


def test_resample_keeps_valid_and_output_types():
    t = _base()
    pt = perturb.apply(t, PerturbationSpec(seed=7, resample_constants=True))
    assert graph.validate(pt.graph, pt.schema) == []
    assert [f.type for f in pt.schema.outputs] == [f.type for f in t.schema.outputs]


def test_make_task_id():
    t = _base()
    task = perturb.make_task(t, PerturbationSpec(seed=123, n_distractors=1))
    assert task.task_id == "B-base#p123"
    assert task.perturbation.n_distractors == 1
