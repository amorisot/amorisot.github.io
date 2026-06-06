"""Agent-loop tests (CHECKPOINT 5) — MockModel only, no network."""

import pytest

from progrecon.config import load_config
from progrecon.data import sampler
from progrecon.harness import agent_loop
from progrecon.runner.model_client import MockModel, ModelClient
from progrecon.types import (
    DataflowGraph,
    FieldSpec,
    GridCell,
    OpNode,
    PerturbationSpec,
    Schema,
    Task,
)

from .conftest import build_transform

CFG = load_config()


def _transform():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=30)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 4}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    return build_transform(schema, g, tid="B-harness")


def _task(t):
    return Task(task_id="B-harness#p0", transform=t, perturbation=PerturbationSpec(seed=0))


def _cell(query_budget=0, model_id="mock"):
    return GridCell(
        n=1, m=1, k=40, corpus="B", complexity_band="low", noise="none",
        mode="scratch", model_id=model_id, query_budget=query_budget,
    )


def _wrap(source):
    return f"Here is my program.\n```python\n{source}\n```\nDONE"


def test_extract_code_and_query():
    assert agent_loop.extract_code("```python\nx=1\n```") == "x=1"
    assert agent_loop.extract_code("no code here") is None
    assert agent_loop.extract_query('QUERY: [{"in_0": 5}]') == [{"in_0": 5}]
    assert agent_loop.extract_query("no query") is None


def test_oracle_mock_solves_in_one_round():
    t = _transform()
    train = sampler.make_samples(t, 40, 1, "train")
    client = ModelClient(CFG, mock_model=MockModel(responses=[_wrap(t.source)]))
    res = agent_loop.run(_task(t), _cell(), client, train, t)
    assert res.outcome == "solved"
    assert res.iterations_used == 1
    assert res.final_source is not None


def test_wrong_program_exhausts_budget():
    t = _transform()
    train = sampler.make_samples(t, 40, 2, "train")
    wrong = "def transform(x):\n    return {'out_0': x['in_0'] + 1000}"
    client = ModelClient(CFG, mock_model=MockModel(responses=[_wrap(wrong)]))
    res = agent_loop.run(_task(t), _cell(), client, train, t)
    assert res.outcome == "budget_exhausted"
    assert res.iterations_used == CFG.harness.b_iter


def test_query_then_solve():
    t = _transform()
    train = sampler.make_samples(t, 40, 3, "train")
    responses = ['QUERY: [{"in_0": 7}, {"in_0": 8}]', _wrap(t.source)]
    client = ModelClient(CFG, mock_model=MockModel(responses=responses))
    res = agent_loop.run(_task(t), _cell(query_budget=2), client, train, t)
    assert res.outcome == "solved"
    assert res.queries_used == 1
    assert res.iterations_used == 2


def test_no_code_then_solve():
    t = _transform()
    train = sampler.make_samples(t, 40, 4, "train")
    responses = ["I think the answer is mod 4.", _wrap(t.source)]
    client = ModelClient(CFG, mock_model=MockModel(responses=responses))
    res = agent_loop.run(_task(t), _cell(), client, train, t)
    assert res.outcome == "solved"
    assert res.iterations_used == 2


def test_firewall_rejects_test_split():
    t = _transform()
    test = sampler.make_samples(t, 20, 1, "test_id")
    client = ModelClient(CFG, mock_model=MockModel(responses=["x"]))
    with pytest.raises(AssertionError):
        agent_loop.run(_task(t), _cell(), client, test, t)
