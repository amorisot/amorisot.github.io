"""Metered model client tests (CHECKPOINT 5) — all via MockModel, no network."""

import pytest

from progrecon.config import load_config
from progrecon.runner.model_client import BudgetExceeded, MockModel, ModelClient

CFG = load_config()


def test_mock_returns_scripted_text_and_meters_zero_for_mock_pricing():
    client = ModelClient(CFG, mock_model=MockModel(responses=["hello"]))
    resp = client.complete(system="sys", messages=[{"role": "user", "content": "hi"}], model_id="mock", max_tokens=100)
    assert resp.text == "hello"
    assert resp.usd_cost == 0.0  # mock pricing is zero
    assert client.total_usd == 0.0
    assert len(client.calls) == 1


def test_cost_metering_uses_model_pricing():
    # Route through the cheap model id (Haiku pricing) but the mock provider.
    cheap_id = CFG.models.cheap.id
    client = ModelClient(CFG, mock_model=MockModel(responses=["x" * 4000]))
    resp = client.complete(system="s" * 4000, messages=[], model_id=cheap_id, max_tokens=100)
    assert resp.input_tokens > 0 and resp.output_tokens > 0
    assert resp.usd_cost > 0.0
    assert client.total_usd == pytest.approx(resp.usd_cost)


def test_global_ceiling_aborts():
    cfg = CFG.model_copy(update={"budget": CFG.budget.model_copy(update={"usd_ceiling": 1e-9})})
    client = ModelClient(cfg, mock_model=MockModel(responses=["y" * 8000]))
    with pytest.raises(BudgetExceeded):
        client.complete(system="s" * 8000, messages=[], model_id=cfg.models.cheap.id, max_tokens=100)
    # the breaching call is still recorded before aborting
    assert client.calls and client.calls[-1].cumulative_usd > cfg.budget.usd_ceiling


def test_mock_requires_a_mock_model():
    client = ModelClient(CFG)  # no mock_model
    with pytest.raises(RuntimeError, match="no MockModel"):
        client.complete(system="s", messages=[], model_id="mock", max_tokens=10)


def test_responder_callable():
    def responder(system, messages):
        return f"saw {len(messages)} messages"

    client = ModelClient(CFG, mock_model=MockModel(responder=responder))
    resp = client.complete(system="s", messages=[{"role": "user", "content": "a"}], model_id="mock", max_tokens=10)
    assert resp.text == "saw 1 messages"
