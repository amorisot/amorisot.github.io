"""The ONE metered, rate-limited, multi-provider model client.

No other module calls a provider SDK directly (build spec §0.2). This client:
* dispatches by provider (``mock`` | ``anthropic``) configured per model_id;
* meters tokens and USD cost per call against the configured pricing;
* enforces a hard GLOBAL USD ceiling — once cumulative spend would exceed it,
  the client raises ``BudgetExceeded`` and the run aborts;
* reads secrets from the environment only;
* logs every call (tokens + cost) for the transcript/manifest.

``MockModel`` is a scripted, network-free responder used by ALL tests and as the
default ``cheap`` provider, so CI never hits a network. The Anthropic provider
is implemented but dormant: it is only reachable when a model is configured with
``provider: anthropic`` AND ``ANTHROPIC_API_KEY`` is present — neither is true by
default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import Config, ModelEntry, get_api_key, get_config

# A mock responder maps (system, messages) -> assistant text.
MockResponder = Callable[[str, list[dict[str, Any]]], str]


class BudgetExceeded(Exception):
    """Raised when a call would push cumulative spend past the USD ceiling."""


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    usd_cost: float


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    usd_cost: float
    cumulative_usd: float


class MockModel:
    """Scripted, network-free responder.

    Construct with either a fixed list of responses (consumed in order, last one
    repeated when exhausted) or a callable ``responder(system, messages) -> str``.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        responder: MockResponder | None = None,
    ):
        if (responses is None) == (responder is None):
            raise ValueError("provide exactly one of responses= or responder=")
        self._responses = responses
        self._responder = responder
        self._i = 0

    def respond(self, system: str, messages: list[dict[str, Any]]) -> str:
        if self._responder is not None:
            return self._responder(system, messages)
        assert self._responses is not None
        if not self._responses:
            return ""
        idx = min(self._i, len(self._responses) - 1)
        self._i += 1
        return self._responses[idx]


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate for the mock path (~4 chars/token)."""
    return max(1, len(text) // 4)


class ModelClient:
    """Single entry point for all model calls."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        mock_model: MockModel | None = None,
    ):
        self.config = config or get_config()
        self.mock_model = mock_model
        self.total_usd: float = 0.0
        self.calls: list[CallRecord] = []
        self._registry: dict[str, ModelEntry] = {self.config.models.cheap.id: self.config.models.cheap}
        for m in self.config.models.roster:
            self._registry[m.id] = m

    # -- provider resolution -------------------------------------------------

    def _entry(self, model_id: str) -> ModelEntry:
        if model_id in self._registry:
            return self._registry[model_id]
        # Unknown id: default to mock so tests/dev never accidentally hit network.
        return ModelEntry(id=model_id, provider="mock", version="n/a", temperature=0.0)

    def _cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        price = self.config.price_for(model_id)
        return (
            input_tokens / 1_000_000 * price.input_per_mtok
            + output_tokens / 1_000_000 * price.output_per_mtok
        )

    # -- public API ----------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: int,
    ) -> ModelResponse:
        entry = self._entry(model_id)
        if entry.provider == "mock":
            text, in_tok, out_tok = self._complete_mock(system, messages)
        elif entry.provider == "anthropic":
            text, in_tok, out_tok = self._complete_anthropic(system, messages, entry, max_tokens)
        else:  # pragma: no cover - guard
            raise ValueError(f"unknown provider {entry.provider!r} for model {model_id!r}")

        cost = self._cost(model_id, in_tok, out_tok)
        self.total_usd += cost
        self.calls.append(
            CallRecord(
                model_id=model_id, provider=entry.provider, input_tokens=in_tok,
                output_tokens=out_tok, usd_cost=cost, cumulative_usd=self.total_usd,
            )
        )
        if self.total_usd > self.config.budget.usd_ceiling:
            raise BudgetExceeded(
                f"cumulative spend ${self.total_usd:.4f} exceeded ceiling "
                f"${self.config.budget.usd_ceiling:.2f} — aborting run"
            )
        return ModelResponse(
            text=text, model_id=model_id, input_tokens=in_tok, output_tokens=out_tok, usd_cost=cost
        )

    # -- providers -----------------------------------------------------------

    def _complete_mock(self, system: str, messages: list[dict[str, Any]]) -> tuple[str, int, int]:
        if self.mock_model is None:
            raise RuntimeError(
                "provider is 'mock' but no MockModel was supplied to ModelClient(mock_model=...)"
            )
        text = self.mock_model.respond(system, messages)
        in_tok = _estimate_tokens(system) + sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)
        out_tok = _estimate_tokens(text)
        return text, in_tok, out_tok

    def _complete_anthropic(
        self, system: str, messages: list[dict[str, Any]], entry: ModelEntry, max_tokens: int
    ) -> tuple[str, int, int]:  # pragma: no cover - never exercised offline / in CI
        key = get_api_key("anthropic")
        if not key:
            raise RuntimeError(
                "provider 'anthropic' requires ANTHROPIC_API_KEY in the environment"
            )
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("install the 'providers' extra to use the anthropic provider") from e

        client = anthropic.Anthropic(api_key=key, max_retries=4)
        kwargs: dict[str, Any] = {
            "model": entry.id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        # Opus 4.7/4.8 reject sampling params; only send temperature where accepted.
        if not entry.id.startswith(("claude-opus-4-7", "claude-opus-4-8")):
            kwargs["temperature"] = entry.temperature
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
