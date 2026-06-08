"""Typed configuration surface.

A single ``config.yaml`` holds every tunable knob; secrets come from the
environment only (never from yaml). ``load_config`` parses the yaml into typed
pydantic models so the rest of the codebase reads configuration through one
validated object.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _CfgBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BudgetConfig(_CfgBase):
    usd_ceiling: float
    per_run_token_cap: int


class ModelEntry(_CfgBase):
    id: str
    provider: str  # "mock" | "anthropic" | ...
    version: str
    temperature: float = 0.0


class ModelsConfig(_CfgBase):
    cheap: ModelEntry
    roster: list[ModelEntry] = Field(default_factory=list)


class PriceEntry(_CfgBase):
    input_per_mtok: float
    output_per_mtok: float


class ComplexityWeights(_CfgBase):
    w_depth: float
    w_arity: float
    w_tier: float
    w_branch: float


class ComplexityBands(_CfgBase):
    low_max: float
    med_max: float


class GridConfig(_CfgBase):
    n: list[int]
    m: list[int]
    k: list[int]
    corpus: list[str]
    complexity_band: list[str]
    noise: list[str]
    mode: list[str]
    query_budget: list[int]
    sweeps: list[dict[str, Any]] = Field(default_factory=list)


class HarnessConfig(_CfgBase):
    b_iter: int
    b_query_default: int
    train_validation_fraction: float = 0.2
    max_output_tokens: int = 4096  # per-CALL output cap (distinct from the per-run budget)


class SandboxConfig(_CfgBase):
    timeout_s: float
    mem_mb: int


class ScoringConfig(_CfgBase):
    rel_tol: float
    test_id_size: int
    test_ood_size: int


class SamplingConfig(_CfgBase):
    ood_shift_factor: float
    noise_eps_sigma: float
    noise_flip_prob: float
    max_branch_coverage_tries: int = 400


class GeneratorPersonality(_CfgBase):
    name: str
    tiers: list[int]
    max_depth: int
    arity_bias: str  # "wide" | "narrow" | "mixed"


class PathsConfig(_CfgBase):
    manifests_dir: str = "manifests"
    logs_dir: str = "logs"


class Config(_CfgBase):
    budget: BudgetConfig
    models: ModelsConfig
    model_pricing: dict[str, PriceEntry]
    complexity_weights: ComplexityWeights
    complexity_bands: ComplexityBands
    grid: GridConfig
    harness: HarnessConfig
    sandbox: SandboxConfig
    scoring: ScoringConfig
    sampling: SamplingConfig
    generators: list[GeneratorPersonality]
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def price_for(self, model_id: str) -> PriceEntry:
        if model_id not in self.model_pricing:
            raise KeyError(
                f"no pricing configured for model_id={model_id!r}; add it to "
                f"config.yaml:model_pricing before running"
            )
        return self.model_pricing[model_id]


def _default_config_path() -> Path:
    """Locate config.yaml: env override, then project root next to pyproject."""
    env = os.environ.get("PROGRECON_CONFIG")
    if env:
        return Path(env)
    # src/progrecon/config.py -> project root is two parents up from src/progrecon
    here = Path(__file__).resolve()
    root = here.parents[2]  # .../prog-recon
    return root / "config.yaml"


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else _default_config_path()
    if not p.exists():
        raise FileNotFoundError(f"config not found at {p}")
    data = yaml.safe_load(p.read_text())
    return Config.model_validate(data)


@lru_cache(maxsize=8)
def get_config(path: str | None = None) -> Config:
    """Cached config accessor (most callers want this)."""
    return load_config(path)


# --- Secrets (env only) ----------------------------------------------------


def get_api_key(provider: str) -> str | None:
    """Return the API key for a provider from the environment, or None.

    Secrets are *never* read from config.yaml. The model client raises a clear
    error if a real provider is requested without its key present.
    """
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider)
    if env_var is None:
        return None
    return os.environ.get(env_var)
