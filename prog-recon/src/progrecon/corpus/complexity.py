"""Complexity index C(T).

C(T) is a weighted sum of the structural components of the ComplexityTuple.
Weights come from ``config.yaml`` (placeholders now; calibrated against the
pilot in Phase 7.1). The component breakdown is exposed so the calibration can
inspect each lever independently.
"""

from __future__ import annotations

from ..config import ComplexityBands, ComplexityWeights, get_config
from ..types import ComplexityTuple, Transform


def components(ct: ComplexityTuple, w: ComplexityWeights) -> dict[str, float]:
    return {
        "depth": w.w_depth * ct.depth,
        "arity": w.w_arity * ct.max_arity,
        "tier": w.w_tier * ct.max_tier,
        "branch": w.w_branch * ct.n_branches,
    }


def C(ct: ComplexityTuple, w: ComplexityWeights | None = None) -> float:
    if w is None:
        w = get_config().complexity_weights
    return float(sum(components(ct, w).values()))


def band(c_value: float, bands: ComplexityBands | None = None) -> str:
    if bands is None:
        bands = get_config().complexity_bands
    if c_value <= bands.low_max:
        return "low"
    if c_value <= bands.med_max:
        return "med"
    return "high"


def transform_complexity(t: Transform, w: ComplexityWeights | None = None) -> float:
    return C(t.complexity, w)


def transform_band(t: Transform) -> str:
    return band(transform_complexity(t))
