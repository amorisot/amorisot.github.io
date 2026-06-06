"""Statistical models for F1–F4 figures and the §8 fits.

STUBBED until real data exists (post-CHECKPOINT 7). These define the intended
interfaces; they raise ``NotImplementedError`` so no half-baked statistics can
masquerade as results before the pilot calibrates the design.
"""

from __future__ import annotations

import pandas as pd

_STUB = (
    "analysis.models is stubbed until post-pilot data exists (CHECKPOINT 8). "
    "Implement against the frozen pre-registration, not before."
)


def fit_dimensionality_curve(df: pd.DataFrame):  # noqa: ANN201
    """Claim 1: exact-reconstruction rate vs n (and complexity). Stub."""
    raise NotImplementedError(_STUB)


def fit_sample_curve(df: pd.DataFrame):  # noqa: ANN201
    """Claim 2: reconstruction vs k (sample size). Stub."""
    raise NotImplementedError(_STUB)


def fit_semantic_effect(df: pd.DataFrame):  # noqa: ANN201
    """Claim 3: semantic vs abstract twin effect (mixed-effects). Stub."""
    raise NotImplementedError(_STUB)


def fit_leakage_interaction(df: pd.DataFrame):  # noqa: ANN201
    """Claim 4: leakage/interaction terms. Stub."""
    raise NotImplementedError(_STUB)
