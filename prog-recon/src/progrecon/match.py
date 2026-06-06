"""Output-matching primitives shared by the harness TEST step and the scorer.

Kept free of any train/test policy so both the harness (which compares against a
*train* validation slice) and ``scoring.equivalence`` (which compares against a
*test* split, and adds the firewall assertion) can reuse the same numeric and
categorical comparison rules.
"""

from __future__ import annotations

import math
from typing import Any

from .types import Schema


def value_match(pred: Any, true: Any, *, rel_tol: float) -> bool:
    """Numeric within rel_tol; bool/categorical exact."""
    if isinstance(true, bool) or isinstance(pred, bool):
        return pred == true
    if isinstance(true, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(pred), float(true), rel_tol=rel_tol, abs_tol=1e-12)
    return pred == true


def row_match(y_pred: Any, y_true: dict[str, Any], schema: Schema, *, rel_tol: float) -> bool:
    """A row passes iff ALL declared output fields match."""
    if not isinstance(y_pred, dict):
        return False
    for f in schema.outputs:
        if f.name not in y_pred:
            return False
        if not value_match(y_pred[f.name], y_true[f.name], rel_tol=rel_tol):
            return False
    return True
