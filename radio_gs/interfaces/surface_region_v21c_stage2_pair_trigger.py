"""Stricter pair-conflict authorization layered over frozen V2.1C Stage-I."""

from __future__ import annotations

from collections.abc import Mapping
import math
from statistics import median
from typing import Any

from radio_gs.scripts import (
    train_surface_region_v21c_two_stage_constrained_adamw as frozen_stage_i,
)


PAIR_DOT_THRESHOLD = -1e-12
MINIMUM_PAIR_CONFLICT_STEPS = 16
PAIR_COSINE_MEDIAN_MAXIMUM_EXCLUSIVE = 0.0


def validate_and_evaluate(value: object) -> dict[str, Any]:
    audit = frozen_stage_i.validate_stage_i_audit_result(value)
    if audit["trigger"]["stage_ii_authorized"] is not True:
        raise ValueError("V2.1C Stage-I formal trigger did not authorize Stage-II")
    pair_dots: list[float] = []
    pair_cosines: list[float] = []
    conflict_steps: list[int] = []
    for row in audit["history"]:
        candidate = row["training"]["adamw_candidate_evidence"]
        dots = candidate.get("gradient_dot_candidate")
        cosines = candidate.get("gradient_cosine_candidate")
        if not isinstance(dots, Mapping) or not isinstance(cosines, Mapping):
            raise ValueError("V2.1C Stage-I lacks pair directional evidence")
        dot = float(dots.get("pairwise"))
        cosine = float(cosines.get("pairwise"))
        if not math.isfinite(dot) or not math.isfinite(cosine) or not -1 <= cosine <= 1:
            raise ValueError("V2.1C Stage-I pair directional evidence is invalid")
        pair_dots.append(dot)
        pair_cosines.append(cosine)
        if dot < PAIR_DOT_THRESHOLD:
            conflict_steps.append(int(row["step"]))
    if len(pair_dots) != frozen_stage_i.OPTIMIZER_STEPS:
        raise ValueError("V2.1C Stage-I pair evidence count differs")
    cosine_median = float(median(pair_cosines))
    checks = {
        "frozen_stage_i_formally_authorized": True,
        "pair_conflict_steps_at_least_16": (
            len(conflict_steps) >= MINIMUM_PAIR_CONFLICT_STEPS
        ),
        "pair_candidate_cosine_median_strictly_negative": (
            cosine_median < PAIR_COSINE_MEDIAN_MAXIMUM_EXCLUSIVE
        ),
    }
    return {
        "audited_steps": len(pair_dots),
        "pair_dot_threshold": PAIR_DOT_THRESHOLD,
        "minimum_pair_conflict_steps": MINIMUM_PAIR_CONFLICT_STEPS,
        "pair_conflict_steps": conflict_steps,
        "pair_conflict_step_count": len(conflict_steps),
        "pair_candidate_cosine_median": cosine_median,
        "pair_candidate_cosine_median_maximum_exclusive": (
            PAIR_COSINE_MEDIAN_MAXIMUM_EXCLUSIVE
        ),
        "checks": checks,
        "stage_ii_authorized": all(checks.values()),
    }


def require_authorized(value: object) -> dict[str, Any]:
    result = validate_and_evaluate(value)
    if result["stage_ii_authorized"] is not True:
        raise ValueError(
            "V2.1C Stage-II requires pair conflict on >=16 steps and "
            "strictly negative median pair candidate cosine"
        )
    return result


__all__ = [
    "MINIMUM_PAIR_CONFLICT_STEPS",
    "PAIR_COSINE_MEDIAN_MAXIMUM_EXCLUSIVE",
    "PAIR_DOT_THRESHOLD",
    "require_authorized",
    "validate_and_evaluate",
]
