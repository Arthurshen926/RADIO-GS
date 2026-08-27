"""Pre-registered v3 promotion gates; benchmark results never tune them."""

from __future__ import annotations

from dataclasses import dataclass

from radio_gs.v3.evaluation.source_heldout import SourceHeldoutMetrics


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityMetric:
    value: float
    higher_is_better: bool
    tolerance: float


def capability_pareto_gate(
    baseline: dict[str, CapabilityMetric],
    candidate: dict[str, float],
) -> GateDecision:
    """Require every real source capability to remain within its tolerance."""

    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("capability gate cohorts differ")
    failures = []
    for name in sorted(baseline):
        before = baseline[name]
        after = float(candidate[name])
        change = after - before.value
        regression = -change if before.higher_is_better else change
        if regression > before.tolerance:
            failures.append(
                f"{name}: capability regression {regression:.8g} exceeds tolerance {before.tolerance:.8g}"
            )
    return GateDecision(not failures, tuple(failures))


def source_heldout_gate(
    baseline: dict[str, SourceHeldoutMetrics],
    candidate: dict[str, SourceHeldoutMetrics],
    *,
    minimum_macro_iou_gain: float = 0.05,
) -> GateDecision:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("source-heldout scene cohorts differ")
    failures: list[str] = []
    gains = [candidate[name].mask_iou - baseline[name].mask_iou for name in sorted(baseline)]
    if sum(gains) / len(gains) < minimum_macro_iou_gain:
        failures.append("scene-macro mask IoU gain below +0.05")
    for name in sorted(baseline):
        before, after = baseline[name], candidate[name]
        if after.mask_iou < before.mask_iou:
            failures.append(f"{name}: mask IoU regressed")
        if after.brier >= before.brier:
            failures.append(f"{name}: Brier did not decrease")
        if after.boundary_f <= before.boundary_f:
            failures.append(f"{name}: boundary F did not increase")
        if after.unknown_fp_mass > before.unknown_fp_mass:
            failures.append(f"{name}: unknown FP mass increased")
    return GateDecision(not failures, tuple(failures))


__all__ = [
    "CapabilityMetric",
    "GateDecision",
    "capability_pareto_gate",
    "source_heldout_gate",
]
