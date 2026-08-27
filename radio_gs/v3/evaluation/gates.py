"""Pre-registered v3 promotion gates; benchmark results never tune them."""

from __future__ import annotations

from dataclasses import dataclass

from radio_gs.v3.evaluation.source_heldout import SourceHeldoutMetrics


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failures: tuple[str, ...]


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


__all__ = ["GateDecision", "source_heldout_gate"]
