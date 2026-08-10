"""Source-only labels for same-axis O0 missing-core completion proposals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


O0_SCORE_MINIMUM = 0.6
O0_CORE_SUPERMAJORITY = 0.75


@dataclass(frozen=True)
class MissingCoreConditionalUtility:
    valid_core_counts: torch.Tensor
    positive_fraction: torch.Tensor
    qualified_region_mask: torch.Tensor
    missing_counts: torch.Tensor
    unit_region_indices: torch.Tensor
    unit_query_indices: torch.Tensor
    unit_primitive_rows: torch.Tensor
    unit_o0_scores: torch.Tensor
    unit_hard_labels: torch.Tensor
    unit_soft_target_mass_fraction: torch.Tensor
    unit_signed_utility: torch.Tensor


def source_missing_core_conditional_utility(
    *,
    o0_scores: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_query_indices: torch.Tensor,
    region_dominant_instance_ids: torch.Tensor,
    primitive_instance_mass: torch.Tensor,
) -> MissingCoreConditionalUtility:
    """Label every missing unit of an independently generated O0 anchor.

    O0 and the region-to-query assignment are complete inputs: instance mass
    is read only after qualification and never influences the score, selected
    scale, positive fraction, or missing-core set.
    """

    scores = torch.as_tensor(o0_scores).detach()
    rows = torch.as_tensor(region_rows).detach()
    core = torch.as_tensor(core_mask).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    query = torch.as_tensor(region_query_indices).detach()
    instance = torch.as_tensor(region_dominant_instance_ids).detach()
    mass = torch.as_tensor(primitive_instance_mass).detach()
    if (
        scores.device.type != "cpu"
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or min(scores.shape) <= 0
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or rows.device.type != "cpu"
        or rows.dtype not in {torch.int32, torch.int64}
        or rows.ndim != 2
        or core.device.type != "cpu"
        or core.dtype != torch.bool
        or core.shape != rows.shape
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or query.device.type != "cpu"
        or query.dtype not in {torch.int32, torch.int64}
        or query.shape != (rows.shape[0],)
        or instance.device.type != "cpu"
        or instance.dtype not in {torch.int32, torch.int64}
        or instance.shape != query.shape
        or mass.device.type != "cpu"
        or not mass.is_floating_point()
        or mass.ndim != 2
        or mass.shape[0] != scores.shape[0]
        or not bool(torch.isfinite(mass).all())
        or bool((mass < 0.0).any())
        or bool((rows[core] < 0).any())
        or bool((rows[core] >= scores.shape[0]).any())
    ):
        raise ValueError("source missing-core conditional-utility inputs differ")
    has_query = query >= 0
    if bool((query[has_query] >= scores.shape[1]).any()):
        raise ValueError("source missing-core query index is out of range")
    has_instance = (instance > 0) & (instance < mass.shape[1])
    safe_rows = rows.long().clamp(min=0, max=scores.shape[0] - 1)
    valid_core = core & valid[safe_rows]
    counts = valid_core.sum(dim=1)
    safe_query = query.long().clamp(min=0, max=scores.shape[1] - 1)
    region_scores = scores[safe_rows, safe_query[:, None]]
    positive = (region_scores > O0_SCORE_MINIMUM) & valid_core
    fraction = positive.sum(dim=1).float() / counts.clamp_min(1).float()
    fraction[~has_query] = 0.0
    qualified = (
        has_query
        & has_instance
        & (counts > 0)
        & (fraction >= O0_CORE_SUPERMAJORITY)
    )
    missing = valid_core & (region_scores <= O0_SCORE_MINIMUM) & has_query[:, None]
    missing_counts = missing.sum(dim=1)
    qualified &= missing_counts > 0

    region_out: list[torch.Tensor] = []
    query_out: list[torch.Tensor] = []
    primitive_out: list[torch.Tensor] = []
    score_out: list[torch.Tensor] = []
    hard_out: list[torch.Tensor] = []
    soft_out: list[torch.Tensor] = []
    for region in torch.where(qualified)[0].tolist():
        primitive = safe_rows[region, missing[region]]
        target = int(instance[region])
        selected_mass = mass[primitive].float()
        total = selected_mass.sum(dim=1)
        evaluable = total > 0.0
        if not bool(evaluable.any()):
            continue
        primitive = primitive[evaluable]
        selected_mass = selected_mass[evaluable]
        total = total[evaluable]
        target_fraction = selected_mass[:, target] / total
        hard = selected_mass.argmax(dim=1) == target
        count = int(primitive.numel())
        region_out.append(torch.full((count,), region, dtype=torch.long))
        query_out.append(
            torch.full((count,), int(query[region]), dtype=torch.long)
        )
        primitive_out.append(primitive.long())
        score_out.append(scores[primitive, int(query[region])])
        hard_out.append(hard.bool())
        soft_out.append(target_fraction.float())

    def _cat(values: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
        return (
            torch.cat(values).to(dtype=dtype).contiguous()
            if values
            else torch.empty(0, dtype=dtype)
        )

    soft = _cat(soft_out, dtype=torch.float32)
    return MissingCoreConditionalUtility(
        valid_core_counts=counts.long().contiguous(),
        positive_fraction=fraction.float().contiguous(),
        qualified_region_mask=qualified.bool().contiguous(),
        missing_counts=missing_counts.long().contiguous(),
        unit_region_indices=_cat(region_out, dtype=torch.long),
        unit_query_indices=_cat(query_out, dtype=torch.long),
        unit_primitive_rows=_cat(primitive_out, dtype=torch.long),
        unit_o0_scores=_cat(score_out, dtype=torch.float32),
        unit_hard_labels=_cat(hard_out, dtype=torch.bool),
        unit_soft_target_mass_fraction=soft,
        unit_signed_utility=(2.0 * soft - 1.0).contiguous(),
    )


__all__ = [
    "MissingCoreConditionalUtility",
    "O0_CORE_SUPERMAJORITY",
    "O0_SCORE_MINIMUM",
    "source_missing_core_conditional_utility",
]
