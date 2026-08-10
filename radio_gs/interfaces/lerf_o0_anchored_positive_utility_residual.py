"""Pure O0-anchored positive-utility region residual readout.

The exact frozen O0 primitive logits remain the canonical capability carrier.
This interface accepts a source-calibrated conservative edge-confidence lower
score and may add only a bounded non-negative residual inside at most three
eligible region cores.  It exposes no target label, mask, renderer, metric, or
query-conditioned parameter entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.lerf_o0_anchored_positive_utility_residual.v1"
INTERNAL_GAIN_THRESHOLDS = (0.0, 0.0, 0.0)
MAXIMUM_REGIONS = 3


def residual_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "canonical_capability": "frozen_o0_primitive_logit",
        "region_semantics": (
            "source_calibrated_conservative_edge_confidence_lower_score"
        ),
        "signed_utility": "positive_part_of_two_times_lower_score_minus_one",
        "gain": "signed_utility_times_unique_valid_marginal_mass_over_source_reference",
        "selection": "strict_gain_above_zero",
        "internal_gain_thresholds": list(INTERNAL_GAIN_THRESHOLDS),
        "maximum_regions": MAXIMUM_REGIONS,
        "residual": "epsilon_logit_times_signed_utility",
        "fusion": "o0_logit_plus_nonnegative_bounded_pointwise_max_residual",
        "query_gate": "reliability_and_feature_ood_and_anchor_agreement_and_stability",
        "fallback": (
            "failed_query_gate_or_no_selected_region_or_invalid_primitive_or_"
            "region_union_outside_is_bitwise_o0"
        ),
        "tie_break": "lower_canonical_region_index",
        "query_conditioned_parameters": False,
        "scene_conditioned_parameters": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(residual_contract())


@dataclass(frozen=True)
class SourceFixedPositiveUtilityConfig:
    """Numerical boundaries frozen once from source-only evidence."""

    epsilon_logit: float
    novel_mass_reference: float
    minimum_reliability: float
    maximum_feature_ood_score: float
    minimum_anchor_agreement: float
    minimum_stability: float

    def __post_init__(self) -> None:
        finite = (
            self.epsilon_logit,
            self.novel_mass_reference,
            self.minimum_reliability,
            self.maximum_feature_ood_score,
            self.minimum_anchor_agreement,
            self.minimum_stability,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("source-fixed positive-utility config must be finite")
        if float(self.epsilon_logit) < 0.0:
            raise ValueError("epsilon_logit must be non-negative")
        if float(self.novel_mass_reference) <= 0.0:
            raise ValueError("novel_mass_reference must be positive")
        unit_values = (
            self.minimum_reliability,
            self.maximum_feature_ood_score,
            self.minimum_anchor_agreement,
            self.minimum_stability,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in unit_values):
            raise ValueError("source-fixed gate thresholds must lie in [0,1]")


@dataclass(frozen=True)
class O0AnchoredPositiveUtilityResidualResult:
    fused_logits: torch.Tensor
    residual_logits: torch.Tensor
    query_gate: torch.Tensor
    selected_region_rows: tuple[tuple[int, ...], ...]
    selected_canonical_region_indices: tuple[tuple[int, ...], ...]
    selected_lower_scores: tuple[tuple[float, ...], ...]
    selected_gains: tuple[tuple[float, ...], ...]
    selected_marginal_primitives: tuple[tuple[int, ...], ...]


def _unit_interval_vector(value: object, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().cpu().contiguous()
    if (
        tensor.ndim != 1
        or tensor.numel() <= 0
        or not bool(torch.isfinite(tensor).all())
        or bool((tensor < 0.0).any())
        or bool((tensor > 1.0).any())
    ):
        raise ValueError(f"{label} must be a finite [Q] vector in [0,1]")
    return tensor


def source_fixed_query_gate(
    *,
    reliability: torch.Tensor,
    feature_ood_score: torch.Tensor,
    anchor_agreement: torch.Tensor,
    stability: torch.Tensor,
    config: SourceFixedPositiveUtilityConfig,
) -> torch.Tensor:
    """Evaluate the four source-fixed query validity diagnostics."""

    if not isinstance(config, SourceFixedPositiveUtilityConfig):
        raise TypeError("config must be SourceFixedPositiveUtilityConfig")
    values = {
        "reliability": _unit_interval_vector(reliability, label="reliability"),
        "feature_ood_score": _unit_interval_vector(
            feature_ood_score, label="feature_ood_score"
        ),
        "anchor_agreement": _unit_interval_vector(
            anchor_agreement, label="anchor_agreement"
        ),
        "stability": _unit_interval_vector(stability, label="stability"),
    }
    if len({tuple(value.shape) for value in values.values()}) != 1:
        raise ValueError("source-fixed query gate diagnostic axes differ")
    return (
        (values["reliability"] >= float(config.minimum_reliability))
        & (
            values["feature_ood_score"]
            <= float(config.maximum_feature_ood_score)
        )
        & (
            values["anchor_agreement"]
            >= float(config.minimum_anchor_agreement)
        )
        & (values["stability"] >= float(config.minimum_stability))
    ).bool().cpu().contiguous()


def _validated_internal_gain_thresholds() -> tuple[float, float, float]:
    thresholds = tuple(float(value) for value in INTERNAL_GAIN_THRESHOLDS)
    if (
        len(thresholds) != MAXIMUM_REGIONS
        or MAXIMUM_REGIONS != 3
        or any(not math.isfinite(value) or value != 0.0 for value in thresholds)
    ):
        raise RuntimeError("positive-utility internal gain policy must be exactly three zeros")
    return thresholds  # type: ignore[return-value]


def _validated_residual_inputs(
    *,
    o0_logits: torch.Tensor,
    region_confidence_lower: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_eligible_mask: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    query_gate: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    o0 = torch.as_tensor(o0_logits).detach()
    lower = torch.as_tensor(region_confidence_lower).detach()
    rows = torch.as_tensor(region_rows).detach()
    core = torch.as_tensor(core_mask).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    eligible = torch.as_tensor(region_eligible_mask).detach()
    canonical = torch.as_tensor(canonical_region_indices).detach()
    gate = torch.as_tensor(query_gate).detach()
    if (
        o0.dtype != torch.float32
        or o0.device.type != "cpu"
        or o0.ndim != 2
        or min(o0.shape) <= 0
        or not bool(torch.isfinite(o0).all())
    ):
        raise ValueError("O0 logits must be finite CPU float32 [N,Q]")
    primitive_count, query_count = o0.shape
    if (
        lower.dtype != torch.float32
        or lower.device.type != "cpu"
        or lower.ndim != 2
        or lower.shape[1] != query_count
        or lower.shape[0] <= 0
        or not bool(torch.isfinite(lower).all())
        or bool((lower < 0.0).any())
        or bool((lower > 1.0).any())
    ):
        raise ValueError(
            "region confidence lower scores must be CPU float32 [R,Q] in [0,1]"
        )
    region_count = int(lower.shape[0])
    if (
        rows.device.type != "cpu"
        or rows.dtype not in {torch.int32, torch.int64}
        or rows.ndim != 2
        or rows.shape[0] != region_count
        or core.device.type != "cpu"
        or core.dtype != torch.bool
        or core.shape != rows.shape
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (primitive_count,)
        or eligible.device.type != "cpu"
        or eligible.dtype != torch.bool
        or eligible.shape != (region_count,)
        or canonical.device.type != "cpu"
        or canonical.dtype != torch.long
        or canonical.shape != (region_count,)
        or bool((canonical < 0).any())
        or int(torch.unique(canonical).numel()) != region_count
        or gate.device.type != "cpu"
        or gate.dtype != torch.bool
        or gate.shape != (query_count,)
    ):
        raise ValueError("O0 positive-utility residual canonical axes differ")
    active_rows = rows[core]
    if (
        active_rows.numel() <= 0
        or bool((active_rows < 0).any())
        or bool((active_rows >= primitive_count).any())
        or not bool(core.any(dim=1).all())
    ):
        raise ValueError("region semantic cores differ")
    for index in range(region_count):
        active = rows[index, core[index]].long()
        if int(torch.unique(active).numel()) != int(active.numel()):
            raise ValueError("region semantic core contains duplicate primitives")
    return (
        o0.contiguous(),
        lower.contiguous(),
        rows.long().contiguous(),
        core.contiguous(),
        valid.contiguous(),
        eligible.contiguous(),
        canonical.contiguous(),
        gate.contiguous(),
    )


def o0_anchored_positive_utility_residual(
    *,
    o0_logits: torch.Tensor,
    region_confidence_lower: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_eligible_mask: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    query_gate: torch.Tensor,
    config: SourceFixedPositiveUtilityConfig,
) -> O0AnchoredPositiveUtilityResidualResult:
    """Greedily select at most three strictly positive-utility regions."""

    if not isinstance(config, SourceFixedPositiveUtilityConfig):
        raise TypeError("config must be SourceFixedPositiveUtilityConfig")
    thresholds = _validated_internal_gain_thresholds()
    (
        o0,
        lower,
        rows,
        core,
        valid,
        eligible,
        canonical,
        gate,
    ) = _validated_residual_inputs(
        o0_logits=o0_logits,
        region_confidence_lower=region_confidence_lower,
        region_rows=region_rows,
        core_mask=core_mask,
        primitive_valid_mask=primitive_valid_mask,
        region_eligible_mask=region_eligible_mask,
        canonical_region_indices=canonical_region_indices,
        query_gate=query_gate,
    )
    primitive_count, query_count = o0.shape
    region_count = int(lower.shape[0])
    safe_rows = rows.clamp(min=0, max=primitive_count - 1)
    valid_core = core & valid[safe_rows]
    usable_region = eligible & valid_core.any(dim=1)
    signed_utility = (2.0 * lower - 1.0).clamp_min(0.0)
    residual = torch.zeros_like(o0)
    row_selections: list[tuple[int, ...]] = []
    canonical_selections: list[tuple[int, ...]] = []
    lower_selections: list[tuple[float, ...]] = []
    gain_selections: list[tuple[float, ...]] = []
    marginal_selections: list[tuple[int, ...]] = []

    for query in range(query_count):
        if not bool(gate[query]):
            row_selections.append(())
            canonical_selections.append(())
            lower_selections.append(())
            gain_selections.append(())
            marginal_selections.append(())
            continue
        covered = torch.zeros(primitive_count, dtype=torch.bool)
        selected = torch.zeros(region_count, dtype=torch.bool)
        chosen_rows: list[int] = []
        chosen_canonical: list[int] = []
        chosen_lower: list[float] = []
        chosen_gains: list[float] = []
        chosen_marginals: list[int] = []
        for threshold in thresholds:
            novel = valid_core & ~covered[safe_rows]
            marginal = novel.sum(dim=1)
            gains = signed_utility[:, query] * (
                marginal.float() / float(config.novel_mass_reference)
            )
            candidate = usable_region & ~selected & (marginal > 0)
            gains = gains.masked_fill(~candidate, -1.0)
            best_gain = float(gains.max())
            if not best_gain > threshold:
                break
            tied = candidate & (gains == best_gain)
            tied_canonical = canonical.masked_fill(
                ~tied, torch.iinfo(torch.long).max
            )
            best_canonical = int(tied_canonical.min())
            best_row = int(torch.nonzero(tied & (canonical == best_canonical))[0])
            selected[best_row] = True
            active = safe_rows[best_row, valid_core[best_row]].long()
            covered[active] = True
            residual_value = float(config.epsilon_logit) * float(
                signed_utility[best_row, query]
            )
            if residual_value > 0.0:
                residual[active, query] = torch.maximum(
                    residual[active, query],
                    torch.full_like(residual[active, query], residual_value),
                )
            chosen_rows.append(best_row)
            chosen_canonical.append(best_canonical)
            chosen_lower.append(float(lower[best_row, query]))
            chosen_gains.append(best_gain)
            chosen_marginals.append(int(marginal[best_row]))
        row_selections.append(tuple(chosen_rows))
        canonical_selections.append(tuple(chosen_canonical))
        lower_selections.append(tuple(chosen_lower))
        gain_selections.append(tuple(chosen_gains))
        marginal_selections.append(tuple(chosen_marginals))

    fused = o0.clone()
    update = residual > 0.0
    fused[update] = o0[update] + residual[update]
    selected_any = tuple(bool(rows_) for rows_ in row_selections)
    for query, has_selection in enumerate(selected_any):
        if not has_selection and not torch.equal(fused[:, query], o0[:, query]):
            raise RuntimeError("empty selection changed O0")
    if (
        bool((residual < 0.0).any())
        or float(residual.max()) > float(config.epsilon_logit) + 1e-7
        or any(len(rows_) > MAXIMUM_REGIONS for rows_ in row_selections)
        or not torch.equal(fused[~update], o0[~update])
        or not torch.equal(fused[:, ~gate], o0[:, ~gate])
        or not torch.equal(residual[~valid], torch.zeros_like(residual[~valid]))
    ):
        raise RuntimeError("O0 positive-utility capability invariant failed")
    return O0AnchoredPositiveUtilityResidualResult(
        fused_logits=fused.contiguous(),
        residual_logits=residual.contiguous(),
        query_gate=gate,
        selected_region_rows=tuple(row_selections),
        selected_canonical_region_indices=tuple(canonical_selections),
        selected_lower_scores=tuple(lower_selections),
        selected_gains=tuple(gain_selections),
        selected_marginal_primitives=tuple(marginal_selections),
    )


__all__ = [
    "CONTRACT_SHA256",
    "INTERNAL_GAIN_THRESHOLDS",
    "MAXIMUM_REGIONS",
    "O0AnchoredPositiveUtilityResidualResult",
    "SCHEMA",
    "SourceFixedPositiveUtilityConfig",
    "o0_anchored_positive_utility_residual",
    "residual_contract",
    "source_fixed_query_gate",
]
