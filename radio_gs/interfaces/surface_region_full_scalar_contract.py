"""Source-only full-scalar overlay for the accepted surface-region base.

This module is deliberately an interface layer.  It neither changes the
accepted base readout nor defines a trainable model.  The exact-marginal
factorized state is available only on the intersection with accepted-base
support; every other support case has an explicit conservative route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

import torch

from radio_gs.training.factorized_radio_cache import (
    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
)

from .factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA,
    FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION,
    FactorizedPrimitiveState,
    validate_factorized_primitive_state_payload,
)


SURFACE_REGION_FULL_SCALAR_SCHEMA = "radio_gs.surface_region_full_scalar.v1"
SURFACE_REGION_FULL_SCALAR_SCHEMA_VERSION = 1
SURFACE_REGION_FULL_SCALAR_DIM = 18
SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA = (
    "radio_gs.surface_region_full_scalar_normalization.v1"
)
SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA_VERSION = 1
_MAD_NORMAL_CONSISTENCY = 1.4826
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


SURFACE_REGION_FULL_SCALAR_NAMES = tuple(
    f"{statistic}_{scalar}"
    for statistic in (
        "anchor",
        "legacy_reliability_weighted_mean",
        "legacy_reliability_weighted_std",
    )
    for scalar in FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES
)
SURFACE_REGION_FULL_SCALAR_NAMES_SHA256 = _canonical_json_sha256(
    list(SURFACE_REGION_FULL_SCALAR_NAMES)
)


def surface_region_full_scalar_contract() -> dict[str, Any]:
    """Return the immutable source-only overlay and routing contract."""

    return {
        "schema": SURFACE_REGION_FULL_SCALAR_SCHEMA,
        "schema_version": SURFACE_REGION_FULL_SCALAR_SCHEMA_VERSION,
        "factorized_primitive_state_schema": FACTORIZED_PRIMITIVE_STATE_SCHEMA,
        "factorized_primitive_state_schema_version": (
            FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
        ),
        "factorized_primitive_state_contract_sha256": (
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        ),
        "factorized_primitive_state_scalar_names": list(
            FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES
        ),
        "factorized_primitive_state_scalar_names_sha256": (
            FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256
        ),
        "required_visibility_purity_authority": dict(
            FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
        ),
        "support_routing": {
            "full_scalar": "accepted_base_valid_and_exact_state_valid",
            "base_only": "accepted_base_valid_and_not_exact_state_valid_fallback_to_accepted_base",
            "exact_only": "exact_state_valid_and_not_accepted_base_valid_abstain",
            "neither": "abstain",
        },
        "summary": {
            "dimension": SURFACE_REGION_FULL_SCALAR_DIM,
            "names": list(SURFACE_REGION_FULL_SCALAR_NAMES),
            "names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
            "parts": [
                "anchor_six_scalars",
                "legacy_geometric_reliability_weighted_population_mean_six_scalars",
                "legacy_geometric_reliability_weighted_population_std_six_scalars",
            ],
            "weight_authority": (
                "FactorizedPrimitiveState.legacy_geometric_reliability"
            ),
        },
        "padding_policy": (
            "token_mask_false_is_exact_zero_and_never_gathered_or_aggregated"
        ),
        "normalization": {
            "schema": SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA,
            "source": "source_train_full_scalar_routes_only",
            "location": "coordinatewise_lower_median",
            "dispersion": "coordinatewise_lower_median_absolute_deviation",
            "positive_mad_scale": "1.4826_times_mad",
            "zero_mad_nonconstant_scale": "one",
        },
        "ood_rule": {
            "score": "linf_absolute_source_robust_normalized_coordinate",
            "threshold": "maximum_source_train_score",
            "constant_coordinate": "any_exact_deviation_is_ood",
            "comparison": "strictly_greater_than_source_maximum",
            "action": "fall_back_to_accepted_base_never_enable_exact_only",
        },
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256 = _canonical_json_sha256(
    surface_region_full_scalar_contract()
)


@dataclass(frozen=True)
class FullScalarSupportRouting:
    overlap_mask: torch.Tensor
    base_only_fallback_mask: torch.Tensor
    exact_only_abstain_mask: torch.Tensor
    neither_abstain_mask: torch.Tensor


@dataclass(frozen=True)
class FullScalarRegionSummary:
    summary: torch.Tensor
    token_scalars: torch.Tensor
    token_overlap_mask: torch.Tensor
    token_base_only_mask: torch.Tensor
    token_exact_only_mask: torch.Tensor
    use_full_scalar_mask: torch.Tensor
    base_fallback_mask: torch.Tensor
    abstain_mask: torch.Tensor


@dataclass(frozen=True)
class FullScalarNormalizationResult:
    normalized: torch.Tensor
    ood_score: torch.Tensor
    ood_mask: torch.Tensor
    use_full_scalar_mask: torch.Tensor
    base_fallback_mask: torch.Tensor


def _validate_exact_state(state: FactorizedPrimitiveState) -> None:
    if not isinstance(state, FactorizedPrimitiveState):
        raise TypeError("full-scalar overlay requires FactorizedPrimitiveState")
    validate_factorized_primitive_state_payload(state.to_payload())
    if (
        state.schema != FACTORIZED_PRIMITIVE_STATE_SCHEMA
        or state.schema_version != FACTORIZED_PRIMITIVE_STATE_SCHEMA_VERSION
        or state.contract_sha256 != FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
    ):
        raise ValueError("full-scalar overlay requires primitive-state schema v2")
    registration_sha256 = str(
        state.metadata.get("registration_responsibility_cache_sha256", "")
    )
    expected_authority = {
        **FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY,
        "registration_responsibility_cache_sha256": registration_sha256,
    }
    if (
        _SHA256.fullmatch(registration_sha256) is None
        or state.metadata.get("visibility_purity_authority") != expected_authority
        or not bool(state.visibility_purity_known.all())
    ):
        raise ValueError(
            "full-scalar overlay requires exact-marginal measured-purity authority"
        )


def build_full_scalar_support_routing(
    accepted_base_valid: torch.Tensor,
    state: FactorizedPrimitiveState,
) -> FullScalarSupportRouting:
    """Partition the global domain without silently expanding either support."""

    _validate_exact_state(state)
    base = torch.as_tensor(accepted_base_valid).detach().cpu()
    if base.dtype != torch.bool or base.shape != state.valid.shape:
        raise ValueError("accepted-base validity must align with exact state")
    exact = state.valid.bool().cpu()
    overlap = base & exact
    base_only = base & ~exact
    exact_only = exact & ~base
    neither = ~base & ~exact
    if not torch.equal(
        overlap.to(torch.int8)
        + base_only.to(torch.int8)
        + exact_only.to(torch.int8)
        + neither.to(torch.int8),
        torch.ones_like(base, dtype=torch.int8),
    ):
        raise RuntimeError("full-scalar support routing is not a partition")
    return FullScalarSupportRouting(overlap, base_only, exact_only, neither)


def aggregate_surface_region_full_scalars(
    state: FactorizedPrimitiveState,
    accepted_base_valid: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor | int,
) -> FullScalarRegionSummary:
    """Aggregate the 6-D exact state into an 18-D overlap-only summary."""

    routing = build_full_scalar_support_routing(accepted_base_valid, state)
    raw_rows = torch.as_tensor(region_rows).detach().cpu()
    raw_mask = torch.as_tensor(token_mask).detach().cpu()
    squeeze = raw_rows.ndim == 1
    if squeeze:
        raw_rows = raw_rows[None]
        raw_mask = raw_mask[None]
    if (
        raw_rows.ndim != 2
        or raw_rows.dtype != torch.long
        or raw_mask.dtype != torch.bool
        or raw_mask.shape != raw_rows.shape
        or raw_rows.shape[1] <= 0
    ):
        raise ValueError("region rows/mask must align as long/bool [B,T]")
    batch_size, width = raw_rows.shape
    if bool((raw_mask.sum(dim=1) <= 0).any()):
        raise ValueError("every full-scalar region requires an active token")
    active_rows = raw_rows[raw_mask]
    if bool((active_rows < 0).any()) or bool((active_rows >= state.valid.numel()).any()):
        raise ValueError("active region row is outside primitive support")
    sorted_rows = raw_rows.masked_fill(~raw_mask, state.valid.numel()).sort(
        dim=1
    ).values
    duplicate_active = (sorted_rows[:, 1:] == sorted_rows[:, :-1]) & (
        sorted_rows[:, 1:] < state.valid.numel()
    )
    if bool(duplicate_active.any()):
        raise ValueError("active region rows must be unique")
    anchor = torch.as_tensor(anchor_index).detach().long().cpu().reshape(-1)
    if anchor.numel() == 1:
        anchor = anchor.expand(batch_size)
    if (
        anchor.shape != (batch_size,)
        or bool((anchor < 0).any())
        or bool((anchor >= width).any())
        or not bool(raw_mask[torch.arange(batch_size), anchor].all())
    ):
        raise ValueError("anchor_index must identify one active token per region")

    # Padding is never used as an index.  Canonical row zero here is only a
    # safe gather placeholder and is overwritten with exact zeros below.
    safe_rows = torch.where(raw_mask, raw_rows, torch.zeros_like(raw_rows))
    token_overlap = raw_mask & routing.overlap_mask[safe_rows]
    token_base_only = raw_mask & routing.base_only_fallback_mask[safe_rows]
    token_exact_only = raw_mask & routing.exact_only_abstain_mask[safe_rows]
    anchor_rows = safe_rows[torch.arange(batch_size), anchor]
    use_full = routing.overlap_mask[anchor_rows]
    base_fallback = routing.base_only_fallback_mask[anchor_rows]
    abstain = routing.exact_only_abstain_mask[anchor_rows] | routing.neither_abstain_mask[
        anchor_rows
    ]

    global_to_compact = torch.full(
        (state.valid.numel(),), -1, dtype=torch.long
    )
    global_to_compact[state.global_rows] = torch.arange(state.global_rows.numel())
    compact_rows = global_to_compact[safe_rows]
    safe_compact = compact_rows.clamp_min(0)
    scalar_source = state.scalar_encoding_input().float().cpu()
    reliability_source = state.legacy_geometric_reliability().float().cpu()
    token_scalars = scalar_source[safe_compact]
    token_scalars = token_scalars.masked_fill(~token_overlap[..., None], 0.0)
    token_weights = reliability_source[safe_compact].masked_fill(~token_overlap, 0.0)
    if not bool(torch.isfinite(token_scalars).all()) or not bool(
        torch.isfinite(token_weights).all()
    ):
        raise ValueError("full-scalar state contains non-finite aggregation values")

    total = token_weights.sum(dim=1, keepdim=True)
    if bool((use_full & (total[:, 0] <= 0)).any()):
        raise ValueError("overlap route lacks positive legacy reliability")
    safe_total = total.clamp_min(torch.finfo(torch.float32).tiny)
    mean = (token_scalars * token_weights[..., None]).sum(dim=1) / safe_total
    variance = (
        (token_scalars - mean[:, None]).square()
        * token_weights[..., None]
    ).sum(dim=1) / safe_total
    anchor_values = token_scalars[
        torch.arange(batch_size), anchor
    ]
    summary = torch.cat(
        (anchor_values, mean, variance.clamp_min(0).sqrt()), dim=1
    ).masked_fill(~use_full[:, None], 0.0)
    if bool(summary[~use_full].ne(0).any()) or bool(
        token_scalars[~raw_mask].ne(0).any()
    ):
        raise RuntimeError("full-scalar fallback/padding leaked state values")

    result = FullScalarRegionSummary(
        summary=summary,
        token_scalars=token_scalars,
        token_overlap_mask=token_overlap,
        token_base_only_mask=token_base_only,
        token_exact_only_mask=token_exact_only,
        use_full_scalar_mask=use_full,
        base_fallback_mask=base_fallback,
        abstain_mask=abstain,
    )
    if not squeeze:
        return result
    return FullScalarRegionSummary(
        summary=result.summary[0],
        token_scalars=result.token_scalars[0],
        token_overlap_mask=result.token_overlap_mask[0],
        token_base_only_mask=result.token_base_only_mask[0],
        token_exact_only_mask=result.token_exact_only_mask[0],
        use_full_scalar_mask=result.use_full_scalar_mask[0],
        base_fallback_mask=result.base_fallback_mask[0],
        abstain_mask=result.abstain_mask[0],
    )


def _float32_sha256(values: torch.Tensor) -> str:
    array = (
        torch.as_tensor(values).detach().float().cpu().contiguous().numpy().astype(
            "<f4", copy=False
        )
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _bool_sha256(values: torch.Tensor) -> str:
    array = torch.as_tensor(values).detach().bool().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_full_scalar_normalization_authority(
    source_train_summaries: torch.Tensor,
    source_train_mask: torch.Tensor,
    *,
    source_state_cohort_sha256: str,
) -> dict[str, Any]:
    """Freeze cross-scene source-train normalization and its OOD envelope.

    The SHA binds a canonical manifest of all scene-disjoint source states,
    never an arbitrary single scene and never the deployment target state.
    """

    values = torch.as_tensor(source_train_summaries).detach().float().cpu()
    mask = torch.as_tensor(source_train_mask).detach().cpu()
    if (
        values.ndim != 2
        or values.shape[1] != SURFACE_REGION_FULL_SCALAR_DIM
        or mask.dtype != torch.bool
        or mask.shape != (values.shape[0],)
        or int(mask.sum()) < 2
        or not bool(torch.isfinite(values).all())
        or _SHA256.fullmatch(str(source_state_cohort_sha256)) is None
    ):
        raise ValueError("source-train full-scalar normalization inputs differ")
    selected = values[mask]
    median = selected.median(dim=0).values
    absolute_deviation = (selected - median).abs()
    mad = absolute_deviation.median(dim=0).values
    minimum = selected.min(dim=0).values
    maximum = selected.max(dim=0).values
    constant = minimum == maximum
    positive_mad = mad > 0
    robust_scale = torch.where(
        positive_mad,
        mad * _MAD_NORMAL_CONSISTENCY,
        torch.ones_like(mad),
    )
    normalized_source = (selected - median) / robust_scale
    variable = ~constant
    source_score = (
        normalized_source[:, variable].abs().amax(dim=1)
        if bool(variable.any())
        else torch.zeros(selected.shape[0])
    )
    return {
        "schema": SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA,
        "schema_version": SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA_VERSION,
        "full_scalar_contract_sha256": SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256,
        "factorized_primitive_state_contract_sha256": (
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        ),
        "source_state_cohort_sha256": str(source_state_cohort_sha256),
        "summary_names": list(SURFACE_REGION_FULL_SCALAR_NAMES),
        "summary_names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
        "source_summary_sha256": _float32_sha256(values),
        "source_selection_mask_sha256": _bool_sha256(mask),
        "source_count": int(mask.sum()),
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "constant_coordinate_mask": constant,
        "source_max_robust_linf": float(source_score.max()),
        "ood_rule": surface_region_full_scalar_contract()["ood_rule"],
        "source_access": {
            "split": "source_train",
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }


def validate_full_scalar_normalization_authority(value: object) -> dict[str, Any]:
    """Validate a normalization authority without trusting descriptive fields."""

    if not isinstance(value, Mapping):
        raise ValueError("full-scalar normalization authority must be a mapping")
    authority = dict(value)
    required = {
        "schema", "schema_version", "full_scalar_contract_sha256",
        "factorized_primitive_state_contract_sha256",
        "source_state_cohort_sha256", "summary_names",
        "summary_names_sha256", "source_summary_sha256",
        "source_selection_mask_sha256", "source_count", "median", "mad",
        "robust_scale", "constant_coordinate_mask",
        "source_max_robust_linf", "ood_rule", "source_access",
    }
    if set(authority) != required:
        raise ValueError("full-scalar normalization authority fields differ")
    source_access = authority.get("source_access")
    expected_source_access = {
        "split": "source_train",
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    if (
        authority.get("schema") != SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA
        or authority.get("schema_version")
        != SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA_VERSION
        or authority.get("full_scalar_contract_sha256")
        != SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256
        or authority.get("factorized_primitive_state_contract_sha256")
        != FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        or authority.get("summary_names") != list(SURFACE_REGION_FULL_SCALAR_NAMES)
        or authority.get("summary_names_sha256")
        != SURFACE_REGION_FULL_SCALAR_NAMES_SHA256
        or authority.get("ood_rule")
        != surface_region_full_scalar_contract()["ood_rule"]
        or source_access != expected_source_access
        or int(authority.get("source_count", 0)) < 2
        or any(
            _SHA256.fullmatch(str(authority.get(name, ""))) is None
            for name in (
                "source_state_cohort_sha256",
                "source_summary_sha256",
                "source_selection_mask_sha256",
            )
        )
    ):
        raise ValueError("full-scalar normalization authority contract differs")
    median = authority.get("median")
    mad = authority.get("mad")
    scale = authority.get("robust_scale")
    constant = authority.get("constant_coordinate_mask")
    if (
        not torch.is_tensor(median) or median.dtype != torch.float32
        or median.shape != (SURFACE_REGION_FULL_SCALAR_DIM,)
        or not torch.is_tensor(mad) or mad.dtype != torch.float32
        or mad.shape != median.shape
        or not torch.is_tensor(scale) or scale.dtype != torch.float32
        or scale.shape != median.shape
        or not torch.is_tensor(constant) or constant.dtype != torch.bool
        or constant.shape != median.shape
        or not all(bool(torch.isfinite(item).all()) for item in (median, mad, scale))
        or bool((mad < 0).any()) or bool((scale <= 0).any())
        or not torch.equal(
            scale,
            torch.where(
                mad > 0,
                mad * _MAD_NORMAL_CONSISTENCY,
                torch.ones_like(mad),
            ),
        )
        or not isinstance(authority.get("source_max_robust_linf"), float)
        or not float(authority["source_max_robust_linf"]) >= 0.0
    ):
        raise ValueError("full-scalar normalization statistics differ")
    return authority


def apply_full_scalar_normalization(
    summaries: torch.Tensor,
    eligible_mask: torch.Tensor,
    authority: Mapping[str, Any],
) -> FullScalarNormalizationResult:
    """Normalize eligible overlap summaries and route OOD back to the base."""

    frozen = validate_full_scalar_normalization_authority(authority)
    values = torch.as_tensor(summaries).detach().float().cpu()
    eligible = torch.as_tensor(eligible_mask).detach().cpu()
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None]
        eligible = eligible.reshape(1)
    if (
        values.ndim != 2
        or values.shape[1] != SURFACE_REGION_FULL_SCALAR_DIM
        or eligible.dtype != torch.bool
        or eligible.shape != (values.shape[0],)
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("full-scalar normalization values/mask differ")
    median = frozen["median"]
    scale = frozen["robust_scale"]
    constant = frozen["constant_coordinate_mask"]
    normalized = (values - median) / scale
    normalized = normalized.masked_fill(constant[None], 0.0)
    normalized = normalized.masked_fill(~eligible[:, None], 0.0)
    variable = ~constant
    score = (
        ((values - median) / scale)[:, variable].abs().amax(dim=1)
        if bool(variable.any())
        else torch.zeros(values.shape[0])
    )
    constant_mismatch = (
        (values[:, constant] != median[constant]).any(dim=1)
        if bool(constant.any())
        else torch.zeros(values.shape[0], dtype=torch.bool)
    )
    ood = eligible & (
        constant_mismatch
        | (score > float(frozen["source_max_robust_linf"]))
    )
    use_full = eligible & ~ood
    normalized = normalized.masked_fill(~use_full[:, None], 0.0)
    score = score.masked_fill(~eligible, 0.0)
    result = FullScalarNormalizationResult(
        normalized=normalized,
        ood_score=score,
        ood_mask=ood,
        use_full_scalar_mask=use_full,
        base_fallback_mask=ood,
    )
    if not squeeze:
        return result
    return FullScalarNormalizationResult(
        normalized=result.normalized[0],
        ood_score=result.ood_score[0],
        ood_mask=result.ood_mask[0],
        use_full_scalar_mask=result.use_full_scalar_mask[0],
        base_fallback_mask=result.base_fallback_mask[0],
    )


__all__ = [
    "SURFACE_REGION_FULL_SCALAR_SCHEMA",
    "SURFACE_REGION_FULL_SCALAR_SCHEMA_VERSION",
    "SURFACE_REGION_FULL_SCALAR_DIM",
    "SURFACE_REGION_FULL_SCALAR_NAMES",
    "SURFACE_REGION_FULL_SCALAR_NAMES_SHA256",
    "SURFACE_REGION_FULL_SCALAR_CONTRACT_SHA256",
    "SURFACE_REGION_FULL_SCALAR_NORMALIZATION_SCHEMA",
    "FullScalarSupportRouting",
    "FullScalarRegionSummary",
    "FullScalarNormalizationResult",
    "surface_region_full_scalar_contract",
    "build_full_scalar_support_routing",
    "aggregate_surface_region_full_scalars",
    "build_full_scalar_normalization_authority",
    "validate_full_scalar_normalization_authority",
    "apply_full_scalar_normalization",
]
