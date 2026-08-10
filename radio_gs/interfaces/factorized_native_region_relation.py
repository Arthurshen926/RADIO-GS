"""Query-free region relation features in the factorized RADIO gauge.

This module is an opt-in sibling of the frozen RegionCoMembershipV2 feature
contract.  It never reconstructs ``exp(log_amplitude) * direction`` and never
opens a query.  Instead, it exposes the native spherical direction relation,
the separate amplitude relation, and explicit observation-state availability.

The output can be appended to a future source-trained relation head.  Merely
importing this module does not alter AcceptedV2 or RegionCoMembershipV2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


INTERFACE_SCHEMA = "radio_gs.factorized_native_region_relation.v1"
INTERFACE_SCHEMA_VERSION = 1
STATE_DIM = len(FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES)
FEATURE_NAMES = (
    "anchor_semantic_direction_cosine",
    "pooled_semantic_direction_cosine",
    "minimum_semantic_direction_concentration",
    "absolute_semantic_direction_concentration_difference",
    "absolute_anchor_log_amplitude_difference",
    "absolute_mean_log_amplitude_difference",
    "minimum_mean_observation_evidence",
    "minimum_mean_visibility_purity_known_value",
    "minimum_visibility_purity_known_fraction",
)
FEATURE_NAMES_SHA256 = canonical_json_sha256(list(FEATURE_NAMES))


def source_access() -> dict[str, bool]:
    return {
        "query_independent": True,
        "raw_radio_vector_reconstructed": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "target_metrics_computed": False,
        "per_scene_hyperparameters": False,
    }


def interface_contract() -> dict[str, Any]:
    return {
        "schema": INTERFACE_SCHEMA,
        "schema_version": INTERFACE_SCHEMA_VERSION,
        "parent_factorized_state_contract_sha256": (
            FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        ),
        "input_gauge": {
            "semantic_direction": "unit_l2",
            "log_amplitude": "separate_scalar_never_multiplied_into_direction",
            "state_missingness": "explicit_known_mask",
        },
        "aggregation": {
            "direction": "uniform_spherical_mean_direction_and_concentration",
            "log_amplitude": "uniform_arithmetic_mean_in_log_space",
            "observation_evidence": "uniform_mean",
            "visibility_purity": "known_value_mean_plus_known_fraction",
        },
        "pair_symmetry": True,
        "feature_names": list(FEATURE_NAMES),
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "legacy_accepted_v2_default_changed": False,
        "legacy_region_comembership_v2_default_changed": False,
        "source_access": source_access(),
    }


INTERFACE_CONTRACT_SHA256 = canonical_json_sha256(interface_contract())


@dataclass(frozen=True)
class FactorizedNativeRegionSummary:
    """Permutation-invariant summaries of factorized-native region tokens."""

    anchor_semantic_direction: torch.Tensor
    pooled_semantic_direction: torch.Tensor
    semantic_direction_concentration: torch.Tensor
    anchor_log_amplitude: torch.Tensor
    mean_log_amplitude: torch.Tensor
    mean_observation_evidence: torch.Tensor
    mean_visibility_purity_known_value: torch.Tensor
    visibility_purity_known_fraction: torch.Tensor


@dataclass(frozen=True)
class FactorizedNativeRegionRelation(FactorizedNativeRegionSummary):
    """Region summaries and aligned symmetric pair features."""

    pair_indices: torch.Tensor
    pair_features: torch.Tensor


def _validate_inputs(
    *,
    unit_direction: torch.Tensor,
    log_amplitude: torch.Tensor,
    state: torch.Tensor,
    state_known_mask: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor | int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    direction_source = torch.as_tensor(unit_direction)
    if not direction_source.is_floating_point():
        raise ValueError("factorized-native direction must be floating point")
    direction = direction_source.detach().float().cpu()
    amplitude = torch.as_tensor(log_amplitude).detach().float().cpu()
    values = torch.as_tensor(state).detach().float().cpu()
    known = torch.as_tensor(state_known_mask).detach().cpu()
    mask = torch.as_tensor(token_mask).detach().cpu()
    region_count = int(direction.shape[0]) if direction.ndim == 3 else -1
    if amplitude.ndim == 3 and amplitude.shape[-1] == 1:
        amplitude = amplitude[..., 0]
    if (
        region_count <= 0
        or direction.shape[-1] <= 0
        or amplitude.shape != direction.shape[:2]
        or values.shape != (*direction.shape[:2], STATE_DIM)
        or known.dtype != torch.bool
        or known.shape != values.shape
        or mask.dtype != torch.bool
        or mask.shape != direction.shape[:2]
        or not bool(mask.any(dim=1).all())
    ):
        raise ValueError("factorized-native region relation axes differ")
    active_direction = direction[mask]
    always_known = known[..., (0, 1, 2, 3, 5)]
    active_values = values[mask]
    tolerance = (
        5e-4
        if direction_source.dtype in {torch.float16, torch.bfloat16}
        else 2e-4
    )
    if (
        not bool(torch.isfinite(active_direction).all())
        or not bool(torch.isfinite(amplitude[mask]).all())
        or not bool(torch.isfinite(values[mask]).all())
        or not torch.allclose(
            torch.linalg.vector_norm(active_direction, dim=-1),
            torch.ones(active_direction.shape[0]),
            rtol=0.0,
            atol=tolerance,
        )
        or bool(direction[~mask].count_nonzero())
        or bool(amplitude[~mask].count_nonzero())
        or bool(values[~mask].count_nonzero())
        or bool(known[~mask].any())
        or bool(values[~known].count_nonzero())
        or not bool(always_known[mask].all())
        or not torch.equal(values[..., 0][mask], amplitude[mask])
        or bool((active_values[:, 1] < 0).any())
        or bool((active_values[:, 1] > 1).any())
        or bool((active_values[:, 2] < 0).any())
        or bool((active_values[:, 3] < 0).any())
        or bool((active_values[:, 3] > 1).any())
        or bool((values[..., 4][known[..., 4]] < 0).any())
        or bool((values[..., 4][known[..., 4]] > 1).any())
        or not torch.equal(values[..., 5][mask], known[..., 4][mask].float())
    ):
        raise ValueError("factorized-native region carriers differ")
    anchor = torch.as_tensor(anchor_index).detach().long().cpu().reshape(-1)
    if anchor.numel() == 1:
        anchor = anchor.expand(region_count)
    batch = torch.arange(region_count)
    if (
        anchor.shape != (region_count,)
        or bool((anchor < 0).any())
        or bool((anchor >= direction.shape[1]).any())
        or not bool(mask[batch, anchor].all())
    ):
        raise ValueError("factorized-native relation anchor differs")
    return direction, amplitude, values, known, mask, anchor


def factorized_native_region_summaries(
    *,
    unit_direction: torch.Tensor,
    log_amplitude: torch.Tensor,
    state: torch.Tensor,
    state_known_mask: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor | int,
) -> FactorizedNativeRegionSummary:
    """Summarize one batch without allocating all scene region-token rows."""

    direction, amplitude, values, known, mask, anchor = _validate_inputs(
        unit_direction=unit_direction,
        log_amplitude=log_amplitude,
        state=state,
        state_known_mask=state_known_mask,
        token_mask=token_mask,
        anchor_index=anchor_index,
    )
    weights = mask.float()
    count = weights.sum(dim=1).clamp_min(1.0)
    mean_direction = (direction * weights[..., None]).sum(dim=1) / count[:, None]
    concentration = torch.linalg.vector_norm(mean_direction, dim=-1)
    pooled_direction = F.normalize(mean_direction, dim=-1)
    mean_amplitude = (amplitude * weights).sum(dim=1) / count
    mean_observation = (values[..., 3] * weights).sum(dim=1) / count

    purity_known = known[..., 4] & mask
    known_count = purity_known.sum(dim=1)
    purity_mean = (values[..., 4] * purity_known.float()).sum(dim=1) / known_count.clamp_min(
        1
    )
    purity_known_fraction = known_count.float() / count
    batch = torch.arange(direction.shape[0])
    return FactorizedNativeRegionSummary(
        anchor_semantic_direction=direction[batch, anchor].float().contiguous(),
        pooled_semantic_direction=pooled_direction.float().contiguous(),
        semantic_direction_concentration=concentration.float().contiguous(),
        anchor_log_amplitude=amplitude[batch, anchor].float().contiguous(),
        mean_log_amplitude=mean_amplitude.float().contiguous(),
        mean_observation_evidence=mean_observation.float().contiguous(),
        mean_visibility_purity_known_value=purity_mean.float().contiguous(),
        visibility_purity_known_fraction=purity_known_fraction.float().contiguous(),
    )


def factorized_native_pair_features(
    summary: FactorizedNativeRegionSummary,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    """Materialize symmetric pair channels from precomputed region summaries."""

    if not isinstance(summary, FactorizedNativeRegionSummary):
        raise TypeError("native pair features require region summaries")
    count = int(summary.pooled_semantic_direction.shape[0])
    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    tensors = (
        summary.anchor_semantic_direction,
        summary.pooled_semantic_direction,
        summary.semantic_direction_concentration,
        summary.anchor_log_amplitude,
        summary.mean_log_amplitude,
        summary.mean_observation_evidence,
        summary.mean_visibility_purity_known_value,
        summary.visibility_purity_known_fraction,
    )
    if (
        count <= 1
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or pairs.shape[1] <= 0
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or any(torch.as_tensor(value).shape[0] != count for value in tensors)
        or any(not bool(torch.isfinite(torch.as_tensor(value)).all()) for value in tensors)
    ):
        raise ValueError("factorized-native pair summary axes differ")
    pair_keys = pairs[0] * count + pairs[1]
    if pair_keys.numel() > 1 and not bool((pair_keys[1:] > pair_keys[:-1]).all()):
        raise ValueError("factorized-native relation pairs must be sorted unique")
    left, right = pairs
    features = torch.stack(
        (
            (
                summary.anchor_semantic_direction[left]
                * summary.anchor_semantic_direction[right]
            ).sum(dim=-1),
            (
                summary.pooled_semantic_direction[left]
                * summary.pooled_semantic_direction[right]
            ).sum(dim=-1),
            torch.minimum(
                summary.semantic_direction_concentration[left],
                summary.semantic_direction_concentration[right],
            ),
            (
                summary.semantic_direction_concentration[left]
                - summary.semantic_direction_concentration[right]
            ).abs(),
            (
                summary.anchor_log_amplitude[left]
                - summary.anchor_log_amplitude[right]
            ).abs(),
            (
                summary.mean_log_amplitude[left]
                - summary.mean_log_amplitude[right]
            ).abs(),
            torch.minimum(
                summary.mean_observation_evidence[left],
                summary.mean_observation_evidence[right],
            ),
            torch.minimum(
                summary.mean_visibility_purity_known_value[left],
                summary.mean_visibility_purity_known_value[right],
            ),
            torch.minimum(
                summary.visibility_purity_known_fraction[left],
                summary.visibility_purity_known_fraction[right],
            ),
        ),
        dim=1,
    ).float().contiguous()
    if features.shape != (pairs.shape[1], len(FEATURE_NAMES)):
        raise RuntimeError("factorized-native pair feature dimension differs")
    return features


def factorized_native_region_relation_features(
    *,
    unit_direction: torch.Tensor,
    log_amplitude: torch.Tensor,
    state: torch.Tensor,
    state_known_mask: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor | int,
    pair_indices: torch.Tensor,
) -> FactorizedNativeRegionRelation:
    """Build native, query-free symmetric relation evidence for region pairs."""

    summary = factorized_native_region_summaries(
        unit_direction=unit_direction,
        log_amplitude=log_amplitude,
        state=state,
        state_known_mask=state_known_mask,
        token_mask=token_mask,
        anchor_index=anchor_index,
    )
    pairs = torch.as_tensor(pair_indices).detach().long().cpu().contiguous()
    features = factorized_native_pair_features(summary, pairs)
    if (
        features.shape != (pairs.shape[1], len(FEATURE_NAMES))
        or not bool(torch.isfinite(features).all())
        or bool((summary.semantic_direction_concentration < 0).any())
        or bool((summary.semantic_direction_concentration > 1.0001).any())
    ):
        raise RuntimeError("factorized-native relation materialization failed")
    return FactorizedNativeRegionRelation(
        anchor_semantic_direction=summary.anchor_semantic_direction,
        anchor_log_amplitude=summary.anchor_log_amplitude,
        pair_indices=pairs.contiguous(),
        pair_features=features,
        pooled_semantic_direction=summary.pooled_semantic_direction,
        semantic_direction_concentration=summary.semantic_direction_concentration,
        mean_log_amplitude=summary.mean_log_amplitude,
        mean_observation_evidence=summary.mean_observation_evidence,
        mean_visibility_purity_known_value=(
            summary.mean_visibility_purity_known_value
        ),
        visibility_purity_known_fraction=summary.visibility_purity_known_fraction,
    )


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_NAMES_SHA256",
    "INTERFACE_CONTRACT_SHA256",
    "INTERFACE_SCHEMA",
    "INTERFACE_SCHEMA_VERSION",
    "FactorizedNativeRegionRelation",
    "FactorizedNativeRegionSummary",
    "factorized_native_pair_features",
    "factorized_native_region_relation_features",
    "factorized_native_region_summaries",
    "interface_contract",
    "source_access",
]
