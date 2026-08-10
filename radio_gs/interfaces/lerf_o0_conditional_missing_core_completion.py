"""Source-frozen conditional completion of exact-O0 missing region cores.

This is the conservative counterpart of unconditional FIX6 completion.  Exact
O0 alone proposes a missing primitive/query cell inside a qualified region
core.  A six-feature monotone selector, fitted on source scene0001 and accepted
on strict heldout scene0004, decides whether that individual proposal may be
completed.  The selector and its inclusive threshold are never refit on the
target benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    MonotoneAdditiveLogistic,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    target_consensus_probability,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.lerf_o0_conditional_missing_core_completion.v1"
EXTERNAL_CACHE_SCHEMA = (
    "radio_gs.lerf_o0_conditional_missing_core_completion_external_scores.v1"
)
O0_SCORE_MINIMUM = 0.6
O0_CORE_SUPERMAJORITY = 0.75
MAXIMUM_MEMBERSHIP_EXPANSION = 3.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def completion_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "proposal": (
            "exact_O0_valid_core_score_less_than_or_equal_to_0p6_in_region_"
            "whose_exact_O0_positive_fraction_is_at_least_0p75"
        ),
        "selector": {
            "model": "source_scene0001_three_fold_monotone_additive_logistic",
            "validation": "strict_source_scene0004_heldout_PASS_required",
            "feature_names": list(SELECTOR_FEATURE_NAMES),
            "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
            "probability": "minimum_probability_across_three_fold_models",
            "threshold": "exact_frozen_scene0001_threshold_inclusive",
            "refit_on_target": False,
        },
        "completion_value": "qualified_region_exact_O0_positive_fraction",
        "overlap": "pointwise_maximum_over_selected_proposals",
        "fusion": "pointwise_maximum_exact_O0_and_selected_completion",
        "fallback": "outside_selected_proposal_cells_is_bitwise_exact_O0",
        "query_order_equivariant": True,
        "graph_or_relation": False,
        "cross_query_argmax": False,
        "scene_or_query_identifier_feature": False,
        "instance_or_target_label_feature": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(completion_contract())


@dataclass(frozen=True)
class ConditionalMissingCoreCompletionResult:
    valid_core_counts: torch.Tensor
    positive_fraction: torch.Tensor
    qualified_anchor_mask: torch.Tensor
    unit_region_indices: torch.Tensor
    unit_query_indices: torch.Tensor
    unit_primitive_rows: torch.Tensor
    selector_probability: torch.Tensor
    selected_unit_mask: torch.Tensor
    unconditional_cell_mask: torch.Tensor
    selected_cell_mask: torch.Tensor
    completion_probability: torch.Tensor
    final_scores: torch.Tensor
    changed_mask: torch.Tensor


def _selector_source_axis(
    *,
    unit_o0: torch.Tensor,
    unit_region: torch.Tensor,
    unit_query: torch.Tensor,
    appearance_concentration: torch.Tensor,
    boundary_concentration: torch.Tensor,
    core_spatial_rms_radius: torch.Tensor,
    selected_query_scale_indices: torch.Tensor,
    full_scalar_source_robust_ood_linf: torch.Tensor,
) -> torch.Tensor:
    """Construct only the frozen selector's six source-axis channels."""

    count = int(unit_o0.numel())
    feature = torch.zeros(
        count, max(SOURCE_UNIT_FEATURE_INDICES) + 1, dtype=torch.float32
    )
    feature[:, 0] = unit_o0
    feature[:, 14] = appearance_concentration[unit_region]
    feature[:, 15] = boundary_concentration[unit_region]
    feature[:, 17] = core_spatial_rms_radius[unit_region]
    feature[:, 9] = selected_query_scale_indices[unit_query].float()
    feature[:, 18] = full_scalar_source_robust_ood_linf[unit_region]
    return feature.contiguous()


def conditional_missing_core_completion(
    *,
    o0_scores: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    appearance_concentration: torch.Tensor,
    boundary_concentration: torch.Tensor,
    core_spatial_rms_radius: torch.Tensor,
    selected_query_scale_indices: torch.Tensor,
    full_scalar_source_robust_ood_linf: torch.Tensor,
    fold_models: tuple[MonotoneAdditiveLogistic, ...],
    threshold_inclusive: float,
) -> ConditionalMissingCoreCompletionResult:
    """Select and complete exact-O0 missing-core proposals without target labels."""

    scores = torch.as_tensor(o0_scores).detach()
    rows = torch.as_tensor(region_rows).detach()
    core = torch.as_tensor(core_mask).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    appearance = torch.as_tensor(appearance_concentration).detach()
    boundary = torch.as_tensor(boundary_concentration).detach()
    radius = torch.as_tensor(core_spatial_rms_radius).detach()
    query_scale = torch.as_tensor(selected_query_scale_indices).detach()
    ood = torch.as_tensor(full_scalar_source_robust_ood_linf).detach()
    region_count = int(rows.shape[0]) if rows.ndim == 2 else -1
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
        or region_count <= 0
        or core.device.type != "cpu"
        or core.dtype != torch.bool
        or core.shape != rows.shape
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or appearance.dtype != torch.float32
        or appearance.shape != (region_count,)
        or boundary.dtype != torch.float32
        or boundary.shape != (region_count,)
        or radius.dtype != torch.float32
        or radius.shape != (region_count,)
        or query_scale.dtype not in {torch.int32, torch.int64}
        or query_scale.shape != (scores.shape[1],)
        or ood.dtype != torch.float32
        or ood.shape != (region_count,)
        or not bool(torch.isfinite(appearance).all())
        or not bool(torch.isfinite(boundary).all())
        or not bool(torch.isfinite(radius).all())
        or not bool(torch.isfinite(ood).all())
        or bool((appearance < 0.0).any())
        or bool((appearance > 1.0).any())
        or bool((boundary < 0.0).any())
        or bool((boundary > 1.0).any())
        or bool((radius < 0.0).any())
        or bool((ood < 0.0).any())
        or bool((query_scale < 0).any())
        or not 0.0 <= float(threshold_inclusive) <= 1.0
        or not bool(core.any(dim=1).all())
        or bool((rows[core] < 0).any())
        or bool((rows[core] >= scores.shape[0]).any())
    ):
        raise ValueError("conditional missing-core completion inputs differ")

    safe_rows = rows.long().clamp(min=0, max=scores.shape[0] - 1)
    valid_core = core & valid[safe_rows]
    counts = valid_core.sum(dim=1)
    fractions = torch.zeros(region_count, scores.shape[1], dtype=torch.float32)
    qualified = torch.zeros_like(fractions, dtype=torch.bool)
    unit_regions: list[torch.Tensor] = []
    unit_queries: list[torch.Tensor] = []
    unit_primitives: list[torch.Tensor] = []
    unit_scores: list[torch.Tensor] = []
    for region in range(region_count):
        active = safe_rows[region, valid_core[region]]
        if active.numel() == 0:
            continue
        active_scores = scores[active]
        fraction = (active_scores > O0_SCORE_MINIMUM).float().mean(dim=0)
        fractions[region] = fraction
        active_queries = torch.where(fraction >= O0_CORE_SUPERMAJORITY)[0]
        for query in active_queries.tolist():
            missing = active[active_scores[:, query] <= O0_SCORE_MINIMUM]
            if missing.numel() == 0:
                continue
            qualified[region, query] = True
            count = int(missing.numel())
            unit_regions.append(torch.full((count,), region, dtype=torch.long))
            unit_queries.append(torch.full((count,), query, dtype=torch.long))
            unit_primitives.append(missing.long())
            unit_scores.append(scores[missing, query])

    def _cat(parts: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        return (
            torch.cat(parts).to(dtype=dtype).contiguous()
            if parts
            else torch.empty(0, dtype=dtype)
        )

    unit_region = _cat(unit_regions, torch.long)
    unit_query = _cat(unit_queries, torch.long)
    unit_primitive = _cat(unit_primitives, torch.long)
    unit_o0 = _cat(unit_scores, torch.float32)
    if unit_region.numel() == 0:
        probability = torch.empty(0, dtype=torch.float32)
        selected = torch.empty(0, dtype=torch.bool)
    else:
        features = _selector_source_axis(
            unit_o0=unit_o0,
            unit_region=unit_region,
            unit_query=unit_query,
            appearance_concentration=appearance,
            boundary_concentration=boundary,
            core_spatial_rms_radius=radius,
            selected_query_scale_indices=query_scale,
            full_scalar_source_robust_ood_linf=ood,
        )
        probability = target_consensus_probability(fold_models, features)
        selected = probability >= float(threshold_inclusive)

    cells = scores.numel()
    unconditional_flat = torch.zeros(cells, dtype=torch.bool)
    selected_flat = torch.zeros(cells, dtype=torch.bool)
    completion_flat = torch.zeros(cells, dtype=torch.float32)
    flat = unit_primitive * scores.shape[1] + unit_query
    if flat.numel():
        unconditional_flat[flat] = True
    if bool(selected.any()):
        chosen_flat = flat[selected]
        chosen_value = fractions[unit_region[selected], unit_query[selected]]
        selected_flat[chosen_flat] = True
        completion_flat.scatter_reduce_(
            0, chosen_flat, chosen_value, reduce="amax", include_self=True
        )
    unconditional_cells = unconditional_flat.reshape_as(scores)
    selected_cells = selected_flat.reshape_as(scores)
    completion = completion_flat.reshape_as(scores)
    final = torch.maximum(scores, completion)
    changed = final > scores
    if (
        bool((changed & ~selected_cells).any())
        or not torch.equal(final[~selected_cells], scores[~selected_cells])
        or bool((selected_cells & ~unconditional_cells).any())
        or bool((final < scores).any())
        or not torch.equal(final[~valid], scores[~valid])
    ):
        raise RuntimeError("conditional missing-core completion invariant failed")
    return ConditionalMissingCoreCompletionResult(
        valid_core_counts=counts.long().contiguous(),
        positive_fraction=fractions.contiguous(),
        qualified_anchor_mask=qualified.contiguous(),
        unit_region_indices=unit_region,
        unit_query_indices=unit_query,
        unit_primitive_rows=unit_primitive,
        selector_probability=probability.contiguous(),
        selected_unit_mask=selected.contiguous(),
        unconditional_cell_mask=unconditional_cells.contiguous(),
        selected_cell_mask=selected_cells.contiguous(),
        completion_probability=completion.contiguous(),
        final_scores=final.contiguous(),
        changed_mask=changed.contiguous(),
    )


def access_audit() -> dict[str, bool]:
    return {
        "exact_O0_cache_opened": True,
        "query_independent_region_capability_and_geometry_opened": True,
        "source_frozen_selector_and_two_source_authorities_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan_or_refit": False,
        "scene_specific_parameters": False,
    }


def build_external_query_score_cache(
    *,
    result: ConditionalMissingCoreCompletionResult,
    o0_valid: torch.Tensor,
    o0_xyz: torch.Tensor,
    query_names: Sequence[str],
    scene_id: str,
    input_authority: Mapping[str, Mapping[str, str]],
    threshold_inclusive: float,
) -> dict[str, Any]:
    scores = result.final_scores.float().cpu().contiguous()
    valid = torch.as_tensor(o0_valid).detach().bool().cpu().contiguous()
    xyz = torch.as_tensor(o0_xyz).detach().float().cpu().contiguous()
    names = tuple(str(value) for value in query_names)
    records = {
        str(name): {"path": str(value["path"]), "sha256": str(value["sha256"])}
        for name, value in input_authority.items()
    }
    if (
        valid.shape != (scores.shape[0],)
        or xyz.shape != (scores.shape[0], 3)
        or len(names) != scores.shape[1]
        or len(set(names)) != len(names)
        or not scene_id
    ):
        raise ValueError("conditional completion external cache axes differ")
    payload = {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "contract": completion_contract(),
        "contract_sha256": CONTRACT_SHA256,
        "query_scores": scores,
        "valid": valid,
        "xyz": xyz,
        "metadata": {
            "scene_id": str(scene_id),
            "query_names": list(names),
            "score_semantics": "exact_O0_max_source_frozen_conditional_completion",
            "input_authority": records,
            "frozen_threshold_inclusive": float(threshold_inclusive),
            "qualified_anchor_counts": result.qualified_anchor_mask.sum(0)
            .long()
            .tolist(),
            "candidate_units": int(result.unit_region_indices.numel()),
            "selected_units": int(result.selected_unit_mask.sum()),
            "selected_unique_cells": int(result.selected_cell_mask.sum()),
            "strictly_changed_cells": int(result.changed_mask.sum()),
        },
        "access_audit": access_audit(),
    }
    payload["channel_sha256"] = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(valid),
        "xyz": tensor_sha256(xyz),
        "query_names": canonical_json_sha256(list(names)),
    }
    return validate_external_query_score_cache(payload)


def validate_external_query_score_cache(value: object) -> dict[str, Any]:
    """Fail closed on evaluator-facing axes, identity, and channel hashes."""

    required = {
        "schema",
        "contract",
        "contract_sha256",
        "query_scores",
        "valid",
        "xyz",
        "metadata",
        "access_audit",
        "channel_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("conditional completion external cache fields differ")
    payload = dict(value)
    scores = torch.as_tensor(payload["query_scores"])
    valid = torch.as_tensor(payload["valid"])
    xyz = torch.as_tensor(payload["xyz"])
    metadata = payload["metadata"]
    if (
        payload["schema"] != EXTERNAL_CACHE_SCHEMA
        or payload["contract"] != completion_contract()
        or payload["contract_sha256"] != CONTRACT_SHA256
        or payload["access_audit"] != access_audit()
        or not isinstance(metadata, Mapping)
        or scores.device.type != "cpu"
        or scores.dtype != torch.float32
        or scores.ndim != 2
        or min(scores.shape) <= 0
        or valid.device.type != "cpu"
        or valid.dtype != torch.bool
        or valid.shape != (scores.shape[0],)
        or xyz.device.type != "cpu"
        or xyz.dtype != torch.float32
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(scores).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
        or not bool(torch.isfinite(xyz).all())
    ):
        raise ValueError("conditional completion external cache contract differs")
    names = metadata.get("query_names")
    records = metadata.get("input_authority")
    if (
        not isinstance(names, list)
        or len(names) != scores.shape[1]
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(records, Mapping)
        or not records
        or not 0.0
        <= float(metadata.get("frozen_threshold_inclusive", float("nan")))
        <= 1.0
    ):
        raise ValueError("conditional completion external cache metadata differs")
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not Path(str(record["path"])).is_absolute()
            or _SHA256.fullmatch(str(record["sha256"])) is None
        ):
            raise ValueError("conditional completion input authority differs")
    expected_hashes = {
        "query_scores": tensor_sha256(scores),
        "valid": tensor_sha256(valid),
        "xyz": tensor_sha256(xyz),
        "query_names": canonical_json_sha256(names),
    }
    if payload["channel_sha256"] != expected_hashes:
        raise ValueError("conditional completion external cache channel changed")
    return payload


__all__ = [
    "CONTRACT_SHA256",
    "ConditionalMissingCoreCompletionResult",
    "EXTERNAL_CACHE_SCHEMA",
    "MAXIMUM_MEMBERSHIP_EXPANSION",
    "O0_CORE_SUPERMAJORITY",
    "O0_SCORE_MINIMUM",
    "SCHEMA",
    "access_audit",
    "build_external_query_score_cache",
    "completion_contract",
    "conditional_missing_core_completion",
    "validate_external_query_score_cache",
]
