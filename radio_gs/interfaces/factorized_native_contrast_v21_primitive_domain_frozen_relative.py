"""Primitive-domain frozen-relative unary for contrast-V2.1 region scores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_contrast_v21_frozen_relative_readout as region_formal
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    vala_knn_smoothed_scores,
    vala_minmax_remap_scores,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_primitive_domain_"
    "frozen_relative_unary.v1"
)
READOUT_SCHEMA_VERSION = 1
SEMANTIC_LEVELS = region_formal.SEMANTIC_LEVELS
KNN_NEIGHBORS = region_formal.KNN_NEIGHBORS
KNN_CHUNK_SIZE = region_formal.KNN_CHUNK_SIZE
MASK_THRESHOLD = region_formal.MASK_THRESHOLD
PROJECTION_RULE = "covering_region_max"


def projection_rule_audit() -> dict[str, Any]:
    return {
        "candidate_rules": {
            "covering_region_max": {
                "bounded_in_raw_probability_range": True,
                "idempotent_under_duplicate_region_support": True,
                "monotone_when_additional_support_is_added": True,
                "preserves_strongest_local_region_evidence": True,
                "requires_unavailable_membership_weights": False,
            },
            "coverage_weighted_mean": {
                "bounded_in_raw_probability_range": True,
                "idempotent_under_duplicate_region_support": False,
                "monotone_when_additional_support_is_added": False,
                "preserves_strongest_local_region_evidence": False,
                "requires_unavailable_membership_weights": True,
            },
        },
        "frozen_rule": PROJECTION_RULE,
        "reason": (
            "max_is_the_unique_considered_rule_preserving_existential_region_"
            "evidence_without_inventing_weights_or_overlap_dilution"
        ),
        "scene_or_query_statistics_used_for_rule_selection": False,
    }


def access_audit() -> dict[str, bool]:
    return {
        "strict_contrast_exact_lineage_validated": True,
        "strict_target_accepted_v2_lineage_validated": True,
        "strict_factorized_primitive_state_lineage_validated": True,
        "primitive_xyz_opened": True,
        "query_axis_is_opaque": True,
        "query_identifiers_forwarded_to_readout": False,
        "query_strings_forwarded_to_readout": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "scene_specific_parameters": False,
        "query_specific_parameters": False,
        "graph_or_relation_applied": False,
        "existing_region_relative_candidate_opened": False,
        "existing_candidate_modified": False,
    }


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "candidate_family": "independent_primitive_domain_frozen_relative_unary",
        "input": {
            "region_raw_relevance": "strict_contrast_exact_R_by_opaque_Q",
            "region_support": "AcceptedV2_region_rows_and_token_mask",
            "region_scale": "AcceptedV2_scale_indices_0_1_2",
            "primitive_xyz_and_valid": "strict_factorized_primitive_state",
        },
        "projection": {
            "rule": PROJECTION_RULE,
            "scope": "independently_per_scale_primitive_and_opaque_query",
            "uncovered_primitive_policy": "missing_and_ineligible_not_zero_observation",
            "rule_audit": projection_rule_audit(),
        },
        "fixed_primitive_rule": {
            "knn_domain": "covered_valid_primitive_xyz_independently_per_scale",
            "smoothing": "0.5_raw_plus_0.5_mean_of_knn_including_self",
            "knn_neighbors": KNN_NEIGHBORS,
            "level_selection": "highest_projected_raw_smoothed_peak_per_opaque_query",
            "level_tie_break": "lowest_scale_index",
            "remap": "per_level_per_query_minmax_then_clip_2u_minus_1_to_0_1",
            "mask_threshold": MASK_THRESHOLD,
            "mask_comparator": "strictly_greater",
        },
        "selected_level_eligibility": {
            "rule": "primitive_is_covered_at_selected_level_and_valid",
            "outside_eligibility_relevance": "exact_zero",
            "outside_eligibility_candidate": False,
        },
        "query_axis": "opaque",
        "threshold_scan": False,
        "scene_specific_parameters": False,
        "query_specific_parameters": False,
        "metric_access": False,
        "graph_or_relation": "none",
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class PrimitiveDomainFrozenRelative:
    projection_coverage_count: torch.Tensor
    projection_coverage: torch.Tensor
    projected_raw_relevance: torch.Tensor
    smoothed_relevance: torch.Tensor
    remapped_relevance: torch.Tensor
    raw_smoothed_peaks: torch.Tensor
    selected_scale_indices: torch.Tensor
    selected_scale_eligibility: torch.Tensor
    relative_relevance: torch.Tensor
    query_gate: torch.Tensor
    unary_candidate_mask: torch.Tensor


def _validated_inputs(
    *,
    region_raw_relevance: torch.Tensor,
    scale_indices: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_xyz: torch.Tensor,
    primitive_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.as_tensor(region_raw_relevance).detach().float().cpu().contiguous()
    scales = torch.as_tensor(scale_indices).detach().long().cpu().contiguous()
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    mask = torch.as_tensor(token_mask).detach().bool().cpu().contiguous()
    xyz = torch.as_tensor(primitive_xyz).detach().float().cpu().contiguous()
    valid = torch.as_tensor(primitive_valid).detach().bool().cpu().contiguous()
    regions = int(raw.shape[0]) if raw.ndim == 2 else -1
    primitives = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    if (
        regions <= 0
        or raw.shape[1] <= 0
        or primitives <= 0
        or scales.shape != (regions,)
        or rows.ndim != 2
        or rows.shape[0] != regions
        or mask.shape != rows.shape
        or xyz.shape != (primitives, 3)
        or valid.shape != (primitives,)
        or not bool(valid.any())
        or not bool(torch.isfinite(raw).all())
        or not bool(torch.isfinite(xyz).all())
        or bool((raw < 0.0).any())
        or bool((raw > 1.0).any())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= primitives).any())
        or bool((~valid[rows[mask]]).any())
        or not torch.equal(torch.unique(scales, sorted=True), torch.arange(SEMANTIC_LEVELS))
        or not bool(mask.any(dim=1).all())
    ):
        raise ValueError("primitive-domain frozen-relative inputs differ")
    return raw, scales, rows, mask, xyz, valid


def project_covering_region_max(
    *,
    region_raw_relevance: torch.Tensor,
    scale_indices: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    num_primitives: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.as_tensor(region_raw_relevance).detach().float().cpu().contiguous()
    scales = torch.as_tensor(scale_indices).detach().long().cpu().contiguous()
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    mask = torch.as_tensor(token_mask).detach().bool().cpu().contiguous()
    primitives = int(num_primitives)
    if (
        raw.ndim != 2
        or scales.shape != (raw.shape[0],)
        or rows.ndim != 2
        or rows.shape[0] != raw.shape[0]
        or mask.shape != rows.shape
        or primitives <= 0
    ):
        raise ValueError("primitive max projection axes differ")
    queries = raw.shape[1]
    projected = torch.zeros(SEMANTIC_LEVELS, primitives, queries, dtype=torch.float32)
    coverage_count = torch.zeros(SEMANTIC_LEVELS, primitives, dtype=torch.int64)
    for level in range(SEMANTIC_LEVELS):
        region_index = torch.where(scales == level)[0]
        level_rows = rows[region_index]
        level_mask = mask[region_index]
        lengths = level_mask.sum(dim=1).long()
        primitive_index = level_rows[level_mask]
        parent = torch.arange(region_index.numel()).repeat_interleave(lengths)
        coverage_count[level].scatter_add_(
            0, primitive_index, torch.ones_like(primitive_index, dtype=torch.int64)
        )
        for query in range(queries):
            values = raw[region_index, query][parent]
            target = torch.full((primitives,), -torch.inf, dtype=torch.float32)
            target.scatter_reduce_(
                0, primitive_index, values, reduce="amax", include_self=True
            )
            target[~torch.isfinite(target)] = 0.0
            projected[level, :, query] = target
    return coverage_count.contiguous(), projected.contiguous()


def primitive_domain_frozen_relative_readout(
    *,
    region_raw_relevance: torch.Tensor,
    scale_indices: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_xyz: torch.Tensor,
    primitive_valid: torch.Tensor,
    chunk_size: int = KNN_CHUNK_SIZE,
) -> PrimitiveDomainFrozenRelative:
    raw, scales, rows, mask, xyz, valid = _validated_inputs(
        region_raw_relevance=region_raw_relevance,
        scale_indices=scale_indices,
        region_rows=region_rows,
        token_mask=token_mask,
        primitive_xyz=primitive_xyz,
        primitive_valid=primitive_valid,
    )
    if int(chunk_size) <= 0:
        raise ValueError("primitive-domain KNN chunk size must be positive")
    counts, projected = project_covering_region_max(
        region_raw_relevance=raw,
        scale_indices=scales,
        region_rows=rows,
        token_mask=mask,
        num_primitives=xyz.shape[0],
    )
    coverage = (counts > 0) & valid[None, :]
    if not bool(coverage.any(dim=1).all()):
        raise ValueError("every primitive semantic level must cover a valid primitive")
    smoothed = torch.zeros_like(projected)
    remapped = torch.zeros_like(projected)
    peaks = torch.empty(SEMANTIC_LEVELS, raw.shape[1], dtype=torch.float32)
    for level in range(SEMANTIC_LEVELS):
        primitive_index = torch.where(coverage[level])[0]
        level_smoothed = vala_knn_smoothed_scores(
            projected[level, primitive_index],
            xyz[primitive_index],
            k=KNN_NEIGHBORS,
            chunk_size=int(chunk_size),
        )
        smoothed[level, primitive_index] = level_smoothed
        remapped[level, primitive_index] = vala_minmax_remap_scores(level_smoothed)
        peaks[level] = level_smoothed.amax(dim=0)
    selected = peaks.argmax(dim=0).long().contiguous()
    primitives, queries = xyz.shape[0], raw.shape[1]
    eligibility = torch.empty(primitives, queries, dtype=torch.bool)
    relative = torch.zeros(primitives, queries, dtype=torch.float32)
    for query in range(queries):
        level = int(selected[query])
        eligibility[:, query] = coverage[level]
        relative[:, query] = torch.where(
            coverage[level], remapped[level, :, query], torch.zeros(primitives)
        )
    candidates = eligibility & (relative > MASK_THRESHOLD)
    gate = candidates.any(dim=0)
    if (
        bool(projected[~coverage[:, :, None].expand_as(projected)].count_nonzero())
        or bool(smoothed[~coverage[:, :, None].expand_as(smoothed)].count_nonzero())
        or bool(remapped[~coverage[:, :, None].expand_as(remapped)].count_nonzero())
        or bool(relative[~eligibility].count_nonzero())
        or bool(candidates[~eligibility].any())
    ):
        raise RuntimeError("primitive-domain selected-level eligibility invariant failed")
    return PrimitiveDomainFrozenRelative(
        projection_coverage_count=counts,
        projection_coverage=coverage.contiguous(),
        projected_raw_relevance=projected,
        smoothed_relevance=smoothed.contiguous(),
        remapped_relevance=remapped.contiguous(),
        raw_smoothed_peaks=peaks.contiguous(),
        selected_scale_indices=selected,
        selected_scale_eligibility=eligibility.contiguous(),
        relative_relevance=relative.contiguous(),
        query_gate=gate.contiguous(),
        unary_candidate_mask=candidates.contiguous(),
    )


def channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    names = (
        "region_raw_relevance", "scale_indices", "region_rows", "token_mask", "primitive_xyz",
        "primitive_valid", "projection_coverage_count", "projection_coverage",
        "projected_raw_relevance", "smoothed_relevance", "remapped_relevance",
        "raw_smoothed_peaks", "selected_scale_indices",
        "selected_scale_eligibility", "relative_relevance", "query_gate",
        "unary_candidate_mask",
    )
    return {name: tensor_sha256(torch.as_tensor(value[name])) for name in names}


def expected_audit(readout: PrimitiveDomainFrozenRelative) -> dict[str, Any]:
    counts = readout.unary_candidate_mask.sum(dim=0)
    return {
        "opaque_query_axes": int(readout.relative_relevance.shape[1]),
        "semantic_levels": SEMANTIC_LEVELS,
        "projection_rule": PROJECTION_RULE,
        "projection_rule_audit": projection_rule_audit(),
        "covered_valid_primitives_per_level": [
            int(readout.projection_coverage[level].sum())
            for level in range(SEMANTIC_LEVELS)
        ],
        "membership_count_max_per_level": [
            int(readout.projection_coverage_count[level].max())
            for level in range(SEMANTIC_LEVELS)
        ],
        "selected_scale_counts": {
            str(level): int((readout.selected_scale_indices == level).sum())
            for level in range(SEMANTIC_LEVELS)
        },
        "query_gate_passed": int(readout.query_gate.sum()),
        "query_gate_failed": int((~readout.query_gate).sum()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_median": int(counts.float().median()),
        "candidate_count_max": int(counts.max()),
        "outside_selected_scale_nonzero": int(
            readout.relative_relevance[~readout.selected_scale_eligibility].count_nonzero()
        ),
        "outside_selected_scale_candidates": int(
            readout.unary_candidate_mask[~readout.selected_scale_eligibility].sum()
        ),
        "query_identifiers_consumed": False,
        "graph_or_relation_applied": False,
        "target_metric_computed": False,
    }


def validate_readout_authority(
    value: object, *, replay: PrimitiveDomainFrozenRelative | None = None
) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "contract", "contract_sha256", "scene_id",
        "physical_space_id", "producer", "execution_authority", "input_authority",
        "query_axis_count", "region_raw_relevance", "scale_indices", "region_rows", "token_mask",
        "primitive_xyz", "primitive_valid", "projection_coverage_count",
        "projection_coverage", "projected_raw_relevance", "smoothed_relevance",
        "remapped_relevance", "raw_smoothed_peaks", "selected_scale_indices",
        "selected_scale_eligibility", "relative_relevance", "query_gate",
        "unary_candidate_mask", "audit", "channel_sha256", "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("primitive-domain frozen-relative authority fields differ")
    payload = dict(value)
    checked = replay or primitive_domain_frozen_relative_readout(
        region_raw_relevance=payload["region_raw_relevance"],
        scale_indices=payload["scale_indices"],
        region_rows=payload["region_rows"],
        token_mask=payload["token_mask"],
        primitive_xyz=payload["primitive_xyz"],
        primitive_valid=payload["primitive_valid"],
    )
    expected = {
        "projection_coverage_count": checked.projection_coverage_count,
        "projection_coverage": checked.projection_coverage,
        "projected_raw_relevance": checked.projected_raw_relevance,
        "smoothed_relevance": checked.smoothed_relevance,
        "remapped_relevance": checked.remapped_relevance,
        "raw_smoothed_peaks": checked.raw_smoothed_peaks,
        "selected_scale_indices": checked.selected_scale_indices,
        "selected_scale_eligibility": checked.selected_scale_eligibility,
        "relative_relevance": checked.relative_relevance,
        "query_gate": checked.query_gate,
        "unary_candidate_mask": checked.unary_candidate_mask,
    }
    if (
        payload.get("schema") != READOUT_SCHEMA
        or payload.get("schema_version") != READOUT_SCHEMA_VERSION
        or payload.get("contract") != readout_contract()
        or payload.get("contract_sha256") != READOUT_CONTRACT_SHA256
        or payload.get("access_audit") != access_audit()
        or int(payload.get("query_axis_count", -1)) != checked.relative_relevance.shape[1]
        or any(not torch.equal(torch.as_tensor(payload[name]), tensor) for name, tensor in expected.items())
        or payload.get("audit") != expected_audit(checked)
        or payload.get("channel_sha256") != channel_sha256(payload)
    ):
        raise ValueError("primitive-domain frozen-relative authority differs")
    return payload


__all__ = [
    "KNN_CHUNK_SIZE", "KNN_NEIGHBORS", "MASK_THRESHOLD", "PROJECTION_RULE",
    "READOUT_CONTRACT_SHA256", "READOUT_SCHEMA", "READOUT_SCHEMA_VERSION",
    "SEMANTIC_LEVELS", "PrimitiveDomainFrozenRelative", "access_audit",
    "channel_sha256", "expected_audit", "primitive_domain_frozen_relative_readout",
    "project_covering_region_max", "projection_rule_audit", "readout_contract",
    "validate_readout_authority",
]
