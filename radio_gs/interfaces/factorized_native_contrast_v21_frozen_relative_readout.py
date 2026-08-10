"""Frozen-relative, scale-eligible readout for contrast-V2.1 regions.

The contrast V2.1 descriptor emits one region axis containing three native
semantic levels.  The raw absolute probability is useful as a descriptor
diagnostic, but it is not the frozen VALA readout: VALA smooths within a
level, selects the level with the largest raw peak, remaps that level, and
then applies a fixed 0.6 threshold.

This module adapts that frozen rule to AcceptedV2 region anchors.  It is
query-opaque and target-GT-free.  Rows outside the selected scale are exactly
ineligible; no graph, relation, or region union is applied here.  Any future
relation consumer must mask seeds, edges, paths, and union rows by the emitted
``selected_scale_eligibility`` channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    vala_knn_smoothed_scores,
    vala_minmax_remap_scores,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_frozen_relative_readout.v1"
)
READOUT_SCHEMA_VERSION = 1
SEMANTIC_LEVELS = 3
KNN_NEIGHBORS = 10
KNN_CHUNK_SIZE = 65536
MASK_THRESHOLD = 0.6
FROZEN_PROTOCOL_RECORD = {
    "path": "/root/RADIO-GS/paper/artifacts/evaluation_protocol_freeze_20260801.yaml",
    "sha256": "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916",
}


def access_audit() -> dict[str, bool]:
    return {
        "strict_contrast_exact_lineage_validated": True,
        "strict_target_accepted_v2_lineage_validated": True,
        "renderer_geometry_and_region_anchor_xyz_opened": True,
        "query_axis_is_opaque": True,
        "query_identifiers_forwarded_to_readout": False,
        "query_strings_forwarded_to_readout": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "scene_specific_parameters": False,
        "graph_or_relation_applied": False,
    }


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "frozen_protocol": dict(FROZEN_PROTOCOL_RECORD),
        "input": {
            "raw_relevance": "float32_closed_unit_interval_R_by_opaque_Q",
            "scale_indices": "strict_AcceptedV2_canonical_region_scale_axis",
            "anchor_xyz": "renderer_xyz_at_AcceptedV2_canonical_anchor_rows",
            "semantic_levels": SEMANTIC_LEVELS,
        },
        "fixed_rule": {
            "smoothing": "0.5_raw_plus_0.5_mean_of_knn_including_self",
            "knn_neighbors": KNN_NEIGHBORS,
            "knn_scope": "independently_within_each_semantic_scale",
            "level_selection": "highest_raw_smoothed_peak_per_opaque_query",
            "level_tie_break": "lowest_scale_index",
            "remap": "per_scale_per_query_minmax_then_clip_2u_minus_1_to_0_1",
            "mask_threshold": MASK_THRESHOLD,
            "mask_comparator": "strictly_greater",
        },
        "scale_eligibility": {
            "rule": "scale_indices_R_equals_selected_scale_Q",
            "outside_selected_scale_relevance": "exact_zero",
            "outside_selected_scale_candidate": False,
            "future_graph_requirement": (
                "seed_edge_path_relation_and_union_must_all_remain_inside_eligibility"
            ),
        },
        "adaptation_boundary": {
            "canonical_VALA_knn_domain": "primitive_xyz_within_each_semantic_level",
            "this_region_bridge_knn_domain": (
                "AcceptedV2_region_anchor_xyz_within_each_semantic_level"
            ),
            "claim_of_bitwise_primitive_equivalence": False,
        },
        "output_candidate": "frozen_relative_unary_only",
        "graph_or_relation": "forbidden_in_v1",
        "query_axis": "opaque",
        "threshold_scan": False,
        "scene_specific_parameters": False,
        "metric_access": False,
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class FrozenRelativeRegionReadout:
    raw_relevance: torch.Tensor
    smoothed_relevance: torch.Tensor
    remapped_relevance: torch.Tensor
    raw_smoothed_peaks: torch.Tensor
    selected_scale_indices: torch.Tensor
    selected_scale_eligibility: torch.Tensor
    relative_relevance: torch.Tensor
    query_gate: torch.Tensor
    unary_candidate_mask: torch.Tensor


def _validated_inputs(
    raw_relevance: torch.Tensor,
    scale_indices: torch.Tensor,
    anchor_xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.as_tensor(raw_relevance).detach().float().cpu().contiguous()
    scales = torch.as_tensor(scale_indices).detach().long().cpu().contiguous()
    xyz = torch.as_tensor(anchor_xyz).detach().float().cpu().contiguous()
    regions = int(raw.shape[0]) if raw.ndim == 2 else -1
    levels = torch.unique(scales, sorted=True)
    if (
        regions <= 0
        or raw.shape[1] <= 0
        or scales.shape != (regions,)
        or xyz.shape != (regions, 3)
        or raw.dtype != torch.float32
        or scales.dtype != torch.int64
        or xyz.dtype != torch.float32
        or not bool(torch.isfinite(raw).all())
        or not bool(torch.isfinite(xyz).all())
        or bool((raw < 0.0).any())
        or bool((raw > 1.0).any())
        or not torch.equal(levels, torch.arange(SEMANTIC_LEVELS))
    ):
        raise ValueError("frozen-relative region readout inputs differ")
    return raw, scales, xyz, levels


def frozen_relative_region_readout(
    *,
    raw_relevance: torch.Tensor,
    scale_indices: torch.Tensor,
    anchor_xyz: torch.Tensor,
    chunk_size: int = KNN_CHUNK_SIZE,
) -> FrozenRelativeRegionReadout:
    """Apply fixed within-scale KNN, peak selection, remap, and eligibility."""

    raw, scales, xyz, levels = _validated_inputs(
        raw_relevance, scale_indices, anchor_xyz
    )
    if int(chunk_size) <= 0:
        raise ValueError("frozen-relative KNN chunk size must be positive")
    regions, queries = raw.shape
    smoothed = torch.empty_like(raw)
    remapped = torch.empty_like(raw)
    peaks = torch.empty(SEMANTIC_LEVELS, queries, dtype=torch.float32)
    for level in levels.tolist():
        rows = torch.where(scales == int(level))[0]
        if rows.numel() <= 0:
            raise ValueError("every frozen semantic level must be nonempty")
        level_smoothed = vala_knn_smoothed_scores(
            raw[rows],
            xyz[rows],
            k=KNN_NEIGHBORS,
            chunk_size=int(chunk_size),
        )
        smoothed[rows] = level_smoothed
        remapped[rows] = vala_minmax_remap_scores(level_smoothed)
        peaks[int(level)] = level_smoothed.amax(dim=0)

    selected = peaks.argmax(dim=0).long().contiguous()
    eligibility = scales[:, None] == selected[None, :]
    relative = torch.where(eligibility, remapped, torch.zeros_like(remapped))
    candidates = eligibility & (relative > MASK_THRESHOLD)
    gate = candidates.any(dim=0)
    if (
        bool(relative[~eligibility].count_nonzero())
        or bool(candidates[~eligibility].any())
        or not torch.equal(relative[eligibility], remapped[eligibility])
    ):
        raise RuntimeError("selected-scale eligibility invariant failed")
    return FrozenRelativeRegionReadout(
        raw_relevance=raw,
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
        "canonical_region_indices",
        "scale_indices",
        "anchor_rows",
        "anchor_xyz",
        "raw_relevance",
        "smoothed_relevance",
        "remapped_relevance",
        "raw_smoothed_peaks",
        "selected_scale_indices",
        "selected_scale_eligibility",
        "relative_relevance",
        "query_gate",
        "unary_candidate_mask",
    )
    return {name: tensor_sha256(torch.as_tensor(value[name])) for name in names}


def validate_readout_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "execution_authority",
        "input_authority",
        "region_fingerprints_sha256",
        "query_axis_count",
        "canonical_region_indices",
        "scale_indices",
        "anchor_rows",
        "anchor_xyz",
        "raw_relevance",
        "smoothed_relevance",
        "remapped_relevance",
        "raw_smoothed_peaks",
        "selected_scale_indices",
        "selected_scale_eligibility",
        "relative_relevance",
        "query_gate",
        "unary_candidate_mask",
        "audit",
        "channel_sha256",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen-relative readout authority fields differ")
    payload = dict(value)
    raw = torch.as_tensor(payload.get("raw_relevance"))
    scales = torch.as_tensor(payload.get("scale_indices"))
    xyz = torch.as_tensor(payload.get("anchor_xyz"))
    checked = frozen_relative_region_readout(
        raw_relevance=raw,
        scale_indices=scales,
        anchor_xyz=xyz,
    )
    canonical = torch.as_tensor(payload.get("canonical_region_indices"))
    anchor_rows = torch.as_tensor(payload.get("anchor_rows"))
    regions, queries = raw.shape
    tensors = {
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
        or not isinstance(payload.get("scene_id"), str)
        or not payload.get("scene_id")
        or not isinstance(payload.get("physical_space_id"), str)
        or not payload.get("physical_space_id")
        or int(payload.get("query_axis_count", -1)) != queries
        or canonical.dtype != torch.int64
        or canonical.device.type != "cpu"
        or canonical.shape != (regions,)
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or anchor_rows.dtype != torch.int64
        or anchor_rows.device.type != "cpu"
        or anchor_rows.shape != (regions,)
        or bool((anchor_rows < 0).any())
        or any(
            not torch.equal(torch.as_tensor(payload[name]), expected)
            for name, expected in tensors.items()
        )
        or payload.get("channel_sha256") != channel_sha256(payload)
    ):
        raise ValueError("frozen-relative readout authority differs")
    audit = payload.get("audit")
    counts = checked.unary_candidate_mask.sum(dim=0)
    expected_audit = {
        "opaque_query_axes": queries,
        "semantic_levels": SEMANTIC_LEVELS,
        "selected_scale_counts": {
            str(level): int((checked.selected_scale_indices == level).sum())
            for level in range(SEMANTIC_LEVELS)
        },
        "query_gate_passed": int(checked.query_gate.sum()),
        "query_gate_failed": int((~checked.query_gate).sum()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_median": int(counts.float().median()),
        "candidate_count_max": int(counts.max()),
        "outside_selected_scale_nonzero": int(
            checked.relative_relevance[~checked.selected_scale_eligibility]
            .count_nonzero()
        ),
        "outside_selected_scale_candidates": int(
            checked.unary_candidate_mask[~checked.selected_scale_eligibility].sum()
        ),
        "graph_or_relation_applied": False,
        "query_identifiers_consumed_by_readout": False,
        "target_metric_computed": False,
    }
    if audit != expected_audit:
        raise ValueError("frozen-relative readout audit differs")
    return payload


__all__ = [
    "FROZEN_PROTOCOL_RECORD",
    "KNN_CHUNK_SIZE",
    "KNN_NEIGHBORS",
    "MASK_THRESHOLD",
    "READOUT_CONTRACT_SHA256",
    "READOUT_SCHEMA",
    "READOUT_SCHEMA_VERSION",
    "SEMANTIC_LEVELS",
    "FrozenRelativeRegionReadout",
    "access_audit",
    "channel_sha256",
    "frozen_relative_region_readout",
    "readout_contract",
    "validate_readout_authority",
]
