"""Query-opaque primitive union for the frozen-relative contrast V2.1 unary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_readout as relative_formal,
)
from radio_gs.querying.multi_region_union_readout import (
    MultiRegionUnionConfig,
    greedy_novelty_union_readout,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = (
    "radio_gs.factorized_native_contrast_v21_"
    "frozen_relative_primitive_readout.v1"
)
READOUT_SCHEMA_VERSION = 1
SCORE_THRESHOLD = relative_formal.MASK_THRESHOLD
MAXIMUM_REGIONS = 8
CANDIDATE_CHUNK_ROWS = 4096


def access_audit() -> dict[str, bool]:
    return {
        "strict_frozen_relative_readout_validated": True,
        "strict_target_accepted_v2_validated": True,
        "factorized_primitive_valid_axis_opened": True,
        "query_axis_is_opaque": True,
        "query_identifiers_opened": False,
        "query_strings_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "graph_or_relation_applied": False,
        "scene_specific_parameters": False,
    }


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "input_schema": relative_formal.READOUT_SCHEMA,
        "input_contract_sha256": relative_formal.READOUT_CONTRACT_SHA256,
        "input_probability": (
            "selected_scale_relative_relevance_strictly_above_0p6_only"
        ),
        "selection": {
            "method": "deterministic_greedy_novelty_union",
            "score_threshold": SCORE_THRESHOLD,
            "threshold_semantics": (
                "input_candidate_mask_is_strict_gt_then_non_candidates_exact_zero"
            ),
            "maximum_regions": MAXIMUM_REGIONS,
            "candidate_chunk_rows": CANDIDATE_CHUNK_ROWS,
            "tie_break": "smaller_canonical_region_row_index",
        },
        "region_support": "AcceptedV2_region_rows_and_token_mask",
        "invalid_primitive_policy": "force_exact_zero_after_union",
        "selected_scale_invariants": {
            "selected_region_must_be_input_unary_candidate": True,
            "selected_region_must_be_selected_scale_eligible": True,
            "cross_scale_seed_edge_path_relation_union": "forbidden",
        },
        "graph_or_relation": "none",
        "query_axis": "opaque",
        "metric_access": False,
        "scene_specific_parameters": False,
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class FrozenRelativePrimitiveReadout:
    candidate_probability: torch.Tensor
    primitive_valid: torch.Tensor
    primitive_membership: torch.Tensor
    selected_region_indices: tuple[tuple[int, ...], ...]
    selected_region_scores: tuple[tuple[float, ...], ...]
    selected_marginal_core_rows: tuple[tuple[int, ...], ...]
    invalid_primitive_memberships_removed: int


def frozen_relative_primitive_readout(
    *,
    relative_relevance: torch.Tensor,
    selected_scale_eligibility: torch.Tensor,
    unary_candidate_mask: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    primitive_valid: torch.Tensor,
) -> FrozenRelativePrimitiveReadout:
    relevance = torch.as_tensor(relative_relevance).detach().float().cpu().contiguous()
    eligibility = (
        torch.as_tensor(selected_scale_eligibility)
        .detach()
        .bool()
        .cpu()
        .contiguous()
    )
    candidates = (
        torch.as_tensor(unary_candidate_mask)
        .detach()
        .bool()
        .cpu()
        .contiguous()
    )
    rows = torch.as_tensor(region_rows).detach().long().cpu().contiguous()
    core = torch.as_tensor(token_mask).detach().bool().cpu().contiguous()
    valid = torch.as_tensor(primitive_valid).detach().bool().cpu().contiguous()
    regions, queries = relevance.shape if relevance.ndim == 2 else (-1, -1)
    if (
        regions <= 0
        or queries <= 0
        or eligibility.shape != relevance.shape
        or candidates.shape != relevance.shape
        or rows.ndim != 2
        or rows.shape[0] != regions
        or core.shape != rows.shape
        or valid.ndim != 1
        or valid.numel() <= 0
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0.0).any())
        or bool((relevance > 1.0).any())
        or bool(candidates[~eligibility].any())
        or bool(relevance[~eligibility].count_nonzero())
        or not torch.equal(candidates, eligibility & (relevance > SCORE_THRESHOLD))
    ):
        raise ValueError("frozen-relative primitive readout inputs differ")
    probability = torch.where(candidates, relevance, torch.zeros_like(relevance))
    union = greedy_novelty_union_readout(
        probability,
        region_rows=rows,
        core_mask=core,
        num_primitives=int(valid.numel()),
        config=MultiRegionUnionConfig(
            score_threshold=SCORE_THRESHOLD,
            maximum_regions=MAXIMUM_REGIONS,
            candidate_chunk_rows=CANDIDATE_CHUNK_ROWS,
        ),
    )
    for query, selected in enumerate(union.selected_region_indices):
        if any(
            not bool(candidates[index, query])
            or not bool(eligibility[index, query])
            for index in selected
        ):
            raise RuntimeError("greedy union escaped selected-scale eligibility")
    membership = union.primitive_membership.clone()
    removed = int(membership[~valid].count_nonzero())
    membership[~valid] = 0.0
    if (
        any(len(indices) > MAXIMUM_REGIONS for indices in union.selected_region_indices)
        or bool(membership[~valid].count_nonzero())
    ):
        raise RuntimeError("frozen-relative primitive union invariant failed")
    return FrozenRelativePrimitiveReadout(
        candidate_probability=probability.contiguous(),
        primitive_valid=valid,
        primitive_membership=membership.contiguous(),
        selected_region_indices=union.selected_region_indices,
        selected_region_scores=union.selected_region_scores,
        selected_marginal_core_rows=union.selected_marginal_core_rows,
        invalid_primitive_memberships_removed=removed,
    )


def channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: tensor_sha256(torch.as_tensor(value[name]))
        for name in (
            "canonical_region_indices",
            "selected_scale_indices",
            "selected_scale_eligibility",
            "relative_relevance",
            "unary_candidate_mask",
            "candidate_probability",
            "region_rows",
            "token_mask",
            "primitive_valid",
            "primitive_membership",
        )
    }


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
        "selected_scale_indices",
        "selected_scale_eligibility",
        "relative_relevance",
        "unary_candidate_mask",
        "candidate_probability",
        "region_rows",
        "token_mask",
        "primitive_valid",
        "primitive_membership",
        "selected_region_indices",
        "selected_region_scores",
        "selected_marginal_core_rows",
        "audit",
        "channel_sha256",
        "access_audit",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("frozen-relative primitive authority fields differ")
    payload = dict(value)
    replay = frozen_relative_primitive_readout(
        relative_relevance=payload["relative_relevance"],
        selected_scale_eligibility=payload["selected_scale_eligibility"],
        unary_candidate_mask=payload["unary_candidate_mask"],
        region_rows=payload["region_rows"],
        token_mask=payload["token_mask"],
        primitive_valid=payload["primitive_valid"],
    )
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    selected_scales = torch.as_tensor(payload["selected_scale_indices"])
    query_count = int(replay.primitive_membership.shape[1])
    if (
        payload.get("schema") != READOUT_SCHEMA
        or payload.get("schema_version") != READOUT_SCHEMA_VERSION
        or payload.get("contract") != readout_contract()
        or payload.get("contract_sha256") != READOUT_CONTRACT_SHA256
        or payload.get("access_audit") != access_audit()
        or int(payload.get("query_axis_count", -1)) != query_count
        or canonical.dtype != torch.int64
        or canonical.ndim != 1
        or canonical.numel() != replay.candidate_probability.shape[0]
        or selected_scales.dtype != torch.int64
        or selected_scales.shape != (query_count,)
        or not torch.equal(
            torch.as_tensor(payload["candidate_probability"]),
            replay.candidate_probability,
        )
        or not torch.equal(
            torch.as_tensor(payload["primitive_membership"]),
            replay.primitive_membership,
        )
        or tuple(tuple(v) for v in payload["selected_region_indices"])
        != replay.selected_region_indices
        or tuple(tuple(v) for v in payload["selected_region_scores"])
        != replay.selected_region_scores
        or tuple(tuple(v) for v in payload["selected_marginal_core_rows"])
        != replay.selected_marginal_core_rows
        or payload.get("channel_sha256") != channel_sha256(payload)
    ):
        raise ValueError("frozen-relative primitive authority differs")
    maximum = max(len(value) for value in replay.selected_region_indices)
    selected_total = sum(len(value) for value in replay.selected_region_indices)
    expected_audit = {
        "opaque_query_axes": query_count,
        "query_gate_passed": int(
            torch.as_tensor(payload["unary_candidate_mask"]).any(dim=0).sum()
        ),
        "maximum_union_regions": maximum,
        "selected_region_total": selected_total,
        "selected_cross_scale_regions": 0,
        "selected_non_candidate_regions": 0,
        "primitive_memberships": int(replay.primitive_membership.sum()),
        "invalid_primitive_memberships_removed": (
            replay.invalid_primitive_memberships_removed
        ),
        "graph_or_relation_applied": False,
        "query_identifiers_consumed": False,
        "target_metric_computed": False,
    }
    if payload.get("audit") != expected_audit:
        raise ValueError("frozen-relative primitive audit differs")
    return payload


__all__ = [
    "CANDIDATE_CHUNK_ROWS",
    "MAXIMUM_REGIONS",
    "READOUT_CONTRACT_SHA256",
    "READOUT_SCHEMA",
    "READOUT_SCHEMA_VERSION",
    "SCORE_THRESHOLD",
    "FrozenRelativePrimitiveReadout",
    "access_audit",
    "channel_sha256",
    "frozen_relative_primitive_readout",
    "readout_contract",
    "validate_readout_authority",
]
