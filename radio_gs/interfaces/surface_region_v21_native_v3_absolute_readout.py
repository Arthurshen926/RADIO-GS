"""Formal query-opaque adapter for V2.1 absolute relevance and native V3.

The adapter consumes an already-materialized ``[region, opaque-query]``
absolute relevance tensor.  It never reads query identifiers or strings.  A
source-promoted native-V3 relation may only add bounded positive completion;
the existing deterministic novelty union then maps the final region field to
primitive membership.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import surface_region_v21_query_relevance as relevance_formal
from radio_gs.interfaces.region_comembership_native_v3_target import (
    FEATURE_SCHEMA,
    INFERENCE_SCHEMA,
    SCHEMA_VERSION as NATIVE_V3_SCHEMA_VERSION,
)
from radio_gs.querying.absolute_relevance_relation_readout import (
    READOUT_CONTRACT_SHA256 as ABSOLUTE_READOUT_CONTRACT_SHA256,
    absolute_relevance_relation_readout,
)
from radio_gs.querying.multi_region_union_readout import (
    MultiRegionUnionConfig,
    greedy_novelty_union_readout,
)
from radio_gs.scripts.infer_region_comembership_native_v3 import (
    validate_inference_authority,
)
from radio_gs.scripts.materialize_region_comembership_features_native_v3 import (
    validate_feature_authority,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = "radio_gs.surface_region_v21_native_v3_absolute_readout.v1"
READOUT_SCHEMA_VERSION = 1
ABSOLUTE_BOUNDARY = 0.5
RELATION_THRESHOLD = 0.85
MAXIMUM_REGIONS = 8
CANDIDATE_CHUNK_ROWS = 4096
SOURCE_SELECTED_METHOD = "dual_path_widest"
APPLIED_PATH_METHOD = "widest_path"


def access_audit() -> dict[str, bool]:
    return {
        "absolute_relevance_tensor_opened": True,
        "native_v3_feature_authority_opened": True,
        "native_v3_inference_authority_opened": True,
        "primitive_valid_opened": True,
        "query_axis_is_opaque": True,
        "query_identifiers_opened": False,
        "query_strings_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "per_scene_hyperparameters": False,
    }


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "input": {
            "absolute_relevance": "float32_closed_unit_interval_R_by_opaque_Q",
            "relation": "strict_validated_promoted_native_v3_authorities",
            "primitive_valid": "canonical_physical_primitive_axis",
        },
        "fixed_rule": {
            "source_selected_method": SOURCE_SELECTED_METHOD,
            "applied_path_method": APPLIED_PATH_METHOD,
            "relation_threshold": RELATION_THRESHOLD,
            "maximum_regions": MAXIMUM_REGIONS,
            "absolute_boundary": ABSOLUTE_BOUNDARY,
            "candidate_chunk_rows": CANDIDATE_CHUNK_ROWS,
        },
        "relation_readout_contract_sha256": ABSOLUTE_READOUT_CONTRACT_SHA256,
        "region_to_primitive": "deterministic_greedy_novelty_union",
        "invariants": {
            "final_relevance_not_below_absolute_unary": True,
            "failed_absolute_gate_exact_unary": True,
            "seed_exact_unary": True,
            "relation_support_bounded_by_eight": True,
            "union_regions_bounded_by_eight": True,
            "invalid_primitive_membership_zero": True,
            "fallback_pair_probability_consumed_without_replacement": True,
            "query_axis_opaque": True,
        },
        "legacy_pipeline_modified": False,
        "access_audit": access_audit(),
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class QueryOpaqueAbsoluteRelevance:
    scene_id: str
    physical_space_id: str
    canonical_region_indices: torch.Tensor
    region_fingerprints_sha256: str
    values: torch.Tensor


@dataclass(frozen=True)
class NativeV3AbsoluteReadout:
    absolute_relevance: torch.Tensor
    final_relevance: torch.Tensor
    seed_region_indices: torch.Tensor
    query_gate: torch.Tensor
    relation_selected_region_masks: torch.Tensor
    relation_path_support: torch.Tensor
    primitive_valid: torch.Tensor
    primitive_membership: torch.Tensor
    union_selected_region_indices: tuple[tuple[int, ...], ...]
    union_selected_region_scores: tuple[tuple[float, ...], ...]
    union_selected_marginal_core_rows: tuple[tuple[int, ...], ...]
    invalid_primitive_memberships_removed: int
    fallback_pair_count: int
    fallback_pairs_above_relation_threshold: int


def _sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} SHA-256 differs")
    return digest


def query_opaque_absolute_relevance_view(
    value: object,
) -> QueryOpaqueAbsoluteRelevance:
    """Validate only region channels; query identifiers remain unopened."""

    if not isinstance(value, Mapping):
        raise ValueError("V2.1 absolute relevance authority must be a mapping")
    required_keys = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "query_execution_authority",
        "input_authority",
        "region_row_ids",
        "canonical_region_indices",
        "region_fingerprints",
        "query_ids",
        "region_absolute_relevance",
        "channel_sha256",
        "access_audit",
    }
    channels = value.get("channel_sha256")
    if (
        set(value.keys()) != required_keys
        or value.get("schema") != relevance_formal.QUERY_RELEVANCE_SCHEMA
        or value.get("schema_version") != 1
        or value.get("contract") != relevance_formal.query_relevance_contract()
        or value.get("contract_sha256")
        != relevance_formal.QUERY_RELEVANCE_CONTRACT_SHA256
        or value.get("access_audit")
        != relevance_formal.query_relevance_access_audit()
        or not isinstance(value.get("scene_id"), str)
        or not value.get("scene_id")
        or not isinstance(value.get("physical_space_id"), str)
        or not value.get("physical_space_id")
        or not isinstance(channels, Mapping)
        or set(channels)
        != {
            "region_row_ids",
            "canonical_region_indices",
            "region_fingerprints",
            "query_ids",
            "region_absolute_relevance",
        }
    ):
        raise ValueError("V2.1 opaque absolute relevance identity differs")
    canonical = torch.as_tensor(value.get("canonical_region_indices"))
    relevance = torch.as_tensor(value.get("region_absolute_relevance"))
    regions = int(canonical.numel()) if canonical.ndim == 1 else -1
    if (
        regions <= 0
        or canonical.dtype != torch.int64
        or canonical.device.type != "cpu"
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or relevance.dtype != torch.float32
        or relevance.device.type != "cpu"
        or relevance.ndim != 2
        or relevance.shape[0] != regions
        or relevance.shape[1] <= 0
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0.0).any())
        or bool((relevance > 1.0).any())
        or channels["canonical_region_indices"] != tensor_sha256(canonical)
        or channels["region_absolute_relevance"] != tensor_sha256(relevance)
    ):
        raise ValueError("V2.1 opaque absolute relevance region channels differ")
    # The query_ids value and its channel are intentionally never accessed.
    fingerprint_sha = _sha256(
        channels["region_fingerprints"], label="region fingerprints"
    )
    return QueryOpaqueAbsoluteRelevance(
        scene_id=str(value.get("scene_id")),
        physical_space_id=str(value.get("physical_space_id")),
        canonical_region_indices=canonical.detach().clone().contiguous(),
        region_fingerprints_sha256=fingerprint_sha,
        values=relevance.detach().clone().contiguous(),
    )


def _validated_native_v3_binding(
    *,
    relevance: QueryOpaqueAbsoluteRelevance,
    feature_authority: Mapping[str, Any],
    inference_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    feature = validate_feature_authority(feature_authority)
    inference = validate_inference_authority(inference_authority)
    rule = inference.get("selected_rule")
    canonical = torch.as_tensor(feature.get("canonical_region_indices"))
    inference_canonical = torch.as_tensor(
        inference.get("canonical_region_indices")
    )
    pairs = torch.as_tensor(feature.get("pair_indices"))
    inference_pairs = torch.as_tensor(inference.get("pair_indices"))
    active = torch.as_tensor(feature.get("native_pair_active_mask"))
    inference_active = torch.as_tensor(inference.get("native_pair_active_mask"))
    fallback = torch.as_tensor(feature.get("legacy_v2_fallback_pair_mask"))
    inference_fallback = torch.as_tensor(
        inference.get("legacy_v2_fallback_pair_mask")
    )
    if (
        feature.get("schema") != FEATURE_SCHEMA
        or inference.get("schema") != INFERENCE_SCHEMA
        or feature.get("schema_version") != NATIVE_V3_SCHEMA_VERSION
        or inference.get("schema_version") != NATIVE_V3_SCHEMA_VERSION
        or feature.get("domain") != "target"
        or inference.get("domain") != "target"
        or feature.get("scene_id") != relevance.scene_id
        or inference.get("scene_id") != relevance.scene_id
        or feature.get("target_execution_authority")
        != inference.get("target_execution_authority")
        or feature.get("region_fingerprints_sha256")
        != relevance.region_fingerprints_sha256
        or inference.get("region_fingerprints_sha256")
        != relevance.region_fingerprints_sha256
        or not torch.equal(canonical, relevance.canonical_region_indices)
        or not torch.equal(inference_canonical, relevance.canonical_region_indices)
        or not torch.equal(pairs, inference_pairs)
        or not torch.equal(active, inference_active)
        or not torch.equal(fallback, inference_fallback)
        or not torch.equal(fallback, ~active)
        or rule
        != {
            "method": SOURCE_SELECTED_METHOD,
            "maximum_regions": MAXIMUM_REGIONS,
            "threshold": RELATION_THRESHOLD,
        }
    ):
        raise ValueError("V2.1 relevance/native-V3 canonical binding differs")
    return feature, inference


def apply_native_v3_absolute_readout(
    *,
    relevance: QueryOpaqueAbsoluteRelevance,
    feature_authority: Mapping[str, Any],
    inference_authority: Mapping[str, Any],
    primitive_valid: torch.Tensor,
) -> NativeV3AbsoluteReadout:
    """Apply monotone native-V3 completion then deterministic region union."""

    if not isinstance(relevance, QueryOpaqueAbsoluteRelevance):
        raise TypeError("relevance must be QueryOpaqueAbsoluteRelevance")
    feature, inference = _validated_native_v3_binding(
        relevance=relevance,
        feature_authority=feature_authority,
        inference_authority=inference_authority,
    )
    source_valid = torch.as_tensor(primitive_valid)
    if source_valid.dtype != torch.bool:
        raise ValueError("native-V3 primitive-valid canonical axis differs")
    valid = source_valid.detach().cpu().contiguous()
    rows = torch.as_tensor(feature["region_rows"]).detach().cpu().contiguous()
    core = torch.as_tensor(feature["token_mask"]).detach().bool().cpu().contiguous()
    if valid.ndim != 1 or valid.numel() <= 0 or rows.shape != core.shape:
        raise ValueError("native-V3 primitive-valid canonical axis differs")
    active_rows = rows[core]
    if (
        active_rows.numel() <= 0
        or bool((active_rows < 0).any())
        or bool((active_rows >= valid.numel()).any())
    ):
        raise ValueError("native-V3 region rows differ from primitive-valid axis")

    relation = absolute_relevance_relation_readout(
        region_absolute_relevance=relevance.values,
        pair_indices=inference["pair_indices"],
        pair_probabilities=inference["pair_probabilities"],
        absolute_boundary=ABSOLUTE_BOUNDARY,
        relation_threshold=RELATION_THRESHOLD,
        maximum_regions=MAXIMUM_REGIONS,
        path_method=APPLIED_PATH_METHOD,
    )
    union = greedy_novelty_union_readout(
        relation.final_relevance,
        region_rows=rows,
        core_mask=core,
        num_primitives=int(valid.numel()),
        config=MultiRegionUnionConfig(
            score_threshold=ABSOLUTE_BOUNDARY,
            maximum_regions=MAXIMUM_REGIONS,
            candidate_chunk_rows=CANDIDATE_CHUNK_ROWS,
        ),
    )
    membership = union.primitive_membership.clone()
    removed = int(membership[~valid].count_nonzero())
    membership[~valid] = 0.0
    fallback = torch.as_tensor(
        inference["legacy_v2_fallback_pair_mask"]
    ).bool()
    probabilities = torch.as_tensor(inference["pair_probabilities"]).float()
    if (
        relation.selected_region_masks.sum(dim=0).max().item() > MAXIMUM_REGIONS
        or any(
            len(indices) > MAXIMUM_REGIONS
            for indices in union.selected_region_indices
        )
        or not bool((relation.final_relevance >= relevance.values).all())
        or not torch.equal(
            relation.final_relevance[
                relation.seed_region_indices,
                torch.arange(relevance.values.shape[1]),
            ],
            relevance.values[
                relation.seed_region_indices,
                torch.arange(relevance.values.shape[1]),
            ],
        )
        or not torch.equal(
            relation.final_relevance[:, ~relation.query_gate],
            relevance.values[:, ~relation.query_gate],
        )
        or bool(membership[~valid].count_nonzero())
    ):
        raise RuntimeError("V2.1 native-V3 readout invariant failed")
    return NativeV3AbsoluteReadout(
        absolute_relevance=relation.absolute_relevance,
        final_relevance=relation.final_relevance,
        seed_region_indices=relation.seed_region_indices,
        query_gate=relation.query_gate,
        relation_selected_region_masks=relation.selected_region_masks,
        relation_path_support=relation.path_support,
        primitive_valid=valid,
        primitive_membership=membership.contiguous(),
        union_selected_region_indices=union.selected_region_indices,
        union_selected_region_scores=union.selected_region_scores,
        union_selected_marginal_core_rows=union.selected_marginal_core_rows,
        invalid_primitive_memberships_removed=removed,
        fallback_pair_count=int(fallback.sum()),
        fallback_pairs_above_relation_threshold=int(
            (fallback & (probabilities >= RELATION_THRESHOLD)).sum()
        ),
    )


def readout_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: tensor_sha256(value[name])
        for name in (
            "canonical_region_indices",
            "absolute_relevance",
            "final_relevance",
            "seed_region_indices",
            "query_gate",
            "relation_selected_region_masks",
            "relation_path_support",
            "primitive_valid",
            "primitive_membership",
        )
    }


def validate_readout_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("V2.1 native-V3 readout authority must be a mapping")
    payload = dict(value)
    tensor_names = {
        "canonical_region_indices",
        "absolute_relevance",
        "final_relevance",
        "seed_region_indices",
        "query_gate",
        "relation_selected_region_masks",
        "relation_path_support",
        "primitive_valid",
        "primitive_membership",
    }
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "producer",
        "input_authority",
        "region_fingerprints_sha256",
        "query_axis_count",
        "selected_rule",
        "applied_rule",
        *tensor_names,
        "union_selected_region_indices",
        "union_selected_region_scores",
        "union_selected_marginal_core_rows",
        "audit",
        "channel_sha256",
        "access_audit",
    }
    if (
        set(payload) != required
        or payload.get("schema") != READOUT_SCHEMA
        or payload.get("schema_version") != READOUT_SCHEMA_VERSION
        or payload.get("contract") != readout_contract()
        or payload.get("contract_sha256") != READOUT_CONTRACT_SHA256
        or payload.get("access_audit") != access_audit()
        or payload.get("selected_rule")
        != {
            "method": SOURCE_SELECTED_METHOD,
            "maximum_regions": MAXIMUM_REGIONS,
            "threshold": RELATION_THRESHOLD,
        }
        or payload.get("applied_rule")
        != {
            "path_method": APPLIED_PATH_METHOD,
            "maximum_regions": MAXIMUM_REGIONS,
            "relation_threshold": RELATION_THRESHOLD,
            "absolute_boundary": ABSOLUTE_BOUNDARY,
        }
        or payload.get("channel_sha256") != readout_channel_sha256(payload)
    ):
        raise ValueError("V2.1 native-V3 readout authority differs")
    unary = torch.as_tensor(payload["absolute_relevance"])
    final = torch.as_tensor(payload["final_relevance"])
    canonical = torch.as_tensor(payload["canonical_region_indices"])
    seeds = torch.as_tensor(payload["seed_region_indices"])
    gate = torch.as_tensor(payload["query_gate"])
    selected = torch.as_tensor(payload["relation_selected_region_masks"])
    support = torch.as_tensor(payload["relation_path_support"])
    valid = torch.as_tensor(payload["primitive_valid"])
    membership = torch.as_tensor(payload["primitive_membership"])
    query_count = int(payload.get("query_axis_count", -1))
    region_count = int(unary.shape[0]) if unary.ndim == 2 else -1
    union_indices = payload["union_selected_region_indices"]
    union_scores = payload["union_selected_region_scores"]
    union_marginals = payload["union_selected_marginal_core_rows"]
    audit = payload.get("audit")
    expected_audit_keys = {
        "opaque_query_axes",
        "query_gate_passed",
        "query_gate_failed_exact_unary",
        "maximum_relation_support_regions",
        "maximum_union_regions",
        "unary_decreases",
        "seed_changes",
        "fallback_pair_count",
        "fallback_pairs_above_relation_threshold",
        "invalid_primitive_memberships_removed",
        "query_identifiers_consumed",
        "query_strings_consumed",
        "target_metric_computed",
    }
    if (
        unary.dtype != torch.float32
        or unary.ndim != 2
        or query_count != unary.shape[1]
        or canonical.dtype != torch.int64
        or canonical.shape != (region_count,)
        or seeds.dtype != torch.int64
        or seeds.shape != (query_count,)
        or bool((seeds < 0).any())
        or bool((seeds >= region_count).any())
        or gate.dtype != torch.bool
        or gate.shape != (query_count,)
        or final.dtype != torch.float32
        or final.shape != unary.shape
        or selected.dtype != torch.bool
        or selected.shape != unary.shape
        or support.dtype != torch.float32
        or support.shape != unary.shape
        or not bool(torch.isfinite(support).all())
        or bool((support < 0.0).any())
        or bool((support > 1.0).any())
        or bool((final < unary).any())
        or bool((final > 1.0).any())
        or not torch.equal(
            final[seeds, torch.arange(query_count)],
            unary[seeds, torch.arange(query_count)],
        )
        or not torch.equal(final[:, ~gate], unary[:, ~gate])
        or bool((selected.sum(dim=0) > MAXIMUM_REGIONS).any())
        or valid.dtype != torch.bool
        or valid.ndim != 1
        or membership.dtype != torch.float32
        or membership.shape != (valid.numel(), query_count)
        or bool(((membership != 0.0) & (membership != 1.0)).any())
        or bool(membership[~valid].count_nonzero())
        or not isinstance(union_indices, (tuple, list))
        or not isinstance(union_scores, (tuple, list))
        or not isinstance(union_marginals, (tuple, list))
        or len(union_indices) != query_count
        or len(union_scores) != query_count
        or len(union_marginals) != query_count
        or any(
            len(rows) > MAXIMUM_REGIONS
            or len(rows) != len(scores)
            or len(rows) != len(marginals)
            or any(int(index) < 0 or int(index) >= region_count for index in rows)
            for rows, scores, marginals in zip(
                union_indices, union_scores, union_marginals
            )
        )
        or not isinstance(audit, Mapping)
        or set(audit) != expected_audit_keys
        or audit.get("opaque_query_axes") != query_count
        or audit.get("query_gate_passed") != int(gate.sum())
        or audit.get("query_gate_failed_exact_unary") != int((~gate).sum())
        or audit.get("maximum_relation_support_regions")
        != int(selected.sum(dim=0).max())
        or audit.get("maximum_union_regions")
        != max(len(rows) for rows in union_indices)
        or audit.get("unary_decreases") != 0
        or audit.get("seed_changes") != 0
        or audit.get("query_identifiers_consumed") is not False
        or audit.get("query_strings_consumed") is not False
        or audit.get("target_metric_computed") is not False
    ):
        raise ValueError("V2.1 native-V3 readout tensors differ")
    return payload


__all__ = [
    "ABSOLUTE_BOUNDARY",
    "APPLIED_PATH_METHOD",
    "MAXIMUM_REGIONS",
    "READOUT_CONTRACT_SHA256",
    "READOUT_SCHEMA",
    "READOUT_SCHEMA_VERSION",
    "RELATION_THRESHOLD",
    "SOURCE_SELECTED_METHOD",
    "NativeV3AbsoluteReadout",
    "QueryOpaqueAbsoluteRelevance",
    "access_audit",
    "apply_native_v3_absolute_readout",
    "query_opaque_absolute_relevance_view",
    "readout_channel_sha256",
    "readout_contract",
    "validate_readout_authority",
]
