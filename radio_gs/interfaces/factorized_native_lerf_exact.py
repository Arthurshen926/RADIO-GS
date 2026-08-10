"""Independent LERF exact-query contracts for factorized-native descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import math
import re
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import factorized_native_target_descriptor as target_formal
from radio_gs.interfaces import factorized_native_target_health as health_formal
from radio_gs.losses import source_global_response_listwise_loss_v21 as relevance_loss
from radio_gs.querying import unified_query
from radio_gs.querying import v21_absolute_relevance_adapter as relevance_adapter
from radio_gs.scripts import build_lerf_region_comembership_external_cache_v21 as union
from radio_gs.scripts import eval_lerf_direct_3d_selection as evaluator
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


QUERY_EXECUTION_SCHEMA = "radio_gs.factorized_native_lerf_exact_query_execution.v1"
QUERY_RELEVANCE_SCHEMA = "radio_gs.factorized_native_lerf_exact_relevance.v1"
EXACT_QUERY_MANIFEST_SCHEMA = "radio_gs.lerf_exact_scene_query_manifest.v1"
EXTERNAL_EXECUTION_SCHEMA = "radio_gs.factorized_native_lerf_external_execution.v1"
EXTERNAL_CACHE_SCHEMA = "radio_gs.factorized_native_lerf_external_scores.v1"
METRIC_EXECUTION_SCHEMA = "radio_gs.factorized_native_lerf_metric_execution.v1"
FROZEN_ALL_QUERY_CACHE = {
    "path": "/root/RADIO-GS/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt",
    "sha256": "d0f70797d01cad76e8a12e69c71730fcdfd867e50c3c4b53e3f7bf797e36506d",
}
FROZEN_CANONICAL_NEGATIVE_BANK = {
    "path": "/root/RADIO-GS/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt",
    "sha256": "18d2aac56b50a9670ffe04b397d23a4652dd44fe8f18ed7a309a82b6c1102b67",
}
FROZEN_PREREGISTRATION = {
    "path": (
        "/root/RADIO-GS/paper/artifacts/"
        "lerf_v21_absolute_relevance_greedy_novelty_union_preregistration_20260807.json"
    ),
    "sha256": "32aeaedd72f667593b1ca64c582f2c3bdde19792b33065fd5047d7e36c299c60",
}
FROZEN_EVALUATOR = {
    "path": "/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py",
    "sha256": "8cb39acf08c4f90f6339002ef32022437f67e7cdaae80fc16206b49abbb917d5",
}
FROZEN_QUERY_DEPENDENCIES = {
    "target_descriptor_interface": Path(target_formal.__file__).resolve(),
    "target_health_interface": Path(health_formal.__file__).resolve(),
    "target_health_materializer": health_formal.HEALTH_AUDIT_IMPLEMENTATION_PATH,
    "absolute_relevance_adapter": Path(relevance_adapter.__file__).resolve(),
    "shared_exact_cosine_scorer": Path(unified_query.__file__).resolve(),
    "source_relevance_loss": Path(relevance_loss.__file__).resolve(),
}
FROZEN_EXTERNAL_DEPENDENCIES = {
    "greedy_novelty_union": Path(union.__file__).resolve(),
}
METRIC_PROTOCOL = {
    "protocol_preset": "vala_paper_3d",
    "selection_mode": "score_threshold",
    "score_threshold": 0.6,
    "score_postprocess": "none",
    "projection_mode": "selected_only_alpha",
    "silhouette_threshold": 10.0 / 255.0,
    "alpha_binarization": "png_uint8_gt10",
    "mask_refinement": "none",
    "official_frames_only": True,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    path, digest = str(value["path"]), str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} file record differs")
    return {"path": path, "sha256": digest}


def canonical_output(value: object, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be absolute and canonical")
    return resolved


def query_access_audit() -> dict[str, bool]:
    return {
        "all_source_arms_validated_before_query_files": True,
        "factorized_native_target_descriptor_opened": True,
        "benchmark_queries_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def external_access_audit() -> dict[str, bool]:
    return {
        **query_access_audit(),
        "query_relevance_opened": True,
        "query_independent_comembership_opened": True,
        "renderer_geometry_opened": True,
        "legacy_o0_query_scores_opened": False,
    }


def query_contract() -> dict[str, Any]:
    return {
        "schema": QUERY_RELEVANCE_SCHEMA,
        "schema_version": 1,
        "descriptor": (
            "factorized_native_target_descriptor_exact_query_view_unit_l2_siglip2"
        ),
        "positive_text": "frozen_exact_scene_subset_of_official_all_query_bank",
        "canonical_negative": "frozen_four_row_official_siglip2_bank",
        "formula": "binary_softmax_positive_vs_max_canonical_negative",
        "scorer": "existing_calibrated_v21_absolute_relevance",
        "logit_scale": relevance_loss.INFERENCE_LOGIT_SCALE,
        "assume_normalized": True,
        "postprocess": "none",
        "absolute_relevance_boundary": union.V21_ABSOLUTE_RELEVANCE_BOUNDARY,
        "query_smoothing": False,
        "scene_minmax_remap": False,
        "query_ranking_normalization": False,
        "metric_access": False,
    }


QUERY_CONTRACT_SHA256 = canonical_json_sha256(query_contract())


def validate_exact_query_manifest(value: object, *, scene_id: str) -> dict[str, Any]:
    """Validate the method-independent manifest frozen before this candidate."""

    required = {
        "schema", "schema_version", "scene_id", "query_ids",
        "query_ids_sha256", "frozen_all_query_cache", "frozen_evaluator",
        "frozen_before_champion_query_execution",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("exact scene query manifest fields differ")
    result = dict(value)
    queries = result["query_ids"]
    if (
        result["schema"] != EXACT_QUERY_MANIFEST_SCHEMA
        or result["schema_version"] != 1
        or result["scene_id"] != scene_id
        or not isinstance(queries, list)
        or not queries
        or len(set(queries)) != len(queries)
        or any(not isinstance(item, str) or not item.strip() for item in queries)
        or result["query_ids_sha256"] != canonical_json_sha256(queries)
        or result["frozen_before_champion_query_execution"] is not True
    ):
        raise ValueError("exact scene query manifest identity differs")
    result["frozen_all_query_cache"] = record(
        result["frozen_all_query_cache"], label="manifest all-query cache"
    )
    result["frozen_evaluator"] = record(
        result["frozen_evaluator"], label="manifest evaluator"
    )
    return result


def query_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "region_row_ids": canonical_json_sha256(value["region_row_ids"]),
        "canonical_region_indices": tensor_sha256(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "query_ids": canonical_json_sha256(value["query_ids"]),
        "region_absolute_relevance": tensor_sha256(value["region_absolute_relevance"]),
    }


def validate_query_relevance(value: object) -> dict[str, Any]:
    required = {
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("factorized-native query relevance fields differ")
    payload = dict(value)
    if (
        payload["schema"] != QUERY_RELEVANCE_SCHEMA
        or payload["schema_version"] != 1
        or payload["contract"] != query_contract()
        or payload["contract_sha256"] != QUERY_CONTRACT_SHA256
        or payload["access_audit"] != query_access_audit()
    ):
        raise ValueError("factorized-native query relevance header differs")
    payload["producer"] = record(payload["producer"], label="query producer")
    payload["query_execution_authority"] = record(
        payload["query_execution_authority"], label="query execution authority"
    )
    inputs = payload["input_authority"]
    names = {
        "target_descriptor",
        "descriptor_health_audit",
        "exact_query_manifest",
        "positive_text_cache",
        "all_query_text_cache",
        "canonical_negative_bank",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("factorized-native query relevance inputs differ")
    payload["input_authority"] = {
        name: record(inputs[name], label=f"query input {name}")
        for name in sorted(names)
    }
    rows = payload["region_row_ids"]
    fingerprints = payload["region_fingerprints"]
    canonical = payload["canonical_region_indices"]
    queries = payload["query_ids"]
    relevance = payload["region_absolute_relevance"]
    regions = len(rows) if isinstance(rows, list) else -1
    query_count = len(queries) if isinstance(queries, list) else -1
    if (
        regions <= 0
        or query_count <= 0
        or len(set(rows)) != regions
        or any(not isinstance(item, str) or not item for item in rows)
        or len(set(queries)) != query_count
        or any(not isinstance(item, str) or not item for item in queries)
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or len(set(fingerprints)) != regions
        or any(_SHA256.fullmatch(str(item)) is None for item in fingerprints)
        or not torch.is_tensor(canonical)
        or canonical.dtype != torch.long
        or canonical.device.type != "cpu"
        or canonical.shape != (regions,)
        or not torch.is_tensor(relevance)
        or relevance.dtype != torch.float32
        or relevance.device.type != "cpu"
        or relevance.shape != (regions, query_count)
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0).any())
        or bool((relevance > 1).any())
    ):
        raise ValueError("factorized-native query relevance tensor differs")
    if payload["channel_sha256"] != query_channel_sha256(payload):
        raise ValueError("factorized-native query relevance SHA-256 differs")
    return payload


def external_contract() -> dict[str, Any]:
    return {
        "schema": EXTERNAL_CACHE_SCHEMA,
        "schema_version": 1,
        "region_readout": "frozen_v21_greedy_novelty_union",
        "semantic_boundary": union.V21_ABSOLUTE_RELEVANCE_BOUNDARY,
        "maximum_regions": union.V21_MAXIMUM_REGIONS,
        "legacy_o0_query_scores": False,
        "output": "binary_primitive_membership_in_exact_query_order",
        "frozen_metric_protocol": dict(METRIC_PROTOCOL),
    }


EXTERNAL_CONTRACT_SHA256 = canonical_json_sha256(external_contract())


def validate_external_cache(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "query_scores",
        "valid",
        "xyz",
        "metadata",
        "selection",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("factorized-native external cache fields differ")
    payload = dict(value)
    if (
        payload["schema"] != EXTERNAL_CACHE_SCHEMA
        or payload["schema_version"] != 1
        or payload["contract"] != external_contract()
        or payload["contract_sha256"] != EXTERNAL_CONTRACT_SHA256
    ):
        raise ValueError("factorized-native external cache contract differs")
    scores, valid, xyz = payload["query_scores"], payload["valid"], payload["xyz"]
    if (
        not torch.is_tensor(scores)
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or scores.ndim != 2
        or min(scores.shape) <= 0
        or not bool(torch.isfinite(scores).all())
        or bool(((scores != 0.0) & (scores != 1.0)).any())
        or not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or valid.device.type != "cpu"
        or valid.shape != (scores.shape[0],)
        or not torch.is_tensor(xyz)
        or xyz.dtype != torch.float32
        or xyz.device.type != "cpu"
        or xyz.shape != (scores.shape[0], 3)
        or not bool(torch.isfinite(xyz).all())
        or bool(scores[~valid].count_nonzero())
    ):
        raise ValueError("factorized-native external cache tensors differ")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "query_names",
        "score_semantics",
        "producer",
        "execution_authority",
    }:
        raise ValueError("factorized-native external cache metadata differs")
    queries = metadata["query_names"]
    if (
        not isinstance(queries, list)
        or len(queries) != scores.shape[1]
        or len(set(queries)) != len(queries)
        or any(not isinstance(item, str) or not item for item in queries)
        or metadata["score_semantics"]
        != "binary_factorized_native_absolute_relevance_greedy_novelty_union"
    ):
        raise ValueError("factorized-native external query semantics differ")
    metadata = dict(metadata)
    metadata["producer"] = record(metadata["producer"], label="external producer")
    metadata["execution_authority"] = record(
        metadata["execution_authority"], label="external execution"
    )
    payload["metadata"] = metadata
    selection = payload["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "region_indices",
        "region_scores",
        "marginal_core_rows",
        "invalid_memberships_removed",
    }:
        raise ValueError("factorized-native external selection differs")
    indices = selection["region_indices"]
    region_scores = selection["region_scores"]
    core_rows = selection["marginal_core_rows"]
    if (
        not isinstance(indices, list)
        or not isinstance(region_scores, list)
        or not isinstance(core_rows, list)
        or len(indices) != scores.shape[1]
        or len(region_scores) != scores.shape[1]
        or len(core_rows) != scores.shape[1]
    ):
        raise ValueError("factorized-native external selection query order differs")
    for query_index in range(scores.shape[1]):
        query_indices = indices[query_index]
        query_scores = region_scores[query_index]
        query_core_rows = core_rows[query_index]
        if (
            not isinstance(query_indices, list)
            or not isinstance(query_scores, list)
            or not isinstance(query_core_rows, list)
            or len(query_indices) != len(query_scores)
            or len(query_indices) != len(query_core_rows)
            or len(query_indices) > union.V21_MAXIMUM_REGIONS
            or any(type(item) is not int or item < 0 for item in query_indices)
            or len(set(query_indices)) != len(query_indices)
            or any(
                type(item) is not float
                or not math.isfinite(item)
                or item < 0.0
                or item > 1.0
                for item in query_scores
            )
            or any(type(item) is not int or item <= 0 for item in query_core_rows)
        ):
            raise ValueError("factorized-native external per-query selection differs")
    if (
        not isinstance(selection["invalid_memberships_removed"], int)
        or selection["invalid_memberships_removed"] < 0
    ):
        raise ValueError("factorized-native external invalid-row audit differs")
    return payload


__all__ = [
    "EXTERNAL_CACHE_SCHEMA",
    "EXTERNAL_CONTRACT_SHA256",
    "EXTERNAL_EXECUTION_SCHEMA",
    "EXACT_QUERY_MANIFEST_SCHEMA",
    "FROZEN_ALL_QUERY_CACHE",
    "FROZEN_CANONICAL_NEGATIVE_BANK",
    "FROZEN_EVALUATOR",
    "FROZEN_EXTERNAL_DEPENDENCIES",
    "FROZEN_PREREGISTRATION",
    "FROZEN_QUERY_DEPENDENCIES",
    "METRIC_EXECUTION_SCHEMA",
    "METRIC_PROTOCOL",
    "QUERY_CONTRACT_SHA256",
    "QUERY_EXECUTION_SCHEMA",
    "QUERY_RELEVANCE_SCHEMA",
    "canonical_output",
    "external_access_audit",
    "external_contract",
    "query_access_audit",
    "query_channel_sha256",
    "query_contract",
    "record",
    "validate_external_cache",
    "validate_exact_query_manifest",
    "validate_query_relevance",
]
