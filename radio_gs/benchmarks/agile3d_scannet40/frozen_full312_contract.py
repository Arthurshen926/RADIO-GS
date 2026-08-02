"""Bindings shared by frozen Ours AGILE3D full312 shards and their merger."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


FROZEN_FULL312_SCHEMA = "ours-agile3d-scannet40-full312-frozen-v1"
FROZEN_FULL312_SCENE_COUNT = 312
FROZEN_FULL312_OBJECT_COUNT = 10_357

# These values determine method predictions or the exact callback/readout.
# Per-scene field/source identities intentionally live in scene_support: the
# 240/480/960 source ladder can select a different immutable field per scene
# without changing the query-time method contract.
FROZEN_METHOD_CONTRACT_KEYS = (
    "field_checkpoint_name",
    "capability_cache_name",
    "support_graph_name",
    "reliability_cache_name",
    "observation_contract",
    "voxel_size_m",
    "max_clicks",
    "click_search_workers",
    "click_policy",
    "clicked_labels_forced",
    "test_set_calibration",
    "world_query",
    "selection_mode",
    "official_coordinate_contract",
    "observation_lift",
    "official_point_readout",
    "readout_candidate_k",
    "readout_support_threshold",
    "evaluation_voxel_size_m",
    "voxel_cell_variance_m2",
    "click_seed_kernel",
    "seed_candidate_k",
    "hard_seed_topk",
    "seed_temperature",
    "hard_seed_threshold",
    "hard_seed_conflict_policy",
    "hard_seed_conflict_margin",
    "prototype_count",
    "prototype_strategy",
    "support_gate_required",
    "minimum_support_fraction",
    "solver_type",
    "laplacian_weight",
    "cg_iterations",
    "support_threshold",
    "unary_edge_contrast",
    "world_point_prototype_mode",
    "world_point_max_prototypes",
    "world_point_prototype_weighting",
    "appearance_unary_weight",
    "boundary_unary_weight",
    "feature_calibration",
    "background_centroids",
    "background_negative_policy",
    "calibration_sample_size",
    "centroid_iterations",
    "score_calibration",
    "score_chunk_size",
    "channel_confidence_mode",
    "negative_spatial_mode",
    "negative_spatial_steps",
    "negative_spatial_decay",
    "spatial_log_weight",
    "spatial_floor",
    "point_readout_constraint",
    "requires_official_extracted_capability_teachers",
)

FROZEN_SOURCE_HASH_KEYS = (
    "field_source_contract_sha256",
    "field_source_frame_manifest_sha256",
    "field_checkpoint_sha256",
    "support_graph_sha256",
    "mpr_observation_contract_sha256",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase 64-character sha256")
    return digest


def build_frozen_method_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in FROZEN_METHOD_CONTRACT_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"frozen method protocol misses keys: {missing}")
    return {
        "schema": FROZEN_FULL312_SCHEMA,
        **{key: protocol[key] for key in FROZEN_METHOD_CONTRACT_KEYS},
    }


def bind_frozen_method_contract(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = build_frozen_method_contract(protocol)
    return contract, canonical_json_sha256(contract)


def build_source_contract_bindings(
    scene_support: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for record in scene_support:
        scene_id = str(record.get("scene_id", ""))
        if not scene_id:
            raise ValueError("source contract binding lacks scene_id")
        if scene_id in bindings:
            raise ValueError(f"duplicate source contract binding: {scene_id}")
        bindings[scene_id] = {
            key: require_sha256(
                record.get(key, ""), label=f"{scene_id}.{key}"
            )
            for key in FROZEN_SOURCE_HASH_KEYS
        }
        optional_reliability = str(
            record.get("primitive_reliability_cache_sha256", "")
        )
        if optional_reliability:
            bindings[scene_id]["primitive_reliability_cache_sha256"] = (
                require_sha256(
                    optional_reliability,
                    label=f"{scene_id}.primitive_reliability_cache_sha256",
                )
            )
    return dict(sorted(bindings.items()))


def source_contract_bindings_sha256(
    scene_support: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], str]:
    bindings = build_source_contract_bindings(scene_support)
    envelope = {"schema": FROZEN_FULL312_SCHEMA, "scenes": bindings}
    return bindings, canonical_json_sha256(envelope)
