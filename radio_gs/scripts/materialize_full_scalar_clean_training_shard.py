#!/usr/bin/env python3
"""Materialize one clean full-scalar trainer-v1 shard without model execution.

This CPU-only boundary consumes three independent, caller-SHA-bound scene
authorities: canonical AcceptedV2 regions/e0, an exact-marginal factorized
primitive state, and precomputed official multi-view SigLIP2 descriptors.  A
fourth authority, the complete 24/8 cohort region/view registry, makes the two
trainer manifests global rather than accidentally scene-local.

The materializer never opens benchmark data, labels, masks, or text queries;
it also never runs AcceptedV2, RADIO, or the SigLIP2 head.  Missing real
authorities fail closed and cannot produce a placeholder shard.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.interfaces.factorized_primitive_state import (
    FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256,
    load_factorized_primitive_state,
)
from radio_gs.interfaces.full_scalar_sparse_teacher_selection import (
    REGION_CAP_PER_SCENE,
    SAMPLING_CONTRACT_SHA256,
    VIEW_CAP_PER_REGION,
    region_identity,
    sampling_contract,
    validate_selection_audit,
    validate_sparse_pair_cardinality,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    aggregate_surface_region_full_scalars,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as trainer
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


ACCEPTED_REGION_SCHEMA = (
    "radio_gs.surface_region_accepted_v2_canonical_region_authority.v2"
)
TEACHER_OBSERVATION_SCHEMA = (
    "radio_gs.surface_region_official_siglip2_multiview_teacher_authority.v2"
)
COHORT_REGISTRY_SCHEMA = (
    "radio_gs.surface_region_full_scalar_clean_cohort_region_view_registry.v1"
)
MATERIALIZATION_RECEIPT_SCHEMA = (
    "radio_gs.surface_region_full_scalar_clean_shard_materialization_receipt.v1"
)
SCHEMA_VERSION = 1
ACCEPTED_REGION_SCHEMA_VERSION = 2
TEACHER_OBSERVATION_SCHEMA_VERSION = 2
TEACHER_REGION_CAP_PER_SCENE = REGION_CAP_PER_SCENE
TEACHER_VIEW_CAP_PER_REGION = VIEW_CAP_PER_REGION
OFFICIAL_RADIO_CHECKPOINT_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _authority_content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("authority_sha256", None)
    return canonical_json_sha256(content)


def _authority_access(*, source_rgb_used: bool) -> dict[str, bool]:
    return {
        "source_rgb_used": bool(source_rgb_used),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "target_heldout_opened": False,
        "text_queries_opened": False,
    }


def accepted_region_authority_contract() -> dict[str, Any]:
    return {
        "schema": ACCEPTED_REGION_SCHEMA,
        "schema_version": ACCEPTED_REGION_SCHEMA_VERSION,
        "producer": "materialize_accepted_v2_canonical_region_authority",
        "accepted_v2_authority": trainer._accepted_v2_authority(),
        "storage": "sampling_bound_at_most_4096_canonical_regions",
        "sampling": official_teacher_sampling_contract(),
        "canonical_order": "ascending_full_canonical_region_index",
        "e0": "official_siglip2_summary_descriptor_float32_unit_l2",
        "region_padding": "token_mask_false_region_row_exact_minus_one",
        "stable_region_id": "scene_plus_sha256_of_canonical_region_identity",
        "semantic_cache_final_output_allowed": False,
        "input_authority": (
            "caller_sha_bound_factorized_geometry_support_graph_accepted_v2_"
            "checkpoint_official_summary_head_and_exact_responsibility_selection"
        ),
        "query_independent": True,
    }


def accepted_region_official_head_authority() -> dict[str, Any]:
    """Return the singleton official head used to turn V2 tokens into e0."""

    return {
        "radio_checkpoint_sha256": OFFICIAL_RADIO_CHECKPOINT_SHA256,
        "adaptor": "clip",
        "summary_head": "official_radio_siglip2_g_summary_projection",
        "descriptor_dim": trainer.DESCRIPTOR_DIM,
        "query_independent": True,
    }


def validate_accepted_region_input_authority(
    value: object,
    *,
    geometry_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the caller-SHA lineage absent from legacy semantic caches."""

    if not isinstance(value, Mapping) or set(value) != {
        "geometry_authority",
        "support_graph_authority",
        "selection_authority",
        "accepted_v2_checkpoint_authority",
        "official_summary_head_authority",
    }:
        raise ValueError("AcceptedV2 canonical region input authority differs")
    authority = dict(value)
    geometry = authority["geometry_authority"]
    support = authority["support_graph_authority"]
    selection = authority["selection_authority"]
    accepted = authority["accepted_v2_checkpoint_authority"]
    head = authority["official_summary_head_authority"]
    if not isinstance(geometry, Mapping) or set(geometry) != {
        "kind",
        "factorized_primitive_state_file_sha256",
        "factorized_primitive_state_contract_sha256",
        "factorized_field_checkpoint_file_sha256",
        "factorized_radio_cache_file_sha256",
        "primitive_row_authority_sha256",
        "geometry_fingerprint",
    }:
        raise ValueError("AcceptedV2 geometry input authority differs")
    if (
        geometry.get("kind") != "factorized_primitive_state_v2"
        or geometry.get("factorized_primitive_state_contract_sha256")
        != FACTORIZED_PRIMITIVE_STATE_CONTRACT_SHA256
        or geometry.get("geometry_fingerprint") != dict(geometry_fingerprint)
    ):
        raise ValueError("AcceptedV2 geometry input contract differs")
    for key in (
        "factorized_primitive_state_file_sha256",
        "factorized_field_checkpoint_file_sha256",
        "factorized_radio_cache_file_sha256",
        "primitive_row_authority_sha256",
    ):
        _require_sha256(geometry.get(key), label=f"AcceptedV2 geometry {key}")
    if not isinstance(support, Mapping) or set(support) != {
        "kind",
        "support_graph_file_sha256",
        "primitive_row_authority_sha256",
    }:
        raise ValueError("AcceptedV2 support graph input authority differs")
    if (
        support.get("kind") != "canonical_query_free_support_graph_v1"
        or support.get("primitive_row_authority_sha256")
        != geometry.get("primitive_row_authority_sha256")
    ):
        raise ValueError("AcceptedV2 support graph row authority differs")
    _require_sha256(
        support.get("support_graph_file_sha256"),
        label="AcceptedV2 support graph file",
    )
    if not isinstance(selection, Mapping) or set(selection) != {
        "kind",
        "exact_marginal_responsibility_authority_file_sha256",
        "exact_marginal_formula_sha256",
        "responsibility_view_records_sha256",
        "sampling_contract_sha256",
    }:
        raise ValueError("AcceptedV2 selection input authority differs")
    if (
        selection.get("kind")
        != "exact_marginal_anchor_visibility_sparse_selection_v1"
        or selection.get("sampling_contract_sha256")
        != SAMPLING_CONTRACT_SHA256
    ):
        raise ValueError("AcceptedV2 selection contract differs")
    for key in (
        "exact_marginal_responsibility_authority_file_sha256",
        "exact_marginal_formula_sha256",
        "responsibility_view_records_sha256",
    ):
        _require_sha256(selection.get(key), label=f"AcceptedV2 selection {key}")
    if accepted != trainer._accepted_v2_authority():
        raise ValueError("AcceptedV2 checkpoint input authority differs")
    if head != accepted_region_official_head_authority():
        raise ValueError("AcceptedV2 official summary head authority differs")
    return {
        "geometry_authority": dict(geometry),
        "support_graph_authority": dict(support),
        "selection_authority": dict(selection),
        "accepted_v2_checkpoint_authority": dict(accepted),
        "official_summary_head_authority": dict(head),
    }


def teacher_observation_authority_contract() -> dict[str, Any]:
    return {
        "schema": TEACHER_OBSERVATION_SCHEMA,
        "schema_version": TEACHER_OBSERVATION_SCHEMA_VERSION,
        "producer": (
            "materialize_official_multiview_siglip2_teacher_authority"
        ),
        "descriptor": "float32_pair_by_1536_unit_l2",
        "descriptor_definition": (
            "accepted_region_exact_marginal_anchor_visible_rgb_crop_reencode_"
            "official_c_radio_first_summary_slot_then_official_siglip2_head"
        ),
        "storage": "sparse_coo_region_view_pairs_no_dense_R_by_V_tensor",
        "row_alignment": "sampled_accepted_region_indices_in_canonical_order",
        "sampling": official_teacher_sampling_contract(),
        "stable_view_id": "scene_plus_sha256_of_source_rgb_view_identity",
        "radio_summary_tensor_allowed": False,
        "whole_image_summary_substitution_allowed": False,
        "region_view_audit": (
            "per_pair_exact_crop_boxes_tlbr_positive_hit_counts_and_visible_"
            "primitive_counts"
        ),
        "online_model_execution_by_shard_materializer": False,
        "query_independent": True,
    }


def official_teacher_sampling_contract() -> dict[str, Any]:
    return sampling_contract()


def cohort_registry_contract() -> dict[str, Any]:
    return {
        "schema": COHORT_REGISTRY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cohort": "exactly_caller_bound_clean_24_train_8_validation",
        "scene_records": "exact_cohort_scene_set_sorted_by_scene_id",
        "artifact_binding": (
            "accepted_region_factorized_state_teacher_observation_file_sha256"
        ),
        "global_manifests": "deterministic_derivation_from_all_scene_records",
        "stable_ids": "globally_unique_region_ids_and_scene_scoped_view_ids",
        "nonvacuous_certificate_prerequisite": {
            "minimum_overlap_teacher_rows_per_train_scene": 1,
            "minimum_overlap_teacher_rows_per_validation_scene": 2,
            "claim_is_only_necessary_not_sufficient": True,
            "in_domain_after_train_normalization_checked_later": True,
        },
        "query_independent": True,
    }


def official_teacher_model_authority() -> dict[str, Any]:
    return {
        "kind": "c_radio_v4_h_official_siglip2_g_multiview_descriptor",
        "radio_checkpoint_sha256": OFFICIAL_RADIO_CHECKPOINT_SHA256,
        "adaptor": "clip",
        "summary_head": "official_radio_siglip2_g_summary_projection",
        "descriptor_dim": trainer.DESCRIPTOR_DIM,
        "input": "source_rgb_region_view",
        "query_independent": True,
    }


def _tensor_sha(value: torch.Tensor) -> str:
    return trainer._tensor_channel_sha256(value)


def accepted_region_channel_sha256(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "accepted_base_valid": _tensor_sha(value["accepted_base_valid"]),
        "canonical_region_indices": _tensor_sha(value["canonical_region_indices"]),
        "region_fingerprints": canonical_json_sha256(value["region_fingerprints"]),
        "region_rows": _tensor_sha(value["region_rows"]),
        "token_mask": _tensor_sha(value["token_mask"]),
        "anchor_index": _tensor_sha(value["anchor_index"]),
        "scale_indices": _tensor_sha(value["scale_indices"]),
        "accepted_v2_e0": _tensor_sha(value["accepted_v2_e0"]),
        "selection_audit": canonical_json_sha256(value["selection_audit"]),
    }


def teacher_observation_channel_sha256(
    value: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "region_fingerprints": canonical_json_sha256(
            value["region_fingerprints"]
        ),
        "canonical_region_indices": _tensor_sha(value["canonical_region_indices"]),
        "view_records": canonical_json_sha256(value["view_records"]),
        "pair_region_indices": _tensor_sha(value["pair_region_indices"]),
        "pair_view_indices": _tensor_sha(value["pair_view_indices"]),
        "pair_descriptors": _tensor_sha(value["pair_descriptors"]),
        "pair_crop_boxes_tlbr": _tensor_sha(
            value["pair_crop_boxes_tlbr"]
        ),
        "pair_support_hit_counts": _tensor_sha(
            value["pair_support_hit_counts"]
        ),
        "pair_visible_primitive_counts": _tensor_sha(
            value["pair_visible_primitive_counts"]
        ),
        "selection_audit": canonical_json_sha256(value["selection_audit"]),
    }


def official_teacher_descriptor_definition() -> dict[str, Any]:
    return {
        "region_rows": "AcceptedV2 canonical active global primitive rows",
        "visibility": (
            "positive_finite_exact_marginal_base_weight_hits_with_anchor_hit_"
            "required"
        ),
        "crop": (
            "tight_union_bbox_of_visible_region_hits_on_authority_grid_mapped_"
            "outward_to_source_rgb"
        ),
        "image_encoder": "official_c_radio_v4_h_crop_reencode",
        "summary_token": "genuine_first_1280d_radio_summary_slot_of_crop",
        "projection": "official_checkpoint__heads_siglip2_g",
        "normalization": "float32_unit_l2",
        "whole_image_summary_substitution_allowed": False,
        "precomputed_radio_summary_allowed": False,
        "semantic_cache_final_output_allowed": False,
        "query_independent": True,
    }


def validate_official_teacher_input_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "source_rgb_scene_authority_file_sha256",
        "source_rgb_scene_authority_content_sha256",
        "factorized_primitive_state_file_sha256",
        "accepted_region_authority_file_sha256",
        "accepted_region_channel_sha256",
        "accepted_region_fingerprints_sha256",
        "exact_marginal_responsibility_authority_file_sha256",
        "official_radio_checkpoint_file_sha256",
        "descriptor_definition",
    }:
        raise ValueError("official teacher input authority differs")
    authority = dict(value)
    for key in (
        "source_rgb_scene_authority_file_sha256",
        "source_rgb_scene_authority_content_sha256",
        "factorized_primitive_state_file_sha256",
        "accepted_region_authority_file_sha256",
        "accepted_region_channel_sha256",
        "accepted_region_fingerprints_sha256",
        "exact_marginal_responsibility_authority_file_sha256",
        "official_radio_checkpoint_file_sha256",
    ):
        _require_sha256(authority.get(key), label=f"official teacher {key}")
    if (
        authority["official_radio_checkpoint_file_sha256"]
        != OFFICIAL_RADIO_CHECKPOINT_SHA256
        or authority.get("descriptor_definition")
        != official_teacher_descriptor_definition()
    ):
        raise ValueError("official teacher descriptor input contract differs")
    return authority


def _canonical_region_identity(
    scene_id: str,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    anchor_index: torch.Tensor,
    scale_indices: torch.Tensor,
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for index in range(region_rows.shape[0]):
        active = region_rows[index][token_mask[index]].tolist()
        anchor = int(region_rows[index, int(anchor_index[index])])
        identities.append(
            region_identity(
                scene_id=scene_id,
                scale_index=int(scale_indices[index]),
                anchor_global_row=anchor,
                active_global_rows=active,
            )
        )
    return identities


def stable_region_fingerprints(value: Mapping[str, Any]) -> list[str]:
    return [
        canonical_json_sha256(identity)
        for identity in _canonical_region_identity(
            str(value["scene_id"]),
            value["region_rows"],
            value["token_mask"],
            value["anchor_index"],
            value["scale_indices"],
        )
    ]


def stable_region_id(scene_id: str, fingerprint: str) -> str:
    return f"{scene_id}:accepted-v2-canonical-v1:{fingerprint}"


def stable_teacher_view_id(scene_id: str, record: Mapping[str, Any]) -> str:
    identity = {
        "scene_id": scene_id,
        "view_contract": "source-rgb-official-siglip2-multiview-v1",
        "frame_id": record["frame_id"],
        "source_relative_path": record["source_relative_path"],
        "source_image_sha256": record["source_image_sha256"],
        "field_frame_authority_sha256": record[
            "field_frame_authority_sha256"
        ],
    }
    return f"{scene_id}:source-rgb:{canonical_json_sha256(identity)}"


def validate_accepted_region_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("AcceptedV2 canonical region authority must be a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "scene_id", "physical_space_id", "accepted_v2_authority",
        "geometry_fingerprint", "accepted_base_valid", "canonical_region_indices",
        "region_fingerprints", "selection_audit", "region_rows",
        "token_mask", "anchor_index", "scale_indices", "accepted_v2_e0",
        "input_authority", "channel_sha256", "source_access",
    }
    if set(payload) != required:
        raise ValueError("AcceptedV2 canonical region authority fields differ")
    contract = accepted_region_authority_contract()
    scene = str(payload.get("scene_id", ""))
    if (
        payload.get("schema") != ACCEPTED_REGION_SCHEMA
        or payload.get("schema_version") != ACCEPTED_REGION_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256") != canonical_json_sha256(contract)
        or payload.get("accepted_v2_authority") != trainer._accepted_v2_authority()
        or payload.get("physical_space_id")
        != trainer.canonical_physical_space_id(scene)
        or payload.get("source_access") != _authority_access(source_rgb_used=False)
    ):
        raise ValueError("AcceptedV2 canonical region authority contract differs")
    valid = payload["accepted_base_valid"]
    canonical_indices = payload["canonical_region_indices"]
    fingerprints = payload["region_fingerprints"]
    rows = payload["region_rows"]
    mask = payload["token_mask"]
    anchor = payload["anchor_index"]
    scales = payload["scale_indices"]
    e0 = payload["accepted_v2_e0"]
    geometry = payload.get("geometry_fingerprint")
    count = int(valid.numel()) if torch.is_tensor(valid) and valid.ndim == 1 else -1
    regions = int(rows.shape[0]) if torch.is_tensor(rows) and rows.ndim == 2 else -1
    if (
        not torch.is_tensor(valid) or valid.dtype != torch.bool
        or not torch.is_tensor(canonical_indices)
        or canonical_indices.dtype != torch.long
        or canonical_indices.shape != (regions,)
        or regions > TEACHER_REGION_CAP_PER_SCENE
        or bool((canonical_indices < 0).any())
        or (regions > 1 and not bool(
            (canonical_indices[1:] > canonical_indices[:-1]).all()
        ))
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or any(_SHA256.fullmatch(str(value)) is None for value in fingerprints)
        or len(set(fingerprints)) != regions
        or not torch.is_tensor(rows) or rows.dtype != torch.long
        or regions <= 0 or rows.shape[1] <= 0
        or not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != rows.shape
        or not torch.is_tensor(anchor) or anchor.dtype != torch.long
        or anchor.shape != (regions,)
        or not torch.is_tensor(scales) or scales.dtype != torch.long
        or scales.shape != (regions,) or bool((scales < 0).any())
        or not torch.is_tensor(e0) or e0.dtype != torch.float32
        or e0.shape != (regions, trainer.DESCRIPTOR_DIM)
        or not bool(torch.isfinite(e0).all())
        or geometry
        != {
            "num_gaussians": count,
            "xyz_sha256": _require_sha256(
                geometry.get("xyz_sha256", "") if isinstance(geometry, Mapping) else "",
                label="AcceptedV2 geometry xyz",
            ),
        }
    ):
        raise ValueError("AcceptedV2 canonical region tensor layout differs")
    payload["input_authority"] = validate_accepted_region_input_authority(
        payload.get("input_authority"),
        geometry_fingerprint=geometry,
    )
    if bool((mask.sum(dim=1) <= 0).any()) or bool(rows[~mask].ne(-1).any()):
        raise ValueError("AcceptedV2 region padding or active support differs")
    active = rows[mask]
    if bool((active < 0).any()) or bool((active >= count).any()):
        raise ValueError("AcceptedV2 region row is outside geometry")
    for region in range(regions):
        values = rows[region][mask[region]]
        if len(set(values.tolist())) != values.numel():
            raise ValueError("AcceptedV2 region repeats an active primitive")
    if (
        bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(mask[torch.arange(regions), anchor].all())
    ):
        raise ValueError("AcceptedV2 region anchor differs")
    anchor_rows = rows[torch.arange(regions), anchor]
    if not bool(valid[anchor_rows].all()):
        raise ValueError("AcceptedV2 e0 row lacks valid base anchor")
    norms = torch.linalg.vector_norm(e0, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0.0):
        raise ValueError("AcceptedV2 e0 is not unit L2")
    identities = _canonical_region_identity(scene, rows, mask, anchor, scales)
    expected_fingerprints = [canonical_json_sha256(item) for item in identities]
    order = [
        (
            item["scale_index"], item["anchor_global_row"],
            tuple(item["active_global_rows"]),
        )
        for item in identities
    ]
    if (
        order != sorted(order)
        or len(set(order)) != len(order)
        or fingerprints != expected_fingerprints
    ):
        raise ValueError("AcceptedV2 canonical region order differs")
    audit = validate_selection_audit(
        payload.get("selection_audit"), selected_count=regions
    )
    if bool(
        (
            canonical_indices
            >= int(audit["canonical_candidate_region_count"])
        ).any()
    ):
        raise ValueError("AcceptedV2 canonical region index exceeds selection domain")
    if payload.get("channel_sha256") != accepted_region_channel_sha256(payload):
        raise ValueError("AcceptedV2 canonical region channel SHA-256 differs")
    return {
        **payload,
        "canonical_region_indices": canonical_indices.detach().cpu().contiguous(),
        "region_fingerprints": list(fingerprints),
        "selection_audit": audit,
    }


def validate_teacher_observation_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("official teacher authority must be a mapping")
    payload = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "scene_id", "physical_space_id", "source_rgb_scene_authority_sha256",
        "teacher_model_authority", "teacher_model_authority_sha256",
        "canonical_region_indices", "region_fingerprints", "view_records",
        "pair_region_indices", "pair_view_indices", "pair_descriptors",
        "pair_crop_boxes_tlbr", "pair_support_hit_counts",
        "pair_visible_primitive_counts", "selection_audit",
        "input_authority", "channel_sha256", "source_access",
    }
    if set(payload) != required:
        raise ValueError("official teacher authority fields differ")
    contract = teacher_observation_authority_contract()
    scene = str(payload.get("scene_id", ""))
    model = official_teacher_model_authority()
    if (
        payload.get("schema") != TEACHER_OBSERVATION_SCHEMA
        or payload.get("schema_version") != TEACHER_OBSERVATION_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256") != canonical_json_sha256(contract)
        or payload.get("physical_space_id")
        != trainer.canonical_physical_space_id(scene)
        or payload.get("teacher_model_authority") != model
        or payload.get("teacher_model_authority_sha256")
        != canonical_json_sha256(model)
        or payload.get("source_access") != _authority_access(source_rgb_used=True)
    ):
        raise ValueError("official teacher authority contract differs")
    _require_sha256(
        payload.get("source_rgb_scene_authority_sha256"),
        label="source RGB scene authority",
    )
    sampled = payload.get("canonical_region_indices")
    fingerprints = payload.get("region_fingerprints")
    views = payload.get("view_records")
    pair_rows = payload.get("pair_region_indices")
    pair_views = payload.get("pair_view_indices")
    descriptors = payload.get("pair_descriptors")
    boxes = payload.get("pair_crop_boxes_tlbr")
    hit_counts = payload.get("pair_support_hit_counts")
    primitive_counts = payload.get("pair_visible_primitive_counts")
    selected_count = int(sampled.numel()) if torch.is_tensor(sampled) else -1
    pair_count = int(pair_rows.numel()) if torch.is_tensor(pair_rows) else -1
    validate_sparse_pair_cardinality(
        selected_region_count=selected_count, pair_count=pair_count
    )
    if (
        not torch.is_tensor(sampled) or sampled.dtype != torch.long
        or sampled.ndim != 1 or not 0 < selected_count <= TEACHER_REGION_CAP_PER_SCENE
        or bool((sampled < 0).any())
        or (selected_count > 1 and not bool((sampled[1:] > sampled[:-1]).all()))
        or not isinstance(fingerprints, list) or len(fingerprints) != selected_count
        or any(_SHA256.fullmatch(str(value)) is None for value in fingerprints)
        or len(set(fingerprints)) != len(fingerprints)
        or not isinstance(views, list) or not views
        or not torch.is_tensor(pair_rows) or pair_rows.dtype != torch.long
        or pair_rows.ndim != 1 or pair_count <= 0
        or not torch.is_tensor(pair_views) or pair_views.dtype != torch.long
        or pair_views.shape != (pair_count,)
        or not torch.is_tensor(descriptors) or descriptors.dtype != torch.float32
        or descriptors.shape != (pair_count, trainer.DESCRIPTOR_DIM)
        or not bool(torch.isfinite(descriptors).all())
        or not torch.is_tensor(boxes) or boxes.dtype != torch.long
        or boxes.shape != (pair_count, 4)
        or not torch.is_tensor(hit_counts) or hit_counts.dtype != torch.long
        or hit_counts.shape != (pair_count,)
        or not torch.is_tensor(primitive_counts)
        or primitive_counts.dtype != torch.long
        or primitive_counts.shape != (pair_count,)
    ):
        raise ValueError("official teacher tensor or row alignment differs")
    payload["input_authority"] = validate_official_teacher_input_authority(
        payload.get("input_authority")
    )
    if (
        payload["input_authority"][
            "source_rgb_scene_authority_content_sha256"
        ]
        != payload["source_rgb_scene_authority_sha256"]
    ):
        raise ValueError("official teacher source RGB content authority differs")
    frozen_views: list[dict[str, Any]] = []
    for record in views:
        if not isinstance(record, Mapping) or set(record) != {
            "frame_id", "source_relative_path", "source_image_sha256",
            "field_frame_authority_sha256",
            "source_image_height", "source_image_width",
            "feature_grid_height", "feature_grid_width",
            "responsibility_view_index", "responsibility_view_file_sha256",
        }:
            raise ValueError("official teacher source view record differs")
        source = Path(str(record["source_relative_path"]))
        if source.is_absolute() or ".." in source.parts or not source.parts:
            raise ValueError("official teacher source view path is unsafe")
        frozen_views.append(
            {
                "frame_id": str(record["frame_id"]),
                "source_relative_path": source.as_posix(),
                "source_image_sha256": _require_sha256(
                    record["source_image_sha256"], label="source RGB image"
                ),
                "field_frame_authority_sha256": _require_sha256(
                    record["field_frame_authority_sha256"],
                    label="field frame authority",
                ),
                "source_image_height": int(record["source_image_height"]),
                "source_image_width": int(record["source_image_width"]),
                "feature_grid_height": int(record["feature_grid_height"]),
                "feature_grid_width": int(record["feature_grid_width"]),
                "responsibility_view_index": int(
                    record["responsibility_view_index"]
                ),
                "responsibility_view_file_sha256": _require_sha256(
                    record["responsibility_view_file_sha256"],
                    label="exact-marginal responsibility view",
                ),
            }
        )
    frame_ids = [record["frame_id"] for record in frozen_views]
    if (
        any(not value for value in frame_ids)
        or frame_ids != sorted(frame_ids)
        or len(set(frame_ids)) != len(frame_ids)
        or any(
            min(
                record["source_image_height"],
                record["source_image_width"],
                record["feature_grid_height"],
                record["feature_grid_width"],
            )
            <= 0
            for record in frozen_views
        )
        or [record["responsibility_view_index"] for record in frozen_views]
        != sorted(record["responsibility_view_index"] for record in frozen_views)
        or len(
            {
                record["responsibility_view_index"]
                for record in frozen_views
            }
        )
        != len(frozen_views)
    ):
        raise ValueError("official teacher source view order differs")
    view_ids = [stable_teacher_view_id(scene, record) for record in frozen_views]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("official teacher stable view IDs collide")
    if (
        bool((pair_rows < 0).any())
        or bool((pair_rows >= selected_count).any())
        or bool((pair_views < 0).any())
        or bool((pair_views >= len(frozen_views)).any())
        or (pair_count > 1 and bool((pair_rows[1:] < pair_rows[:-1]).any()))
        or bool((hit_counts <= 0).any())
        or bool((primitive_counts <= 0).any())
        or bool((boxes < 0).any())
        or bool((boxes[:, 2] <= boxes[:, 0]).any())
        or bool((boxes[:, 3] <= boxes[:, 1]).any())
    ):
        raise ValueError("official teacher region-view crop evidence differs")
    row_counts = torch.bincount(pair_rows, minlength=selected_count)
    if bool((row_counts <= 0).any()) or bool(
        (row_counts > TEACHER_VIEW_CAP_PER_REGION).any()
    ):
        raise ValueError("official teacher sparse row coverage differs")
    pair_keys = pair_rows * len(frozen_views) + pair_views
    if pair_keys.unique().numel() != pair_count:
        raise ValueError("official teacher repeats a sparse region-view pair")
    for pair_index, view_index in enumerate(pair_views.tolist()):
        record = frozen_views[int(view_index)]
        if (
            int(boxes[pair_index, 2]) > record["source_image_height"]
            or int(boxes[pair_index, 3]) > record["source_image_width"]
        ):
            raise ValueError("official teacher crop exceeds source RGB dimensions")
    norms = torch.linalg.vector_norm(descriptors, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-4, rtol=0.0):
        raise ValueError("official teacher descriptor is not unit L2")
    audit = payload.get("selection_audit")
    if not isinstance(audit, Mapping) or set(audit) != {
        "accepted_selection_audit", "pair_count", "maximum_views_per_region",
    }:
        raise ValueError("official teacher selection audit differs")
    accepted_audit = validate_selection_audit(
        audit.get("accepted_selection_audit"), selected_count=selected_count
    )
    if (
        int(audit.get("pair_count", -1)) != pair_count
        or int(audit.get("maximum_views_per_region", -1))
        != int(row_counts.max())
    ):
        raise ValueError("official teacher selection audit counts differ")
    if payload.get("channel_sha256") != teacher_observation_channel_sha256(payload):
        raise ValueError("official teacher channel SHA-256 differs")
    return {
        **payload,
        "canonical_region_indices": sampled.detach().cpu().contiguous(),
        "view_records": frozen_views,
        "pair_region_indices": pair_rows.detach().cpu().contiguous(),
        "pair_view_indices": pair_views.detach().cpu().contiguous(),
        "pair_descriptors": descriptors.detach().cpu().contiguous(),
        "pair_crop_boxes_tlbr": boxes.detach().cpu().contiguous(),
        "pair_support_hit_counts": hit_counts.detach().cpu().contiguous(),
        "pair_visible_primitive_counts": primitive_counts.detach().cpu().contiguous(),
        "selection_audit": {
            **dict(audit),
            "accepted_selection_audit": accepted_audit,
        },
    }


def source_state_artifact_sha256(
    *, accepted_region_file_sha256: str, factorized_state_file_sha256: str
) -> str:
    return canonical_json_sha256(
        {
            "contract": "accepted-v2-region-plus-exact-factorized-state-v1",
            "accepted_region_authority_file_sha256": _require_sha256(
                accepted_region_file_sha256, label="AcceptedV2 region file"
            ),
            "factorized_state_file_sha256": _require_sha256(
                factorized_state_file_sha256, label="factorized state file"
            ),
        }
    )


def validate_teacher_accepted_sampling_alignment(
    teacher: Mapping[str, Any], accepted: Mapping[str, Any]
) -> None:
    """Reject any drift in the shared sparse row-selection authority."""

    if (
        teacher["region_fingerprints"] != accepted["region_fingerprints"]
        or not torch.equal(
            teacher["canonical_region_indices"],
            accepted["canonical_region_indices"],
        )
        or teacher["selection_audit"]["accepted_selection_audit"]
        != accepted["selection_audit"]
    ):
        raise ValueError("teacher rows differ from AcceptedV2 canonical regions")


def validate_cohort_region_view_registry(
    value: object,
    *,
    cohort_authority: Mapping[str, Any] | None = None,
    cohort_authority_file_sha256: str = "",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("clean cohort region/view registry must be a mapping")
    registry = dict(value)
    required = {
        "schema", "schema_version", "contract", "contract_sha256",
        "cohort_authority_sha256", "cohort_authority_file_sha256",
        "teacher_model_authority_sha256", "scene_records", "source_access",
        "authority_sha256",
    }
    if set(registry) != required:
        raise ValueError("clean cohort region/view registry fields differ")
    contract = cohort_registry_contract()
    if (
        registry.get("schema") != COHORT_REGISTRY_SCHEMA
        or registry.get("schema_version") != SCHEMA_VERSION
        or registry.get("contract") != contract
        or registry.get("contract_sha256") != canonical_json_sha256(contract)
        or registry.get("source_access") != _authority_access(source_rgb_used=True)
        or registry.get("teacher_model_authority_sha256")
        != canonical_json_sha256(official_teacher_model_authority())
    ):
        raise ValueError("clean cohort region/view registry contract differs")
    _require_sha256(registry.get("cohort_authority_sha256"), label="cohort content")
    _require_sha256(registry.get("cohort_authority_file_sha256"), label="cohort file")
    records = registry.get("scene_records")
    if not isinstance(records, list) or len(records) != (
        trainer.TRAIN_SCENE_COUNT + trainer.VALIDATION_SCENE_COUNT
    ):
        raise ValueError("clean cohort registry must cover exact 24+8 scenes")
    frozen: list[dict[str, Any]] = []
    all_region_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "scene_id", "physical_space_id", "split",
            "accepted_region_authority_file_sha256",
            "factorized_state_file_sha256",
            "teacher_observation_authority_file_sha256",
            "source_state_artifact_sha256", "teacher_model_authority_sha256",
            "eligible_overlap_teacher_row_count", "region_records",
        }:
            raise ValueError("clean cohort registry scene record differs")
        scene = str(record["scene_id"])
        split = str(record["split"])
        accepted_sha = _require_sha256(
            record["accepted_region_authority_file_sha256"],
            label="registry AcceptedV2 region file",
        )
        state_sha = _require_sha256(
            record["factorized_state_file_sha256"],
            label="registry factorized state file",
        )
        if (
            record["physical_space_id"] != trainer.canonical_physical_space_id(scene)
            or split not in {"source_train", "source_validation"}
            or record["source_state_artifact_sha256"]
            != source_state_artifact_sha256(
                accepted_region_file_sha256=accepted_sha,
                factorized_state_file_sha256=state_sha,
            )
            or record["teacher_model_authority_sha256"]
            != registry["teacher_model_authority_sha256"]
        ):
            raise ValueError("clean cohort registry scene authority differs")
        _require_sha256(
            record["teacher_observation_authority_file_sha256"],
            label="registry teacher observation file",
        )
        region_records = record["region_records"]
        if not isinstance(region_records, list) or not region_records:
            raise ValueError("clean cohort registry region records differ")
        scene_regions: list[dict[str, Any]] = []
        eligible_count = 0
        for region in region_records:
            if not isinstance(region, Mapping) or set(region) != {
                "region_fingerprint", "region_row_id", "teacher_view_ids",
                "eligible_overlap_teacher",
            }:
                raise ValueError("clean cohort registry region record differs")
            fingerprint = _require_sha256(
                region["region_fingerprint"], label="registry region fingerprint"
            )
            row_id = stable_region_id(scene, fingerprint)
            views = region["teacher_view_ids"]
            eligible = region["eligible_overlap_teacher"]
            if (
                region["region_row_id"] != row_id
                or not isinstance(eligible, bool)
                or not isinstance(views, list)
                or any(
                    not isinstance(view, str)
                    or not view.startswith(f"{scene}:source-rgb:")
                    or _SHA256.fullmatch(view.rsplit(":", 1)[-1]) is None
                    for view in views
                )
                or len(set(views)) != len(views)
                or (eligible and not views)
                or (not eligible and bool(views))
            ):
                raise ValueError("clean cohort registry stable row/view ID differs")
            eligible_count += int(eligible)
            scene_regions.append(
                {
                    "region_fingerprint": fingerprint,
                    "region_row_id": row_id,
                    "teacher_view_ids": list(views),
                    "eligible_overlap_teacher": eligible,
                }
            )
            all_region_ids.append(row_id)
        if [item["region_row_id"] for item in scene_regions] != sorted(
            item["region_row_id"] for item in scene_regions
        ):
            raise ValueError("clean cohort registry region records are not sorted")
        minimum = 2 if split == "source_validation" else 1
        if eligible_count != int(record["eligible_overlap_teacher_row_count"]):
            raise ValueError("clean cohort registry eligible row count differs")
        if eligible_count < minimum:
            raise ValueError(
                "clean cohort registry is nonvacuous-certificate insufficient"
            )
        frozen.append({**record, "region_records": scene_regions})
    scenes = [str(record["scene_id"]) for record in frozen]
    if scenes != sorted(scenes) or len(set(scenes)) != len(scenes):
        raise ValueError("clean cohort registry scene order differs")
    if len(set(all_region_ids)) != len(all_region_ids):
        raise ValueError("clean cohort registry repeats a stable region ID")
    if len({str(record["physical_space_id"]) for record in frozen}) != len(frozen):
        raise ValueError("clean cohort registry repeats a physical space")
    if cohort_authority is not None:
        train = [record["scene_id"] for record in frozen if record["split"] == "source_train"]
        validation = [
            record["scene_id"]
            for record in frozen
            if record["split"] == "source_validation"
        ]
        if (
            train != cohort_authority["source_train_scene_ids"]
            or validation != cohort_authority["source_validation_scene_ids"]
            or registry["cohort_authority_sha256"]
            != cohort_authority["authority_sha256"]
            or registry["cohort_authority_file_sha256"]
            != _require_sha256(cohort_authority_file_sha256, label="cohort file")
        ):
            raise ValueError("clean cohort registry/cohort authority differs")
    expected_authority = _require_sha256(
        registry.get("authority_sha256"), label="cohort registry authority"
    )
    if _authority_content_sha256(registry) != expected_authority:
        raise ValueError("clean cohort registry content SHA-256 differs")
    return {**registry, "scene_records": frozen}


def build_cohort_region_view_registry(
    *,
    cohort_authority: Mapping[str, Any],
    cohort_authority_file_sha256: str,
    scene_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble a registry only after all 32 formal scene declarations exist."""

    cohort = trainer.validate_cohort_authority_payload(cohort_authority)
    payload: dict[str, Any] = {
        "schema": COHORT_REGISTRY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contract": cohort_registry_contract(),
        "contract_sha256": canonical_json_sha256(cohort_registry_contract()),
        "cohort_authority_sha256": cohort["authority_sha256"],
        "cohort_authority_file_sha256": _require_sha256(
            cohort_authority_file_sha256, label="cohort file"
        ),
        "teacher_model_authority_sha256": canonical_json_sha256(
            official_teacher_model_authority()
        ),
        "scene_records": sorted(
            [dict(record) for record in scene_records],
            key=lambda record: str(record.get("scene_id", "")),
        ),
        "source_access": _authority_access(source_rgb_used=True),
    }
    payload["authority_sha256"] = _authority_content_sha256(payload)
    return validate_cohort_region_view_registry(
        payload,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_authority_file_sha256,
    )


def derive_global_manifests(
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = validate_cohort_region_view_registry(registry)
    source = {
        "schema": trainer.SOURCE_STATE_MANIFEST_SCHEMA,
        "schema_version": trainer.SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract": trainer.source_state_manifest_contract(),
        "contract_sha256": trainer.SOURCE_STATE_MANIFEST_CONTRACT_SHA256,
        "scene_records": [
            {
                "scene_id": record["scene_id"],
                "physical_space_id": record["physical_space_id"],
                "artifact_sha256": record["source_state_artifact_sha256"],
            }
            for record in frozen["scene_records"]
        ],
        "region_records": sorted(
            [
                {
                    "region_row_id": region["region_row_id"],
                    "scene_id": record["scene_id"],
                }
                for record in frozen["scene_records"]
                for region in record["region_records"]
            ],
            key=lambda item: item["region_row_id"],
        ),
        "source_access": trainer._source_manifest_access(),
    }
    source["authority_sha256"] = trainer._manifest_content_sha256(source)
    source = trainer.validate_source_state_manifest(source)
    teacher = {
        "schema": trainer.TEACHER_MANIFEST_SCHEMA,
        "schema_version": trainer.SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract": trainer.teacher_manifest_contract(),
        "contract_sha256": trainer.TEACHER_MANIFEST_CONTRACT_SHA256,
        "teacher_model_authority_sha256": frozen[
            "teacher_model_authority_sha256"
        ],
        "region_view_records": sorted(
            [
                {
                    "region_row_id": region["region_row_id"],
                    "scene_id": record["scene_id"],
                    "teacher_view_ids": list(region["teacher_view_ids"]),
                }
                for record in frozen["scene_records"]
                for region in record["region_records"]
            ],
            key=lambda item: item["region_row_id"],
        ),
        "source_access": trainer._source_manifest_access(),
    }
    teacher["authority_sha256"] = trainer._manifest_content_sha256(teacher)
    teacher = trainer.validate_teacher_manifest(teacher)
    return source, teacher


def _load_torch_authority(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
    validator: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, observed, source = load_torch_mapping(
        path,
        expected_sha256=_require_sha256(expected_sha256, label=label),
        map_location="cpu",
        label=label,
    )
    return validator(payload), {
        "path": str(source), "sha256": observed, "size_bytes": source.stat().st_size
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    cohort, cohort_file = trainer.load_cohort_authority(
        args.cohort_authority,
        expected_sha256=args.expected_cohort_authority_sha256,
    )
    registry_value, registry_sha, registry_path = load_json_object(
        args.cohort_region_view_registry,
        expected_sha256=_require_sha256(
            args.expected_cohort_region_view_registry_sha256,
            label="cohort region/view registry",
        ),
        label="full-scalar clean cohort region/view registry",
    )
    registry = validate_cohort_region_view_registry(
        registry_value,
        cohort_authority=cohort,
        cohort_authority_file_sha256=cohort_file["sha256"],
    )
    accepted, accepted_file = _load_torch_authority(
        args.accepted_region_authority,
        args.expected_accepted_region_authority_sha256,
        label="AcceptedV2 canonical region authority",
        validator=validate_accepted_region_authority,
    )
    teacher, teacher_file = _load_torch_authority(
        args.teacher_observation_authority,
        args.expected_teacher_observation_authority_sha256,
        label="official multiview SigLIP2 teacher authority",
        validator=validate_teacher_observation_authority,
    )
    state = load_factorized_primitive_state(
        args.factorized_state,
        expected_sha256=_require_sha256(
            args.expected_factorized_state_sha256, label="factorized state"
        ),
        expected_field_checkpoint_sha256=_require_sha256(
            args.expected_field_checkpoint_sha256, label="field checkpoint"
        ),
        expected_factorized_radio_cache_sha256=_require_sha256(
            args.expected_factorized_radio_cache_sha256,
            label="factorized RADIO cache",
        ),
    )
    state_file = file_record(args.factorized_state)
    scene = accepted["scene_id"]
    if teacher["scene_id"] != scene:
        raise ValueError("AcceptedV2 and teacher scene IDs differ")
    train = cohort["source_train_scene_ids"]
    validation = cohort["source_validation_scene_ids"]
    if (scene in train) == (scene in validation):
        raise ValueError("materialization scene is not a unique cohort member")
    split = "source_train" if scene in train else "source_validation"
    if (
        state.metadata["geometry_fingerprint"] != accepted["geometry_fingerprint"]
        or state.valid.shape != accepted["accepted_base_valid"].shape
    ):
        raise ValueError("AcceptedV2 and exact-state geometry differs")
    fingerprints = accepted["region_fingerprints"]
    validate_teacher_accepted_sampling_alignment(teacher, accepted)
    teacher_inputs = teacher["input_authority"]
    if (
        teacher_inputs["accepted_region_authority_file_sha256"]
        != accepted_file["sha256"]
        or teacher_inputs["accepted_region_channel_sha256"]
        != canonical_json_sha256(accepted["channel_sha256"])
        or teacher_inputs["accepted_region_fingerprints_sha256"]
        != canonical_json_sha256(fingerprints)
    ):
        raise ValueError("teacher/AcceptedV2 caller-SHA lineage differs")
    summary = aggregate_surface_region_full_scalars(
        state,
        accepted["accepted_base_valid"],
        accepted["region_rows"],
        accepted["token_mask"],
        accepted["anchor_index"],
    )
    eligible = summary.use_full_scalar_mask.bool()
    pair_rows = teacher["pair_region_indices"]
    pair_counts = torch.bincount(pair_rows, minlength=len(fingerprints))
    if bool((pair_counts[~eligible] > 0).any()):
        raise ValueError("base-only region carries an official teacher observation")
    if bool((pair_counts[eligible] <= 0).any()):
        raise ValueError("exact overlap region lacks an official teacher observation")
    if not bool(eligible.all()):
        raise ValueError("sampled AcceptedV2 authority includes non-overlap rows")
    minimum = 2 if split == "source_validation" else 1
    if int(eligible.sum()) < minimum:
        raise ValueError("scene is nonvacuous-certificate insufficient")
    view_ids = [stable_teacher_view_id(scene, item) for item in teacher["view_records"]]
    region_ids = [stable_region_id(scene, value) for value in fingerprints]
    observed_regions = sorted(
        [
            {
                "region_fingerprint": fingerprint,
                "region_row_id": region_id,
                "teacher_view_ids": [
                    view_ids[int(view)]
                    for view in teacher["pair_view_indices"][pair_rows == row]
                ],
                "eligible_overlap_teacher": bool(eligible[row]),
            }
            for row, (fingerprint, region_id) in enumerate(zip(fingerprints, region_ids))
        ],
        key=lambda item: item["region_row_id"],
    )
    registry_scene = [
        record for record in registry["scene_records"] if record["scene_id"] == scene
    ]
    if len(registry_scene) != 1:
        raise ValueError("registry has no unique materialization scene")
    registry_scene = registry_scene[0]
    expected_source_sha = source_state_artifact_sha256(
        accepted_region_file_sha256=accepted_file["sha256"],
        factorized_state_file_sha256=state_file["sha256"],
    )
    if (
        registry_scene["split"] != split
        or registry_scene["accepted_region_authority_file_sha256"]
        != accepted_file["sha256"]
        or registry_scene["factorized_state_file_sha256"] != state_file["sha256"]
        or registry_scene["teacher_observation_authority_file_sha256"]
        != teacher_file["sha256"]
        or registry_scene["source_state_artifact_sha256"] != expected_source_sha
        or registry_scene["teacher_model_authority_sha256"]
        != teacher["teacher_model_authority_sha256"]
        or registry_scene["eligible_overlap_teacher_row_count"]
        != int(eligible.sum())
        or registry_scene["region_records"] != observed_regions
    ):
        raise ValueError("scene materialization differs from frozen cohort registry")
    source_manifest, teacher_manifest = derive_global_manifests(registry)
    return {
        "scene_id": scene,
        "physical_space_id": accepted["physical_space_id"],
        "split": split,
        "cohort": cohort,
        "cohort_file": cohort_file,
        "registry": registry,
        "registry_file": {
            "path": str(registry_path), "sha256": registry_sha,
            "size_bytes": registry_path.stat().st_size,
        },
        "accepted": accepted,
        "accepted_file": accepted_file,
        "state": state,
        "state_file": state_file,
        "teacher": teacher,
        "teacher_file": teacher_file,
        "summary": summary,
        "eligible": eligible,
        "region_ids": region_ids,
        "view_ids": view_ids,
        "source_manifest": source_manifest,
        "teacher_manifest": teacher_manifest,
        "nonvacuous_prerequisite": {
            "eligible_overlap_teacher_rows": int(eligible.sum()),
            "minimum_required": minimum,
            "passed": True,
            "training_certificate_claimed": False,
            "in_domain_after_normalization_pending": True,
        },
    }


def _require_outputs_absent(paths: Sequence[str | Path]) -> None:
    existing = [str(Path(path).resolve()) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "full-scalar materialization refuses to clobber outputs: "
            + ", ".join(existing)
        )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    outputs = [
        args.output_shard,
        args.output_source_state_manifest,
        args.output_teacher_manifest,
        args.output_receipt,
    ]
    _require_outputs_absent(outputs)
    prepared = preflight(args)
    if bool(getattr(args, "preflight_only", False)):
        return {
            "scene_id": prepared["scene_id"],
            "split": prepared["split"],
            "registry_file": prepared["registry_file"],
            "nonvacuous_prerequisite": prepared["nonvacuous_prerequisite"],
            "outputs_written": False,
        }

    write_frozen_json(
        args.output_source_state_manifest, prepared["source_manifest"]
    )
    write_frozen_json(args.output_teacher_manifest, prepared["teacher_manifest"])
    source_manifest_file = file_record(args.output_source_state_manifest)
    teacher_manifest_file = file_record(args.output_teacher_manifest)
    teacher = prepared["teacher"]
    pair_rows = teacher["pair_region_indices"].clone()
    pair_descriptors = teacher["pair_descriptors"].clone()
    pair_view_ids = [
        prepared["view_ids"][int(view)]
        for view in teacher["pair_view_indices"]
    ]
    shard: dict[str, Any] = {
        "schema": trainer.TRAINING_SHARD_SCHEMA,
        "schema_version": trainer.TRAINING_SHARD_SCHEMA_VERSION,
        "contract": trainer.training_shard_contract(),
        "contract_sha256": trainer.TRAINING_SHARD_CONTRACT_SHA256,
        "split": prepared["split"],
        "accepted_v2_e0": prepared["accepted"]["accepted_v2_e0"].clone(),
        "raw_full_scalar_summary": prepared["summary"].summary.float().clone(),
        "eligible": prepared["eligible"].clone(),
        "official_multiview_siglip2_teacher_pair_region_indices": pair_rows,
        "official_multiview_siglip2_teacher_pair_descriptors": pair_descriptors,
        "scene_ids": [prepared["scene_id"]] * len(prepared["region_ids"]),
        "region_row_ids": list(prepared["region_ids"]),
        "teacher_pair_view_ids": pair_view_ids,
        "sampling_audit": {
            "scene_id": prepared["scene_id"],
            "sampling_contract_sha256": SAMPLING_CONTRACT_SHA256,
            "canonical_region_indices_sha256": _tensor_sha(
                prepared["accepted"]["canonical_region_indices"]
            ),
            "accepted_selection_audit": dict(
                prepared["accepted"]["selection_audit"]
            ),
            "selected_region_count": len(prepared["region_ids"]),
            "pair_count": int(pair_rows.numel()),
            "maximum_views_per_region": int(
                torch.bincount(
                    pair_rows, minlength=len(prepared["region_ids"])
                ).max()
            ),
        },
        "lineage": {
            "accepted_v2_authority": trainer._accepted_v2_authority(),
            "source_state_cohort_authority_sha256": prepared[
                "source_manifest"
            ]["authority_sha256"],
            "source_state_manifest_file_sha256": source_manifest_file["sha256"],
            "cohort_authority_sha256": prepared["cohort"]["authority_sha256"],
            "cohort_authority_file_sha256": prepared["cohort_file"]["sha256"],
            "teacher_authority_sha256": prepared["teacher_manifest"][
                "authority_sha256"
            ],
            "teacher_manifest_file_sha256": teacher_manifest_file["sha256"],
        },
        "source_access": trainer._source_access(prepared["split"]),
    }
    shard["channel_sha256"] = trainer.training_shard_channel_sha256(shard)
    trainer.validate_training_shard_payload(shard, expected_split=prepared["split"])
    write_torch_noclobber(args.output_shard, shard)
    receipt = {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene_id": prepared["scene_id"],
        "physical_space_id": prepared["physical_space_id"],
        "split": prepared["split"],
        "inputs": {
            "cohort_authority": file_record(args.cohort_authority),
            "cohort_region_view_registry": prepared["registry_file"],
            "accepted_region_authority": prepared["accepted_file"],
            "factorized_state": prepared["state_file"],
            "teacher_observation_authority": prepared["teacher_file"],
            "implementation": file_record(Path(__file__).resolve()),
        },
        "outputs": {
            "training_shard": file_record(args.output_shard),
            "source_state_manifest": source_manifest_file,
            "teacher_manifest": teacher_manifest_file,
        },
        "nonvacuous_prerequisite": prepared["nonvacuous_prerequisite"],
        "source_access": {
            **trainer._source_manifest_access(),
            "source_rgb_opened_by_materializer": False,
            "online_model_execution": False,
        },
    }
    receipt["authority_sha256"] = _authority_content_sha256(receipt)
    write_frozen_json(args.output_receipt, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-authority", required=True)
    parser.add_argument("--expected-cohort-authority-sha256", required=True)
    parser.add_argument("--cohort-region-view-registry", required=True)
    parser.add_argument(
        "--expected-cohort-region-view-registry-sha256", required=True
    )
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--expected-accepted-region-authority-sha256", required=True)
    parser.add_argument("--factorized-state", required=True)
    parser.add_argument("--expected-factorized-state-sha256", required=True)
    parser.add_argument("--expected-field-checkpoint-sha256", required=True)
    parser.add_argument("--expected-factorized-radio-cache-sha256", required=True)
    parser.add_argument("--teacher-observation-authority", required=True)
    parser.add_argument(
        "--expected-teacher-observation-authority-sha256", required=True
    )
    parser.add_argument("--output-shard", required=True)
    parser.add_argument("--output-source-state-manifest", required=True)
    parser.add_argument("--output-teacher-manifest", required=True)
    parser.add_argument("--output-receipt", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(materialize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
