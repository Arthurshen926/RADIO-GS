"""Target-safe AcceptedV2 canonical-region authority.

The frozen source authority deliberately accepts only ScanNet source-scene
identities.  This sibling schema preserves the exact AcceptedV2 tensors,
selection and input lineage while requiring an explicit caller-bound target
physical-space identity.  It contains no query, label, mask or metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import torch

from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


TARGET_ACCEPTED_V2_SCHEMA = "radio_gs.target_accepted_v2_canonical_region_authority.v1"
TARGET_ACCEPTED_V2_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def target_access_audit() -> dict[str, bool]:
    return {
        "query_independent": True,
        "target_geometry_authorities_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "target_metrics_computed": False,
    }


def target_accepted_v2_contract() -> dict[str, Any]:
    return {
        "schema": TARGET_ACCEPTED_V2_SCHEMA,
        "schema_version": TARGET_ACCEPTED_V2_SCHEMA_VERSION,
        "mathematical_payload": shard.accepted_region_authority_contract(),
        "mathematical_payload_sha256": canonical_json_sha256(
            shard.accepted_region_authority_contract()
        ),
        "identity": {
            "scene_id": "nonempty_caller_bound_target_scene_id",
            "physical_space_id": (
                "dataset_id:scene_id:geometry-checkpoint-sha256:digest"
            ),
            "scannet_name_inference_allowed": False,
        },
        "query_relevance_computed": False,
        "access_audit": target_access_audit(),
    }


TARGET_ACCEPTED_V2_CONTRACT_SHA256 = canonical_json_sha256(
    target_accepted_v2_contract()
)


def _identity(value: object, *, label: str) -> str:
    text = str(value)
    if not text or text != text.strip() or len(text) > 256 or "\x00" in text:
        raise ValueError(f"target AcceptedV2 {label} differs")
    return text


def target_physical_space_authority(
    *, dataset_id: object, scene_id: object, geometry_checkpoint_sha256: object
) -> dict[str, str]:
    dataset = _identity(dataset_id, label="dataset ID")
    scene = _identity(scene_id, label="scene ID")
    digest = str(geometry_checkpoint_sha256)
    if ":" in dataset or ":" in scene or _SHA256.fullmatch(digest) is None:
        raise ValueError("target AcceptedV2 physical-space authority differs")
    return {
        "kind": "target_geometry_checkpoint_v1",
        "dataset_id": dataset,
        "scene_id": scene,
        "geometry_checkpoint_sha256": digest,
        "physical_space_id": (f"{dataset}:{scene}:geometry-checkpoint-sha256:{digest}"),
    }


def _producer(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("target AcceptedV2 producer record differs")
    path = str(value["path"])
    digest = str(value["sha256"])
    if not Path(path).is_absolute() or _SHA256.fullmatch(digest) is None:
        raise ValueError("target AcceptedV2 producer record differs")
    return {"path": path, "sha256": digest}


def validate_target_accepted_v2_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("target AcceptedV2 authority must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "scene_id",
        "physical_space_id",
        "physical_space_authority",
        "producer",
        "accepted_v2_authority",
        "geometry_fingerprint",
        "accepted_base_valid",
        "canonical_region_indices",
        "region_fingerprints",
        "selection_audit",
        "region_rows",
        "token_mask",
        "anchor_index",
        "scale_indices",
        "accepted_v2_e0",
        "input_authority",
        "channel_sha256",
        "access_audit",
    }
    contract = target_accepted_v2_contract()
    if (
        set(payload) != required
        or payload.get("schema") != TARGET_ACCEPTED_V2_SCHEMA
        or payload.get("schema_version") != TARGET_ACCEPTED_V2_SCHEMA_VERSION
        or payload.get("contract") != contract
        or payload.get("contract_sha256") != TARGET_ACCEPTED_V2_CONTRACT_SHA256
        or payload.get("access_audit") != target_access_audit()
        or payload.get("accepted_v2_authority")
        != shard.trainer._accepted_v2_authority()
    ):
        raise ValueError("target AcceptedV2 contract differs")
    scene = _identity(payload["scene_id"], label="scene ID")
    physical = _identity(payload["physical_space_id"], label="physical-space ID")
    physical_raw = payload["physical_space_authority"]
    if not isinstance(physical_raw, Mapping) or set(physical_raw) != {
        "kind",
        "dataset_id",
        "scene_id",
        "geometry_checkpoint_sha256",
        "physical_space_id",
    }:
        raise ValueError("target AcceptedV2 physical-space authority differs")
    physical_authority = target_physical_space_authority(
        dataset_id=physical_raw["dataset_id"],
        scene_id=physical_raw["scene_id"],
        geometry_checkpoint_sha256=physical_raw["geometry_checkpoint_sha256"],
    )
    if (
        physical_raw != physical_authority
        or physical_authority["scene_id"] != scene
        or physical_authority["physical_space_id"] != physical
    ):
        raise ValueError("target AcceptedV2 physical-space binding differs")
    payload["physical_space_authority"] = physical_authority
    payload["producer"] = _producer(payload["producer"])
    valid = payload["accepted_base_valid"]
    canonical = payload["canonical_region_indices"]
    fingerprints = payload["region_fingerprints"]
    rows = payload["region_rows"]
    mask = payload["token_mask"]
    anchor = payload["anchor_index"]
    scales = payload["scale_indices"]
    descriptor = payload["accepted_v2_e0"]
    count = int(valid.numel()) if torch.is_tensor(valid) and valid.ndim == 1 else -1
    regions = int(rows.shape[0]) if torch.is_tensor(rows) and rows.ndim == 2 else -1
    geometry = payload.get("geometry_fingerprint")
    if (
        not torch.is_tensor(valid)
        or valid.dtype != torch.bool
        or not torch.is_tensor(canonical)
        or canonical.dtype != torch.long
        or canonical.shape != (regions,)
        or regions <= 0
        or regions > shard.TEACHER_REGION_CAP_PER_SCENE
        or bool((canonical < 0).any())
        or (regions > 1 and not bool((canonical[1:] > canonical[:-1]).all()))
        or not isinstance(fingerprints, list)
        or len(fingerprints) != regions
        or any(_SHA256.fullmatch(str(item)) is None for item in fingerprints)
        or len(set(fingerprints)) != regions
        or not torch.is_tensor(rows)
        or rows.dtype != torch.long
        or rows.shape[1] <= 0
        or not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or mask.shape != rows.shape
        or not torch.is_tensor(anchor)
        or anchor.dtype != torch.long
        or anchor.shape != (regions,)
        or not torch.is_tensor(scales)
        or scales.dtype != torch.long
        or scales.shape != (regions,)
        or bool((scales < 0).any())
        or not torch.is_tensor(descriptor)
        or descriptor.dtype != torch.float32
        or descriptor.shape != (regions, shard.trainer.DESCRIPTOR_DIM)
        or not bool(torch.isfinite(descriptor).all())
        or not isinstance(geometry, Mapping)
        or set(geometry) != {"num_gaussians", "xyz_sha256"}
        or int(geometry["num_gaussians"]) != count
        or _SHA256.fullmatch(str(geometry["xyz_sha256"])) is None
    ):
        raise ValueError("target AcceptedV2 tensor layout differs")
    payload["input_authority"] = shard.validate_accepted_region_input_authority(
        payload["input_authority"], geometry_fingerprint=dict(geometry)
    )
    if bool((mask.sum(dim=1) <= 0).any()) or bool(rows[~mask].ne(-1).any()):
        raise ValueError("target AcceptedV2 padding differs")
    active = rows[mask]
    if bool((active < 0).any()) or bool((active >= count).any()):
        raise ValueError("target AcceptedV2 active row is outside geometry")
    for region in range(regions):
        values = rows[region][mask[region]]
        if int(torch.unique(values).numel()) != int(values.numel()):
            raise ValueError("target AcceptedV2 repeats an active row")
    if (
        bool((anchor < 0).any())
        or bool((anchor >= rows.shape[1]).any())
        or not bool(mask[torch.arange(regions), anchor].all())
    ):
        raise ValueError("target AcceptedV2 anchor differs")
    anchor_rows = rows[torch.arange(regions), anchor]
    if not bool(valid[anchor_rows].all()):
        raise ValueError("target AcceptedV2 anchor lacks a valid base descriptor")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("target AcceptedV2 descriptor is not unit L2")
    identities = shard._canonical_region_identity(scene, rows, mask, anchor, scales)
    expected_fingerprints = [canonical_json_sha256(item) for item in identities]
    order = [
        (
            item["scale_index"],
            item["anchor_global_row"],
            tuple(item["active_global_rows"]),
        )
        for item in identities
    ]
    if (
        order != sorted(order)
        or len(set(order)) != len(order)
        or fingerprints != expected_fingerprints
    ):
        raise ValueError("target AcceptedV2 canonical region order differs")
    audit = shard.validate_selection_audit(
        payload["selection_audit"], selected_count=regions
    )
    if bool((canonical >= int(audit["canonical_candidate_region_count"])).any()):
        raise ValueError("target AcceptedV2 canonical index exceeds its domain")
    if payload["channel_sha256"] != shard.accepted_region_channel_sha256(payload):
        raise ValueError("target AcceptedV2 channel SHA-256 differs")
    return {
        **payload,
        "scene_id": scene,
        "physical_space_id": physical,
        "canonical_region_indices": canonical.detach().cpu().contiguous(),
        "region_fingerprints": list(fingerprints),
        "selection_audit": audit,
    }


__all__ = [
    "TARGET_ACCEPTED_V2_CONTRACT_SHA256",
    "TARGET_ACCEPTED_V2_SCHEMA",
    "TARGET_ACCEPTED_V2_SCHEMA_VERSION",
    "target_accepted_v2_contract",
    "target_access_audit",
    "target_physical_space_authority",
    "validate_target_accepted_v2_authority",
]
