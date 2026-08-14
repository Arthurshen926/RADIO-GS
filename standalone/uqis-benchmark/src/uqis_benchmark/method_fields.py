"""Representation accounting for UQIS method systems.

The benchmark may score either a genuinely universal field or a complete
method system made of modality-scoped fields.  This module keeps those claims
distinct and makes the latter pay its full persistent-storage cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    BENCHMARK_VERSION,
    BENCHMARK_VERSION_V2_CANDIDATE,
    canonical_json_sha256,
    sha256_file,
)


MODALITIES = ("text", "image", "point_2d", "point_3d")
REPRESENTATION_SCOPES = (
    "single_universal_field",
    "modality_specific_multi_field",
)
FIELD_INVENTORY_SCHEMA_V1 = "scannet_uqis_method_field_inventory_v1"
FIELD_INVENTORY_SCHEMA_V2 = "scannet_uqis_method_field_inventory_v2"
# Preserve the v0.1 construction/result authority.  New dependency-set
# inventories must opt into FIELD_INVENTORY_SCHEMA_V2 explicitly.
FIELD_INVENTORY_SCHEMA = FIELD_INVENTORY_SCHEMA_V1


def ludvig_modality_field_plan() -> tuple[dict[str, Any], ...]:
    """Return the frozen per-scene field-family plan for the LUDVIG comparator."""

    return (
        {
            "field_id": "ludvig_clip_text_field",
            "field_family": "ludvig_clip_language_field",
            "modalities": ["text"],
        },
        {
            "field_id": "ludvig_dino_prompt_image_field",
            "field_family": "ludvig_dinov2_visual_field",
            "modalities": ["image", "point_2d", "point_3d"],
        },
    )


def _validate_digest(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"relative_path", "bytes", "sha256"}
    if set(raw) != expected:
        raise ValueError("field artifact schema changed")
    relative = Path(str(raw["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
        raise ValueError("field artifact path must be a contained relative path")
    size = int(raw["bytes"])
    if size < 0:
        raise ValueError("field artifact byte size must be non-negative")
    return {
        "relative_path": relative.as_posix(),
        "bytes": size,
        "sha256": _validate_digest(raw["sha256"], "field artifact hash"),
    }


def _validate_method_field_inventory_v1(
    payload: Mapping[str, Any],
    *,
    expected_scene_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and recompute all field multiplicity/storage accounting."""

    top_keys = {
        "schema_version",
        "benchmark_version",
        "status",
        "method_system_id",
        "method_identity_sha256",
        "representation_scope",
        "scene_count",
        "scenes",
        "totals",
        "inventory_sha256",
    }
    if set(payload) != top_keys:
        raise ValueError("method field inventory top-level schema changed")
    if payload.get("schema_version") != FIELD_INVENTORY_SCHEMA:
        raise ValueError("method field inventory schema version changed")
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("method field inventory benchmark version changed")
    if payload.get("status") != "complete":
        raise ValueError("method field inventory is incomplete")
    method_system_id = str(payload.get("method_system_id", "")).strip()
    if not method_system_id:
        raise ValueError("method_system_id must be non-empty")
    method_identity_sha256 = _validate_digest(
        payload.get("method_identity_sha256"), "method identity"
    )
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("method field inventory must contain scenes")

    scenes: list[dict[str, Any]] = []
    inferred_scopes: set[str] = set()
    all_artifacts: dict[str, int] = {}
    total_fields = 0
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping) or set(raw_scene) != {
            "scene_id",
            "fields",
            "field_count",
            "persistent_bytes",
        }:
            raise ValueError("method scene field schema changed")
        scene_id = str(raw_scene["scene_id"])
        raw_fields = raw_scene["fields"]
        if not scene_id or not isinstance(raw_fields, list) or not raw_fields:
            raise ValueError("method scene must contain one or more fields")
        assignments: dict[str, str] = {}
        fields: list[dict[str, Any]] = []
        scene_artifacts: dict[str, int] = {}
        field_ids: set[str] = set()
        for raw_field in raw_fields:
            if not isinstance(raw_field, Mapping) or set(raw_field) != {
                "field_id",
                "field_family",
                "modalities",
                "mapping_receipt_sha256",
                "artifacts",
            }:
                raise ValueError("method field schema changed")
            field_id = str(raw_field["field_id"]).strip()
            family = str(raw_field["field_family"]).strip()
            if not field_id or field_id in field_ids or not family:
                raise ValueError("field identities must be non-empty and unique per scene")
            field_ids.add(field_id)
            modalities = tuple(map(str, raw_field["modalities"]))
            if not modalities or len(set(modalities)) != len(modalities):
                raise ValueError("a field must serve distinct modalities")
            for modality in modalities:
                if modality not in MODALITIES or modality in assignments:
                    raise ValueError("each UQIS modality must map to exactly one field")
                assignments[modality] = field_id
            artifacts = [_validate_artifact(row) for row in raw_field["artifacts"]]
            if not artifacts:
                raise ValueError("each field must bind at least one persistent artifact")
            for artifact in artifacts:
                digest, size = artifact["sha256"], artifact["bytes"]
                if digest in scene_artifacts and scene_artifacts[digest] != size:
                    raise ValueError("one artifact digest declares inconsistent sizes")
                scene_artifacts[digest] = size
                if digest in all_artifacts and all_artifacts[digest] != size:
                    raise ValueError("one artifact digest declares inconsistent sizes")
                all_artifacts[digest] = size
            fields.append(
                {
                    "field_id": field_id,
                    "field_family": family,
                    "modalities": list(modalities),
                    "mapping_receipt_sha256": _validate_digest(
                        raw_field["mapping_receipt_sha256"], "mapping receipt"
                    ),
                    "artifacts": artifacts,
                }
            )
        if set(assignments) != set(MODALITIES):
            raise ValueError(f"{scene_id}: modality-to-field coverage is incomplete")
        scope = (
            "single_universal_field"
            if len(fields) == 1 and set(fields[0]["modalities"]) == set(MODALITIES)
            else "modality_specific_multi_field"
        )
        inferred_scopes.add(scope)
        persistent_bytes = sum(scene_artifacts.values())
        if int(raw_scene["field_count"]) != len(fields):
            raise ValueError(f"{scene_id}: declared field count changed")
        if int(raw_scene["persistent_bytes"]) != persistent_bytes:
            raise ValueError(f"{scene_id}: declared persistent storage changed")
        total_fields += len(fields)
        scenes.append(
            {
                "scene_id": scene_id,
                "fields": fields,
                "field_count": len(fields),
                "persistent_bytes": persistent_bytes,
            }
        )
    scene_ids = [scene["scene_id"] for scene in scenes]
    if len(set(scene_ids)) != len(scene_ids) or scene_ids != sorted(scene_ids):
        raise ValueError("method field inventory scenes must be unique and sorted")
    if expected_scene_ids is not None and scene_ids != sorted(map(str, expected_scene_ids)):
        raise ValueError("method field inventory scene coverage changed")
    if len(inferred_scopes) != 1:
        raise ValueError("representation scope must be consistent across scenes")
    inferred_scope = next(iter(inferred_scopes))
    if payload.get("representation_scope") != inferred_scope:
        raise ValueError("declared representation scope disagrees with field assignment")
    totals = {
        "field_count": total_fields,
        # Fields are deployed per scene; identical bytes in two scene states
        # are not credited as one universal representation.
        "persistent_bytes": sum(scene["persistent_bytes"] for scene in scenes),
    }
    if payload.get("scene_count") != len(scenes) or payload.get("totals") != totals:
        raise ValueError("method field inventory totals changed")
    normalized_without_hash = {
        "schema_version": FIELD_INVENTORY_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "method_system_id": method_system_id,
        "method_identity_sha256": method_identity_sha256,
        "representation_scope": inferred_scope,
        "scene_count": len(scenes),
        "scenes": scenes,
        "totals": totals,
    }
    if payload.get("inventory_sha256") != canonical_json_sha256(normalized_without_hash):
        raise ValueError("method field inventory digest changed")
    return {**normalized_without_hash, "inventory_sha256": payload["inventory_sha256"]}


def _validate_method_field_inventory_v2(
    payload: Mapping[str, Any],
    *,
    expected_scene_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate v0.2 modality-to-field dependency sets and union storage."""

    top_keys = {
        "schema_version", "benchmark_version", "status", "method_system_id",
        "method_identity_sha256", "representation_scope", "scene_count",
        "scenes", "totals", "inventory_sha256",
    }
    if set(payload) != top_keys:
        raise ValueError("method field inventory top-level schema changed")
    if payload.get("schema_version") != FIELD_INVENTORY_SCHEMA_V2:
        raise ValueError("method field inventory schema version changed")
    benchmark_version = str(payload.get("benchmark_version", ""))
    if benchmark_version not in {BENCHMARK_VERSION, BENCHMARK_VERSION_V2_CANDIDATE}:
        raise ValueError("method field inventory benchmark version changed")
    if payload.get("status") != "complete":
        raise ValueError("method field inventory is incomplete")
    method_system_id = str(payload.get("method_system_id", "")).strip()
    if not method_system_id:
        raise ValueError("method_system_id must be non-empty")
    method_identity_sha256 = _validate_digest(
        payload.get("method_identity_sha256"), "method identity"
    )
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("method field inventory must contain scenes")

    scenes: list[dict[str, Any]] = []
    inferred_scopes: set[str] = set()
    total_fields = 0
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping) or set(raw_scene) != {
            "scene_id", "fields", "modality_field_dependencies",
            "field_count", "persistent_bytes",
        }:
            raise ValueError("method scene field schema changed")
        scene_id = str(raw_scene["scene_id"])
        raw_fields = raw_scene["fields"]
        if not scene_id or not isinstance(raw_fields, list) or not raw_fields:
            raise ValueError("method scene must contain one or more fields")
        fields: list[dict[str, Any]] = []
        field_ids: set[str] = set()
        scene_artifacts: dict[str, int] = {}
        for raw_field in raw_fields:
            if not isinstance(raw_field, Mapping) or set(raw_field) != {
                "field_id", "field_family", "mapping_receipt_sha256", "artifacts",
            }:
                raise ValueError("method field schema changed")
            field_id = str(raw_field["field_id"]).strip()
            family = str(raw_field["field_family"]).strip()
            if not field_id or field_id in field_ids or not family:
                raise ValueError("field identities must be non-empty and unique per scene")
            field_ids.add(field_id)
            artifacts = [_validate_artifact(row) for row in raw_field["artifacts"]]
            if not artifacts:
                raise ValueError("each field must bind at least one persistent artifact")
            for artifact in artifacts:
                digest, size = artifact["sha256"], artifact["bytes"]
                if digest in scene_artifacts and scene_artifacts[digest] != size:
                    raise ValueError("one artifact digest declares inconsistent sizes")
                scene_artifacts[digest] = size
            fields.append(
                {
                    "field_id": field_id,
                    "field_family": family,
                    "mapping_receipt_sha256": _validate_digest(
                        raw_field["mapping_receipt_sha256"], "mapping receipt"
                    ),
                    "artifacts": artifacts,
                }
            )
        raw_dependencies = raw_scene["modality_field_dependencies"]
        if not isinstance(raw_dependencies, Mapping) or set(raw_dependencies) != set(MODALITIES):
            raise ValueError(f"{scene_id}: modality field dependency coverage is incomplete")
        dependencies: dict[str, list[str]] = {}
        for modality in MODALITIES:
            raw_values = raw_dependencies[modality]
            if not isinstance(raw_values, list) or not raw_values:
                raise ValueError(f"{scene_id}/{modality}: dependency set must be non-empty")
            values = list(map(str, raw_values))
            if len(set(values)) != len(values):
                raise ValueError(f"{scene_id}/{modality}: dependency set contains duplicates")
            undeclared = sorted(set(values) - field_ids)
            if undeclared:
                raise ValueError(
                    f"{scene_id}/{modality}: dependency set references undeclared field {undeclared}"
                )
            dependencies[modality] = values
        only_field = next(iter(field_ids)) if len(field_ids) == 1 else None
        scope = (
            "single_universal_field"
            if only_field is not None
            and all(values == [only_field] for values in dependencies.values())
            else "modality_specific_multi_field"
        )
        inferred_scopes.add(scope)
        persistent_bytes = sum(scene_artifacts.values())
        if int(raw_scene["field_count"]) != len(fields):
            raise ValueError(f"{scene_id}: declared field count changed")
        if int(raw_scene["persistent_bytes"]) != persistent_bytes:
            raise ValueError(f"{scene_id}: declared persistent storage changed")
        total_fields += len(fields)
        scenes.append(
            {
                "scene_id": scene_id,
                "fields": fields,
                "modality_field_dependencies": dependencies,
                "field_count": len(fields),
                "persistent_bytes": persistent_bytes,
            }
        )
    scene_ids = [scene["scene_id"] for scene in scenes]
    if len(set(scene_ids)) != len(scene_ids) or scene_ids != sorted(scene_ids):
        raise ValueError("method field inventory scenes must be unique and sorted")
    if expected_scene_ids is not None and scene_ids != sorted(map(str, expected_scene_ids)):
        raise ValueError("method field inventory scene coverage changed")
    if len(inferred_scopes) != 1:
        raise ValueError("representation scope must be consistent across scenes")
    inferred_scope = next(iter(inferred_scopes))
    if payload.get("representation_scope") != inferred_scope:
        raise ValueError("declared representation scope disagrees with field dependencies")
    totals = {
        "field_count": total_fields,
        "persistent_bytes": sum(scene["persistent_bytes"] for scene in scenes),
    }
    if payload.get("scene_count") != len(scenes) or payload.get("totals") != totals:
        raise ValueError("method field inventory totals changed")
    normalized_without_hash = {
        "schema_version": FIELD_INVENTORY_SCHEMA_V2,
        "benchmark_version": benchmark_version,
        "status": "complete",
        "method_system_id": method_system_id,
        "method_identity_sha256": method_identity_sha256,
        "representation_scope": inferred_scope,
        "scene_count": len(scenes),
        "scenes": scenes,
        "totals": totals,
    }
    if payload.get("inventory_sha256") != canonical_json_sha256(normalized_without_hash):
        raise ValueError("method field inventory digest changed")
    return {**normalized_without_hash, "inventory_sha256": payload["inventory_sha256"]}


def validate_method_field_inventory(
    payload: Mapping[str, Any],
    *,
    expected_scene_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate either immutable v0.1 assignments or v0.2 dependency sets."""

    if payload.get("schema_version") == FIELD_INVENTORY_SCHEMA_V1:
        return _validate_method_field_inventory_v1(
            payload, expected_scene_ids=expected_scene_ids
        )
    if payload.get("schema_version") == FIELD_INVENTORY_SCHEMA_V2:
        return _validate_method_field_inventory_v2(
            payload, expected_scene_ids=expected_scene_ids
        )
    raise ValueError("method field inventory schema version changed")


def bind_field_artifact(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Create a content/size binding for one persistent field artifact."""

    source = Path(path).resolve()
    boundary = Path(root).resolve()
    if not source.is_file() or not source.is_relative_to(boundary):
        raise ValueError("field artifact must be a file contained by its field root")
    return {
        "relative_path": source.relative_to(boundary).as_posix(),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }
