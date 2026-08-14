"""Seal the official UQIS construction without enabling public evaluation.

The construction authority is the narrow seam between dataset construction
and method mapping.  It proves which content-bound assets produced the frozen
cohort, queries, exclusions, and evaluator domain.  It deliberately does not
mint method-visible formal query bundles or authorize an evaluator run.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    BENCHMARK_VERSION,
    COHORT_DERIVATION_LEDGER,
    FROZEN_PROTOCOL_CONFIG,
    PREREGISTERED_TEST_SCENES,
    audit_release,
    canonical_json_sha256,
    sha256_file,
)


AUTHORITY_SCHEMA_VERSION = "scannet_uqis_construction_authority_v1"
SCENE_RECORD_KEYS = {
    "scene_id",
    "mesh_xyz_path",
    "mesh_instance_ids_path",
    "query_frame_ids",
    "withheld_frame_ids",
    "field_frame_ids",
    "max_query_frames",
}
TARGET_RECORD_KEYS = {
    "scene_id",
    "instance_id",
    "nyu40_class_id",
    "raw_semantic_label",
    "mesh_vertex_count",
    "size_bucket",
    "same_class_distractor_instance_ids",
    "query_frame_id",
    "expression",
    "expression_annotation_id",
    "expression_source",
    "expression_view_independent",
    "expression_view_dependence_rule",
    "crop_rgb_path",
    "crop_rgb_sha256",
    "camera_to_world",
    "camera_intrinsics",
    "raster_size",
    "positive_pixel_uv",
    "click_depth_m",
    "point_world_xyz",
    "projection_pixels",
    "projection_fraction",
    "projection_purity",
    "field_surface_coverage",
    "field_visibility_count",
}
SCENE_RECEIPT_KEYS = {
    "schema_version",
    "benchmark_version",
    "scene_id",
    "status",
    "method_predictions_opened",
    "sources",
    "full_sensor_frame_count",
    "nonfinite_pose_frame_count",
    "sparse_projection_frame_count",
    "coverage_surface_frame_count",
    "query_frame_ids",
    "withheld_frame_count",
    "field_frame_count",
    "target_instance_ids",
    "protocol_config",
    "receipt_sha256",
}
SOURCE_KEYS = {"path", "bytes", "sha256"}
AUTHORITY_KEYS = {
    "schema_version",
    "benchmark_version",
    "status",
    "construction_formal_eligible",
    "public_formal_evaluation_enabled",
    "remaining_evaluation_authority_requirements",
    "protocol_config",
    "protocol_config_sha256",
    "scene_count",
    "target_count",
    "query_count",
    "scene_order",
    "construction_inputs",
    "nr3d",
    "cohort_ledger",
    "scene_derivation_receipt_sha256",
    "verified_scene_sources",
    "candidate_release",
    "authority_sha256",
}


def _read_exact(path: Path, keys: set[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON {path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError(f"{path.name}: top-level schema changed")
    return payload


def _verify_binding(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != SOURCE_KEYS:
        raise ValueError(f"{label}: source binding schema changed")
    path = Path(str(binding["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"{label}: missing source {path}")
    if int(binding["bytes"]) != path.stat().st_size:
        raise ValueError(f"{label}: source size changed")
    digest = sha256_file(path)
    if binding["sha256"] != digest:
        raise ValueError(f"{label}: source hash changed")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def audit_construction_authority(
    authority_path: str | Path, *, check_files: bool = True
) -> dict[str, Any]:
    """Revalidate a sealed construction authority before downstream mapping."""

    errors: list[str] = []
    path = Path(authority_path).resolve()
    try:
        authority = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "errors": [f"unreadable authority: {error}"]}
    if not isinstance(authority, dict) or set(authority) != AUTHORITY_KEYS:
        return {"valid": False, "errors": ["construction authority schema changed"]}
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if authority["authority_sha256"] != canonical_json_sha256(body):
        errors.append("construction authority digest changed")
    if (
        authority["schema_version"] != AUTHORITY_SCHEMA_VERSION
        or authority["benchmark_version"] != BENCHMARK_VERSION
        or authority["status"] != "construction_authority_sealed"
        or authority["construction_formal_eligible"] is not True
        or authority["public_formal_evaluation_enabled"] is not False
        or tuple(authority["scene_order"]) != PREREGISTERED_TEST_SCENES
        or authority["scene_count"] != len(PREREGISTERED_TEST_SCENES)
        or authority["query_count"] != 4 * authority["target_count"]
    ):
        errors.append("construction authority identity/status changed")
    if authority["protocol_config_sha256"] != canonical_json_sha256(
        authority["protocol_config"]
    ):
        errors.append("construction authority protocol digest changed")
    if not check_files:
        errors.append("authority source hashes were skipped")
    else:
        bindings: list[tuple[str, Any]] = [
            (f"construction/{name}", binding)
            for name, binding in authority["construction_inputs"].items()
        ]
        bindings.extend(
            [
                ("nr3d", authority["nr3d"]),
                (
                    "cohort_ledger",
                    {key: authority["cohort_ledger"][key] for key in SOURCE_KEYS},
                ),
                (
                    "candidate_release",
                    {key: authority["candidate_release"][key] for key in SOURCE_KEYS},
                ),
            ]
        )
        for scene_id, sources in authority["verified_scene_sources"].items():
            for name, binding in sources.items():
                if binding is not None:
                    bindings.append((f"{scene_id}/{name}", binding))
        for label, binding in bindings:
            try:
                _verify_binding(binding, label=label)
            except (ValueError, KeyError, TypeError) as error:
                errors.append(str(error))
        candidate_path = Path(authority["candidate_release"]["path"])
        candidate_audit = audit_release(candidate_path.parent, check_files=True)
        if not candidate_audit.get("valid"):
            errors.append("candidate release no longer passes fresh audit")
    return {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "authority_sha256": authority.get("authority_sha256"),
        "scene_count": authority.get("scene_count"),
        "target_count": authority.get("target_count"),
        "query_count": authority.get("query_count"),
        "valid": not errors,
        "errors": errors,
    }


def _validate_cohort_ledger(path: Path) -> dict[str, Any]:
    ledger = _read_exact(
        path,
        {
            "schema_version",
            "benchmark_version",
            "selection_information",
            "formal_method_predictions_opened",
            "final_scene_order",
            "decisions",
        },
    )
    if ledger["schema_version"] != "scannet_uqis_cohort_derivation_v1":
        raise ValueError("cohort ledger schema version changed")
    if ledger["benchmark_version"] != BENCHMARK_VERSION:
        raise ValueError("cohort ledger benchmark identity changed")
    if ledger["selection_information"] != "official_scannet_geometry_plus_nr3d_annotations_only":
        raise ValueError("cohort ledger selection information changed")
    if ledger["formal_method_predictions_opened"] is not False:
        raise ValueError("cohort selection opened formal method predictions")
    if tuple(ledger["final_scene_order"]) != PREREGISTERED_TEST_SCENES:
        raise ValueError("cohort ledger scene order changed")
    if ledger["decisions"] != list(COHORT_DERIVATION_LEDGER):
        raise ValueError("cohort derivation decisions changed")
    return ledger


def seal_construction_authority(
    construction_root: str | Path,
    candidate_release_root: str | Path,
    cohort_ledger_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify every construction input and emit one immutable authority file."""

    construction = Path(construction_root).resolve()
    candidate = Path(candidate_release_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty authority: {output}")

    scenes_payload = _read_exact(
        construction / "scene_records.json", {"benchmark_version", "scenes"}
    )
    targets_payload = _read_exact(
        construction / "target_records.json", {"benchmark_version", "targets"}
    )
    receipts_payload = _read_exact(
        construction / "construction_receipts.json",
        {"benchmark_version", "nr3d", "cohort_derivation_ledger", "scenes"},
    )
    if {
        scenes_payload["benchmark_version"],
        targets_payload["benchmark_version"],
        receipts_payload["benchmark_version"],
    } != {BENCHMARK_VERSION}:
        raise ValueError("construction benchmark identity changed")

    scenes = scenes_payload["scenes"]
    targets = targets_payload["targets"]
    receipts = receipts_payload["scenes"]
    if not all(isinstance(row, dict) and set(row) == SCENE_RECORD_KEYS for row in scenes):
        raise ValueError("scene-record schema changed")
    if not all(isinstance(row, dict) and set(row) == TARGET_RECORD_KEYS for row in targets):
        raise ValueError("target-record schema changed")
    if not all(isinstance(row, dict) and set(row) == SCENE_RECEIPT_KEYS for row in receipts):
        raise ValueError("scene-receipt schema changed")
    scene_ids = tuple(row["scene_id"] for row in scenes)
    if scene_ids != PREREGISTERED_TEST_SCENES or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("construction scene cohort/order changed")
    receipt_by_scene = {row["scene_id"]: row for row in receipts}
    if tuple(receipt_by_scene) != PREREGISTERED_TEST_SCENES:
        raise ValueError("receipt scene cohort/order changed")

    # Compare the serialized protocol identity. JSON intentionally normalizes
    # the dataclass's RGB tuple to a list in every persisted manifest.
    frozen_config = json.loads(json.dumps(asdict(FROZEN_PROTOCOL_CONFIG)))
    target_ids_by_scene: dict[str, list[int]] = {scene_id: [] for scene_id in scene_ids}
    for target in targets:
        scene_id = str(target["scene_id"])
        if scene_id not in target_ids_by_scene:
            raise ValueError(f"target references unknown scene {scene_id}")
        target_ids_by_scene[scene_id].append(int(target["instance_id"]))
        if target["expression_source"] != "nr3d" or target["expression_view_independent"] is not True:
            raise ValueError(f"{scene_id}: text authority is not view-independent Nr3D")
        crop = Path(str(target["crop_rgb_path"])).resolve()
        if not crop.is_file() or sha256_file(crop) != target["crop_rgb_sha256"]:
            raise ValueError(f"{scene_id}: query crop hash changed")

    verified_sources: dict[str, dict[str, Any]] = {}
    receipt_hashes: dict[str, str] = {}
    for scene in scenes:
        scene_id = scene["scene_id"]
        receipt = receipt_by_scene[scene_id]
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt["receipt_sha256"] != canonical_json_sha256(body):
            raise ValueError(f"{scene_id}: derivation receipt digest changed")
        if (
            receipt["schema_version"] != "scannet_uqis_official_scene_derivation_v1"
            or receipt["benchmark_version"] != BENCHMARK_VERSION
            or receipt["status"] != "construction_complete"
            or receipt["method_predictions_opened"] is not False
            or receipt["protocol_config"] != frozen_config
        ):
            raise ValueError(f"{scene_id}: derivation receipt identity changed")
        if receipt["query_frame_ids"] != scene["query_frame_ids"]:
            raise ValueError(f"{scene_id}: query-frame receipt differs")
        if receipt["withheld_frame_count"] != len(scene["withheld_frame_ids"]):
            raise ValueError(f"{scene_id}: withheld-frame receipt differs")
        if receipt["field_frame_count"] != len(scene["field_frame_ids"]):
            raise ValueError(f"{scene_id}: field-frame receipt differs")
        if sorted(receipt["target_instance_ids"]) != sorted(target_ids_by_scene[scene_id]):
            raise ValueError(f"{scene_id}: target receipt differs")
        if set(receipt["sources"]) != {
            "sens", "mesh", "aggregation", "segmentation", "query_frame_derivation_receipt"
        }:
            raise ValueError(f"{scene_id}: derivation source schema changed")
        scene_sources = {}
        for name, binding in receipt["sources"].items():
            if binding is None:
                if name != "query_frame_derivation_receipt":
                    raise ValueError(f"{scene_id}: required source {name} missing")
                scene_sources[name] = None
            else:
                scene_sources[name] = _verify_binding(binding, label=f"{scene_id}/{name}")
        verified_sources[scene_id] = scene_sources
        receipt_hashes[scene_id] = receipt["receipt_sha256"]

    nr3d_binding = _verify_binding(receipts_payload["nr3d"], label="nr3d")
    if receipts_payload["cohort_derivation_ledger"] != list(COHORT_DERIVATION_LEDGER):
        raise ValueError("construction cohort ledger differs from frozen ledger")
    ledger = _validate_cohort_ledger(Path(cohort_ledger_path).resolve())

    candidate_audit = audit_release(candidate, check_files=True)
    if not candidate_audit.get("valid"):
        raise ValueError("candidate release audit failed: " + "; ".join(candidate_audit.get("errors", [])))
    candidate_release = _read_exact(
        candidate / "release.json",
        {
            "benchmark_version", "split_role", "release_tier",
            "formal_benchmark_eligible", "protocol_config", "protocol_config_sha256",
            "query_id_salt_sha256", "status", "formal_release_implemented",
            "scene_count", "target_count", "query_count", "manifest_sha256", "audit",
        },
    )
    if (
        candidate_release["benchmark_version"] != BENCHMARK_VERSION
        or candidate_release["protocol_config"] != frozen_config
        or candidate_release["scene_count"] != len(scenes)
        or candidate_release["target_count"] != len(targets)
        or candidate_release["query_count"] != 4 * len(targets)
        or candidate_release["formal_benchmark_eligible"] is not False
    ):
        raise ValueError("candidate release differs from official construction")
    public_targets = json.loads((candidate / "target_manifest.public.json").read_text())
    if public_targets.get("readiness_errors") != []:
        raise ValueError("construction has unresolved formal-readiness errors")

    input_bindings = {
        name: _file_binding(construction / name)
        for name in (
            "scene_records.json", "target_records.json", "construction_receipts.json"
        )
    }
    authority_body = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "construction_authority_sealed",
        "construction_formal_eligible": True,
        "public_formal_evaluation_enabled": False,
        "remaining_evaluation_authority_requirements": [
            "evaluator_private_secret_query_id_refreeze",
            "physically_separated_method_and_evaluator_bundles",
            "per_query_sandbox_runtime_receipts",
            "sealed_complete_method_field_inventory",
            "evaluator_owned_one_shot_release_ledger",
        ],
        "protocol_config": frozen_config,
        "protocol_config_sha256": canonical_json_sha256(frozen_config),
        "scene_count": len(scenes),
        "target_count": len(targets),
        "query_count": 4 * len(targets),
        "scene_order": list(scene_ids),
        "construction_inputs": input_bindings,
        "nr3d": nr3d_binding,
        "cohort_ledger": {
            **_file_binding(Path(cohort_ledger_path).resolve()),
            "payload_sha256": canonical_json_sha256(ledger),
        },
        "scene_derivation_receipt_sha256": receipt_hashes,
        "verified_scene_sources": verified_sources,
        "candidate_release": {
            **_file_binding(candidate / "release.json"),
            "manifest_sha256": candidate_release["manifest_sha256"],
            "audit": candidate_audit,
            "purpose": "evaluator_private_construction_validation_only",
        },
    }
    authority = {
        **authority_body,
        "authority_sha256": canonical_json_sha256(authority_body),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "construction_authority.json"
    path.write_text(
        json.dumps(authority, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return authority
