#!/usr/bin/env python3
"""Fail-closed stage validation for the ScanNet paper-8 reconstruction queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.config import load_config
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_OBSERVATION_CONTRACT_NAME,
)
from radio_gs.training.tensor_cache_io import validate_mpr_cache_payload
from radio_gs.utils.immutable_artifacts import load_torch_mapping


STAMP_SCHEMA = "scannet-canonical-mpr-v3-paper8-stage-validation-v1"
PLAN_POLICY = "deterministic_even_interior_frame_manifest_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _resolved(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_feature(args: argparse.Namespace) -> str:
    root = Path(args.feature_dir).resolve()
    manifest_path = root / "frame_manifest.json"
    manifest = _json(manifest_path)
    _require(manifest.get("scene") == args.scene, "feature scene differs")
    _require(int(manifest.get("num_frames", -1)) == args.expected_frames, "feature frame count differs")
    frames = manifest.get("frames")
    _require(isinstance(frames, list) and len(frames) == args.expected_frames, "feature manifest frames differ")
    excluded = {str(value) for value in manifest.get("excluded_image_stems", [])}
    expected_excluded = {
        value.strip()
        for value in str(args.expected_excluded_stems).split(",")
        if value.strip()
    }
    _require(excluded == expected_excluded, "feature exclusion set differs")
    _require(_resolved(manifest.get("image_dir", "")) == _resolved(Path(args.scene_root) / "color"), "feature image source differs")
    _require(float(manifest.get("resolution_scale", -1.0)) == 1.0, "feature extraction resolution scale differs")
    _require(not manifest.get("features", {}).get("adaptors", []), "project_raw bundle must not contain an alternate adaptor route")
    radio = manifest.get("radio", {})
    _require(isinstance(radio, dict), "feature RADIO provenance is absent")
    _require(radio.get("version") == "c-radio_v4-h", "feature RADIO version differs")
    _require(
        radio.get("checkpoint_sha256")
        == "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9",
        "feature RADIO checkpoint differs",
    )
    bundle = manifest.get("output_bundle")
    _require(isinstance(bundle, dict), "feature output bundle is absent")
    bundle_sha = str(manifest.get("output_bundle_sha256", ""))
    _require(bundle_sha == _canonical_json_sha256(bundle), "feature output bundle digest differs")
    _require(
        bundle_sha == args.expected_output_bundle_sha256,
        "feature output bundle differs from the frozen scene authority",
    )
    _require(bundle.get("contract") == "radio-feature-output-bundle-v1", "feature output bundle contract differs")
    bundle_frames = bundle.get("frames")
    _require(isinstance(bundle_frames, list) and len(bundle_frames) == args.expected_frames, "feature bundle frame count differs")
    for record in bundle_frames:
        _require(isinstance(record, dict), "feature bundle frame record is malformed")
        marker = root / str(record.get("marker_relative_path", ""))
        _require(marker.is_file(), f"feature commit marker is missing: {marker}")
        _require(_sha256(marker) == record.get("marker_sha256"), f"feature commit marker digest differs: {marker}")
        signature = record.get("feature_signature", {})
        backbone = signature.get("backbone", {}) if isinstance(signature, dict) else {}
        _require(backbone.get("grid") == [60, 81] and backbone.get("dim") == 1280, "feature grid/signature differs")
        tensors = record.get("tensors")
        _require(isinstance(tensors, list) and len(tensors) == 2, "raw feature frame tensor set differs")
        for tensor in tensors:
            tensor_path = root / str(tensor.get("relative_path", ""))
            _require(tensor_path.is_file() and tensor_path.stat().st_size > 0, f"feature tensor is absent: {tensor_path}")

    config = load_config(args.config)
    _require(_resolved(config.scene_root) == _resolved(args.scene_root), "config scene_root differs")
    _require(_resolved(config.feature_dir) == str(root), "config feature_dir differs")
    _require(_resolved(config.rgb_dir) == _resolved(Path(args.scene_root) / "color"), "config rgb_dir differs")
    _require(int(config.feature_height) == 60 and int(config.feature_width) == 81, "config feature grid differs")
    return bundle_sha


def validate_plan(path: str | Path, *, feature_dir: str | Path | None = None) -> dict[str, Any]:
    payload = _json(path)
    _require(payload.get("schema_version") == 1, "validation plan schema differs")
    _require(payload.get("policy") == PLAN_POLICY, "validation plan policy differs")
    _require(payload.get("requested_validation_views") == 4, "validation view count differs")
    selected = payload.get("validation_frame_ids")
    _require(isinstance(selected, list) and len(selected) == 4, "validation plan must contain four frames")
    _require(len(set(int(value) for value in selected)) == 4, "validation frames are not unique")
    manifest = Path(payload.get("frame_manifest", ""))
    _require(manifest.is_file() and _sha256(manifest) == payload.get("frame_manifest_sha256"), "validation plan feature manifest binding differs")
    if feature_dir is not None:
        _require(_resolved(payload.get("feature_dir", "")) == _resolved(feature_dir), "validation plan feature directory differs")
    for key in ("benchmark_labels_opened", "benchmark_masks_opened", "query_opened"):
        _require(payload.get(key) is False, f"validation plan safety flag differs: {key}")
    return payload


def _false_safety(payload: Mapping[str, Any], *, label: str) -> None:
    for key in ("benchmark_masks_opened", "text_queries_opened"):
        _require(payload.get(key) is False, f"{label} safety flag differs: {key}")


def _metadata_common(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    _require(_resolved(metadata.get("config", "")) == _resolved(args.config), "MPR config provenance differs")
    _require(_resolved(metadata.get("checkpoint", "")) == _resolved(args.geometry_checkpoint), "MPR geometry provenance differs")
    _require(metadata.get("feature_output_bundle_sha256") == args.feature_output_bundle_sha256, "MPR feature bundle provenance differs")
    contract = metadata.get("observation_lifting_contract")
    _require(isinstance(contract, dict) and contract.get("name") == CANONICAL_OBSERVATION_CONTRACT_NAME, "MPR observation contract differs")
    _require(metadata.get("aggregation_mode") == "raster_gaussian_top1", "MPR aggregation differs")
    _require(metadata.get("registration_weight_mode") == "alpha_depth", "MPR registration weights differ")
    _require(metadata.get("raster_view_fusion") == "contribution_mean", "MPR view fusion differs")
    _require(metadata.get("normalize_each_view") is True, "MPR normalization differs")
    plan = validate_plan(args.validation_plan)
    expected_excluded = sorted(int(value) for value in plan["validation_frame_ids"])
    _require(sorted(int(value) for value in metadata.get("excluded_frame_ids", [])) == expected_excluded, "MPR held-out frames differ")
    _false_safety(metadata, label="MPR")


def _load_mpr(args: argparse.Namespace, feature_space: str) -> dict[str, Any]:
    payload, _digest, _source = load_torch_mapping(args.path, map_location="cpu", label=f"{feature_space} MPR")
    validate_mpr_cache_payload(payload, expected_feature_space=feature_space, require_reliability=True, require_formal_safety=True)
    metadata = payload.get("metadata", {})
    _require(isinstance(metadata, dict), "MPR metadata is malformed")
    _metadata_common(args, metadata)
    responsibility_sha = _sha256(args.responsibility_cache)
    _require(metadata.get("registration_responsibility_cache_sha256") == responsibility_sha, "MPR responsibility digest differs")
    _require(_resolved(metadata.get("registration_responsibility_cache", "")) == _resolved(args.responsibility_cache), "MPR responsibility path differs")
    if feature_space in {"dino_v3", "sam3"}:
        _require(metadata.get("capability_map_source") == "project_raw", "capability MPR source is not project_raw")
        _require(metadata.get("capability_projection_before_mpr") is True, "capability projection order differs")
        _require(metadata.get("official_adaptor_checkpoint_sha256") == args.radio_checkpoint_sha256, "capability adaptor checkpoint differs")
    return payload


def _load_responsibility(args: argparse.Namespace) -> dict[str, Any]:
    payload, _digest, _source = load_torch_mapping(args.path, map_location="cpu", label="MPR responsibility")
    _require(payload.get("schema_version") == 1, "responsibility schema differs")
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), "responsibility metadata is malformed")
    _require(metadata.get("assignment_mode") == "raster_gaussian_top1", "responsibility assignment mode differs")
    _require(metadata.get("registration_weight_mode") == "alpha_depth", "responsibility weighting differs")
    _require(_resolved(metadata.get("checkpoint", "")) == _resolved(args.geometry_checkpoint), "responsibility geometry differs")
    plan = validate_plan(args.validation_plan)
    _require(sorted(metadata.get("excluded_frame_ids", [])) == sorted(plan["validation_frame_ids"]), "responsibility held-out frames differ")
    assignments = payload.get("assignments")
    _require(isinstance(assignments, list) and assignments, "responsibility assignments are absent")
    for item in assignments:
        _require(isinstance(item, dict), "responsibility assignment is malformed")
        _require(all(torch.is_tensor(item.get(key)) for key in ("gaussian_ids", "pixel_ids", "weights")), "responsibility tensors are absent")
    return payload


def _load_field_v1(args: argparse.Namespace) -> dict[str, Any]:
    payload, _digest, _source = load_torch_mapping(args.path, map_location="cpu", label="canonical field v1")
    _require(payload.get("schema_version") == 1, "field-v1 schema differs")
    architecture = payload.get("architecture", {})
    _require(architecture.get("coefficient_dim") == 256, "field-v1 coefficient dimension differs")
    _require(architecture.get("local_dim") == 128, "field-v1 local dimension differs")
    _require(architecture.get("use_fusion") is True, "field-v1 primitive fusion is absent")
    _require(payload.get("capability_target_mode") == "official_adaptor_then_geometry_matched_mpr", "field-v1 capability supervision differs")
    _require(payload.get("mpr_cache_sha256") in (None, "", _sha256(args.raw_mpr)), "field-v1 raw MPR digest differs")
    if payload.get("feature_output_bundle_sha256") not in (None, ""):
        _require(payload.get("feature_output_bundle_sha256") == args.feature_output_bundle_sha256, "field-v1 feature bundle differs")
    targets = payload.get("capability_mpr_targets", {})
    _require(targets.get("dino_v3", {}).get("sha256") == _sha256(args.dino_mpr), "field-v1 DINO MPR differs")
    _require(targets.get("sam3", {}).get("sha256") == _sha256(args.sam3_mpr), "field-v1 SAM3 MPR differs")
    training = payload.get("training_config")
    if isinstance(training, dict):
        expected = {
            "observation_contract": "canonical-mpr-v1",
            "coefficient_dim": 256,
            "local_dim": 128,
            "primitive_fusion": True,
            "official_capability_loss": True,
            "epochs": 20,
            "min_epochs": 5,
            "target_cosine": 0.985,
            "seed": 0,
        }
        _require(all(training.get(key) == value for key, value in expected.items()), "field-v1 training configuration differs")
    _false_safety(payload, label="field-v1")
    return payload


def _load_field_v2(args: argparse.Namespace) -> dict[str, Any]:
    payload, _digest, _source = load_torch_mapping(args.path, map_location="cpu", label="canonical field v2")
    _require(payload.get("schema_version") == 1, "field-v2 schema differs")
    render = payload.get("render_optimization")
    _require(isinstance(render, dict), "field-v2 render optimization is absent")
    expected = {
        "render_view_policy": "all_nonbenchmark",
        "selection_policy": "capability_pareto",
        "max_mpr_drop": 0.005,
        "train_basis": False,
        "train_fusion": False,
    }
    _require(all(render.get(key) == value for key, value in expected.items()), "field-v2 frozen selection configuration differs")
    _require(render.get("max_capability_drop") == 0.002, "field-v2 capability Pareto drop differs")
    _require(_resolved(render.get("config", "")) == _resolved(args.config), "field-v2 config differs")
    _require(_resolved(render.get("geometry_checkpoint", "")) == _resolved(args.geometry_checkpoint), "field-v2 geometry differs")
    plan = validate_plan(args.validation_plan)
    _require(sorted(render.get("validation_frames", [])) == sorted(plan["validation_frame_ids"]), "field-v2 validation frames differ")
    capability = render.get("official_render_capability", {})
    _require(capability.get("enabled") is True, "field-v2 capability render loss is absent")
    _require(capability.get("teacher_map_source") == "project_raw", "field-v2 capability source differs")
    weights = capability.get("adaptor_weights", {})
    _require(weights == {"dino_v3": 0.2, "sam3": 0.2}, "field-v2 capability weights differ")
    _require(capability.get("local_affinity_weight") == 0.25, "field-v2 local affinity weight differs")
    _require(capability.get("local_radius") == 1, "field-v2 local radius differs")
    _require(capability.get("local_balance_quantile") == 0.0, "field-v2 local balance differs")
    semantic = render.get("semantic_capability", {})
    _require(not semantic.get("enabled", False), "legacy semantic teacher must be disabled")
    _false_safety(payload, label="field-v2")
    return payload


def _load_capability(args: argparse.Namespace) -> dict[str, Any]:
    payload, _digest, _source = load_torch_mapping(args.path, map_location="cpu", label="canonical capability cache")
    required = {"schema_version", "xyz", "valid", "appearance_dino_v3", "boundary_sam3", "metadata"}
    _require(required.issubset(payload), "capability cache keys differ")
    metadata = payload["metadata"]
    _require(metadata.get("source") == "canonical_radio_field_official_frozen_capability_views", "capability cache source differs")
    _require(metadata.get("field_checkpoint_sha256") == _sha256(args.field_v2), "capability field digest differs")
    _require(metadata.get("radio_checkpoint_sha256") == args.radio_checkpoint_sha256, "capability RADIO checkpoint differs")
    _require(metadata.get("render_capability_teacher_source") == "project_raw", "capability render source differs")
    _false_safety(metadata, label="capability")
    return payload


def _load_graph(args: argparse.Namespace) -> dict[str, Any]:
    payload, _digest, _source = load_torch_mapping(args.path, map_location="cpu", label="canonical support graph")
    required = {"schema_version", "global_rows", "num_global_rows", "edge_index", "edge_weight", "metadata"}
    _require(required.issubset(payload), "support graph keys differ")
    metadata = payload["metadata"]
    _require(_resolved(metadata.get("capability_cache", "")) == _resolved(args.capability_cache), "support graph capability source differs")
    affinity = metadata.get("capability_affinity", {})
    _require(affinity.get("mode") == "signed_hash" and affinity.get("output_dim") == 256, "support graph affinity representation differs")
    config = metadata.get("graph_config", {})
    _require(config.get("neighbors") == 16, "support graph neighbor count differs")
    _require(config.get("topology_mode") == "symmetric_union", "support graph topology differs")
    _false_safety(metadata, label="support graph")
    return payload


def _dependency_state(args: argparse.Namespace) -> dict[str, str]:
    state = {
        "config_sha256": _sha256(args.config),
        "geometry_checkpoint_sha256": args.geometry_checkpoint_sha256,
        "feature_output_bundle_sha256": args.feature_output_bundle_sha256,
        "validation_plan_sha256": _sha256(args.validation_plan),
        "radio_checkpoint_sha256": args.radio_checkpoint_sha256,
    }
    paths_by_kind = {
        "raw_mpr": [args.responsibility_cache],
        "dino_mpr": [args.responsibility_cache],
        "sam3_mpr": [args.responsibility_cache],
        "field_v1": [args.raw_mpr, args.dino_mpr, args.sam3_mpr],
        "field_v2": [args.raw_mpr, args.field_v1],
        "capability": [args.raw_mpr, args.field_v2],
        "support_graph": [args.capability_cache],
    }
    for path in paths_by_kind.get(args.kind, []):
        resolved = _resolved(path)
        state[resolved] = _sha256(resolved)
    sidecar = Path(str(args.path) + ".json")
    if sidecar.is_file():
        state["sidecar_sha256"] = _sha256(sidecar)
    return state


def validate_artifact(args: argparse.Namespace) -> str:
    path = Path(args.path)
    _require(path.is_file() and path.stat().st_size > 0, f"artifact is absent: {path}")
    if args.kind == "validation_frames":
        validate_plan(path)
        return _sha256(path)
    artifact_sha = _sha256(path)
    dependencies = _dependency_state(args)
    stamp_path = Path(str(path) + ".paper8_validation.json")
    expected_stamp = {
        "schema": STAMP_SCHEMA,
        "kind": args.kind,
        "artifact": str(path.resolve()),
        "artifact_sha256": artifact_sha,
        "dependencies": dependencies,
    }
    if stamp_path.is_file() and _json(stamp_path) == expected_stamp:
        return artifact_sha
    loaders = {
        "responsibility": _load_responsibility,
        "raw_mpr": lambda value: _load_mpr(value, "radio"),
        "dino_mpr": lambda value: _load_mpr(value, "dino_v3"),
        "sam3_mpr": lambda value: _load_mpr(value, "sam3"),
        "field_v1": _load_field_v1,
        "field_v2": _load_field_v2,
        "capability": _load_capability,
        "support_graph": _load_graph,
    }
    loaders[args.kind](args)
    if args.write_stamp:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{stamp_path.name}.", suffix=".tmp", dir=stamp_path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(json.dumps(expected_stamp, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, stamp_path)
        finally:
            temporary.unlink(missing_ok=True)
    return artifact_sha


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    feature = subparsers.add_parser("feature")
    feature.add_argument("--scene", required=True)
    feature.add_argument("--feature-dir", required=True)
    feature.add_argument("--config", required=True)
    feature.add_argument("--scene-root", required=True)
    feature.add_argument("--expected-frames", type=int, required=True)
    feature.add_argument("--expected-excluded-stems", default="")
    feature.add_argument("--expected-output-bundle-sha256", required=True)
    feature.add_argument("--print-sha256", action="store_true")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--path", required=True)
    plan.add_argument("--print-csv", action="store_true")
    artifact = subparsers.add_parser("artifact")
    artifact.add_argument("--kind", choices=["validation_frames", "responsibility", "raw_mpr", "dino_mpr", "sam3_mpr", "field_v1", "field_v2", "capability", "support_graph"], required=True)
    artifact.add_argument("--path", required=True)
    artifact.add_argument("--scene", required=True)
    artifact.add_argument("--config", required=True)
    artifact.add_argument("--geometry-checkpoint", required=True)
    artifact.add_argument("--geometry-checkpoint-sha256", required=True)
    artifact.add_argument("--feature-output-bundle-sha256", required=True)
    artifact.add_argument("--validation-plan", required=True)
    artifact.add_argument("--responsibility-cache", required=True)
    artifact.add_argument("--raw-mpr", required=True)
    artifact.add_argument("--dino-mpr", required=True)
    artifact.add_argument("--sam3-mpr", required=True)
    artifact.add_argument("--field-v1", required=True)
    artifact.add_argument("--field-v2", required=True)
    artifact.add_argument("--capability-cache", required=True)
    artifact.add_argument("--radio-checkpoint", required=True)
    artifact.add_argument("--radio-checkpoint-sha256", required=True)
    artifact.add_argument("--write-stamp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "feature":
        digest = validate_feature(args)
        print(digest if args.print_sha256 else json.dumps({"output_bundle_sha256": digest}, indent=2))
    elif args.command == "plan":
        payload = validate_plan(args.path)
        frames = payload["validation_frame_ids"]
        print(",".join(str(int(value)) for value in frames) if args.print_csv else json.dumps(payload, indent=2))
    else:
        print(validate_artifact(args))


if __name__ == "__main__":
    main()
