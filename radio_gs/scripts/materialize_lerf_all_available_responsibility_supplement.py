#!/usr/bin/env python3
"""Render omitted-only exact responsibilities for all-available LERF views."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from radio_gs.config import load_config
from radio_gs.querying.all_available_source_view_authority import (
    file_record_value,
    load_reference_inputs,
)
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    SparseExactMarginalAuthorityWriter,
    sparse_exact_marginal_implementation_sha256,
)
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _gaussian_state_sha256,
    _sha256_tensor_rows,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import SimpleRadioDataset
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    sha256_file,
    write_frozen_json,
)


AUTHORITY_SCHEMA = (
    "radio_gs.lerf_all_available_responsibility_supplement_execution.v1"
)
SUPPLEMENT_CONTRACT = (
    "radio_gs.lerf_omitted_source_view_exact_marginal_supplement.v1"
)
SCHEMA_VERSION = 1


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_view_domain": "feature_manifest_order_minus_frozen_excluded_frames",
        "rendered_view_domain": "exactly_all_available_minus_legacy120",
        "responsibility": "exact_front_to_back_sparse_marginal",
        "legacy_authority_modified": False,
        "feature_independent": True,
        "query_independent": True,
        "per_scene_or_per_query_hyperparameters": False,
        "target_metric_execution_authorized": False,
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def access_audit(*, gpu_used: bool) -> dict[str, bool]:
    return {
        "source_feature_manifest_opened": True,
        "source_feature_tensors_opened": False,
        "source_poses_opened": bool(gpu_used),
        "source_geometry_opened": bool(gpu_used),
        "target_images_opened": False,
        "text_queries_opened": False,
        "labels_or_masks_opened": False,
        "target_metrics_opened": False,
        "gpu_used": bool(gpu_used),
    }


def _new(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result):
        raise ValueError(f"{label} must be canonical absolute")
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"{label} already exists: {result}")
    return result


def prepare_inputs(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="all-available supplement execution authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "method_contract",
        "method_contract_sha256",
        "reference_execution_authority",
        "output_responsibility_authority",
        "execution",
        "source_only_materialization_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("all-available supplement authority fields differ")
    authority = dict(raw)
    implementation = file_record_value(
        authority["implementation"], label="supplement implementation"
    )
    reference = file_record_value(
        authority["reference_execution_authority"],
        label="supplement reference execution authority",
    )
    execution = authority.get("execution")
    output_raw = str(authority.get("output_responsibility_authority", ""))
    output = Path(output_raw).expanduser().resolve()
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("status")
        != "authorized_source_only_omitted_view_responsibility"
        or implementation != file_record(Path(__file__).resolve())
        or authority.get("method_contract") != method_contract()
        or authority.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or not isinstance(execution, Mapping)
        or set(execution)
        != {
            "physical_gpu",
            "cuda_visible_devices",
            "program_device",
            "thermal_poll_seconds",
            "maximum_temperature_c",
        }
        or int(execution.get("physical_gpu", -1)) not in (0, 1)
        or str(execution.get("cuda_visible_devices"))
        != str(execution.get("physical_gpu"))
        or execution.get("program_device") != "cuda:0"
        or int(execution.get("thermal_poll_seconds", -1)) != 300
        or int(execution.get("maximum_temperature_c", -1)) != 88
        or output_raw != str(output)
        or authority.get("source_only_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != access_audit(gpu_used=False)
    ):
        raise ValueError("all-available supplement authority header differs")
    prepared = load_reference_inputs(
        reference["path"],
        expected_sha256=reference["sha256"],
        load_tensor_payloads=False,
    )
    if (
        authority.get("scene_id") != prepared["authority"]["scene_id"]
        or not prepared["domain_audit"].omitted_frames
    ):
        raise ValueError("supplement scene has no exact omitted-view domain")
    return {
        "authority": authority,
        "authority_record": {"path": str(source), "sha256": digest},
        "reference": prepared,
        "output": str(output),
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="supplement authority")
    output = _new(
        args.output_responsibility_authority,
        label="supplement responsibility authority",
    )
    reference = {
        "path": str(Path(args.reference_execution_authority).expanduser().resolve()),
        "sha256": str(args.reference_execution_authority_sha256),
    }
    prepared = load_reference_inputs(
        reference["path"],
        expected_sha256=reference["sha256"],
        load_tensor_payloads=False,
    )
    if not prepared["domain_audit"].omitted_frames:
        raise ValueError("legacy authority already covers all available source views")
    physical_gpu = int(args.physical_gpu)
    if physical_gpu not in (0, 1):
        raise ValueError("physical GPU must be 0 or 1")
    payload = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_source_only_omitted_view_responsibility",
        "scene_id": prepared["authority"]["scene_id"],
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "reference_execution_authority": reference,
        "output_responsibility_authority": str(output),
        "execution": {
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": str(physical_gpu),
            "program_device": "cuda:0",
            "thermal_poll_seconds": 300,
            "maximum_temperature_c": 88,
        },
        "source_only_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": access_audit(gpu_used=False),
    }
    write_frozen_json(authority_output, payload)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {
        "status": "authorized",
        "authority": record,
        "omitted_frame_count": len(prepared["domain_audit"].omitted_frames),
        "omitted_frame_ids": list(prepared["domain_audit"].omitted_frames),
        "output_responsibility_authority": str(output),
    }


def _dataset_for_manifest(prepared: Mapping[str, Any]) -> SimpleRadioDataset:
    records = prepared["records"]
    config = load_config(records["scene_config"]["path"])
    feature_dir = Path(records["feature_manifest"]["path"]).parent
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    pose_file_path = Path(raw_pose_file) if raw_pose_file else None
    pose_file = (
        str(pose_file_path)
        if pose_file_path is not None and pose_file_path.is_file()
        else None
    )
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    pose_dir_path = Path(raw_pose_dir) if raw_pose_dir else None
    fallback_pose_dir = feature_dir / "poses_w2c"
    pose_dir = (
        str(pose_dir_path)
        if pose_dir_path is not None and pose_dir_path.is_dir()
        else str(fallback_pose_dir)
        if fallback_pose_dir.is_dir()
        else None
    )
    feature_frames = [
        int(record["frame_idx"]) for record in prepared["feature_manifest"]["frames"]
    ]
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(
            int(prepared["responsibility"]["metadata"]["feature_height"]),
            int(prepared["responsibility"]["metadata"]["feature_width"]),
        ),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=feature_frames,
    )
    if list(map(int, dataset.frame_indices)) != feature_frames:
        raise ValueError("pose dataset order differs from feature manifest")
    return dataset


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    output = _new(prepared["output"], label="supplement responsibility authority")
    execution = prepared["authority"]["execution"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != execution["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from supplement authority")
    if not torch.cuda.is_available():
        raise RuntimeError("all-available responsibility supplement requires CUDA")
    device = torch.device("cuda:0")
    reference = prepared["reference"]
    records = reference["records"]
    model, _codec, renderer, _sharpener, _refiner, config, _hybrid = (
        load_render_pipeline(
            records["scene_config"]["path"],
            records["renderer_geometry_checkpoint"]["path"],
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
            expected_checkpoint_sha256=records["renderer_geometry_checkpoint"]["sha256"],
        )
    )
    dataset = _dataset_for_manifest(reference)
    omitted = list(reference["domain_audit"].omitted_frames)
    dataset_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    selected = [dataset_index[frame] for frame in omitted]
    poses = torch.stack(
        [torch.from_numpy(dataset.poses_w2c[index]).float().cpu() for index in selected]
    )
    legacy_metadata = reference["responsibility"]["metadata"]
    feature_height = int(legacy_metadata["feature_height"])
    feature_width = int(legacy_metadata["feature_width"])
    xyz_sha = _sha256_tensor_rows(model.get_xyz())
    state_sha = _gaussian_state_sha256(model)
    if (
        xyz_sha != legacy_metadata["xyz_sha256"]
        or state_sha != legacy_metadata["gaussian_state_sha256"]
        or int(model.get_xyz().shape[0])
        != int(reference["responsibility"]["num_gaussians"])
        or int(getattr(config, "feature_height", renderer.image_height))
        != feature_height
        or int(getattr(config, "feature_width", renderer.image_width)) != feature_width
    ):
        raise ValueError("loaded supplement geometry differs from legacy authority")
    compositor_source = Path(
        inspect.getsourcefile(inspect.unwrap(rasterize_single_view_contributions)) or ""
    ).resolve()
    metadata = {
        "schema_version": 1,
        "supplement_contract": SUPPLEMENT_CONTRACT,
        "assignment_mode": "exact_front_to_back_sparse_marginal",
        "registration_weight_mode": "exact_front_to_back_marginal_responsibility",
        "config": legacy_metadata["config"],
        "checkpoint": legacy_metadata["checkpoint"],
        "geometry_checkpoint_sha256": legacy_metadata["geometry_checkpoint_sha256"],
        "selected_dataset_indices": selected,
        "selected_frame_indices": omitted,
        "excluded_frame_ids": list(legacy_metadata["excluded_frame_ids"]),
        "feature_height": feature_height,
        "feature_width": feature_width,
        "post_compositor_alpha_threshold": 0.0,
        "depth_filter": "not_applied_to_exact_compositor_hits",
        "pose_sha256": _sha256_tensor_rows(poses),
        "intrinsics_sha256": _sha256_tensor_rows(
            renderer.scaled_intrinsics(feature_width, feature_height)
        ),
        "xyz_sha256": xyz_sha,
        "gaussian_state_sha256": state_sha,
        "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "builder_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "compositor_implementation_sha256": sha256_file(compositor_source),
        "authority_implementation_sha256": sparse_exact_marginal_implementation_sha256(),
        "legacy_responsibility_authority": dict(
            records["responsibility_authority"]
        ),
        "feature_manifest": dict(records["feature_manifest"]),
        "feature_independent": True,
        "query_independent": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "target_metrics_opened": False,
    }
    writer = SparseExactMarginalAuthorityWriter(
        output,
        metadata=metadata,
        frame_indices=omitted,
        num_gaussians=int(model.get_xyz().shape[0]),
        num_pixels=feature_height * feature_width,
    )
    for view_index, pose in enumerate(poses):
        if view_index in writer.completed_view_indices:
            continue
        hits = rasterize_single_view_contributions(
            model,
            renderer,
            pose.to(device),
            height=feature_height,
            width=feature_width,
        )
        writer.add_view(
            view_index,
            hits["gaussian_ids"],
            hits["pixel_ids"],
            hits["weights"],
        )
        if (view_index + 1) % 25 == 0 or view_index + 1 == len(omitted):
            print(
                json.dumps(
                    {
                        "event": "supplement_view_progress",
                        "views_complete": view_index + 1,
                        "views_total": len(omitted),
                    }
                ),
                flush=True,
            )
        del hits
        torch.cuda.empty_cache()
    authority_path, digest = writer.finalize()
    return {
        "status": "complete_source_only_omitted_view_responsibility",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": dict(prepared["authority_record"]),
        "responsibility_authority": {
            "path": str(authority_path),
            "sha256": digest,
        },
        "omitted_frame_count": len(omitted),
        "elapsed_seconds": time.monotonic() - started,
        "access_audit": access_audit(gpu_used=True),
        "metric_execution_authorized": False,
        "metric_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--reference-execution-authority", required=True)
    build.add_argument("--reference-execution-authority-sha256", required=True)
    build.add_argument("--output-responsibility-authority", required=True)
    build.add_argument("--physical-gpu", type=int, choices=(0, 1), required=True)
    build.add_argument("--authority-output", required=True)
    build.set_defaults(handler=build_authority)
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--execution-authority-sha256", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "METHOD_CONTRACT_SHA256",
    "SUPPLEMENT_CONTRACT",
    "access_audit",
    "method_contract",
    "prepare_inputs",
]
