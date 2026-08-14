"""Stage only authority-authorized RGB/pose observations for LUDVIG mapping."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .ludvig_mapping_plan import LUDVIG_MAPPING_PLAN_SCHEMA
from .protocol import BENCHMARK_VERSION, canonical_json_sha256, sha256_file


STAGE_SCHEMA = "scannet_uqis_ludvig_mapping_observations_v1"
COLOR_EXTENSIONS = {1: "png", 2: "jpg"}


def _binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _stage_scene(job: dict[str, Any], output: Path) -> dict[str, Any]:
    scene_id = job["scene_id"]
    sens = Path(job["sens"]["path"]).resolve()
    if sha256_file(sens) != job["sens"]["sha256"]:
        raise ValueError(f"{scene_id}: .sens changed after mapping plan freeze")
    selected = set(job["selected_mapping_frame_ids"])
    scene_root = output / scene_id
    color_root = scene_root / "color"
    pose_root = scene_root / "pose"
    color_root.mkdir(parents=True)
    pose_root.mkdir(parents=True)
    records = []
    with sens.open("rb") as handle:
        if struct.unpack("I", handle.read(4))[0] != 4:
            raise ValueError(f"{scene_id}: unsupported .sens version")
        name_length = struct.unpack("Q", handle.read(8))[0]
        handle.seek(name_length, 1)
        intrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(64)), np.float32).reshape(4, 4)
        extrinsic_color = np.asarray(struct.unpack("f" * 16, handle.read(64)), np.float32).reshape(4, 4)
        intrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(64)), np.float32).reshape(4, 4)
        extrinsic_depth = np.asarray(struct.unpack("f" * 16, handle.read(64)), np.float32).reshape(4, 4)
        color_compression = struct.unpack("i", handle.read(4))[0]
        handle.seek(4 + 4 * 4 + 4, 1)
        frame_count = struct.unpack("Q", handle.read(8))[0]
        extension = COLOR_EXTENSIONS.get(color_compression)
        if extension is None:
            raise ValueError(f"{scene_id}: unsupported color compression {color_compression}")
        for index in range(frame_count):
            pose = np.asarray(struct.unpack("f" * 16, handle.read(64)), np.float32).reshape(4, 4)
            handle.seek(16, 1)
            color_bytes = struct.unpack("Q", handle.read(8))[0]
            depth_bytes = struct.unpack("Q", handle.read(8))[0]
            color_payload = handle.read(color_bytes)
            handle.seek(depth_bytes, 1)
            frame_id = f"{index:06d}"
            if frame_id not in selected:
                continue
            if not np.isfinite(pose).all():
                raise ValueError(f"{scene_id}/{frame_id}: selected pose is non-finite")
            color_path = color_root / f"{frame_id}.{extension}"
            pose_path = pose_root / f"{frame_id}.npy"
            color_path.write_bytes(color_payload)
            np.save(pose_path, pose, allow_pickle=False)
            records.append(
                {
                    "frame_id": frame_id,
                    "color": _binding(color_path, output),
                    "camera_to_world": _binding(pose_path, output),
                }
            )
    if [row["frame_id"] for row in records] != job["selected_mapping_frame_ids"]:
        raise ValueError(f"{scene_id}: staged frame coverage/order changed")
    intrinsic_root = scene_root / "intrinsic"
    intrinsic_root.mkdir()
    matrices = {
        "intrinsic_color": intrinsic_color,
        "extrinsic_color": extrinsic_color,
        "intrinsic_depth": intrinsic_depth,
        "extrinsic_depth": extrinsic_depth,
    }
    matrix_bindings = {}
    for name, matrix in matrices.items():
        path = intrinsic_root / f"{name}.npy"
        np.save(path, matrix, allow_pickle=False)
        matrix_bindings[name] = _binding(path, output)
    return {
        "scene_id": scene_id,
        "construction_scene_receipt_sha256": job["construction_scene_receipt_sha256"],
        "legal_field_frame_ids_sha256": job["legal_field_frame_ids_sha256"],
        "selected_mapping_frame_ids_sha256": job["selected_mapping_frame_ids_sha256"],
        "frame_count": len(records),
        "frames": records,
        "camera_matrices": matrix_bindings,
    }


def stage_ludvig_mapping_observations(
    mapping_plan_path: str | Path,
    output_dir: str | Path,
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    plan_path = Path(mapping_plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if expected_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("LUDVIG mapping plan differs from external expected digest")
    if (
        plan.get("schema_version") != LUDVIG_MAPPING_PLAN_SCHEMA
        or plan.get("benchmark_version") != BENCHMARK_VERSION
        or plan.get("status") != "mapping_planned_not_run"
        or plan.get("plan_sha256") != canonical_json_sha256(body)
    ):
        raise ValueError("LUDVIG mapping plan identity/digest changed")
    jobs_by_scene = {}
    for job in plan["jobs"]:
        job_body = {key: value for key, value in job.items() if key != "job_sha256"}
        if job["job_sha256"] != canonical_json_sha256(job_body):
            raise ValueError("LUDVIG mapping job digest changed")
        previous = jobs_by_scene.setdefault(job["scene_id"], job)
        shared = (
            "sens", "selected_mapping_frame_ids", "selected_mapping_frame_ids_sha256",
            "legal_field_frame_ids_sha256", "construction_scene_receipt_sha256",
        )
        if any(previous[key] != job[key] for key in shared):
            raise ValueError(f"{job['scene_id']}: CLIP/DINO observation plans differ")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    scenes = [_stage_scene(job, output) for job in jobs_by_scene.values()]
    receipt_body = {
        "schema_version": STAGE_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "mapping_observations_staged",
        "authorized_mapping_observations_only": True,
        "query_rgb_opened": False,
        "evaluator_labels_opened": False,
        "mapping_plan_sha256": plan["plan_sha256"],
        "mapping_plan_file_sha256": sha256_file(plan_path),
        "method_identity_sha256": plan["method_identity_sha256"],
        "scene_count": len(scenes),
        "frame_count": sum(scene["frame_count"] for scene in scenes),
        "scenes": scenes,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_json_sha256(receipt_body)}
    (output / "mapping_observation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
