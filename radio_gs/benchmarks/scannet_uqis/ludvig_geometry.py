"""Stage and train authority-bound Gaussian geometry for UQIS LUDVIG fields."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import rotation_matrix_to_qvec

from .ludvig_mapping_stage import STAGE_SCHEMA
from .protocol import BENCHMARK_VERSION, canonical_json_sha256, sha256_file


GEOMETRY_STAGING_SCHEMA = "scannet_uqis_ludvig_geometry_staging_v1"
GEOMETRY_RUN_SCHEMA = "scannet_uqis_ludvig_geometry_run_v1"
PINNED_3DGS_COMMIT = "f7a116fb1397d9842239127d39dc212f93171f70"
PINNED_TRAIN_ENTRYPOINT_SHA256 = (
    "c5a61947e2abcf56bf83451ae9633799d96894910ea2982a01f209c47cec462d"
)


def _load_observation_receipt(
    path: Path, *, expected_receipt_sha256: str
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema_version") != STAGE_SCHEMA
        or receipt.get("benchmark_version") != BENCHMARK_VERSION
        or receipt.get("status") != "mapping_observations_staged"
        or receipt.get("receipt_sha256") != expected_receipt_sha256
        or receipt.get("receipt_sha256") != canonical_json_sha256(body)
    ):
        raise ValueError("mapping-observation receipt identity/digest changed")
    return receipt


def _verify_relative(root: Path, binding: dict[str, Any], label: str) -> Path:
    relative = Path(str(binding.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: path escapes mapping-observation root")
    path = root / relative
    if (
        not path.is_file()
        or path.stat().st_size != int(binding.get("bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise ValueError(f"{label}: observation binding changed")
    return path


def stage_ludvig_geometry_scene(
    observation_receipt_path: str | Path,
    *,
    expected_observation_receipt_sha256: str,
    construction_authority_path: str | Path,
    expected_construction_authority_sha256: str,
    scene_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a COLMAP text scene from one legal mapping-observation inventory."""

    receipt_path = Path(observation_receipt_path).resolve()
    observation_root = receipt_path.parent
    receipt = _load_observation_receipt(
        receipt_path, expected_receipt_sha256=expected_observation_receipt_sha256
    )
    authority_path = Path(construction_authority_path).resolve()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_body = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    if (
        authority.get("authority_sha256") != expected_construction_authority_sha256
        or authority.get("authority_sha256") != canonical_json_sha256(authority_body)
        or authority.get("construction_formal_eligible") is not True
    ):
        raise ValueError("construction authority identity/digest changed")
    scene_receipts = [row for row in receipt["scenes"] if row["scene_id"] == scene_id]
    if len(scene_receipts) != 1:
        raise ValueError(f"mapping observations contain no unique scene {scene_id}")
    scene = scene_receipts[0]
    if (
        scene["construction_scene_receipt_sha256"]
        != authority["scene_derivation_receipt_sha256"].get(scene_id)
    ):
        raise ValueError("scene derivation authority changed")
    source = authority["verified_scene_sources"][scene_id]["mesh"]
    mesh = Path(source["path"])
    if (
        not mesh.is_file()
        or mesh.stat().st_size != source["bytes"]
        or sha256_file(mesh) != source["sha256"]
    ):
        raise ValueError("authority-bound ScanNet mesh changed")

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    images = output / "images"
    sparse = output / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    matrices = {
        name: np.load(
            _verify_relative(observation_root, binding, f"{scene_id}/{name}"),
            allow_pickle=False,
        )
        for name, binding in scene["camera_matrices"].items()
    }
    intrinsic = np.asarray(matrices["intrinsic_color"], dtype=np.float64)
    if intrinsic.shape != (4, 4) or not np.isfinite(intrinsic).all():
        raise ValueError("color intrinsic is invalid")
    frame_records = []
    image_lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] intentionally empty; poses come from ScanNet .sens",
    ]
    raster_size = None
    for rank, frame in enumerate(scene["frames"]):
        color = _verify_relative(
            observation_root, frame["color"], f"{scene_id}/{frame['frame_id']}/color"
        )
        pose_path = _verify_relative(
            observation_root,
            frame["camera_to_world"],
            f"{scene_id}/{frame['frame_id']}/pose",
        )
        pose = np.load(pose_path, allow_pickle=False).astype(np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"{scene_id}/{frame['frame_id']}: pose is invalid")
        with Image.open(color) as image:
            size = image.size
        if raster_size is None:
            raster_size = size
        elif raster_size != size:
            raise ValueError(f"{scene_id}: mapping RGB dimensions differ")
        name = f"{rank:06d}_{frame['frame_id']}{color.suffix.lower()}"
        os.symlink(str(color), str(images / name))
        w2c = np.linalg.inv(pose)
        qvec = rotation_matrix_to_qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]
        image_lines.append(
            f"{rank + 1} "
            + " ".join(f"{value:.17g}" for value in (*qvec, *tvec))
            + f" 1 {name}"
        )
        image_lines.append("")
        frame_records.append(
            {
                "rank": rank,
                "frame_id": frame["frame_id"],
                "staged_name": name,
                "source_sha256": frame["color"]["sha256"],
            }
        )
    if raster_size is None:
        raise ValueError(f"{scene_id}: no mapping observations")
    width, height = raster_size
    (sparse / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {intrinsic[0,0]:.17g} "
        f"{intrinsic[1,1]:.17g} {intrinsic[0,2]:.17g} {intrinsic[1,2]:.17g}\n",
        encoding="ascii",
    )
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="ascii")
    # Graphdeco's loader silently drops ScanNet PLYs that also carry mesh
    # faces. Derive a point-only PLY from the byte-bound official vertex
    # element without changing coordinates or colors.
    source_ply = PlyData.read(str(mesh))
    source_vertices = source_ply["vertex"].data
    graphdeco_dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
    )
    graphdeco_vertices = np.zeros(len(source_vertices), dtype=graphdeco_dtype)
    for name in ("x", "y", "z", "red", "green", "blue"):
        graphdeco_vertices[name] = source_vertices[name]
    point_cloud_path = sparse / "points3D.ply"
    PlyData(
        [PlyElement.describe(graphdeco_vertices, "vertex")],
        text=False,
        byte_order="<",
    ).write(str(point_cloud_path))
    body = {
        "schema_version": GEOMETRY_STAGING_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "geometry_staging_complete",
        "scene_id": scene_id,
        "construction_authority_sha256": expected_construction_authority_sha256,
        "mapping_observation_receipt_sha256": expected_observation_receipt_sha256,
        "construction_scene_receipt_sha256": scene[
            "construction_scene_receipt_sha256"
        ],
        "camera_model": "PINHOLE",
        "pose_conversion": "ScanNet_camera_to_world_inverted_to_COLMAP_world_to_camera",
        "raster_size": [width, height],
        "frame_count": len(frame_records),
        "frames": frame_records,
        "initial_point_cloud": {
            "role": "point_only_derivative_of_official_scannet_colored_mesh_vertices",
            "source_path": str(mesh),
            "source_bytes": mesh.stat().st_size,
            "source_sha256": source["sha256"],
            "derived_path": str(point_cloud_path),
            "derived_bytes": point_cloud_path.stat().st_size,
            "derived_sha256": sha256_file(point_cloud_path),
            "vertex_count": len(source_vertices),
            "derived_normal_rule": "zero_normals_required_by_original_graphdeco_loader",
            "coordinates_and_rgb_preserved": True,
        },
        "cameras_txt_sha256": sha256_file(sparse / "cameras.txt"),
        "images_txt_sha256": sha256_file(sparse / "images.txt"),
        "query_frames_opened": False,
        "evaluator_labels_opened": False,
    }
    manifest = {**body, "receipt_sha256": canonical_json_sha256(body)}
    (output / "geometry_staging_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_ludvig_geometry_training(
    staging_dir: str | Path,
    *,
    expected_staging_receipt_sha256: str,
    ludvig_upstream: str | Path,
    python: str | Path,
    output_dir: str | Path,
    iterations: int,
    device_index: int = 0,
) -> dict[str, Any]:
    """Run vendored original-3DGS geometry; short runs remain smoke-only."""

    staging = Path(staging_dir).resolve()
    staging_receipt_path = staging / "geometry_staging_receipt.json"
    staging_receipt = json.loads(staging_receipt_path.read_text(encoding="utf-8"))
    staging_body = {
        key: value for key, value in staging_receipt.items() if key != "receipt_sha256"
    }
    if (
        staging_receipt.get("receipt_sha256") != expected_staging_receipt_sha256
        or staging_receipt.get("receipt_sha256") != canonical_json_sha256(staging_body)
    ):
        raise ValueError("geometry staging receipt identity/digest changed")
    if iterations <= 0 or iterations > 30000:
        raise ValueError("geometry iterations must be in [1,30000]")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    upstream = Path(ludvig_upstream).resolve()
    training_root = (
        upstream
        if (upstream / "train.py").is_file()
        else upstream / "gaussiansplatting"
    )
    entrypoint = training_root / "train.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"missing Gaussian training entrypoint: {entrypoint}")
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(training_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "-C", str(training_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError("unable to verify Gaussian training checkout") from error
    implementation = {
        "git_commit": git_commit,
        "git_clean": not bool(git_status),
        "train_entrypoint": str(entrypoint),
        "train_entrypoint_sha256": sha256_file(entrypoint),
    }
    implementation["identity_sha256"] = canonical_json_sha256(implementation)
    if iterations == 30000 and (
        git_commit != PINNED_3DGS_COMMIT
        or git_status
        or implementation["train_entrypoint_sha256"]
        != PINNED_TRAIN_ENTRYPOINT_SHA256
    ):
        raise ValueError("formal geometry requires the pinned clean original-3DGS checkout")
    command = [
        str(Path(python).resolve()),
        str(entrypoint),
        "--source_path", str(staging),
        "--model_path", str(output / "model"),
        "--iterations", str(iterations),
        "--test_iterations", "-1",
        "--save_iterations", str(iterations),
        "--port", str(6010 + device_index),
        "--quiet",
    ]
    output.mkdir(parents=True)
    log = output / "training.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(upstream),
            str(training_root),
            inherited_pythonpath,
        )
        if value
    )
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=training_root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.monotonic() - started
    point_cloud = output / "model" / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if completed.returncode or not point_cloud.is_file():
        raise RuntimeError(f"geometry training failed; see {log}")
    formal_eligible = (
        iterations == 30000
        and git_commit == PINNED_3DGS_COMMIT
        and not git_status
        and implementation["train_entrypoint_sha256"]
        == PINNED_TRAIN_ENTRYPOINT_SHA256
    )
    body = {
        "schema_version": GEOMETRY_RUN_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "geometry_complete" if formal_eligible else "geometry_smoke_complete",
        "formal_field_eligible": formal_eligible,
        "scene_id": staging_receipt["scene_id"],
        "geometry_staging_receipt_sha256": expected_staging_receipt_sha256,
        "iterations": iterations,
        "implementation": implementation,
        "command": command,
        "elapsed_seconds": elapsed,
        "point_cloud": {
            "path": str(point_cloud),
            "bytes": point_cloud.stat().st_size,
            "sha256": sha256_file(point_cloud),
        },
        "training_log": {
            "path": str(log),
            "bytes": log.stat().st_size,
            "sha256": sha256_file(log),
        },
    }
    manifest = {**body, "receipt_sha256": canonical_json_sha256(body)}
    (output / "geometry_run_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
