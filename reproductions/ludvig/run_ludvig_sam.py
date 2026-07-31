#!/usr/bin/env python3
"""Fail-closed launcher for the released LUDVIG-SAM evaluation.

This wrapper does not change RADIO-GS fields or query code.  It validates the
upstream commit, creates an isolated config/output directory, records protocol
visibility, and serializes GPU 0 with the repository-wide lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import time
from typing import Any

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks"
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "protocol_audit_20260731" / "ludvig"
LOCK_PATH = Path("/tmp/radio-gs-gpu0.lock")
DEFAULT_DRIVER_LIBRARY_DIR = Path("/root/baselines/LUDVIG/.driver535")
UPSTREAM_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
NVOS_SCENES = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
SPIN_SCENES = (
    "orchids",
    "leaves",
    "fern",
    "room",
    "horns",
    "fortress",
    "fork",
    "pinecone",
    "truck",
    "lego",
)
SPIN_ANNOTATION_FOLDERS = {
    "orchids": "orchids (llff)",
    "leaves": "leaves (llff)",
    "fern": "fern (llff)",
    "room": "room (llff)",
    "horns": "horns (llff)",
    "fortress": "fortress (llff)",
    "fork": "fork (nerf_supervision)",
    "pinecone": "pinecone (nerf_real_360)",
    "truck": "Truck (Tanks & Temples)",
    "lego": "lego_real_night_radial",
}
SPIN_SOURCE_RELATIVE = {
    "orchids": "SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/orchids",
    "leaves": "SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/leaves",
    "fern": "SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/fern",
    "room": "SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/room",
    "horns": "SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/horns",
    "fortress": "SPIn-NeRF/source_images/llff_google_drive/extracted/nerf_llff_data/fortress",
    "fork": "SPIn-NeRF/source_images/nerf_supervision/extracted/nerf_supervision_data/fork",
    "pinecone": "SPIn-NeRF/source_images/nerf_real_360/extracted/pinecone",
    "truck": "SPIn-NeRF/source_images/tandt/extracted/tandt/truck",
    "lego": "SPIn-NeRF/source_images/lego_real_night_radial/lego_real_night_radial",
}
NVOS_IMAGE_SIZE = {
    scene: (1600, 1199) for scene in NVOS_SCENES
}
SPIN_IMAGE_SIZE = {
    "orchids": (1600, 1199),
    "leaves": (1600, 1199),
    "fern": (1600, 1199),
    "room": (1600, 1200),
    "horns": (1600, 1199),
    "fortress": (1600, 1199),
    "fork": (1600, 1202),
    "pinecone": (1600, 1199),
    "truck": (979, 546),
    "lego": (1015, 764),
}
COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


class ProtocolError(RuntimeError):
    """Raised before GPU work when a run would be mislabeled or incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _driver_library(driver_library_dir: Path) -> Path:
    for name in ("libcuda.so.1", "libcuda.so"):
        candidate = driver_library_dir / name
        if candidate.exists():
            return candidate
    raise ProtocolError(
        f"No libcuda.so.1 or libcuda.so in --driver-library-dir "
        f"{driver_library_dir}"
    )


def _runtime_environment(args: argparse.Namespace) -> tuple[dict[str, str], Path]:
    driver_library_dir = args.driver_library_dir.resolve()
    driver_library = _driver_library(driver_library_dir)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["LD_LIBRARY_PATH"] = str(driver_library_dir) + os.pathsep + environment.get(
        "LD_LIBRARY_PATH", ""
    )
    if args.pythonpath:
        environment["PYTHONPATH"] = str(args.pythonpath.resolve()) + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
    return environment, driver_library


def _validate_upstream(checkout: Path) -> None:
    if _git_head(checkout) != UPSTREAM_COMMIT:
        raise ProtocolError(
            f"LUDVIG checkout must be pinned to {UPSTREAM_COMMIT}; "
            f"found {_git_head(checkout)}"
        )
    entrypoint = checkout / "ludvig_uplift.py"
    source = entrypoint.read_text(encoding="utf-8")
    if '"--seed"' not in source or "reproducibility(args.seed)" not in source:
        raise ProtocolError(
            "The audited reproduction patch is not applied; "
            "run `git apply reproductions/ludvig/patches/"
            "0001-reproduction-seeds-and-json-results.patch` in the checkout."
        )


def _ensure_link(link: Path, target: Path) -> None:
    if not target.exists():
        raise ProtocolError(f"Required input does not exist: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ProtocolError(f"Existing staging link has the wrong target: {link}")
        return
    if link.exists():
        raise ProtocolError(f"Refusing to replace existing staging path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def _read_colmap_cameras_binary(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", handle.read(24)
            )
            if model_id not in COLMAP_CAMERA_MODELS:
                raise ProtocolError(
                    f"Unknown COLMAP camera model id {model_id} in {path}"
                )
            model, num_params = COLMAP_CAMERA_MODELS[model_id]
            params = list(struct.unpack(f"<{num_params}d", handle.read(8 * num_params)))
            records.append(
                {
                    "camera_id": camera_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                }
            )
    return records


def _read_colmap_image_poses(path: Path) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            image_id = struct.unpack("<i", handle.read(4))[0]
            qvec = struct.unpack("<4d", handle.read(32))
            tvec = struct.unpack("<3d", handle.read(24))
            camera_id = struct.unpack("<i", handle.read(4))[0]
            name_bytes = bytearray()
            while True:
                byte = handle.read(1)
                if not byte:
                    raise ProtocolError(f"Truncated COLMAP image name in {path}")
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
            name = name_bytes.decode("utf-8")
            (num_points,) = struct.unpack("<Q", handle.read(8))
            handle.seek(24 * num_points, os.SEEK_CUR)
            poses[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
            }
    return poses


def _max_pose_delta(
    original: dict[str, dict[str, Any]],
    converted: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    if set(original) != set(converted):
        raise ProtocolError(
            "Undistorted COLMAP conversion changed the registered image cohort"
        )
    max_qvec_delta = 0.0
    max_tvec_delta = 0.0
    for name in original:
        max_qvec_delta = max(
            max_qvec_delta,
            max(
                abs(a - b)
                for a, b in zip(original[name]["qvec"], converted[name]["qvec"])
            ),
        )
        max_tvec_delta = max(
            max_tvec_delta,
            max(
                abs(a - b)
                for a, b in zip(original[name]["tvec"], converted[name]["tvec"])
            ),
        )
    return max_qvec_delta, max_tvec_delta


def _scaled_pinhole_audit(
    camera: dict[str, Any], target_width: int, target_height: int
) -> dict[str, Any]:
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = camera["params"][0]
        cx, cy = camera["params"][1:3]
    elif camera["model"] == "PINHOLE":
        fx, fy, cx, cy = camera["params"][:4]
    else:
        raise ProtocolError(
            f"LUDVIG requires PINHOLE/SIMPLE_PINHOLE, found {camera['model']}"
        )
    source_width = float(camera["width"])
    source_height = float(camera["height"])
    center_error = max(
        abs(cx - source_width / 2.0),
        abs(cy - source_height / 2.0),
    )
    if center_error > 1e-3:
        raise ProtocolError(
            "LUDVIG's loader discards cx/cy, but the staged camera principal "
            f"point is not centered (max error {center_error:.6g}px)"
        )

    source_aspect = source_height / source_width
    target_aspect = target_height / target_width
    if source_aspect > target_aspect:
        crop_width = source_width
        crop_height = source_width * target_aspect
    else:
        crop_height = source_height
        crop_width = source_height / target_aspect
    scaled_fx = fx * target_width / crop_width
    scaled_fy = fy * target_height / crop_height
    if not all(math.isfinite(item) and item > 0 for item in (scaled_fx, scaled_fy)):
        raise ProtocolError("Invalid scaled focal length in staged COLMAP model")
    return {
        "source_width": camera["width"],
        "source_height": camera["height"],
        "source_fx": fx,
        "source_fy": fy,
        "source_cx": cx,
        "source_cy": cy,
        "source_principal_point_center_error_px": center_error,
        "target_width": target_width,
        "target_height": target_height,
        "center_crop_width": crop_width,
        "center_crop_height": crop_height,
        "effective_target_fx": scaled_fx,
        "effective_target_fy": scaled_fy,
        "effective_target_cx": target_width / 2.0,
        "effective_target_cy": target_height / 2.0,
    }


def _stage_nvos_pinhole_colmap(
    source_scene: Path,
    staging_scene: Path,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    converted_sparse = source_scene / "dense" / "sparse"
    original_sparse = source_scene / "sparse" / "0"
    camera_records = _read_colmap_cameras_binary(
        converted_sparse / "cameras.bin"
    )
    if len(camera_records) != 1:
        raise ProtocolError(
            "Audited NVOS staging expects exactly one shared COLMAP camera; "
            f"found {len(camera_records)}"
        )
    camera = camera_records[0]
    intrinsics_audit = _scaled_pinhole_audit(
        camera, target_width, target_height
    )

    images_dir = source_scene / "images"
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ProtocolError(f"No RGB images in {images_dir}")
    image_sizes = set()
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_sizes.add(image.size)
    expected_size = (camera["width"], camera["height"])
    if image_sizes != {expected_size}:
        raise ProtocolError(
            f"RGB dimensions {sorted(image_sizes)} do not match staged COLMAP "
            f"camera dimensions {expected_size}"
        )

    original_poses = _read_colmap_image_poses(original_sparse / "images.bin")
    converted_poses = _read_colmap_image_poses(converted_sparse / "images.bin")
    rgb_names = {path.name for path in image_paths}
    if rgb_names != set(converted_poses):
        raise ProtocolError(
            "Staged RGB/COLMAP image names differ: "
            f"missing_rgb={sorted(set(converted_poses) - rgb_names)}, "
            f"unregistered_rgb={sorted(rgb_names - set(converted_poses))}"
        )
    max_qvec_delta, max_tvec_delta = _max_pose_delta(
        original_poses, converted_poses
    )
    if max_qvec_delta > 1e-12 or max_tvec_delta > 1e-12:
        raise ProtocolError(
            "Undistorted COLMAP conversion changed camera poses beyond 1e-12: "
            f"q={max_qvec_delta}, t={max_tvec_delta}"
        )

    _ensure_link(staging_scene / "images", images_dir)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        _ensure_link(
            staging_scene / "sparse" / "0" / name,
            converted_sparse / name,
        )
    return {
        "strategy": "stage_existing_colmap_undistortion_output",
        "source_scene": str(source_scene.resolve()),
        "converted_sparse_source": str(converted_sparse.resolve()),
        "staged_scene": str(staging_scene.resolve()),
        "camera_model": camera["model"],
        "registered_images": len(converted_poses),
        "rgb_images": len(image_paths),
        "rgb_dimensions": list(expected_size),
        "max_qvec_delta_vs_original_sparse": max_qvec_delta,
        "max_tvec_delta_vs_original_sparse": max_tvec_delta,
        "pose_equivalence_tolerance": 1e-12,
        "intrinsics": intrinsics_audit,
        "source_sha256": {
            name: _sha256(converted_sparse / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "raw_dataset_modified": False,
    }


def _image_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise ProtocolError(f"Missing RGB directory: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ProtocolError(f"No RGB images in {directory}")
    return {path.name: _sha256(path) for path in paths}


def _stage_spin_llff_pinhole_colmap(
    spin_source_scene: Path,
    converted_source_scene: Path,
    staging_scene: Path,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    """Reuse a COLMAP undistortion only after proving raw-scene identity."""

    spin_sparse = spin_source_scene / "sparse" / "0"
    converted_raw_sparse = converted_source_scene / "sparse" / "0"
    sparse_hashes = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        spin_path = spin_sparse / name
        converted_path = converted_raw_sparse / name
        if not spin_path.is_file() or not converted_path.is_file():
            raise ProtocolError(
                f"Missing raw COLMAP file needed for SPIn equivalence: {name}"
            )
        spin_hash = _sha256(spin_path)
        converted_hash = _sha256(converted_path)
        if spin_hash != converted_hash:
            raise ProtocolError(
                "SPIn source and proposed PINHOLE conversion do not share "
                f"the same raw {name}"
            )
        sparse_hashes[name] = spin_hash

    spin_rgb_hashes = _image_hashes(spin_source_scene / "images")
    converted_raw_rgb_hashes = _image_hashes(
        converted_source_scene / "images_distort"
    )
    if spin_rgb_hashes != converted_raw_rgb_hashes:
        missing = sorted(set(converted_raw_rgb_hashes) - set(spin_rgb_hashes))
        extra = sorted(set(spin_rgb_hashes) - set(converted_raw_rgb_hashes))
        changed = sorted(
            name
            for name in set(spin_rgb_hashes) & set(converted_raw_rgb_hashes)
            if spin_rgb_hashes[name] != converted_raw_rgb_hashes[name]
        )
        raise ProtocolError(
            "SPIn raw RGBs differ from the proposed conversion source: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    pinhole = _stage_nvos_pinhole_colmap(
        converted_source_scene,
        staging_scene,
        target_width,
        target_height,
    )
    return {
        "strategy": "reuse_verified_identical_llff_colmap_undistortion",
        "spin_source_scene": str(spin_source_scene.resolve()),
        "converted_source_scene": str(converted_source_scene.resolve()),
        "raw_sparse_sha256": sparse_hashes,
        "raw_rgb_sha256": spin_rgb_hashes,
        "raw_rgb_images": len(spin_rgb_hashes),
        "raw_scene_identity_proven": True,
        "pinhole": pinhole,
        "raw_dataset_modified": False,
    }


def _resolve_inputs(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    benchmark_root = args.benchmark_root.resolve()
    if args.benchmark == "nvos":
        if getattr(args, "stage_spin_llff_pinhole_from", None) is not None:
            raise ProtocolError(
                "--stage-spin-llff-pinhole-from is only valid for SPIn-NeRF"
            )
        if args.scene not in NVOS_SCENES:
            raise ProtocolError(f"Unknown NVOS task: {args.scene}")
        source_scene = "horns" if args.scene.startswith("horns_") else args.scene
        colmap_dir = (
            benchmark_root
            / "NVOS"
            / "llff_undistorted"
            / f"{source_scene}_undistort"
        )
        mask_pattern = (
            benchmark_root
            / "NVOS"
            / "official_annotations"
            / "llff"
            / "{{}}"
            / "{}"
        )
        default_gs = (
            benchmark_root
            / "gaussfm_jobs"
            / "nvos_strict_unseen_v1"
            / "scenes"
            / args.scene
            / "geometry"
            / "point_cloud"
            / "iteration_30000"
            / "point_cloud.ply"
        )
        if args.geometry_protocol == "released_all_view":
            if args.gs_source is None:
                raise ProtocolError(
                    "--geometry-protocol released_all_view requires --gs-source; "
                    "the local default checkpoint is strict-unseen geometry."
                )
            strict_unseen_exact_match = False
            geometry_note = "all-view checkpoint supplied by caller"
        else:
            if args.gs_source is not None:
                raise ProtocolError(
                    "--gs-source is only accepted with --geometry-protocol released_all_view"
                )
            args.gs_source = default_gs
            strict_unseen_exact_match = False
            geometry_note = (
                "hybrid diagnostic: strict-unseen 3DGS geometry, but released "
                "LUDVIG uplifting still sees the target RGB"
            )
        width, height = NVOS_IMAGE_SIZE[args.scene]
        colmap_staging = None
        if args.stage_nvos_pinhole:
            staged_colmap_dir = (
                run_dir / "staging" / "colmap_pinhole_undistorted"
            )
            colmap_staging = _stage_nvos_pinhole_colmap(
                colmap_dir,
                staged_colmap_dir,
                width,
                height,
            )
            colmap_dir = staged_colmap_dir
        return {
            "colmap_dir": colmap_dir,
            "colmap_staging": colmap_staging,
            "mask_pattern": str(mask_pattern),
            "gs_source": args.gs_source,
            "width": width,
            "height": height,
            "strict_unseen_exact_match": strict_unseen_exact_match,
            "target_rgb_visible_during_gaussian_splatting_training": (
                args.geometry_protocol == "released_all_view"
            ),
            "geometry_note": geometry_note,
            "cohort": list(NVOS_SCENES),
            "reference_mask_calibration": False,
            "aggregation": "equal_weight_macro_over_8_tasks",
        }

    if getattr(args, "stage_nvos_pinhole", False):
        raise ProtocolError("--stage-nvos-pinhole is only valid for NVOS")
    if args.scene not in SPIN_SCENES:
        raise ProtocolError(f"Unknown SPIn-NeRF scene: {args.scene}")
    if args.gs_source is None:
        raise ProtocolError(
            "SPIn-NeRF has no local pretrained 3DGS checkpoint; pass an all-view "
            "30k --gs-source. Do not silently reuse a checkpoint in another camera frame."
        )
    staging = run_dir / "staging" / "SPIn-NeRF_masks"
    annotation = (
        benchmark_root
        / "SPIn-NeRF"
        / "multiview_annotations"
        / SPIN_ANNOTATION_FOLDERS[args.scene]
    )
    _ensure_link(staging / args.scene, annotation)
    width, height = SPIN_IMAGE_SIZE[args.scene]
    colmap_dir = benchmark_root / SPIN_SOURCE_RELATIVE[args.scene]
    if not colmap_dir.is_dir():
        raise ProtocolError(f"Missing SPIn source scene: {colmap_dir}")
    colmap_staging = None
    converted_source = getattr(args, "stage_spin_llff_pinhole_from", None)
    if converted_source is not None:
        staged_colmap_dir = (
            run_dir / "staging" / "colmap_pinhole_undistorted"
        )
        colmap_staging = _stage_spin_llff_pinhole_colmap(
            colmap_dir,
            Path(converted_source).resolve(),
            staged_colmap_dir,
            width,
            height,
        )
        colmap_dir = staged_colmap_dir
    else:
        cameras = _read_colmap_cameras_binary(
            colmap_dir / "sparse" / "0" / "cameras.bin"
        )
        unsupported = sorted(
            {camera["model"] for camera in cameras}
            - {"PINHOLE", "SIMPLE_PINHOLE"}
        )
        if unsupported:
            raise ProtocolError(
                "SPIn source has an unsupported distorted COLMAP model "
                f"{unsupported}; pass --stage-spin-llff-pinhole-from with a "
                "verified conversion of this exact raw scene"
            )
    return {
        "colmap_dir": colmap_dir,
        "colmap_staging": colmap_staging,
        "mask_pattern": str(staging / "{}"),
        "gs_source": args.gs_source,
        "width": width,
        "height": height,
        "strict_unseen_exact_match": False,
        "target_rgb_visible_during_gaussian_splatting_training": True,
        "geometry_note": "released all-view 3DGS geometry supplied by caller",
        "cohort": list(SPIN_SCENES),
        "reference_mask_calibration": True,
        "aggregation": "frame_mean_then_equal_weight_macro_over_10_scenes",
    }


def _config(args: argparse.Namespace, run_dir: Path, inputs: dict[str, Any]) -> dict:
    feature = {
        "name": "predictors.sam.SAMDataset",
        "thres": 1 if args.benchmark == "nvos" else 0.4,
        "multimask_output": args.benchmark == "spin",
        "sam_ckpt": str(args.sam_checkpoint.resolve()),
        "scribble": inputs["mask_pattern"],
    }
    evaluation = {
        "name": (
            "evaluation.spin_nvos.segmentation.SegmentationNVOSSAM"
            if args.benchmark == "nvos"
            else "evaluation.spin_nvos.segmentation.SegmentationSPInSAM"
        ),
        "segmentation_3d": True,
        "maskdir": inputs["mask_pattern"],
    }
    if args.benchmark == "nvos":
        evaluation["thresholding"] = 75
    return {
        "tag": "sam",
        "dst_dir": str(run_dir),
        "prune_gaussians": 0.5,
        "feature": feature,
        "evaluation": evaluation,
    }


def launch(args: argparse.Namespace) -> Path:
    run_family_dir = (
        args.output_root.resolve()
        / args.benchmark
        / args.geometry_protocol
        / args.scene
        / f"seed_{args.seed}"
    )
    if args.attempt_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.attempt_id):
            raise ProtocolError(
                "--attempt-id must contain only letters, digits, '.', '_' or '-'"
            )
        run_dir = run_family_dir / "attempts" / args.attempt_id
    else:
        run_dir = run_family_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing_status = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("status")
        raise ProtocolError(
            f"Refusing to overwrite existing attempt ({existing_status}) at "
            f"{manifest_path}; use a new --attempt-id"
        )
    _validate_upstream(args.upstream.resolve())
    inputs = _resolve_inputs(args, run_dir)
    for name in ("colmap_dir", "gs_source"):
        if not Path(inputs[name]).exists():
            raise ProtocolError(f"Missing {name}: {inputs[name]}")
    if not args.sam_checkpoint.exists():
        raise ProtocolError(f"Missing SAM checkpoint: {args.sam_checkpoint}")

    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(args, run_dir, inputs), sort_keys=False),
        encoding="utf-8",
    )
    command = [
        str(args.python.resolve()),
        str(args.upstream.resolve() / "ludvig_uplift.py"),
        "--colmap_dir",
        str(inputs["colmap_dir"]),
        "--gs_source",
        str(inputs["gs_source"]),
        "--config",
        str(config_path),
        "--height",
        str(inputs["height"]),
        "--width",
        str(inputs["width"]),
        "--tag",
        args.scene,
        "--seed",
        str(args.seed),
    ]
    manifest = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "benchmark": "NVOS" if args.benchmark == "nvos" else "SPIn-NeRF",
        "scene": args.scene,
        "seed": args.seed,
        "method": "LUDVIG-SAM",
        "protocol_id": "ludvig_official_online_multiview_v1",
        "geometry_protocol": args.geometry_protocol,
        "geometry_note": inputs["geometry_note"],
        "attempt_id": args.attempt_id,
        "prior_attempt_manifests": [
            str(path)
            for path in sorted(run_family_dir.rglob("run_manifest.json"))
            if path != manifest_path
        ],
        "colmap_staging": inputs["colmap_staging"],
        "strict_unseen_exact_match": inputs["strict_unseen_exact_match"],
        "target_rgb_visible_during_gaussian_splatting_training": inputs[
            "target_rgb_visible_during_gaussian_splatting_training"
        ],
        "target_rgb_visible_during_uplifting": True,
        "target_view_2d_foundation_model_calls": True,
        "reference_mask_calibration": inputs["reference_mask_calibration"],
        "target_masks_scoring_only": True,
        "aggregation": inputs["aggregation"],
        "cohort": inputs["cohort"],
        "upstream_commit": UPSTREAM_COMMIT,
        "gs_source": str(Path(inputs["gs_source"]).resolve()),
        "gs_source_sha256": _sha256(Path(inputs["gs_source"])),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "sam_checkpoint_sha256": _sha256(args.sam_checkpoint),
        "command": command,
    }
    if args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    environment, driver_library = _runtime_environment(args)
    manifest["driver_library_dir"] = str(args.driver_library_dir.resolve())
    manifest["driver_library"] = str(driver_library.resolve())
    manifest["driver_library_sha256"] = _sha256(driver_library)
    manifest["cuda_visible_devices"] = environment["CUDA_VISIBLE_DEVICES"]
    manifest["queued_at"] = _utc_now()
    queue_started_epoch = time.time()
    wall_started = time.monotonic()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log_path = run_dir / "stdout_stderr.log"
    gpu_started_marker = run_dir / "gpu_started_at.txt"
    locked_script = (
        "date -u +%Y-%m-%dT%H:%M:%S.%NZ"
        f" > {shlex.quote(str(gpu_started_marker))}; "
        f"exec {shlex.join(command)}"
    )
    locked_command = [
        "flock",
        str(LOCK_PATH),
        "-c",
        locked_script,
    ]
    try:
        with log_path.open("w") as log_handle:
            completed = subprocess.run(
                locked_command,
                cwd=args.upstream.resolve(),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["completed_at"] = _utc_now()
        manifest["wall_time_seconds"] = time.monotonic() - wall_started
        manifest["log"] = str(log_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise

    completed_epoch = time.time()
    manifest["status"] = "complete" if completed.returncode == 0 else "failed"
    manifest["returncode"] = completed.returncode
    manifest["completed_at"] = _utc_now()
    manifest["wall_time_seconds"] = time.monotonic() - wall_started
    if gpu_started_marker.exists():
        gpu_started_epoch = gpu_started_marker.stat().st_mtime
        manifest["gpu_started_at"] = gpu_started_marker.read_text(
            encoding="utf-8"
        ).strip()
        manifest["queue_wait_seconds"] = max(
            0.0, gpu_started_epoch - queue_started_epoch
        )
        manifest["gpu_wall_time_seconds"] = max(
            0.0, completed_epoch - gpu_started_epoch
        )
    manifest["log"] = str(log_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("nvos", "spin"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path)
    parser.add_argument(
        "--attempt-id",
        help=(
            "Optional immutable retry identifier. Existing attempt manifests "
            "are never overwritten."
        ),
    )
    parser.add_argument(
        "--stage-nvos-pinhole",
        action="store_true",
        help=(
            "Stage the dataset's existing dense/sparse undistorted PINHOLE "
            "model under this attempt; the downloaded dataset is not modified."
        ),
    )
    parser.add_argument(
        "--stage-spin-llff-pinhole-from",
        type=Path,
        help=(
            "Stage an existing PINHOLE conversion for SPIn-NeRF only after "
            "proving its raw sparse model and RGBs are byte-identical to the "
            "selected SPIn source scene."
        ),
    )
    parser.add_argument(
        "--driver-library-dir",
        type=Path,
        default=DEFAULT_DRIVER_LIBRARY_DIR,
        help=(
            "Directory containing the kernel-driver-compatible libcuda.so.1; "
            "prepended to LD_LIBRARY_PATH for the locked child only."
        ),
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("/root/baselines/VALA/ckpts/sam_vit_h_4b8939.pth"),
    )
    parser.add_argument("--gs-source", type=Path)
    parser.add_argument(
        "--geometry-protocol",
        choices=("released_all_view", "strict_geometry_hybrid_diagnostic"),
        default="released_all_view",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(launch(parse_args()))
    except ProtocolError as error:
        raise SystemExit(f"protocol error: {error}") from error
