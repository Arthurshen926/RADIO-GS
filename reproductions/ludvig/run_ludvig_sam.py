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

from PIL import Image, JpegImagePlugin, features
from PIL import __version__ as PILLOW_VERSION
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks"
)
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "protocol_audit_20260731" / "ludvig"
LOCK_PATH = Path("/tmp/radio-gs-gpu0.lock")
DEFAULT_DRIVER_LIBRARY_DIR = Path("/root/baselines/LUDVIG/.driver535")
UPSTREAM_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
OFFICIAL_3DGS_COMMIT = "f7a116fb1397d9842239127d39dc212f93171f70"
SAM_VIT_H_CHECKPOINT_SHA256 = (
    "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
)
UPSTREAM_PATCH = (
    ROOT
    / "reproductions"
    / "ludvig"
    / "patches"
    / "0001-reproduction-seeds-and-json-results.patch"
)
UPSTREAM_PATCH_SHA256 = (
    "2c21257316c6f65d25eea2bbd98481bd3e42f0d84df23a13c1bd1cb645e7d602"
)
UPSTREAM_PATCHED_FILE_SHA256 = {
    "evaluation/spin_nvos/base.py": (
        "7ecd469119b4ee87e3cc9cf5764426ec6c8a1d9118db072beab4d8335ea0d353"
    ),
    "evaluation/spin_nvos/segmentation.py": (
        "ecc9d7b1a29faa7d1091e14e9c5a50036180c5ba8671ba396d0292075a559b0e"
    ),
    "ludvig_uplift.py": (
        "ae2eb5af2050e619a8d3a6f5bb04d228b4c090425cfcaf30f25aa9a859cddd3e"
    ),
    "predictors/sam.py": (
        "3cbf8bda6f7334086c3ba7c117a1b604ed12757c351fff96054a6a2f484684b9"
    ),
    "utils/image.py": (
        "6047b23c26fcece6bc451961a532b302e6aeb0dcdce0c5c89a7e46f71eed87c1"
    ),
    "utils/solver.py": (
        "6b71c91c5e4dbe50b2995f6b9428c9cd9bd6940ab58b1f16fe788dfc50b1c70c"
    ),
}
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
NVOS_TASK_TO_GEOMETRY_SCENE = {
    "fern": "fern",
    "flower": "flower",
    "fortress": "fortress",
    "horns_center": "horns",
    "horns_left": "horns",
    "leaves": "leaves",
    "orchids": "orchids",
    "trex": "trex",
}
NVOS_GEOMETRY_REGISTERED_IMAGES = {
    "fern": 20,
    "flower": 34,
    "fortress": 42,
    "horns": 62,
    "leaves": 26,
    "orchids": 25,
    "trex": 55,
}
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
SPIN_GEOMETRY_REGISTERED_IMAGES = {
    "orchids": 25,
    "leaves": 26,
    "fern": 20,
    "room": 41,
    "horns": 62,
    "fortress": 42,
    # fork is intentionally absent: the benchmark asset is unavailable.
    "pinecone": 99,
    "truck": 251,
    "lego": 102,
}
SPIN_NVOS_SHARED_LLFF_GEOMETRIES = frozenset(
    {"fern", "fortress", "horns", "leaves", "orchids", "room"}
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
SPIN_ROOM_PINHOLE_IMAGE_SIZE = (4005, 3003)
SPIN_ROOM_RGB_PREPROCESSING = (
    "fractional_center_crop_bicubic_preserve_jpeg_tables_v1"
)
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
    physical_gpu = int(getattr(args, "physical_gpu", 0))
    if physical_gpu not in (0, 1):
        raise ProtocolError("--physical-gpu must be 0 or 1")
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["LD_LIBRARY_PATH"] = str(driver_library_dir) + os.pathsep + environment.get(
        "LD_LIBRARY_PATH", ""
    )
    if args.pythonpath:
        environment["PYTHONPATH"] = str(args.pythonpath.resolve()) + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
    return environment, driver_library


def _validate_pythonpath(pythonpath: Path | None) -> dict[str, str]:
    """Fail before GPU work when the audited rasterizer is not importable."""

    if pythonpath is None:
        raise ProtocolError(
            "--pythonpath is required and must contain the audited patched "
            "diff_gaussian_rasterization extension"
        )
    resolved = pythonpath.resolve()
    package = resolved / "diff_gaussian_rasterization"
    init = package / "__init__.py"
    extensions = sorted(package.glob("_C*.so"))
    if not init.is_file() or len(extensions) != 1:
        raise ProtocolError(
            "--pythonpath must contain exactly one compiled audited "
            "diff_gaussian_rasterization extension"
        )
    return {
        "root": str(resolved),
        "package_init_sha256": _sha256(init),
        "extension": str(extensions[0].resolve()),
        "extension_sha256": _sha256(extensions[0]),
    }


def _upstream_patch_provenance(checkout: Path) -> dict[str, Any]:
    if not UPSTREAM_PATCH.is_file() or _sha256(UPSTREAM_PATCH) != UPSTREAM_PATCH_SHA256:
        raise ProtocolError(
            f"Audited LUDVIG reproduction patch changed: {UPSTREAM_PATCH}"
        )
    patched_paths = sorted(UPSTREAM_PATCHED_FILE_SHA256)
    staged = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    modified = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if staged or sorted(modified) != patched_paths:
        raise ProtocolError(
            "LUDVIG checkout tracked changes do not exactly match the audited "
            f"reproduction patch: staged={staged}, modified={modified}"
        )
    diff = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--", *patched_paths],
        check=True,
        capture_output=True,
    ).stdout
    diff_hash = hashlib.sha256(diff).hexdigest()
    if diff_hash != UPSTREAM_PATCH_SHA256:
        raise ProtocolError(
            "LUDVIG checkout diff does not byte-match the audited "
            f"reproduction patch: {diff_hash}"
        )
    file_hashes = {
        relative: _sha256(checkout / relative) for relative in patched_paths
    }
    if file_hashes != UPSTREAM_PATCHED_FILE_SHA256:
        raise ProtocolError(
            "LUDVIG patched evaluator/entrypoint hashes changed: "
            f"{file_hashes}"
        )
    return {
        "patch": str(UPSTREAM_PATCH),
        "patch_sha256": UPSTREAM_PATCH_SHA256,
        "tracked_diff_sha256": diff_hash,
        "patched_file_sha256": file_hashes,
        "staged_tracked_changes": False,
        "other_tracked_changes": False,
    }


def _validate_upstream(checkout: Path) -> dict[str, Any]:
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
    return _upstream_patch_provenance(checkout)


def _validate_sam_checkpoint(checkpoint: Path) -> str:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ProtocolError(f"Missing SAM checkpoint: {checkpoint}")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != SAM_VIT_H_CHECKPOINT_SHA256:
        raise ProtocolError(
            "LUDVIG-SAM reproduction requires the audited ViT-H SAM "
            f"checkpoint SHA256 {SAM_VIT_H_CHECKPOINT_SHA256}; found "
            f"{checkpoint_hash}"
        )
    return checkpoint_hash


def _validate_released_all_view_training_manifest(
    manifest_path: Path,
    gs_source: Path,
    geometry_scene: str,
) -> dict[str, Any]:
    """Bind a released-all-view label to a verified original-3DGS run."""

    manifest_path = manifest_path.resolve()
    gs_source = gs_source.resolve()
    if not manifest_path.is_file():
        raise ProtocolError(
            "released_all_view requires an existing --gs-training-manifest"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ProtocolError(
            "released_all_view training manifest must have status=complete"
        )
    if (
        payload.get("method") != "original-3DGS"
        or payload.get("geometry_protocol") != "released_all_view"
    ):
        raise ProtocolError(
            "training manifest is not a released-all-view original-3DGS run"
        )
    training_benchmark = payload.get("benchmark")
    if training_benchmark == "NVOS":
        expected_registered_views = NVOS_GEOMETRY_REGISTERED_IMAGES.get(
            geometry_scene
        )
    elif training_benchmark == "SPIn-NeRF":
        expected_registered_views = SPIN_GEOMETRY_REGISTERED_IMAGES.get(
            geometry_scene
        )
    else:
        raise ProtocolError(
            f"training manifest has unsupported benchmark: {training_benchmark}"
        )
    if expected_registered_views is None:
        raise ProtocolError(
            "training manifest geometry is not in the frozen benchmark asset "
            f"contract: {training_benchmark}/{geometry_scene}"
        )
    declared_geometry_scene = payload.get("geometry_scene")
    legacy_geometry_scene_fallback = False
    if declared_geometry_scene is None:
        if (
            payload.get("benchmark") == "NVOS"
            and payload.get("scene") == "fern"
            and geometry_scene == "fern"
        ):
            # One fern training predated the explicit geometry_scene field.
            declared_geometry_scene = "fern"
            legacy_geometry_scene_fallback = True
        else:
            raise ProtocolError(
                "training manifest must explicitly record geometry_scene; "
                "the legacy fallback is restricted to the existing NVOS fern run"
            )
    if (
        payload.get("scene") != geometry_scene
        or declared_geometry_scene != geometry_scene
    ):
        raise ProtocolError(
            "training manifest geometry scene "
            f"{declared_geometry_scene} does not match {geometry_scene}"
        )

    source = payload.get("source_provenance", {})
    if source.get("commit") != OFFICIAL_3DGS_COMMIT:
        raise ProtocolError(
            "training manifest does not pin the audited original-3DGS commit"
        )
    if source.get("ludvig_commit") != UPSTREAM_COMMIT:
        raise ProtocolError(
            "training manifest does not bind the audited LUDVIG source"
        )

    protocol = payload.get("effective_training_protocol", {})
    required_protocol = {
        "held_out_training_views": 0,
        "eval_split_enabled": False,
        "iterations": 30000,
        "resolution_argument": -1,
        "algorithm_source_modified": False,
    }
    for key, expected in required_protocol.items():
        actual = protocol.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ProtocolError(
                f"training manifest has incompatible {key}: "
                f"expected {expected}, found {actual}"
            )
    registered_views = protocol.get("registered_training_views")
    if (
        type(registered_views) is not int
        or registered_views != expected_registered_views
    ):
        raise ProtocolError(
            "training manifest camera count does not match the frozen asset "
            f"contract: expected {expected_registered_views}, found "
            f"{registered_views}"
        )

    output = payload.get("training_output", {})
    output_point_cloud = Path(str(output.get("point_cloud", ""))).resolve()
    if output_point_cloud != gs_source:
        raise ProtocolError(
            "training manifest point_cloud does not match --gs-source"
        )
    if not gs_source.is_file():
        raise ProtocolError(f"Missing released-all-view point cloud: {gs_source}")
    checkpoint_sha256 = _sha256(gs_source)
    if output.get("point_cloud_sha256") != checkpoint_sha256:
        raise ProtocolError(
            "training manifest point-cloud hash does not match --gs-source"
        )
    cfg_args = output.get("cfg_args", {})
    if (
        cfg_args.get("eval") is not False
        or type(cfg_args.get("resolution")) is not int
        or cfg_args.get("resolution") != -1
    ):
        raise ProtocolError(
            "training output cfg_args must record eval=false and resolution=-1"
        )
    if (
        type(output.get("registered_all_view_cameras")) is not int
        or output.get("registered_all_view_cameras") != registered_views
    ):
        raise ProtocolError(
            "training output camera count does not match the effective protocol"
        )
    if output.get("target_rgb_visible_during_training") is not True:
        raise ProtocolError(
            "released-all-view training output must declare target RGB visible"
        )

    return {
        "verified": True,
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": _sha256(manifest_path),
        "training_status": payload["status"],
        "method": payload["method"],
        "benchmark": training_benchmark,
        "scene": payload["scene"],
        "geometry_scene": declared_geometry_scene,
        "legacy_geometry_scene_fallback": legacy_geometry_scene_fallback,
        "official_3dgs_commit": source["commit"],
        "ludvig_commit": source["ludvig_commit"],
        "iterations": protocol["iterations"],
        "registered_training_views": registered_views,
        "held_out_training_views": protocol["held_out_training_views"],
        "eval_split_enabled": protocol["eval_split_enabled"],
        "resolution_argument": protocol["resolution_argument"],
        "algorithm_source_modified": protocol["algorithm_source_modified"],
        "point_cloud": str(gs_source),
        "point_cloud_sha256": checkpoint_sha256,
    }


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


def _stage_spin_room_exact_rgb(
    source_images_dir: Path,
    staging_images_dir: Path,
    camera: dict[str, Any],
    intrinsics_audit: dict[str, Any],
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    """Materialize the exact 1600x1200 RGB geometry used by room masks.

    LUDVIG computes the room FOV as a centered 4:3 crop of the released
    4005x3003 PINHOLE camera, but its independent max-axis RGB resize floors
    the image to 1600x1199.  Make the FOV crop explicit before evaluation so
    RGB, camera, prompt, and mask tensors all share the official 1600x1200
    contract.  The half-pixel crop on each horizontal edge preserves the
    centered principal point exactly.
    """

    source_size = (camera["width"], camera["height"])
    target_size = (target_width, target_height)
    if source_size != SPIN_ROOM_PINHOLE_IMAGE_SIZE:
        raise ProtocolError(
            "SPIn room RGB preprocessing requires the audited 4005x3003 "
            f"PINHOLE source; found {source_size}"
        )
    if target_size != SPIN_IMAGE_SIZE["room"]:
        raise ProtocolError(
            "SPIn room RGB preprocessing requires the official 1600x1200 "
            f"target; found {target_size}"
        )

    crop_width = float(intrinsics_audit["center_crop_width"])
    crop_height = float(intrinsics_audit["center_crop_height"])
    if not (
        math.isclose(crop_width, 4004.0, abs_tol=1e-9)
        and math.isclose(crop_height, 3003.0, abs_tol=1e-9)
    ):
        raise ProtocolError(
            "SPIn room intrinsics no longer imply the audited 4004x3003 "
            f"center crop: found {crop_width}x{crop_height}"
        )
    crop_left = (source_size[0] - crop_width) / 2.0
    crop_top = (source_size[1] - crop_height) / 2.0
    crop_box = (
        crop_left,
        crop_top,
        crop_left + crop_width,
        crop_top + crop_height,
    )
    if crop_box != (0.5, 0.0, 4004.5, 3003.0):
        raise ProtocolError(f"Unexpected SPIn room center-crop box: {crop_box}")

    source_paths = sorted(
        path
        for path in source_images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    source_hashes = {path.name: _sha256(path) for path in source_paths}
    if staging_images_dir.exists() or staging_images_dir.is_symlink():
        raise ProtocolError(
            "Refusing to replace existing SPIn room RGB staging path: "
            f"{staging_images_dir}"
        )
    if staging_images_dir.resolve() == source_images_dir.resolve():
        raise ProtocolError("SPIn room RGB staging cannot overwrite its source")
    staging_images_dir.mkdir(parents=True)

    jpeg_sampling: dict[str, int] = {}
    jpeg_quantization_sha256: dict[str, str] = {}
    for source_path in source_paths:
        with Image.open(source_path) as source_image:
            if source_image.size != source_size:
                raise ProtocolError(
                    f"SPIn room RGB {source_path.name} changed dimensions: "
                    f"expected {source_size}, found {source_image.size}"
                )
            if source_image.format != "JPEG" or source_image.mode != "RGB":
                raise ProtocolError(
                    "Audited SPIn room preprocessing requires RGB JPEG inputs; "
                    f"found {source_image.format}/{source_image.mode} for "
                    f"{source_path.name}"
                )
            quantization = getattr(source_image, "quantization", None)
            if not quantization:
                raise ProtocolError(
                    f"Missing JPEG quantization tables for {source_path}"
                )
            sampling = JpegImagePlugin.get_sampling(source_image)
            if sampling not in {0, 1, 2}:
                raise ProtocolError(
                    f"Unsupported JPEG sampling {sampling} for {source_path}"
                )
            qtables = {
                int(index): list(table)
                for index, table in quantization.items()
            }
            jpeg_sampling[source_path.name] = sampling
            jpeg_quantization_sha256[source_path.name] = hashlib.sha256(
                json.dumps(
                    qtables,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            resized = source_image.resize(
                target_size,
                resample=Image.Resampling.BICUBIC,
                box=crop_box,
            )
            resized.save(
                staging_images_dir / source_path.name,
                format="JPEG",
                quality=-1,
                subsampling=sampling,
                qtables=qtables,
                optimize=False,
                progressive=False,
            )

    staged_paths = sorted(
        path
        for path in staging_images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if {path.name for path in staged_paths} != set(source_hashes):
        raise ProtocolError("SPIn room RGB preprocessing changed image names")
    for staged_path in staged_paths:
        with Image.open(staged_path) as staged_image:
            if staged_image.size != target_size or staged_image.format != "JPEG":
                raise ProtocolError(
                    f"Invalid staged SPIn room RGB {staged_path}: "
                    f"{staged_image.size}/{staged_image.format}"
                )
    if {path.name: _sha256(path) for path in source_paths} != source_hashes:
        raise ProtocolError("SPIn room source RGBs changed during staging")

    return {
        "strategy": SPIN_ROOM_RGB_PREPROCESSING,
        "scene": "room",
        "source_rgb_directory": str(source_images_dir.resolve()),
        "staged_rgb_directory": str(staging_images_dir.resolve()),
        "source_rgb_dimensions": list(source_size),
        "center_crop_dimensions": [crop_width, crop_height],
        "center_crop_box": list(crop_box),
        "center_crop_coordinate_policy": "fractional_geometric_center",
        "target_rgb_dimensions": list(target_size),
        "resize_implementation": "Pillow.Image.resize",
        "resize_resampling": "BICUBIC",
        "pillow_version": PILLOW_VERSION,
        "libjpeg_version": features.version_codec("jpg"),
        "output_encoding": "JPEG",
        "output_quality": -1,
        "output_optimize": False,
        "output_progressive": False,
        "jpeg_sampling_by_image": jpeg_sampling,
        "jpeg_quantization_sha256_by_image": jpeg_quantization_sha256,
        "source_rgb_sha256": source_hashes,
        "staged_rgb_sha256": {
            path.name: _sha256(path) for path in staged_paths
        },
        "source_dataset_modified": False,
    }


def _stage_nvos_pinhole_colmap(
    source_scene: Path,
    staging_scene: Path,
    target_width: int,
    target_height: int,
    rgb_preprocessing: str | None = None,
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

    rgb_preprocessing_audit = None
    if rgb_preprocessing is None:
        _ensure_link(staging_scene / "images", images_dir)
    elif rgb_preprocessing == SPIN_ROOM_RGB_PREPROCESSING:
        rgb_preprocessing_audit = _stage_spin_room_exact_rgb(
            images_dir,
            staging_scene / "images",
            camera,
            intrinsics_audit,
            target_width,
            target_height,
        )
    else:
        raise ProtocolError(
            f"Unknown RGB preprocessing strategy: {rgb_preprocessing}"
        )
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        _ensure_link(
            staging_scene / "sparse" / "0" / name,
            converted_sparse / name,
        )
    audit = {
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
    if rgb_preprocessing_audit is not None:
        audit["source_rgb_dimensions"] = list(expected_size)
        audit["rgb_dimensions"] = [target_width, target_height]
        audit["rgb_preprocessing"] = rgb_preprocessing_audit
    return audit


def _stage_spin_truck_pinhole_colmap(
    source_scene: Path,
    staging_scene: Path,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    """Stage the official Graphdeco Truck input without changing calibration.

    ``tandt_db.zip`` stores 979x546 RGBs alongside the centered 1957x1091
    PINHOLE calibration from which they were downsampled.  Original 3DGS uses
    the RGB dimensions for its tensors and the COLMAP focal ratios for the
    field of view.  Preserve that released representation exactly instead of
    rewriting its intrinsics or resampling the images again.
    """

    expected_rgb_size = SPIN_IMAGE_SIZE["truck"]
    expected_camera_size = (1957, 1091)
    if (target_width, target_height) != expected_rgb_size:
        raise ProtocolError(
            "Truck staging requires the frozen 979x546 evaluation size; "
            f"found {target_width}x{target_height}"
        )

    source_sparse = source_scene / "sparse" / "0"
    camera_records = _read_colmap_cameras_binary(
        source_sparse / "cameras.bin"
    )
    if len(camera_records) != 1:
        raise ProtocolError(
            "Audited Truck staging expects exactly one shared COLMAP camera; "
            f"found {len(camera_records)}"
        )
    camera = camera_records[0]
    if camera["model"] != "PINHOLE":
        raise ProtocolError(
            "Audited Truck staging requires the native PINHOLE model; "
            f"found {camera['model']}"
        )
    camera_size = (camera["width"], camera["height"])
    if camera_size != expected_camera_size:
        raise ProtocolError(
            "Truck COLMAP metadata dimensions changed: expected "
            f"{expected_camera_size}, found {camera_size}"
        )
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
    if image_sizes != {expected_rgb_size}:
        raise ProtocolError(
            "Truck RGB dimensions changed: expected "
            f"{expected_rgb_size}, found {sorted(image_sizes)}"
        )

    poses = _read_colmap_image_poses(source_sparse / "images.bin")
    rgb_names = {path.name for path in image_paths}
    if rgb_names != set(poses):
        raise ProtocolError(
            "Staged Truck RGB/COLMAP image names differ: "
            f"missing_rgb={sorted(set(poses) - rgb_names)}, "
            f"unregistered_rgb={sorted(rgb_names - set(poses))}"
        )
    camera_ids = {pose["camera_id"] for pose in poses.values()}
    if camera_ids != {camera["camera_id"]}:
        raise ProtocolError(
            "Truck registered images do not all use the audited shared camera"
        )

    _ensure_link(staging_scene / "images", images_dir)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        _ensure_link(
            staging_scene / "sparse" / "0" / name,
            source_sparse / name,
        )
    return {
        "strategy": "stage_native_graphdeco_truck_pinhole",
        "source_scene": str(source_scene.resolve()),
        "source_sparse": str(source_sparse.resolve()),
        "staged_scene": str(staging_scene.resolve()),
        "camera_model": camera["model"],
        "camera_metadata_dimensions": list(expected_camera_size),
        "registered_images": len(poses),
        "rgb_images": len(image_paths),
        "rgb_dimensions": list(expected_rgb_size),
        "released_rgb_downsampling": "ceil_half_from_colmap_metadata",
        "intrinsics": intrinsics_audit,
        "source_sha256": {
            name: _sha256(source_sparse / name)
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "rgb_sha256": _image_hashes(images_dir),
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
    rgb_preprocessing: str | None = None,
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
        rgb_preprocessing=rgb_preprocessing,
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
        if getattr(args, "stage_spin_truck_pinhole", False):
            raise ProtocolError(
                "--stage-spin-truck-pinhole is only valid for SPIn-NeRF"
            )
        if args.scene not in NVOS_SCENES:
            raise ProtocolError(f"Unknown NVOS task: {args.scene}")
        source_scene = NVOS_TASK_TO_GEOMETRY_SCENE[args.scene]
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
            "geometry_scene": source_scene,
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
    stage_truck = getattr(args, "stage_spin_truck_pinhole", False)
    stage_lego = getattr(args, "stage_spin_lego_pinhole", False)
    if sum((converted_source is not None, stage_truck, stage_lego)) > 1:
        raise ProtocolError(
            "Choose exactly one SPIn COLMAP staging strategy"
        )
    if stage_truck:
        if args.scene != "truck":
            raise ProtocolError(
                "--stage-spin-truck-pinhole is restricted to the frozen "
                "Graphdeco Truck asset"
            )
        staged_colmap_dir = (
            run_dir / "staging" / "colmap_native_truck_pinhole"
        )
        colmap_staging = _stage_spin_truck_pinhole_colmap(
            colmap_dir,
            staged_colmap_dir,
            width,
            height,
        )
        if colmap_staging.get("registered_images") != (
            SPIN_GEOMETRY_REGISTERED_IMAGES["truck"]
        ):
            raise ProtocolError(
                "Truck staging does not contain the frozen 251-view cohort"
            )
        colmap_dir = staged_colmap_dir
    elif stage_lego:
        if args.scene != "lego":
            raise ProtocolError(
                "--stage-spin-lego-pinhole is restricted to the frozen "
                "Lego official undistortion"
            )
        # Local import avoids a module cycle: the Lego staging helper reuses
        # the frozen COLMAP readers defined above in this wrapper.
        from reproductions.ludvig.stage_spin_lego_official_undistortion import (
            _stage_spin_lego_pinhole_colmap,
        )

        staged_colmap_dir = run_dir / "staging" / "colmap_native_lego_pinhole"
        colmap_staging = _stage_spin_lego_pinhole_colmap(
            colmap_dir,
            annotation,
            staged_colmap_dir,
            width,
            height,
        )
        if colmap_staging.get("registered_images") != (
            SPIN_GEOMETRY_REGISTERED_IMAGES["lego"]
        ):
            raise ProtocolError(
                "Lego staging does not contain the frozen 102-view cohort"
            )
        colmap_dir = staged_colmap_dir
    elif converted_source is not None:
        staged_colmap_dir = (
            run_dir / "staging" / "colmap_pinhole_undistorted"
        )
        colmap_staging = _stage_spin_llff_pinhole_colmap(
            colmap_dir,
            Path(converted_source).resolve(),
            staged_colmap_dir,
            width,
            height,
            rgb_preprocessing=(
                SPIN_ROOM_RGB_PREPROCESSING
                if args.scene == "room"
                else None
            ),
        )
        if args.scene == "room":
            pinhole = colmap_staging.get("pinhole", {})
            if pinhole.get("registered_images") != (
                SPIN_GEOMETRY_REGISTERED_IMAGES["room"]
            ):
                raise ProtocolError(
                    "SPIn room staging does not contain the frozen 41-view cohort"
                )
            if pinhole.get("rgb_dimensions") != [width, height]:
                raise ProtocolError(
                    "SPIn room staging did not materialize exact 1600x1200 RGBs"
                )
        colmap_dir = staged_colmap_dir
    else:
        if args.scene == "truck":
            raise ProtocolError(
                "Truck requires --stage-spin-truck-pinhole so its native "
                "half-resolution PINHOLE contract is audited"
            )
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
        "geometry_scene": args.scene,
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
    config = {
        "tag": "sam",
        "dst_dir": str(run_dir),
        "feature": feature,
    }
    if not bool(getattr(args, "materialize_only", False)):
        config["evaluation"] = evaluation
    if not bool(getattr(args, "retain_full_carrier", False)):
        config["prune_gaussians"] = 0.5
    return config


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
    upstream_patch_provenance = _validate_upstream(args.upstream.resolve())
    pythonpath_provenance = _validate_pythonpath(args.pythonpath)
    inputs = _resolve_inputs(args, run_dir)
    for name in ("colmap_dir", "gs_source"):
        if not Path(inputs[name]).exists():
            raise ProtocolError(f"Missing {name}: {inputs[name]}")
    sam_checkpoint_sha256 = _validate_sam_checkpoint(args.sam_checkpoint)
    released_training_provenance = None
    if args.geometry_protocol == "released_all_view":
        if args.gs_training_manifest is None:
            raise ProtocolError(
                "--geometry-protocol released_all_view requires "
                "--gs-training-manifest"
            )
        released_training_provenance = (
            _validate_released_all_view_training_manifest(
                args.gs_training_manifest,
                Path(inputs["gs_source"]),
                inputs["geometry_scene"],
            )
        )
        expected_training_benchmark = (
            "NVOS" if args.benchmark == "nvos" else "SPIn-NeRF"
        )
        if (
            released_training_provenance["benchmark"]
            != expected_training_benchmark
        ):
            staging = inputs.get("colmap_staging")
            verified_spin_nvos_reuse = (
                args.benchmark == "spin"
                and released_training_provenance["benchmark"] == "NVOS"
                and inputs["geometry_scene"] in SPIN_NVOS_SHARED_LLFF_GEOMETRIES
                and isinstance(staging, dict)
                and staging.get("strategy")
                == "reuse_verified_identical_llff_colmap_undistortion"
                and staging.get("raw_scene_identity_proven") is True
            )
            if not verified_spin_nvos_reuse:
                raise ProtocolError(
                    "training manifest benchmark does not match the requested "
                    f"evaluation: expected {expected_training_benchmark}, found "
                    f"{released_training_provenance['benchmark']}"
                )
            staging_sha256 = hashlib.sha256(
                json.dumps(
                    staging,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            released_training_provenance["cross_benchmark_asset_reuse"] = {
                "verified": True,
                "training_benchmark": "NVOS",
                "evaluation_benchmark": "SPIn-NeRF",
                "geometry_scene": inputs["geometry_scene"],
                "colmap_staging_sha256": staging_sha256,
            }
        else:
            released_training_provenance["cross_benchmark_asset_reuse"] = None
    elif args.gs_training_manifest is not None:
        raise ProtocolError(
            "--gs-training-manifest is only valid with released_all_view"
        )

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
        "geometry_scene": inputs["geometry_scene"],
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
        "materialize_only": bool(getattr(args, "materialize_only", False)),
        "target_masks_opened_by_materializer": False
        if bool(getattr(args, "materialize_only", False))
        else None,
        "retain_full_carrier": bool(
            getattr(args, "retain_full_carrier", False)
        ),
        "reference_mask_calibration": inputs["reference_mask_calibration"],
        "target_masks_scoring_only": True,
        "aggregation": inputs["aggregation"],
        "cohort": inputs["cohort"],
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_patch_provenance": upstream_patch_provenance,
        "pythonpath_provenance": pythonpath_provenance,
        "gs_source": str(Path(inputs["gs_source"]).resolve()),
        "gs_source_sha256": _sha256(Path(inputs["gs_source"])),
        "released_all_view_training_provenance": released_training_provenance,
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "sam_checkpoint_sha256": sam_checkpoint_sha256,
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
    physical_gpu = int(getattr(args, "physical_gpu", 0))
    lock_path = Path(f"/tmp/radio-gs-gpu{physical_gpu}.lock")
    locked_command = [
        "flock",
        str(lock_path),
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
    parser.add_argument(
        "--physical-gpu",
        type=int,
        choices=(0, 1),
        default=0,
        help="Physical GPU exposed as cuda:0 inside the audited child.",
    )
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
        "--stage-spin-truck-pinhole",
        action="store_true",
        help=(
            "Stage and audit the native 251-view Graphdeco Truck PINHOLE "
            "asset, including its frozen half-resolution RGB contract."
        ),
    )
    parser.add_argument(
        "--stage-spin-lego-pinhole",
        action="store_true",
        help=(
            "Stage and audit Lego's official 102-view PINHOLE undistortion, "
            "including the exact 0_/1_ split-prefix mapping."
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
        "--retain-full-carrier",
        action="store_true",
        help=(
            "Do not prune low-visibility Gaussians after uplifting. This is "
            "required when the query field will be rendered on the same frozen "
            "full carrier, so every carrier row receives an explicit hypothesis."
        ),
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help=(
            "Persist the uplifted query field and carrier without running the "
            "upstream evaluator. This keeps target masks sealed for a separate "
            "receipt-first frozen-renderer evaluation."
        ),
    )
    parser.add_argument(
        "--gs-training-manifest",
        type=Path,
        help=(
            "Completed original-3DGS training manifest whose validated 30k "
            "point cloud is passed via --gs-source. Required for "
            "released_all_view and forbidden for hybrid geometry."
        ),
    )
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
