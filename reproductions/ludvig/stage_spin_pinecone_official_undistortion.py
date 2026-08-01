#!/usr/bin/env python3
"""Materialize and audit Pinecone's pinned COLMAP undistortion once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from reproductions.ludvig.run_ludvig_sam import (
    ProtocolError,
    _image_hashes,
    _read_colmap_cameras_binary,
    _read_colmap_image_poses,
    _sha256,
    _stage_spin_llff_pinhole_colmap,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "output"
    / "protocol_audit_20260731"
    / "ludvig"
    / "spin"
    / "released_all_view"
    / "pinecone"
    / "undistortion"
    / "attempts"
)
RAW_RELATIVE = (
    Path("SPIn-NeRF")
    / "source_images"
    / "nerf_real_360"
    / "extracted"
    / "pinecone"
)
ANNOTATION_RELATIVE = (
    Path("SPIn-NeRF")
    / "multiview_annotations"
    / "pinecone (nerf_real_360)"
)
ADAPTER_RELATIVE = (
    Path("SPIn-NeRF")
    / "protocol_derived"
    / "pinecone_colmap_3p6_undistorted_v2"
)
SPARSE_NAMES = ("cameras.bin", "images.bin", "points3D.bin")
EXPECTED_RAW_SPARSE_SHA256 = {
    "cameras.bin": "173d3047620ad45fe1f453ecf1fd2a28a22b0d19f72062707311479d255c6d64",
    "images.bin": "d175b790d2b99929dcc79b9f751d830824a767291fb1896f833e05a198a7dd17",
    "points3D.bin": "4505cdacaa7f274588440ed768018a7bf3ee317226c3f2e39ea13675192131dc",
}


class PineconeStagingError(RuntimeError):
    """Raised before accepting a derived Pinecone asset."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_link(link: Path, target: Path, *, directory: bool = False) -> None:
    target = target.resolve()
    if not target.exists():
        raise PineconeStagingError(f"Missing link target: {target}")
    if link.is_symlink():
        if link.resolve() != target:
            raise PineconeStagingError(f"Existing link has wrong target: {link}")
        return
    if link.exists():
        raise PineconeStagingError(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=directory)


def _directory_tree_hash(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain_annotation_hashes(directory: Path) -> dict[str, str]:
    masks = sorted(
        path
        for path in directory.glob("*.png")
        if not path.stem.endswith(("_cutout", "_pseudo"))
    )
    if len(masks) != 99:
        raise PineconeStagingError(f"Expected 99 plain masks, found {len(masks)}")
    for path in masks:
        with Image.open(path) as image:
            values = set(image.getdata())
        if values - {0, 1}:
            raise PineconeStagingError(
                f"Pinecone mask is not binary 0/1: {path.name} -> {values}"
            )
    return {path.name: _sha256(path) for path in masks}


def _colmap_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()[0].strip()


def stage(args: argparse.Namespace) -> Path:
    benchmark_root = args.benchmark_root.resolve()
    raw_scene = benchmark_root / RAW_RELATIVE
    annotation_scene = benchmark_root / ANNOTATION_RELATIVE
    raw_sparse = raw_scene / "sparse" / "0"
    raw_images = raw_scene / "images"
    adapter_link = benchmark_root / ADAPTER_RELATIVE
    attempt = args.output_root.resolve() / args.attempt_id
    manifest_path = attempt / "undistortion_manifest.json"
    if attempt.exists():
        raise PineconeStagingError(f"Immutable attempt already exists: {attempt}")

    colmap = args.colmap.resolve()
    if not colmap.is_file() or not os.access(colmap, os.X_OK):
        raise PineconeStagingError(f"COLMAP executable is unavailable: {colmap}")
    raw_sparse_hashes = {name: _sha256(raw_sparse / name) for name in SPARSE_NAMES}
    if raw_sparse_hashes != EXPECTED_RAW_SPARSE_SHA256:
        raise PineconeStagingError("Pinecone raw sparse hashes changed")
    raw_rgb_hashes = _image_hashes(raw_images)
    if len(raw_rgb_hashes) != 99:
        raise PineconeStagingError(f"Expected 99 raw RGBs, found {len(raw_rgb_hashes)}")
    annotation_hashes = _plain_annotation_hashes(annotation_scene)
    if {Path(name).stem for name in raw_rgb_hashes} != {
        Path(name).stem for name in annotation_hashes
    }:
        raise PineconeStagingError("Raw RGB and annotation stems are not bijective")

    raw_cameras = _read_colmap_cameras_binary(raw_sparse / "cameras.bin")
    raw_poses = _read_colmap_image_poses(raw_sparse / "images.bin")
    if (
        len(raw_cameras) != 1
        or raw_cameras[0]["model"] != "SIMPLE_RADIAL"
        or [raw_cameras[0]["width"], raw_cameras[0]["height"]] != [4032, 3024]
        or len(raw_poses) != 99
    ):
        raise PineconeStagingError("Pinecone raw camera contract changed")

    attempt.mkdir(parents=True)
    log_path = attempt / "colmap_stdout_stderr.log"
    command = [
        str(colmap),
        "image_undistorter",
        "--image_path",
        str(raw_images),
        "--input_path",
        str(raw_sparse),
        "--output_path",
        str(attempt),
        "--output_type",
        "COLMAP",
        "--blank_pixels",
        "0",
        "--min_scale",
        "0.2",
        "--max_scale",
        "2",
        "--max_image_size",
        "-1",
    ]
    started_at = _utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise PineconeStagingError(
            f"COLMAP image_undistorter failed with {completed.returncode}"
        )

    generated_sparse = attempt / "sparse"
    for name in SPARSE_NAMES:
        if not (generated_sparse / name).is_file():
            raise PineconeStagingError(f"COLMAP omitted generated {name}")
        _safe_link(generated_sparse / "0" / name, raw_sparse / name)
    _safe_link(attempt / "images_distort", raw_images, directory=True)
    _safe_link(attempt / "dense" / "sparse", generated_sparse, directory=True)

    try:
        equivalence = _stage_spin_llff_pinhole_colmap(
            raw_scene,
            attempt,
            attempt / "verified_training_input",
            1600,
            1199,
        )
    except ProtocolError as error:
        raise PineconeStagingError(str(error)) from error
    pinhole = equivalence["pinhole"]
    if (
        pinhole["camera_model"] not in {"PINHOLE", "SIMPLE_PINHOLE"}
        or pinhole["registered_images"] != 99
        or pinhole["rgb_images"] != 99
        or pinhole["rgb_dimensions"] != [4015, 3011]
        or pinhole["max_qvec_delta_vs_original_sparse"] > 1e-12
        or pinhole["max_tvec_delta_vs_original_sparse"] > 1e-12
    ):
        raise PineconeStagingError("Generated Pinecone PINHOLE audit changed")

    effective_size = [
        int(4015 / (4015 / 1600)),
        int(3011 / (4015 / 1600)),
    ]
    if effective_size != [1600, 1199]:
        raise PineconeStagingError(f"Unexpected 3DGS resolution: {effective_size}")
    if {name: _sha256(raw_sparse / name) for name in SPARSE_NAMES} != raw_sparse_hashes:
        raise PineconeStagingError("Raw sparse files changed during undistortion")
    if _image_hashes(raw_images) != raw_rgb_hashes:
        raise PineconeStagingError("Raw RGB files changed during undistortion")

    _safe_link(adapter_link, attempt, directory=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "SPIn-NeRF",
        "scene": "pinecone",
        "attempt_id": args.attempt_id,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "command": command,
        "colmap": {
            "version": _colmap_version(colmap),
            "executable": str(colmap),
            "executable_sha256": _sha256(colmap),
        },
        "raw": {
            "scene": str(raw_scene.resolve()),
            "camera_model": "SIMPLE_RADIAL",
            "camera_dimensions": [4032, 3024],
            "registered_images": 99,
            "sparse_sha256": raw_sparse_hashes,
            "rgb_tree_sha256": _directory_tree_hash(raw_rgb_hashes),
            "source_dataset_modified": False,
        },
        "annotations": {
            "scene": str(annotation_scene.resolve()),
            "plain_masks": 99,
            "reference": "IMG_7238",
            "scored_targets": 98,
            "rgb_stem_bijection": True,
            "plain_mask_tree_sha256": _directory_tree_hash(annotation_hashes),
        },
        "undistorted": {
            "scene": str(attempt),
            "adapter_link": str(adapter_link),
            "camera_model": pinhole["camera_model"],
            "camera_dimensions": [4015, 3011],
            "registered_images": 99,
            "effective_original_3dgs_resolution": effective_size,
            "max_qvec_delta_vs_raw": pinhole[
                "max_qvec_delta_vs_original_sparse"
            ],
            "max_tvec_delta_vs_raw": pinhole[
                "max_tvec_delta_vs_original_sparse"
            ],
            "sparse_sha256": pinhole["source_sha256"],
            "rgb_tree_sha256": _directory_tree_hash(
                _image_hashes(attempt / "images")
            ),
        },
        "equivalence_audit": equivalence,
        "log": str(log_path),
        "log_sha256": _sha256(log_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--colmap", type=Path, default=Path("/usr/bin/colmap"))
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(stage(parse_args()))
    except (PineconeStagingError, ProtocolError) as error:
        raise SystemExit(f"protocol error: {error}") from error
