#!/usr/bin/env python3
"""Recover the audited Truck 30k manifest after the legacy size assertion.

This is deliberately a postflight-only tool.  It never invokes training or
CUDA, accepts only the frozen Truck contract, and writes only the existing
training manifest when ``--apply`` is explicit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from reproductions.ludvig.train_nvos_all_view_3dgs import (
    NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT,
    TrainingProtocolError,
    _sha256,
    _validate_training_output,
)


EXPECTED_RGB_RESOLUTION = (979, 546)
EXPECTED_CAMERA_METADATA_RESOLUTION = (1957, 1091)
EXPECTED_REGISTERED_IMAGES = 251
LEGACY_VALIDATION_ERROR = (
    "Training cameras do not match the staged source resolution: "
    "expected (979, 546), found [(1957, 1091)]"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PostflightRecoveryError(RuntimeError):
    """Raised without writing when the failed attempt is not the exact target."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostflightRecoveryError(message)


def _require_sha256(value: str, label: str) -> None:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA256 digest",
    )


def _validated_recovery_manifest(
    run_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_point_cloud_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Return a complete candidate manifest without mutating the attempt."""

    _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
    _require_sha256(
        expected_point_cloud_sha256,
        "expected_point_cloud_sha256",
    )
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "training_manifest.json"
    _require(manifest_path.is_file(), f"Missing training manifest: {manifest_path}")
    found_manifest_sha256 = _sha256(manifest_path)
    _require(
        found_manifest_sha256 == expected_manifest_sha256,
        "Training manifest changed before recovery: expected "
        f"{expected_manifest_sha256}, found {found_manifest_sha256}",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostflightRecoveryError(
            f"Cannot parse training manifest: {manifest_path}"
        ) from error
    _require(isinstance(manifest, dict), "Training manifest must be an object")

    expected_identity = {
        "schema_version": 1,
        "status": "failed_validation",
        "method": "original-3DGS",
        "benchmark": "SPIn-NeRF",
        "scene": "truck",
        "geometry_scene": "truck",
        "geometry_protocol": "released_all_view",
        "expected_registered_images": EXPECTED_REGISTERED_IMAGES,
        "evaluation_render_resolution": list(EXPECTED_RGB_RESOLUTION),
        "source_asset_contract": NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT,
        "returncode": 0,
        "error_type": "TrainingProtocolError",
        "error": LEGACY_VALIDATION_ERROR,
    }
    for key, expected in expected_identity.items():
        _require(
            type(manifest.get(key)) is type(expected)
            and manifest.get(key) == expected,
            f"Recovery identity mismatch for {key}: expected {expected!r}, "
            f"found {manifest.get(key)!r}",
        )
    _require(
        manifest.get("attempt_id") == run_dir.name,
        "Manifest attempt_id does not match the run directory",
    )
    _require(
        "training_output" not in manifest,
        "Refusing to overwrite an existing training_output receipt",
    )

    log_path = run_dir / "stdout_stderr.log"
    _require(
        Path(manifest.get("log", "")).resolve() == log_path,
        "Manifest log path does not resolve to the attempt log",
    )
    _require(log_path.is_file(), f"Missing training log: {log_path}")
    _require(
        _sha256(log_path) == manifest.get("log_sha256"),
        "Training log hash changed after the failed validation",
    )

    camera_audit = manifest.get("camera_audit")
    _require(isinstance(camera_audit, dict), "Missing camera_audit receipt")
    required_camera_audit = {
        "strategy": "stage_native_graphdeco_truck_pinhole",
        "camera_model": "PINHOLE",
        "camera_metadata_dimensions": list(
            EXPECTED_CAMERA_METADATA_RESOLUTION
        ),
        "registered_images": EXPECTED_REGISTERED_IMAGES,
        "rgb_images": EXPECTED_REGISTERED_IMAGES,
        "rgb_dimensions": list(EXPECTED_RGB_RESOLUTION),
        "released_rgb_downsampling": "ceil_half_from_colmap_metadata",
        "raw_dataset_modified": False,
    }
    for key, expected in required_camera_audit.items():
        _require(
            type(camera_audit.get(key)) is type(expected)
            and camera_audit.get(key) == expected,
            f"Camera audit mismatch for {key}",
        )

    staged_scene = run_dir / "staging" / "colmap_pinhole_undistorted"
    _require(
        Path(camera_audit.get("staged_scene", "")).resolve() == staged_scene,
        "Camera audit staged_scene does not resolve inside the attempt",
    )
    sparse_hashes = camera_audit.get("source_sha256")
    expected_sparse_names = {"cameras.bin", "images.bin", "points3D.bin"}
    _require(
        isinstance(sparse_hashes, dict)
        and set(sparse_hashes) == expected_sparse_names,
        "Incomplete staged sparse hash receipt",
    )
    for name in sorted(expected_sparse_names):
        path = staged_scene / "sparse" / "0" / name
        _require(path.is_file(), f"Missing staged sparse asset: {path}")
        _require(
            _sha256(path) == sparse_hashes[name],
            f"Staged sparse asset changed: {name}",
        )

    rgb_hashes = camera_audit.get("rgb_sha256")
    _require(
        isinstance(rgb_hashes, dict)
        and len(rgb_hashes) == EXPECTED_REGISTERED_IMAGES,
        "Incomplete staged RGB hash receipt",
    )
    image_dir = staged_scene / "images"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    _require(
        {path.name for path in image_paths} == set(rgb_hashes),
        "Staged RGB names changed after training",
    )
    for path in image_paths:
        _require(
            _sha256(path) == rgb_hashes[path.name],
            f"Staged RGB changed after training: {path.name}",
        )

    model_path = run_dir / "model"
    try:
        training_output = _validate_training_output(
            run_dir,
            model_path,
            expected_registered_images=EXPECTED_REGISTERED_IMAGES,
            expected_source_resolution=EXPECTED_RGB_RESOLUTION,
            expected_camera_metadata_resolution=(
                EXPECTED_CAMERA_METADATA_RESOLUTION
            ),
        )
    except TrainingProtocolError as error:
        raise PostflightRecoveryError(str(error)) from error
    _require(
        training_output["point_cloud_sha256"]
        == expected_point_cloud_sha256,
        "Final point cloud hash does not match the reviewed recovery pin",
    )
    cfg_args = training_output["cfg_args"]
    _require(
        Path(cfg_args.get("source_path", "")).resolve() == staged_scene,
        "cfg_args source_path does not resolve to the staged scene",
    )
    _require(
        Path(cfg_args.get("model_path", "")).resolve() == model_path,
        "cfg_args model_path does not resolve to the attempt model",
    )
    cameras = json.loads((model_path / "cameras.json").read_text(encoding="utf-8"))
    camera_names = {
        camera.get("img_name")
        for camera in cameras
        if isinstance(camera, dict)
    }
    _require(
        camera_names == {Path(name).stem for name in rgb_hashes},
        "cameras.json names do not match the staged RGB receipt",
    )

    protocol = manifest.get("effective_training_protocol")
    _require(isinstance(protocol, dict), "Missing effective training protocol")
    required_protocol = {
        "registered_training_views": EXPECTED_REGISTERED_IMAGES,
        "held_out_training_views": 0,
        "eval_split_enabled": False,
        "iterations": 30000,
        "resolution_argument": -1,
        "source_camera_resolution": list(EXPECTED_RGB_RESOLUTION),
        "effective_resolution": list(EXPECTED_RGB_RESOLUTION),
        "evaluation_render_resolution": list(EXPECTED_RGB_RESOLUTION),
        "save_iterations": [30000],
    }
    for key, expected in required_protocol.items():
        _require(
            type(protocol.get(key)) is type(expected)
            and protocol.get(key) == expected,
            f"Effective training protocol mismatch for {key}",
        )

    recovered = dict(manifest)
    recovered_protocol = dict(protocol)
    recovered_protocol["source_rgb_resolution"] = list(
        EXPECTED_RGB_RESOLUTION
    )
    recovered_protocol["camera_metadata_resolution"] = list(
        EXPECTED_CAMERA_METADATA_RESOLUTION
    )
    recovered["effective_training_protocol"] = recovered_protocol
    recovered["training_output"] = training_output
    recovered["postflight_recovery"] = {
        "reason": (
            "Legacy wrapper compared Graphdeco cameras.json COLMAP metadata "
            "dimensions with the released half-resolution RGB dimensions"
        ),
        "previous_status": manifest["status"],
        "previous_error_type": manifest["error_type"],
        "previous_error": manifest["error"],
        "pre_recovery_manifest_sha256": found_manifest_sha256,
        "reviewed_point_cloud_sha256": expected_point_cloud_sha256,
        "recovery_tool": str(Path(__file__).resolve()),
        "recovery_tool_sha256": _sha256(Path(__file__).resolve()),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "gpu_work_performed": False,
        "model_files_modified": False,
    }
    recovered["status"] = "complete"
    recovered.pop("error_type", None)
    recovered.pop("error", None)
    return manifest_path, recovered


def recover(
    run_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_point_cloud_sha256: str,
    apply: bool,
) -> dict[str, Any]:
    """Validate a recovery candidate and optionally atomically close it out."""

    manifest_path, recovered = _validated_recovery_manifest(
        run_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_point_cloud_sha256=expected_point_cloud_sha256,
    )
    if not apply:
        return recovered

    _require(
        _sha256(manifest_path) == expected_manifest_sha256,
        "Training manifest changed during recovery validation",
    )
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.postflight-{os.getpid()}.tmp"
    )
    _require(not temporary.exists(), f"Recovery temporary path exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(recovered, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, manifest_path.stat().st_mode & 0o777)
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return recovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-point-cloud-sha256", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace only training_manifest.json after validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recovered = recover(
        args.run_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_point_cloud_sha256=args.expected_point_cloud_sha256,
        apply=args.apply,
    )
    output = recovered["training_output"]
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "check-only",
                "status": recovered["status"],
                "registered_all_view_cameras": output[
                    "registered_all_view_cameras"
                ],
                "source_rgb_resolution": output["source_rgb_resolution"],
                "camera_metadata_resolution": output[
                    "camera_metadata_resolution"
                ],
                "point_cloud_vertices": output["point_cloud_vertices"],
                "point_cloud_sha256": output["point_cloud_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
