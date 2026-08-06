#!/usr/bin/env python3
"""Versioned exact-W export through one caller-bound explicit carrier bundle.

The frozen legacy exporter resolves a queue layout.  This wrapper leaves that
entrypoint untouched and materializes a private, verified queue overlay whose
three carrier assets are immutable symlinks to caller-hashed files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from radio_gs.interfaces.prompt_responsibility_cache import sha256_file
from radio_gs.scripts.export_prompt_responsibility_cache import export


REGISTRATION = "prompt_responsibility_explicit_carrier_overlay_v1"


def _require_sha256(value: str, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _verified(path: str, expected: str, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing {label}: {source}")
    required = _require_sha256(expected, label=f"expected {label}")
    actual = sha256_file(source)
    if actual != required:
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return source


def _immutable_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source:
            raise FileExistsError(f"carrier overlay differs: {destination}")
        return
    os.symlink(str(source), str(destination))


def _write_json_noclobber(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"explicit-carrier report differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = _verified(args.scene_config, args.scene_config_sha256, label="scene config")
    checkpoint = _verified(
        args.scene_checkpoint,
        args.scene_checkpoint_sha256,
        label="scene checkpoint",
    )
    camera_map = _verified(args.camera_map, args.camera_map_sha256, label="camera map")
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("explicit-carrier exact-W output/report already exists")

    overlay_root = output.parent / "carrier_overlay_v1"
    overlay_scene = overlay_root / "scenes" / str(args.scene_id)
    _immutable_symlink(config, overlay_scene / "gaussfm_main_track.yaml")
    _immutable_symlink(
        checkpoint,
        overlay_scene / "feature_field" / "checkpoints" / "best.pth",
    )
    _immutable_symlink(camera_map, overlay_scene / "rgb_to_colmap_camera_mapping.json")

    legacy_args = SimpleNamespace(
        manifest=args.manifest,
        queue_root=str(overlay_root),
        scene_id=args.scene_id,
        output=str(output),
        report=None,
        device=args.device,
        cpu_staging_lock=args.cpu_staging_lock,
        telemetry_log=args.telemetry_log,
        execution_log=args.execution_log,
        expected_prompt_type=args.expected_prompt_type,
        expected_reference_frame_id=args.expected_reference_frame_id,
        expected_reference_mask_sha256=args.expected_reference_mask_sha256,
        expected_native_height=args.expected_native_height,
        expected_native_width=args.expected_native_width,
        overwrite=False,
    )
    report = export(legacy_args)
    report["carrier_override_authority"] = {
        "registration": REGISTRATION,
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "overlay_root": str(overlay_root),
        "config": {"path": str(config), "sha256": args.scene_config_sha256},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": args.scene_checkpoint_sha256,
        },
        "camera_map": {"path": str(camera_map), "sha256": args.camera_map_sha256},
        "all_three_verified_before_legacy_export": True,
        "legacy_exporter_modified": False,
    }
    _write_json_noclobber(report_path, report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--scene-id", required=True)
    result.add_argument("--scene-config", required=True)
    result.add_argument("--scene-config-sha256", required=True)
    result.add_argument("--scene-checkpoint", required=True)
    result.add_argument("--scene-checkpoint-sha256", required=True)
    result.add_argument("--camera-map", required=True)
    result.add_argument("--camera-map-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--report", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--cpu-staging-lock")
    result.add_argument("--telemetry-log")
    result.add_argument("--execution-log")
    result.add_argument(
        "--expected-prompt-type",
        choices=("reference_binary_mask",),
        required=True,
    )
    result.add_argument("--expected-reference-frame-id", required=True)
    result.add_argument("--expected-reference-mask-sha256", required=True)
    result.add_argument("--expected-native-height", type=int, required=True)
    result.add_argument("--expected-native-width", type=int, required=True)
    return result


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
