#!/usr/bin/env python3
"""Run a fixed NVOS carrier-native bridge queue with bounded NFS staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


REPO = Path("/root/RADIO-GS")
PYTHON = Path("/root/miniconda3/envs/cybersim_agent/bin/python")
MANIFEST = Path("/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json")
QUEUE_ROOT = Path("/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1")
EXACT_ROOT = Path("/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260821/nvos_ludvig_carrier_native_bridge_exact_w_full8_v1")
BRIDGE_ROOT = Path("/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260821/nvos_ludvig_carrier_native_bridge_v1")
TARGET_ROOT = Path("/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260821/nvos_ludvig_carrier_native_bridge_target_v1")
NATIVE_AUDIT = REPO / "paper/artifacts/nvos_ludvig_published_compatible_asset_audit_20260804.json"
METHOD_MANIFEST = Path("/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260821/nvos_ludvig_region_reliability_gate_v1_full8/prediction_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage(source: Path, directory: Path, expected: str | None = None) -> tuple[Path, str]:
    target = directory / source.name
    shutil.copyfile(source, target)
    digest = _sha256(target)
    if expected is not None and digest != expected:
        raise ValueError(f"staged file differs: {source}")
    if target.stat().st_size != source.stat().st_size:
        raise ValueError(f"staged file size differs: {source}")
    return target, digest


def _run(command: list[str], log: Path, gpu: int) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("xb") as handle:
        result = subprocess.run(
            command, cwd=REPO, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); inspect {log}")


def _record_by_scene(payload: dict, scene: str) -> dict:
    matches = [row for row in payload["records"] if row["scene_id"] == scene]
    if len(matches) != 1:
        raise ValueError(f"Method-v1 record differs for {scene}")
    return matches[0]


def run_scene(scene: str, gpu: int) -> None:
    native = json.loads(NATIVE_AUDIT.read_text(encoding="utf-8"))["scenes"][scene]["runs"]["0"]
    method = _record_by_scene(
        json.loads(METHOD_MANIFEST.read_text(encoding="utf-8")), scene
    )
    scene_root = QUEUE_ROOT / "scenes" / scene
    checkpoint = scene_root / "feature_field/checkpoints/best.pth"
    config = scene_root / "gaussfm_main_track.yaml"
    camera = scene_root / "rgb_to_colmap_camera_mapping.json"
    exact_cache = EXACT_ROOT / "cache" / f"{scene}.pt"
    exact_report = EXACT_ROOT / "reports" / f"{scene}.export.json"
    bridge_output = BRIDGE_ROOT / scene
    target_output = TARGET_ROOT / scene
    logs = EXACT_ROOT / "logs"
    if target_output.joinpath("target_diagnostic.json").is_file():
        return

    with tempfile.TemporaryDirectory(
        prefix=f"nvos_bridge_{scene}_", dir="/tmp"
    ) as temporary_name:
        staging = Path(temporary_name)
        checkpoint_local, checkpoint_sha = _stage(checkpoint, staging)
        if not exact_report.is_file():
            if exact_cache.exists():
                raise FileExistsError(f"exact cache exists without report: {exact_cache}")
            _run(
                [
                    str(PYTHON), "-m", "radio_gs.scripts.export_prompt_responsibility_cache",
                    "--manifest", str(MANIFEST), "--queue-root", str(QUEUE_ROOT),
                    "--scene-id", scene, "--output", str(exact_cache),
                    "--report", str(exact_report), "--device", "cuda:0",
                    "--cpu-staging-lock", "/tmp/nvos_ludvig_bridge_exact_w_cpu.lock",
                    "--expected-prompt-type", "positive_negative_scribbles",
                    "--geometry-checkpoint-local-copy", str(checkpoint_local),
                    "--expected-geometry-checkpoint-sha256", checkpoint_sha,
                    "--local-artifact-staging-dir", "/tmp",
                ],
                logs / f"{scene}.export.log", gpu,
            )
        report = json.loads(exact_report.read_text(encoding="utf-8"))
        authority = report["authority"]
        if (
            authority["scene_id"] != scene
            or authority["target_rgb_opened"] is not False
            or authority["target_mask_opened"] is not False
            or authority["geometry_checkpoint_sha256"] != checkpoint_sha
        ):
            raise ValueError(f"exact-W report authority differs for {scene}")

        ply = Path(native["native_gaussian_geometry"]["path"])
        scalar = Path(native["primitive_scalar_uplift"]["path"])
        ply_sha = native["native_gaussian_geometry"]["sha256"]
        scalar_sha = native["primitive_scalar_uplift"]["sha256"]
        ply_local, _ = _stage(ply, staging, ply_sha)
        scalar_local, _ = _stage(scalar, staging, scalar_sha)
        if not bridge_output.joinpath("receipt.json").is_file():
            _run(
                [
                    str(PYTHON), "-m", "radio_gs.scripts.materialize_nvos_ludvig_carrier_native_bridge",
                    "--scene-id", scene, "--reference-frame", authority["frame_id"],
                    "--native-ply", str(ply), "--native-ply-sha256", ply_sha,
                    "--native-ply-local-copy", str(ply_local),
                    "--native-scalar", str(scalar), "--native-scalar-sha256", scalar_sha,
                    "--native-scalar-local-copy", str(scalar_local),
                    "--dataset-manifest", str(MANIFEST),
                    "--dataset-manifest-sha256", authority["source_sha256"]["benchmark_manifest"],
                    "--current-config", str(config),
                    "--current-config-sha256", authority["source_sha256"]["gaussfm_config"],
                    "--camera-mapping", str(camera),
                    "--camera-mapping-sha256", authority["source_sha256"]["camera_mapping"],
                    "--current-responsibility", str(exact_cache),
                    "--current-responsibility-sha256", report["file_sha256"],
                    "--device", "cuda:0", "--output-dir", str(bridge_output),
                ],
                logs / f"{scene}.bridge.log", gpu,
            )
        receipt_path = bridge_output / "receipt.json"
        receipt_sha = _sha256(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        state = receipt["outputs"]["primitive_state"]
        selector = native["rendered_binary_selector"]
        prediction = method["method_v1_prediction"]
        if not target_output.joinpath("target_diagnostic.json").is_file():
            _run(
                [
                    str(PYTHON), "-m", "radio_gs.scripts.evaluate_nvos_ludvig_carrier_native_bridge",
                    "--scene-id", scene, "--bridge-receipt", str(receipt_path),
                    "--bridge-receipt-sha256", receipt_sha,
                    "--bridge-state", state["path"], "--bridge-state-sha256", state["sha256"],
                    "--manifest", str(MANIFEST), "--manifest-sha256", _sha256(MANIFEST),
                    "--queue-root", str(QUEUE_ROOT),
                    "--geometry-checkpoint-local-copy", str(checkpoint_local),
                    "--expected-geometry-checkpoint-sha256", checkpoint_sha,
                    "--native-target-selector", selector["path"],
                    "--native-target-selector-sha256", selector["sha256"],
                    "--method-v1-prediction", prediction["path"],
                    "--method-v1-prediction-sha256", prediction["sha256"],
                    "--threshold-parameter", "75", "--device", "cuda:0",
                    "--output-dir", str(target_output),
                ],
                logs / f"{scene}.target.log", gpu,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    args = parser.parse_args(argv)
    for scene in args.scenes:
        run_scene(scene, args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
