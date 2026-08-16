#!/usr/bin/env python3
"""Run the frozen Method-v1 NVOS full8 readout behind one pre-GT barrier.

All eight final fields and their Method-v1 gates are verified before any
feature render starts.  All factorized feature renders and signed field
prompts are then sealed before the transient SAM stage opens target RGB.  The
scorer is the only stage allowed to open target masks, after it independently
verifies the complete ordered batch and every field/prediction receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from radio_gs.five_benchmark_method_v1 import validate_method_authority
from radio_gs.scripts.run_nvos_method_v1_scene import (
    GPU_THERMAL_ENV,
    METHOD_AUTHORITY,
    NVOS_AUTHORITY,
    REPO_ROOT,
    resolve_scene_assets,
)
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


DATASET_MANIFEST = Path(
    "/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/"
    "nvos_strict_unseen_v1.json"
)
DEFAULT_FIELD_ROOT = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/" "core_method_v1/nvos"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_FIELD_ROOT / "method_v1_readout/full8_20260816"
FINAL_FIELD_NAME = "generic_text_response_w005_s0_64.pth"
SAM3_CHECKPOINT = REPO_ROOT / "checkpoints/sam3_modelscope/sam3.pt"
SAM3_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
SAM3_RUNTIME_ENV = {
    "RADIO_GS_PYTHON": "/root/miniconda3/envs/easy3d/bin/python",
    "RADIO_GS_SITE_PACKAGES": "/root/miniconda3/envs/easy3d/lib/python3.11/site-packages",
    "RADIO_GS_LD_LIBRARY_PATH": "/root/miniconda3/envs/easy3d/lib",
    "RADIO_GS_SAM3_SOURCE": "/root/external/sam3",
}
STAGES = ("render", "signed_prompt", "transient_sam", "score")


@dataclass(frozen=True)
class ReadoutScene:
    scene_id: str
    camera_map: Path
    config: Path
    geometry: Path
    final_field: Path
    final_field_sha256: str
    gate: Path


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value, _digest, _source = load_json_object(path, label=label)
    return value


def frozen_scene_order(method_authority: Path = METHOD_AUTHORITY) -> tuple[str, ...]:
    authority = _load_object(method_authority, label="Method-v1 authority")
    validate_method_authority(authority)
    scenes = tuple(str(value) for value in authority["frozen_cohorts"]["nvos"])
    if len(scenes) != 8 or len(set(scenes)) != 8:
        raise ValueError("Method-v1 NVOS cohort is not an ordered full8")
    return scenes


def resolve_ready_scenes(
    *,
    field_root: Path,
    method_authority: Path = METHOD_AUTHORITY,
    nvos_authority: Path = NVOS_AUTHORITY,
) -> tuple[ReadoutScene, ...]:
    """Fail closed unless every frozen scene has a hash-valid final field gate."""

    method_sha = sha256_file(method_authority)
    exact = _load_object(nvos_authority, label="NVOS exact authority")
    exact_rows = {str(row["scene_id"]): row for row in exact["scenes"]}
    ready: list[ReadoutScene] = []
    for scene_id in frozen_scene_order(method_authority):
        assets = resolve_scene_assets(scene_id)
        scene_root = field_root / scene_id
        config = scene_root / "method_v1.yaml"
        final_field = scene_root / FINAL_FIELD_NAME
        gate = scene_root / "method_v1_gate.json"
        gate_payload = _load_object(gate, label=f"{scene_id} Method-v1 gate")
        field_sha = sha256_file(final_field)
        if (
            gate_payload.get("status") != "pass"
            or gate_payload.get("benchmark") != "NVOS"
            or gate_payload.get("scene") != scene_id
            or Path(str(gate_payload.get("field", ""))).resolve()
            != final_field.resolve()
            or gate_payload.get("field_sha256") != field_sha
            or Path(str(gate_payload.get("method_authority", ""))).resolve()
            != method_authority.resolve()
            or gate_payload.get("method_authority_sha256") != method_sha
        ):
            raise ValueError(f"{scene_id} Method-v1 gate differs")
        row = exact_rows[scene_id]
        camera_map = Path(row["camera_map"]["path"]).resolve(strict=True)
        if sha256_file(camera_map) != row["camera_map"]["sha256"]:
            raise ValueError(f"{scene_id} camera-map SHA-256 differs")
        if not config.is_file():
            raise FileNotFoundError(config)
        ready.append(
            ReadoutScene(
                scene_id=scene_id,
                camera_map=camera_map,
                config=config.resolve(),
                geometry=assets.geometry,
                final_field=final_field.resolve(),
                final_field_sha256=field_sha,
                gate=gate.resolve(),
            )
        )
    return tuple(ready)


def render_command(scene: ReadoutScene, output_root: Path) -> list[str]:
    return [
        "radio_gs/scripts/render_promptable_nvs_features.py",
        "--manifest",
        str(DATASET_MANIFEST),
        "--scene-id",
        scene.scene_id,
        "--camera-map",
        str(scene.camera_map),
        "--config",
        str(scene.config),
        "--checkpoint",
        str(scene.geometry),
        "--canonical-field-checkpoint",
        str(scene.final_field),
        "--canonical-field-checkpoint-schema",
        "factorized-v2",
        "--expected-canonical-field-checkpoint-sha256",
        scene.final_field_sha256,
        "--output-dir",
        str(output_root / "rendered_features"),
        "--device",
        "cuda:0",
    ]


def signed_prompt_command(
    scenes: Sequence[ReadoutScene], output_root: Path
) -> list[str]:
    command = [
        "radio_gs/scripts/predict_promptable_nvs_feature_readout.py",
        "--manifest",
        str(DATASET_MANIFEST),
        "--output-dir",
        str(output_root / "signed_field_prompt"),
        "--feature-root",
        str(output_root / "rendered_features"),
        "--feature-pattern",
        "{scene_id}/{camera_name}.pt",
        "--feature-layout",
        "chw",
        "--require-render-authority",
        "--method-name",
        "RADIO-GS Method-v1 signed field prompt",
    ]
    for scene in scenes:
        command.extend(("--scene-id", scene.scene_id))
    return command


def transient_sam_command(
    scenes: Sequence[ReadoutScene], output_root: Path
) -> list[str]:
    command = [
        "radio_gs/scripts/predict_nvos_method_v1_transient_sam.py",
        "--manifest",
        str(DATASET_MANIFEST),
        "--signed-field-prompt-manifest",
        str(output_root / "signed_field_prompt/prediction_manifest.json"),
        "--output-dir",
        str(output_root / "transient_sam"),
        "--method-authority",
        str(METHOD_AUTHORITY),
        "--checkpoint",
        str(SAM3_CHECKPOINT),
        "--expected-checkpoint-sha256",
        SAM3_CHECKPOINT_SHA256,
        "--device",
        "cuda:0",
    ]
    for scene in scenes:
        command.extend(("--scene-id", scene.scene_id))
    return command


def score_command(output_root: Path) -> list[str]:
    return [
        "radio_gs/scripts/score_nvos_method_v1_full8.py",
        "--manifest",
        str(DATASET_MANIFEST),
        "--prediction-manifest",
        str(output_root / "transient_sam/prediction_manifest.json"),
        "--method-authority",
        str(METHOD_AUTHORITY),
        "--output",
        str(output_root / "method_v1_nvos_full8_results.json"),
    ]


def _validate_render(scene: ReadoutScene, output_root: Path) -> bool:
    report_path = (
        output_root / "rendered_features" / scene.scene_id / "render_manifest.json"
    )
    if not report_path.is_file():
        partial = report_path.parent
        if partial.exists() and any(partial.iterdir()):
            raise RuntimeError(f"partial render output must be audited: {partial}")
        return False
    report = _load_object(report_path, label=f"{scene.scene_id} render authority")
    if (
        report.get("kind") != "promptable_nvs_gaussfm_render"
        or report.get("scene_id") != scene.scene_id
        or report.get("canonical_field_checkpoint_schema") != "factorized-v2"
        or Path(str(report.get("canonical_field_checkpoint", ""))).resolve()
        != scene.final_field
        or report.get("canonical_field_checkpoint_sha256") != scene.final_field_sha256
        or report.get("safety", {}).get("evaluation_ground_truth_opened") is not False
        or report.get("safety", {}).get("rgb_files_opened") is not False
    ):
        raise ValueError(f"{scene.scene_id} existing render authority differs")
    outputs = report.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError(f"{scene.scene_id} render does not contain prompt and target")
    for row in outputs:
        path = Path(str(row.get("feature_path", ""))).resolve(strict=True)
        if sha256_file(path) != row.get("feature_sha256"):
            raise ValueError(f"{scene.scene_id} rendered feature SHA-256 differs")
    return True


def _validate_prediction_manifest(
    path: Path,
    *,
    kind: str,
    scene_order: Sequence[str],
) -> bool:
    if not path.is_file():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise RuntimeError(
                f"partial prediction output must be audited: {path.parent}"
            )
        return False
    payload = _load_object(path, label=kind)
    if payload.get("kind") != kind:
        raise ValueError(f"existing {kind} manifest differs")
    if kind == "promptable_nvs_continuous_score_predictions":
        safety = payload.get("safety", {})
        if (
            safety.get("evaluation_performed") is not False
            or safety.get("evaluation_ground_truth_opened") is not False
        ):
            raise ValueError("signed prompt manifest safety contract differs")
        actual = tuple(
            str(value)
            for value in payload.get("input", {}).get("selected_scene_ids", [])
        )
    else:
        if payload.get("evaluation_performed") is not False:
            raise ValueError("transient manifest evaluation boundary differs")
        actual = tuple(str(value) for value in payload.get("predictions", {}))
        if (
            payload.get("target_mask_opened") is not False
            or payload.get("target_metric_opened") is not False
        ):
            raise ValueError("transient manifest safety contract differs")
    if actual != tuple(scene_order):
        raise ValueError(f"existing {kind} manifest is not the ordered full8")
    return True


def _run(
    command: Iterable[str],
    *,
    log_path: Path,
    gpu: int | None,
    runtime_env: Mapping[str, str] | None = None,
) -> None:
    command = [str(value) for value in command]
    environment = os.environ.copy()
    if gpu is None:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        full_command = [
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
            *command,
        ]
    else:
        environment.update(
            {
                **GPU_THERMAL_ENV,
                "GPU": str(gpu),
                "GPU_TELEMETRY_LOG": str(log_path.with_suffix(".gpu_telemetry.csv")),
                "GPU_OWNER_AUDIT_LOG": str(log_path.with_suffix(".gpu_owner.csv")),
            }
        )
        full_command = [
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"),
            "--",
            "env",
            f"CUDA_VISIBLE_DEVICES={gpu}",
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
            *command,
        ]
    if runtime_env is not None:
        environment.update(runtime_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(full_command) + "\n")
        handle.flush()
        subprocess.run(
            full_command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    field_root = Path(args.field_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = resolve_ready_scenes(field_root=field_root)
    scene_order = tuple(scene.scene_id for scene in scenes)
    stop_index = STAGES.index(args.stop_after)

    for scene in scenes:
        if not _validate_render(scene, output_root):
            print(f"{scene.scene_id}: rendering sealed factorized-v2 field", flush=True)
            _run(
                render_command(scene, output_root),
                log_path=output_root / "logs" / f"render_{scene.scene_id}.log",
                gpu=args.gpu,
            )
            if not _validate_render(scene, output_root):
                raise RuntimeError(f"{scene.scene_id} render was not sealed")
    if stop_index == 0:
        return {"completed_stage": STAGES[0], "scene_order": scene_order}

    signed_manifest = output_root / "signed_field_prompt/prediction_manifest.json"
    if not _validate_prediction_manifest(
        signed_manifest,
        kind="promptable_nvs_continuous_score_predictions",
        scene_order=scene_order,
    ):
        print("full8: sealing signed field prompts before target RGB", flush=True)
        _run(
            signed_prompt_command(scenes, output_root),
            log_path=output_root / "logs/signed_prompt.log",
            gpu=None,
        )
    if stop_index == 1:
        return {"completed_stage": STAGES[1], "scene_order": scene_order}

    transient_manifest = output_root / "transient_sam/prediction_manifest.json"
    if not _validate_prediction_manifest(
        transient_manifest,
        kind="promptable_nvs_method_v1_transient_sam_predictions",
        scene_order=scene_order,
    ):
        print(
            "full8: signed prompts sealed; opening target RGB for transient SAM",
            flush=True,
        )
        _run(
            transient_sam_command(scenes, output_root),
            log_path=output_root / "logs/transient_sam.log",
            gpu=args.gpu,
            runtime_env=SAM3_RUNTIME_ENV,
        )
    if stop_index == 2:
        return {"completed_stage": STAGES[2], "scene_order": scene_order}

    result_path = output_root / "method_v1_nvos_full8_results.json"
    if result_path.exists():
        result = _load_object(result_path, label="Method-v1 NVOS full8 result")
        if result.get("artifact_type") != "radio_gs_method_v1_nvos_full8_results":
            raise ValueError("existing NVOS full8 result differs")
    else:
        print(
            "full8: complete prediction barrier sealed; opening target masks once",
            flush=True,
        )
        _run(
            score_command(output_root),
            log_path=output_root / "logs/score.log",
            gpu=None,
        )
    return {
        "completed_stage": STAGES[-1],
        "scene_order": scene_order,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", default=str(DEFAULT_FIELD_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
