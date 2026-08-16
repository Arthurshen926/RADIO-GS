#!/usr/bin/env python3
"""Run the frozen SPIn Method-v1 full9 readout behind one pre-GT barrier."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence

from radio_gs.scripts.materialize_spin9_method_v1_signed_field import (
    _field_binding,
    _validate_existing as _validate_signed_scene,
)
from radio_gs.scripts.run_nvos_method_v1_full8_readout import SAM3_RUNTIME_ENV
from radio_gs.scripts.run_nvos_method_v1_scene import GPU_THERMAL_ENV
from radio_gs.scripts.run_spin9_method_v1_scene import (
    DATASET_MANIFEST,
    DEFAULT_RUN_ROOT,
    METHOD_AUTHORITY,
    REPO_ROOT,
)
from radio_gs.scripts.score_spin9_method_v1_full9 import verify_full9_before_gt
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "method_v1_readout/full9_20260816"
STAGES = ("signed_field", "transient_sam", "score")


def frozen_scene_order() -> tuple[str, ...]:
    dataset, _digest, _source = load_json_object(
        DATASET_MANIFEST, label="SPIn Available-Nine dataset manifest"
    )
    authority, _authority_digest, _authority_source = load_json_object(
        METHOD_AUTHORITY, label="Method-v1 authority"
    )
    order = tuple(str(value) for value in dataset["protocol"]["cohort"])
    method = tuple(
        str(value) for value in authority["frozen_cohorts"]["spin_nerf_available9"]
    )
    if len(order) != 9 or len(set(order)) != 9 or set(order) != set(method):
        raise ValueError("SPIn Method-v1 Available-Nine cohort differs")
    return order


def resolve_ready_full9(field_root: Path) -> dict[str, dict[str, Any]]:
    """Hash all nine final fields before any reference mask or target RGB opens."""

    return {
        scene_id: _field_binding(scene_id, field_root)
        for scene_id in frozen_scene_order()
    }


def signed_command(*, scene_id: str, field_root: Path, output_root: Path) -> list[str]:
    return [
        "radio_gs/scripts/materialize_spin9_method_v1_signed_field.py",
        "--scene",
        scene_id,
        "--field-root",
        str(field_root),
        "--output-root",
        str(output_root / "signed_field"),
        "--scratch-root",
        str(output_root / "scratch"),
        "--device",
        "cuda:0",
    ]


def transient_command(output_root: Path) -> list[str]:
    return [
        "radio_gs/scripts/predict_spin9_method_v1_transient_sam.py",
        "--signed-root",
        str(output_root / "signed_field"),
        "--output-root",
        str(output_root / "transient_sam"),
        "--device",
        "cuda:0",
    ]


def score_command(output_root: Path) -> list[str]:
    return [
        "radio_gs/scripts/score_spin9_method_v1_full9.py",
        "--manifest",
        str(DATASET_MANIFEST),
        "--prediction-manifest",
        str(output_root / "transient_sam/prediction_manifest.json"),
        "--method-authority",
        str(METHOD_AUTHORITY),
        "--output",
        str(output_root / "method_v1_spin9_full9_results.json"),
    ]


def _run(
    command: Iterable[str],
    *,
    log_path: Path,
    gpu: int | None,
    runtime_env: dict[str, str] | None = None,
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
    bindings = resolve_ready_full9(field_root)
    scene_order = tuple(bindings)
    stop_index = STAGES.index(args.stop_after)

    for scene_id in scene_order:
        scene_root = output_root / "signed_field/scenes" / scene_id
        receipt = _validate_signed_scene(scene_root, scene_id=scene_id)
        if receipt is None:
            print(
                f"{scene_id}: sealing scalar field prompts without target RGB",
                flush=True,
            )
            _run(
                signed_command(
                    scene_id=scene_id,
                    field_root=field_root,
                    output_root=output_root,
                ),
                log_path=output_root / "logs" / f"signed_{scene_id}.log",
                gpu=args.gpu,
            )
            receipt = _validate_signed_scene(scene_root, scene_id=scene_id)
        if (
            receipt is None
            or receipt["field"]["sha256"] != bindings[scene_id]["field_sha256"]
        ):
            raise RuntimeError(f"{scene_id} signed-field receipt was not sealed")
    if stop_index == 0:
        return {"completed_stage": STAGES[0], "scene_order": scene_order}

    transient_manifest = output_root / "transient_sam/prediction_manifest.json"
    if transient_manifest.is_file():
        verify_full9_before_gt(
            dataset_manifest_path=DATASET_MANIFEST,
            prediction_manifest_path=transient_manifest,
            method_authority_path=METHOD_AUTHORITY,
        )
    else:
        print(
            "full9: all scalar prompts sealed; opening reference/target RGB for transient SAM",
            flush=True,
        )
        _run(
            transient_command(output_root),
            log_path=output_root / "logs/transient_sam.log",
            gpu=args.gpu,
            runtime_env=SAM3_RUNTIME_ENV,
        )
        verify_full9_before_gt(
            dataset_manifest_path=DATASET_MANIFEST,
            prediction_manifest_path=transient_manifest,
            method_authority_path=METHOD_AUTHORITY,
        )
    if stop_index == 1:
        return {"completed_stage": STAGES[1], "scene_order": scene_order}

    result_path = output_root / "method_v1_spin9_full9_results.json"
    if result_path.is_file():
        result, _digest, _source = load_json_object(
            result_path, label="Method-v1 SPIn full9 result"
        )
        if result.get("artifact_type") != "radio_gs_method_v1_spin9_full9_results":
            raise ValueError("existing Method-v1 SPIn full9 result differs")
    else:
        print(
            "full9: complete prediction barrier sealed; opening evaluation masks once",
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
    parser.add_argument("--field-root", default=str(DEFAULT_RUN_ROOT))
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
