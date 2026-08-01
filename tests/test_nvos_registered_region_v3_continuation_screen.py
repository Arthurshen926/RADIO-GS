from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from radio_gs.scripts.screen_nvos_registered_region_v3_continuation import (
    CANDIDATE_ELIGIBILITY,
    CANDIDATE_ID,
    FULL_SCENES,
    PARENT_V2_PROPAGATED_IOU,
    SCREEN_CONTRACT,
    SCREEN_CONTRACT_SHA256,
    SCREEN_SCENES,
    screen,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_SCRIPT = (
    REPO_ROOT / "radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py"
)
AGGREGATE_SCRIPT = (
    REPO_ROOT / "radio_gs/scripts/aggregate_registered_prompt_closeout.py"
)
EVALUATOR = REPO_ROOT / "radio_gs/scripts/eval_nvos_gaussian_first.py"
RUNNER = REPO_ROOT / "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh"
CANDIDATE_CONTRACT = (
    REPO_ROOT / "paper/artifacts/nvos_registered_region_v3_candidate_20260731.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_result(
    root: Path,
    *,
    scene: str,
    propagated_iou: float,
    manifest: dict,
    manifest_sha256: str,
) -> Path:
    result_root = root / scene / "eval_full_mask_random_walker"
    artifact_root = result_root / "artifacts"
    artifact_root.mkdir(parents=True)
    final_score = artifact_root / "final.npy"
    stage_scores = {
        "unary_prior": artifact_root / "unary_prior.npy",
        "propagated": artifact_root / "propagated.npy",
        "connected": artifact_root / "connected.npy",
    }
    final_score.write_bytes(b"propagated-score")
    stage_scores["unary_prior"].write_bytes(b"unary-score")
    stage_scores["propagated"].write_bytes(b"propagated-score")
    stage_scores["connected"].write_bytes(b"connected-score")

    method_contract = {
        "schema_version": 2,
        "candidate_id": CANDIDATE_ID,
        "evaluator": "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "evaluator_sha256": _sha256(EVALUATOR),
        "implementation_sha256": {},
        "radio_checkpoint_sha256": manifest["radio_checkpoint_sha256"],
        "candidate_run_manifest_sha256": manifest_sha256,
        "candidate_method_contract_sha256": _json_sha256(manifest["method_contract"]),
        "candidate_eligibility": CANDIDATE_ELIGIBILITY,
        "shared_solver": {"registered_readout_stage": "propagated"},
    }
    method_sha256 = _json_sha256(method_contract)
    dataset_contract = {
        "schema_version": 1,
        "benchmark": "nvos",
        "protocol_hash": "protocol",
        "legacy_protocol_hash": "protocol",
        "benchmark_manifest_sha256": manifest["benchmark_manifest_sha256"],
        "cohort": list(FULL_SCENES),
    }
    dataset_sha256 = _json_sha256(dataset_contract)
    evaluation_contract = {
        "schema_version": 1,
        "method_config_sha256": method_sha256,
        "dataset_protocol_sha256": dataset_sha256,
        "final_readout": "propagated",
        "pixel_threshold": {"value": 0.5, "comparison": "greater_or_equal"},
    }
    evaluation_sha256 = _json_sha256(evaluation_contract)
    frame = {
        "frame_id": "frame0",
        "foreground_iou": propagated_iou,
        "pixel_accuracy": 0.9,
    }
    stage_values = {
        "unary_prior": max(0.0, propagated_iou - 0.05),
        "propagated": propagated_iou,
        "connected": max(0.0, propagated_iou - 0.10),
    }
    result = {
        "scene_id": scene,
        "protocol_hash": "protocol",
        "legacy_protocol_hash": "protocol",
        "foreground_iou": propagated_iou,
        "pixel_accuracy": 0.9,
        "score_threshold": 0.5,
        "shared_solver": {"registered_readout_stage": "propagated"},
        "frames": [frame],
        "stage_metrics": {
            stage: {
                "foreground_iou": value,
                "pixel_accuracy": 0.9,
                "frames": [
                    {
                        "frame_id": "frame0",
                        "foreground_iou": value,
                        "pixel_accuracy": 0.9,
                    }
                ],
            }
            for stage, value in stage_values.items()
        },
        "score_paths": {"frame0": str(final_score.resolve())},
        "score_sha256": {"frame0": _sha256(final_score)},
        "stage_score_paths": {
            stage: {"frame0": str(path.resolve())}
            for stage, path in stage_scores.items()
        },
        "stage_score_sha256": {
            stage: {"frame0": _sha256(path)} for stage, path in stage_scores.items()
        },
        "safety": {
            "target_ground_truth_opened_before_prediction_write": False,
            "target_rgb_opened": False,
            "target_camera_used_as_support": False,
            "test_calibration": False,
            "candidate_eligibility": CANDIDATE_ELIGIBILITY,
            "frozen_diagnostic_eligible": True,
            "main_result_eligible": False,
        },
        "method_contract": method_contract,
        "method_config_sha256": method_sha256,
        "dataset_protocol_contract": dataset_contract,
        "dataset_protocol_sha256": dataset_sha256,
        "evaluation_protocol_contract": evaluation_contract,
        "evaluation_protocol_sha256": evaluation_sha256,
        "run_manifest_sha256": manifest_sha256,
    }
    result_path = result_root / f"{scene}_evaluation.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def _build_screen_run(
    tmp_path: Path,
    values: dict[str, float],
) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    root = tmp_path / "run"
    root.mkdir()
    queue = tmp_path / "queue_plan.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": scene} for scene in FULL_SCENES],
            }
        ),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark_manifest.json"
    benchmark.write_text(
        json.dumps({"benchmark": "nvos", "protocol_hash": "protocol"}),
        encoding="utf-8",
    )
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output = root / "three_scene_screen.json"
    partial = root / "partial_completion.json"
    method_contract = {
        "schema_version": 1,
        "method": CANDIDATE_ID,
        "final_readout": "propagated",
    }
    manifest = {
        "schema_version": 2,
        "candidate": CANDIDATE_ID,
        "eligibility": CANDIDATE_ELIGIBILITY,
        "scenes": list(FULL_SCENES),
        "queue_plan": str(queue.resolve()),
        "queue_plan_sha256": _sha256(queue),
        "benchmark_manifest": str(benchmark.resolve()),
        "benchmark_manifest_sha256": _sha256(benchmark),
        "radio_checkpoint_sha256": "r" * 64,
        "method_contract": method_contract,
        "runner": str(runner.resolve()),
        "runner_sha256": _sha256(runner),
        "implementation_sources": {
            "radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py": _sha256(
                SCREEN_SCRIPT
            ),
            "radio_gs/scripts/aggregate_registered_prompt_closeout.py": _sha256(
                AGGREGATE_SCRIPT
            ),
            "radio_gs/scripts/eval_nvos_gaussian_first.py": _sha256(EVALUATOR),
        },
        "continuation_screen": {
            "script": str(SCREEN_SCRIPT.resolve()),
            "script_sha256": _sha256(SCREEN_SCRIPT),
            "candidate_contract": str(CANDIDATE_CONTRACT.resolve()),
            "candidate_contract_sha256": _sha256(CANDIDATE_CONTRACT),
            "contract": SCREEN_CONTRACT,
            "contract_sha256": SCREEN_CONTRACT_SHA256,
            "output": str(output.resolve()),
            "partial_completion_output": str(partial.resolve()),
        },
    }
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = _sha256(manifest_path)
    results = {
        scene: _write_result(
            root,
            scene=scene,
            propagated_iou=values[scene],
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
        for scene in SCREEN_SCENES
    }
    return root, manifest_path, output, partial, results


def _run_screen(
    root: Path,
    manifest: Path,
    output: Path,
    partial: Path,
) -> dict:
    return screen(
        result_root=root,
        run_manifest=manifest,
        candidate_contract=CANDIDATE_CONTRACT,
        output=output,
        partial_completion_output=partial,
    )


def test_continue_requires_macro_two_wins_and_no_regression(tmp_path: Path) -> None:
    values = dict(PARENT_V2_PROPAGATED_IOU)
    values["fern"] += 0.01
    values["flower"] += 0.01
    root, manifest, output, partial, _ = _build_screen_run(tmp_path, values)

    report = _run_screen(root, manifest, output, partial)

    assert report["decision"] == "continue_full_eight"
    assert report["continue_full_eight"] is True
    assert report["strict_scene_wins"] == 2
    assert report["scene_regressions"] == 0
    assert all(report["gates"].values())
    assert not partial.exists()


def test_regression_rejects_even_when_macro_and_two_wins_pass(tmp_path: Path) -> None:
    values = dict(PARENT_V2_PROPAGATED_IOU)
    values["fern"] += 0.05
    values["flower"] += 0.05
    values["fortress"] -= 0.01
    root, manifest, output, partial, _ = _build_screen_run(tmp_path, values)

    report = _run_screen(root, manifest, output, partial)

    assert report["gates"]["macro_strictly_improved"] is True
    assert report["gates"]["at_least_two_strict_scene_wins"] is True
    assert report["gates"]["no_scene_regression"] is False
    assert report["decision"] == "reject_stop_after_three"
    completion = json.loads(partial.read_text(encoding="utf-8"))
    assert completion["normal_stop"] is True
    assert completion["completed_scenes"] == list(SCREEN_SCENES)
    assert completion["unrun_scenes"] == list(FULL_SCENES[3:])
    assert completion["run_manifest_sha256"] == _sha256(manifest)
    assert completion["three_scene_screen_sha256"] == _sha256(output)


def test_one_win_rejects_despite_strict_macro_gain(tmp_path: Path) -> None:
    values = dict(PARENT_V2_PROPAGATED_IOU)
    values["fern"] += 0.03
    root, manifest, output, partial, _ = _build_screen_run(tmp_path, values)

    report = _run_screen(root, manifest, output, partial)

    assert report["gates"]["macro_strictly_improved"] is True
    assert report["strict_scene_wins"] == 1
    assert report["decision"] == "reject_stop_after_three"


def test_score_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    values = {scene: PARENT_V2_PROPAGATED_IOU[scene] + 0.01 for scene in SCREEN_SCENES}
    root, manifest, output, partial, results = _build_screen_run(tmp_path, values)
    result = json.loads(results["fern"].read_text(encoding="utf-8"))
    Path(result["score_paths"]["frame0"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="score artifact SHA mismatch"):
        _run_screen(root, manifest, output, partial)


def test_run_manifest_candidate_tamper_fails_closed(tmp_path: Path) -> None:
    values = {scene: PARENT_V2_PROPAGATED_IOU[scene] + 0.01 for scene in SCREEN_SCENES}
    root, manifest, output, partial, _ = _build_screen_run(tmp_path, values)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidate"] = "registered-region-v2"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind the frozen v3 screen"):
        _run_screen(root, manifest, output, partial)


def test_runner_calls_screen_after_fortress_and_keeps_thermal_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")

    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"' in source
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-60}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-74}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-68}"' in source
    assert 'GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-77}"' in source
    assert 'GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-75}"' in source
    assert 'GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"' in source
    assert 'GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-300.5}"' in source
    assert 'GPU_PEER_MAX_MEMORY_MIB="${GPU_PEER_MAX_MEMORY_MIB:-0}"' in source
    assert 'GPU_PEER_MAX_UTIL_PCT="${GPU_PEER_MAX_UTIL_PCT:-100}"' in source
    assert "physical GPU1 already has compute owner(s)" in source
    assert 'MAIN_OUTPUT_ROOT="/root/RADIO-GS/output"' in source
    assert 'GLOBAL_GPU1_LOCK="$MAIN_OUTPUT_ROOT/.physical_gpu1.lock"' in source
    assert "surface_gpu1_lock_supervisor.py" in source
    assert "RADIO_GS_GPU1_SINGLETON_FD" in source
    assert "verify-inherited" in source
    assert 'exec {gpu1_lock}<>"$GLOBAL_GPU1_LOCK"' not in source
    assert 'exec {run_lock}>"$LOCK_ROOT/run.lock"' in source
    screen_branch = source.index('if [[ "$scene" == "fortress" ]]')
    aggregate = source.index("aggregate_registered_prompt_closeout.py", screen_branch)
    assert screen_branch < aggregate
    assert 'if [[ "$screen_decision" == "reject_stop_after_three" ]]' in source
    assert "partial_completion.json" in source
    assert "SCREEN_CONTRACT_SHA256" in source
    assert "radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py" in source
    assert '"implementation_sources": implementation' in source
    assert '"runtime_closure": runtime_closure' in source
