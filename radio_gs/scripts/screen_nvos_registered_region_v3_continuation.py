#!/usr/bin/env python3
"""Apply the frozen three-scene continuation gate for registered-region-v3.

This script is CPU-only.  It validates the immutable run manifest and the
strict, score-hash-bound fern/flower/fortress reports before reading their
propagated IoU.  The remaining five scenes may run only when the v3 macro is
strictly above the frozen v2 macro, at least two scenes strictly improve, and
no scene regresses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from radio_gs.scripts.aggregate_registered_prompt_closeout import (
    _validate_strict_result,
)


CANDIDATE_ID = "registered-region-v3"
CANDIDATE_ELIGIBILITY = "diagnostic_until_disjoint_registered_prompt_gate"
FULL_SCENES = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
SCREEN_SCENES = FULL_SCENES[:3]
REMAINING_SCENES = FULL_SCENES[3:]
PARENT_V2_PROPAGATED_IOU = {
    "fern": 0.44801025143203155,
    "flower": 0.7886652885090718,
    "fortress": 0.6967529158458324,
}
PARENT_V2_MACRO = 0.6444761519289787
SCREEN_CONTRACT = {
    "schema_version": 1,
    "screen": "registered-region-v3-three-scene-continuation",
    "candidate_id": CANDIDATE_ID,
    "parent_candidate_id": "registered-region-v2",
    "full_scene_order": list(FULL_SCENES),
    "diagnostic_scene_order": list(SCREEN_SCENES),
    "remaining_scene_order": list(REMAINING_SCENES),
    "metric": "stage_metrics.propagated.foreground_iou",
    "parent_v2_propagated_iou": PARENT_V2_PROPAGATED_IOU,
    "parent_v2_macro": PARENT_V2_MACRO,
    "macro_rule": "v3_macro_strictly_greater_than_parent_v2_macro",
    "minimum_strict_scene_wins": 2,
    "scene_regressions_allowed": 0,
    "comparison_tolerance": 0.0,
    "continue_decision": "continue_full_eight",
    "reject_decision": "reject_stop_after_three",
    "selection_stage_is_frozen": True,
    "threshold_or_stage_tuning_allowed": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SCREEN_CONTRACT_SHA256 = _json_sha256(SCREEN_CONTRACT)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_candidate_yaml(path: Path, declared_sha256: object) -> None:
    if not path.is_file() or declared_sha256 != _sha256(path):
        raise ValueError("candidate YAML path/SHA mismatch")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate YAML must be a mapping")
    diagnostic = payload.get("diagnostic_screen")
    if (
        payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("parent_candidate") != "registered-region-v2"
        or not isinstance(diagnostic, dict)
        or diagnostic.get("scenes") != list(SCREEN_SCENES)
        or diagnostic.get("parent_v2_propagated_iou")
        != {
            **PARENT_V2_PROPAGATED_IOU,
            "macro": PARENT_V2_MACRO,
        }
    ):
        raise ValueError("candidate YAML continuation constants drifted")
    rule = str(diagnostic.get("continuation_rule", ""))
    for required in (
        "strictly greater than v2",
        "at least two of three",
        "no scene is below its v2 value",
        "stop without a full-eight expansion",
    ):
        if required not in rule:
            raise ValueError("candidate YAML continuation rule drifted")


def _validate_run_manifest(
    path: Path,
    *,
    result_root: Path,
    output: Path,
    partial_completion_output: Path | None,
    candidate_contract: Path,
) -> tuple[dict[str, Any], str, str]:
    manifest = _read_json(path, label="run manifest")
    manifest_sha256 = _sha256(path)
    repo = Path(__file__).resolve().parents[2]
    queue_path = Path(str(manifest.get("queue_plan", ""))).resolve()
    benchmark_path = Path(str(manifest.get("benchmark_manifest", ""))).resolve()
    runner = Path(str(manifest.get("runner", ""))).resolve()
    implementation = manifest.get("implementation_sources")
    method_contract = manifest.get("method_contract")
    continuation = manifest.get("continuation_screen")
    screen_relative = (
        "radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py"
    )
    aggregate_relative = "radio_gs/scripts/aggregate_registered_prompt_closeout.py"
    evaluator_relative = "radio_gs/scripts/eval_nvos_gaussian_first.py"
    if (
        manifest.get("schema_version") != 2
        or manifest.get("candidate") != CANDIDATE_ID
        or manifest.get("eligibility") != CANDIDATE_ELIGIBILITY
        or manifest.get("scenes") != list(FULL_SCENES)
        or not isinstance(method_contract, dict)
        or not method_contract
        or not queue_path.is_file()
        or manifest.get("queue_plan_sha256") != _sha256(queue_path)
        or not benchmark_path.is_file()
        or manifest.get("benchmark_manifest_sha256") != _sha256(benchmark_path)
        or not runner.is_file()
        or manifest.get("runner_sha256") != _sha256(runner)
        or not isinstance(implementation, dict)
        or implementation.get(screen_relative) != _sha256(repo / screen_relative)
        or implementation.get(aggregate_relative) != _sha256(repo / aggregate_relative)
        or implementation.get(evaluator_relative) != _sha256(repo / evaluator_relative)
        or not isinstance(continuation, dict)
    ):
        raise ValueError("run manifest does not bind the frozen v3 screen")
    queue = _read_json(queue_path, label="queue plan")
    benchmark = _read_json(benchmark_path, label="benchmark manifest")
    if (
        queue.get("benchmark") != "nvos"
        or [str(row.get("scene_id")) for row in queue.get("scenes", [])]
        != list(FULL_SCENES)
        or queue.get("protocol_hash") != benchmark.get("protocol_hash")
    ):
        raise ValueError("run manifest queue/benchmark cohort mismatch")
    screen_path = Path(str(continuation.get("script", ""))).resolve()
    yaml_path = Path(str(continuation.get("candidate_contract", ""))).resolve()
    declared_partial = str(continuation.get("partial_completion_output", ""))
    expected_partial = (
        str(partial_completion_output.resolve())
        if partial_completion_output is not None
        else declared_partial
    )
    if (
        screen_path != Path(__file__).resolve()
        or continuation.get("script_sha256") != _sha256(screen_path)
        or continuation.get("contract") != SCREEN_CONTRACT
        or continuation.get("contract_sha256") != SCREEN_CONTRACT_SHA256
        or Path(str(continuation.get("output", ""))).resolve() != output.resolve()
        or declared_partial != expected_partial
        or yaml_path != candidate_contract.resolve()
    ):
        raise ValueError("run manifest continuation-screen contract mismatch")
    _validate_candidate_yaml(yaml_path, continuation.get("candidate_contract_sha256"))
    if result_root.resolve() != path.parent.resolve():
        raise ValueError("result root must be the immutable run-manifest directory")
    return manifest, manifest_sha256, _json_sha256(method_contract)


def _validate_result(
    path: Path,
    *,
    scene: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    method_contract_sha256: str,
) -> tuple[dict[str, Any], tuple[str, str, str]]:
    result = _read_json(path, label=f"{scene} result")
    hashes = _validate_strict_result(
        result,
        expected_scene_id=scene,
        expected_candidate_id=CANDIDATE_ID,
        expected_candidate_eligibility=CANDIDATE_ELIGIBILITY,
        expected_candidate_method_contract_sha256=method_contract_sha256,
    )
    method_contract = result["method_contract"]
    method_implementation = method_contract.get("implementation_sha256")
    implementation = manifest["implementation_sources"]
    evaluator = str(method_contract.get("evaluator", ""))
    dataset_contract = result["dataset_protocol_contract"]
    queue = _read_json(Path(str(manifest["queue_plan"])), label="queue plan")
    if (
        result.get("run_manifest_sha256") != manifest_sha256
        or method_contract.get("candidate_run_manifest_sha256") != manifest_sha256
        or not isinstance(method_implementation, dict)
        or implementation.get(evaluator) != method_contract.get("evaluator_sha256")
        or any(
            implementation.get(relative) != digest
            for relative, digest in method_implementation.items()
        )
        or manifest.get("radio_checkpoint_sha256")
        != method_contract.get("radio_checkpoint_sha256")
        or result.get("protocol_hash") != queue.get("protocol_hash")
        or result.get("legacy_protocol_hash") != queue.get("protocol_hash")
        or dataset_contract.get("legacy_protocol_hash") != queue.get("protocol_hash")
        or dataset_contract.get("benchmark") != "nvos"
        or dataset_contract.get("cohort") != list(FULL_SCENES)
        or dataset_contract.get("benchmark_manifest_sha256")
        != manifest.get("benchmark_manifest_sha256")
    ):
        raise ValueError(f"{scene}: result/run-manifest provenance mismatch")
    safety = result.get("safety", {})
    for key in (
        "target_ground_truth_opened_before_prediction_write",
        "target_rgb_opened",
        "target_camera_used_as_support",
        "test_calibration",
    ):
        if safety.get(key) is not False:
            raise ValueError(f"{scene}: unsafe result declaration {key}")
    return result, hashes


def _write_immutable_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path, label=f"existing {path.name}") != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def screen(
    *,
    result_root: Path,
    run_manifest: Path,
    candidate_contract: Path,
    output: Path,
    partial_completion_output: Path | None = None,
) -> dict[str, Any]:
    result_root = result_root.resolve()
    run_manifest = run_manifest.resolve()
    candidate_contract = candidate_contract.resolve()
    output = output.resolve()
    partial_completion_output = (
        partial_completion_output.resolve()
        if partial_completion_output is not None
        else None
    )
    manifest, manifest_sha256, method_contract_sha256 = _validate_run_manifest(
        run_manifest,
        result_root=result_root,
        output=output,
        partial_completion_output=partial_completion_output,
        candidate_contract=candidate_contract,
    )
    prior_screen = (
        _read_json(output, label="existing three-scene screen")
        if output.is_file()
        else None
    )
    unexpected = []
    for scene in REMAINING_SCENES:
        path = (
            result_root
            / scene
            / "eval_full_mask_random_walker"
            / f"{scene}_evaluation.json"
        )
        if path.exists():
            unexpected.append(scene)
    if unexpected and prior_screen is None:
        raise ValueError(
            "continuation screen must precede remaining scenes: "
            + ", ".join(unexpected)
        )
    if (
        unexpected
        and prior_screen.get("decision") != SCREEN_CONTRACT["continue_decision"]
    ):
        raise ValueError("remaining scenes exist without a prior continue decision")

    rows: list[dict[str, Any]] = []
    v3_values: list[float] = []
    wins = 0
    regressions = 0
    method_hashes: set[str] = set()
    evaluation_hashes: set[str] = set()
    dataset_hashes: set[str] = set()
    for scene in SCREEN_SCENES:
        result_path = (
            result_root
            / scene
            / "eval_full_mask_random_walker"
            / f"{scene}_evaluation.json"
        )
        if not result_path.is_file():
            raise ValueError(f"missing continuation-screen scene: {scene}")
        result, hashes = _validate_result(
            result_path,
            scene=scene,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            method_contract_sha256=method_contract_sha256,
        )
        method_hashes.add(hashes[0])
        evaluation_hashes.add(hashes[1])
        dataset_hashes.add(hashes[2])
        value = float(result["stage_metrics"]["propagated"]["foreground_iou"])
        parent = PARENT_V2_PROPAGATED_IOU[scene]
        if value > parent:
            outcome = "strict_win"
            wins += 1
        elif value < parent:
            outcome = "regression"
            regressions += 1
        else:
            outcome = "tie"
        v3_values.append(value)
        rows.append(
            {
                "scene_id": scene,
                "v3_propagated_iou": value,
                "parent_v2_propagated_iou": parent,
                "delta": value - parent,
                "outcome": outcome,
                "result": str(result_path.resolve()),
                "result_sha256": _sha256(result_path),
                "method_config_sha256": hashes[0],
                "evaluation_protocol_sha256": hashes[1],
                "dataset_protocol_sha256": hashes[2],
            }
        )
    if (
        len(method_hashes) != 1
        or len(evaluation_hashes) != 1
        or len(dataset_hashes) != 1
    ):
        raise ValueError("three-scene result contracts are inconsistent")
    v3_macro = math.fsum(v3_values) / len(v3_values)
    macro_strictly_improved = v3_macro > PARENT_V2_MACRO
    enough_wins = wins >= int(SCREEN_CONTRACT["minimum_strict_scene_wins"])
    no_scene_regression = regressions == 0
    should_continue = macro_strictly_improved and enough_wins and no_scene_regression
    decision = (
        str(SCREEN_CONTRACT["continue_decision"])
        if should_continue
        else str(SCREEN_CONTRACT["reject_decision"])
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "screen": SCREEN_CONTRACT["screen"],
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "continue_full_eight": should_continue,
        "run_manifest": str(run_manifest),
        "run_manifest_sha256": manifest_sha256,
        "candidate_contract": str(candidate_contract),
        "candidate_contract_sha256": _sha256(candidate_contract),
        "screen_script": str(Path(__file__).resolve()),
        "screen_script_sha256": _sha256(Path(__file__).resolve()),
        "screen_contract": SCREEN_CONTRACT,
        "screen_contract_sha256": SCREEN_CONTRACT_SHA256,
        "parent_v2_macro": PARENT_V2_MACRO,
        "v3_macro": v3_macro,
        "macro_delta": v3_macro - PARENT_V2_MACRO,
        "strict_scene_wins": wins,
        "scene_regressions": regressions,
        "gates": {
            "macro_strictly_improved": macro_strictly_improved,
            "at_least_two_strict_scene_wins": enough_wins,
            "no_scene_regression": no_scene_regression,
        },
        "method_config_sha256": next(iter(method_hashes)),
        "evaluation_protocol_sha256": next(iter(evaluation_hashes)),
        "dataset_protocol_sha256": next(iter(dataset_hashes)),
        "scenes": rows,
        "remaining_scenes_started_before_decision": False,
    }
    _write_immutable_atomic(output, payload)

    if should_continue:
        if partial_completion_output is not None and partial_completion_output.exists():
            raise ValueError("continue decision conflicts with partial completion")
    elif partial_completion_output is not None:
        partial = {
            "schema_version": 1,
            "completion_status": "diagnostic_rejected_after_three_scene_screen",
            "candidate_id": CANDIDATE_ID,
            "normal_stop": True,
            "completed_scenes": list(SCREEN_SCENES),
            "unrun_scenes": list(REMAINING_SCENES),
            "full_aggregate_written": False,
            "run_manifest": str(run_manifest),
            "run_manifest_sha256": manifest_sha256,
            "three_scene_screen": str(output),
            "three_scene_screen_sha256": _sha256(output),
            "screen_contract_sha256": SCREEN_CONTRACT_SHA256,
            "screen_script_sha256": _sha256(Path(__file__).resolve()),
            "decision": decision,
        }
        _write_immutable_atomic(partial_completion_output, partial)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial-completion-output", type=Path, default=None)
    args = parser.parse_args()
    payload = screen(
        result_root=args.result_root,
        run_manifest=args.run_manifest,
        candidate_contract=args.candidate_contract,
        output=args.output,
        partial_completion_output=args.partial_completion_output,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
