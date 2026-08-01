#!/usr/bin/env python3
"""Aggregate a frozen registered-prompt closeout without changing predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _require_contract_digest(
    result: dict[str, Any],
    *,
    scene_id: str,
    contract_key: str,
    digest_key: str,
) -> str:
    contract = result.get(contract_key)
    if not isinstance(contract, dict):
        raise ValueError(f"{scene_id}: missing or invalid {contract_key}")
    declared = result.get(digest_key)
    if declared is None:
        raise ValueError(f"{scene_id}: missing {digest_key}")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"{scene_id}: invalid {digest_key}")
    actual = _json_sha256(contract)
    if declared != actual:
        raise ValueError(
            f"{scene_id}: {digest_key} does not match {contract_key}"
        )
    return declared


def _unit_metric(value: object, *, scene_id: str, label: str) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{scene_id}: {label} must be numeric") from error
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise ValueError(
            f"{scene_id}: {label} must be finite and in [0, 1], got {value!r}"
        )
    return metric


def _validate_strict_result(
    result: dict[str, Any],
    *,
    expected_scene_id: str,
    expected_candidate_id: str,
    expected_candidate_eligibility: str,
    expected_candidate_method_contract_sha256: str,
) -> tuple[str, str, str]:
    reported_scene_id = result.get("scene_id")
    if reported_scene_id != expected_scene_id:
        raise ValueError(
            f"{expected_scene_id}: scene_id mismatch "
            f"(report has {reported_scene_id!r})"
        )

    safety = result.get("safety")
    if (
        not isinstance(safety, dict)
        or safety.get("frozen_diagnostic_eligible") is not True
    ):
        raise ValueError(
            f"{expected_scene_id}: "
            "safety.frozen_diagnostic_eligible must be true"
        )
    if (
        safety.get("candidate_eligibility")
        != expected_candidate_eligibility
        or safety.get("main_result_eligible") is not False
    ):
        raise ValueError(
            f"{expected_scene_id}: candidate promotion eligibility mismatch"
        )

    shared_solver = result.get("shared_solver")
    if (
        not isinstance(shared_solver, dict)
        or shared_solver.get("registered_readout_stage") != "propagated"
    ):
        raise ValueError(
            f"{expected_scene_id}: "
            "shared_solver.registered_readout_stage must be propagated"
        )

    method_hash = _require_contract_digest(
        result,
        scene_id=expected_scene_id,
        contract_key="method_contract",
        digest_key="method_config_sha256",
    )
    evaluation_protocol_hash = _require_contract_digest(
        result,
        scene_id=expected_scene_id,
        contract_key="evaluation_protocol_contract",
        digest_key="evaluation_protocol_sha256",
    )
    dataset_protocol_hash = _require_contract_digest(
        result,
        scene_id=expected_scene_id,
        contract_key="dataset_protocol_contract",
        digest_key="dataset_protocol_sha256",
    )
    method_contract = result["method_contract"]
    if (
        method_contract.get("candidate_id") != expected_candidate_id
        or method_contract.get("candidate_method_contract_sha256")
        != expected_candidate_method_contract_sha256
        or method_contract.get("candidate_eligibility")
        != expected_candidate_eligibility
    ):
        raise ValueError(
            f"{expected_scene_id}: candidate method declaration mismatch"
        )
    method_solver = method_contract.get("shared_solver")
    if (
        not isinstance(method_solver, dict)
        or method_solver.get("registered_readout_stage") != "propagated"
    ):
        raise ValueError(
            f"{expected_scene_id}: method contract does not bind propagated readout"
        )
    evaluation_contract = result["evaluation_protocol_contract"]
    if (
        evaluation_contract.get("method_config_sha256") != method_hash
        or evaluation_contract.get("dataset_protocol_sha256")
        != dataset_protocol_hash
        or evaluation_contract.get("final_readout") != "propagated"
    ):
        raise ValueError(
            f"{expected_scene_id}: evaluation protocol contract cross-link mismatch"
        )
    pixel_threshold = evaluation_contract.get("pixel_threshold")
    if (
        not isinstance(pixel_threshold, dict)
        or pixel_threshold.get("comparison") != "greater_or_equal"
        or not math.isclose(
            float(pixel_threshold.get("value", math.nan)),
            float(result.get("score_threshold", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            f"{expected_scene_id}: evaluation pixel threshold mismatch"
        )

    final_metrics = {
        name: _unit_metric(
            result.get(name),
            scene_id=expected_scene_id,
            label=name,
        )
        for name in ("foreground_iou", "pixel_accuracy")
    }
    stages = result.get("stage_metrics")
    if not isinstance(stages, dict) or not stages:
        raise ValueError(f"{expected_scene_id}: stage_metrics must be a non-empty dict")
    required_stages = {"unary_prior", "propagated", "connected"}
    if not required_stages.issubset(stages):
        raise ValueError(
            f"{expected_scene_id}: required stage metrics are absent"
        )
    validated_stages: dict[str, dict[str, float]] = {}
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            raise ValueError(
                f"{expected_scene_id}: stage_metrics.{stage_name} must be a dict"
            )
        validated_stages[str(stage_name)] = {
            metric_name: _unit_metric(
                stage.get(metric_name),
                scene_id=expected_scene_id,
                label=f"stage_metrics.{stage_name}.{metric_name}",
            )
            for metric_name in ("foreground_iou", "pixel_accuracy")
        }
    propagated = validated_stages.get("propagated")
    if propagated is None:
        raise ValueError(
            f"{expected_scene_id}: stage_metrics.propagated is required"
        )
    for metric_name, final_value in final_metrics.items():
        if not math.isclose(
            final_value,
            propagated[metric_name],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{expected_scene_id}: final {metric_name} does not match "
                f"stage_metrics.propagated.{metric_name}"
            )

    final_frames = result.get("frames")
    if not isinstance(final_frames, list) or not final_frames:
        raise ValueError(f"{expected_scene_id}: final frame metrics are required")
    frame_ids: list[str] = []
    validated_final_frames: list[dict[str, float]] = []
    for final_frame in final_frames:
        if not isinstance(final_frame, dict):
            raise ValueError(
                f"{expected_scene_id}: frame metric records must be dicts"
            )
        frame_id = str(final_frame.get("frame_id", ""))
        if not frame_id or frame_id in frame_ids:
            raise ValueError(
                f"{expected_scene_id}: final frame IDs are empty or duplicated"
            )
        frame_ids.append(frame_id)
        validated_final_frames.append(
            {
                metric_name: _unit_metric(
                    final_frame.get(metric_name),
                    scene_id=expected_scene_id,
                    label=f"frames.{frame_id}.{metric_name}",
                )
                for metric_name in ("foreground_iou", "pixel_accuracy")
            }
        )
    for metric_name, aggregate in final_metrics.items():
        frame_mean = math.fsum(
            frame[metric_name] for frame in validated_final_frames
        ) / len(validated_final_frames)
        if not math.isclose(
            aggregate,
            frame_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{expected_scene_id}: final {metric_name} does not match "
                "its per-frame mean"
            )

    validated_stage_frames: dict[str, list[dict[str, float]]] = {}
    for stage_name in sorted(required_stages):
        raw_frames = stages[stage_name].get("frames")
        if not isinstance(raw_frames, list) or len(raw_frames) != len(frame_ids):
            raise ValueError(
                f"{expected_scene_id}: stage_metrics.{stage_name}.frames "
                "must match the final frame count"
            )
        stage_frames: list[dict[str, float]] = []
        for index, stage_frame in enumerate(raw_frames):
            if not isinstance(stage_frame, dict):
                raise ValueError(
                    f"{expected_scene_id}: stage frame metric records must be dicts"
                )
            frame_id = str(stage_frame.get("frame_id", ""))
            if frame_id != frame_ids[index]:
                raise ValueError(
                    f"{expected_scene_id}: stage_metrics.{stage_name} "
                    "frame IDs differ from final frames"
                )
            stage_frames.append(
                {
                    metric_name: _unit_metric(
                        stage_frame.get(metric_name),
                        scene_id=expected_scene_id,
                        label=(
                            f"stage_metrics.{stage_name}.frames."
                            f"{frame_id}.{metric_name}"
                        ),
                    )
                    for metric_name in ("foreground_iou", "pixel_accuracy")
                }
            )
        validated_stage_frames[stage_name] = stage_frames
        for metric_name, aggregate in validated_stages[stage_name].items():
            frame_mean = math.fsum(
                frame[metric_name] for frame in stage_frames
            ) / len(stage_frames)
            if not math.isclose(
                aggregate,
                frame_mean,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{expected_scene_id}: stage_metrics.{stage_name}."
                    f"{metric_name} does not match its per-frame mean"
                )

    for final_frame, propagated_frame in zip(
        validated_final_frames,
        validated_stage_frames["propagated"],
    ):
        for metric_name in ("foreground_iou", "pixel_accuracy"):
            if not math.isclose(
                final_frame[metric_name],
                propagated_frame[metric_name],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{expected_scene_id}: final/propagated frame metric mismatch"
                )

    score_paths = result.get("score_paths")
    score_sha256 = result.get("score_sha256")
    stage_score_paths = result.get("stage_score_paths")
    stage_score_sha256 = result.get("stage_score_sha256")
    if not all(
        isinstance(value, dict)
        for value in (
            score_paths,
            score_sha256,
            stage_score_paths,
            stage_score_sha256,
        )
    ):
        raise ValueError(
            f"{expected_scene_id}: score artifact provenance is incomplete"
        )
    if set(score_paths) != set(frame_ids) or set(score_sha256) != set(
        frame_ids
    ):
        raise ValueError(
            f"{expected_scene_id}: final score artifact frame set mismatch"
        )
    if (
        not required_stages.issubset(stage_score_paths)
        or not required_stages.issubset(stage_score_sha256)
    ):
        raise ValueError(
            f"{expected_scene_id}: required stage score artifacts are absent"
        )
    for frame_id in frame_ids:
        final_path = Path(str(score_paths[frame_id])).expanduser().resolve()
        final_digest = score_sha256[frame_id]
        if (
            not isinstance(final_digest, str)
            or len(final_digest) != 64
            or not final_path.is_file()
            or _sha256(final_path) != final_digest
        ):
            raise ValueError(
                f"{expected_scene_id}: final score artifact SHA mismatch"
            )
        for stage_name in required_stages:
            per_stage_paths = stage_score_paths[stage_name]
            per_stage_sha256 = stage_score_sha256[stage_name]
            if (
                not isinstance(per_stage_paths, dict)
                or not isinstance(per_stage_sha256, dict)
                or set(per_stage_paths) != set(frame_ids)
                or set(per_stage_sha256) != set(frame_ids)
            ):
                raise ValueError(
                    f"{expected_scene_id}: {stage_name} artifact frame set mismatch"
                )
            stage_path = Path(
                str(per_stage_paths[frame_id])
            ).expanduser().resolve()
            stage_digest = per_stage_sha256[frame_id]
            if (
                not isinstance(stage_digest, str)
                or len(stage_digest) != 64
                or not stage_path.is_file()
                or _sha256(stage_path) != stage_digest
            ):
                raise ValueError(
                    f"{expected_scene_id}: "
                    f"{stage_name} score artifact SHA mismatch"
                )
            if stage_name == "propagated" and stage_digest != final_digest:
                raise ValueError(
                    f"{expected_scene_id}: "
                    "final and propagated score artifacts differ"
                )

    return method_hash, evaluation_protocol_hash, dataset_protocol_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-plan", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-candidate",
        default="registered-region-v1",
        help="Candidate identifier required in the strict run manifest/reports.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Optional immutable run manifest that every strict report must bind.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--require-method-config",
        action="store_true",
        help=(
            "Require every scene report to carry the same full method-config "
            "digest in addition to the benchmark/data protocol hash."
        ),
    )
    args = parser.parse_args()
    if args.require_method_config and args.run_manifest is None:
        parser.error("--require-method-config requires --run-manifest")

    queue_path = args.queue_plan.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    queue = _read_json(queue_path)
    run_manifest_path = (
        args.run_manifest.expanduser().resolve()
        if args.run_manifest is not None
        else None
    )
    run_manifest_sha256 = (
        _sha256(run_manifest_path)
        if run_manifest_path is not None
        else None
    )
    run_manifest = (
        _read_json(run_manifest_path)
        if run_manifest_path is not None
        else None
    )
    candidate_eligibility: str | None = None
    candidate_method_contract_sha256: str | None = None
    expected = [str(row["scene_id"]) for row in queue["scenes"]]
    protocol_hash = str(queue["protocol_hash"])
    if args.require_method_config:
        assert run_manifest is not None
        benchmark_manifest = Path(
            str(run_manifest.get("benchmark_manifest", ""))
        ).resolve()
        implementation = run_manifest.get("implementation_sources")
        candidate_method_contract = run_manifest.get("method_contract")
        aggregate_relative = (
            "radio_gs/scripts/aggregate_registered_prompt_closeout.py"
        )
        runner = Path(str(run_manifest.get("runner", ""))).resolve()
        if (
            run_manifest.get("candidate") != args.expected_candidate
            or run_manifest.get("eligibility")
            != "diagnostic_until_disjoint_registered_prompt_gate"
            or not isinstance(candidate_method_contract, dict)
            or not candidate_method_contract
            or Path(str(run_manifest.get("queue_plan", ""))).resolve()
            != queue_path
            or run_manifest.get("queue_plan_sha256") != _sha256(queue_path)
            or run_manifest.get("scenes") != expected
            or not benchmark_manifest.is_file()
            or run_manifest.get("benchmark_manifest_sha256")
            != _sha256(benchmark_manifest)
            or _read_json(benchmark_manifest).get("protocol_hash")
            != protocol_hash
            or not isinstance(implementation, dict)
            or implementation.get(aggregate_relative)
            != _sha256(Path(__file__).resolve())
            or not runner.is_file()
            or run_manifest.get("runner_sha256") != _sha256(runner)
        ):
            raise ValueError("run manifest does not match the frozen queue/dataset")
        candidate_eligibility = str(run_manifest["eligibility"])
        candidate_method_contract_sha256 = _json_sha256(
            candidate_method_contract
        )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    method_hashes: set[str] = set()
    missing_method_hashes: list[str] = []
    evaluation_protocol_hashes: set[str] = set()
    dataset_protocol_hashes: set[str] = set()

    for scene_id in expected:
        path = (
            result_root
            / scene_id
            / "eval_full_mask_random_walker"
            / f"{scene_id}_evaluation.json"
        )
        if not path.is_file():
            missing.append(scene_id)
            continue
        result = _read_json(path)
        if result.get("protocol_hash") != protocol_hash:
            raise ValueError(f"{scene_id}: protocol hash mismatch")
        safety = result.get("safety", {})
        forbidden = {
            "target_ground_truth_opened_before_prediction_write": False,
            "target_rgb_opened": False,
            "target_camera_used_as_support": False,
            "test_calibration": False,
        }
        for key, required in forbidden.items():
            if safety.get(key) is not required:
                raise ValueError(f"{scene_id}: unsafe {key}={safety.get(key)!r}")
        if args.require_method_config:
            (
                method_hash,
                evaluation_protocol_hash,
                dataset_protocol_hash,
            ) = _validate_strict_result(
                result,
                expected_scene_id=scene_id,
                expected_candidate_id=str(args.expected_candidate),
                expected_candidate_eligibility=str(candidate_eligibility),
                expected_candidate_method_contract_sha256=str(
                    candidate_method_contract_sha256
                ),
            )
            evaluation_protocol_hashes.add(evaluation_protocol_hash)
            dataset_protocol_hashes.add(dataset_protocol_hash)
            if (
                result.get("run_manifest_sha256") != run_manifest_sha256
                or result["method_contract"].get(
                    "candidate_run_manifest_sha256"
                )
                != run_manifest_sha256
            ):
                raise ValueError(
                    f"{scene_id}: run_manifest_sha256 mismatch"
                )
            assert run_manifest is not None
            implementation = run_manifest.get("implementation_sources")
            method_contract = result["method_contract"]
            evaluator = str(method_contract.get("evaluator", ""))
            method_implementation = method_contract.get(
                "implementation_sha256"
            )
            if (
                not isinstance(implementation, dict)
                or not isinstance(method_implementation, dict)
                or implementation.get(evaluator)
                != method_contract.get("evaluator_sha256")
                or any(
                    implementation.get(relative) != digest
                    for relative, digest in method_implementation.items()
                )
                or run_manifest.get("radio_checkpoint_sha256")
                != method_contract.get("radio_checkpoint_sha256")
            ):
                raise ValueError(
                    f"{scene_id}: method/run-manifest provenance mismatch"
                )
            dataset_contract = result["dataset_protocol_contract"]
            if (
                result.get("legacy_protocol_hash") != protocol_hash
                or dataset_contract.get("legacy_protocol_hash")
                != protocol_hash
                or dataset_contract.get("benchmark") != queue.get("benchmark")
                or dataset_contract.get("cohort") != expected
                or dataset_contract.get("benchmark_manifest_sha256")
                != run_manifest.get("benchmark_manifest_sha256")
            ):
                raise ValueError(
                    f"{scene_id}: dataset/run-manifest provenance mismatch"
                )
        else:
            raw_method_hash = result.get("method_config_sha256")
            method_hash = (
                raw_method_hash.strip()
                if isinstance(raw_method_hash, str)
                else ""
            )
        if method_hash:
            method_hashes.add(method_hash)
        else:
            missing_method_hashes.append(scene_id)
        stages = result["stage_metrics"]
        rows.append(
            {
                "scene_id": scene_id,
                "foreground_iou": float(result["foreground_iou"]),
                "pixel_accuracy": float(result["pixel_accuracy"]),
                "unary_iou": float(stages["unary_prior"]["foreground_iou"]),
                "propagated_iou": float(stages["propagated"]["foreground_iou"]),
                "connected_iou": float(stages["connected"]["foreground_iou"]),
                "method_config_sha256": method_hash or None,
                "result": str(path),
                "result_sha256": _sha256(path),
            }
        )

    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} scenes: {', '.join(missing)}")
    if args.require_method_config and missing_method_hashes:
        raise ValueError(
            "missing method_config_sha256 for scenes: "
            + ", ".join(missing_method_hashes)
        )
    if method_hashes and missing_method_hashes:
        raise ValueError(
            "cannot mix method-digested and legacy scene reports: "
            + ", ".join(missing_method_hashes)
        )
    if len(method_hashes) > 1:
        raise ValueError(
            "registered-prompt scene reports use multiple method configurations: "
            + ", ".join(sorted(method_hashes))
        )
    if len(evaluation_protocol_hashes) > 1:
        raise ValueError(
            "registered-prompt scene reports use multiple evaluation protocols: "
            + ", ".join(sorted(evaluation_protocol_hashes))
        )
    if len(dataset_protocol_hashes) > 1:
        raise ValueError(
            "registered-prompt scene reports use multiple dataset protocols: "
            + ", ".join(sorted(dataset_protocol_hashes))
        )
    summary = {
        "schema_version": 2 if args.require_method_config else 1,
        "contract_validation": (
            "strict" if args.require_method_config else "legacy_unverified"
        ),
        "benchmark": queue.get("benchmark"),
        "candidate": (
            str(args.expected_candidate)
            if args.require_method_config
            else None
        ),
        "protocol_hash": protocol_hash,
        "method_config_sha256": (
            next(iter(method_hashes)) if method_hashes else None
        ),
        "evaluation_protocol_sha256": (
            next(iter(evaluation_protocol_hashes))
            if evaluation_protocol_hashes
            else None
        ),
        "dataset_protocol_sha256": (
            next(iter(dataset_protocol_hashes))
            if dataset_protocol_hashes
            else None
        ),
        "queue_plan": str(queue_path),
        "queue_plan_sha256": _sha256(queue_path),
        "run_manifest": (
            str(run_manifest_path) if run_manifest_path is not None else None
        ),
        "run_manifest_sha256": run_manifest_sha256,
        "candidate_method_contract_sha256": (
            candidate_method_contract_sha256
            if args.require_method_config
            else None
        ),
        "candidate_eligibility": (
            candidate_eligibility
            if args.require_method_config
            else "legacy_unverified"
        ),
        "frozen_diagnostic_eligible": (
            True if args.require_method_config else None
        ),
        "main_result_eligible": (
            False if args.require_method_config else None
        ),
        "score_artifact_validation": (
            "sha256_and_propagated_identity"
            if args.require_method_config
            else "legacy_unverified"
        ),
        "expected_scene_count": len(expected),
        "completed_scene_count": len(rows),
        "complete": not missing,
        "missing_scenes": missing,
        "macro": {
            "foreground_iou": _mean([row["foreground_iou"] for row in rows]),
            "pixel_accuracy": _mean([row["pixel_accuracy"] for row in rows]),
            "unary_iou": _mean([row["unary_iou"] for row in rows]),
            "propagated_iou": _mean([row["propagated_iou"] for row in rows]),
            "connected_iou": _mean([row["connected_iou"] for row in rows]),
        },
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
