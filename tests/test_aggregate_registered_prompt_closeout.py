from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from radio_gs.scripts.aggregate_registered_prompt_closeout import main


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result(
    root: Path,
    scene_id: str,
    *,
    method_hash: str | None = "auto",
    evaluation_protocol_hash: str | None = "auto",
    dataset_protocol_hash: str | None = "auto",
    frozen_diagnostic_eligible: bool = True,
    main_result_eligible: bool = False,
    final_foreground_iou: float = 0.6,
    contract_overrides: dict[str, dict[str, Any]] | None = None,
) -> Path:
    path = (
        root
        / scene_id
        / "eval_full_mask_random_walker"
        / f"{scene_id}_evaluation.json"
    )
    path.parent.mkdir(parents=True)
    contracts: dict[str, dict[str, Any]] = {
        "method_contract": {
            "schema_version": 1,
            "method": "registered-region-v1",
            "candidate_id": "registered-region-v1",
            "evaluator": "eval.py",
            "evaluator_sha256": "eval-sha",
            "implementation_sha256": {},
            "radio_checkpoint_sha256": "radio-sha",
            "candidate_run_manifest_sha256": "",
            "candidate_method_contract_sha256": "",
            "candidate_eligibility": (
                "diagnostic_until_disjoint_registered_prompt_gate"
            ),
            "shared_solver": {
                "registered_readout_stage": "propagated",
            },
        },
        "dataset_protocol_contract": {
            "schema_version": 1,
            "benchmark": "nvos",
            "protocol_hash": "protocol",
            "legacy_protocol_hash": "protocol",
            "benchmark_manifest_sha256": "",
            "cohort": [],
        },
    }
    for key, value in (contract_overrides or {}).items():
        if key != "evaluation_protocol_contract":
            contracts[key] = value
    default_evaluation_contract = {
        "schema_version": 1,
        "final_readout": "propagated",
        "method_config_sha256": _json_sha256(
            contracts["method_contract"]
        ),
        "dataset_protocol_sha256": _json_sha256(
            contracts["dataset_protocol_contract"]
        ),
        "pixel_threshold": {
            "value": 0.5,
            "comparison": "greater_or_equal",
        },
    }
    contracts["evaluation_protocol_contract"] = default_evaluation_contract
    for key, value in (contract_overrides or {}).items():
        if key == "evaluation_protocol_contract":
            contracts[key] = value
    frame_metrics = [
        {
            "frame_id": "frame0",
            "foreground_iou": 0.6,
            "pixel_accuracy": 0.9,
        }
    ]
    artifact_root = path.parent / "artifacts"
    artifact_root.mkdir()
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
    payload = {
        "scene_id": scene_id,
        "protocol_hash": "protocol",
        "legacy_protocol_hash": "protocol",
        "foreground_iou": final_foreground_iou,
        "pixel_accuracy": 0.9,
        "score_threshold": 0.5,
        "stage_metrics": {
            name: {
                "foreground_iou": value,
                "pixel_accuracy": 0.9,
                "frames": (
                    frame_metrics
                    if name == "propagated"
                    else [
                        {
                            "frame_id": "frame0",
                            "foreground_iou": value,
                            "pixel_accuracy": 0.9,
                        }
                    ]
                ),
            }
            for name, value in {
                "unary_prior": 0.5,
                "propagated": 0.6,
                "connected": 0.4,
            }.items()
        },
        "shared_solver": {
            "registered_readout_stage": "propagated",
        },
        "frames": frame_metrics,
        "score_paths": {"frame0": str(final_score.resolve())},
        "score_sha256": {"frame0": _file_sha256(final_score)},
        "stage_score_paths": {
            stage: {"frame0": str(stage_path.resolve())}
            for stage, stage_path in stage_scores.items()
        },
        "stage_score_sha256": {
            stage: {"frame0": _file_sha256(stage_path)}
            for stage, stage_path in stage_scores.items()
        },
        "safety": {
            "target_ground_truth_opened_before_prediction_write": False,
            "target_rgb_opened": False,
            "target_camera_used_as_support": False,
            "test_calibration": False,
            "candidate_eligibility": (
                "diagnostic_until_disjoint_registered_prompt_gate"
            ),
            "frozen_diagnostic_eligible": frozen_diagnostic_eligible,
            "main_result_eligible": main_result_eligible,
        },
        **contracts,
    }
    if method_hash is not None:
        payload["method_config_sha256"] = (
            _json_sha256(contracts["method_contract"])
            if method_hash == "auto"
            else method_hash
        )
    if evaluation_protocol_hash is not None:
        payload["evaluation_protocol_sha256"] = (
            _json_sha256(contracts["evaluation_protocol_contract"])
            if evaluation_protocol_hash == "auto"
            else evaluation_protocol_hash
        )
    if dataset_protocol_hash is not None:
        payload["dataset_protocol_sha256"] = (
            _json_sha256(contracts["dataset_protocol_contract"])
            if dataset_protocol_hash == "auto"
            else dataset_protocol_hash
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(
    monkeypatch: pytest.MonkeyPatch,
    queue: Path,
    root: Path,
    output: Path,
    *,
    require_method: bool,
    preserve_evaluation_crosslink: bool = False,
    run_manifest_candidate: str = "registered-region-v1",
    candidate_method_digest_override: str | None = None,
) -> None:
    argv = [
        "aggregate_registered_prompt_closeout.py",
        "--queue-plan",
        str(queue),
        "--result-root",
        str(root),
        "--output",
        str(output),
    ]
    if require_method:
        queue_payload = json.loads(queue.read_text(encoding="utf-8"))
        benchmark_manifest = queue.parent / "benchmark_manifest.json"
        benchmark_manifest.write_text(
            json.dumps(
                {
                    "benchmark": queue_payload.get("benchmark"),
                    "protocol_hash": queue_payload.get("protocol_hash"),
                }
            ),
            encoding="utf-8",
        )
        runner = queue.parent / "runner.sh"
        runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        aggregate_path = Path(main.__code__.co_filename).resolve()
        run_manifest = queue.parent / "run_manifest.json"
        run_payload = {
            "schema_version": 1,
            "candidate": run_manifest_candidate,
            "eligibility": (
                "diagnostic_until_disjoint_registered_prompt_gate"
            ),
            "method_contract": {
                "method": "registered-region-v1",
                "final_readout": "propagated",
            },
            "scenes": [
                str(row["scene_id"]) for row in queue_payload["scenes"]
            ],
            "queue_plan": str(queue.resolve()),
            "queue_plan_sha256": _file_sha256(queue),
            "benchmark_manifest": str(benchmark_manifest.resolve()),
            "benchmark_manifest_sha256": _file_sha256(benchmark_manifest),
            "radio_checkpoint_sha256": "radio-sha",
            "implementation_sources": {
                "eval.py": "eval-sha",
                "radio_gs/scripts/aggregate_registered_prompt_closeout.py": (
                    _file_sha256(aggregate_path)
                ),
            },
            "runner": str(runner.resolve()),
            "runner_sha256": _file_sha256(runner),
        }
        run_manifest.write_text(json.dumps(run_payload), encoding="utf-8")
        run_sha256 = _file_sha256(run_manifest)
        for result_path in root.glob(
            "*/eval_full_mask_random_walker/*_evaluation.json"
        ):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            validity = {
                digest_key: payload.get(digest_key)
                == _json_sha256(payload[contract_key])
                for contract_key, digest_key in (
                    ("method_contract", "method_config_sha256"),
                    (
                        "evaluation_protocol_contract",
                        "evaluation_protocol_sha256",
                    ),
                    (
                        "dataset_protocol_contract",
                        "dataset_protocol_sha256",
                    ),
                )
            }
            payload["run_manifest_sha256"] = run_sha256
            payload["method_contract"][
                "candidate_run_manifest_sha256"
            ] = run_sha256
            payload["method_contract"][
                "candidate_method_contract_sha256"
            ] = (
                candidate_method_digest_override
                or _json_sha256(run_payload["method_contract"])
            )
            payload["method_contract"]["candidate_eligibility"] = (
                run_payload["eligibility"]
            )
            payload["safety"]["candidate_eligibility"] = run_payload[
                "eligibility"
            ]
            method_sha256 = _json_sha256(payload["method_contract"])
            dataset_contract = payload["dataset_protocol_contract"]
            dataset_contract.update(
                {
                    "legacy_protocol_hash": queue_payload["protocol_hash"],
                    "benchmark": queue_payload["benchmark"],
                    "benchmark_manifest_sha256": _file_sha256(
                        benchmark_manifest
                    ),
                    "cohort": [
                        str(row["scene_id"])
                        for row in queue_payload["scenes"]
                    ],
                }
            )
            dataset_sha256 = _json_sha256(dataset_contract)
            evaluation_contract = payload[
                "evaluation_protocol_contract"
            ]
            if not preserve_evaluation_crosslink:
                evaluation_contract["method_config_sha256"] = method_sha256
                evaluation_contract[
                    "dataset_protocol_sha256"
                ] = dataset_sha256
            payload["legacy_protocol_hash"] = queue_payload["protocol_hash"]
            for contract_key, digest_key, digest in (
                (
                    "method_contract",
                    "method_config_sha256",
                    method_sha256,
                ),
                (
                    "dataset_protocol_contract",
                    "dataset_protocol_sha256",
                    dataset_sha256,
                ),
                (
                    "evaluation_protocol_contract",
                    "evaluation_protocol_sha256",
                    _json_sha256(evaluation_contract),
                ),
            ):
                if validity[digest_key]:
                    payload[digest_key] = digest
            result_path.write_text(json.dumps(payload), encoding="utf-8")
        argv.extend(["--run-manifest", str(run_manifest)])
        argv.append("--require-method-config")
    monkeypatch.setattr(sys, "argv", argv)
    main()


def test_aggregate_binds_one_method_digest_across_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}, {"scene_id": "b"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a")
    _write_result(root, "b")
    output = tmp_path / "summary.json"

    _run(monkeypatch, queue, root, output, require_method=True)

    summary = json.loads(output.read_text(encoding="utf-8"))
    first_report = json.loads(
        (
            root / "a" / "eval_full_mask_random_walker" / "a_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == 2
    assert summary["contract_validation"] == "strict"
    assert summary["candidate"] == "registered-region-v1"
    assert summary["candidate_eligibility"] == (
        "diagnostic_until_disjoint_registered_prompt_gate"
    )
    assert summary["frozen_diagnostic_eligible"] is True
    assert summary["main_result_eligible"] is False
    assert (
        summary["score_artifact_validation"]
        == "sha256_and_propagated_identity"
    )
    assert (
        summary["method_config_sha256"]
        == first_report["method_config_sha256"]
    )
    assert (
        summary["evaluation_protocol_sha256"]
        == first_report["evaluation_protocol_sha256"]
    )
    assert (
        summary["dataset_protocol_sha256"]
        == first_report["dataset_protocol_sha256"]
    )


def test_aggregate_rejects_mixed_or_missing_method_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}, {"scene_id": "b"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a")
    _write_result(root, "b", method_hash=None)

    with pytest.raises(ValueError, match="missing method_config_sha256"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


@pytest.mark.parametrize(
    ("hash_name", "error_match"),
    [
        ("method_hash", "method_config_sha256 does not match method_contract"),
        (
            "evaluation_protocol_hash",
            "evaluation_protocol_sha256 does not match evaluation_protocol_contract",
        ),
        (
            "dataset_protocol_hash",
            "dataset_protocol_sha256 does not match dataset_protocol_contract",
        ),
    ],
)
def test_aggregate_rejects_forged_contract_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hash_name: str,
    error_match: str,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a", **{hash_name: "forged"})

    with pytest.raises(ValueError, match=error_match):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


def test_aggregate_rejects_ineligible_main_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a", frozen_diagnostic_eligible=False)

    with pytest.raises(
        ValueError,
        match="frozen_diagnostic_eligible must be true",
    ):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


def test_aggregate_rejects_wrong_run_manifest_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a")

    with pytest.raises(ValueError, match="run manifest does not match"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
            run_manifest_candidate="another-candidate",
        )


def test_aggregate_rejects_candidate_method_declaration_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a")

    with pytest.raises(ValueError, match="candidate method declaration"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
            candidate_method_digest_override="forged",
        )


def test_aggregate_rejects_modified_score_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    result_path = _write_result(root, "a")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    Path(result["score_paths"]["frame0"]).write_bytes(b"modified")

    with pytest.raises(ValueError, match="score artifact SHA mismatch"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


def test_aggregate_rejects_final_metric_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(root, "a", final_foreground_iou=0.61)

    with pytest.raises(ValueError, match="final foreground_iou does not match"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


def test_aggregate_rejects_joint_final_and_propagated_aggregate_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    result_path = _write_result(root, "a")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["foreground_iou"] = 0.61
    result["stage_metrics"]["propagated"]["foreground_iou"] = 0.61
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="per-frame mean"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


@pytest.mark.parametrize("stage_name", ["unary_prior", "connected"])
def test_aggregate_rejects_diagnostic_stage_aggregate_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    result_path = _write_result(root, "a")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["stage_metrics"][stage_name]["foreground_iou"] = 0.55
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="per-frame mean"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


def test_aggregate_rejects_diagnostic_stage_frame_set_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    result_path = _write_result(root, "a")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["stage_metrics"]["connected"]["frames"][0]["frame_id"] = (
        "forged"
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="frame IDs differ"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
        )


def test_aggregate_rejects_evaluation_contract_cross_link_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": "protocol",
                "scenes": [{"scene_id": "a"}],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "results"
    _write_result(
        root,
        "a",
        contract_overrides={
            "evaluation_protocol_contract": {
                "schema_version": 1,
                "final_readout": "propagated",
                "method_config_sha256": "another-method",
                "dataset_protocol_sha256": "another-dataset",
                "pixel_threshold": {
                    "value": 0.5,
                    "comparison": "greater_or_equal",
                },
            }
        },
    )

    with pytest.raises(ValueError, match="cross-link mismatch"):
        _run(
            monkeypatch,
            queue,
            root,
            tmp_path / "summary.json",
            require_method=True,
            preserve_evaluation_crosslink=True,
        )
