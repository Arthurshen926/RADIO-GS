from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import radio_gs.scripts.aggregate_nvos_forward_beta_full8_nonexact as aggregator_module
from radio_gs.scripts.aggregate_nvos_forward_beta_full8_nonexact import (
    ARTIFACT_TYPE,
    ELIGIBILITY,
    EXPECTED_SCORING_CONTRACT,
    FIXED_BLOCKERS,
    ForwardBetaAggregationError,
    aggregate_forward_beta_full8,
    main,
)
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    STRICT_TASKS,
    build_authority,
    canonical_json_sha256,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_score(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _forward_contract() -> dict[str, object]:
    return {
        "mode": "beta_coverage_v1",
        "status": ELIGIBILITY,
        "strict_unseen_eligible": False,
        "selection_applied_to_main_output": False,
        "required_final_readout": "propagated",
        "scoring_adapter": deepcopy(EXPECTED_SCORING_CONTRACT),
    }


@pytest.fixture(autouse=True)
def _synthetic_scene_receipt_validator(monkeypatch):
    def validate(path, *, run_manifest, scene, result):
        receipt_path = Path(path).resolve()
        receipt = _load(receipt_path)
        assert receipt["scene"] == scene
        assert Path(receipt["run_manifest"]).resolve() == Path(run_manifest).resolve()
        result_path = Path(receipt["result"]).resolve()
        assert result_path == Path(result).resolve()
        return {
            "receipt": {
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
            },
            "payload": {
                "status": (
                    "beta_report_nonpromoted_cuda_attested_exclusive_owner_"
                    "postchecked"
                ),
                "result": {
                    "path": str(result_path),
                    "sha256": _sha256(result_path),
                },
                "gpu_identity": {"uuid": "GPU-synthetic"},
                "owner_audit": {"child_owner_pids": ["123"]},
                "cuda_attestation": {"sha256": "a" * 64},
                "postcheck": {"sha256": "b" * 64},
                "promotion": {
                    "main_result_eligible": False,
                    "frozen_diagnostic_eligible": False,
                    "strict_unseen_protocol_exact_match": False,
                },
            },
        }

    monkeypatch.setattr(aggregator_module, "validate_scene_receipt", validate)


def _make_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    method_contract = {
        "support_mode": "canonical_support",
        "final_readout": "propagated",
        "selection_applied_to_main_output": False,
        "registered_forward_unary": _forward_contract(),
    }
    method_sha = canonical_json_sha256(method_contract)
    authority = build_authority(
        candidate_method_sha256=method_sha,
        scoring_contract=EXPECTED_SCORING_CONTRACT,
        repo_root=Path(__file__).resolve().parents[1],
    )
    authority_sha = canonical_json_sha256(authority)
    manifest = {
        "schema_version": 1,
        "candidate": "forward-beta-v1",
        "eligibility": ELIGIBILITY,
        "scenes": list(STRICT_TASKS),
        "method_contract": method_contract,
        "registered_forward_protocol_authority": authority,
        "registered_forward_protocol_authority_sha256": authority_sha,
    }
    manifest_path = tmp_path / "run_manifest.json"
    _write(manifest_path, manifest)
    manifest_sha = _sha256(manifest_path)

    result_root = tmp_path / "results"
    receipt_root = tmp_path / "receipts"
    reports: dict[str, Path] = {}
    dataset_sha = "d" * 64
    for index, scene in enumerate(STRICT_TASKS):
        frame_id = f"{scene}-frame"
        report_dir = result_root / scene / "eval_full_mask_random_walker"
        main_score_path = report_dir / "scores" / scene / f"{frame_id}.npy"
        stage_score_paths_on_disk = {
            stage: report_dir / "stage_scores" / stage / scene / f"{frame_id}.npy"
            for stage in ("unary_prior", "propagated", "connected")
        }
        propagated_payload = f"propagated-score-{scene}".encode("utf-8")
        _write_score(main_score_path, propagated_payload)
        _write_score(stage_score_paths_on_disk["propagated"], propagated_payload)
        _write_score(
            stage_score_paths_on_disk["unary_prior"],
            f"unary-score-{scene}".encode("utf-8"),
        )
        _write_score(
            stage_score_paths_on_disk["connected"],
            f"connected-score-{scene}".encode("utf-8"),
        )
        propagated_iou = 0.1 * (index + 1)
        propagated_accuracy = 0.5 + 0.01 * index
        unary_iou = max(0.0, propagated_iou - 0.02)
        connected_iou = min(1.0, propagated_iou + 0.01)
        frames = [
            {
                "frame_id": frame_id,
                "foreground_iou": propagated_iou,
                "pixel_accuracy": propagated_accuracy,
            }
        ]
        stages = {
            "unary_prior": {
                "foreground_iou": unary_iou,
                "pixel_accuracy": propagated_accuracy - 0.01,
                "frames": [
                    {
                        "frame_id": frame_id,
                        "foreground_iou": unary_iou,
                        "pixel_accuracy": propagated_accuracy - 0.01,
                    }
                ],
            },
            "propagated": {
                "foreground_iou": propagated_iou,
                "pixel_accuracy": propagated_accuracy,
                "frames": frames,
            },
            "connected": {
                "foreground_iou": connected_iou,
                "pixel_accuracy": propagated_accuracy,
                "frames": [
                    {
                        "frame_id": frame_id,
                        "foreground_iou": connected_iou,
                        "pixel_accuracy": propagated_accuracy,
                    }
                ],
            },
        }
        solver = {
            "registered_readout_stage": "propagated",
            "registered_forward_unary": _forward_contract(),
        }
        report_method = {
            "candidate_run_manifest_sha256": manifest_sha,
            "candidate_method_contract_sha256": method_sha,
            "candidate_eligibility": ELIGIBILITY,
            "registered_forward_protocol_authority": authority,
            "registered_forward_protocol_authority_sha256": authority_sha,
            "shared_solver": solver,
        }
        report_method_sha = canonical_json_sha256(report_method)
        evaluation = {
            "method_config_sha256": report_method_sha,
            "dataset_protocol_sha256": dataset_sha,
            "final_readout": "propagated",
            "registered_forward_protocol_authority_sha256": authority_sha,
            "strict_unseen_protocol_exact_match": False,
            "score_semantics": EXPECTED_SCORING_CONTRACT["score_semantics"],
            "prediction_representation": EXPECTED_SCORING_CONTRACT[
                "prediction_representation"
            ],
            "pixel_threshold": EXPECTED_SCORING_CONTRACT["threshold"],
            "resize_to_ground_truth": "cv2.INTER_NEAREST",
        }
        score_paths = {frame_id: str(main_score_path.resolve())}
        score_hashes = {frame_id: _sha256(main_score_path)}
        report = {
            "scene_id": scene,
            "method": "gaussian_first_beta_centered_posterior",
            "registered_forward_protocol_authority": authority,
            "registered_forward_protocol_authority_sha256": authority_sha,
            "run_manifest_sha256": manifest_sha,
            "method_contract": report_method,
            "method_config_sha256": report_method_sha,
            "shared_solver": solver,
            "dataset_protocol_sha256": dataset_sha,
            "evaluation_protocol_contract": evaluation,
            "evaluation_protocol_sha256": canonical_json_sha256(evaluation),
            "foreground_iou": propagated_iou,
            "pixel_accuracy": propagated_accuracy,
            "frames": frames,
            "stage_metrics": stages,
            "score_paths": score_paths,
            "score_sha256": score_hashes,
            "stage_score_paths": {
                stage: {frame_id: str(path.resolve())}
                for stage, path in stage_score_paths_on_disk.items()
            },
            "stage_score_sha256": {
                stage: {frame_id: _sha256(path)}
                for stage, path in stage_score_paths_on_disk.items()
            },
            "safety": {
                "candidate_eligibility": ELIGIBILITY,
                "frozen_diagnostic_eligible": False,
                "main_result_eligible": False,
                "strict_unseen_eligible": False,
                "strict_unseen_protocol_exact_match": False,
                "registered_forward_protocol_authority_sha256": authority_sha,
                "target_ground_truth_opened_before_prediction_write": False,
                "target_rgb_opened": False,
                "target_camera_used_as_support": False,
                "test_calibration": False,
            },
        }
        path = report_dir / f"{scene}_evaluation.json"
        _write(path, report)
        reports[scene] = path
        _write(
            receipt_root / scene / "scene_receipt.json",
            {
                "scene": scene,
                "run_manifest": str(manifest_path.resolve()),
                "result": str(path.resolve()),
            },
        )
    return manifest_path, result_root, receipt_root, reports


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_aggregates_exact_full8_with_equal_scene_macro_and_safe_labels(
    tmp_path: Path,
) -> None:
    manifest, root, receipts, _reports = _make_fixture(tmp_path)

    summary = aggregate_forward_beta_full8(
        run_manifest_path=manifest, result_root=root, receipt_root=receipts
    )

    assert summary["artifact_type"] == ARTIFACT_TYPE
    assert summary["status"] == ELIGIBILITY
    assert summary["strict_unseen_protocol_exact_match"] is False
    assert summary["strict_unseen_exact_match_blockers"] == FIXED_BLOCKERS
    assert summary["frozen_diagnostic_eligible"] is False
    assert summary["main_result_eligible"] is False
    assert summary["expected_tasks"] == list(STRICT_TASKS)
    assert summary["completed_task_count"] == 8
    assert summary["aggregation"]["weighting"] == "equal_weight"
    assert summary["main_output"]["stage"] == "propagated"
    assert summary["main_output"]["macro"]["foreground_iou"] == (pytest.approx(0.45))
    assert summary["diagnostics"]["connected_macro"]["role"] == (
        "diagnostic_only_not_applied_to_main_output"
    )
    assert all(
        row["diagnostics"]["connected"]["role"]
        == "diagnostic_only_not_applied_to_main_output"
        for row in summary["tasks"]
    )
    assert summary["authority_binding"][
        "payload_and_sha_match_run_manifest_and_all_eight_reports"
    ]
    assert summary["scene_gpu_authority"]["validated_receipt_count"] == 8
    assert summary["scene_gpu_authority"][
        "gpu_owner_attestation_postcheck_chain_validated"
    ]
    assert all("scene_gpu_authority" in row for row in summary["tasks"])
    assert summary["external_comparator_fence"]["binding_role"] == (
        "external_method_comparator_only"
    )
    assert set(summary["external_comparator_fence"]["candidate_binding"].values()) == {
        None
    }


def test_evaluator_main_and_propagated_copies_bind_by_content_not_path(
    tmp_path: Path,
) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["fern"])
    frame_id = report["frames"][0]["frame_id"]
    main_path = Path(report["score_paths"][frame_id])
    propagated_path = Path(
        report["stage_score_paths"]["propagated"][frame_id]
    )

    assert main_path != propagated_path
    assert main_path.read_bytes() == propagated_path.read_bytes()
    assert report["score_sha256"][frame_id] == (
        report["stage_score_sha256"]["propagated"][frame_id]
    )
    assert aggregate_forward_beta_full8(
        run_manifest_path=manifest,
        result_root=root,
        receipt_root=receipts,
    )["complete"]


def test_propagated_score_content_tamper_fails_closed(tmp_path: Path) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["fern"])
    frame_id = report["frames"][0]["frame_id"]
    propagated_path = Path(
        report["stage_score_paths"]["propagated"][frame_id]
    )
    propagated_path.write_bytes(b"tampered-propagated-score")

    with pytest.raises(
        ForwardBetaAggregationError,
        match=r"fern: propagated/fern-frame score artifact SHA256 differs",
    ):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_self_consistent_propagated_tamper_still_cannot_replace_main(
    tmp_path: Path,
) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["fern"])
    frame_id = report["frames"][0]["frame_id"]
    propagated_path = Path(
        report["stage_score_paths"]["propagated"][frame_id]
    )
    propagated_path.write_bytes(b"tampered-propagated-score")
    report["stage_score_sha256"]["propagated"][frame_id] = _sha256(
        propagated_path
    )
    _write(reports["fern"], report)

    with pytest.raises(
        ForwardBetaAggregationError,
        match=r"propagated/fern-frame score content is not the main artifact content",
    ):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_propagated_path_substitution_fails_even_when_content_matches(
    tmp_path: Path,
) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["fern"])
    frame_id = report["frames"][0]["frame_id"]
    report["stage_score_paths"]["propagated"][frame_id] = report[
        "score_paths"
    ][frame_id]
    _write(reports["fern"], report)

    with pytest.raises(
        ForwardBetaAggregationError,
        match=r"propagated/fern-frame score artifact path differs",
    ):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_missing_or_extra_task_report_fails_closed(tmp_path: Path) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    reports["fern"].unlink()
    with pytest.raises(ForwardBetaAggregationError, match="missing"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )

    _manifest, root, receipts, _reports = _make_fixture(tmp_path / "extra")
    extra = root / "ninth" / "eval_full_mask_random_walker/ninth_evaluation.json"
    _write(extra, {"scene_id": "ninth"})
    with pytest.raises(ForwardBetaAggregationError, match="extra"):
        aggregate_forward_beta_full8(
            run_manifest_path=_manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_missing_or_extra_scene_receipt_fails_closed(tmp_path: Path) -> None:
    manifest, root, receipts, _reports = _make_fixture(tmp_path)
    (receipts / "leaves" / "scene_receipt.json").unlink()
    with pytest.raises(ForwardBetaAggregationError, match="receipt set.*missing"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )

    manifest, root, receipts, _reports = _make_fixture(tmp_path / "extra")
    _write(
        receipts / "ninth" / "scene_receipt.json",
        {"scene": "ninth", "run_manifest": str(manifest), "result": "none"},
    )
    with pytest.raises(ForwardBetaAggregationError, match="receipt set.*extra"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_scene_authority_payload_or_sha_drift_fails(tmp_path: Path) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["flower"])
    report["registered_forward_protocol_authority_sha256"] = "b" * 64
    _write(reports["flower"], report)

    with pytest.raises(ForwardBetaAggregationError, match="top-level authority"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_strict_or_blocker_relabel_in_manifest_authority_fails(tmp_path: Path) -> None:
    manifest_path, root, receipts, _reports = _make_fixture(tmp_path)
    manifest = _load(manifest_path)
    authority = manifest["registered_forward_protocol_authority"]
    authority["strict_unseen_protocol_exact_match"] = True
    authority["strict_unseen_exact_match_blockers"] = []
    manifest["registered_forward_protocol_authority_sha256"] = canonical_json_sha256(
        authority
    )
    _write(manifest_path, manifest)

    with pytest.raises(ForwardBetaAggregationError, match="authority is invalid"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest_path,
            result_root=root,
            receipt_root=receipts,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("frozen_diagnostic_eligible", True), ("main_result_eligible", True)],
)
def test_frozen_or_main_promotion_flag_fails(
    tmp_path: Path, field: str, value: bool
) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["fortress"])
    report["safety"][field] = value
    _write(reports["fortress"], report)

    with pytest.raises(ForwardBetaAggregationError, match="safety labels"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_connected_cannot_replace_propagated_main_output(tmp_path: Path) -> None:
    manifest, root, receipts, reports = _make_fixture(tmp_path)
    report = _load(reports["horns_center"])
    report["foreground_iou"] = report["stage_metrics"]["connected"]["foreground_iou"]
    _write(reports["horns_center"], report)

    with pytest.raises(ForwardBetaAggregationError, match="propagated stage"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest,
            result_root=root,
            receipt_root=receipts,
        )


def test_ludvig_candidate_binding_forgery_fails_before_aggregation(
    tmp_path: Path,
) -> None:
    manifest_path, root, receipts, _reports = _make_fixture(tmp_path)
    manifest = _load(manifest_path)
    authority = manifest["registered_forward_protocol_authority"]
    authority["external_comparator_provenance"]["candidate_binding"] = {
        "canonical_task_id": "spatial_nvos_ludvig",
        "registry_row": "forged",
        "promptable_registry_row": "forged",
    }
    manifest["registered_forward_protocol_authority_sha256"] = canonical_json_sha256(
        authority
    )
    _write(manifest_path, manifest)

    with pytest.raises(ForwardBetaAggregationError, match="authority is invalid"):
        aggregate_forward_beta_full8(
            run_manifest_path=manifest_path,
            result_root=root,
            receipt_root=receipts,
        )


def test_cli_writes_immutable_non_exact_output(tmp_path: Path) -> None:
    manifest, root, receipts, _reports = _make_fixture(tmp_path)
    output = tmp_path / "closeout.json"

    assert (
        main(
            [
                "--run-manifest",
                str(manifest),
                "--result-root",
                str(root),
                "--receipt-root",
                str(receipts),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    saved = _load(output)
    assert saved["status"] == ELIGIBILITY
    assert saved["main_result_eligible"] is False
    with pytest.raises(FileExistsError):
        main(
            [
                "--run-manifest",
                str(manifest),
                "--result-root",
                str(root),
                "--receipt-root",
                str(receipts),
                "--output",
                str(output),
            ]
        )
