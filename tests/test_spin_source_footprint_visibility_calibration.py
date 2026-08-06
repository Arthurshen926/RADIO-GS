from __future__ import annotations

from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest
import torch

from radio_gs.querying.spin_source_footprint_visibility_calibration import (
    MATCHED_OOF_ARTIFACT_TYPE,
    MAX_FOLD_THRESHOLD_SPAN,
    build_crossfit_calibration,
    compute_matched_oof_support,
    matched_oof_method_contract,
    reference_threshold_grid,
    select_responsibility_weighted_threshold,
    visibility_adaptive_threshold,
    visibility_calibrated_prediction,
)
from radio_gs.scripts.build_spin_source_footprint_matched_oof import json_sha256
from radio_gs.scripts.build_spin_source_footprint_matched_oof import file_sha256
from radio_gs.scripts.build_spin_source_footprint_visibility_calibration import (
    build as build_visibility_calibration,
)
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256


def test_reference_threshold_grid_is_the_frozen_release_grid() -> None:
    grid = reference_threshold_grid()
    assert len(grid) == 97
    assert grid[0] == pytest.approx(0.99)
    assert grid[-1] == pytest.approx(0.03)
    assert all(left > right for left, right in zip(grid, grid[1:]))
    contract = matched_oof_method_contract()
    assert contract["feature_bandwidth"] == 0.5
    assert contract["regularizer_bandwidth"] == 1.0
    assert contract["effective_knn_columns"] == 201
    assert contract["parameter_scan"] is False


def test_matched_oof_clears_evidence_and_logistic_weight_before_k201() -> None:
    count = 202
    generator = torch.Generator().manual_seed(20260805)
    features = torch.randn((count, 4), generator=generator)
    # K201 including self, with one deterministic omitted row per node.
    neighbors = torch.empty((count, 201), dtype=torch.long)
    all_rows = torch.arange(count)
    for row in range(count):
        neighbors[row] = all_rows[all_rows != ((row + 1) % count)]
    positive = torch.zeros(count)
    positive[:20] = torch.linspace(0.1, 1.0, 20)
    reference = torch.ones(count)
    heldout = torch.zeros(count, dtype=torch.bool)
    heldout[5:15] = True

    result = compute_matched_oof_support(
        features,
        neighbors,
        positive,
        reference,
        heldout,
        device="cpu",
    )

    assert torch.equal(result.heldout, heldout)
    assert torch.count_nonzero(result.training_positive_weight[heldout]) == 0
    assert torch.count_nonzero(result.training_reference_weight[heldout]) == 0
    assert result.probability.shape == (count,)
    assert result.query_compatibility.shape == (count,)
    assert torch.isfinite(result.probability).all()
    assert torch.all((0 <= result.probability) & (result.probability <= 1))


def test_matched_oof_requires_exact_k201_and_nonempty_holdout() -> None:
    features = torch.eye(5)
    with pytest.raises(ValueError, match="K201"):
        compute_matched_oof_support(
            features,
            torch.arange(5)[:, None],
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            torch.ones(5),
            torch.tensor([False, True, False, False, False]),
        )


def test_weighted_threshold_uses_soft_responsibility_and_first_tie() -> None:
    selection = select_responsibility_weighted_threshold(
        torch.tensor([0.9, 0.8, 0.7, 0.2]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.ones(4, dtype=torch.bool),
        thresholds=(0.85, 0.75, 0.65),
    )
    assert selection.threshold == 0.75
    assert selection.weighted_soft_iou == 1.0

    tie = select_responsibility_weighted_threshold(
        torch.tensor([0.9, 0.1]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.ones(2, dtype=torch.bool),
        thresholds=(0.8, 0.7),
    )
    assert tie.threshold == 0.8


def _fold_payload(fold: int) -> dict[str, object]:
    fold_ids = torch.tensor([0, 1, 2, 0, 1, 2])
    heldout = fold_ids == fold
    probability = torch.zeros(6)
    # Every fold has one positive score 0.8 and one negative score 0.2.
    probability[heldout] = torch.tensor([0.8, 0.2])
    contract = matched_oof_method_contract()
    return {
        "artifact_type": MATCHED_OOF_ARTIFACT_TYPE,
        "heldout_fold": fold,
        "scene_id": "lego",
        "protocol_hash": "p" * 64,
        "capability_cache_sha256": "c" * 64,
        "support_graph_sha256": "g" * 64,
        "source_evidence_authority_sha256": "e" * 64,
        "source_footprint_fold_authority_sha256": "f" * 64,
        "query_diffusion_knn_sha256": "k" * 64,
        "query_diffusion_relation_sha256": "r" * 64,
        "method_contract_sha256": json_sha256(contract),
        "heldout_training_positive_weight_sum": 0.0,
        "heldout_training_reference_weight_sum": 0.0,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "valid": torch.ones(6, dtype=torch.bool),
        "global_rows": torch.arange(6),
        "fold_ids": fold_ids,
        "observed": torch.ones(6, dtype=torch.bool),
        "heldout": heldout,
        "reference_weight": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        "population_positive_weight": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "population_negative_weight": torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        "matched_query_diffusion_probability": probability,
    }


def test_crossfit_calibration_pools_disjoint_folds_and_reports_stability() -> None:
    result = build_crossfit_calibration(
        {fold: _fold_payload(fold) for fold in range(3)}
    )
    assert result.t_completion == pytest.approx(0.8)
    assert result.fold_thresholds == pytest.approx((0.8, 0.8, 0.8))
    assert result.threshold_span == 0.0
    assert result.stable is True
    assert MAX_FOLD_THRESHOLD_SPAN == 0.10
    assert int(result.pooled_eligible.sum()) == 6
    assert torch.equal(
        result.source_visible,
        torch.tensor([True, True, True, True, True, False]),
    )


def test_crossfit_calibration_fails_closed_on_authority_or_clear_tamper() -> None:
    folds = {fold: _fold_payload(fold) for fold in range(3)}
    changed = copy.deepcopy(folds)
    changed[1]["query_diffusion_relation_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="authority differs"):
        build_crossfit_calibration(changed)

    leaked = copy.deepcopy(folds)
    leaked[2]["heldout_training_reference_weight_sum"] = 1e-9
    with pytest.raises(ValueError, match="not cleared"):
        build_crossfit_calibration(leaked)


def test_visibility_threshold_has_exact_endpoints_and_no_exponent() -> None:
    coverage = torch.tensor([0.0, 0.25, 1.0])
    threshold = visibility_adaptive_threshold(
        coverage,
        t_seen=0.71,
        t_completion=0.61,
    )
    assert threshold.tolist() == pytest.approx([0.61, 0.635, 0.71])
    prediction = visibility_calibrated_prediction(
        torch.tensor([0.60, 0.64, 0.72]),
        coverage,
        t_seen=0.71,
        t_completion=0.61,
    )
    assert torch.equal(prediction, torch.tensor([False, True, True]))


def test_visibility_threshold_rejects_out_of_range_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        visibility_adaptive_threshold(
            torch.tensor([1.01]),
            t_seen=0.71,
            t_completion=0.61,
        )


def test_visibility_calibration_builder_accepts_only_pre_metric_seen_threshold(
    tmp_path: Path,
) -> None:
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "registration": (
                    "spin_source_footprint_crossfit_visibility_calibration_v1"
                )
            }
        ),
        encoding="utf-8",
    )
    fold_paths: list[Path] = []
    for fold in range(3):
        payload = _fold_payload(fold)
        payload["method_contract"] = matched_oof_method_contract()
        tensor_names = (
            "valid",
            "global_rows",
            "fold_ids",
            "observed",
            "heldout",
            "reference_weight",
            "population_positive_weight",
            "population_negative_weight",
            "matched_query_diffusion_probability",
        )
        payload["tensor_sha256"] = {
            name: tensor_sha256(payload[name]) for name in tensor_names
        }
        path = tmp_path / f"fold_{fold}.pt"
        torch.save(payload, path)
        fold_paths.append(path)
    premetric = tmp_path / "pre_metric.json"
    premetric.write_text(
        json.dumps(
            {
                "artifact_type": "nvos_pre_metric_prediction_receipt_v1",
                "scene_id": "lego",
                "protocol_hash": "p" * 64,
                "sealed_before_target_ground_truth_open": True,
                "target_rgb_opened": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
                "method_contract": {
                    "score_threshold": 0.71,
                    "query_conditioned_diffusion": {
                        "kernel": "ludvig_release_compat"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "calibration.pt"
    arguments = Namespace(
        preregistration=str(preregistration),
        preregistration_sha256=file_sha256(preregistration),
        fold_0=str(fold_paths[0]),
        fold_0_sha256=file_sha256(fold_paths[0]),
        fold_1=str(fold_paths[1]),
        fold_1_sha256=file_sha256(fold_paths[1]),
        fold_2=str(fold_paths[2]),
        fold_2_sha256=file_sha256(fold_paths[2]),
        full_fit_pre_metric_receipt=str(premetric),
        full_fit_pre_metric_receipt_sha256=file_sha256(premetric),
        output=str(output),
    )
    receipt = build_visibility_calibration(arguments)
    assert receipt["status"] == "pass_stable_source_only_calibration"
    assert receipt["t_seen"] == 0.71
    assert receipt["t_completion"] == pytest.approx(0.8)
    assert receipt["target_mask_opened"] is False
    assert output.is_file()

    unsafe = json.loads(premetric.read_text(encoding="utf-8"))
    unsafe["target_metric_opened"] = True
    unsafe_path = tmp_path / "unsafe_pre_metric.json"
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
    arguments.full_fit_pre_metric_receipt = str(unsafe_path)
    arguments.full_fit_pre_metric_receipt_sha256 = file_sha256(unsafe_path)
    arguments.output = str(tmp_path / "unsafe_calibration.pt")
    with pytest.raises(ValueError, match="not target blind"):
        build_visibility_calibration(arguments)
