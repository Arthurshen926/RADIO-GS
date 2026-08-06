from __future__ import annotations

from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.spin_source_footprint_quantile_calibration import (
    INPUT_MATCHED_OOF_ARTIFACT_TYPE,
    MAX_FOLD_QUANTILE_THRESHOLD_SPAN,
    WeightedRightECDF,
    build_quantile_prediction_fields,
    build_quantile_oof_calibration,
    compute_full_fit_quantile_gauge,
    map_target_score_to_source_quantile,
    quantile_method_contract,
    quantile_threshold_grid,
    quantile_visibility_adaptive_threshold,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    build as build_quantile_authority,
    file_sha256,
    json_sha256,
)
from radio_gs.scripts.eval_spin_source_footprint_quantile_target_prediction import (
    exact_four_connected_boundary,
)


def test_weighted_right_ecdf_has_exact_tie_semantics() -> None:
    cdf = WeightedRightECDF.fit(
        torch.tensor([0.1, 0.2, 0.2, 0.9]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert cdf.source_rows == 4
    assert cdf.total_weight == 10.0
    assert cdf.support.tolist() == pytest.approx([0.1, 0.2, 0.9])
    assert cdf.map(torch.tensor([0.0, 0.1, 0.199, 0.2, 0.8, 0.9])).tolist() == (
        pytest.approx([0.0, 0.1, 0.1, 0.6, 0.6, 1.0])
    )


def test_weighted_right_ecdf_is_invariant_to_strict_monotone_score_gauge() -> None:
    score = torch.tensor([0.05, 0.2, 0.4, 0.6, 0.9])
    weight = torch.tensor([2.0, 1.0, 3.0, 4.0, 2.0])
    query = torch.tensor([0.1, 0.4, 0.7])
    raw = WeightedRightECDF.fit(score, weight).map(query)
    transformed = WeightedRightECDF.fit(score.square(), weight).map(query.square())
    assert torch.equal(raw, transformed)


def test_weighted_right_ecdf_ignores_zero_weight_rows() -> None:
    first = WeightedRightECDF.fit(
        torch.tensor([0.1, 0.3]), torch.tensor([1.0, 1.0])
    )
    second = WeightedRightECDF.fit(
        torch.tensor([0.1, 0.2, 0.3]), torch.tensor([1.0, 0.0, 1.0])
    )
    query = torch.tensor([0.15, 0.25, 0.35])
    assert torch.equal(first.map(query), second.map(query))


def _fold_payload(fold: int) -> dict[str, object]:
    fold_ids = torch.arange(12) // 4
    heldout = fold_ids == fold
    base_score = torch.tensor([0.9, 0.8, 0.2, 0.1] * 3)
    # Three sharply different absolute gauges with identical within-fold ranks.
    power = (1.0, 2.0, 4.0)[fold]
    probability = base_score.pow(power)
    reference = torch.ones(12)
    training_reference = reference.clone()
    training_reference[heldout] = 0.0
    training_positive = torch.tensor([1.0, 1.0, 0.0, 0.0] * 3)
    training_positive[heldout] = 0.0
    return {
        "artifact_type": INPUT_MATCHED_OOF_ARTIFACT_TYPE,
        "heldout_fold": fold,
        "scene_id": "lego",
        "protocol_hash": "p" * 64,
        "capability_cache_sha256": "c" * 64,
        "support_graph_sha256": "g" * 64,
        "source_evidence_authority_sha256": "e" * 64,
        "source_footprint_fold_authority_sha256": "f" * 64,
        "query_diffusion_knn_sha256": "k" * 64,
        "query_diffusion_relation_sha256": "r" * 64,
        "method_contract_sha256": "m" * 64,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "valid": torch.ones(12, dtype=torch.bool),
        "global_rows": torch.arange(12),
        "fold_ids": fold_ids,
        "observed": torch.ones(12, dtype=torch.bool),
        "heldout": heldout,
        "reference_weight": reference,
        "population_positive_weight": torch.tensor(
            [1.0, 1.0, 0.0, 0.0] * 3
        ),
        "population_negative_weight": torch.tensor(
            [0.0, 0.0, 1.0, 1.0] * 3
        ),
        "matched_query_diffusion_probability": probability,
        "training_positive_weight": training_positive,
        "training_reference_weight": training_reference,
    }


def test_quantile_oof_removes_fold_specific_monotone_absolute_gauge() -> None:
    result = build_quantile_oof_calibration(
        {fold: _fold_payload(fold) for fold in range(3)}
    )
    assert [item.threshold for item in result.fold_diagnostics] == pytest.approx(
        [0.75, 0.75, 0.75]
    )
    assert result.t_completion_quantile == pytest.approx(0.75)
    assert result.threshold_span == 0.0
    assert result.stable is True
    assert MAX_FOLD_QUANTILE_THRESHOLD_SPAN == 0.10


def test_quantile_oof_fails_closed_on_authority_or_heldout_leak() -> None:
    folds = {fold: _fold_payload(fold) for fold in range(3)}
    changed = copy.deepcopy(folds)
    changed[1]["query_diffusion_knn_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="authority differs"):
        build_quantile_oof_calibration(changed)

    leaked = copy.deepcopy(folds)
    leaked[2]["training_reference_weight"][leaked[2]["heldout"]] = 1e-6
    with pytest.raises(ValueError, match="not cleared"):
        build_quantile_oof_calibration(leaked)


def test_full_fit_gauge_maps_seen_and_target_by_the_same_source_cdf() -> None:
    count = 202
    generator = torch.Generator().manual_seed(20260805)
    features = torch.randn((count, 4), generator=generator)
    neighbors = torch.empty((count, 201), dtype=torch.long)
    all_rows = torch.arange(count)
    for row in range(count):
        neighbors[row] = all_rows[all_rows != ((row + 1) % count)]
    positive = torch.zeros(count)
    positive[:20] = torch.linspace(0.1, 1.0, 20)
    reference = torch.ones(count)

    result = compute_full_fit_quantile_gauge(
        features,
        neighbors,
        positive,
        reference,
        t_seen_raw=0.71,
        device="cpu",
    )
    expected = float(result.ecdf.map(0.71))
    assert result.t_seen_quantile == expected
    assert torch.equal(
        map_target_score_to_source_quantile(
            torch.tensor([0.2, 0.71, 0.9]), result.ecdf
        ),
        result.ecdf.map(torch.tensor([0.2, 0.71, 0.9])),
    )


def test_quantile_contract_has_no_hidden_calibration_hyperparameter() -> None:
    contract = quantile_method_contract()
    assert len(quantile_threshold_grid()) == 97
    assert contract["cdf_definition"] == "weighted_right_continuous_empirical_cdf"
    assert contract["cdf_smoothing"] is False
    assert contract["cdf_temperature"] is None
    assert contract["cdf_bandwidth"] is None
    assert contract["parameter_scan"] is False
    threshold = quantile_visibility_adaptive_threshold(
        torch.tensor([0.0, 0.25, 1.0]),
        t_seen_quantile=0.9,
        t_completion_quantile=0.7,
    )
    assert threshold.tolist() == pytest.approx([0.7, 0.75, 0.9])


def test_quantile_prediction_uses_a_continuous_margin_before_resize() -> None:
    cdf = WeightedRightECDF.fit(
        torch.tensor([0.1, 0.2, 0.4, 0.8]),
        torch.ones(4),
    )
    fields = build_quantile_prediction_fields(
        torch.tensor([[0.2, 0.8], [0.1, 0.4]]),
        torch.tensor([[0.0, 1.0], [0.5, 0.25]]),
        cdf,
        t_seen_quantile=0.9,
        t_completion_quantile=0.7,
    )
    torch.testing.assert_close(
        fields.continuous_margin,
        fields.score_quantile - fields.spatial_threshold_quantile,
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        fields.low_resolution_prediction,
        fields.continuous_margin >= 0,
    )
    torch.testing.assert_close(
        fields.spatial_threshold_quantile,
        torch.tensor([[0.7, 0.9], [0.8, 0.75]], dtype=torch.float64),
    )


def test_exact_four_connected_boundary_marks_both_transition_sides() -> None:
    mask = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=torch.uint8,
    ).numpy()
    boundary = exact_four_connected_boundary(mask)
    assert boundary.tolist() == [
        [False, True, True],
        [True, True, True],
        [True, True, False],
    ]


def test_quantile_authority_builder_is_independent_and_immutable(
    tmp_path: Path,
) -> None:
    preregistration = tmp_path / "v2_preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "registration": (
                    "spin_source_footprint_crossfit_quantile_calibration_v2"
                )
            }
        ),
        encoding="utf-8",
    )
    stopped_v1 = tmp_path / "stopped_v1.json"
    stopped_v1.write_text(json.dumps({"status": "stop"}), encoding="utf-8")
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
        "training_positive_weight",
        "training_reference_weight",
    )
    fold_paths = []
    for fold in range(3):
        payload = _fold_payload(fold)
        payload["tensor_sha256"] = {
            name: tensor_sha256(payload[name]) for name in tensor_names
        }
        path = tmp_path / f"fold_{fold}.pt"
        torch.save(payload, path)
        fold_paths.append(path)
    output = tmp_path / "quantile_oof.pt"
    args = Namespace(
        preregistration=str(preregistration),
        preregistration_sha256=file_sha256(preregistration),
        stopped_v1_result=str(stopped_v1),
        stopped_v1_result_sha256=file_sha256(stopped_v1),
        fold_0=str(fold_paths[0]),
        fold_0_sha256=file_sha256(fold_paths[0]),
        fold_1=str(fold_paths[1]),
        fold_1_sha256=file_sha256(fold_paths[1]),
        fold_2=str(fold_paths[2]),
        fold_2_sha256=file_sha256(fold_paths[2]),
        output=str(output),
    )
    receipt = build_quantile_authority(args)
    assert receipt["status"] == "pass_stable_source_only_quantile_gauge"
    assert receipt["t_completion_quantile"] == pytest.approx(0.75)
    assert receipt["target_mask_opened"] is False
    assert receipt["method_contract_sha256"] == json_sha256(
        quantile_method_contract()
    )
    assert output.is_file()
    assert build_quantile_authority(args)["artifact_sha256"] == receipt[
        "artifact_sha256"
    ]
