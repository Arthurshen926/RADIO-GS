from types import SimpleNamespace

import pytest
import torch

from radio_gs.v4.carrier import Camera, ProjectionTable
from radio_gs.v4.completion import (
    OracleIdentityCompletionMLP,
    PartialObjectMembership,
    completion_metrics,
)
from radio_gs.v4.training.train_scannet_completion_oracle import (
    RGB_GEOMETRY_LAYOUT,
    RGB_RADIO_GEOMETRY_LAYOUT,
    _heldout_2d_metrics,
    _pooled_categorical_confusion,
    _pool_soft_iou_sufficient_statistics,
    _render_valid_posterior,
    _runtime,
    _select_local_features,
    _soft_iou_token_statistics_from_components,
)


def _all_unknown(labels: torch.Tensor, token_count: int) -> PartialObjectMembership:
    return PartialObjectMembership.from_oracle_visibility(
        labels,
        torch.zeros_like(labels, dtype=torch.bool),
        token_count=token_count,
    )


def test_primary_assignment_is_threshold_free_argmax_and_legacy_is_named_at_0p5():
    labels = torch.tensor([0, 1, 1, -1, -1])
    partial = _all_unknown(labels, token_count=2)
    membership = torch.tensor([
        [0.40, 0.30],  # correct token 0, but below 0.5
        [0.20, 0.41],  # correct token 1, but below 0.5
        [0.40, 0.35],  # wrong token 0
        [0.40, 0.10],  # correct null
        [0.40, 0.41],  # token hallucination on retained-set null
    ])
    null = torch.tensor([0.30, 0.39, 0.25, 0.50, 0.19])

    metrics = completion_metrics(
        membership, partial, labels, null_probability=null, assignment_threshold=0.5
    )

    assert metrics["unknown_assignment_precision"] == pytest.approx(2 / 4)
    assert metrics["unknown_retained_object_coverage"] == pytest.approx(1.0)
    assert metrics["unknown_correct_assignment_recall"] == pytest.approx(2 / 3)
    assert metrics["assigned_unknown_object_top1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["unknown_retained_set_null_recall"] == pytest.approx(1 / 2)
    assert metrics["assigned_unknown_count"] == 4
    assert metrics["unknown_correct_token_count"] == 2
    assert metrics["unknown_wrong_token_on_object_count"] == 1
    assert metrics["unknown_token_on_null_count"] == 1
    assert metrics["unknown_null_on_object_count"] == 0
    assert metrics["unknown_correct_null_count"] == 1
    assert metrics["assigned_unknown_count_at_0p5"] == 0
    assert metrics["unknown_assignment_precision_at_0p5"] == 0
    assert metrics["unknown_retained_object_coverage_at_0p5"] == 0
    assert metrics["unknown_only_soft_3d_miou"] > 0

    with pytest.raises(ValueError, match="frozen at 0.5"):
        completion_metrics(
            membership, partial, labels, null_probability=null,
            assignment_threshold=0.4,
        )


def test_visible_mask_gap_and_never_visible_strata_report_separate_confusions():
    labels = torch.tensor([0, 1, 1, -1, -1])
    partial = _all_unknown(labels, token_count=2)
    membership = torch.tensor([
        [0.40, 0.30], [0.20, 0.41], [0.40, 0.35], [0.40, 0.10], [0.40, 0.41]
    ])
    null = torch.tensor([0.30, 0.39, 0.25, 0.50, 0.19])
    strata = {
        "visible_but_unmasked": torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
        "never_visible": torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
    }

    metrics = completion_metrics(
        membership,
        partial,
        labels,
        null_probability=null,
        unknown_strata=strata,
    )

    assert metrics["visible_but_unmasked_element_count"] == 2
    assert metrics["visible_but_unmasked_correct_token_count"] == 2
    assert metrics["visible_but_unmasked_assignment_precision"] == 1
    assert metrics["never_visible_element_count"] == 3
    assert metrics["never_visible_wrong_token_on_object_count"] == 1
    assert metrics["never_visible_token_on_null_count"] == 1
    assert metrics["never_visible_correct_null_count"] == 1
    assert metrics["visible_but_unmasked_soft_3d_miou"] > 0
    assert metrics["never_visible_soft_3d_miou"] >= 0

    with pytest.raises(ValueError, match="must be disjoint"):
        completion_metrics(
            membership,
            partial,
            labels,
            null_probability=null,
            unknown_strata={
                "visible_but_unmasked": torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool),
                "never_visible": torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool),
            },
        )


class _ProjectionCarrier:
    def __init__(self, projections: dict[str, ProjectionTable]):
        self.projections = projections

    def project(self, camera: Camera) -> ProjectionTable:
        return self.projections[camera.key]


def _camera_record(key: str) -> dict:
    return {
        "key": key,
        "intrinsic": torch.eye(3),
        "camera_to_world": torch.eye(4),
        "height": 1,
        "width": 1,
    }


def _one_pixel_projection(element_ids, weights, *, num_elements: int) -> ProjectionTable:
    count = len(element_ids)
    return ProjectionTable(
        element_ids=torch.tensor(element_ids),
        pixel_ids=torch.zeros(count, dtype=torch.long),
        depths=torch.ones(count),
        weights=torch.tensor(weights),
        num_elements=num_elements,
        height=1,
        width=1,
    )


def test_validity_filter_preserves_all_surface_mass_in_render_denominator():
    camera = Camera("view", torch.eye(3), torch.eye(4), 1, 1)
    carrier = _ProjectionCarrier({
        "view": _one_pixel_projection([0, 1], [0.5, 0.5], num_elements=2)
    })
    rendered = _render_valid_posterior(
        carrier,
        torch.tensor([[1.0], [1.0]]),
        torch.tensor([True, False]),
        camera,
    )
    assert rendered.shape == (1, 1, 1)
    assert float(rendered[0, 0, 0]) == pytest.approx(0.5)


def test_heldout_iou_aggregates_per_token_across_views_and_counts_absent_fp():
    projection = _one_pixel_projection([0], [1.0], num_elements=1)
    runtime = {
        "partial": SimpleNamespace(eligible_elements=torch.tensor([True])),
        "carrier": _ProjectionCarrier({"seen": projection, "absent": projection}),
        "payload": {
            "heldout_cameras": [_camera_record("seen"), _camera_record("absent")],
            "heldout_mesh_target_rasters": [
                torch.tensor([[[1.0, 0.0]]]),
                torch.tensor([[[0.0, 0.0]]]),
            ],
        },
    }
    # Both tokens are predicted in both views. Token 0 has IoU 1/2 because its
    # absent-view FP remains in the union; token 1 is FP-only with IoU 0.
    metrics = _heldout_2d_metrics(runtime, torch.tensor([[1.0, 1.0]]))
    assert metrics["heldout_2d_soft_miou"] == pytest.approx(0.25)
    assert metrics["heldout_2d_cross_view_token_soft_miou"] == pytest.approx(0.25)
    assert metrics["heldout_2d_evaluated_token_count"] == 2
    assert metrics["heldout_2d_target_present_token_count"] == 1
    assert metrics["heldout_2d_false_positive_only_token_count"] == 1
    assert metrics["heldout_2d_aggregation"] == (
        "sum_intersection_union_per_token_across_all_heldout_views"
    )
    statistics = metrics["heldout_2d_soft_iou_sufficient_statistics"]
    assert statistics["numeric_dtype"] == "float64"
    assert statistics["domain_unit"] == "heldout_pixel_token"
    false_positive_only = statistics["token_statistics"][1]
    assert false_positive_only == {
        "token_index": 1,
        "intersection": 0.0,
        "prediction_mass": 2.0,
        "target_mass": 0.0,
        "union": 2.0,
        "evaluated": True,
    }


def test_soft_iou_sufficient_statistics_pool_scene_tokens_and_union_mass():
    statistics = {
        cohort: _soft_iou_token_statistics_from_components(
            torch.tensor([2.0, 0.0], dtype=torch.float64),
            torch.tensor([3.0, 1.0], dtype=torch.float64),
            torch.tensor([3.0, 0.0], dtype=torch.float64),
            domain_unit=(
                "heldout_pixel_token"
                if cohort == "heldout_2d"
                else "surface_element_token"
            ),
        )
        for cohort in (
            "full_3d",
            "unknown_3d",
            "visible_but_unmasked_3d",
            "never_visible_3d",
            "heldout_2d",
        )
    }
    records = [
        {
            "learned_completion": {
                "soft_iou_sufficient_statistics": statistics,
            }
        }
    ]

    pooled = _pool_soft_iou_sufficient_statistics(
        records, "learned_completion"
    )["full_3d"]

    assert pooled["scene_token_macro_soft_iou"] == pytest.approx(0.25)
    assert pooled[
        "union_summed_element_or_pixel_token_micro_soft_iou"
    ] == pytest.approx(0.4)
    assert pooled["counts"]["evaluated_scene_token_count"] == 2
    assert pooled["counts"]["target_present_scene_token_count"] == 1
    assert pooled["counts"]["false_positive_only_scene_token_count"] == 1
    assert pooled["sums"] == {
        "intersection": 2.0,
        "prediction_mass": 4.0,
        "target_mass": 3.0,
        "union": 5.0,
    }


def test_raw_null_logit_is_not_shifted_by_token_cardinality():
    scorer = OracleIdentityCompletionMLP(3, hidden_dimension=4, dropout=0)
    for parameter in scorer.parameters():
        torch.nn.init.zeros_(parameter)
    for token_count in (2, 5):
        logits = scorer.categorical_logits(torch.zeros(1, token_count, 3))
        assert float(logits[0, -1].detach()) == 0
        assert float(torch.softmax(logits, -1)[0, -1].detach()) == pytest.approx(
            1 / (token_count + 1)
        )
    assert scorer.token_cardinality_normalization == "none_raw_learned_null_logit"


def test_local_feature_mode_is_explicit_and_selects_only_frozen_layout_columns():
    features = torch.arange(2 * 71, dtype=torch.float32).reshape(2, 71)
    payload = {
        "local_features": features,
        "configuration": {"local_feature_layout": list(RGB_RADIO_GEOMETRY_LAYOUT)},
    }
    rgb, rgb_layout = _select_local_features(payload, "rgb_geometry")
    radio, radio_layout = _select_local_features(payload, "rgb_radio_geometry")
    assert rgb.shape == (2, 7)
    assert torch.equal(rgb, torch.cat((features[:, :4], features[:, -3:]), -1))
    assert rgb_layout == RGB_GEOMETRY_LAYOUT
    assert torch.equal(radio, features)
    assert radio_layout == RGB_RADIO_GEOMETRY_LAYOUT

    legacy = {
        "local_features": features[:, :7],
        "configuration": {"local_feature_layout": list(RGB_GEOMETRY_LAYOUT)},
    }
    with pytest.raises(ValueError, match="requires the sealed F71"):
        _select_local_features(legacy, "rgb_radio_geometry")
    with pytest.raises(ValueError, match="unsupported local feature mode"):
        _select_local_features(payload, "implicit")


def test_runtime_uses_mask_support_for_membership_and_source_visibility_for_strata():
    membership_observed = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    source_visible = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    features = torch.zeros(4, 71)
    features[:, 3] = source_visible.float()
    payload = {
        "scene_id": "synthetic_scene",
        "centres": torch.tensor([
            [0.00, 0.00, 1.0],
            [0.04, 0.00, 1.0],
            [1.00, 0.00, 1.0],
            [1.04, 0.00, 1.0],
        ]),
        "normals": torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1).clone(),
        "local_features": features,
        "token_index": torch.tensor([0, 0, 1, 1]),
        "object_ids": [10, 20],
        "completion_valid": torch.ones(4, dtype=torch.bool),
        "source_visible": source_visible,
        "feature_available": source_visible,
        "appearance_available": source_visible,
        "membership_observed": membership_observed,
        "mask_supported": membership_observed,
        "observed_visible": membership_observed,
        "configuration": {
            "voxel_size": 0.04,
            "maximum_splat_radius": 1,
            "surface_band_voxels": 1.5,
            "maximum_contributors_per_pixel": 8,
            "local_feature_layout": list(RGB_RADIO_GEOMETRY_LAYOUT),
        },
    }
    runtime = _runtime(payload, "rgb_geometry")
    assert runtime["local_features"].shape == (4, 7)
    assert runtime["partial"].element_is_observed.tolist() == [True, False, True, False]
    assert runtime["unknown_strata"]["visible_but_unmasked"].tolist() == [
        False, True, False, False
    ]
    assert runtime["unknown_strata"]["never_visible"].tolist() == [
        False, False, False, True
    ]


def test_raw_confusion_supports_exact_pooled_metrics():
    labels = torch.tensor([0, 1, 1, -1, -1])
    partial = _all_unknown(labels, token_count=2)
    membership = torch.tensor([
        [0.40, 0.30], [0.20, 0.41], [0.40, 0.35], [0.40, 0.10], [0.40, 0.41]
    ])
    null = torch.tensor([0.30, 0.39, 0.25, 0.50, 0.19])
    metrics = completion_metrics(membership, partial, labels, null_probability=null)
    records = [
        {"learned_completion": metrics},
        {"learned_completion": metrics},
    ]
    pooled = _pooled_categorical_confusion(records, "learned_completion", "unknown")
    assert pooled["counts"]["element_count"] == 10
    assert pooled["counts"]["correct_token_count"] == 4
    assert pooled["metrics"]["assignment_precision"] == pytest.approx(0.5)
    assert pooled["metrics"]["correct_assignment_recall"] == pytest.approx(2 / 3)
