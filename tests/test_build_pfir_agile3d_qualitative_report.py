from __future__ import annotations

from PIL import Image
import pytest
import numpy as np

from radio_gs.scripts import build_pfir_agile3d_qualitative_report as report


def _pfir_rows() -> tuple[list[dict], list[dict]]:
    ranking = [
        {
            "query_id": "z_success",
            "scene_id": "scene_z",
            "rank": 1,
            "same_category": True,
        },
        {
            "query_id": "a_success",
            "scene_id": "scene_a",
            "rank": 1,
            "same_category": True,
        },
        {
            "query_id": "z_gap",
            "scene_id": "scene_z",
            "rank": 1,
            "same_category": False,
        },
        {
            "query_id": "a_gap",
            "scene_id": "scene_a",
            "rank": 1,
            "same_category": False,
        },
        {
            "query_id": "z_confusion_hard",
            "scene_id": "scene_z",
            "rank": 8,
            "same_category": True,
        },
        {
            "query_id": "a_confusion_hard",
            "scene_id": "scene_a",
            "rank": 8,
            "same_category": True,
        },
    ]
    selection = [
        {"query_id": "z_success", "iou": 0.70},
        {"query_id": "a_success", "iou": 0.70},
        {"query_id": "z_gap", "iou": 0.01},
        {"query_id": "a_gap", "iou": 0.01},
        {
            "query_id": "z_confusion_hard",
            "iou": 0.02,
            "same_category_distractor_success": False,
        },
        {
            "query_id": "a_confusion_hard",
            "iou": 0.02,
            "same_category_distractor_success": False,
        },
    ]
    return ranking, selection


def test_select_pfir_cases_uses_fixed_predicates_and_query_id_ties() -> None:
    ranking, selection = _pfir_rows()

    selected = report.select_pfir_cases(ranking, selection)

    assert [row["kind"] for row in selected] == [
        "success",
        "rank_mask_gap",
        "same_class_confusion",
    ]
    assert [row["query_id"] for row in selected] == [
        "a_success",
        "a_gap",
        "a_confusion_hard",
    ]


def test_select_agile_cases_is_deterministic_under_input_permutation() -> None:
    rows = [
        {"scene_id": "scene_high", "object_id": 2, "trajectory": {15: 0.95}},
        {"scene_id": "scene_mid", "object_id": 1, "trajectory": {15: 0.50}},
        {"scene_id": "scene_low", "object_id": 3, "trajectory": {15: 0.02}},
    ]
    coverage = [
        {"scene_id": "scene_high", "feature_coverage": 0.95},
        {"scene_id": "scene_mid", "feature_coverage": 0.50},
        {"scene_id": "scene_low", "feature_coverage": 0.10},
    ]

    selected = report.select_agile_cases(rows, coverage)
    reversed_selected = report.select_agile_cases(list(reversed(rows)), coverage)

    assert [row["kind"] for row in selected] == [
        "high_coverage_success",
        "middle_coverage_middle_iou",
        "low_coverage_failure",
    ]
    assert [(row["scene_id"], row["object_id"]) for row in selected] == [
        ("scene_high", 2),
        ("scene_mid", 1),
        ("scene_low", 3),
    ]
    assert selected == reversed_selected


def test_mask_error_colors_distinguish_all_mask_outcomes() -> None:
    prediction = np.array([True, True, False, False])
    target = np.array([True, False, True, False])

    colors = report.mask_error_colors(prediction, target)

    assert tuple(colors[0]) == report.TP_COLOR
    assert tuple(colors[1]) == report.FP_COLOR
    assert tuple(colors[2]) == report.FN_COLOR
    assert tuple(colors[3]) == report.BG_COLOR


def test_assert_trajectory_matches_rejects_metric_drift() -> None:
    with pytest.raises(AssertionError, match="click 2"):
        report.assert_trajectory_matches({1: 0.1, 2: 0.2}, {1: 0.1, 2: 0.3})


def test_assert_trajectory_matches_accepts_json_and_integer_click_keys() -> None:
    report.assert_trajectory_matches({"1": 0.1, "2": 0.2}, {1: 0.1, 2: 0.2})


def test_replay_interactive_masks_keeps_forced_clicks_and_requested_snapshots() -> None:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    target = np.array([True, True, False])

    replay = report.replay_interactive_masks(
        coordinates,
        target,
        target,
        np.arange(3, dtype=np.int64),
        lambda _xyz, _previous, _clicks: np.zeros(3, dtype=bool),
        max_clicks=3,
        capture_click_counts=(1, 2, 3),
        click_workers=1,
    )

    assert replay["trajectory"] == {1: 0.5, 2: 1.0, 3: 1.0}
    assert replay["snapshots"][1]["prediction"].tolist() == [True, False, False]
    assert replay["snapshots"][2]["prediction"].tolist() == [True, True, False]
    assert replay["snapshots"][3]["prediction"].tolist() == [True, True, False]
    assert [click.is_positive for click in replay["snapshots"][2]["clicks"]] == [True, True]


def test_render_mask_view_returns_rgb_image_with_mask_overlay() -> None:
    xyz = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = np.full((4, 3), 128, dtype=np.uint8)
    image = report.render_mask_view(
        xyz,
        rgb,
        np.array([True, False, False, False]),
        np.array([True, False, False, False]),
        size=(64, 48),
        overlay="error",
    )

    pixels = np.asarray(image)
    assert image.mode == "RGB"
    assert image.size == (64, 48)
    assert np.any(np.all(pixels == report.TP_COLOR, axis=-1))


def test_render_mask_view_focus_mask_enlarges_target_without_changing_canvas_size() -> None:
    xyz = np.array(
        [
            [-5.0, -5.0, 0.0],
            [5.0, -5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [-0.30, -0.30, 1.0],
            [0.30, -0.30, 1.0],
            [-0.30, 0.30, 1.0],
            [0.30, 0.30, 1.0],
        ],
        dtype=np.float32,
    )
    rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    target = np.array([False, False, False, False, True, True, True, True])

    full_scene = report.render_mask_view(
        xyz, rgb, target, target, size=(160, 120), overlay="error"
    )
    focused = report.render_mask_view(
        xyz,
        rgb,
        target,
        target,
        size=(160, 120),
        overlay="error",
        focus_mask=target,
    )

    full_tp = np.all(np.asarray(full_scene) == report.TP_COLOR, axis=-1).sum()
    focused_tp = np.all(np.asarray(focused) == report.TP_COLOR, axis=-1).sum()
    assert focused.size == full_scene.size == (160, 120)
    assert focused_tp > full_tp * 2


def test_build_markdown_marks_gt_as_evaluator_only_and_links_both_tasks() -> None:
    markdown = report.build_markdown(
        {
            "pfir": {"cases": [{"kind": "success", "query_id": "q"}]},
            "agile3d": {"cases": [{"kind": "high_coverage_success", "scene_id": "s"}]},
        }
    )

    assert "PFIR" in markdown
    assert "AGILE3D" in markdown
    assert "evaluator-only" in markdown
    assert "target-centered crops" in markdown
    assert "pfir_mask_comparison.png" in markdown
    assert "agile3d_click_replay.png" in markdown


def test_write_outputs_publishes_png_markdown_and_audit(tmp_path) -> None:
    output = report.write_outputs(
        tmp_path,
        Image.new("RGB", (16, 16), "white"),
        Image.new("RGB", (16, 16), "black"),
        {"pfir": {"cases": []}, "agile3d": {"cases": []}},
    )

    assert output["pfir"].is_file()
    assert output["agile3d"].is_file()
    assert output["markdown"].is_file()
    assert output["audit"].is_file()
