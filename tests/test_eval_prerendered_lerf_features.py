import json

import pytest

from radio_gs.scripts import eval_prerendered_lerf_features as eval_features
from radio_gs.scripts.eval_occamlgs_lerf_checkpoint import (
    ProtocolError,
    _read_namespace_config,
    validate_label_camera_roles,
)


def test_load_lerf_objects_merges_repeated_labels(tmp_path):
    label_root = tmp_path / "label" / "teatime"
    label_root.mkdir(parents=True)
    (label_root / "frame_00002.json").write_text(
        json.dumps(
            {
                "info": {"name": "frame_00002.jpg", "width": 4, "height": 4},
                "objects": [
                    {
                        "category": "cup",
                        "bbox": [0, 0, 1, 1],
                        "segmentation": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    },
                    {
                        "category": "cup",
                        "bbox": [2, 2, 3, 3],
                        "segmentation": [[2, 2], [3, 2], [3, 3], [2, 3]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    frames = eval_features.load_lerf_objects(label_root, frames=["frame_00002"])

    assert [obj.query for obj in frames["frame_00002"]] == ["cup"]
    cup = frames["frame_00002"][0]
    assert cup.mask.sum() == 8
    assert cup.bboxes == [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)]


def test_evaluate_relevance_maps_computes_iou_and_localization():
    obj = eval_features.LerfObject(
        frame="frame_00002",
        query="cup",
        mask=eval_features.np.array([[1, 1], [0, 0]], dtype=bool),
        bboxes=[(0.0, 0.0, 1.0, 0.0)],
    )
    relevance = eval_features.np.array(
        [
            [[[0.2, 0.2], [0.2, 0.2]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ],
        dtype=eval_features.np.float32,
    )

    result = eval_features.evaluate_relevance_maps(
        {"frame_00002": [obj]},
        {"frame_00002": relevance},
        mask_thresh=0.4,
        activation_kernel=1,
        smooth_kernel=1,
    )

    assert result["macro"]["miou"] == pytest.approx(1.0)
    assert result["macro"]["loc_acc"] == pytest.approx(1.0)
    assert result["macro"]["aggregation"] == "query_weighted_micro"
    assert result["query_micro"] == result["macro"]
    assert result["scene_macro"]["aggregation"] == "scene_equal_macro"
    assert result["frames"]["frame_00002"]["objects"][0]["chosen_level"] == 1


def test_evaluate_relevance_maps_upsamples_feature_grid_to_mask_size():
    obj = eval_features.LerfObject(
        frame="frame_00002",
        query="cup",
        mask=eval_features.np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        bboxes=[(0.0, 0.0, 1.0, 1.0)],
    )
    relevance = eval_features.np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=eval_features.np.float32)

    result = eval_features.evaluate_relevance_maps(
        {"frame_00002": [obj]},
        {"frame_00002": relevance},
        mask_thresh=0.4,
        activation_kernel=1,
        smooth_kernel=1,
        resize_policy="bilinear_compat",
    )

    assert result["macro"]["objects"] == 1
    assert result["macro"]["loc_acc"] == pytest.approx(1.0)


def test_evaluate_relevance_maps_reports_missing_frame_as_error():
    with pytest.raises(FileNotFoundError):
        eval_features.evaluate_relevance_maps(
            {"frame_00002": []},
            {},
            mask_thresh=0.4,
            activation_kernel=1,
            smooth_kernel=1,
        )


def test_level_selection_uses_raw_activated_peak_not_normalized_peak():
    obj = eval_features.LerfObject(
        frame="frame_00002",
        query="cup",
        mask=eval_features.np.array([[1, 0], [0, 0]], dtype=bool),
        bboxes=[(0.0, 0.0, 0.0, 0.0)],
    )
    # Both non-constant levels min-max to a peak of one. The released
    # evaluator selects level 1 because its activated raw peak is larger.
    relevance = eval_features.np.array(
        [
            [[[0.2, 0.1], [0.1, 0.1]]],
            [[[0.9, 0.1], [0.1, 0.1]]],
        ],
        dtype=eval_features.np.float32,
    )

    result = eval_features.evaluate_relevance_maps(
        {"frame_00002": [obj]},
        {"frame_00002": relevance},
        mask_thresh=0.4,
        activation_kernel=1,
        smooth_kernel=1,
    )

    item = result["frames"]["frame_00002"]["objects"][0]
    assert item["chosen_level"] == 1
    assert item["level_scores"] == pytest.approx([0.2, 0.9])


def test_aggregate_scene_results_separates_scene_macro_and_query_micro():
    aggregate = eval_features.aggregate_scene_results(
        {
            "scene_a": {"query_micro": {"miou": 1.0, "loc_acc": 1.0, "objects": 1}},
            "scene_b": {"query_micro": {"miou": 0.0, "loc_acc": 0.0, "objects": 3}},
        }
    )

    assert aggregate["scene_macro"]["miou"] == pytest.approx(0.5)
    assert aggregate["scene_macro"]["loc_acc"] == pytest.approx(0.5)
    assert aggregate["query_micro"]["miou"] == pytest.approx(0.25)
    assert aggregate["query_micro"]["loc_acc"] == pytest.approx(0.25)
    assert aggregate["macro"] == aggregate["query_micro"]


def test_occam_paper_profile_records_full_threshold_and_kernel_protocol():
    args = eval_features.build_arg_parser().parse_args(
        [
            "--label-root",
            "/labels",
            "--feature-dirs",
            "/features/1",
            "/features/2",
            "/features/3",
            "--protocol-profile",
            "occam_langsplat_paper",
            "--output-json",
            "/tmp/result.json",
        ]
    )

    config = eval_features.resolve_protocol_config(args)

    assert config["mask_thresh"] == pytest.approx(0.5)
    assert config["activation_kernel"] == 30
    assert config["smooth_kernel"] == 7
    assert config["feature_mode"] == "raw"
    assert config["filter_implementation"] == "opencv_filter2d"
    assert config["mask_smoothing_implementation"] == "langsplat_legacy"
    assert config["resize_policy"] == "error_on_mismatch"
    assert config["level_selection"] == "argmax activated raw-relevance peak"
    assert config["mask_threshold_comparison"] == "strict_greater_than"
    assert config["feature_resize"] == (
        "forbidden; rendered relevance and annotation mask shapes must match"
    )


def test_langsplat_legacy_smoothing_matches_released_loop_at_boundaries():
    mask = eval_features.torch.tensor(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
        ],
        dtype=eval_features.torch.uint8,
    )

    expected = mask.numpy().copy()
    height, width = expected.shape
    radius = 3
    for row in range(height):
        for col in range(width):
            window = mask.numpy()[
                max(0, row - radius) : min(row + radius + 1, height - 1),
                max(0, col - radius) : min(col + radius + 1, width - 1),
            ]
            expected[row, col] = eval_features.np.argmax(
                eval_features.np.bincount(window.reshape(-1))
            )

    actual = eval_features._smooth_mask(
        mask,
        kernel_size=7,
        implementation="langsplat_legacy",
    )

    assert eval_features.np.array_equal(actual.numpy(), expected.astype(bool))


def test_strict_profiles_reject_relevance_annotation_shape_mismatch():
    obj = eval_features.LerfObject(
        frame="frame_00002",
        query="cup",
        mask=eval_features.np.zeros((4, 4), dtype=bool),
        bboxes=[],
    )
    relevance = eval_features.np.zeros((1, 1, 2, 2), dtype=eval_features.np.float32)

    with pytest.raises(ValueError, match="relevance shape"):
        eval_features.evaluate_relevance_maps(
            {"frame_00002": [obj]},
            {"frame_00002": relevance},
            mask_thresh=0.5,
            activation_kernel=1,
            smooth_kernel=1,
            resize_policy="error_on_mismatch",
        )


def test_occam_camera_manifest_resolves_exact_name_and_records_split_role():
    manifest = validate_label_camera_roles(
        ["frame_00001", "frame_00002"],
        ["frame_00001.jpg"],
        ["frame_00002.jpg"],
    )

    assert manifest["frame_00001"] == {
        "resolved_camera_name": "frame_00001.jpg",
        "camera_role": "train",
    }
    assert manifest["frame_00002"] == {
        "resolved_camera_name": "frame_00002.jpg",
        "camera_role": "test",
    }
    with pytest.raises(ValueError, match="both train and test"):
        validate_label_camera_roles(
            ["frame_00002"],
            ["frame_00002.jpg"],
            ["frame_00002.jpg"],
        )
    with pytest.raises(ValueError, match="require-test-only"):
        validate_label_camera_roles(
            ["frame_00001"],
            ["frame_00001.jpg"],
            [],
            require_test_only=True,
        )


def test_occam_checkpoint_cfg_parser_is_non_executing(tmp_path):
    config = tmp_path / "cfg_args"
    config.write_text(
        "Namespace(source_path='/data/teatime', eval=False, feature_level=2)\n",
        encoding="utf-8",
    )

    assert _read_namespace_config(config) == {
        "source_path": "/data/teatime",
        "eval": False,
        "feature_level": 2,
    }

    config.write_text("__import__('os').system('false')\n", encoding="utf-8")
    with pytest.raises(ProtocolError):
        _read_namespace_config(config)
