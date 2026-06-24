import json

import pytest

from radio_gs.scripts import eval_prerendered_lerf_features as eval_features


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
