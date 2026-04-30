import numpy as np

from radio_gs.scannet_constants import (
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
    compute_split_metrics,
    remap_nyu40_labels,
)


def test_opengaussian_splits_match_protocol_ids():
    assert OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36
    ]
    assert OPENGAUSSIAN_NYU40_CLASS_SPLITS["15"] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34
    ]
    assert OPENGAUSSIAN_NYU40_CLASS_SPLITS["10"] == [
        1, 2, 4, 5, 6, 7, 8, 9, 10, 33
    ]


def test_remap_nyu40_labels_ignores_zero_and_non_target_ids():
    labels = np.array([0, 1, 2, 39, 33, 34], dtype=np.int32)
    remapped, valid = remap_nyu40_labels(labels, OPENGAUSSIAN_NYU40_CLASS_SPLITS["10"])

    assert valid.tolist() == [False, True, True, False, True, False]
    assert remapped[valid].tolist() == [0, 1, 9]


def test_compute_split_metrics_averages_gt_present_classes():
    gt = np.array([1, 1, 2, 2, 33, 33], dtype=np.int32)
    pred = np.array([1, 2, 2, 2, 1, 33], dtype=np.int32)

    metrics = compute_split_metrics(
        pred_labels=pred,
        gt_labels=gt,
        split_ids=[1, 2, 33],
    )

    assert metrics["num_valid"] == 6
    assert np.isclose(metrics["per_class"]["1"]["iou"], 1 / 3)
    assert np.isclose(metrics["per_class"]["2"]["iou"], 2 / 3)
    assert np.isclose(metrics["per_class"]["33"]["iou"], 1 / 2)
    assert np.isclose(metrics["miou"], (1 / 3 + 2 / 3 + 1 / 2) / 3)


def test_compute_split_metrics_ignores_prediction_only_absent_gt_classes():
    gt = np.array([1, 1, 1, 2], dtype=np.int32)
    pred = np.array([1, 1, 33, 2], dtype=np.int32)

    metrics = compute_split_metrics(
        pred_labels=pred,
        gt_labels=gt,
        split_ids=[1, 2, 33],
    )

    assert metrics["per_class"]["33"]["iou"] is None
    assert np.isclose(metrics["miou"], (2 / 3 + 1.0) / 2)
