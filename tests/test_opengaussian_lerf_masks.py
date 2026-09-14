import numpy as np
from PIL import Image
import pytest
from radio_gs.scripts.evaluate_opengaussian_lerf_masks import FRAMES, score_masks


def test_public_metric_missing_prediction_empty_union_and_strict_cutoff(tmp_path):
    gt, pred = tmp_path / "gt", tmp_path / "pred"
    pred.mkdir()
    for index, frame in enumerate(FRAMES["figurines"]):
        folder = gt / f"frame_{frame:05d}"
        folder.mkdir(parents=True)
        target = np.full((16, 16), 255 if index < 3 else 0, dtype=np.uint8)
        Image.fromarray(target).save(folder / "object.jpg", quality=100)
        if index == 2:
            continue
        prediction = np.zeros_like(target)
        if index == 0:
            prediction[:8] = 255
        elif index == 1:
            prediction[:4] = 255
        Image.fromarray(prediction).save(pred / f"frame_{frame:05d}_object.png")
    result = score_masks(gt, pred, "figurines")
    assert result["mean_iou"] == pytest.approx(.1875)
    assert result["acc_at_025"] == .25
    assert result["acc_at_05"] == 0
    assert result["observations"][2]["prediction_missing"]


def test_missing_gt_cannot_silently_shrink_evaluation(tmp_path):
    with pytest.raises(FileNotFoundError, match="GT frame"):
        score_masks(tmp_path, tmp_path, "ramen")
