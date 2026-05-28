import json

import cv2
import numpy as np

from radio_gs.scripts import compose_lerf_main_qualitative as compose


def _write_label(path, category):
    payload = {
        "info": {"height": 4, "width": 4},
        "objects": [
            {
                "category": category,
                "segmentation": [[1, 1], [3, 1], [3, 3], [1, 3]],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_make_case_row_loads_baseline_and_compact_masks(tmp_path):
    label_root = tmp_path / "labels"
    scene_dir = label_root / "figurines"
    scene_dir.mkdir(parents=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :] = (20, 30, 40)
    cv2.imwrite(str(scene_dir / "frame_00001.jpg"), rgb)
    _write_label(scene_dir / "frame_00001.json", "green apple")

    baseline_root = tmp_path / "drsplat"
    baseline_dir = (
        baseline_root
        / compose.DR_SPLAT_SCENE_DIR["figurines"]
        / "predictions_mask_0.4"
        / "renders_silhouette"
        / "frame_00001"
    )
    baseline_dir.mkdir(parents=True)
    baseline_mask = np.zeros((4, 4), dtype=np.uint8)
    baseline_mask[1:4, 1:4] = 255
    cv2.imwrite(str(baseline_dir / "green apple.png"), baseline_mask)

    ours_root = tmp_path / "ours"
    ours_dir = ours_root / "pred_masks" / "thr0p35" / "figurines"
    ours_dir.mkdir(parents=True)
    ours_mask = np.zeros((4, 4), dtype=np.uint8)
    ours_mask[1:4, 1:3] = 255
    cv2.imwrite(str(ours_dir / "frame_00001_green apple.png"), ours_mask)

    row, manifest = compose.make_case_row(
        compose.QualCase("figurines", "00001", "green apple"),
        label_root=label_root,
        baseline_root=baseline_root,
        baseline="dr_splat",
        baseline_label="Dr. Splat",
        ours_root=ours_root,
        ours_selection="thr0p35",
        panel_width=40,
        panel_height=30,
    )

    assert row.shape == (30, 160, 3)
    assert manifest["baseline_iou"] == 1.0
    assert manifest["ours_iou"] == 0.6667
    assert manifest["baseline_mask"].endswith("green apple.png")
    assert manifest["ours_mask"].endswith("frame_00001_green apple.png")


def test_parse_cases_rejects_malformed_case():
    try:
        compose.parse_cases(["figurines:00152"])
    except ValueError as exc:
        assert "scene:frame:query" in str(exc)
    else:
        raise AssertionError("malformed case was accepted")
