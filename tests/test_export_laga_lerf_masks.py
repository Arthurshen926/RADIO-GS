import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from radio_gs.scripts import export_laga_lerf_masks as export


def _write_label(path: Path, *, query: str = "frog cup") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {"width": 8, "height": 6, "name": f"{path.stem}.jpg"},
        "objects": [
            {
                "category": query,
                "bbox": [1, 1, 4, 4],
                "segmentation": [[1, 1], [4, 1], [4, 4], [1, 4]],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_root_matches_nested_lerf_evaluator_layout(tmp_path):
    root = export.prediction_root(tmp_path / "models", "ramen", "0.5")

    assert root == tmp_path / "models" / "ramen" / "predictions_mask_0.5" / "renders_silhouette"


def test_load_scene_frame_prompts_uses_official_lerf_frames(tmp_path):
    _write_label(tmp_path / "label" / "ramen" / "frame_00006.json", query="nori")
    _write_label(tmp_path / "label" / "ramen" / "frame_00999.json", query="ignored")

    prompts = export.load_scene_frame_prompts(tmp_path, "ramen")

    assert prompts == {"frame_00006": ["nori"]}


def test_camera_lookup_accepts_exact_name_and_suffix():
    cameras = [
        SimpleNamespace(image_name="frame_00006"),
        SimpleNamespace(image_name="frame_00024.jpg"),
    ]

    lookup = export.camera_lookup(cameras)

    assert export.resolve_camera(lookup, "frame_00006") is cameras[0]
    assert export.resolve_camera(lookup, "frame_00024") is cameras[1]


def test_scene_camera_lookup_includes_train_and_test_cameras():
    train_camera = SimpleNamespace(image_name="frame_00002.jpg")
    test_camera = SimpleNamespace(image_name="frame_00041.jpg")

    class Scene:
        def getTrainCameras(self):
            return [train_camera]

        def getTestCameras(self):
            return [test_camera]

    lookup = export.scene_camera_lookup(Scene())

    assert export.resolve_camera(lookup, "frame_00002") is train_camera
    assert export.resolve_camera(lookup, "frame_00041") is test_camera


def test_normalize_scores_is_stable_for_constant_and_nonconstant_inputs():
    constant = np.ones((4, 1), dtype=np.float32)
    varied = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

    assert np.array_equal(export.normalize_scores(constant), np.zeros_like(constant))
    normalized = export.normalize_scores(varied, clip_quantile=0.0)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
