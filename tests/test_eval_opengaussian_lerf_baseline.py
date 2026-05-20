import json

import pytest
from PIL import Image

from radio_gs.scripts import eval_opengaussian_lerf_baseline as eval_lerf


def _write_mask(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels, mode="L")
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=100, subsampling=0)
    else:
        image.save(path)


def test_evaluate_scene_counts_missing_prediction_as_zero(tmp_path):
    gt_root = tmp_path / "label" / "figurines" / "gt"
    pred_root = tmp_path / "pred"
    eval_lerf.SCENE_GT_FRAMES["figurines"] = ["frame_00041"]

    _write_mask(gt_root / "frame_00041" / "object_a.jpg", eval_lerf.np.array([[255, 0], [255, 0]], dtype=eval_lerf.np.uint8))
    _write_mask(gt_root / "frame_00041" / "object_b.jpg", eval_lerf.np.array([[255, 255], [0, 0]], dtype=eval_lerf.np.uint8))
    _write_mask(pred_root / "frame_00041_object_a.png", eval_lerf.np.array([[255, 0], [0, 255]], dtype=eval_lerf.np.uint8))

    result = eval_lerf.evaluate_scene(gt_root, pred_root, "figurines")

    assert result.count == 2
    assert result.missing == 1
    assert result.miou == pytest.approx(1 / 6)
    assert result.acc025 == pytest.approx(0.5)
    assert result.acc05 == pytest.approx(0.0)
    assert result.objects[1].iou == 0.0
    assert result.objects[1].missing is True


def test_evaluate_run_computes_macro_across_scenes(tmp_path):
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    original_frames = eval_lerf.SCENE_GT_FRAMES.copy()
    eval_lerf.SCENE_GT_FRAMES.clear()
    eval_lerf.SCENE_GT_FRAMES.update(
        {
            "figurines": ["frame_00041"],
            "ramen": ["frame_00006"],
        }
    )
    try:
        _write_mask(
            lerf_root / "label" / "figurines" / "gt" / "frame_00041" / "object.jpg",
            eval_lerf.np.array([[255, 0], [0, 0]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            model_root / "figurines" / "text2obj" / "ours_70000" / "renders_cluster_silhouette" / "frame_00041_object.png",
            eval_lerf.np.array([[255, 0], [0, 0]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            lerf_root / "label" / "ramen" / "gt" / "frame_00006" / "object.jpg",
            eval_lerf.np.array([[255, 255], [0, 0]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            model_root / "ramen" / "text2obj" / "ours_70000" / "renders_cluster_silhouette" / "frame_00006_object.png",
            eval_lerf.np.array([[0, 0], [255, 255]], dtype=eval_lerf.np.uint8),
        )

        report = eval_lerf.evaluate_run(lerf_root, model_root, ["figurines", "ramen"], iteration=70000)
    finally:
        eval_lerf.SCENE_GT_FRAMES.clear()
        eval_lerf.SCENE_GT_FRAMES.update(original_frames)

    assert report["scenes"]["figurines"]["miou"] == pytest.approx(1.0)
    assert report["scenes"]["ramen"]["miou"] == pytest.approx(0.0)
    assert report["macro"]["miou"] == pytest.approx(0.5)
    assert report["macro"]["acc025"] == pytest.approx(0.5)
    assert report["macro"]["count"] == 2


def test_cli_writes_json_report(tmp_path):
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    out_path = tmp_path / "report.json"
    original_frames = eval_lerf.SCENE_GT_FRAMES.copy()
    eval_lerf.SCENE_GT_FRAMES.clear()
    eval_lerf.SCENE_GT_FRAMES.update({"figurines": ["frame_00041"]})
    try:
        _write_mask(
            lerf_root / "label" / "figurines" / "gt" / "frame_00041" / "object.jpg",
            eval_lerf.np.array([[255]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            model_root / "figurines" / "text2obj" / "ours_3" / "renders_cluster_silhouette" / "frame_00041_object.png",
            eval_lerf.np.array([[255]], dtype=eval_lerf.np.uint8),
        )

        exit_code = eval_lerf.main(
            [
                "--lerf-root",
                str(lerf_root),
                "--model-root",
                str(model_root),
                "--scenes",
                "figurines",
                "--iteration",
                "3",
                "--output-json",
                str(out_path),
            ]
        )
    finally:
        eval_lerf.SCENE_GT_FRAMES.clear()
        eval_lerf.SCENE_GT_FRAMES.update(original_frames)

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["scenes"]["figurines"]["miou"] == 1.0
    assert payload["macro"]["count"] == 1


def test_evaluate_run_rasterizes_local_polygon_json_labels(tmp_path):
    lerf_root = tmp_path / "lerf_ovs"
    model_root = tmp_path / "models"
    original_frames = eval_lerf.SCENE_GT_FRAMES.copy()
    eval_lerf.SCENE_GT_FRAMES.clear()
    eval_lerf.SCENE_GT_FRAMES.update({"figurines": ["frame_00041"]})
    try:
        label_root = lerf_root / "label" / "figurines"
        label_root.mkdir(parents=True)
        (label_root / "frame_00041.json").write_text(
            json.dumps(
                {
                    "info": {"width": 2, "height": 2},
                    "objects": [
                        {
                            "category": "red apple",
                            "segmentation": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_mask(
            model_root / "figurines" / "text2obj" / "ours_70000" / "renders_cluster_silhouette" / "frame_00041_red apple.png",
            eval_lerf.np.array([[255, 255], [255, 255]], dtype=eval_lerf.np.uint8),
        )

        report = eval_lerf.evaluate_run(lerf_root, model_root, ["figurines"], iteration=70000)
    finally:
        eval_lerf.SCENE_GT_FRAMES.clear()
        eval_lerf.SCENE_GT_FRAMES.update(original_frames)

    assert report["scenes"]["figurines"]["count"] == 1
    assert report["scenes"]["figurines"]["miou"] == pytest.approx(1.0)
    assert report["macro"]["acc025"] == pytest.approx(1.0)
