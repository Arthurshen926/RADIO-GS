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


def test_evaluate_scene_reads_nested_frame_query_masks(tmp_path):
    from radio_gs.scripts import eval_drsplat_lerf_masks as eval_drsplat

    gt_root = tmp_path / "label" / "figurines" / "gt"
    pred_root = tmp_path / "predictions_mask_0.4" / "renders_silhouette"
    original_frames = eval_drsplat.SCENE_GT_FRAMES.copy()
    eval_drsplat.SCENE_GT_FRAMES.clear()
    eval_drsplat.SCENE_GT_FRAMES.update({"figurines": ["frame_00041"]})
    try:
        _write_mask(
            gt_root / "frame_00041" / "red apple.jpg",
            eval_lerf.np.array([[255, 0], [255, 0]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            gt_root / "frame_00041" / "green apple.jpg",
            eval_lerf.np.array([[255, 255], [0, 0]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            pred_root / "frame_00041" / "red apple.png",
            eval_lerf.np.array([[255, 0], [0, 255]], dtype=eval_lerf.np.uint8),
        )

        result = eval_drsplat.evaluate_scene(gt_root, pred_root, "figurines")
    finally:
        eval_drsplat.SCENE_GT_FRAMES.clear()
        eval_drsplat.SCENE_GT_FRAMES.update(original_frames)

    assert result.count == 2
    assert result.missing == 1
    assert result.miou == pytest.approx(1 / 6)
    assert result.acc025 == pytest.approx(0.5)
    assert result.acc05 == pytest.approx(0.0)
    by_query = {item.query: item for item in result.objects}
    assert by_query["red apple"].pred_path.endswith("frame_00041/red apple.png")
    assert by_query["green apple"].missing is True


def test_evaluate_run_resolves_suffixed_drsplat_scene_directory(tmp_path):
    from radio_gs.scripts import eval_drsplat_lerf_masks as eval_drsplat

    lerf_root = tmp_path / "lerf_ovs"
    pred_root = tmp_path / "drsplat_runs"
    out_path = tmp_path / "report.json"
    out_md_path = tmp_path / "report.md"
    original_frames = eval_drsplat.SCENE_GT_FRAMES.copy()
    eval_drsplat.SCENE_GT_FRAMES.clear()
    eval_drsplat.SCENE_GT_FRAMES.update({"figurines": ["frame_00041"]})
    try:
        _write_mask(
            lerf_root / "label" / "figurines" / "gt" / "frame_00041" / "red apple.jpg",
            eval_lerf.np.array([[255, 0], [0, 0]], dtype=eval_lerf.np.uint8),
        )
        _write_mask(
            pred_root
            / "figurines_1_lerfcompat_topk45_weight_128"
            / "none"
            / "predictions_mask_0.4"
            / "renders_silhouette"
            / "frame_00041"
            / "red apple.png",
            eval_lerf.np.array([[255, 0], [0, 0]], dtype=eval_lerf.np.uint8),
        )

        report = eval_drsplat.evaluate_run(
            lerf_root,
            pred_root,
            ["figurines"],
            mask_thresh="0.4",
            ablation_type="none",
        )
        exit_code = eval_drsplat.main(
            [
                "--lerf-root",
                str(lerf_root),
                "--pred-root",
                str(pred_root),
                "--scenes",
                "figurines",
                "--output-json",
                str(out_path),
                "--output-md",
                str(out_md_path),
            ]
        )
    finally:
        eval_drsplat.SCENE_GT_FRAMES.clear()
        eval_drsplat.SCENE_GT_FRAMES.update(original_frames)

    assert report["scenes"]["figurines"]["miou"] == pytest.approx(1.0)
    assert report["scenes"]["figurines"]["pred_root"].endswith("renders_silhouette")
    assert report["macro"]["count"] == 1
    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["macro"]["acc025"] == pytest.approx(1.0)
    markdown = out_md_path.read_text(encoding="utf-8")
    assert "Dr. Splat LERF Mask Summary" in markdown
    assert "| figurines | 1.0000 | 1.0000 | 1.0000 | 1 | 0 |" in markdown
    assert "| Macro | 1.0000 | 1.0000 | 1.0000 | 1 | 0 |" in markdown


def test_evaluate_scene_resizes_prediction_masks_to_gt_shape(tmp_path):
    from radio_gs.scripts import eval_drsplat_lerf_masks as eval_drsplat

    gt_root = tmp_path / "label" / "figurines" / "gt"
    pred_root = tmp_path / "predictions_mask_0.4" / "renders_silhouette"
    original_frames = eval_drsplat.SCENE_GT_FRAMES.copy()
    eval_drsplat.SCENE_GT_FRAMES.clear()
    eval_drsplat.SCENE_GT_FRAMES.update({"figurines": ["frame_00041"]})
    try:
        _write_mask(
            gt_root / "frame_00041" / "red apple.jpg",
            eval_lerf.np.array(
                [
                    [255, 255, 0, 0],
                    [255, 255, 0, 0],
                    [255, 255, 0, 0],
                    [255, 255, 0, 0],
                ],
                dtype=eval_lerf.np.uint8,
            ),
        )
        _write_mask(
            pred_root / "frame_00041" / "red apple.png",
            eval_lerf.np.array([[255, 0], [255, 0]], dtype=eval_lerf.np.uint8),
        )

        result = eval_drsplat.evaluate_scene(gt_root, pred_root, "figurines")
    finally:
        eval_drsplat.SCENE_GT_FRAMES.clear()
        eval_drsplat.SCENE_GT_FRAMES.update(original_frames)

    assert result.count == 1
    assert result.missing == 0
    assert result.miou == pytest.approx(1.0)
    assert result.acc025 == pytest.approx(1.0)
    assert result.acc05 == pytest.approx(1.0)
