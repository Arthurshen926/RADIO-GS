import json
from pathlib import Path

import pytest

from radio_gs.scripts import build_lerf_query_breakdown as breakdown


def _write_scene(path: Path, scene: str, details):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scene": {
                    "scene": scene,
                    "image_height": 100,
                    "image_width": 100,
                    "results": {"thr0p25": {"query_details": details}},
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_breakdown_groups_by_footprint_and_label_tags(tmp_path):
    scene_path = tmp_path / "figurines.json"
    _write_scene(
        scene_path,
        "figurines",
        [
            {
                "frame": "frame_00041",
                "category": "glass of water",
                "iou": 0.5,
                "gt_pixels": 50,
                "selected_gaussians": 10,
            },
            {
                "frame": "frame_00041",
                "category": "red cup",
                "iou": 0.25,
                "gt_pixels": 500,
                "selected_gaussians": 30,
            },
            {
                "frame": "frame_00042",
                "category": "bear nose",
                "iou": 0.0,
                "gt_pixels": 5000,
                "selected_gaussians": 5,
            },
        ],
    )

    summary = breakdown.build_summary([scene_path], selection="thr0p25")

    assert summary["object_weighted"]["count"] == 3
    assert summary["object_weighted"]["miou"] == pytest.approx(0.25)
    assert summary["scene_mean"]["scene_count"] == 1
    assert summary["scene_mean"]["miou"] == pytest.approx(0.25)
    assert summary["footprint_bins"]["tiny"]["count"] == 1
    assert summary["footprint_bins"]["small"]["count"] == 1
    assert summary["footprint_bins"]["large"]["count"] == 1
    assert summary["label_groups"]["reflective_or_transparent"]["count"] == 1
    assert summary["label_groups"]["container_or_part"]["count"] == 2


def test_scene_mean_is_scene_weighted_not_object_weighted(tmp_path):
    scene_a = tmp_path / "scene_a.json"
    scene_b = tmp_path / "scene_b.json"
    _write_scene(
        scene_a,
        "figurines",
        [
            {"frame": "frame_00001", "category": "object a", "iou": 0.0, "gt_pixels": 100, "selected_gaussians": 1},
            {"frame": "frame_00002", "category": "object b", "iou": 0.0, "gt_pixels": 100, "selected_gaussians": 1},
            {"frame": "frame_00003", "category": "object c", "iou": 1.0, "gt_pixels": 100, "selected_gaussians": 1},
        ],
    )
    _write_scene(
        scene_b,
        "ramen",
        [
            {"frame": "frame_00004", "category": "object d", "iou": 1.0, "gt_pixels": 100, "selected_gaussians": 1},
        ],
    )

    summary = breakdown.build_summary([scene_a, scene_b], selection="thr0p25")

    assert summary["object_weighted"]["count"] == 4
    assert summary["object_weighted"]["miou"] == pytest.approx(0.5)
    assert summary["object_weighted"]["acc025"] == pytest.approx(0.5)
    assert summary["scene_mean"]["scene_count"] == 2
    assert summary["scene_mean"]["miou"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert summary["scene_mean"]["acc025"] == pytest.approx((1 / 3 + 1.0) / 2)


def test_cli_writes_json_and_markdown(tmp_path):
    scene_path = tmp_path / "ramen.json"
    out_json = tmp_path / "breakdown.json"
    out_md = tmp_path / "breakdown.md"
    _write_scene(
        scene_path,
        "ramen",
        [
            {
                "frame": "frame_00006",
                "category": "wavy noodles",
                "iou": 0.75,
                "gt_pixels": 1500,
                "selected_gaussians": 20,
            }
        ],
    )

    exit_code = breakdown.main(
        [
            "--inputs",
            str(scene_path),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    markdown = out_md.read_text(encoding="utf-8")
    assert payload["object_weighted"]["count"] == 1
    assert payload["scene_mean"]["scene_count"] == 1
    assert "LERF Query Breakdown" in markdown
    assert "object-weighted mIoU" in markdown
    assert "scene-mean mIoU" in markdown
    assert "| texture_like | 1 | 0.7500 |" in markdown
