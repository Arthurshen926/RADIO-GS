from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from radio_gs.scripts import build_alpha_depth_case_figure as figure


def test_select_cases_keeps_scene_coverage_before_filling() -> None:
    cases = [
        {"scene": "A", "category": "a1", "boundary_error": 1.0, "discontinuity_error_boundary_mean": 0.9},
        {"scene": "A", "category": "a2", "boundary_error": 1.0, "discontinuity_error_boundary_mean": 0.8},
        {"scene": "B", "category": "b1", "boundary_error": 0.9, "discontinuity_error_boundary_mean": 0.7},
        {"scene": "C", "category": "c1", "boundary_error": 0.8, "discontinuity_error_boundary_mean": 0.6},
    ]

    selected = figure.select_cases(cases, max_cases=3)

    assert [case["category"] for case in selected] == ["a1", "b1", "c1"]


def test_build_manifest_resolves_overlay_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    overlay = source_root / "geometry_overlays" / "thr0p25" / "scene" / "case.png"
    overlay.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(overlay)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "worst_geometry_cases": [
                    {
                        "scene": "Scene",
                        "frame": "frame_00001",
                        "category": "object",
                        "iou": 0.1,
                        "boundary_error": 0.9,
                        "discontinuity_error_boundary_mean": 0.8,
                        "geometry_overlay_path": "geometry_overlays/thr0p25/scene/case.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = figure.build_manifest(report, source_root=source_root, max_cases=1)

    assert manifest["cases"][0]["overlay_path"].endswith("case.png")
    assert Path(manifest["cases"][0]["overlay_path"]).exists()


def test_write_case_figure_outputs_image_and_manifest(tmp_path: Path) -> None:
    overlay = tmp_path / "case.png"
    Image.new("RGB", (24, 16), "blue").save(overlay)
    manifest = {
        "cases": [
            {
                "scene": "Scene",
                "frame": "frame_00001",
                "category": "object",
                "iou": 0.1,
                "boundary_error": 0.9,
                "overlay_path": str(overlay),
            }
        ]
    }

    paths = figure.write_case_figure(
        manifest,
        output_png=tmp_path / "figure.png",
        output_json=tmp_path / "manifest.json",
        output_md=tmp_path / "manifest.md",
        tile_width=64,
    )

    assert paths["png"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["cases"][0]["category"] == "object"
    assert "object" in paths["markdown"].read_text(encoding="utf-8")
