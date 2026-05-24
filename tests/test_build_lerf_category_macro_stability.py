from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.scripts import build_lerf_category_macro_stability as stability


def _write_lerf_result(path: Path, scene: str, rendered: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "scenes": {
                    scene: {
                        "rendered": rendered,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_summary_reports_scene_sample_and_category_macro(tmp_path: Path) -> None:
    scene_a = tmp_path / "scene_a.json"
    scene_b = tmp_path / "scene_b.json"
    _write_lerf_result(
        scene_a,
        "scene_a",
        {
            "loc_acc": 0.8,
            "miou": 0.6,
            "loc_total": 3,
            "n_iou_samples": 3,
            "per_category": {
                "cat1": {"loc_acc": 1.0, "miou": 0.8, "n_samples": 2},
                "cat2": {"loc_acc": 0.0, "miou": 0.2, "n_samples": 1},
            },
        },
    )
    _write_lerf_result(
        scene_b,
        "scene_b",
        {
            "loc_acc": 0.6,
            "miou": 0.4,
            "loc_total": 4,
            "n_iou_samples": 4,
            "per_category": {
                "cat1": {"loc_acc": 0.0, "miou": 0.4, "n_samples": 1},
                "cat3": {"loc_acc": 1.0, "miou": 0.6, "n_samples": 3},
            },
        },
    )

    summary = stability.build_summary([scene_a, scene_b], readout="rendered", bootstrap_iters=100, seed=7)

    assert summary["scene_macro"]["miou"] == pytest.approx(0.5)
    assert summary["sample_weighted"]["miou"] == pytest.approx((0.6 * 3 + 0.4 * 4) / 7, abs=1e-4)
    assert summary["scene_category_macro"]["miou"] == pytest.approx(0.5)
    assert summary["category_rows"][0]["category"] == "cat1"
    assert summary["category_rows"][0]["miou"] == pytest.approx((0.8 * 2 + 0.4) / 3, abs=1e-4)
    assert summary["worst_categories"][0]["category"] == "cat2"
    assert summary["stability"]["category_minus_sample_miou"] == pytest.approx(
        0.5 - ((0.6 * 3 + 0.4 * 4) / 7),
        abs=1e-4,
    )
    assert len(summary["bootstrap"]["scene_macro_miou_ci95"]) == 2


def test_write_reports_outputs_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "scene.json"
    _write_lerf_result(
        source,
        "scene",
        {
            "loc_acc": 1.0,
            "miou": 0.5,
            "loc_total": 2,
            "n_iou_samples": 2,
            "per_category": {
                "object": {"loc_acc": 1.0, "miou": 0.5, "n_samples": 2},
            },
        },
    )

    summary = stability.build_summary([source], readout="rendered", bootstrap_iters=10, seed=1)
    paths = stability.write_reports(
        summary,
        output_json=tmp_path / "stability.json",
        output_md=tmp_path / "stability.md",
    )

    assert paths["json"].exists()
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Category-Macro Stability" in markdown
    assert "object" in markdown
