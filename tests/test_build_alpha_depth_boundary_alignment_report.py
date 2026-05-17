from __future__ import annotations

import json
from pathlib import Path

from radio_gs.scripts import build_alpha_depth_boundary_alignment_report as report


def _write_scene(path: Path, scene: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scene": {
                    "scene": scene,
                    "results": {
                        "thr0p25": {
                            "miou": 0.5,
                            "boundary_f": 0.5,
                            "trimap_iou": 0.4,
                            "n": len(rows),
                            "query_details": rows,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_sweep(path: Path, source: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "pad16",
                        "fixed_global_threshold": {
                            "selection": "thr0p25",
                            "macro_miou": 0.5,
                            "macro_boundary_f": 0.5,
                            "macro_trimap_iou": 0.4,
                            "rows": [
                                {
                                    "scene": "figurines",
                                    "selection": "thr0p25",
                                    "miou": 0.5,
                                    "boundary_f": 0.5,
                                    "trimap_iou": 0.4,
                                    "n": 3,
                                    "source": str(source),
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_summary_computes_geometry_alignment_when_query_metrics_exist(tmp_path: Path) -> None:
    scene_json = tmp_path / "figurines" / "lerf_direct_3d_selection_results.json"
    sweep = tmp_path / "sweep.json"
    _write_scene(
        scene_json,
        "figurines",
        [
            {
                "frame": "frame_00001",
                "category": "apple",
                "iou": 0.9,
                "boundary_f": 0.9,
                "trimap_iou": 0.7,
                "geometry_valid": 1,
                "discontinuity_gt_boundary_mean": 0.8,
                "discontinuity_error_boundary_mean": 0.1,
                "alpha_edge_gt_boundary_mean": 0.7,
                "depth_edge_gt_boundary_mean": 0.2,
                "geometry_overlay_path": "geometry_overlays/thr0p25/figurines/frame_00001_apple.png",
            },
            {
                "frame": "frame_00002",
                "category": "pear",
                "iou": 0.4,
                "boundary_f": 0.4,
                "trimap_iou": 0.3,
                "geometry_valid": 1,
                "discontinuity_gt_boundary_mean": 0.5,
                "discontinuity_error_boundary_mean": 0.6,
                "alpha_edge_gt_boundary_mean": 0.4,
                "depth_edge_gt_boundary_mean": 0.5,
            },
            {
                "frame": "frame_00003",
                "category": "chair",
                "iou": 0.1,
                "boundary_f": 0.1,
                "trimap_iou": 0.1,
                "geometry_valid": 1,
                "discontinuity_gt_boundary_mean": 0.2,
                "discontinuity_error_boundary_mean": 0.9,
                "alpha_edge_gt_boundary_mean": 0.1,
                "depth_edge_gt_boundary_mean": 0.8,
            },
        ],
    )
    _write_sweep(sweep, scene_json)

    summary = report.build_summary(sweep, run_label="pad16", selection="thr0p25")

    assert summary["geometry_alignment_status"] == "available"
    assert summary["query_count"] == 3
    assert summary["geometry_query_count"] == 3
    assert summary["query_correlations"]["boundary_error_vs_discontinuity_error_boundary_mean"] > 0.9
    assert summary["discontinuity_buckets"]["low"]["count"] == 1
    assert summary["worst_geometry_cases"][0]["category"] == "chair"


def test_summary_marks_not_available_without_geometry_query_metrics(tmp_path: Path) -> None:
    scene_json = tmp_path / "figurines" / "lerf_direct_3d_selection_results.json"
    sweep = tmp_path / "sweep.json"
    _write_scene(
        scene_json,
        "figurines",
        [
            {
                "frame": "frame_00001",
                "category": "apple",
                "iou": 0.9,
                "boundary_f": 0.9,
                "trimap_iou": 0.7,
            }
        ],
    )
    _write_sweep(sweep, scene_json)

    summary = report.build_summary(sweep, run_label="pad16", selection="thr0p25")
    markdown = report.build_markdown(summary)

    assert summary["geometry_alignment_status"] == "not_available"
    assert summary["geometry_query_count"] == 0
    assert "Alpha/depth geometry maps are not available" in markdown


def test_write_outputs_records_all_formats(tmp_path: Path) -> None:
    summary = {
        "sweep_source": "sweep.json",
        "run_label": "pad16",
        "selection": "thr0p25",
        "geometry_alignment_status": "not_available",
        "query_count": 0,
        "geometry_query_count": 0,
        "scene_rows": [],
        "query_correlations": {},
        "discontinuity_buckets": {},
        "worst_geometry_cases": [],
    }

    paths = report.write_outputs(
        summary,
        tmp_path / "report.md",
        tmp_path / "report.json",
        tmp_path / "report.tex",
    )

    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert paths["latex"].exists()
