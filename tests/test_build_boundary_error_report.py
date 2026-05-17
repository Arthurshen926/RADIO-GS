from __future__ import annotations

import json
from pathlib import Path

from radio_gs.scripts import build_boundary_error_report as report


def _write_scene(path: Path, scene: str, queries: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    miou = sum(float(row["iou"]) for row in queries) / len(queries)
    boundary = sum(float(row["boundary_f"]) for row in queries) / len(queries)
    trimap = sum(float(row["trimap_iou"]) for row in queries) / len(queries)
    path.write_text(
        json.dumps(
            {
                "scene": {
                    "scene": scene,
                    "results": {
                        "thr0p25": {
                            "miou": miou,
                            "boundary_f": boundary,
                            "trimap_iou": trimap,
                            "n": len(queries),
                            "query_details": queries,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_sweep(path: Path, fig_source: Path, waldo_source: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "pad16",
                        "source_root": "fake_pad16",
                        "fixed_global_threshold": {
                            "selection": "thr0p25",
                            "macro_miou": 0.5,
                            "macro_boundary_f": 0.5,
                            "macro_trimap_iou": 0.425,
                            "rows": [
                                {
                                    "scene": "figurines",
                                    "selection": "thr0p25",
                                    "miou": 0.7,
                                    "boundary_f": 0.8,
                                    "trimap_iou": 0.6,
                                    "n": 2,
                                    "source": str(fig_source),
                                },
                                {
                                    "scene": "waldo_kitchen",
                                    "selection": "thr0p25",
                                    "miou": 0.3,
                                    "boundary_f": 0.2,
                                    "trimap_iou": 0.25,
                                    "n": 2,
                                    "source": str(waldo_source),
                                },
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_build_summary_extracts_scene_and_query_boundary_errors(tmp_path: Path) -> None:
    fig = tmp_path / "fig" / "lerf_direct_3d_selection_results.json"
    waldo = tmp_path / "waldo" / "lerf_direct_3d_selection_results.json"
    sweep = tmp_path / "sweep.json"
    _write_scene(
        fig,
        "figurines",
        [
            {
                "frame": "frame_00001",
                "category": "apple",
                "iou": 0.8,
                "boundary_f": 0.9,
                "trimap_iou": 0.7,
                "gt_pixels": 100,
                "pred_pixels": 100,
                "overselect_ratio": 1.0,
            },
            {
                "frame": "frame_00002",
                "category": "pear",
                "iou": 0.6,
                "boundary_f": 0.7,
                "trimap_iou": 0.5,
                "gt_pixels": 100,
                "pred_pixels": 300,
                "overselect_ratio": 3.0,
            },
        ],
    )
    _write_scene(
        waldo,
        "waldo_kitchen",
        [
            {
                "frame": "frame_00003",
                "category": "knife",
                "iou": 0.2,
                "boundary_f": 0.1,
                "trimap_iou": 0.2,
                "gt_pixels": 10,
                "pred_pixels": 2,
                "overselect_ratio": 0.2,
            },
            {
                "frame": "frame_00004",
                "category": "sink",
                "iou": 0.4,
                "boundary_f": 0.3,
                "trimap_iou": 0.3,
                "gt_pixels": 1000,
                "pred_pixels": 1000,
                "overselect_ratio": 1.0,
            },
        ],
    )
    _write_sweep(sweep, fig, waldo)

    summary = report.build_summary(sweep, run_label="pad16", selection="thr0p25")

    assert summary["scene_rows"][0]["scene"] == "Figurines"
    assert summary["scene_rows"][0]["boundary_error"] == 0.2
    assert summary["scene_rows"][1]["trimap_error"] == 0.75
    assert summary["query_count"] == 4
    assert summary["query_correlations"]["iou_vs_boundary_f"] > 0.9
    assert summary["overselect_buckets"]["under"]["count"] == 1
    assert summary["overselect_buckets"]["balanced"]["count"] == 2
    assert summary["overselect_buckets"]["over"]["count"] == 1
    assert summary["area_buckets"]["small"]["count"] == 2
    assert summary["worst_boundary_cases"][0]["category"] == "knife"


def test_markdown_and_latex_state_alpha_depth_limitation(tmp_path: Path) -> None:
    summary = {
        "sweep_source": "sweep.json",
        "run_label": "pad16",
        "selection": "thr0p25",
        "macro": {"miou": 0.5, "boundary_f": 0.5, "trimap_iou": 0.4},
        "scene_rows": [],
        "query_count": 0,
        "query_correlations": {},
        "overselect_buckets": {},
        "area_buckets": {},
        "worst_boundary_cases": [],
        "alpha_depth_status": "not_available",
    }

    markdown = report.build_markdown(summary)
    latex = report.build_latex_table(summary)

    assert "Boundary Error Readout" in markdown
    assert "Alpha/depth discontinuity maps are not present" in markdown
    assert "\\label{tab:boundary_error_readout}" in latex


def test_write_outputs_records_all_formats(tmp_path: Path) -> None:
    summary = {
        "sweep_source": "sweep.json",
        "run_label": "pad16",
        "selection": "thr0p25",
        "macro": {},
        "scene_rows": [],
        "query_count": 0,
        "query_correlations": {},
        "overselect_buckets": {},
        "area_buckets": {},
        "worst_boundary_cases": [],
        "alpha_depth_status": "not_available",
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
