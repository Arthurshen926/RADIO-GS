import json
from pathlib import Path

import torch

from radio_gs.scripts import build_direct3d_confidence_coverage_report as report


def _write_scene(
    root: Path,
    *,
    scene: str,
    categories: list[str],
    scores: torch.Tensor,
    mean_valid_views: float,
    registered_fraction: float,
    miou: float,
    details: list[dict],
) -> None:
    cache_path = root / f"{scene}_scores.pt"
    torch.save(
        {
            "version": 1,
            "scores": scores,
            "metadata": {"scene": scene},
            "registration_stats": {},
        },
        cache_path,
    )
    per_category = {}
    for cat in categories:
        cat_details = [row for row in details if row["category"] == cat]
        per_category[cat] = {
            "miou": sum(float(row["iou"]) for row in cat_details) / len(cat_details),
            "n": len(cat_details),
            "selected_gaussians": 3,
        }
    scene_dir = root / "results" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "lerf_direct_3d_selection_results.json").write_text(
        json.dumps(
            {
                "scene": {
                    "scene": scene,
                    "categories": categories,
                    "registration": {
                        "mean_valid_views": mean_valid_views,
                        "max_valid_views": mean_valid_views + 2,
                        "registered_fraction": registered_fraction,
                        "registered_gaussians": 10,
                        "total_gaussians": 20,
                        "num_frames": 5,
                    },
                    "score_cache": {"enabled": True, "path": str(cache_path), "status": "hit"},
                    "results": {
                        "thr0p25": {
                            "miou": miou,
                            "acc025": 0.5,
                            "n": len(details),
                            "per_category": per_category,
                            "query_details": details,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_build_summary_joins_view_coverage_and_teacher_score_confidence(tmp_path: Path) -> None:
    _write_scene(
        tmp_path,
        scene="low_view",
        categories=["cup", "bowl"],
        scores=torch.tensor([[0.10, 0.05], [0.20, 0.30], [0.80, 0.40], [0.70, 0.20]]),
        mean_valid_views=2.0,
        registered_fraction=0.25,
        miou=0.2,
        details=[
            {"frame": "frame_00001", "frame_id": 1, "category": "cup", "iou": 0.1, "pred_pixels": 0, "gt_pixels": 100, "selected_gaussians": 3},
            {"frame": "frame_00001", "frame_id": 1, "category": "bowl", "iou": 0.3, "pred_pixels": 10, "gt_pixels": 300, "selected_gaussians": 3},
        ],
    )
    _write_scene(
        tmp_path,
        scene="high_view",
        categories=["mug", "plate"],
        scores=torch.tensor([[0.95, 0.05], [0.85, 0.15], [0.05, 0.88], [0.10, 0.82]]),
        mean_valid_views=8.0,
        registered_fraction=0.75,
        miou=0.8,
        details=[
            {"frame": "frame_00002", "frame_id": 2, "category": "mug", "iou": 0.9, "pred_pixels": 20, "gt_pixels": 200, "selected_gaussians": 3},
            {"frame": "frame_00002", "frame_id": 2, "category": "plate", "iou": 0.7, "pred_pixels": 30, "gt_pixels": 400, "selected_gaussians": 3},
        ],
    )

    summary = report.build_summary(tmp_path / "results", selection="thr0p25")

    assert [row["scene"] for row in summary["scene_rows"]] == ["high_view", "low_view"]
    assert summary["scene_correlations"]["mean_valid_views_vs_miou"] == 1.0
    cup = next(row for row in summary["query_rows"] if row["category"] == "cup")
    assert cup["max_score"] == 0.8
    assert cup["top1pct_mean_score"] == 0.8
    assert cup["top1pct_mean_margin"] == 0.4
    bowl = next(row for row in summary["query_rows"] if row["category"] == "bowl")
    assert bowl["top1pct_mean_margin"] == -0.4
    assert summary["confidence_buckets"]["low"]["zero_prediction_rate"] == 0.5
    assert summary["confidence_buckets"]["high"]["mean_iou"] == 0.9
    assert summary["text_ambiguity_buckets"]["ambiguous"]["mean_iou"] == 0.2
    assert summary["text_ambiguity_buckets"]["distinct"]["mean_iou"] == 0.9


def test_build_markdown_includes_coverage_and_confidence_tables(tmp_path: Path) -> None:
    _write_scene(
        tmp_path,
        scene="scene",
        categories=["cup"],
        scores=torch.tensor([[0.1], [0.9]]),
        mean_valid_views=3.0,
        registered_fraction=0.5,
        miou=0.4,
        details=[
            {"frame": "frame_00001", "frame_id": 1, "category": "cup", "iou": 0.4, "pred_pixels": 2, "gt_pixels": 10, "selected_gaussians": 1}
        ],
    )
    summary = report.build_summary(tmp_path / "results", selection="thr0p25")

    markdown = report.build_markdown(summary)

    assert "Direct3D Confidence and Coverage Analysis" in markdown
    assert "## Scene View-Coverage" in markdown
    assert "## Teacher-Score Confidence Buckets" in markdown
    assert "## Text-Ambiguity Buckets" in markdown
    assert "| scene | 3.0000 |" in markdown


def test_build_latex_table_includes_scene_and_bucket_rows() -> None:
    summary = {
        "selection": "thr0p25",
        "top_score_ratio": 0.01,
        "scene_rows": [
            {
                "scene": "waldo_kitchen",
                "n": 22,
                "mean_valid_views": 5.7542,
                "miou": 0.2429,
                "acc025": 0.4091,
            }
        ],
        "confidence_buckets": {
            "low": {
                "n": 70,
                "mean_score": 0.18,
                "mean_iou": 0.4345,
                "acc025": 0.6143,
            }
        },
        "text_ambiguity_buckets": {
            "ambiguous": {
                "n": 70,
                "mean_score": 0.01,
                "mean_iou": 0.3000,
                "acc025": 0.5000,
            }
        },
    }

    latex = report.build_latex_table(summary)

    assert "\\label{tab:direct3d_confidence_coverage}" in latex
    assert "Scene & waldo\\_kitchen & 22 & 5.7542 & 0.2429 & 0.4091" in latex
    assert "Teacher bucket & low & 70 & 0.1800 & 0.4345 & 0.6143" in latex
    assert "Text margin & ambiguous & 70 & 0.0100 & 0.3000 & 0.5000" in latex


def test_write_outputs_records_json_and_markdown(tmp_path: Path) -> None:
    summary = {
        "selection": "thr0p25",
        "source_root": "root",
        "top_score_ratio": 0.01,
        "scene_rows": [],
        "query_rows": [],
        "confidence_buckets": {},
        "text_ambiguity_buckets": {},
        "scene_correlations": {},
        "low_confidence_failures": [],
        "ambiguous_failures": [],
    }

    paths = report.write_outputs(
        summary,
        tmp_path / "report.md",
        tmp_path / "report.json",
        tmp_path / "report.tex",
    )

    assert paths["markdown"].exists()
    assert paths["latex"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["selection"] == "thr0p25"
