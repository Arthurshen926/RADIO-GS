import csv
import json
from pathlib import Path

import pytest

from radio_gs.scripts.aggregate_scannet_pointcloud_results import (
    DEFAULT_TARGETS,
    aggregate_results,
    render_markdown,
    write_csv,
)


def _write_result(
    root: Path,
    dirname: str,
    scene: str,
    split_values: dict[str, tuple[float, float, int]],
    *,
    query_mode: str = "gaussian_index",
    opacity_mode: str = "label_index",
) -> Path:
    result_dir = root / dirname
    result_dir.mkdir(parents=True)
    payload = {
        "timestamp": "2026-04-30 12:00:00",
        "args": {"scene": scene},
        "macro": {},
        "scenes": {
            scene: {
                "scene": scene,
                "query_mode": query_mode,
                "opacity_filter": {"mode": opacity_mode, "enabled": True},
                "splits": {
                    split: {"miou": miou, "macc": macc, "num_valid": num_valid}
                    for split, (miou, macc, num_valid) in split_values.items()
                },
            }
        },
    }
    path = result_dir / "scannet_pointcloud_radio_gs_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregates_json_macro_official_flags_and_targets(tmp_path):
    _write_result(
        tmp_path,
        "scene0001_00_v43fair_best_hybrid_pointwise_gidx_official",
        "scene0001_00",
        {
            "19": (0.40, 0.50, 19),
            "15": (0.20, 0.30, 15),
            "10": (0.50, 0.60, 10),
        },
    )
    _write_result(
        tmp_path,
        "scene0002_00_v43fair_best_hybrid_pointwise_gidx_official",
        "scene0002_00",
        {
            "19": (0.20, 0.40, 19),
            "15": (0.50, 0.70, 15),
            "10": (0.30, 0.40, 10),
        },
        query_mode="knn",
        opacity_mode="query_top1",
    )

    summary = aggregate_results(
        eval_root=tmp_path,
        patterns=["*_v43fair_best_hybrid_pointwise_gidx_official"],
        targets=DEFAULT_TARGETS,
        require_official=True,
        expected_scenes=["scene0001_00", "scene0002_00", "scene0003_00"],
    )

    assert [row.scene for row in summary.rows] == ["scene0001_00", "scene0002_00"]
    assert summary.macro["19"]["miou"] == pytest.approx(0.30)
    assert summary.macro["15"]["macc"] == pytest.approx(0.50)
    assert summary.macro["10"]["num_valid"] == pytest.approx(10)
    assert summary.rows[0].split_metrics["19"].passes_target is True
    assert summary.rows[1].split_metrics["19"].passes_target is False
    assert summary.rows[0].official_ok is True
    assert summary.rows[1].official_ok is False
    assert summary.rows[1].official_issues == [
        "query_mode=knn",
        "opacity_filter.mode=query_top1",
    ]
    assert summary.missing_scenes == ["scene0003_00"]


def test_markdown_and_csv_include_split_fields_and_protocol_note(tmp_path):
    _write_result(
        tmp_path,
        "scene0001_00_v43fair_best_hybrid_pointwise_gidx_official",
        "scene0001_00",
        {
            "19": (0.3052, 0.50, 19),
            "15": (0.3149, 0.30, 15),
            "10": (0.4001, 0.60, 10),
        },
    )
    summary = aggregate_results(
        eval_root=tmp_path,
        patterns=["*_v43fair_best_hybrid_pointwise_gidx_official"],
        targets=DEFAULT_TARGETS,
        require_official=True,
        expected_scenes=[],
    )

    markdown = render_markdown(
        summary,
        protocol_note="v43 is a label-supervised diagnostic / upper-bound.",
    )
    assert "v43 is a label-supervised diagnostic / upper-bound." in markdown
    assert "| scene | split19 mIoU | split19 mAcc | split19 num_valid | split19 target |" in markdown
    assert "| scene0001_00 | 0.3052 | 0.5000 | 19 | pass |" in markdown
    assert "user/local target" in markdown
    assert "Missing expected scenes: none" in markdown

    csv_path = tmp_path / "summary.csv"
    write_csv(summary, csv_path)
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert rows[0]["scene"] == "scene0001_00"
    assert rows[0]["split19_miou"] == "0.3052"
    assert rows[0]["split15_passes_target"] == "false"
    assert rows[0]["split10_passes_target"] == "true"
    assert rows[0]["official_ok"] == "true"
