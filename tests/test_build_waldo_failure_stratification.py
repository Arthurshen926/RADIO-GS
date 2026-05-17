import json
from pathlib import Path

from radio_gs.scripts import build_waldo_failure_stratification as strat


def _write_result(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scene": {
                    "scene": "waldo_kitchen",
                    "results": {
                        "thr0p25": {
                            "miou": 0.25,
                            "acc025": 0.4,
                            "query_details": [
                                {
                                    "category": "spoon",
                                    "iou": 0.0,
                                    "pred_pixels": 0,
                                    "gt_pixels": 900,
                                    "overselect_ratio": 0.0,
                                },
                                {
                                    "category": "knife",
                                    "iou": 0.2,
                                    "pred_pixels": 2000,
                                    "gt_pixels": 6000,
                                    "overselect_ratio": 0.33,
                                },
                                {
                                    "category": "yellow desk",
                                    "iou": 0.1,
                                    "pred_pixels": 12000,
                                    "gt_pixels": 30000,
                                    "overselect_ratio": 0.4,
                                },
                                {
                                    "category": "pot",
                                    "iou": 0.0,
                                    "pred_pixels": 0,
                                    "gt_pixels": 20000,
                                    "overselect_ratio": 0.0,
                                },
                            ],
                        }
                    },
                }
            }
        )
    )


def test_summarize_stratifies_by_object_size_and_zero_predictions(tmp_path: Path) -> None:
    result = tmp_path / "waldo.json"
    _write_result(result)

    summary = strat.summarize_result(result, selection="thr0p25")

    assert summary["scene"] == "waldo_kitchen"
    assert summary["query_count"] == 4
    assert summary["zero_prediction_rate"] == 0.5
    assert summary["size_buckets"]["small"]["n"] == 1
    assert summary["size_buckets"]["small"]["zero_prediction_rate"] == 1.0
    assert summary["size_buckets"]["medium"]["mean_iou"] == 0.2
    assert summary["size_buckets"]["large"]["n"] == 2
    assert summary["worst_zero_prediction_categories"] == ["spoon", "pot"]


def test_build_markdown_includes_bucket_table(tmp_path: Path) -> None:
    result = tmp_path / "waldo.json"
    _write_result(result)
    summary = strat.summarize_result(result, selection="thr0p25")

    markdown = strat.build_markdown(summary)

    assert "Waldo Kitchen Failure Stratification" in markdown
    assert "| small | 1 | 0.0000 | 1.0000 |" in markdown
    assert "spoon, pot" in markdown


def test_write_outputs_records_json_and_markdown(tmp_path: Path) -> None:
    result = tmp_path / "waldo.json"
    _write_result(result)
    summary = strat.summarize_result(result, selection="thr0p25")

    paths = strat.write_outputs(summary, tmp_path / "waldo.md", tmp_path / "waldo_summary.json")

    assert paths["markdown"].exists()
    payload = json.loads(paths["json"].read_text())
    assert payload["selection"] == "thr0p25"
