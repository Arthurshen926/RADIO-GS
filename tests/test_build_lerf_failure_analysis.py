import json
from pathlib import Path

from radio_gs.scripts import build_lerf_failure_analysis as failure


def test_load_category_rows_extracts_rendered_failures(tmp_path: Path) -> None:
    result_path = tmp_path / "lerf_ovs_results.json"
    result_path.write_text(
        json.dumps(
            {
                "scenes": {
                    "toy_scene": {
                        "rendered": {
                            "per_category": {
                                "small object": {
                                    "loc_acc": 0.0,
                                    "miou": 0.1,
                                    "n_samples": 2,
                                },
                                "large object": {
                                    "loc_acc": 1.0,
                                    "miou": 0.8,
                                    "n_samples": 3,
                                },
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rows = failure.load_category_rows("Toy", result_path)

    assert len(rows) == 2
    assert rows[0].scene == "Toy"
    assert rows[0].category == "small object"
    assert rows[0].loc_acc == 0.0


def test_select_fragile_rows_prioritizes_failed_low_miou_categories() -> None:
    rows = [
        failure.CategoryRow("Toy", "good", 1.0, 0.8, 4),
        failure.CategoryRow("Toy", "missed", 0.0, 0.2, 2),
        failure.CategoryRow("Toy", "fragile", 0.5, 0.1, 2),
    ]

    selected = failure.select_fragile_rows(rows, limit=2)

    assert [row.category for row in selected] == ["missed", "fragile"]
