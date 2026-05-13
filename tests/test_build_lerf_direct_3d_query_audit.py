import json
from pathlib import Path

from radio_gs.scripts import build_lerf_direct_3d_query_audit as audit


def _write_scene(root: Path, scene: str, tag: str, details: list[dict]) -> None:
    scene_dir = root / scene
    scene_dir.mkdir(parents=True)
    scene_dir.joinpath("lerf_direct_3d_selection_results.json").write_text(
        json.dumps(
            {
                "scene": {
                    "results": {
                        tag: {
                            "miou": sum(item["iou"] for item in details) / len(details),
                            "acc025": sum(item["iou"] > 0.25 for item in details) / len(details),
                            "query_details": details,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_load_query_rows_reads_scene_query_details(tmp_path: Path) -> None:
    _write_scene(
        tmp_path,
        "figurines",
        "meanstd2p5",
        [
            {"frame": "frame_00001", "category": "apple", "iou": 0.5, "pred_pixels": 20, "gt_pixels": 10, "overselect_ratio": 2.0},
            {"frame": "frame_00002", "category": "cup", "iou": 0.0, "pred_pixels": 0, "gt_pixels": 15, "overselect_ratio": 0.0},
        ],
    )

    rows = audit.load_query_rows(tmp_path, ["figurines"], "meanstd2p5")

    assert [row.category for row in rows] == ["apple", "cup"]
    assert rows[0].scene == "Figurines"
    assert rows[1].zero_pred is True


def test_summarize_rows_reports_ci_and_failure_rates(tmp_path: Path) -> None:
    rows = [
        audit.QueryRow("Figurines", "frame_00001", "apple", 0.5, 20, 10, 2.0),
        audit.QueryRow("Figurines", "frame_00002", "cup", 0.0, 0, 15, 0.0),
        audit.QueryRow("Figurines", "frame_00003", "plate", 0.2, 50, 10, 5.0),
    ]

    summary = audit.summarize_rows(rows)

    assert summary["n"] == 3
    assert summary["zero_pred_rate"] == 1 / 3
    assert summary["acc025"] == 1 / 3
    assert summary["mean_overselect_ratio"] > 2.0
