import json

import numpy as np

from radio_gs.scripts.eval_lerf_score_quality_diagnostic import (
    _index_query_rows,
    grouped_average_precision,
    oracle_threshold_iou,
)


def test_grouped_average_precision_is_tie_invariant() -> None:
    scores = np.array([0.9, 0.5, 0.5, 0.1])
    target_a = np.array([1, 1, 0, 0], dtype=bool)
    target_b = np.array([1, 0, 1, 0], dtype=bool)
    assert grouped_average_precision(scores, target_a) == grouped_average_precision(
        scores, target_b
    )
    assert np.isclose(grouped_average_precision(scores, target_a), 5.0 / 6.0)


def test_oracle_threshold_respects_equal_score_groups() -> None:
    result = oracle_threshold_iou(
        np.array([0.9, 0.5, 0.5, 0.1]),
        np.array([1, 1, 0, 0], dtype=bool),
    )
    assert np.isclose(result["iou"], 2.0 / 3.0)
    assert result["threshold"] == 0.5
    assert result["selected_pixels"] == 3


def test_index_query_rows_reads_frozen_result_shape(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "scene": {
                    "results": {
                        "thr0p6": {
                            "query_details": [
                                {"frame_id": 3, "category": "cup", "iou": 0.5}
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows = _index_query_rows(path)
    assert rows[(3, "cup")]["iou"] == 0.5
