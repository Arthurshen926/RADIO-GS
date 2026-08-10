from __future__ import annotations

import pytest

from radio_gs.scripts import (
    materialize_lerf_o1_native_v3_scale_aware_fusion_cross_scene as materializer,
)


def _result(*, target_metrics_opened: bool = False) -> tuple[dict, dict, dict]:
    positive = {"path": "/tmp/teatime_positive.pt", "sha256": "a" * 64}
    negative = {"path": "/tmp/teatime_negative.pt", "sha256": "b" * 64}
    result = {
        "schema": "radio_gs.lerf_o1_o2_streaming_result.v1",
        "schema_version": 1,
        "status": "complete_source_only_premetric_o1_o2_streaming",
        "scene_id": "teatime",
        "outputs": {"o1_positive": positive, "o1_negative": negative},
        "metric_executed": False,
        "metric_execution_authorized": False,
        "access_audit": {
            "target_ground_truth_opened": False,
            "target_images_opened": False,
            "target_masks_opened": False,
            "target_metrics_opened": target_metrics_opened,
            "target_quality_readout_executed": False,
        },
    }
    return result, positive, negative


def test_cross_scene_o1_materialization_binding() -> None:
    result, positive, negative = _result()
    materializer.validate_o1_materialization_result(
        result,
        scene_id="teatime",
        positive_record=positive,
        negative_record=negative,
    )


def test_cross_scene_o1_materialization_rejects_metric_opening() -> None:
    result, positive, negative = _result(target_metrics_opened=True)
    with pytest.raises(ValueError, match="cross-scene O1 materialization binding"):
        materializer.validate_o1_materialization_result(
            result,
            scene_id="teatime",
            positive_record=positive,
            negative_record=negative,
        )
