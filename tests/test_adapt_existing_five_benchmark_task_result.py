from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.scripts.adapt_existing_five_benchmark_task_result import (
    TaskResultAdapterError,
    adapt,
)


def _joint(task: str, scenes: list[str]) -> dict:
    return {
        "joint_candidate": {
            "candidate_sha256": "a" * 64,
            "identity": {"typed_readout_map": {task: "readout-v1"}},
        },
        "deployment_fields": {
            "task_references": {task: scenes},
            "instances": [
                {"dataset_instance_id": scene, "universal_field": {"sha256": f"{index + 1:064x}"}}
                for index, scene in enumerate(scenes)
            ],
        },
    }


def _adapt(tmp_path: Path, task: str, source: dict, scenes: list[str]) -> dict:
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return adapt(task=task, joint_receipt=_joint(task, scenes), source=source, source_path=path)


def test_lerf2d_complete_result_remains_contract_ineligible(tmp_path: Path) -> None:
    scenes = ["lerf:figurines", "lerf:ramen", "lerf:teatime", "lerf:waldo_kitchen"]
    source = {"lerf2d_frozen_full4": {"scene_count": 4, "sample_count": 208, "sample_micro_miou": .314, "scene_macro_miou": .319, "localization_accuracy": .88, "scene_results": {name.split(":")[1]: {} for name in scenes}}}
    row = _adapt(tmp_path, "lerf2d", source, scenes)
    assert row["status"] == "complete"
    assert row["exact_evaluation_contract"] is False
    assert row["adapter_assessment"]["result_complete"] is True
    assert row["adapter_assessment"]["development_contract_eligible"] is False


def test_lerf3d_single_level_cannot_become_three_level_row(tmp_path: Path) -> None:
    scenes = ["figurines", "ramen", "teatime", "waldo_kitchen"]
    source = {"lerf3d_frozen_full4": {"scene_count": 4, "readout": "primitive_text_score_on_frozen_3d_support", "sample_micro_miou": .3345, "sample_micro_acc025": .5769, "sample_micro_acc050": .2740, "scene_results": {scene: {} for scene in scenes}}}
    row = _adapt(tmp_path, "lerf3d", source, [f"lerf:{scene}" for scene in scenes])
    assert row["exact_evaluation_contract"] is False
    assert row["adapter_assessment"]["metrics"]["semantic_levels"] == 1
    assert "three" in row["adapter_assessment"]["contract_gaps"][0]


def test_scannet_frozen_full8_is_development_eligible_not_paper(tmp_path: Path) -> None:
    metrics = {split: {"miou": .3, "macc": .6} for split in ("19", "15", "10")}
    source = {"scannet_ovs_paper8": {"method_v1_complete": 8, "scene_results": {str(i): {} for i in range(8)}, "evaluation_protocol": "frozen_and_reproduced", "paper8_macro": {"vala_pseudo_volume": metrics}}}
    row = _adapt(tmp_path, "scannet_ovs", source, [f"scannet:{i}" for i in range(8)])
    assert row["exact_evaluation_contract"] is True
    assert row["adapter_assessment"]["development_contract_eligible"] is True
    assert row["adapter_assessment"]["paper_contract_eligible"] is False


def test_nvos_rgb_assisted_is_legal_development_but_not_strict_paper(tmp_path: Path) -> None:
    source = {
        "status": "development_candidate_promoted_by_preregistered_gate",
        "result": {"macro_foreground_iou": .8176, "pixel_accuracy": .97, "scene_foreground_iou": {"fern": .8}},
        "eligibility": {"target_rgb_assisted": True},
        "prediction_batch": {"all_eight_predictions_sealed_before_first_target_mask_open": True},
    }
    row = _adapt(tmp_path, "nvos", source, ["nvos:fern"])
    assert row["authorized_target_access"] is True
    assert row["prediction_barrier_passed"] is True
    assert row["adapter_assessment"]["development_contract_eligible"] is True
    assert row["adapter_assessment"]["paper_contract_eligible"] is False
    assert "strict-unseen" in row["adapter_assessment"]["contract_gaps"][0]


def test_nvos_fails_closed_without_prediction_barrier(tmp_path: Path) -> None:
    source = {"status": "development_candidate_promoted_by_preregistered_gate", "result": {}, "eligibility": {"target_rgb_assisted": True}, "prediction_batch": {"all_eight_predictions_sealed_before_first_target_mask_open": False}}
    with pytest.raises(TaskResultAdapterError, match="barrier"):
        _adapt(tmp_path, "nvos", source, ["nvos:fern"])
