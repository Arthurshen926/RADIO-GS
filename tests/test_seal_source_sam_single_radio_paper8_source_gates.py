import json
from pathlib import Path

import pytest

from radio_gs.scripts.seal_source_sam_single_radio_paper8_source_gates import seal


def _report() -> dict:
    return {
        "source_sam_relative_gate_decision": {"all_source_gates_passed": True},
        "source_only_sam_relative_structure": {
            "persistent_semantic_feature": "canonical_radio_only",
            "teacher_payload_saved": False,
            "query_time_source_rgb": False,
            "query_time_target_rgb": False,
            "manifest": {"path": "/source.json", "sha256": "a" * 64},
        },
        "initial_field_checkpoint": {
            "source_final_metrics": {"mean_cosine": 0.90, "p05_cosine": 0.80},
            "source_final_capability_metrics": {
                "dino_v3_target_mean_cosine": 0.91,
                "sam3_target_mean_cosine": 0.89,
            },
        },
        "final_metrics": {"mean_cosine": 0.91, "p05_cosine": 0.81},
        "final_capability_metrics": {
            "dino_v3_target_mean_cosine": 0.92,
            "sam3_target_mean_cosine": 0.90,
        },
        "control_source_sam_relative_pair_metrics": {"sam_relation_gap": 0.1},
        "final_source_sam_relative_pair_metrics": {"sam_relation_gap": 0.2},
        "control_source_sam_relative_metrics": {
            "sam_relative_gap": 0.1,
            "sam_relative_violation_rate": 0.4,
        },
        "final_source_sam_relative_metrics": {
            "sam_relative_gap": 0.2,
            "sam_relative_violation_rate": 0.3,
        },
    }


def test_seals_exact_eight_single_radio_candidates(tmp_path: Path) -> None:
    prereg = tmp_path / "prereg.json"
    prereg.write_text("{}")
    specs = []
    for index in range(8):
        checkpoint = tmp_path / f"scene{index}.pth"
        checkpoint.write_bytes(bytes([index]))
        Path(str(checkpoint) + ".json").write_text(json.dumps(_report()))
        specs.append(f"scene{index:04d}_00={checkpoint}")
    output = tmp_path / "seal.json"
    result = seal(specs, str(prereg), str(output))
    assert result["all_source_gates_passed"] is True
    assert len(result["scenes"]) == 8
    assert result["scenes"][0]["deltas"]["global_relation_gap"] > 0


def test_rejects_failed_or_rgb_dependent_candidate(tmp_path: Path) -> None:
    prereg = tmp_path / "prereg.json"
    prereg.write_text("{}")
    specs = []
    for index in range(8):
        checkpoint = tmp_path / f"scene{index}.pth"
        checkpoint.write_bytes(bytes([index]))
        report = _report()
        if index == 3:
            report["source_only_sam_relative_structure"]["query_time_target_rgb"] = True
        Path(str(checkpoint) + ".json").write_text(json.dumps(report))
        specs.append(f"scene{index:04d}_00={checkpoint}")
    with pytest.raises(ValueError, match="query-time RGB"):
        seal(specs, str(prereg), str(tmp_path / "seal.json"))
