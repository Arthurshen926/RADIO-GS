from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.training.source_only_sam_relative_structure import (
    SAM_RELATIVE_MARGIN,
    SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256,
    SourceSamRelativeTeacher,
    build_scale_matched_triplets,
    evaluate_source_sam_relative_full_gates,
    evaluate_source_sam_relative_gates,
    source_only_sam_relative_contract,
    source_sam_relative_batch_loss,
    source_sam_relative_metrics,
    validate_source_only_sam_relative_manifest,
)
from radio_gs.training.source_only_sam_structure import SourceSamStructureTeacher


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> dict:
    base = tmp_path / "base.json"
    base.write_text("{}", encoding="utf-8")
    contract = source_only_sam_relative_contract()
    return {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract": contract,
        "contract_sha256": SOURCE_ONLY_SAM_RELATIVE_CONTRACT_SHA256,
        "status": "preregistered_training_not_started",
        "scene_id": "synthetic",
        "base_structure_manifest": {"path": str(base), "sha256": _sha(base)},
        "loss": contract["loss"],
        "source_gates": {
            "radio_reconstruction_no_regression": {
                "mean_cosine_max_regression": 0.005,
                "p05_cosine_max_regression": 0.01,
            },
            "official_capability_no_regression": {
                "mean_cosine_max_regression": 0.005,
                "p05_cosine_max_regression": 0.01,
            },
            "global_same_cosine_non_decrease": True,
            "global_separate_cosine_non_increase": True,
            "global_relation_gap_strict_improvement": True,
            "scale_triplet_gap_strict_improvement": True,
            "scale_triplet_violation_strict_decrease": True,
            "six_task_benchmark_gate": (
                "closed_until_all_source_gates_pass_then_frozen_one_shot"
            ),
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "official_sam_opened_during_mapping": True,
            "query_time_source_rgb_opened": False,
            "query_time_target_rgb_opened": False,
            "benchmark_query_or_text_opened": False,
            "benchmark_target_or_evaluation_rgb_opened": False,
            "benchmark_ground_truth_opened": False,
            "benchmark_labels_or_masks_opened": False,
            "benchmark_metrics_or_predictions_opened": False,
        },
        "execution": {
            "gpu_started": False,
            "per_scene_or_per_task_tuning": False,
            "output_no_clobber": True,
            "teacher_payload_saved_in_checkpoint": False,
            "v1_candidate_checkpoint_used": False,
        },
    }


def test_relative_manifest_is_single_radio_and_evaluation_rgb_free(tmp_path: Path) -> None:
    payload = _manifest(tmp_path)
    result = validate_source_only_sam_relative_manifest(payload)
    assert result["contract"]["persistent_semantic_feature"] == "canonical_radio_only"
    assert result["access_audit"]["source_rgb_opened_during_mapping"] is True
    assert result["access_audit"]["query_time_source_rgb_opened"] is False
    assert result["access_audit"]["query_time_target_rgb_opened"] is False


def test_relative_manifest_rejects_v1_candidate_initialization(tmp_path: Path) -> None:
    payload = _manifest(tmp_path)
    payload["execution"]["v1_candidate_checkpoint_used"] = True
    with pytest.raises(ValueError, match="execution contract"):
        validate_source_only_sam_relative_manifest(payload)


def test_scale_matched_triplets_do_not_mix_hierarchy_bins() -> None:
    # At scale 0, anchor 0 has same neighbor 1 and boundary neighbor 2.  At
    # scale 1, its matching pair is 3/4.  The compiler must not cross-pair.
    edge = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
    same = torch.tensor([[3.0, 0.0], [0.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    separate = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 5.0]])
    triplet, weight, scale = build_scale_matched_triplets(
        global_edge=edge,
        same_votes=same,
        separate_votes=separate,
        valid=torch.ones(5, dtype=torch.bool),
    )
    entries = {
        (int(scale[i]), *(int(value) for value in triplet[:, i]))
        for i in range(triplet.shape[1])
    }
    assert entries == {(0, 0, 1, 2), (1, 0, 3, 4)}
    assert torch.all(weight > 0)


class _ToyField(torch.nn.Module):
    def __init__(self, radio: torch.Tensor) -> None:
        super().__init__()
        self.local_codes = torch.nn.Parameter(radio.clone())

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        return self.local_codes[rows]


def _teacher(field: _ToyField) -> SourceSamRelativeTeacher:
    pair = SourceSamStructureTeacher(
        global_edge_index=torch.tensor([[0, 0], [1, 2]]),
        same_relation=torch.tensor([True, False]),
        edge_weight=torch.ones(2),
        relation_cache_source=Path("synthetic.pt"),
        relation_cache_sha256="0" * 64,
    )
    with torch.no_grad():
        direction = torch.nn.functional.normalize(field.local_codes.float(), dim=-1)
        positive = (direction[0] * direction[1]).sum().reshape(1)
        negative = (direction[0] * direction[2]).sum().reshape(1)
    return SourceSamRelativeTeacher(
        pair_teacher=pair,
        triplet_index=torch.tensor([[0], [1], [2]]),
        triplet_weight=torch.ones(1),
        scale_bin=torch.zeros(1, dtype=torch.long),
        pair_control_cosine=torch.cat([positive, negative]),
        positive_control_cosine=positive,
        negative_control_cosine=negative,
    )


def test_relative_loss_is_zero_when_ordered_and_control_preserved() -> None:
    radio = torch.zeros(3, 1280)
    radio[0, 0] = 1
    radio[1, 0] = 1
    radio[2, 1] = 1
    field = _ToyField(radio)
    teacher = _teacher(field)
    loss, stats = source_sam_relative_batch_loss(
        field,
        teacher=teacher,
        triplet_indices=torch.tensor([0]),
        guard_edge_indices=torch.tensor([0, 1]),
    )
    assert float(loss) == 0.0
    assert float(stats["ranking_loss"]) == 0.0
    assert SAM_RELATIVE_MARGIN == 0.05


def test_relative_loss_repairs_violation_with_tangent_gradient() -> None:
    radio = torch.zeros(3, 1280)
    radio[0, 0] = 1
    radio[1, 0] = 0.6
    radio[1, 1] = 0.8
    radio[2, 0] = 0.9
    radio[2, 1] = 0.4358899
    field = _ToyField(radio)
    teacher = _teacher(field)
    loss, _ = source_sam_relative_batch_loss(
        field,
        teacher=teacher,
        triplet_indices=torch.tensor([0]),
        guard_edge_indices=torch.tensor([0, 1]),
    )
    assert float(loss) > 0
    loss.backward()
    for row in range(3):
        gradient = field.local_codes.grad[row]
        value = field.local_codes.detach()[row]
        assert abs(float((gradient * value).sum())) < 1e-5


def test_control_relative_guard_penalizes_only_harmful_directions() -> None:
    radio = torch.zeros(3, 1280)
    radio[0, 0] = 1
    radio[1, 0] = 1
    radio[2, 1] = 1
    control = _ToyField(radio)
    teacher = _teacher(control)
    candidate = _ToyField(radio)
    with torch.no_grad():
        # Decrease the same cosine and increase the separate cosine.
        candidate.local_codes[1, 0] = 0.8
        candidate.local_codes[1, 1] = 0.6
        candidate.local_codes[2, 0] = 0.2
    loss, stats = source_sam_relative_batch_loss(
        candidate,
        teacher=teacher,
        triplet_indices=torch.tensor([0]),
        guard_edge_indices=torch.tensor([0, 1]),
    )
    assert float(loss) > 0
    assert float(stats["same_guard_loss"]) > 0
    assert float(stats["separate_guard_loss"]) > 0


def test_relative_gates_require_global_and_scale_improvements() -> None:
    control_pair = {
        "sam_structure_edges": 10,
        "sam_same_edges": 5,
        "sam_separate_edges": 5,
        "sam_same_mean_cosine": 0.7,
        "sam_separate_mean_cosine": 0.4,
        "sam_relation_gap": 0.3,
    }
    candidate_pair = {
        **control_pair,
        "sam_same_mean_cosine": 0.71,
        "sam_separate_mean_cosine": 0.38,
        "sam_relation_gap": 0.33,
    }
    control_relative = {
        "sam_relative_triplets": 8,
        "sam_relative_gap": 0.1,
        "sam_relative_violation_rate": 0.4,
    }
    candidate_relative = {
        "sam_relative_triplets": 8,
        "sam_relative_gap": 0.12,
        "sam_relative_violation_rate": 0.3,
    }
    result = evaluate_source_sam_relative_gates(
        control_pair=control_pair,
        candidate_pair=candidate_pair,
        control_relative=control_relative,
        candidate_relative=candidate_relative,
    )
    assert result["all_structure_gates_passed"]
    candidate_relative["sam_relative_violation_rate"] = 0.4
    assert not evaluate_source_sam_relative_gates(
        control_pair=control_pair,
        candidate_pair=candidate_pair,
        control_relative=control_relative,
        candidate_relative=candidate_relative,
    )["all_structure_gates_passed"]


def test_full_gate_also_requires_radio_and_capability_preservation(tmp_path: Path) -> None:
    structure = {"all_structure_gates_passed": True}
    result = evaluate_source_sam_relative_full_gates(
        manifest=_manifest(tmp_path),
        control_primary={"mean_cosine": 0.96, "p05_cosine": 0.92},
        candidate_primary={"mean_cosine": 0.959, "p05_cosine": 0.919},
        control_capability={
            "dino_target_mean_cosine": 0.97,
            "dino_target_p05_cosine": 0.94,
        },
        candidate_capability={
            "dino_target_mean_cosine": 0.969,
            "dino_target_p05_cosine": 0.939,
        },
        structure_decision=structure,
    )
    assert result["all_source_gates_passed"]
    failed = evaluate_source_sam_relative_full_gates(
        manifest=_manifest(tmp_path),
        control_primary={"mean_cosine": 0.96, "p05_cosine": 0.92},
        candidate_primary={"mean_cosine": 0.95, "p05_cosine": 0.919},
        control_capability={"dino_target_mean_cosine": 0.97},
        candidate_capability={"dino_target_mean_cosine": 0.969},
        structure_decision=structure,
    )
    assert not failed["benchmark_gate_opened"]
