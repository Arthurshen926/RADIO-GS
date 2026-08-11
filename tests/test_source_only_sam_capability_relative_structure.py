from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.training.source_only_sam_capability_relative_structure import (
    SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256,
    SourceSamCapabilityRelativeTeacher,
    capability_geometric_mean_relation,
    evaluate_source_sam_capability_relative_full_gates,
    evaluate_source_sam_capability_relative_gates,
    source_only_sam_capability_relative_contract,
    source_sam_capability_relative_batch_loss,
    validate_source_only_sam_capability_relative_manifest,
)
from radio_gs.training.source_only_sam_relative_structure import (
    SAM_RELATIVE_MARGIN,
    SourceSamRelativeTeacher,
)
from radio_gs.training.source_only_sam_structure import SourceSamStructureTeacher


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> dict:
    base = tmp_path / "base.json"
    checkpoint = tmp_path / "radio.pt"
    base.write_text("{}")
    checkpoint.write_bytes(b"official-radio")
    contract = source_only_sam_capability_relative_contract()
    return {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract": contract,
        "contract_sha256": SOURCE_ONLY_SAM_CAPABILITY_RELATIVE_CONTRACT_SHA256,
        "status": "preregistered_training_not_started",
        "scene_id": "synthetic",
        "base_relative_manifest": {"path": str(base), "sha256": _sha(base)},
        "official_adaptor_checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha(checkpoint),
        },
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
            "global_capability_same_relation_non_decrease": True,
            "global_capability_separate_relation_non_increase": True,
            "global_capability_relation_gap_strict_improvement": True,
            "capability_triplet_gap_strict_improvement": True,
            "capability_triplet_violation_strict_decrease": True,
            "six_task_benchmark_gate": (
                "closed_after_source_gate_checkpoint_seal_no_benchmark_in_this_stage"
            ),
        },
        "access_audit": {
            "source_rgb_opened_during_mapping": True,
            "official_sam_opened_during_mapping": True,
            "official_dino_and_sam_adaptors_opened_during_training": True,
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
            "v2_candidate_checkpoint_used": False,
        },
    }


class _FrozenViews(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dino = torch.nn.Linear(1280, 8, bias=False)
        self.sam = torch.nn.Linear(1280, 8, bias=False)
        generator = torch.Generator().manual_seed(4)
        with torch.no_grad():
            self.dino.weight.copy_(torch.randn(8, 1280, generator=generator))
            self.sam.weight.copy_(torch.randn(8, 1280, generator=generator))
        self.requires_grad_(False)

    def project_dino_primitives(self, radio: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.dino(radio), dim=-1)

    def project_sam3_primitives(self, radio: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.sam(radio), dim=-1)


class _ToyField(torch.nn.Module):
    def __init__(self, radio: torch.Tensor) -> None:
        super().__init__()
        self.local_codes = torch.nn.Parameter(radio.clone())

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        return self.local_codes[rows]


def _teacher(field: _ToyField, views: _FrozenViews) -> SourceSamCapabilityRelativeTeacher:
    pair = SourceSamStructureTeacher(
        global_edge_index=torch.tensor([[0, 0], [1, 2]]),
        same_relation=torch.tensor([True, False]),
        edge_weight=torch.ones(2),
        relation_cache_source=Path("synthetic.pt"),
        relation_cache_sha256="0" * 64,
    )
    triplet = torch.tensor([[0], [1], [2]])
    relative = SourceSamRelativeTeacher(
        pair_teacher=pair,
        triplet_index=triplet,
        triplet_weight=torch.ones(1),
        scale_bin=torch.zeros(1, dtype=torch.long),
        pair_control_cosine=torch.zeros(2),
        positive_control_cosine=torch.zeros(1),
        negative_control_cosine=torch.zeros(1),
    )
    with torch.no_grad():
        pair_relation = capability_geometric_mean_relation(
            field.local_codes, pair.global_edge_index, official_views=views
        )
        positive = capability_geometric_mean_relation(
            field.local_codes, triplet[[0, 1]], official_views=views
        )
        negative = capability_geometric_mean_relation(
            field.local_codes, triplet[[0, 2]], official_views=views
        )
    return SourceSamCapabilityRelativeTeacher(
        relative_teacher=relative,
        pair_control_relation=pair_relation,
        positive_control_relation=positive,
        negative_control_relation=negative,
    )


def test_v3_manifest_is_single_radio_and_benchmark_closed(tmp_path: Path) -> None:
    manifest = validate_source_only_sam_capability_relative_manifest(
        _manifest(tmp_path)
    )
    assert manifest["contract"]["persistent_semantic_feature"] == "canonical_radio_only"
    assert manifest["contract"]["relation_space"]["adaptors_frozen"] is True
    assert manifest["access_audit"]["benchmark_ground_truth_opened"] is False
    changed = _manifest(tmp_path)
    changed["execution"]["v2_candidate_checkpoint_used"] = True
    with pytest.raises(ValueError, match="execution"):
        validate_source_only_sam_capability_relative_manifest(changed)


def test_geometric_mean_relation_uses_both_frozen_capabilities() -> None:
    views = _FrozenViews()
    radio = torch.randn(3, 1280, generator=torch.Generator().manual_seed(1))
    edge = torch.tensor([[0, 0], [1, 2]])
    relation = capability_geometric_mean_relation(radio, edge, official_views=views)
    dino = views.project_dino_primitives(radio)
    sam = views.project_sam3_primitives(radio)
    expected = torch.sqrt(
        (
            0.5 * (1 + (dino[edge[0]] * dino[edge[1]]).sum(-1))
            * 0.5
            * (1 + (sam[edge[0]] * sam[edge[1]]).sum(-1))
        ).clamp_min(0)
    )
    assert torch.allclose(relation, expected)
    assert all(not parameter.requires_grad for parameter in views.parameters())


def test_capability_relative_loss_has_only_tangent_radio_gradient() -> None:
    views = _FrozenViews()
    radio = torch.randn(3, 1280, generator=torch.Generator().manual_seed(2))
    field = _ToyField(radio)
    teacher = _teacher(field, views)
    # Force both triplet and guards active without changing the immutable axis.
    teacher = SourceSamCapabilityRelativeTeacher(
        relative_teacher=teacher.relative_teacher,
        pair_control_relation=torch.tensor([1.0, 0.0]),
        positive_control_relation=teacher.positive_control_relation,
        negative_control_relation=teacher.negative_control_relation,
    )
    loss, stats = source_sam_capability_relative_batch_loss(
        field,
        official_views=views,
        teacher=teacher,
        triplet_indices=torch.tensor([0]),
        guard_edge_indices=torch.tensor([0, 1]),
    )
    assert float(loss) > 0
    assert set(stats) == {
        "capability_ranking_loss",
        "capability_same_guard_loss",
        "capability_separate_guard_loss",
        "triplets",
        "guard_edges",
    }
    loss.backward()
    for row in range(3):
        gradient = field.local_codes.grad[row]
        value = field.local_codes.detach()[row]
        assert abs(float((gradient * value).sum())) < 2e-5
    assert all(parameter.grad is None for parameter in views.parameters())
    assert SAM_RELATIVE_MARGIN == 0.05


def test_v3_gates_require_capability_triplet_and_no_regression(tmp_path: Path) -> None:
    control_pair = {
        "sam_capability_structure_edges": 20,
        "sam_capability_same_mean_relation": 0.7,
        "sam_capability_separate_mean_relation": 0.5,
        "sam_capability_relation_gap": 0.2,
    }
    candidate_pair = {
        **control_pair,
        "sam_capability_same_mean_relation": 0.71,
        "sam_capability_separate_mean_relation": 0.48,
        "sam_capability_relation_gap": 0.23,
    }
    control_relative = {
        "sam_capability_relative_triplets": 8,
        "sam_capability_relative_gap": 0.08,
        "sam_capability_relative_violation_rate": 0.4,
    }
    candidate_relative = {
        "sam_capability_relative_triplets": 8,
        "sam_capability_relative_gap": 0.1,
        "sam_capability_relative_violation_rate": 0.3,
    }
    structure = evaluate_source_sam_capability_relative_gates(
        control_pair=control_pair,
        candidate_pair=candidate_pair,
        control_relative=control_relative,
        candidate_relative=candidate_relative,
    )
    assert structure["all_structure_gates_passed"]
    assert structure["benchmark_gate_opened"] is False
    full = evaluate_source_sam_capability_relative_full_gates(
        manifest=_manifest(tmp_path),
        control_primary={"mean_cosine": 0.99, "p05_cosine": 0.98},
        candidate_primary={"mean_cosine": 0.989, "p05_cosine": 0.979},
        control_capability={"dino_target_mean_cosine": 0.98},
        candidate_capability={"dino_target_mean_cosine": 0.979},
        structure_decision=structure,
    )
    assert full["all_source_gates_passed"]
    assert full["checkpoint_seal_allowed"]
    assert full["benchmark_gate_opened"] is False
