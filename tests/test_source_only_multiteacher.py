from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from radio_gs.training.source_only_multiteacher import (
    SOURCE_ONLY_MULTITEACHER_CONTRACT_SHA256,
    SOURCE_ONLY_MULTITEACHER_SCHEMA,
    SOURCE_ONLY_MULTITEACHER_SCHEMA_VERSION,
    PrimitiveDescriptorTeacher,
    _evenly_spaced_edge_selection,
    evaluate_source_only_gates,
    primitive_source_descriptor_loss,
    source_direction_only_radio,
    source_only_multiteacher_contract,
    validate_source_only_multiteacher_manifest,
)
from radio_gs.utils.immutable_artifacts import file_record


def _manifest(tmp_path: Path) -> dict:
    records = {}
    for name in (
        "field_control",
        "factorized_radio_cache",
        "factorized_primitive_state",
        "primitive_descriptor_teacher",
        "relation_graph",
        "official_radio_checkpoint",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        records[name] = file_record(path)
    return {
        "schema": SOURCE_ONLY_MULTITEACHER_SCHEMA,
        "schema_version": SOURCE_ONLY_MULTITEACHER_SCHEMA_VERSION,
        "contract": source_only_multiteacher_contract(),
        "contract_sha256": SOURCE_ONLY_MULTITEACHER_CONTRACT_SHA256,
        "status": "preregistered_gpu_not_started",
        "scene_id": "sentinel",
        **records,
        "loss": source_only_multiteacher_contract()["loss"],
        "source_gates": {
            "raw_radio_no_regression": "relative_to_hash_bound_control",
            "gauge_no_regression": "relative_to_hash_bound_control",
            "dino_sam_no_regression": "relative_to_hash_bound_control",
            "descriptor_improvement": "strict_mean_improvement",
            "descriptor_tail_no_regression": "p05_non_regression",
            "semantic_relation_no_regression": "mean_and_p95_non_regression",
            "scalar_authority_unchanged": "bitwise_same_state_sidecar",
            "benchmark_gate": (
                "closed_until_every_source_gate_passes_then_one_shot_frozen_metric"
            ),
        },
        "access_audit": {
            "source_rgb_or_features_may_have_been_opened": True,
            "benchmark_query_or_text_opened": False,
            "benchmark_target_rgb_opened": False,
            "benchmark_ground_truth_opened": False,
            "benchmark_labels_or_masks_opened": False,
            "benchmark_metrics_or_predictions_opened": False,
        },
        "execution": {
            "gpu_started": False,
            "per_scene_tuning": False,
            "output_no_clobber": True,
            "implementation": {
                "multiteacher": file_record(
                    Path("radio_gs/training/source_only_multiteacher.py").resolve()
                ),
                "trainer": file_record(
                    Path("radio_gs/scripts/train_canonical_radio_field.py").resolve()
                ),
                "manifest_validator": file_record(
                    Path(
                        "radio_gs/scripts/validate_source_only_multiteacher_manifest.py"
                    ).resolve()
                ),
            },
        },
    }


def test_manifest_is_source_only_and_weights_are_not_tunable(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    validated = validate_source_only_multiteacher_manifest(value)
    assert validated["loss"]["descriptor_weight"] == 0.20
    assert validated["loss"]["relation_weight"] == 0.05

    contaminated = dict(value)
    contaminated["access_audit"] = {
        **value["access_audit"],
        "benchmark_query_or_text_opened": True,
    }
    with pytest.raises(ValueError, match="access audit"):
        validate_source_only_multiteacher_manifest(contaminated)

    tuned = dict(value)
    tuned["loss"] = {**value["loss"], "descriptor_weight": 0.25}
    with pytest.raises(ValueError, match="loss differs"):
        validate_source_only_multiteacher_manifest(tuned)


def test_evenly_spaced_edge_selection_is_fixed_and_endpoint_complete() -> None:
    selected = _evenly_spaced_edge_selection(11, 5)
    assert torch.equal(selected, torch.tensor([0, 2, 5, 7, 10]))
    assert torch.equal(
        _evenly_spaced_edge_selection(3, 5), torch.tensor([0, 1, 2])
    )
    assert torch.equal(_evenly_spaced_edge_selection(11, 1), torch.tensor([0]))


class _FirstTwoProjector(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values[:, :2], dim=-1)


def test_primitive_descriptor_loss_uses_only_aligned_source_rows(tmp_path: Path) -> None:
    source = tmp_path / "teacher.pt"
    source.write_bytes(b"teacher")
    descriptor = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1
    )
    teacher = PrimitiveDescriptorTeacher(
        scene_id="sentinel",
        global_rows=torch.tensor([1, 3]),
        descriptor=descriptor,
        valid=torch.tensor([True, True]),
        retained_view_count=torch.tensor([4, 2], dtype=torch.uint8),
        directional_resultant=torch.tensor([1.0, 0.5]),
        global_to_local=torch.tensor([-1, 0, -1, 1]),
        source=source,
        sha256="0" * 64,
    )
    radio = torch.zeros(3, 1280)
    radio[0, 0] = 1.0  # global row 0: absent from the teacher
    radio[1, 0] = 1.0  # global row 1: exact target
    radio[2, 0] = 1.0  # global row 3: orthogonal to target
    loss, stats = primitive_source_descriptor_loss(
        radio,
        torch.tensor([0, 1, 3]),
        projector=_FirstTwoProjector(),
        teacher=teacher,
    )
    assert int(stats["active_rows"]) == 2
    assert 0.0 < float(loss) < 1.0
    assert torch.isclose(stats["mean_cosine"], torch.tensor(0.5))


def test_teacher_batch_keeps_ambiguous_rows_trainable(tmp_path: Path) -> None:
    source = tmp_path / "teacher.pt"
    source.write_bytes(b"teacher")
    teacher = PrimitiveDescriptorTeacher(
        scene_id="sentinel",
        global_rows=torch.tensor([0]),
        descriptor=torch.tensor([[1.0, 0.0]]),
        valid=torch.tensor([True]),
        retained_view_count=torch.tensor([1], dtype=torch.uint8),
        directional_resultant=torch.tensor([0.0]),
        global_to_local=torch.tensor([0]),
        source=source,
        sha256="0" * 64,
    )
    _target, active, weight = teacher.batch(torch.tensor([0]))
    assert bool(active[0])
    assert float(weight[0]) == 0.5
    with pytest.raises(ValueError, match="out of range"):
        teacher.batch(torch.tensor([1]))


def test_source_gate_decision_uses_only_preregistered_source_metrics(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["factorized_primitive_state"]["sha256"] = "a" * 64
    manifest["source_gates"] = {
        "raw_radio_no_regression": {
            "candidate_mean_cosine_min": 0.99,
            "candidate_p05_cosine_min": 0.98,
        },
        "gauge_no_regression": {
            "candidate_mean_abs_log_amplitude_error_max": 0.03,
            "candidate_p95_abs_log_amplitude_error_max": 0.06,
        },
        "dino_sam_no_regression": {
            "maximum_mean_cosine_drop": 0.002,
            "maximum_p05_cosine_drop": 0.002,
            "dino_v3_control_mean_cosine": 0.98,
            "dino_v3_control_p05_cosine": 0.96,
            "sam3_control_mean_cosine": 0.97,
            "sam3_control_p05_cosine": 0.95,
        },
        "descriptor_improvement": {
            "required_hash_bound_support_intersection_rows": 8,
            "candidate_mean_cosine_minus_hash_bound_control_min": 0.002,
        },
        "descriptor_tail_no_regression": {
            "candidate_p05_cosine_minus_hash_bound_control_min": -0.001,
        },
        "semantic_relation_no_regression": {
            "required_deterministic_relation_edges": 16,
            "candidate_mean_abs_error_minus_hash_bound_control_max": 0.0,
            "candidate_p95_abs_error_minus_hash_bound_control_max": 0.002,
        },
        "scalar_authority_unchanged": {
            "required_factorized_primitive_state_sha256": "a" * 64,
            "required_valid_rows": 10,
        },
        "benchmark_gate": (
            "closed_until_every_source_gate_passes_then_one_shot_frozen_metric"
        ),
    }
    decision = evaluate_source_only_gates(
        manifest=manifest,
        control_source_metrics={
            "source_descriptor_mean_cosine": 0.90,
            "source_descriptor_p05_cosine": 0.80,
            "source_relation_mean_abs_error": 0.10,
            "source_relation_p95_abs_error": 0.20,
        },
        candidate_primary_metrics={
            "mean_cosine": 0.995,
            "p05_cosine": 0.985,
            "mean_abs_log_amplitude_error": 0.02,
            "p95_abs_log_amplitude_error": 0.05,
        },
        candidate_capability_metrics={
            "dino_v3_target_mean_cosine": 0.979,
            "dino_v3_target_p05_cosine": 0.959,
            "sam3_target_mean_cosine": 0.969,
            "sam3_target_p05_cosine": 0.949,
        },
        candidate_source_metrics={
            "source_descriptor_rows": 8,
            "source_descriptor_mean_cosine": 0.903,
            "source_descriptor_p05_cosine": 0.800,
            "source_relation_edges": 16,
            "source_relation_mean_abs_error": 0.09,
            "source_relation_p95_abs_error": 0.201,
        },
        primitive_state_summary={"valid_rows": 10},
    )
    assert decision["all_source_gates_passed"] is True
    assert decision["benchmark_gate_opened"] is True
    assert decision["benchmark_metric_read"] is False


def test_source_direction_objective_has_zero_gauge_gradient_but_trains_direction(
) -> None:
    radio = torch.zeros(2, 1280, requires_grad=True)
    with torch.no_grad():
        radio[0, :2] = torch.tensor([3.0, 4.0])
        radio[1, :2] = torch.tensor([2.0, -1.0])
    locked = source_direction_only_radio(radio)
    assert torch.allclose(locked, radio.detach(), atol=1e-6, rtol=1e-6)
    # A deliberately gauge-sensitive linear objective would have a radial
    # gradient without the detached-norm reparameterization.
    loss = locked[:, 0].sum() + 0.25 * locked[:, 1].sum()
    loss.backward()
    assert radio.grad is not None
    radial = (radio.grad * radio.detach()).sum(dim=-1)
    assert torch.allclose(radial, torch.zeros_like(radial), atol=1e-6)
    assert bool((torch.linalg.vector_norm(radio.grad, dim=-1) > 0).all())

    # The unchanged primary amplitude objective still owns the radial degree
    # of freedom, so gauge remains trainable by raw RADIO supervision.
    raw = radio.detach().clone().requires_grad_(True)
    raw_loss = torch.log(torch.linalg.vector_norm(raw, dim=-1)).sum()
    raw_loss.backward()
    assert raw.grad is not None
    assert bool(((raw.grad * raw.detach()).sum(dim=-1).abs() > 0.9).all())
