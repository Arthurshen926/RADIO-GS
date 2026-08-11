from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.build_source_official_sam_authority import (
    INVENTORY_SCHEMA,
    build_authority,
    validate_inventory,
)
from radio_gs.training.source_only_sam_structure import (
    OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA,
    SAM_STRUCTURE_SEPARATION_MARGIN,
    SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256,
    SourceSamStructureTeacher,
    evaluate_source_sam_structure_gates,
    load_source_only_sam_structure_bundle,
    source_only_sam_structure_contract,
    source_sam_structure_batch_loss,
    validate_single_radio_checkpoint_payload,
    validate_source_only_sam_structure_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, torch.Tensor, torch.Tensor, str, str]:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    graph_path = tmp_path / "graph.pt"
    graph = {
        "xyz": xyz,
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
    }
    torch.save(graph, graph_path)
    relation_path = tmp_path / "relation.pt"
    torch.save(
        {
            "schema_version": 2,
            "edge_index": graph["edge_index"],
            "same_votes": torch.tensor([[3.0, 1.0], [0.0, 1.0], [2.0, 1.0]]),
            "separate_votes": torch.tensor([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0]]),
            "metadata": {
                "teacher": "official_sam3_multimask_scale_ordered_regions",
                "query_free": True,
                "labels_opened": False,
                "instances_opened": False,
                "text_opened": False,
                "membership_lifting": "raster_adjoint",
                "scene_graph_sha256": _sha(graph_path),
            },
        },
        relation_path,
    )
    mask_path = tmp_path / "source-mask.pt"
    torch.save({"official": True}, mask_path)
    authority_path = tmp_path / "sam-authority.json"
    _write_json(
        authority_path,
        {
            "schema": OFFICIAL_SAM_BUILD_AUTHORITY_SCHEMA,
            "official_sam": True,
            "source_rgb_only": True,
            "query_free": True,
            "target_or_evaluation_rgb_opened": False,
            "benchmark_query_opened": False,
            "benchmark_gt_or_metric_opened": False,
            "teacher_artifacts_training_only": True,
            "source_mask_caches": [_record(mask_path)],
        },
    )
    control = tmp_path / "control.pt"
    factorized = tmp_path / "factorized.pt"
    torch.save({}, control)
    torch.save({}, factorized)
    contract = source_only_sam_structure_contract()
    manifest = {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract": contract,
        "contract_sha256": SOURCE_ONLY_SAM_STRUCTURE_CONTRACT_SHA256,
        "status": "preregistered_training_not_started",
        "scene_id": "synthetic",
        "field_control": _record(control),
        "canonical_radio_cache": _record(factorized),
        "relation_cache": _record(relation_path),
        "relation_graph": _record(graph_path),
        "official_sam_build_authority": _record(authority_path),
        "loss": contract["loss"],
        "source_gates": {
            "radio_reconstruction_no_regression": True,
            "gauge_no_regression": True,
            "sam_same_cosine_non_decrease": True,
            "sam_separate_cosine_non_increase": True,
            "sam_relation_gap_strict_improvement": True,
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
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, xyz, torch.ones(4, dtype=torch.bool), _sha(factorized), _sha(control)


def test_manifest_is_single_radio_source_only_and_rgb_free_at_query_time(tmp_path: Path) -> None:
    manifest_path, *_ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_source_only_sam_structure_manifest(payload)
    assert validated["contract"]["persistent_semantic_feature"] == "canonical_radio_only"
    assert validated["access_audit"]["source_rgb_opened_during_mapping"] is True
    assert validated["access_audit"]["query_time_source_rgb_opened"] is False
    assert validated["access_audit"]["query_time_target_rgb_opened"] is False


def test_manifest_rejects_query_time_source_rgb(tmp_path: Path) -> None:
    manifest_path, *_ = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["access_audit"]["query_time_source_rgb_opened"] = True
    with pytest.raises(ValueError, match="access audit"):
        validate_source_only_sam_structure_manifest(payload)


def test_loader_collapses_votes_and_abstains_on_ambiguous_edges(tmp_path: Path) -> None:
    manifest_path, xyz, valid, factorized_sha, control_sha = _fixture(tmp_path)
    bundle = load_source_only_sam_structure_bundle(
        manifest_path,
        expected_sha256=_sha(manifest_path),
        expected_xyz=xyz,
        expected_valid=valid,
        expected_canonical_radio_cache_sha256=factorized_sha,
        expected_field_checkpoint_sha256=control_sha,
    )
    # Edge 2 has same=3, separate=1 and exactly 0.5 dominance, so it remains.
    assert bundle.teacher.num_edges == 3
    assert bundle.teacher.same_relation.tolist() == [True, False, True]
    assert bundle.teacher.global_edge_index.tolist() == [[0, 1, 2], [1, 2, 3]]


class _ToyField(torch.nn.Module):
    def __init__(self, radio: torch.Tensor) -> None:
        super().__init__()
        self.local_codes = torch.nn.Parameter(radio.clone())

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        return self.local_codes[rows]


def test_structure_loss_pulls_same_and_pushes_cross_boundary_with_tangent_gradient() -> None:
    radio = torch.zeros(4, 1280)
    radio[0, 0] = 1.0
    radio[1, 0] = 0.8
    radio[1, 1] = 0.6
    radio[2, 0] = 1.0
    radio[3, 1] = 1.0
    field = _ToyField(radio)
    teacher = SourceSamStructureTeacher(
        global_edge_index=torch.tensor([[0, 1], [1, 2]]),
        same_relation=torch.tensor([True, False]),
        edge_weight=torch.ones(2),
        relation_cache_source=Path("synthetic.pt"),
        relation_cache_sha256="0" * 64,
    )
    loss, stats = source_sam_structure_batch_loss(
        field, teacher=teacher, edge_indices=torch.arange(2)
    )
    assert float(loss) > 0
    assert int(stats["same_edges"]) == 1
    assert int(stats["separate_edges"]) == 1
    loss.backward()
    for row in (0, 1, 2):
        gradient = field.local_codes.grad[row]
        value = field.local_codes.detach()[row]
        assert abs(float((gradient * value).sum())) < 1e-5
    assert SAM_STRUCTURE_SEPARATION_MARGIN == 0.25


def test_structure_gate_requires_both_directions_and_gap() -> None:
    control = {
        "sam_structure_edges": 10,
        "sam_same_edges": 5,
        "sam_separate_edges": 5,
        "sam_same_mean_cosine": 0.70,
        "sam_separate_mean_cosine": 0.40,
        "sam_relation_gap": 0.30,
    }
    candidate = {
        **control,
        "sam_same_mean_cosine": 0.72,
        "sam_separate_mean_cosine": 0.35,
        "sam_relation_gap": 0.37,
    }
    assert evaluate_source_sam_structure_gates(
        control=control, candidate=candidate
    )["all_structure_gates_passed"]
    candidate["sam_separate_mean_cosine"] = 0.41
    assert not evaluate_source_sam_structure_gates(
        control=control, candidate=candidate
    )["all_structure_gates_passed"]


def test_checkpoint_rejects_persistent_sam_feature_tensor() -> None:
    validate_single_radio_checkpoint_payload(
        {
            "state_dict": {"decoder.basis": torch.zeros(2, 2)},
            "source_only_sam_structure": {
                "teacher_payload_saved": False,
                "relation_edges": 12,
            },
        }
    )
    with pytest.raises(ValueError, match="teacher feature tensors"):
        validate_single_radio_checkpoint_payload(
            {"state_dict": {"sam_feature_bank": torch.zeros(2, 2)}}
        )


def test_source_authority_binds_explicit_official_cache_inventory(tmp_path: Path) -> None:
    cache = tmp_path / "frame7.pt"
    torch.save(
        {
            "metadata": {
                "schema_version": 2,
                "source": "official_sam3_interactive_grid_multimask_hierarchy",
                "official_decoder": True,
                "query_free": True,
                "image": str(tmp_path / "frame7.jpg"),
                "checkpoint_sha256": "a" * 64,
            }
        },
        cache,
    )
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "scene_id": "scene",
        "source_frames": ["frame7"],
        "source_split_authority": _record(cache),
        "target_or_evaluation_frames_excluded": True,
        "benchmark_queries_excluded": True,
        "mask_caches": [_record(cache)],
    }
    authority = build_authority(
        inventory,
        inventory_path=tmp_path / "inventory.json",
        inventory_sha256="b" * 64,
    )
    assert authority["source_rgb_only"] is True
    assert authority["target_or_evaluation_rgb_opened"] is False
    assert authority["teacher_artifacts_training_only"] is True


def test_source_authority_allows_numeric_alias_for_resolved_lerf_image(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "1.pt"
    torch.save(
        {
            "metadata": {
                "schema_version": 2,
                "source": "official_sam3_interactive_grid_multimask_hierarchy",
                "official_decoder": True,
                "query_free": True,
                "image": str(tmp_path / "frame_00001.jpg"),
                "checkpoint_sha256": "a" * 64,
            }
        },
        cache,
    )
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "scene_id": "figurines",
        "source_frames": ["1"],
        "source_split_authority": _record(cache),
        "target_or_evaluation_frames_excluded": True,
        "benchmark_queries_excluded": True,
        "mask_caches": [_record(cache)],
    }
    authority = build_authority(
        inventory,
        inventory_path=tmp_path / "inventory.json",
        inventory_sha256="b" * 64,
    )
    assert authority["source_frames"] == ["1"]
    assert authority["source_images"] == [str(tmp_path / "frame_00001.jpg")]


def test_source_inventory_requires_explicit_target_exclusion(tmp_path: Path) -> None:
    cache = tmp_path / "frame7.pt"
    torch.save({}, cache)
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "scene_id": "scene",
        "source_frames": ["frame7"],
        "source_split_authority": _record(cache),
        "target_or_evaluation_frames_excluded": False,
        "benchmark_queries_excluded": True,
        "mask_caches": [_record(cache)],
    }
    with pytest.raises(ValueError, match="contract"):
        validate_inventory(inventory)
