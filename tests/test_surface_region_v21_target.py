from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    target_physical_space_authority,
)
from radio_gs.interfaces import surface_region_v21_target as formal
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import materialize_full_scalar_clean_training_shard as shard
from radio_gs.scripts.materialize_surface_region_v21_target_descriptor import (
    apply_v21_canonical_forward,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


def _normalization() -> dict:
    return {
        "median": torch.zeros(30),
        "robust_scale": torch.ones(30),
        "constant_coordinate_mask": torch.zeros(30, dtype=torch.bool),
        "source_max_robust_linf": 2.0,
    }


def test_complete_canonical_forward_updates_only_eligible_in_envelope_rows() -> None:
    torch.manual_seed(0)
    base = F.normalize(torch.randn(4, 1536), dim=-1)
    context = torch.zeros(4, 1280)
    context[:3, 0] = 1.0
    full_scalar = torch.zeros(4, 18)
    full_scalar[1, 0] = 3.0
    statistics = torch.zeros(4, 12)
    eligible = torch.tensor([True, True, False, True])
    typed = torch.tensor([True, True, True, False])
    model = SurfaceRegionAcceptedV2TypedContextResidualV1()
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-0.2, 0.2, 1536))
    result = apply_v21_canonical_forward(
        accepted_v2_e0=base,
        pooled_context_radio_direction=context,
        raw_full_scalar_summary=full_scalar,
        typed_context_statistics=statistics,
        full_scalar_eligible=eligible,
        typed_context_valid=typed,
        model=model,
        normalization=_normalization(),
        batch_size=1,
    )
    assert result["active_update_mask"].tolist() == [True, False, False, False]
    assert result["normalization_ood_mask"].tolist() == [False, True, False, False]
    assert result["effective_ood_mask"].tolist() == [False, True, True, False]
    assert result["immutable_fallback_mask"].tolist() == [False, True, True, True]
    assert result["fallback_bitwise_equal"] is True
    assert torch.equal(result["semantic_descriptor"][1:], base[1:])
    assert not torch.equal(result["semantic_descriptor"][0], base[0])


def _execution(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    records = {}
    for name in formal.TARGET_INPUT_NAMES:
        target = tmp_path / f"{name}.pt"
        target.write_bytes(name.encode("utf-8"))
        records[name] = file_record(target)
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_v21_source_promotion_for_query_free_descriptor_only",
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:geometry-checkpoint-sha256:" + "9" * 64,
        "source_pilot_result": {"path": "/source/result.json", "sha256": "1" * 64},
        "implementation": file_record(formal.TARGET_IMPLEMENTATION_PATH),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.TARGET_IMPLEMENTATION_DEPENDENCIES.items()
        },
        "preregistration": file_record(formal.TARGET_PREREGISTRATION_PATH),
        "target_inputs": records,
        "target_descriptor_output": str((tmp_path / "descriptor.pt").resolve()),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_descriptor_access_audit(),
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    return path, records


def test_source_gate_rejection_occurs_before_any_target_file_record(
    monkeypatch, tmp_path: Path
) -> None:
    path, _ = _execution(tmp_path)
    opened = {"target": 0}

    def reject(*args, **kwargs):
        raise ValueError("source promotion rejected")

    def target_open(*args, **kwargs):
        opened["target"] += 1
        raise AssertionError("target record opened before source gate")

    monkeypatch.setattr(formal, "validate_source_pilot_chain", reject)
    monkeypatch.setattr(formal, "validate_file_record", target_open)
    with pytest.raises(ValueError, match="source promotion rejected"):
        formal.validate_target_execution_authority(
            path, expected_sha256=sha256_file(path)
        )
    assert opened["target"] == 0


def test_target_execution_binds_promoted_checkpoint_and_normalization(
    monkeypatch, tmp_path: Path
) -> None:
    path, records = _execution(tmp_path)
    monkeypatch.setattr(
        formal,
        "validate_source_pilot_chain",
        lambda *args, **kwargs: {
            "source_promotion_authorized": True,
            "checkpoint": records["v21_checkpoint"],
            "normalization_authority": records["v21_normalization"],
        },
    )
    authority = formal.validate_target_execution_authority(
        path,
        expected_sha256=sha256_file(path),
        expected_scene_id="figurines",
        expected_output=tmp_path / "descriptor.pt",
    )
    assert authority["target_inputs"] == records
    assert authority["query_execution_authorized"] is False


def test_target_descriptor_output_validates_routing_and_no_query() -> None:
    geometry_sha = "9" * 64
    physical = target_physical_space_authority(
        dataset_id="lerf",
        scene_id="figurines",
        geometry_checkpoint_sha256=geometry_sha,
    )
    descriptor = torch.zeros(2, shard.trainer.DESCRIPTOR_DIM)
    descriptor[0, 0] = descriptor[1, 1] = 1.0
    masks = {
        "full_scalar_eligible_mask": torch.tensor([True, False]),
        "typed_context_valid_mask": torch.tensor([True, True]),
        "normalization_ood_mask": torch.tensor([False, False]),
        "effective_ood_mask": torch.tensor([False, True]),
        "active_update_mask": torch.tensor([True, False]),
        "immutable_fallback_mask": torch.tensor([False, True]),
        "descriptor_changed_mask": torch.tensor([True, False]),
    }
    record = {"path": "/synthetic/input.pt", "sha256": "1" * 64}
    payload = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": formal.target_descriptor_contract(),
        "contract_sha256": formal.TARGET_DESCRIPTOR_CONTRACT_SHA256,
        "scene_id": "figurines",
        "physical_space_id": physical["physical_space_id"],
        "physical_space_authority": physical,
        "producer": record,
        "target_execution_authority": record,
        "input_authority": {name: record for name in formal.TARGET_INPUT_NAMES},
        "region_row_ids": ["r0", "r1"],
        "canonical_region_indices": torch.tensor([0, 1]),
        "region_fingerprints": ["2" * 64, "3" * 64],
        "semantic_descriptor": descriptor,
        **masks,
        "fallback_bitwise_equal": True,
        "routing_audit": {
            "regions": 2,
            "full_scalar_eligible": 1,
            "typed_context_valid": 2,
            "normalization_ood": 0,
            "effective_ood": 1,
            "active_update": 1,
            "immutable_fallback": 1,
            "descriptor_changed": 1,
        },
        "channel_sha256": {},
        "access_audit": formal.target_descriptor_access_audit(),
    }
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    validated = formal.validate_target_descriptor_authority(payload)
    assert validated["access_audit"]["query_relevance_computed"] is False

    payload["descriptor_changed_mask"][1] = True
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    with pytest.raises(ValueError, match="routing masks"):
        formal.validate_target_descriptor_authority(payload)
