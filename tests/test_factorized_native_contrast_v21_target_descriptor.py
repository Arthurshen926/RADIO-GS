from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as formal
from radio_gs.models.factorized_native_gauge_state_readout import DIRECTION_ONLY
from radio_gs.scripts import (
    train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 as source,
)


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def _payload() -> dict:
    torch.manual_seed(71)
    descriptor = F.normalize(torch.randn(3, 1536), dim=-1)
    exact = torch.tensor([True, False, True])
    changed = torch.tensor([True, False, True])
    physical = {
        "kind": "target_geometry_checkpoint_v1",
        "dataset_id": "lerf3d",
        "scene_id": "figurines",
        "geometry_checkpoint_sha256": "f" * 64,
        "physical_space_id": (
            "lerf3d:figurines:geometry-checkpoint-sha256:" + "f" * 64
        ),
    }
    payload = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": formal.target_descriptor_contract(),
        "contract_sha256": formal.TARGET_DESCRIPTOR_CONTRACT_SHA256,
        "scene_id": "figurines",
        "physical_space_id": physical["physical_space_id"],
        "physical_space_authority": physical,
        "producer": _record("a"),
        "target_execution_authority": _record("b"),
        "input_authority": {
            name: _record(str(index + 1))
            for index, name in enumerate(formal.DESCRIPTOR_INPUT_NAMES)
        },
        "source_arm": DIRECTION_ONLY,
        "source_selected_step": 60,
        "source_gate_audit": {
            "result_schema": source.RESULT_SCHEMA,
            "checkpoint_schema": source.CHECKPOINT_SCHEMA,
            "schema_version": 21,
            "status": "source_only_contrast_v21_promotion_candidate_complete",
            "source_only_passed": True,
        },
        "region_row_ids": ["figurines:r0", "figurines:r1", "figurines:r2"],
        "canonical_region_indices": torch.tensor([0, 2, 9]),
        "region_fingerprints": ["1" * 64, "2" * 64, "3" * 64],
        "semantic_descriptor": descriptor,
        "exact_state_anchor_mask": exact,
        "active_update_mask": exact.clone(),
        "immutable_fallback_mask": ~exact,
        "descriptor_changed_mask": changed,
        "fallback_bitwise_equal": True,
        "routing_audit": {
            "regions": 3,
            "exact_state_anchor": 2,
            "active_update": 2,
            "immutable_fallback": 1,
            "descriptor_changed": 2,
        },
        "access_audit": formal.target_descriptor_access_audit(),
    }
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    return payload


def _eligible_validation(score: float) -> dict:
    return {
        "macro_mean_teacher_centered_residual_cosine": score,
        "macro_teacher_centered_pair_gram_correlation": score,
        "macro_mean_all_view_cosine": score,
        "selection": {"eligible": True},
    }


def test_result_header_requires_exact_schema21_and_selected_source_step() -> None:
    steps = [0, *range(5, 61, 5)]
    history = [
        {
            "step": step,
            "validation": (
                _eligible_validation(float(step))
                if step == 60
                else {"selection": {"eligible": False}}
            ),
            "model_state_dict_sha256": f"{step:064x}",
        }
        for step in steps
    ]
    result = {
        "schema": source.RESULT_SCHEMA,
        "schema_version": 21,
        "status": "source_only_contrast_v21_promotion_candidate_complete",
        "arm": DIRECTION_ONLY,
        "training_contract": source.training_contract(),
        "training_contract_sha256": formal.canonical_json_sha256(
            source.training_contract()
        ),
        "execution_authority": _record("a"),
        "normalization": _record("b"),
        "contrast_reference": _record("c"),
        "checkpoint": _record("d"),
        "selected_step": 60,
        "history": history,
        "last_training_step": {"step": 60, "per_scene": {}},
        "benchmark_opened": False,
        "source_access": source.source_access(),
    }
    checked = formal._validate_result_header(result, record=_record("e"))
    assert checked["selected_step"] == 60
    result["schema_version"] = 2
    with pytest.raises(ValueError, match="contract differs"):
        formal._validate_result_header(result, record=_record("e"))


def test_descriptor_has_exact_query_view_without_impersonating_v1() -> None:
    checked = formal.validate_target_descriptor_authority(_payload())
    view = formal.exact_query_descriptor_view(checked)
    assert view["schema"] == formal.EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA
    assert view["source_descriptor_schema"] == formal.TARGET_DESCRIPTOR_SCHEMA
    assert torch.equal(view["semantic_descriptor"], checked["semantic_descriptor"])
    assert formal.target_descriptor_contract()["legacy_v1_consumer_changed"] is False


def test_descriptor_rejects_non_direction_source_or_fallback_change() -> None:
    payload = _payload()
    payload["source_arm"] = "direction_plus_log_amplitude"
    with pytest.raises(ValueError, match="contract differs"):
        formal.validate_target_descriptor_authority(payload)
    payload = _payload()
    payload["descriptor_changed_mask"][1] = True
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    with pytest.raises(ValueError, match="routing masks"):
        formal.validate_target_descriptor_authority(payload)


def test_execution_gate_runs_source_before_target_open(monkeypatch, tmp_path: Path) -> None:
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_contrast_v21_source_promotion_for_query_free_target",
        "source_contrast_v21_result": _record("a"),
        "implementation": _record("b"),
        "implementation_dependencies": {
            name: _record("c") for name in formal.TARGET_IMPLEMENTATION_DEPENDENCIES
        },
        "target_inputs": {
            "target_accepted_v2": _record("d"),
            "factorized_primitive_state": _record("e"),
        },
        "target_descriptor_output": str(tmp_path / "descriptor.pt"),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_descriptor_access_audit(),
    }
    order: list[str] = []

    monkeypatch.setattr(
        formal,
        "load_json_object",
        lambda *args, **kwargs: (authority, "f" * 64, tmp_path / "authority.json"),
    )

    def source_stop(*args, **kwargs):
        order.append("source")
        raise RuntimeError("source stop")

    def target_open(*args, **kwargs):
        order.append("target")
        raise AssertionError("target opened before source gate")

    monkeypatch.setattr(formal, "validate_source_contrast_v21_result", source_stop)
    monkeypatch.setattr(formal, "validate_file_record", target_open)
    with pytest.raises(RuntimeError, match="source stop"):
        formal.validate_target_execution_authority(
            tmp_path / "authority.json", expected_sha256="f" * 64
        )
    assert order == ["source"]
