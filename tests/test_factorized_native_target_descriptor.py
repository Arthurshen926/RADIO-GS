from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_gauge_state_readout as readout
from radio_gs.interfaces import factorized_native_target_descriptor as formal
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
)
from radio_gs.scripts.materialize_factorized_native_target_descriptor import (
    apply_factorized_native_canonical_forward,
)


def _normalization():
    return readout.validate_source_normalization(
        {
            "schema": readout.NORMALIZATION_SCHEMA,
            "schema_version": readout.NORMALIZATION_SCHEMA_VERSION,
            "interface_contract_sha256": readout.INTERFACE_CONTRACT_SHA256,
            "source_state_cohort_authority_sha256": "a" * 64,
            "state_names": list(readout.FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES),
            "state_names_sha256": readout.FACTORIZED_PRIMITIVE_STATE_SCALAR_NAMES_SHA256,
            "state_median": torch.zeros(6),
            "state_robust_scale": torch.ones(6),
            "log_amplitude_median": torch.zeros(1),
            "log_amplitude_robust_scale": torch.ones(1),
            "known_count_by_state_column": [3] * 6,
            "fit_scene_count": 4,
            "source_access": readout.source_access(),
        }
    )


def _state() -> FactorizedPrimitiveState:
    torch.manual_seed(3)
    direction = F.normalize(torch.randn(3, 1280), dim=-1)
    return FactorizedPrimitiveState(
        xyz=torch.zeros(4, 3),
        valid=torch.tensor([True, True, False, True]),
        global_rows=torch.tensor([0, 1, 3]),
        semantic_direction=direction,
        predicted_log_amplitude=torch.tensor([0.1, -0.2, 0.3]),
        directional_dispersion=torch.tensor([0.1, 0.2, 0.3]),
        log_amplitude_std=torch.tensor([0.2, 0.3, 0.4]),
        observation_evidence=torch.tensor([0.8, 0.7, 0.6]),
        visibility_purity_value=torch.tensor([0.9, 0.8, 0.7]),
        visibility_purity_known=torch.ones(3, dtype=torch.bool),
        metadata={"query_independent": True},
    )


class _FakeHead(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 1280:
            raise ValueError("fake head input differs")
        return torch.cat((value, value[..., :256]), dim=-1)


def _record(letter: str) -> dict[str, str]:
    return {"path": f"/tmp/{letter}", "sha256": letter * 64}


def _payload(forward: dict[str, torch.Tensor | bool]) -> dict:
    physical = {
        "kind": "target_geometry_checkpoint_v1",
        "dataset_id": "lerf3d",
        "scene_id": "teatime",
        "geometry_checkpoint_sha256": "f" * 64,
        "physical_space_id": (
            "lerf3d:teatime:geometry-checkpoint-sha256:" + "f" * 64
        ),
    }
    fingerprints = ["1" * 64, "2" * 64]
    masks = {
        name: forward[name]
        for name in (
            "exact_state_anchor_mask",
            "active_update_mask",
            "immutable_fallback_mask",
            "descriptor_changed_mask",
        )
    }
    payload = {
        "schema": formal.TARGET_DESCRIPTOR_SCHEMA,
        "schema_version": 1,
        "contract": formal.target_descriptor_contract(),
        "contract_sha256": formal.TARGET_DESCRIPTOR_CONTRACT_SHA256,
        "scene_id": "teatime",
        "physical_space_id": physical["physical_space_id"],
        "physical_space_authority": physical,
        "producer": _record("a"),
        "target_execution_authority": _record("b"),
        "input_authority": {
            "source_arm_results": {
                arm: _record(str(index + 3))
                for index, arm in enumerate(FACTORIZED_NATIVE_READOUT_ARMS)
            },
            "winner_checkpoint": _record("6"),
            "winner_normalization": _record("7"),
            "official_radio_checkpoint": _record("8"),
            "target_accepted_v2": _record("9"),
            "factorized_primitive_state": _record("c"),
        },
        "winner_arm": FACTORIZED_NATIVE_READOUT_ARMS[0],
        "winner_selected_step": 4,
        "winner_source_ranking": {
            "macro_mean_all_view_cosine": 0.8,
            "macro_p05_row_mean_all_view_cosine": 0.7,
            "arm_order_tie_break": 0,
        },
        "region_row_ids": ["teatime:r0", "teatime:r1"],
        "canonical_region_indices": torch.tensor([0, 1]),
        "region_fingerprints": fingerprints,
        "semantic_descriptor": forward["semantic_descriptor"],
        **masks,
        "fallback_bitwise_equal": True,
        "routing_audit": {
            "regions": 2,
            "exact_state_anchor": int(masks["exact_state_anchor_mask"].sum()),
            "active_update": int(masks["active_update_mask"].sum()),
            "immutable_fallback": int(masks["immutable_fallback_mask"].sum()),
            "descriptor_changed": int(masks["descriptor_changed_mask"].sum()),
        },
        "access_audit": formal.target_descriptor_access_audit(),
    }
    payload["channel_sha256"] = formal.target_descriptor_channel_sha256(payload)
    return payload


def test_forward_uses_exact_anchor_and_preserves_bitwise_fallback() -> None:
    torch.manual_seed(5)
    base = F.normalize(torch.randn(2, 1536), dim=-1)
    model = readout.build_model("direction_only", _normalization())
    result = apply_factorized_native_canonical_forward(
        accepted_v2_e0=base,
        region_rows=torch.tensor([[0, 1], [2, 3]]),
        token_mask=torch.ones(2, 2, dtype=torch.bool),
        anchor_index=torch.tensor([0, 0]),
        state=_state(),
        model=model,
        head=_FakeHead(),
        batch_size=1,
    )
    assert torch.equal(result["exact_state_anchor_mask"], torch.tensor([True, False]))
    assert torch.equal(result["active_update_mask"], torch.tensor([True, False]))
    assert torch.equal(result["semantic_descriptor"][1], base[1])
    assert result["fallback_bitwise_equal"] is True
    assert torch.allclose(
        torch.linalg.vector_norm(result["semantic_descriptor"], dim=-1),
        torch.ones(2),
        atol=2e-4,
        rtol=0.0,
    )


def test_descriptor_view_is_exact_query_compatible_and_honest() -> None:
    torch.manual_seed(7)
    base = F.normalize(torch.randn(2, 1536), dim=-1)
    forward = {
        "semantic_descriptor": base,
        "exact_state_anchor_mask": torch.tensor([True, False]),
        "active_update_mask": torch.tensor([True, False]),
        "immutable_fallback_mask": torch.tensor([False, True]),
        "descriptor_changed_mask": torch.tensor([True, False]),
        "fallback_bitwise_equal": True,
    }
    payload = _payload(forward)
    checked = formal.validate_target_descriptor_authority(payload)
    view = formal.exact_query_descriptor_view(checked)
    assert view["schema"] == formal.EXACT_QUERY_DESCRIPTOR_VIEW_SCHEMA
    assert view["semantic_descriptor"].shape == (2, 1536)
    assert torch.equal(view["semantic_descriptor"], base)
    assert checked["schema"] != "radio_gs.surface_region_v21_target_descriptor_authority.v1"
    assert formal.target_descriptor_contract()["legacy_default_changed"] is False


def test_descriptor_validator_rejects_update_on_fallback() -> None:
    base = F.normalize(torch.randn(2, 1536), dim=-1)
    forward = {
        "semantic_descriptor": base,
        "exact_state_anchor_mask": torch.tensor([True, False]),
        "active_update_mask": torch.tensor([True, False]),
        "immutable_fallback_mask": torch.tensor([False, True]),
        "descriptor_changed_mask": torch.tensor([True, True]),
        "fallback_bitwise_equal": True,
    }
    payload = _payload(forward)
    with pytest.raises(ValueError, match="routing masks"):
        formal.validate_target_descriptor_authority(payload)


def test_execution_gate_calls_source_before_target_records(monkeypatch, tmp_path: Path) -> None:
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_three_arm_source_selection_for_query_free_target",
        "source_arm_results": {
            arm: _record(str(index + 3))
            for index, arm in enumerate(FACTORIZED_NATIVE_READOUT_ARMS)
        },
        "implementation": _record("a"),
        "implementation_dependencies": {
            name: _record("b") for name in formal.TARGET_IMPLEMENTATION_DEPENDENCIES
        },
        "target_inputs": {
            "target_accepted_v2": _record("c"),
            "factorized_primitive_state": _record("d"),
        },
        "target_descriptor_output": str(tmp_path / "descriptor.pt"),
        "materialization_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "access_audit": formal.target_descriptor_access_audit(),
    }
    order: list[str] = []

    def fake_load(*args, **kwargs):
        return authority, "e" * 64, tmp_path / "authority.json"

    def stop_at_source(*args, **kwargs):
        order.append("source")
        raise RuntimeError("source stop")

    def target_open(*args, **kwargs):
        order.append("target")
        raise AssertionError("target opened before source gate")

    monkeypatch.setattr(formal, "load_json_object", fake_load)
    monkeypatch.setattr(formal, "validate_source_arm_winner", stop_at_source)
    monkeypatch.setattr(formal, "validate_file_record", target_open)
    with pytest.raises(RuntimeError, match="source stop"):
        formal.validate_target_execution_authority(
            tmp_path / "authority.json", expected_sha256="e" * 64
        )
    assert order == ["source"]
