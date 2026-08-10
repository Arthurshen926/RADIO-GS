from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.interfaces import region_comembership_native_v3_target as formal
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.scripts.infer_region_comembership_native_v3 import (
    exact_anchor_fallback_fusion,
    query_free_graph_health,
)
from radio_gs.scripts.materialize_region_comembership_features_native_v3 import (
    materialize_native_pair_block,
)


def _state() -> FactorizedPrimitiveState:
    return FactorizedPrimitiveState(
        xyz=torch.zeros(4, 3),
        valid=torch.tensor([True, False, True, True]),
        global_rows=torch.tensor([0, 2, 3]),
        semantic_direction=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        ),
        predicted_log_amplitude=torch.tensor([0.1, 0.2, 0.3]),
        directional_dispersion=torch.tensor([0.1, 0.2, 0.3]),
        log_amplitude_std=torch.tensor([0.2, 0.3, 0.4]),
        observation_evidence=torch.tensor([0.8, 0.7, 0.6]),
        visibility_purity_value=torch.tensor([0.9, 0.0, 0.7]),
        visibility_purity_known=torch.tensor([True, False, True]),
        metadata={"query_independent": True},
    )


def test_native_pair_block_requires_two_exact_canonical_anchors() -> None:
    pairs = torch.tensor([[0, 0, 1], [1, 2, 2]])
    block = materialize_native_pair_block(
        state=_state(),
        region_rows=torch.tensor([[0, 2], [1, 2], [3, 2]]),
        token_mask=torch.ones(3, 2, dtype=torch.bool),
        canonical_anchor_index=torch.zeros(3, dtype=torch.long),
        pair_indices=pairs,
        batch_size=1,
    )
    assert torch.equal(
        block["exact_state_anchor_mask"], torch.tensor([True, False, True])
    )
    assert torch.equal(
        block["native_pair_active_mask"], torch.tensor([False, True, False])
    )
    assert not bool(
        block["native_pair_features"][
            block["legacy_v2_fallback_pair_mask"]
        ].count_nonzero()
    )
    assert bool(block["native_pair_features"][1].abs().sum() > 0)


def test_exact_anchor_fallback_is_bitwise_parent_v2_for_probability_and_edge() -> None:
    native = torch.tensor([0.1, 0.9, 0.7, 0.2])
    parent = torch.tensor([0.3, 0.4, 0.8, 0.6])
    parent_edge = torch.tensor([False, True, True, False])
    active = torch.tensor([True, False, True, False])
    probability, edge = exact_anchor_fallback_fusion(
        native_probability=native,
        parent_probability=parent,
        parent_accepted_edge_mask=parent_edge,
        native_pair_active_mask=active,
        native_threshold=0.75,
    )
    assert torch.equal(probability[~active], parent[~active])
    assert torch.equal(edge[~active], parent_edge[~active])
    assert torch.equal(probability[active], native[active])
    assert torch.equal(edge[active], torch.tensor([False, False]))


def test_query_free_health_reports_components_without_labels_or_queries() -> None:
    pairs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    health = query_free_graph_health(
        region_count=4,
        pair_indices=pairs,
        probability=torch.tensor([0.9, 0.2, 0.8]),
        accepted_edge_mask=torch.tensor([True, False, True]),
        native_pair_active_mask=torch.tensor([True, True, False]),
        parent_accepted_edge_mask=torch.tensor([False, False, True]),
    )
    assert health["accepted_edges"] == 2
    assert health["connected_components_including_isolates"] == 2
    assert health["largest_component_regions"] == 2
    assert health["edge_additions_vs_parent_v2"] == 1
    assert health["query_readout_executed"] is False
    assert health["target_metric_computed"] is False


def test_target_execution_gate_stops_before_target_records_on_source_failure(
    monkeypatch, tmp_path: Path
) -> None:
    record = {"path": "/frozen", "sha256": "a" * 64}
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": formal.TARGET_EXECUTION_STATUS,
        "scene_id": "figurines",
        "source_result": record,
        "promoted_checkpoint": record,
        "builder_implementation": record,
        "implementation_dependencies": {
            name: record for name in formal.TARGET_IMPLEMENTATION_PATHS
        },
        "target_inputs": {name: record for name in formal.TARGET_INPUT_NAMES},
        "target_feature_output": str(tmp_path / "feature.pt"),
        "target_inference_output": str(tmp_path / "inference.pt"),
        "feature_materialization_authorized": True,
        "checkpoint_inference_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "fallback_contract": formal.fallback_contract(),
        "access_audit": formal.access_audit(target_opened=True),
    }
    order: list[str] = []

    def fake_load(*args, **kwargs):
        return authority, "b" * 64, tmp_path / "authority.json"

    def source_stop(*args, **kwargs):
        order.append("source")
        raise RuntimeError("source promotion stop")

    def target_open(*args, **kwargs):
        order.append("target")
        raise AssertionError("target opened before source promotion")

    monkeypatch.setattr(formal, "load_json_object", fake_load)
    monkeypatch.setattr(formal, "validate_source_promotion", source_stop)
    monkeypatch.setattr(formal, "validate_file_record", target_open)
    with pytest.raises(RuntimeError, match="source promotion stop"):
        formal.validate_target_execution_authority(
            tmp_path / "authority.json",
            expected_sha256="b" * 64,
            scene_id="figurines",
        )
    assert order == ["source"]
