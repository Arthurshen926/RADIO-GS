from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces.factorized_native_region_relation import (
    FEATURE_NAMES as NATIVE_FEATURE_NAMES,
    FEATURE_NAMES_SHA256 as NATIVE_FEATURE_NAMES_SHA256,
    INTERFACE_CONTRACT_SHA256,
)
from radio_gs.interfaces.factorized_primitive_state import FactorizedPrimitiveState
from radio_gs.models.region_comembership_native_v3 import PAIR_FEATURE_NAMES
from radio_gs.models.region_comembership_v2 import (
    PAIR_FEATURE_NAMES as V2_PAIR_FEATURE_NAMES,
)
from radio_gs.scripts import materialize_source_region_comembership_native_v3 as builder
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
)


def _state() -> FactorizedPrimitiveState:
    return FactorizedPrimitiveState(
        xyz=torch.zeros(5, 3),
        valid=torch.tensor([True, False, True, False, True]),
        global_rows=torch.tensor([0, 2, 4]),
        semantic_direction=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float16
        ),
        predicted_log_amplitude=torch.tensor([1.0, 2.0, 3.0]),
        directional_dispersion=torch.tensor([0.1, 0.2, 0.3]),
        log_amplitude_std=torch.tensor([0.1, 0.2, 0.3]),
        observation_evidence=torch.tensor([0.8, 0.7, 0.6]),
        visibility_purity_value=torch.tensor([0.9, 0.0, 0.7]),
        visibility_purity_known=torch.tensor([True, False, True]),
        metadata={},
    )


def test_indexed_gather_uses_compact_state_and_exact_missingness() -> None:
    state = _state()
    index = builder._factorized_state_index(state)
    gathered = builder._gather_indexed_factorized_native_region_inputs(
        state,
        index,
        torch.tensor([[0, 2, -1], [4, 3, -1]]),
        torch.tensor([[True, True, False], [True, True, False]]),
        torch.tensor([0, 0]),
    )
    assert gathered.unit_direction.dtype == torch.float32
    assert gathered.token_mask.tolist() == [[True, True, False], [True, False, False]]
    assert gathered.state_known_mask[0, 1, 4].item() is False
    assert gathered.state[0, 1, 4].item() == 0.0
    assert torch.equal(gathered.log_amplitude, gathered.state[..., 0])


def _payload(tmp_path: Path) -> dict:
    proof = tmp_path / "proof.bin"
    proof.write_bytes(b"frozen")
    record = file_record(proof)
    pairs = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.int64)
    v2_features = torch.zeros(3, len(V2_PAIR_FEATURE_NAMES), dtype=torch.float32)
    native_features = torch.zeros(3, len(NATIVE_FEATURE_NAMES), dtype=torch.float32)
    identity = {
        "schema": builder.SCHEMA,
        "schema_version": builder.SCHEMA_VERSION,
        "scene_id": "scene0001_00",
        "split": "source_train",
        "producer": record,
        "input_authority": {
            "parent_v2_source_authority": record,
            "accepted_v2": record,
            "factorized_state": record,
        },
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "pair_feature_names_sha256": canonical_json_sha256(list(PAIR_FEATURE_NAMES)),
        "native_feature_names": list(NATIVE_FEATURE_NAMES),
        "native_feature_names_sha256": NATIVE_FEATURE_NAMES_SHA256,
        "native_interface_contract_sha256": INTERFACE_CONTRACT_SHA256,
        "primitive_metric_contract": dict(builder.PRIMITIVE_METRIC_CONTRACT),
        "source_access": builder.source_access(),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": ["a", "b", "c"],
        "canonical_region_indices": torch.arange(3, dtype=torch.int64),
        "region_rows": torch.arange(3, dtype=torch.int64)[:, None],
        "token_mask": torch.ones(3, 1, dtype=torch.bool),
        "dominant_instance_ids": torch.tensor([1, 1, 2], dtype=torch.int64),
        "instance_purity": torch.ones(3, dtype=torch.float32),
        "instance_label_coverage": torch.ones(3, dtype=torch.float32),
        "instance_observed": torch.ones(3, dtype=torch.bool),
        "pair_indices": pairs,
        "v2_pair_features": v2_features,
        "native_pair_features": native_features,
        "pair_features": torch.cat((v2_features, native_features), dim=1),
        "same_instance_targets": torch.tensor([True, False, False]),
        "pair_evidence_weights": torch.ones(3, dtype=torch.float32),
        "primitive_instance_flat_keys": torch.tensor([1, 4, 8], dtype=torch.int64),
        "primitive_instance_mass": torch.ones(3, dtype=torch.float32),
        "primitive_count": 3,
        "instance_columns_including_zero": 3,
        "channel_sha256": {},
        "audit": {},
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in builder.TENSOR_CHANNELS
    }
    return payload


def test_native_v3_source_authority_validates_combined_axis_and_fails_closed(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    value = builder.validate_source_region_comembership_native_v3(payload)
    assert value["pair_features"].shape == (3, 30)
    assert value["source_access"]["target_metrics_computed"] is False
    tampered = deepcopy(payload)
    tampered["native_pair_features"][0, 0] = 1.0
    tampered["channel_sha256"]["native_pair_features"] = tensor_sha256(
        tampered["native_pair_features"]
    )
    with pytest.raises(ValueError):
        builder.validate_source_region_comembership_native_v3(tampered)
