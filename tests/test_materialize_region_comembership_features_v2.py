from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.models.region_comembership_v1 import (
    PAIR_FEATURE_NAMES as V1_PAIR_FEATURE_NAMES,
)
from radio_gs.models.region_comembership_v2 import PAIR_FEATURE_NAMES
from radio_gs.scripts import infer_region_comembership_v1 as v1_inference
from radio_gs.scripts import materialize_region_capability_descriptors_v2 as capability
from radio_gs.scripts import materialize_region_comembership_features_v1 as v1
from radio_gs.scripts.materialize_region_comembership_features_v2 import (
    combine_feature_authorities,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


def _v1_feature() -> dict:
    accepted = {"path": "/not-opened.pt", "sha256": "a" * 64}
    identity = {
        "schema": v1.SCHEMA,
        "schema_version": 1,
        "scene_id": "scene0001_00",
        "domain": "source_parity",
        "producer": file_record(Path(v1.__file__).resolve()),
        "target_execution_authority": None,
        "input_authority": {"accepted_v2": accepted},
        "candidate_policy": {
            "descriptor_neighbors": 16,
            "centroid_neighbors": 16,
            "anchor_support_edges": True,
        },
        "feature_names": list(V1_PAIR_FEATURE_NAMES),
        "feature_names_sha256": canonical_json_sha256(list(V1_PAIR_FEATURE_NAMES)),
        "source_access": v1._source_access("source_parity"),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": ["a", "b", "c"],
        "canonical_region_indices": torch.arange(3, dtype=torch.int64),
        "region_rows": torch.arange(3, dtype=torch.int64)[:, None],
        "token_mask": torch.ones(3, 1, dtype=torch.bool),
        "pair_indices": torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.int64),
        "pair_features": torch.zeros(3, len(V1_PAIR_FEATURE_NAMES)),
        "channel_sha256": {},
        "audit": {},
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name])
        for name in (
            "canonical_region_indices",
            "region_rows",
            "token_mask",
            "pair_indices",
            "pair_features",
        )
    }
    v1_inference.validate_feature_authority(payload)
    return payload


def _descriptor() -> dict:
    identity = {
        "schema": capability.SCHEMA,
        "schema_version": 2,
        "scene_id": "scene0001_00",
        "producer": file_record(Path(capability.__file__).resolve()),
        "input_authority": {
            "accepted_v2": {"path": "/not-opened.pt", "sha256": "a" * 64},
            "capability_bank": {"path": "/not-opened.npz", "sha256": "b" * 64},
            "factorized_field_checkpoint_sha256": "c" * 64,
            "primitive_row_authority_sha256": "d" * 64,
        },
        "pooling_contract": {
            "primitive_normalization": "explicit_l2",
            "aggregation": "uniform_mean_over_unpadded_region_tokens",
            "direction": "l2_normalized_mean",
            "concentration": "l2_norm_of_mean_primitive_unit_directions",
            "storage": "float16_direction_float32_concentration",
        },
        "source_access": capability.source_access(),
    }
    payload = {
        **identity,
        "content_authority_sha256": canonical_json_sha256(identity),
        "region_fingerprints": ["a", "b", "c"],
        "canonical_region_indices": torch.arange(3, dtype=torch.int64),
        "region_rows": torch.arange(3, dtype=torch.int64)[:, None],
        "token_mask": torch.ones(3, 1, dtype=torch.bool),
        "appearance_direction": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float16
        ),
        "boundary_direction": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float16
        ),
        "appearance_concentration": torch.tensor([0.9, 0.8, 0.7]),
        "boundary_concentration": torch.tensor([0.6, 0.5, 0.4]),
        "channel_sha256": {},
        "audit": {},
    }
    payload["channel_sha256"] = {
        name: tensor_sha256(payload[name]) for name in capability.CHANNEL_NAMES
    }
    capability.validate_region_capability_descriptor_authority(payload)
    return payload


def test_combine_appends_six_capability_channels_in_frozen_order() -> None:
    combined = combine_feature_authorities(v1=_v1_feature(), descriptor=_descriptor())
    assert combined.shape == (3, len(PAIR_FEATURE_NAMES))
    assert torch.equal(combined[:, : len(V1_PAIR_FEATURE_NAMES)], torch.zeros(3, 15))
    torch.testing.assert_close(
        combined[:, -6:],
        torch.tensor(
            [
                [1.0, 0.0, 0.8, 0.5, 0.1, 0.1],
                [0.0, -1.0, 0.7, 0.4, 0.2, 0.2],
                [0.0, 0.0, 0.7, 0.4, 0.1, 0.1],
            ]
        ),
        atol=1e-6,
        rtol=0,
    )


def test_combine_rejects_capability_axis_drift() -> None:
    descriptor = _descriptor()
    descriptor["region_fingerprints"] = ["b", "a", "c"]
    with pytest.raises(ValueError, match="axes"):
        combine_feature_authorities(v1=_v1_feature(), descriptor=descriptor)
