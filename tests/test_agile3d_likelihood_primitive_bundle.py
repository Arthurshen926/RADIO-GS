from __future__ import annotations

import numpy as np
import pytest
import torch

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_training_dataset import (
    _load_primitive_bundle,
)
from radio_gs.benchmarks.agile3d_scannet40.materialize_likelihood_primitive_bundle import (
    BUNDLE_SCHEMA,
    TENSOR_KEYS,
    _tensor_sha256,
    build_bundle_payload,
    geometry_candidate_mappings,
)


def _synthetic_payload() -> dict[str, object]:
    primitive_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    point_xyz = torch.tensor(
        [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.9, 0.0]]
    )
    candidates, primitive_to_point = geometry_candidate_mappings(
        primitive_xyz, point_xyz, candidate_k=2
    )
    return build_bundle_payload(
        scene_id="scene0000_00",
        primitive_xyz=primitive_xyz,
        primitive_covariance=torch.eye(3).repeat(3, 1, 1) * 0.01,
        primitive_opacity=torch.tensor([0.8, 0.9, 0.7]),
        appearance=torch.arange(12, dtype=torch.float16).reshape(3, 4),
        boundary=torch.arange(6, dtype=torch.float16).reshape(3, 2),
        prior_probability=torch.full((3,), 0.5, dtype=torch.float16),
        coverage=torch.tensor([0.5, 0.6, 0.7], dtype=torch.float16),
        reliability=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float16),
        global_rows=torch.tensor([2, 7, 9], dtype=torch.int32),
        official_point_xyz=point_xyz,
        point_candidate_indices=candidates,
        primitive_to_point_index=primitive_to_point,
        provenance={"source_assets": {}, "capability_source": "synthetic"},
    )


def test_bundle_contains_gaussian_capabilities_and_geometry_mappings() -> None:
    payload = _synthetic_payload()
    assert payload["artifact_type"] == BUNDLE_SCHEMA
    assert set(payload["tensor_records"]) == set(TENSOR_KEYS)
    assert payload["point_candidate_indices"].tolist() == [[0, 1], [1, 0], [2, 0]]
    assert payload["primitive_to_point_index"].tolist() == [0, 1, 2]
    assert payload["safety"] == {
        "query_independent": True,
        "object_id_used": False,
        "clicks_opened": False,
        "gt_labels_opened": False,
        "ply_label_property_opened": False,
        "test_labels_opened": False,
        "target_masks_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "point_as_primitive_used": False,
        "cuda_used": False,
    }
    for forbidden in ("object_id", "clicks", "labels", "point_target"):
        assert forbidden not in payload


def test_tensor_sha_is_layout_independent_for_equal_logical_rows() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    noncontiguous = value.t().contiguous().t()
    assert not noncontiguous.is_contiguous()
    assert _tensor_sha256(value, chunk_rows=2) == _tensor_sha256(
        noncontiguous, chunk_rows=3
    )


def test_bundle_rejects_query_or_label_provenance() -> None:
    payload = _synthetic_payload()
    kwargs = {key: payload[key] for key in TENSOR_KEYS}
    with pytest.raises(ValueError, match="query/object/label"):
        build_bundle_payload(
            scene_id="scene0000_00",
            **kwargs,
            provenance={"labels": [1, 2, 3]},
        )


def test_bundle_is_compatible_with_training_builder(tmp_path) -> None:
    payload = _synthetic_payload()
    path = tmp_path / "scene0000_00.pt"
    torch.save(payload, path)
    bundle, points, adapter = _load_primitive_bundle(
        path,
        point_count=3,
        point_xyz_world=np.asarray(payload["official_point_xyz"]),
    )
    assert adapter == "canonical_primitive_bundle_v1"
    assert bundle["primitive_xyz"].shape == (3, 3)
    assert bundle["primitive_covariance"].shape == (3, 3, 3)
    assert points.shape == (3, 3)


def test_bundle_rejects_non_neutral_query_prior() -> None:
    payload = _synthetic_payload()
    kwargs = {key: payload[key] for key in TENSOR_KEYS}
    kwargs["prior_probability"] = torch.tensor([0.2, 0.5, 0.5])
    with pytest.raises(ValueError, match="neutral 0.5 prior"):
        build_bundle_payload(
            scene_id="scene0000_00",
            **kwargs,
            provenance={"source_assets": {}},
        )
