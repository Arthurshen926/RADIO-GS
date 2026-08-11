from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.data.scannet_source_region_semantics import (
    MESH_NEAREST_MAX_DISTANCE_METERS,
    build_source_region_semantic_sidecar,
    build_development_region_semantic_sidecar,
    official_vertex_nyu40_labels,
    validate_source_region_semantic_sidecar,
    validate_development_region_semantic_sidecar,
)


def _lineage(tmp_path: Path, *, development: bool = False) -> dict[str, Path]:
    result = {}
    for name in (
        "accepted_region_authority",
        "factorized_field_authority",
        "official_mesh",
        "official_segmentation",
        "official_aggregation",
        "official_label_tsv",
        "official_train_split",
    ):
        path = tmp_path / name
        path.write_bytes(f"frozen-{name}".encode())
        result[name] = path
    if development:
        path = tmp_path / "source_gate_receipt"
        path.write_bytes(b"sealed-source-gates")
        result["source_gate_receipt"] = path
    return result


def test_official_segments_and_aggregation_produce_nyu40_vertex_labels() -> None:
    labels, audit = official_vertex_nyu40_labels(
        scene_id="scene0001_00",
        vertex_count=4,
        segmentation={
            "sceneId": "scene0001_00",
            "segIndices": [10, 10, 11, 12],
        },
        aggregation={
            "sceneId": "scannet.scene0001_00",
            "segGroups": [
                {"label": "wall", "segments": [10]},
                {"label": "chair", "segments": [11]},
            ],
        },
        raw_to_nyu40={"wall": 1, "chair": 5},
    )
    np.testing.assert_array_equal(labels, np.array([1, 1, 5, 0]))
    assert audit["annotated_mesh_vertex_count"] == 3


def test_region_sidecar_retains_mixed_soft_distribution(tmp_path: Path) -> None:
    geometry = {"num_gaussians": 4, "xyz_sha256": "a" * 64}
    accepted = {
        "scene_id": "scene0001_00",
        "geometry_fingerprint": geometry,
        "region_rows": torch.tensor([[0, 1], [2, 3], [1, 2]]),
        "token_mask": torch.ones((3, 2), dtype=torch.bool),
        "canonical_region_indices": torch.tensor([3, 7, 9]),
        "region_fingerprints": ["a", "b", "c"],
    }
    xyz = torch.tensor(
        [
            [0.000, 0.0, 0.0],
            [0.010, 0.0, 0.0],
            [1.000, 0.0, 0.0],
            [1.010, 0.0, 0.0],
        ]
    )
    factorized = {
        "xyz": xyz,
        "geometry_fingerprint": geometry,
        "factorized_radio": {"valid": torch.ones(4, dtype=torch.bool)},
    }
    sidecar = validate_source_region_semantic_sidecar(
        build_source_region_semantic_sidecar(
            scene_id="scene0001_00",
            accepted_region_payload=accepted,
            factorized_field_payload=factorized,
            official_mesh_xyz=xyz.numpy(),
            official_vertex_labels=np.array([1, 5, 2, 2]),
            lineage_paths=_lineage(tmp_path),
        )
    )
    torch.testing.assert_close(
        sidecar["nyu40_class_distribution"][0, [1, 5]],
        torch.tensor([0.5, 0.5]),
    )
    assert sidecar["semantic_purity"][0] == 0.5
    assert sidecar["statistics"]["mixed_valid_region_count"] == 2
    assert sidecar["source_access"]["agile3d_instance_ids_opened"] is False
    assert sidecar["statistics"]["member_nearest_distance_p95_meters"] < (
        MESH_NEAREST_MAX_DISTANCE_METERS
    )


def test_source_sidecar_rejects_development_scene_before_opening_labels(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="semantic cohort"):
        build_source_region_semantic_sidecar(
            scene_id="scene0003_00",
            accepted_region_payload={},
            factorized_field_payload={},
            official_mesh_xyz=np.zeros((1, 3), dtype=np.float32),
            official_vertex_labels=np.zeros(1, dtype=np.int64),
            lineage_paths=_lineage(tmp_path),
        )


def test_development_sidecar_is_separate_and_forbids_parameter_callback(
    tmp_path: Path,
) -> None:
    geometry = {"num_gaussians": 2, "xyz_sha256": "b" * 64}
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    payload = validate_development_region_semantic_sidecar(
        build_development_region_semantic_sidecar(
            scene_id="scene0003_00",
            accepted_region_payload={
                "scene_id": "scene0003_00",
                "geometry_fingerprint": geometry,
                "region_rows": torch.tensor([[0], [1]]),
                "token_mask": torch.ones((2, 1), dtype=torch.bool),
                "canonical_region_indices": torch.tensor([0, 1]),
                "region_fingerprints": ["d0", "d1"],
            },
            factorized_field_payload={
                "xyz": xyz,
                "geometry_fingerprint": geometry,
                "factorized_radio": {"valid": torch.ones(2, dtype=torch.bool)},
            },
            official_mesh_xyz=xyz.numpy(),
            official_vertex_labels=np.array([1, 5]),
            lineage_paths=_lineage(tmp_path, development=True),
        )
    )
    assert payload["partition"] == "development"
    assert payload["source_access"]["development_semantic_labels_opened"] is True
    assert payload["source_access"]["parameter_callback_allowed"] is False
