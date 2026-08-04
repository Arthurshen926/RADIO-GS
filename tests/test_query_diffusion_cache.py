import torch
import hashlib

from radio_gs.interfaces.query_diffusion_cache import (
    build_exact_euclidean_knn,
    load_query_diffusion_knn_cache,
    load_query_diffusion_relation_cache,
    tensor_sha256,
)


def test_official_neighbor_parameter_200_means_201_columns_with_self():
    xyz = torch.arange(0, 603, dtype=torch.float32).reshape(201, 3)
    neighbors = build_exact_euclidean_knn(
        xyz, num_neighbors=200, include_self=True, workers=1
    )
    assert neighbors.shape == (201, 201)
    expected = torch.arange(201, dtype=torch.int32)
    # Every row contains every node; its own row is retained by release semantics.
    assert all(torch.equal(row.sort().values, expected) for row in neighbors)


def test_knn_cache_loader_binds_geometry_rows_and_release_k(tmp_path):
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    rows = torch.tensor([1, 3, 5, 8])
    neighbors = build_exact_euclidean_knn(xyz, num_neighbors=2, workers=1)
    path = tmp_path / "knn.pt"
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "query_conditioned_diffusion_euclidean_knn",
            "neighbor_indices": neighbors,
            "global_rows": rows,
            "num_global_rows": 10,
            "xyz_sha256": tensor_sha256(xyz),
            "metadata": {
                "source_graph_sha256": "abc",
                "official_num_neighbors_parameter": 2,
                "include_self": True,
                "query_independent": True,
                "labels_opened": False,
                "target_masks_opened": False,
                "target_metrics_opened": False,
            },
        },
        path,
    )
    cache = load_query_diffusion_knn_cache(
        path,
        expected_global_rows=rows,
        expected_xyz=xyz,
        expected_source_graph_sha256="abc",
        expected_num_neighbors=2,
    )
    assert cache.num_nodes == 4
    assert cache.effective_k == 3


def test_tensor_sha256_chunked_digest_matches_contiguous_bytes():
    values = torch.arange(35, dtype=torch.float16).reshape(5, 7)
    expected = hashlib.sha256(values.numpy().tobytes(order="C")).hexdigest()
    assert tensor_sha256(values) == expected


def test_relation_cache_loader_binds_rows_geometry_field_and_capability(tmp_path):
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    rows = torch.tensor([2, 5, 9])
    features = torch.arange(12, dtype=torch.float16).reshape(3, 4)
    capability = tmp_path / "capability.pt"
    capability.touch()
    path = tmp_path / "relation.pt"
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "query_conditioned_diffusion_relation_features",
            "features": features,
            "global_rows": rows,
            "num_global_rows": 12,
            "xyz_sha256": tensor_sha256(xyz),
            "metadata": {
                "source_graph_sha256": "graph-digest",
                "field_checkpoint_sha256": "field-digest",
                "source_capability_cache": str(capability.resolve()),
                "output_dimension": 4,
                "query_independent": True,
                "labels_opened": False,
                "target_masks_opened": False,
                "target_metrics_opened": False,
            },
        },
        path,
    )
    cache = load_query_diffusion_relation_cache(
        path,
        expected_global_rows=rows,
        expected_xyz=xyz,
        expected_source_graph_sha256="graph-digest",
        expected_field_checkpoint_sha256="field-digest",
        expected_source_capability_cache=capability,
    )
    assert cache.feature_dimension == 4
    assert cache.features.dtype == torch.float16
    torch.testing.assert_close(cache.features, features)
    try:
        load_query_diffusion_relation_cache(
            path,
            expected_global_rows=rows,
            expected_xyz=xyz,
            expected_source_graph_sha256="graph-digest",
            expected_field_checkpoint_sha256="wrong-field",
            expected_source_capability_cache=capability,
        )
    except ValueError as error:
        assert "canonical-field hash" in str(error)
    else:
        raise AssertionError("field mismatch must fail closed")
