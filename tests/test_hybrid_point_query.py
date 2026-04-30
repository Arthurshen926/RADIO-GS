import torch
import pytest

from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian


def _toy_hybrid_model() -> HybridFeatureGaussian:
    model = HybridFeatureGaussian(
        latent_dim=4,
        hash_output_dim=4,
        fine_dim=4,
        coarse_dim=4,
        output_dim=8,
        num_levels=1,
        features_per_level=2,
        log2_hashmap_size=4,
        base_resolution=2,
        max_resolution=2,
        fine_hidden_dim=4,
        coarse_hidden_dim=4,
        fusion_hidden_dim=8,
    )
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    model._scaling = torch.log(torch.full((3, 3), 0.2))
    model._opacity = torch.full((3, 1), 4.0)
    model._features_dc = torch.zeros(3, 1, 3)
    model._features_rest = torch.empty(0)
    model._latent = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )
    return model.eval()


def test_query_gaussian_points_matches_screen_space_decode():
    model = _toy_hybrid_model()
    indices = torch.tensor([0, 2])

    queried = model.query_gaussian_points(indices)

    latent_map = model.get_latent()[indices].T.reshape(1, model.latent_dim, 2, 1)
    points_norm = model.normalize_world_positions(model.get_xyz()[indices])
    position_map = points_norm.T.reshape(1, 3, 2, 1)
    manual = model.decode_screen_space(latent_map, position_map)
    manual = manual.squeeze(0).squeeze(-1).T

    assert queried.shape == (2, 8)
    assert torch.allclose(queried, manual, atol=1e-6)


def test_query_gaussian_points_can_decode_at_override_positions():
    model = _toy_hybrid_model()
    indices = torch.tensor([0, 2])
    query_points = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )

    queried = model.query_gaussian_points(indices, points_xyz=query_points)

    manual = model._decode_point_features(
        model.get_latent()[indices],
        model.normalize_world_positions(query_points),
    )
    assert queried.shape == (2, 8)
    assert torch.allclose(queried, manual, atol=1e-6)


def test_query_gaussian_points_rejects_mismatched_override_positions():
    model = _toy_hybrid_model()

    with pytest.raises(ValueError, match="points_xyz"):
        model.query_gaussian_points(
            torch.tensor([0, 2]),
            points_xyz=torch.zeros(1, 3),
        )


def test_query_compact_points_k1_matches_exact_gaussian_query():
    model = _toy_hybrid_model()
    points = model.get_xyz()[[0, 1]]

    queried = model.query_compact_points(points, k=1)
    exact = model.query_gaussian_points(torch.tensor([0, 1]))

    assert queried.shape == (2, 8)
    assert torch.allclose(queried, exact, atol=1e-5)


def test_query_compact_points_return_aux_for_standard_fusion_head():
    model = _toy_hybrid_model()
    result = model.query_compact_points(model.get_xyz()[[0]], k=2, return_aux=True)

    assert result["features"].shape == (1, 8)
    assert result["gaussian_indices"].shape == (1, 2)
    assert result["weights"].shape == (1, 2)
    assert result["mahalanobis_dist2"].shape == (1, 2)
    assert result["density"].shape == (1, 2)


def test_query_gaussian_points_with_semantic_adaptor_is_order_and_chunk_independent():
    torch.manual_seed(7)
    model = HybridFeatureGaussian(
        latent_dim=4,
        hash_output_dim=4,
        fine_dim=4,
        coarse_dim=4,
        output_dim=8,
        num_levels=1,
        features_per_level=2,
        log2_hashmap_size=4,
        base_resolution=2,
        max_resolution=2,
        fine_hidden_dim=4,
        coarse_hidden_dim=4,
        fusion_hidden_dim=8,
        decoupled_heads=True,
        use_semantic_adaptor=True,
        semantic_adaptor_hidden_dim=4,
        semantic_adaptor_use_geometry_guidance=True,
    ).eval()
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
    model._scaling = torch.log(torch.full((4, 3), 0.2))
    model._opacity = torch.full((4, 1), 4.0)
    model._features_dc = torch.zeros(4, 1, 3)
    model._features_rest = torch.empty(0)
    model._latent = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    )
    adaptor = model.fusion_head.semantic_adaptor
    assert adaptor is not None
    adaptor.confidence_net = torch.nn.Sequential(
        torch.nn.Conv2d(9, 1, kernel_size=3, padding=1, bias=False)
    )
    with torch.no_grad():
        adaptor.confidence_net[0].weight.zero_()
        adaptor.confidence_net[0].weight[0, 0, 2, 1] = 5.0

    all_at_once = model.query_gaussian_points(torch.tensor([0, 1, 2, 3]))
    chunked = torch.cat(
        [
            model.query_gaussian_points(torch.tensor([0, 1])),
            model.query_gaussian_points(torch.tensor([2, 3])),
        ],
        dim=0,
    )
    order = torch.tensor([2, 0, 3, 1])
    shuffled = model.query_gaussian_points(order)
    unshuffled = torch.empty_like(shuffled)
    unshuffled[order] = shuffled

    assert torch.allclose(all_at_once, chunked, atol=1e-6)
    assert torch.allclose(all_at_once, unshuffled, atol=1e-6)


def test_query_compact_points_uses_gaussian_rotation_for_weights():
    model = _toy_hybrid_model()
    model._xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    model._scaling = torch.log(torch.tensor([[0.1, 1.0, 1.0], [0.1, 1.0, 1.0]], dtype=torch.float32))
    angle = torch.tensor(torch.pi / 2.0)
    model._rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [torch.cos(angle / 2.0), 0.0, 0.0, torch.sin(angle / 2.0)],
        ],
        dtype=torch.float32,
    )
    model._opacity = torch.full((2, 1), 4.0)
    model._latent = torch.nn.Parameter(torch.eye(2, 4, dtype=torch.float32))

    result = model.query_compact_points(torch.tensor([[0.0, 0.2, 0.0]]), k=2, return_aux=True)
    indices = result["gaussian_indices"][0].tolist()
    identity_pos = indices.index(0)
    rotated_pos = indices.index(1)

    assert result["mahalanobis_dist2"][0, identity_pos] < result["mahalanobis_dist2"][0, rotated_pos]
    assert result["weights"][0, identity_pos] > result["weights"][0, rotated_pos]


def test_query_compact_points_can_prune_oversampled_candidates_by_density():
    model = _toy_hybrid_model()
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model._scaling = torch.log(
        torch.tensor(
            [
                [0.01, 0.01, 0.01],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
    )
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    model._opacity = torch.full((2, 1), 4.0)
    model._latent = torch.nn.Parameter(torch.eye(2, 4, dtype=torch.float32))

    point = torch.tensor([[0.0, 0.2, 0.0]], dtype=torch.float32)
    nearest_only = model.query_compact_points(point, k=1, return_aux=True)
    oversampled = model.query_compact_points(
        point,
        k=1,
        candidate_k=2,
        return_aux=True,
    )

    assert nearest_only["gaussian_indices"].tolist() == [[0]]
    assert oversampled["gaussian_indices"].tolist() == [[1]]
    assert oversampled["density"][0, 0] > nearest_only["density"][0, 0]


def test_query_compact_points_handles_empty_input():
    model = _toy_hybrid_model()

    out = model.query_compact_points(torch.empty(0, 3), k=8)
    aux = model.query_compact_points(torch.empty(0, 3), k=8, return_aux=True)

    assert out.shape == (0, 8)
    assert aux["features"].shape == (0, 8)
    assert aux["gaussian_indices"].shape == (0, 0)
