import pytest
import torch

from radio_gs.querying.query_conditioned_diffusion import (
    QueryConditionedDiffusionConfig,
    _undirected_boolean_propagation,
    cap_positive_reference_evidence,
    gate_knn_similarity,
    knn_feature_distances,
    ludvig_release_position_normalize,
    normalize_node_features,
    rbf_knn_feature_similarity,
    rbf_similarity_from_distances,
    run_query_conditioned_diffusion,
    solve_continuous_query_support,
    weighted_logistic_query_compatibility,
)


def _explicit_symmetrized_boolean_step(
    active: torch.Tensor,
    neighbors: torch.Tensor,
    directed_edge_mask: torch.Tensor,
) -> torch.Tensor:
    node_count, neighbor_count = neighbors.shape
    rows = torch.arange(node_count).repeat_interleave(neighbor_count)
    cols = neighbors.reshape(-1)
    retained = directed_edge_mask.reshape(-1).float()
    adjacency = torch.sparse_coo_tensor(
        torch.stack(
            [torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0
        ),
        torch.cat([retained, retained]),
        (node_count, node_count),
    ).coalesce()
    return torch.sparse.mm(adjacency, active.float()[:, None]).squeeze(1) > 0


def test_release_maxpos_cap_uses_argsort_and_integer_positive_fraction():
    evidence = torch.tensor([0.0, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.9, 1.0])
    capped = cap_positive_reference_evidence(
        evidence, max_positive_fraction=0.2
    )
    assert int((capped > 0).sum()) == 2
    torch.testing.assert_close(capped[capped > 0], torch.tensor([0.9, 1.0]))
    # Release ``[:-0]`` removes nothing for fewer than 1/fraction positives.
    unchanged = cap_positive_reference_evidence(
        torch.tensor([0.0, 0.4, 0.8]), max_positive_fraction=0.1
    )
    torch.testing.assert_close(unchanged, torch.tensor([0.0, 0.4, 0.8]))


def test_release_position_normalization_uses_knn_column_position_not_node_degree():
    similarities = torch.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]])
    actual = ludvig_release_position_normalize(similarities, eps=1e-8)
    expected = similarities / (
        similarities.sum(1, keepdim=True).sqrt()
        * similarities.sum(0, keepdim=True).sqrt()
    )
    torch.testing.assert_close(actual, expected)
    # The two K columns have different sums.  A graph-node degree
    # normalization could not produce this released-code denominator.
    assert not torch.allclose(actual[:, 0], similarities[:, 0] / similarities.sum(1))


def test_query_gate_is_exact_endpoint_geometric_mean():
    similarities = torch.ones(3, 2)
    neighbors = torch.tensor([[0, 1], [1, 2], [2, 0]])
    compatibility = torch.tensor([1.0, 0.25, 0.04])
    actual = gate_knn_similarity(similarities, neighbors, compatibility)
    expected = torch.tensor([[1.0, 0.5], [0.25, 0.1], [0.04, 0.2]])
    torch.testing.assert_close(actual, expected)


def test_rbf_median_matches_positive_source_rows_in_release():
    features = normalize_node_features(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    )
    neighbors = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    positive = torch.tensor([True, False, False])
    actual = rbf_knn_feature_similarity(
        features,
        neighbors,
        feature_bandwidth=2.0,
        positive_reference_mask=positive,
    )
    distances = torch.linalg.vector_norm(features[:, None] - features[neighbors], dim=-1)
    median = distances[positive].median()
    expected = torch.exp(-distances.square() / (2.0 * median.square()))
    torch.testing.assert_close(actual, expected)


def test_cached_knn_distances_reproduce_direct_rbf_exactly():
    features = normalize_node_features(
        torch.tensor([[1.0, 0.2], [0.1, 1.0], [-1.0, 0.3], [0.4, -1.0]])
    )
    neighbors = torch.tensor(
        [[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1]]
    )
    positive = torch.tensor([True, False, True, False])
    distances = knn_feature_distances(
        features, neighbors, distance_chunk_size=2
    )
    cached = rbf_similarity_from_distances(
        distances,
        feature_bandwidth=2.0,
        positive_reference_mask=positive,
    )
    direct = rbf_knn_feature_similarity(
        features,
        neighbors,
        feature_bandwidth=2.0,
        positive_reference_mask=positive,
        distance_chunk_size=2,
    )
    torch.testing.assert_close(cached, direct, rtol=0, atol=0)


def test_weighted_logistic_uses_only_signed_reference_rows_and_returns_probability():
    features = normalize_node_features(
        torch.tensor(
            [
                [2.0, 0.0],
                [1.0, 0.1],
                [-2.0, 0.0],
                [-1.0, -0.1],
                [0.5, 0.0],
            ]
        )
    )
    evidence = torch.tensor([1.0, 0.5, -1.0, -0.5, 0.0])
    weights = torch.tensor([2.0, 1.0, 2.0, 1.0, 0.0])
    probability = weighted_logistic_query_compatibility(
        features,
        evidence,
        weights,
        logistic_c=1.0,
        regularizer_bandwidth=1.0,
    )
    assert probability.shape == (5,)
    assert bool(((probability >= 0) & (probability <= 1)).all())
    assert probability[0] > probability[2]
    with pytest.raises(ValueError, match="both signed classes"):
        weighted_logistic_query_compatibility(
            features,
            evidence.clamp_min(0),
            weights,
            logistic_c=1.0,
            fit_population="signed_nonzero",
        )


def test_positive_only_release_fit_includes_all_nodes_and_allows_zero_weight_rows():
    features = normalize_node_features(
        torch.tensor([[2.0, 0.0], [1.0, 0.1], [-2.0, 0.0], [-1.0, -0.1]])
    )
    evidence = torch.tensor([1.0, 0.5, 0.0, 0.0])
    weights = torch.tensor([2.0, 1.0, 0.4, 0.0])
    explicit = weighted_logistic_query_compatibility(
        features,
        evidence,
        weights,
        logistic_c=1.0,
        fit_population="all_nodes_positive_only",
    )
    automatic = weighted_logistic_query_compatibility(
        features,
        evidence,
        weights,
        logistic_c=1.0,
        fit_population="auto_release",
    )
    torch.testing.assert_close(automatic, explicit)
    assert explicit[0] > explicit[2]


def test_rbf_chunking_is_numerically_identical_across_chunk_sizes():
    generator = torch.Generator().manual_seed(7)
    features = normalize_node_features(torch.randn(11, 13, generator=generator))
    neighbors = torch.randint(0, 11, (11, 5), generator=generator)
    mask = torch.arange(11) < 4
    small = rbf_knn_feature_similarity(
        features,
        neighbors,
        feature_bandwidth=2.0,
        positive_reference_mask=mask,
        distance_chunk_size=2,
    )
    large = rbf_knn_feature_similarity(
        features,
        neighbors,
        feature_bandwidth=2.0,
        positive_reference_mask=mask,
        distance_chunk_size=64,
    )
    torch.testing.assert_close(small, large, atol=0, rtol=0)


def test_release_compat_binarization_returns_p_on_seed_reachable_component():
    # Nodes 0-1 form one component and node 2 has only a self edge.  The
    # released postprocessing returns P on the seed-reachable component.
    neighbors = torch.tensor([[0, 1], [1, 0], [2, 2]])
    similarities = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    initial = torch.tensor([1.0, 0.0, 0.0])
    compatibility = torch.tensor([0.9, 0.7, 0.3])
    output = run_query_conditioned_diffusion(
        initial,
        neighbors,
        similarities,
        compatibility,
        config=QueryConditionedDiffusionConfig(
            iterations=3,
            feature_bandwidth=1.0,
            regularizer_bandwidth=1.0,
            edge_binarize_threshold=1e-5,
        ),
    )
    torch.testing.assert_close(output[:, 0], torch.tensor([0.9, 0.7, 0.0]))


def test_implicit_release_reachability_matches_explicit_coo_at_every_step():
    # Deliberately omit some self edges so this covers exact-step propagation,
    # not merely monotone connected-component closure.
    neighbors = torch.tensor(
        [[1, 2], [0, 3], [3, 4], [1, 5], [2, 4], [0, 5]]
    )
    directed = torch.tensor(
        [
            [True, False],
            [True, True],
            [True, False],
            [False, True],
            [True, True],
            [True, False],
        ]
    )
    initial = torch.tensor([True, False, False, False, False, False])
    expected = initial
    for steps in range(1, 7):
        expected = _explicit_symmetrized_boolean_step(expected, neighbors, directed)
        actual = _undirected_boolean_propagation(
            initial,
            neighbors,
            directed,
            iterations=steps,
            row_chunk_size=2,
        ).squeeze(1)
        torch.testing.assert_close(actual, expected)


def test_release_memory_optimization_matches_explicit_sparse_final_output():
    generator = torch.Generator().manual_seed(17)
    node_count, neighbor_count = 13, 4
    neighbors = torch.randint(
        0, node_count, (node_count, neighbor_count), generator=generator
    )
    neighbors[:, 0] = torch.arange(node_count)
    similarities = torch.rand(
        node_count, neighbor_count, generator=generator
    ).clamp_min(0.05)
    compatibility = torch.rand(node_count, generator=generator).clamp_min(0.1)
    initial = torch.zeros(node_count)
    initial[[0, 7]] = torch.tensor([1.0, 0.4])
    config = QueryConditionedDiffusionConfig(
        iterations=5,
        edge_binarize_threshold=0.03,
    )

    gated = gate_knn_similarity(similarities, neighbors, compatibility)
    normalized = ludvig_release_position_normalize(gated, eps=config.eps)
    retained = normalized > float(config.edge_binarize_threshold)
    explicit_active = initial > 0
    for _ in range(config.iterations):
        explicit_active = _explicit_symmetrized_boolean_step(
            explicit_active, neighbors, retained
        )
    expected = explicit_active.float() * compatibility
    actual = run_query_conditioned_diffusion(
        initial,
        neighbors,
        similarities,
        compatibility,
        config=config,
    ).squeeze(1)
    torch.testing.assert_close(actual, expected)


def test_clean_symmetric_kernel_is_distinct_from_release_compatibility_kernel():
    neighbors = torch.tensor([[0, 1], [1, 0], [2, 1]])
    similarities = torch.tensor([[1.0, 0.8], [1.0, 0.8], [1.0, 0.2]])
    initial = torch.tensor([1.0, 0.0, 0.0])
    compatibility = torch.tensor([0.9, 0.7, 0.3])
    release = run_query_conditioned_diffusion(
        initial,
        neighbors,
        similarities,
        compatibility,
        config=QueryConditionedDiffusionConfig(iterations=2),
    )
    clean = run_query_conditioned_diffusion(
        initial,
        neighbors,
        similarities,
        compatibility,
        config=QueryConditionedDiffusionConfig(
            kernel="symmetric_normalized",
            iterations=2,
            edge_binarize_threshold=None,
        ),
    )
    assert not torch.allclose(release, clean)


def test_continuous_convex_solver_locks_reliable_reference_and_fills_unknown():
    # Pad the endpoint rows to a rectangular kNN cache with self edges.
    neighbors = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 1, 2]])
    similarities = torch.ones_like(neighbors, dtype=torch.float32)
    observation_probability = torch.tensor([1.0, 0.5, 0.0])
    observation_confidence = torch.tensor([1.0, 0.0, 1.0])
    compatibility = torch.ones(3)
    output = solve_continuous_query_support(
        observation_probability,
        observation_confidence,
        neighbors,
        similarities,
        compatibility,
        config=QueryConditionedDiffusionConfig(
            kernel="continuous_convex_v2",
            edge_binarize_threshold=None,
            cg_iterations=128,
            cg_tolerance=1e-7,
            solver_row_chunk_size=2,
        ),
    )
    assert output[0].item() == 1.0
    assert output[2].item() == 0.0
    assert output[1].item() == pytest.approx(0.5, abs=0.01)


def test_continuous_query_gate_changes_only_pairwise_completion_not_hard_unary():
    neighbors = torch.tensor([[0, 1, 0], [1, 0, 2], [2, 1, 2]])
    similarities = torch.ones_like(neighbors, dtype=torch.float32)
    probability = torch.tensor([1.0, 0.5, 0.0])
    confidence = torch.tensor([1.0, 0.0, 1.0])
    config = QueryConditionedDiffusionConfig(
        kernel="continuous_convex_v2",
        edge_binarize_threshold=None,
        cg_iterations=128,
        cg_tolerance=1e-7,
        solver_row_chunk_size=2,
    )
    ungated = solve_continuous_query_support(
        probability,
        confidence,
        neighbors,
        similarities,
        torch.ones(3),
        config=config,
    )
    foreground_gated = solve_continuous_query_support(
        probability,
        confidence,
        neighbors,
        similarities,
        torch.tensor([1.0, 1.0, 0.0]),
        config=config,
    )
    assert foreground_gated[0].item() == 1.0
    assert foreground_gated[2].item() == 0.0
    assert foreground_gated[1] > ungated[1] + 0.2


def test_continuous_convex_solver_preserves_soft_unary_amplitude():
    neighbors = torch.arange(3)[:, None]
    output = solve_continuous_query_support(
        torch.tensor([0.8, 0.45, 0.2]),
        torch.ones(3),
        neighbors,
        torch.ones(3, 1),
        torch.tensor([0.9, 0.5, 0.1]),
        config=QueryConditionedDiffusionConfig(
            kernel="continuous_convex_v2",
            edge_binarize_threshold=None,
            cg_iterations=16,
        ),
    )
    # With no non-self relation, the convex readout is the observation itself,
    # unlike release reachability's binary support times P.
    torch.testing.assert_close(output, torch.tensor([0.8, 0.45, 0.2]))


def test_continuous_implicit_pcg_matches_dense_hard_eliminated_energy():
    neighbors = torch.tensor(
        [[0, 1, 2], [1, 0, 2], [2, 1, 3], [3, 2, 1]]
    )
    similarities = torch.tensor(
        [[1.0, 0.8, 0.2], [1.0, 0.7, 0.5],
         [1.0, 0.6, 0.9], [1.0, 0.4, 0.3]]
    )
    compatibility = torch.tensor([0.95, 0.75, 0.55, 0.25])
    probability = torch.tensor([1.0, 0.65, 0.5, 0.0])
    confidence = torch.tensor([1.0, 0.4, 0.0, 1.0])
    config = QueryConditionedDiffusionConfig(
        kernel="continuous_convex_v2",
        edge_binarize_threshold=None,
        laplacian_weight=0.7,
        cg_iterations=256,
        cg_tolerance=1e-8,
        solver_row_chunk_size=2,
    )
    actual = solve_continuous_query_support(
        probability,
        confidence,
        neighbors,
        similarities,
        compatibility,
        config=config,
    )

    node_count = probability.numel()
    rows = torch.arange(node_count)[:, None]
    gated = similarities * torch.sqrt(
        compatibility[:, None] * compatibility[neighbors]
    )
    gated = gated * (neighbors != rows)
    degree = gated.sum(dim=1)
    degree.index_add_(0, neighbors.reshape(-1), gated.reshape(-1))
    conductance = gated / torch.sqrt(
        degree[:, None].clamp_min(config.eps)
        * degree[neighbors].clamp_min(config.eps)
    )
    laplacian = torch.zeros(node_count, node_count)
    for source in range(node_count):
        for position in range(neighbors.shape[1]):
            destination = int(neighbors[source, position])
            weight = conductance[source, position]
            difference = torch.zeros(node_count)
            difference[source] = 1.0
            difference[destination] -= 1.0
            laplacian += weight * torch.outer(difference, difference)

    unary_target = torch.where(confidence > 0, probability, compatibility)
    unary_confidence = confidence + (1.0 - confidence) * config.unobserved_fidelity
    fixed = torch.tensor([True, False, False, True])
    free = ~fixed
    fixed_values = torch.tensor([1.0, 0.0, 0.0, 0.0])
    system = torch.diag(unary_confidence) + config.laplacian_weight * laplacian
    right = unary_confidence * unary_target
    expected = fixed_values.clone()
    expected[free] = torch.linalg.solve(
        system[free][:, free],
        right[free] - system[free][:, fixed] @ fixed_values[fixed],
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
