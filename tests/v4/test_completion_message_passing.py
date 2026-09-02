import inspect

import pytest
import torch
from torch.nn import functional as F

from radio_gs.v4.carrier import SurfaceVoxelCarrier
from radio_gs.v4.completion import (
    EDGE_FEATURE_DIMENSION,
    EDGE_FEATURE_LAYOUT,
    EXTENT_GATE_INITIAL_LOGIT,
    EdgeCompatibilityMLP,
    SurfaceMessagePassing,
    build_query_free_edge_features,
    validate_surface_voxel_adjacency,
)


def _source_facts(source_visible: torch.Tensor):
    source_visible = torch.as_tensor(source_visible, dtype=torch.bool)
    element_count = source_visible.numel()
    centres = torch.stack(
        (
            torch.arange(element_count, dtype=torch.float32) * 0.04,
            torch.zeros(element_count),
            torch.ones(element_count),
        ),
        dim=-1,
    )
    normals = torch.tensor([0.0, 0.0, 1.0]).expand(element_count, -1).clone()
    features = torch.zeros(element_count, 71)
    features[:, 3] = source_visible.float()
    for index in torch.where(source_visible)[0].tolist():
        features[index, :3] = torch.tensor(
            [0.1 * (index + 1), -0.05 * index, 0.2]
        )
        features[index, 4 + (index % 64)] = 1.0
    features[:, -3:] = normals
    return centres, normals, features, source_visible


def _chain_edges(element_count: int) -> torch.Tensor:
    forward = torch.stack(
        (torch.arange(element_count - 1), torch.arange(1, element_count))
    )
    return torch.cat((forward, forward.flip(0)), dim=1)


def test_edge_features_are_query_free_auditable_and_reverse_symmetric():
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, True, False])
    )
    edges = _chain_edges(3)
    result = build_query_free_edge_features(
        edges,
        centres,
        normals,
        features,
        visible,
        voxel_size=0.04,
    )
    assert result.shape == (4, EDGE_FEATURE_DIMENSION)
    assert len(EDGE_FEATURE_LAYOUT) == EDGE_FEATURE_DIMENSION == 19
    assert torch.isfinite(result).all()
    # The first/reverse-first and second/reverse-second facts are identical.
    torch.testing.assert_close(result[0], result[2], rtol=0, atol=0)
    torch.testing.assert_close(result[1], result[3], rtol=0, atol=0)
    # One unavailable endpoint disables RGB/RADIO compatibility while retaining
    # explicit availability facts.
    layout = {name: index for index, name in enumerate(EDGE_FEATURE_LAYOUT)}
    one_missing = result[1]
    assert float(one_missing[layout["exactly_one_source_available"]]) == 1.0
    assert float(one_missing[layout["both_source_available"]]) == 0.0
    assert float(one_missing[layout["rgb_cosine_when_both_available"]]) == 0.0
    assert float(one_missing[layout["radio_cosine_when_both_available"]]) == 0.0
    # The feature API has no membership, identity, label, token, or query input.
    parameter_names = set(inspect.signature(build_query_free_edge_features).parameters)
    assert not parameter_names & {
        "labels",
        "token_index",
        "membership",
        "query",
        "target",
    }


def test_two_steps_transport_distribution_two_hops_and_exactly_clamp_observations():
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False, False, True])
    )
    edges = _chain_edges(4)
    # Two object tokens followed by one explicit null category.
    unary = torch.tensor(
        [
            [0.2, 0.2, 0.6],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.3, 0.5],
        ]
    )
    clamp_mask = torch.tensor([True, False, False, True])
    clamp = torch.zeros_like(unary)
    clamp[0, 0] = 1.0  # observed positive for token 0; token 1/null are negative.
    clamp[3, -1] = 1.0  # observed background: exact null, all tokens negative.
    module = SurfaceMessagePassing(step_count=2, edge_hidden_dimension=8, dropout=0)
    for parameter in module.edge_compatibility.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        module.step_logits.zero_()
    output = module(
        unary,
        edges,
        centres,
        normals,
        features,
        visible,
        clamp_mask,
        clamp,
        voxel_size=0.04,
    )
    assert len(output.step_probabilities) == 2
    assert output.edge_logits.shape == (edges.shape[1],)
    torch.testing.assert_close(
        output.edge_weights, torch.full_like(output.edge_weights, 0.5)
    )
    for step in output.step_probabilities:
        torch.testing.assert_close(step.sum(-1), torch.ones(4), rtol=0, atol=1e-6)
        assert bool((step >= 0).all())
        assert torch.equal(step[clamp_mask], clamp[clamp_mask])
    # Node 2 cannot see node 0 in one hop.  Its token-0 mass arrives only in the
    # second bounded propagation step.
    assert float(output.step_probabilities[0][2, 0].detach()) == 0.0
    assert float(output.step_probabilities[1][2, 0].detach()) > 0.0
    assert float(output.step_seed_reachabilities[0][2, 0]) == 0.0
    assert float(output.step_seed_reachabilities[1][2, 0]) > 0.0
    assert output.extent_weights.shape == output.extent_logits.shape == (4, 2)
    assert output.seed_reachability.shape == (4, 2)
    assert torch.isfinite(output.extent_logits).all()
    assert torch.equal(output.probabilities[clamp_mask], clamp[clamp_mask])
    assert torch.isfinite(output.log_probabilities).all()


@pytest.mark.parametrize("step_count", [2, 3])
def test_edge_and_render_losses_are_differentiable_but_frozen_unary_stays_frozen(
    step_count,
):
    torch.manual_seed(7)
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False, True, False])
    )
    edges = _chain_edges(4)
    unary = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ],
        requires_grad=True,
    )
    clamp_mask = torch.tensor([True, False, True, False])
    clamp = torch.zeros_like(unary.detach())
    clamp[0, 0] = 1
    clamp[2, 1] = 1
    module = SurfaceMessagePassing(
        step_count=step_count, edge_hidden_dimension=12, dropout=0
    )
    output = module(
        unary,
        edges,
        centres,
        normals,
        features,
        visible,
        clamp_mask,
        clamp,
        voxel_size=0.04,
    )
    categorical_or_render_loss = F.nll_loss(
        output.log_probabilities[~clamp_mask], torch.tensor([0, 1])
    )
    same_instance_edge_target = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    edge_loss = F.binary_cross_entropy_with_logits(
        output.edge_logits, same_instance_edge_target
    )
    (categorical_or_render_loss + edge_loss).backward()
    assert unary.grad is None
    assert module.step_logits.grad is not None
    assert torch.isfinite(module.step_logits.grad).all()
    edge_gradients = [
        parameter.grad
        for parameter in module.edge_compatibility.parameters()
        if parameter.grad is not None
    ]
    assert edge_gradients
    assert all(torch.isfinite(gradient).all() for gradient in edge_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in edge_gradients) > 0


def test_no_edges_runs_soft_seed_extent_suppression_and_exact_clamps():
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False])
    )
    unary = torch.tensor([[0.4, 0.6], [0.25, 0.75]])
    clamp_mask = torch.tensor([True, False])
    clamp = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    module = SurfaceMessagePassing(step_count=2, edge_hidden_dimension=8)
    output = module(
        unary,
        torch.empty(2, 0, dtype=torch.long),
        centres,
        normals,
        features,
        visible,
        clamp_mask,
        clamp,
        voxel_size=0.04,
    )
    assert output.edge_logits.shape == output.edge_weights.shape == (0,)
    assert output.edge_features.shape == (0, EDGE_FEATURE_DIMENSION)
    torch.testing.assert_close(output.probabilities[0], clamp[0], rtol=0, atol=0)
    assert float(output.probabilities[1, 0]) < float(unary[1, 0])
    assert float(output.probabilities[1, -1]) > float(unary[1, -1])
    assert float(output.seed_reachability[1, 0]) == 0.0
    initial_retention = 1.0 - float(torch.sigmoid(torch.tensor(-3.0)))
    assert initial_retention >= 0.94
    assert float(output.extent_weights[1, 0]) == pytest.approx(initial_retention)
    assert all(
        float(step[1, 0]) >= 0.94 for step in output.step_extent_weights
    )
    assert float(output.probabilities[1, 0] / unary[1, 0]) >= 0.9


def test_adjacency_contract_is_bidirectional_duplicate_free_and_six_neighbour():
    centres, _, _, _ = _source_facts(torch.tensor([True, True, True]))
    validate_surface_voxel_adjacency(
        _chain_edges(3), centres, voxel_size=0.04
    )
    with pytest.raises(ValueError, match="reverse edge"):
        validate_surface_voxel_adjacency(
            torch.tensor([[0], [1]]), centres, voxel_size=0.04
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_surface_voxel_adjacency(
            torch.tensor([[0, 1, 0], [1, 0, 1]]),
            centres,
            voxel_size=0.04,
        )
    diagonal = centres.clone()
    diagonal[1, 1] += 0.04
    with pytest.raises(ValueError, match="six-neighbour"):
        validate_surface_voxel_adjacency(
            torch.tensor([[0, 1], [1, 0]]), diagonal, voxel_size=0.04
        )


def test_adjacency_validation_replays_carrier_cpu_float32_boundary_authority():
    voxel_size = 0.04
    positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)
    # Production meshes contain coordinates one float32 ULP from voxel faces.
    # The validator must consume the exact edge list emitted by the carrier and
    # replay its CPU float32 key convention, even if caller tensors later live
    # on a GPU.
    left = torch.nextafter(torch.tensor(0.08), negative_infinity)
    right = torch.nextafter(torch.tensor(0.12), negative_infinity)
    z = torch.nextafter(torch.tensor(1.0), positive_infinity)
    centres = torch.tensor([[left, 0.0, z], [right, 0.0, z]])
    carrier = SurfaceVoxelCarrier(
        centres,
        voxel_size,
        normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        maximum_splat_radius=1,
        surface_band_voxels=1.0,
        maximum_contributors_per_pixel=2,
    )
    edges = carrier.neighbors().edge_index
    assert edges.shape == (2, 2)
    validate_surface_voxel_adjacency(edges, centres, voxel_size=voxel_size)

    # Exact production regression from scene0050_02: CUDA floor placed the
    # second endpoint in x-key 105 while the carrier-authoritative CPU float32
    # path (and its emitted edge) place it in key 106.
    production_centres = torch.tensor(
        [
            [4.2220001220703125, 0.1663036048412323, 1.096000075340271],
            [4.239999771118164, 0.16545572876930237, 1.0947500467300415],
        ],
        dtype=torch.float32,
    )
    production_carrier = SurfaceVoxelCarrier(
        production_centres,
        voxel_size,
        normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        maximum_splat_radius=1,
        surface_band_voxels=1.0,
        maximum_contributors_per_pixel=2,
    )
    production_edges = production_carrier.neighbors().edge_index
    assert production_edges.shape == (2, 2)
    validate_surface_voxel_adjacency(
        production_edges, production_centres, voxel_size=voxel_size
    )


def test_architecture_receipt_fixes_local_role_and_forbidden_mechanisms():
    receipt = SurfaceMessagePassing(step_count=2).architecture_receipt()
    assert receipt["role"] == "local_surface_boundary_and_observed_seed_support_residual"
    assert receipt["global_identity_and_extent_authority"] == "frozen_K_plus_null_unary"
    assert receipt["target_membership_in_edge_or_extent_features"] is False
    assert receipt["query_in_edge_or_extent_features"] is False
    assert receipt["hard_threshold"] is False
    assert receipt["hard_radius_or_envelope"] is False
    assert receipt["connected_components"] is False
    assert receipt["extent_gate_initial_logit"] == EXTENT_GATE_INITIAL_LOGIT == -3.0
    assert receipt["extent_gate_initial_strength"] < 0.06


def test_contract_rejects_non_f71_or_target_like_implicit_facts():
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False])
    )
    edges = _chain_edges(2)
    with pytest.raises(ValueError, match="sealed F71"):
        build_query_free_edge_features(
            edges,
            centres,
            normals,
            features[:, :-1],
            visible,
            voxel_size=0.04,
        )
    broken_availability = features.clone()
    broken_availability[1, 3] = 1
    with pytest.raises(ValueError, match="availability"):
        build_query_free_edge_features(
            edges,
            centres,
            normals,
            broken_availability,
            visible,
            voxel_size=0.04,
        )
    broken_normal = normals.clone()
    broken_normal[0, 0] = 1
    with pytest.raises(ValueError, match="normal channels"):
        build_query_free_edge_features(
            edges,
            centres,
            broken_normal,
            features,
            visible,
            voxel_size=0.04,
        )


def test_contract_rejects_invalid_probability_and_clamp_states():
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False])
    )
    edges = _chain_edges(2)
    module = SurfaceMessagePassing(step_count=2, edge_hidden_dimension=8)
    common = (
        edges,
        centres,
        normals,
        features,
        visible,
        torch.tensor([True, False]),
    )
    with pytest.raises(ValueError, match="simplex"):
        module(
            torch.tensor([[0.4, 0.4], [0.5, 0.5]]),
            *common,
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            voxel_size=0.04,
        )
    with pytest.raises(ValueError, match=r"exact K\+null one-hot"):
        module(
            torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            *common,
            torch.tensor([[0.5, 0.5], [0.0, 0.0]]),
            voxel_size=0.04,
        )
    with pytest.raises(ValueError, match="step_count"):
        SurfaceMessagePassing(step_count=1)
    with pytest.raises(ValueError, match="sealed F71"):
        SurfaceMessagePassing(feature_dimension=70)
    with pytest.raises(ValueError, match="sealed 19-D"):
        EdgeCompatibilityMLP(input_dimension=18)
