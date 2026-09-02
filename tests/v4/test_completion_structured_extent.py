from __future__ import annotations

import inspect

import pytest
import torch
from torch.nn import functional as F

from radio_gs.v4.completion import (
    EDGE_FEATURE_DIMENSION,
    STRUCTURED_EXTENT_ITERATION_COUNT,
    STRUCTURED_EXTENT_MODES,
    TokenConditionedStructuredExtent,
)


def _source_facts(source_visible: torch.Tensor):
    source_visible = torch.as_tensor(source_visible, dtype=torch.bool)
    element_count = int(source_visible.numel())
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
    if element_count <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    forward = torch.stack(
        (torch.arange(element_count - 1), torch.arange(1, element_count))
    )
    return torch.cat((forward, forward.flip(0)), dim=1)


def _two_token_inputs(*, unary_requires_grad: bool = False):
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False, False, False, True])
    )
    unary = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.55, 0.10, 0.35],
            [0.20, 0.20, 0.60],
            [0.10, 0.55, 0.35],
            [0.0, 1.0, 0.0],
        ],
        requires_grad=unary_requires_grad,
    )
    observed_positive = torch.zeros(5, 2, dtype=torch.bool)
    observed_positive[0, 0] = True
    observed_positive[4, 1] = True
    clamp_mask = torch.tensor([True, False, True, False, True])
    clamp = torch.zeros_like(unary.detach())
    clamp[0, 0] = 1
    clamp[2, -1] = 1
    clamp[4, 1] = 1
    return {
        "unary_probabilities": unary,
        "edge_index": _chain_edges(5),
        "centres": centres,
        "normals": normals,
        "local_features": features,
        "source_visible": visible,
        "observed_positive": observed_positive,
        "clamp_mask": clamp_mask,
        "clamp_probabilities": clamp,
        "voxel_size": 0.04,
        "completion_confidence_cap": 0.95,
    }


def test_forward_is_fixed_two_step_simplex_and_exact_observed_clamp():
    torch.manual_seed(5)
    inputs = _two_token_inputs()
    model = TokenConditionedStructuredExtent(
        embedding_dimension=12,
        edge_hidden_dimension=16,
        edge_chunk_size=2,
    )
    output = model(**inputs)
    assert STRUCTURED_EXTENT_ITERATION_COUNT == model.iteration_count == 2
    assert len(output.step_probabilities) == 2
    assert len(output.step_realized_full_mass) == 2
    assert len(output.step_dual_biases) == 2
    assert output.probabilities.shape == output.log_probabilities.shape == (5, 3)
    assert output.token_context.shape == (2, 12)
    assert output.mass_context.shape == (2, 12)
    assert output.node_token_affinity.shape == (5, 2)
    assert output.edge_features.shape == (8, EDGE_FEATURE_DIMENSION)
    assert output.base_edge_logits.shape == (8,)
    assert output.token_edge_logits.shape == (8, 2)
    assert output.predicted_full_mass.shape == (2,)
    assert output.predicted_log_full_mass.shape == (2,)
    assert output.predicted_dual_posterior_mass.shape == (2,)
    assert output.predicted_completed_membership_mass.shape == (2,)
    assert output.realized_full_mass.shape == (2,)
    assert output.realized_posterior_mass.shape == (2,)
    torch.testing.assert_close(
        output.probabilities.sum(-1), torch.ones(5), rtol=0, atol=1e-6
    )
    mask = inputs["clamp_mask"]
    assert torch.equal(
        output.probabilities[mask], inputs["clamp_probabilities"][mask]
    )
    assert float(output.clamp_max_error.detach()) == 0.0
    assert bool((output.predicted_full_mass >= torch.tensor([1.0, 1.0])).all())
    assert torch.isfinite(output.log_probabilities).all()
    without_edge_output = model(**inputs, return_token_edge_logits=False)
    assert without_edge_output.token_edge_logits is None
    assert without_edge_output.edge_features.shape == (8, EDGE_FEATURE_DIMENSION)
    torch.testing.assert_close(
        without_edge_output.probabilities,
        output.probabilities,
        rtol=2e-6,
        atol=2e-7,
    )
    # The trainable residual starts safely near the frozen pointwise identity;
    # exact observations remain bitwise clamped above.
    unknown = ~inputs["clamp_mask"]
    assert float(
        (
            output.probabilities[unknown]
            - inputs["unary_probabilities"][unknown]
        )
        .abs()
        .max()
        .detach()
    ) < 0.002


def test_token_permutation_is_an_identity_equivariance_not_an_index_code():
    torch.manual_seed(11)
    inputs = _two_token_inputs()
    model = TokenConditionedStructuredExtent(
        embedding_dimension=10, edge_hidden_dimension=13, dropout=0
    ).eval()
    original = model(**inputs)
    permutation = torch.tensor([1, 0])
    categorical_permutation = torch.tensor([1, 0, 2])
    permuted_inputs = dict(inputs)
    permuted_inputs["unary_probabilities"] = inputs["unary_probabilities"][
        :, categorical_permutation
    ]
    permuted_inputs["observed_positive"] = inputs["observed_positive"][
        :, permutation
    ]
    permuted_inputs["clamp_probabilities"] = inputs["clamp_probabilities"][
        :, categorical_permutation
    ]
    permuted = model(**permuted_inputs)
    torch.testing.assert_close(
        permuted.probabilities,
        original.probabilities[:, categorical_permutation],
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        permuted.token_context, original.token_context[permutation]
    )
    torch.testing.assert_close(
        permuted.mass_context, original.mass_context[permutation]
    )
    torch.testing.assert_close(
        permuted.node_token_affinity,
        original.node_token_affinity[:, permutation],
    )
    torch.testing.assert_close(
        permuted.token_edge_logits,
        original.token_edge_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.predicted_full_mass, original.predicted_full_mass[permutation]
    )
    torch.testing.assert_close(
        permuted.predicted_completed_membership_mass,
        original.predicted_completed_membership_mass[permutation],
    )


def test_token_conditioned_symmetric_edge_score_and_shared_edge_control():
    model = TokenConditionedStructuredExtent(
        embedding_dimension=8,
        edge_hidden_dimension=8,
        mode="full",
        edge_chunk_size=1,
    )
    for parameter in model.edge_network.parameters():
        torch.nn.init.zeros_(parameter)
    # Zero base with positive agreement/disagreement coefficients.
    edge_features = torch.zeros(4, EDGE_FEATURE_DIMENSION)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    affinity = torch.tensor([[2.0, -2.0], [1.5, -1.0], [-2.0, 2.0]])
    base, token = model.score_token_edges(edge_features, edge_index, affinity)
    assert base.shape == (4,)
    assert token.shape == (4, 2)
    # Reversing an edge cannot change a symmetric token-conditioned score.
    torch.testing.assert_close(token[0], token[1], rtol=0, atol=0)
    torch.testing.assert_close(token[2], token[3], rtol=0, atol=0)
    # The same physical edge has different compatibility for different tokens.
    assert float(token[0, 0].detach()) != float(token[0, 1].detach())
    selected_base, selected_token = model.score_token_edges(
        edge_features,
        edge_index,
        affinity,
        edge_ids=torch.tensor([0, 2, 3]),
        token_ids=torch.tensor([1, 0, 1]),
    )
    torch.testing.assert_close(selected_base, base[torch.tensor([0, 2, 3])])
    torch.testing.assert_close(
        selected_token, token[torch.tensor([0, 2, 3]), torch.tensor([1, 0, 1])]
    )

    shared = TokenConditionedStructuredExtent(
        embedding_dimension=8,
        edge_hidden_dimension=8,
        mode="shared_edge_plus_mass",
        edge_chunk_size=3,
    )
    shared.load_state_dict(model.state_dict())
    shared_base, shared_token = shared.score_token_edges(
        edge_features, edge_index, affinity
    )
    torch.testing.assert_close(shared_base, base)
    torch.testing.assert_close(
        shared_token, base[:, None].expand_as(shared_token), rtol=0, atol=0
    )


def test_observed_one_hot_retains_strong_wrong_token_negative_affinity():
    torch.manual_seed(23)
    inputs = _two_token_inputs()
    model = TokenConditionedStructuredExtent(
        embedding_dimension=12, edge_hidden_dimension=16, dropout=0
    ).eval()
    output = model(**inputs)
    # Row zero is an immutable positive for token zero: token one is an exact
    # categorical negative even though both its probability and null are zero.
    assert float(output.node_token_affinity[0, 0]) > 10
    assert float(output.node_token_affinity[0, 1]) < -10
    assert float(output.node_token_affinity[4, 1]) > 10
    assert float(output.node_token_affinity[4, 0]) < -10


def test_learned_soft_mass_dual_moves_coverage_and_bypass_is_exact():
    centres, normals, features, visible = _source_facts(
        torch.tensor([True, False, False])
    )
    inputs = {
        "unary_probabilities": torch.tensor(
            [[1.0, 0.0], [0.2, 0.8], [0.2, 0.8]]
        ),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "centres": centres,
        "normals": normals,
        "local_features": features,
        "source_visible": visible,
        "observed_positive": torch.tensor([[True], [False], [False]]),
        "clamp_mask": torch.tensor([True, False, False]),
        "clamp_probabilities": torch.tensor(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ),
        "voxel_size": 0.04,
        "completion_confidence_cap": 0.95,
    }
    low = TokenConditionedStructuredExtent(
        embedding_dimension=8, edge_hidden_dimension=8, mode="full"
    )
    high = TokenConditionedStructuredExtent(
        embedding_dimension=8, edge_hidden_dimension=8, mode="full"
    )
    high.load_state_dict(low.state_dict())
    with torch.no_grad():
        low.mass_head[-1].bias.fill_(-10)
        high.mass_head[-1].bias.fill_(5)
        low.dual_step_parameters.fill_(10)
        high.dual_step_parameters.fill_(10)
        low.transport_step_parameters.zero_()
        high.transport_step_parameters.zero_()
    low_output = low(**inputs)
    high_output = high(**inputs)
    assert float(high_output.predicted_full_mass[0]) > float(
        low_output.predicted_full_mass[0]
    )
    torch.testing.assert_close(
        high_output.predicted_dual_posterior_mass,
        high_output.predicted_full_mass,
    )
    torch.testing.assert_close(
        high_output.predicted_completed_membership_mass,
        torch.tensor([1.0])
        + 0.95 * (high_output.predicted_full_mass - torch.tensor([1.0])),
    )
    assert float(high_output.dual_bias[0]) > float(low_output.dual_bias[0])
    assert float(high_output.realized_full_mass[0]) > float(
        low_output.realized_full_mass[0]
    )
    torch.testing.assert_close(
        high_output.realized_full_mass,
        torch.tensor([1.0])
        + 0.95 * (high_output.realized_posterior_mass - torch.tensor([1.0])),
    )

    bypass = TokenConditionedStructuredExtent(
        embedding_dimension=8,
        edge_hidden_dimension=8,
        mode="token_conditioned_edge_plus_mass_bypass",
    )
    bypass.load_state_dict(high.state_dict())
    bypass_output = bypass(**inputs)
    assert torch.equal(bypass_output.dual_bias, torch.zeros(1))
    assert all(
        torch.equal(value, torch.zeros(1))
        for value in bypass_output.step_dual_biases
    )
    # The capacity-matched control still reports the same learned mass head.
    torch.testing.assert_close(
        bypass_output.predicted_full_mass, high_output.predicted_full_mass
    )


def test_mass_auxiliary_branch_cannot_train_the_bypassed_edge_representation():
    torch.manual_seed(29)
    inputs = _two_token_inputs()
    model = TokenConditionedStructuredExtent(
        embedding_dimension=10,
        edge_hidden_dimension=12,
        dropout=0,
        mode="token_conditioned_edge_plus_mass_bypass",
    )
    output = model(**inputs, return_token_edge_logits=False)
    output.predicted_log_full_mass.square().mean().backward()

    for module in (model.mass_encoder, model.mass_head):
        gradients = [
            parameter.grad for parameter in module.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert sum(float(value.abs().sum()) for value in gradients) > 0
    for module in (
        model.node_encoder,
        model.token_encoder,
        model.node_affinity_projection,
        model.token_affinity_projection,
        model.edge_network,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_chunking_is_numerically_invariant_and_gradients_do_not_reach_unary():
    torch.manual_seed(19)
    inputs = _two_token_inputs(unary_requires_grad=True)
    chunked = TokenConditionedStructuredExtent(
        embedding_dimension=9,
        edge_hidden_dimension=11,
        edge_chunk_size=1,
    )
    whole = TokenConditionedStructuredExtent(
        embedding_dimension=9,
        edge_hidden_dimension=11,
        edge_chunk_size=1024,
    )
    whole.load_state_dict(chunked.state_dict())
    chunked_output = chunked(**inputs)
    whole_output = whole(**inputs)
    torch.testing.assert_close(
        chunked_output.probabilities,
        whole_output.probabilities,
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        chunked_output.token_edge_logits,
        whole_output.token_edge_logits,
        rtol=0,
        atol=0,
    )

    unknown = ~inputs["clamp_mask"]
    categorical = F.nll_loss(
        chunked_output.log_probabilities[unknown], torch.tensor([0, 1])
    )
    edge_objective = chunked_output.token_edge_logits.square().mean()
    mass_objective = F.smooth_l1_loss(
        chunked_output.predicted_log_full_mass,
        torch.log(torch.tensor([2.5, 2.5])),
    )
    (categorical + 0.1 * edge_objective + mass_objective).backward()
    assert inputs["unary_probabilities"].grad is None
    required_modules = (
        chunked.node_encoder,
        chunked.token_encoder,
        chunked.mass_encoder,
        chunked.edge_network,
        chunked.mass_head,
    )
    for module in required_modules:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert chunked.transport_step_parameters.grad is not None
    assert chunked.dual_step_parameters.grad is not None
    assert torch.isfinite(chunked.transport_step_parameters.grad).all()
    assert torch.isfinite(chunked.dual_step_parameters.grad).all()


def test_contract_exposes_three_causal_modes_and_no_forbidden_input_or_mechanism():
    assert STRUCTURED_EXTENT_MODES == (
        "full",
        "shared_edge_plus_mass",
        "token_conditioned_edge_plus_mass_bypass",
    )
    forward_parameters = set(
        inspect.signature(TokenConditionedStructuredExtent.forward).parameters
    )
    assert not forward_parameters & {
        "labels",
        "target_membership",
        "target_rgb",
        "heldout_rgb",
        "query",
        "text",
        "radius",
        "threshold",
        "root_cap",
    }
    for mode in STRUCTURED_EXTENT_MODES:
        receipt = TokenConditionedStructuredExtent(mode=mode).architecture_receipt()
        assert receipt["iteration_count"] == 2
        assert receipt["target_membership_input"] is False
        assert receipt["heldout_rgb_input"] is False
        assert receipt["external_query_input"] is False
        assert receipt["hard_threshold"] is False
        assert receipt["hard_radius_or_envelope"] is False
        assert receipt["connected_components"] is False
        assert receipt["token_or_root_cap"] is False
        assert receipt["v3_dependency"] is False
    with pytest.raises(ValueError, match="unsupported"):
        TokenConditionedStructuredExtent(mode="threshold_sweep")
    with pytest.raises(ValueError, match="sealed F71"):
        TokenConditionedStructuredExtent(feature_dimension=70)
    with pytest.raises(ValueError, match="edge_chunk_size"):
        TokenConditionedStructuredExtent(edge_chunk_size=0)
