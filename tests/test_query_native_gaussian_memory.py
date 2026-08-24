import torch

from radio_gs.interfaces.query_packet import QueryPacket
from radio_gs.models.query_native_gaussian_memory import (
    AnchorConditionedExtentDecoder,
    AnchorPacket,
    CounterfactualSelectiveRiskEstimator,
    FixedCosineQueryProjection,
    GaussianGeometry,
    LowRankSceneCanonicalizer,
    ModalityQueryAdapter,
    QueryNativeGaussianPosteriorDecoder,
    QueryPairEligibilityGate,
    QuerySetCategoricalDecoder,
    QuerySetEligibilityGate,
    TextAnchorIdentityAdapter,
    compile_peak_local_anchor_packet,
)
from radio_gs.scripts.train_evaluate_query_native_membership_decoder import (
    _cross_view_episodes,
    _mutually_exclusive_negative_supports,
    _proposal_pairs,
)
from radio_gs.scripts.train_scannet_query_native_shared_decoder import _loss as _categorical_loss
from radio_gs.scripts.evaluate_scannet_constrained_top2_source_gate import constrained_top2
from radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder import (
    _blend_score,
    _calibrate_selective_blend,
)


def test_cross_view_episode_uses_confirmed_target_and_explicit_negatives_only() -> None:
    authority = {
        "proposal_views": torch.tensor([0, 1, 1]),
        "edge_left": torch.tensor([0, 0]),
        "edge_right": torch.tensor([1, 2]),
        "edge_label": torch.tensor([1, 0]),
    }
    support = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])]
    query, target, negatives = _cross_view_episodes(
        authority, authority["proposal_views"], support
    )
    assert list(zip(query.tolist(), target.tolist())) == [(0, 1), (1, 0)]
    assert torch.equal(negatives[0], torch.tensor([4, 5]))
    assert negatives[1].numel() == 0


def test_selective_text_blend_contains_exact_primitive_fallback() -> None:
    truth = torch.tensor([1.0, 1.0, 0.0, 0.0])
    posterior = [[(torch.tensor([0.2, 0.1, 0.9, 0.8]), truth)]]
    primitive = [[(torch.tensor([0.8, 0.7, 0.2, 0.1]), truth)]]
    calibration = _calibrate_selective_blend(
        posterior, primitive, 0.5, 0.5, 3, 0.0,
    )
    assert calibration["weight"] == 0.0
    signed = _blend_score(
        posterior[0][0][0], primitive[0][0][0], calibration, 0.5, 0.5,
    )
    assert torch.equal(signed >= 0, primitive[0][0][0] >= 0.5)


def test_peak_local_anchor_packet_expands_to_fixed_k_without_distant_instance() -> None:
    identity = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.99])
    xyz = torch.tensor([[0.0, 0.0, float(index) * 0.01] for index in range(6)] + [[9.0, 0.0, 0.0]])
    packet = compile_peak_local_anchor_packet(identity, xyz, topk=4, radius_fraction=1e-5)
    packet.validate(identity.numel())
    assert packet.rows.numel() == 4
    assert 6 not in packet.rows.tolist()


def test_anchor_conditioned_extent_has_exact_zero_authority_replay() -> None:
    torch.manual_seed(9)
    decoder = AnchorConditionedExtentDecoder(latent_dim=4, reliability_dim=2, key_dim=3, hidden_dim=5)
    latent, reliability, identity = torch.randn(8, 4), torch.randn(8, 2), torch.randn(8)
    xyz = torch.randn(8, 3)
    packet = AnchorPacket(torch.tensor([0, 1]), identity[:2], 0, 1.0)
    logits = decoder(
        latent, reliability, identity, packet, GaussianGeometry(xyz), torch.zeros(8),
    )
    assert torch.equal(logits, identity)


def test_text_anchor_adapter_is_low_rank_normalized_and_identity_initialized() -> None:
    value=torch.randn(4,13)
    adapter=TextAnchorIdentityAdapter(embedding_dim=13,rank=3)
    output=adapter(value)
    torch.testing.assert_close(output,torch.nn.functional.normalize(value.float(),dim=-1))
    torch.testing.assert_close(output.norm(dim=1),torch.ones(4))
    assert adapter.down.weight.numel()+adapter.up.weight.numel()==2*13*3


def test_scannet_constrained_top2_changes_only_two_authorized_coordinates() -> None:
    baseline=torch.tensor([[.8,.7,.1,.0],[.1,.2,.9,.3]])
    candidate=torch.tensor([[.2,.9,.8,.0],[.0,.1,.8,.2]])
    output=constrained_top2(baseline,candidate,1.0)
    assert torch.equal(output[0,2:],baseline[0,2:])
    assert torch.equal(output[1],baseline[1])
    torch.testing.assert_close(output[0,1]-output[0,0],candidate[0,1]-candidate[0,0])


def test_query_set_decoder_is_permutation_equivariant_and_cardinality_free() -> None:
    torch.manual_seed(3)
    model = QuerySetCategoricalDecoder(
        latent_dim=4, reliability_dim=2, query_dim=6, hidden_dim=5, pair_hidden_dim=4
    )
    latent, reliability = torch.randn(7, 4), torch.randn(7, 2)
    query, baseline = torch.randn(5, 6), torch.randn(7, 5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    direct = model(latent, reliability, query, baseline)
    permuted = model(latent, reliability, query[permutation], baseline[:, permutation])
    assert torch.allclose(permuted, direct[:, permutation], atol=1e-6)
    assert model(latent, reliability, query[:3], baseline[:, :3]).shape == (7, 3)


def test_factorized_query_set_decoder_remains_permutation_equivariant() -> None:
    torch.manual_seed(4)
    model = QuerySetCategoricalDecoder(
        latent_dim=4, reliability_dim=2, query_dim=6, hidden_dim=5,
        pair_hidden_dim=4, factorized_identity_competition=True,
    )
    latent, reliability = torch.randn(7, 4), torch.randn(7, 2)
    query, baseline = torch.randn(5, 6), torch.randn(7, 5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    direct = model(latent, reliability, query, baseline)
    permuted = model(latent, reliability, query[permutation], baseline[:, permutation])
    assert torch.allclose(permuted, direct[:, permutation], atol=1e-6)


def test_decision_preserving_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(6)
    model = QuerySetCategoricalDecoder(
        latent_dim=4, reliability_dim=2, query_dim=6, hidden_dim=5,
        pair_hidden_dim=4, factorized_identity_competition=True,
    )
    baseline = torch.randn(8, 3)
    target = baseline.clone()
    target[:4] = target[:4].roll(1, dims=1)
    data = {
        "latent": torch.randn(8, 4),
        "reliability": torch.randn(8, 2),
        "baseline": [baseline],
        "target": [target],
        "changed": [(target - baseline).abs().amax(1) > 0],
        "decision_changed": [target.argmax(1) != baseline.argmax(1)],
        "query_holdout": [torch.zeros(3, dtype=torch.bool)],
        "significance": torch.linspace(0.1, 1.0, 8),
    }
    loss = _categorical_loss(
        model, None, data, 0, 0, torch.arange(8), torch.randn(3, 6),
        3.0, 0.25, True, 0.25, 0.25, 1.0, torch.device("cpu"),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(value.grad is not None for value in model.parameters())


def test_query_set_gate_is_permutation_invariant() -> None:
    torch.manual_seed(5)
    gate = QuerySetEligibilityGate(latent_dim=4, reliability_dim=2, query_dim=6, hidden_dim=5)
    latent, reliability = torch.randn(7, 4), torch.randn(7, 2)
    query, baseline = torch.randn(5, 6), torch.randn(7, 5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(
        gate(latent, reliability, query, baseline),
        gate(latent, reliability, query[permutation], baseline[:, permutation]), atol=1e-6,
    )


def test_query_pair_gate_is_permutation_equivariant() -> None:
    torch.manual_seed(6)
    gate = QueryPairEligibilityGate(latent_dim=4, reliability_dim=2, query_dim=6, hidden_dim=5)
    latent, reliability = torch.randn(7, 4), torch.randn(7, 2)
    query, baseline = torch.randn(5, 6), torch.randn(7, 5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(
        gate(latent, reliability, query, baseline)[:, permutation],
        gate(latent, reliability, query[permutation], baseline[:, permutation]), atol=1e-6,
    )


def test_query_packet_validation_and_prompt_seed_authority() -> None:
    torch.manual_seed(7)
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=4, reliability_dim=2, query_dim=3, hidden_dim=5
    )
    seed = torch.tensor([float("nan"), 1.0, 0.0, float("nan")])
    packet = QueryPacket(torch.randn(2, 3), "prompt", seed_probability=seed)
    logits, identity = decoder(torch.randn(4, 4), torch.randn(4, 2), packet)
    assert logits.shape == identity.shape == (4,)
    assert logits[1] > 8.0
    assert logits[2] < -8.0


def test_modality_adapter_changes_encoder_dimension_only() -> None:
    adapter = ModalityQueryAdapter(input_dim=6, query_dim=3)
    assert adapter(torch.randn(4, 6)).shape == (4, 3)


def test_scene_canonicalizer_is_zero_init_and_not_gaussian_indexed() -> None:
    canonicalizer = LowRankSceneCanonicalizer(num_scenes=3, latent_dim=6, rank=2)
    latent = torch.randn(5, 6)
    assert torch.equal(canonicalizer(latent, 1), latent)
    parameter_count = sum(value.numel() for value in canonicalizer.parameters())
    assert parameter_count == 3 * 4 + 2 * 6 * 2


def test_identity_prior_is_replayed_before_extent_training() -> None:
    torch.manual_seed(11)
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=4, reliability_dim=2, query_dim=3, hidden_dim=5
    )
    prior = torch.tensor([-0.4, 0.2, 0.8, 0.1])
    logits, identity = decoder(
        torch.randn(4, 4), torch.randn(4, 2), QueryPacket(torch.randn(1, 3), "image"),
        identity_prior=prior,
    )
    assert torch.equal(identity, prior)
    assert torch.equal(logits, prior)


def test_identity_uses_bounded_learned_temperature_not_dimension_scaling() -> None:
    torch.manual_seed(13)
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=4, reliability_dim=2, query_dim=8, hidden_dim=5,
        initial_temperature=0.07,
    )
    latent = torch.randn(6, 4)
    reliability = torch.randn(6, 2)
    tokens = torch.randn(1, 8)
    _logits, identity = decoder(latent, reliability, QueryPacket(tokens, "text"))
    with torch.no_grad():
        key = torch.nn.functional.normalize(
            decoder.gaussian_key(decoder.latent_norm(latent.float())), dim=-1
        )
        query = torch.nn.functional.normalize(tokens.float(), dim=-1)
        expected = (key @ query.T).squeeze(1) / decoder.identity_temperature()
    assert torch.allclose(decoder.identity_temperature(), torch.tensor(0.07), atol=1e-6)
    assert torch.allclose(identity, expected, atol=1e-6)
    with torch.no_grad():
        decoder.log_temperature.fill_(10.0)
    assert decoder.identity_temperature() == 0.2
    with torch.no_grad():
        decoder.log_temperature.fill_(-10.0)
    assert decoder.identity_temperature() == 0.02


def test_geometry_decoder_preserves_top_identity_anchors_and_prompt_seeds() -> None:
    torch.manual_seed(17)
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=4, reliability_dim=2, query_dim=3, hidden_dim=5, topk_anchors=2
    )
    with torch.no_grad():
        decoder.extent[-1].weight.normal_()
        decoder.extent[-1].bias.fill_(-3.0)
    prior = torch.tensor([-0.4, 0.2, 0.8, 0.1])
    seed = torch.tensor([float("nan"), 1.0, float("nan"), float("nan")])
    geometry = GaussianGeometry(
        xyz=torch.tensor([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
            [2.0, 0.0, 0.0], [2.1, 0.0, 0.0],
        ]),
        scales=torch.ones(4, 3) * 0.1,
        opacity=torch.tensor([0.8, 0.9, 0.7, 0.6]),
    )
    logits, identity = decoder(
        torch.randn(4, 4), torch.randn(4, 2),
        QueryPacket(torch.randn(1, 3), "prompt", seed_probability=seed),
        identity_prior=prior, geometry=geometry,
    )
    assert logits[2] >= identity[2]
    assert logits[1] > 8.0


def test_geometry_validation_fails_closed_on_bad_row_domain() -> None:
    decoder = QueryNativeGaussianPosteriorDecoder(
        latent_dim=4, reliability_dim=2, query_dim=3, hidden_dim=5
    )
    packet = QueryPacket(torch.randn(1, 3), "image")
    try:
        decoder(
            torch.randn(4, 4), torch.randn(4, 2), packet,
            geometry=GaussianGeometry(torch.randn(3, 3)),
        )
    except ValueError as error:
        assert "xyz differs" in str(error)
    else:
        raise AssertionError("bad geometry row domain was accepted")


def test_membership_training_keeps_unlabeled_visible_rows_unknown() -> None:
    supports = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4])]
    probabilities = [torch.ones(2), torch.ones(2), torch.ones(1)]
    views = torch.tensor([0, 0, 0])
    observed = torch.ones(1, 7, dtype=torch.bool)
    semantic = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1]])
    negative = _mutually_exclusive_negative_supports(
        supports, views, semantic,
        negative_semantic_max=0.2, negative_max_support_iou=0.05,
    )
    rows, target = _proposal_pairs(
        0, supports, probabilities, views, negative,
        8, 8, torch.Generator().manual_seed(1),
    )
    assert set(rows[target == 0].tolist()) == {2, 3}
    assert 4 not in rows.tolist()
    assert 5 not in rows.tolist()
    assert 6 not in rows.tolist()


def test_counterfactual_risk_estimator_has_three_explicit_outcomes() -> None:
    estimator = CounterfactualSelectiveRiskEstimator(
        latent_dim=7, reliability_dim=5, decision_feature_dim=9, hidden_dim=11,
    )
    logits = estimator(torch.randn(4, 7), torch.randn(4, 5), torch.randn(4, 9))
    assert logits.shape == (4, 3)
    try:
        estimator(torch.randn(4, 8), torch.randn(4, 5), torch.randn(4, 9))
    except ValueError as error:
        assert "latent input differs" in str(error)
    else:
        raise AssertionError("bad risk-estimator latent domain was accepted")


def test_fixed_query_projection_is_deterministic_and_normalized() -> None:
    value = torch.randn(5, 17)
    left = FixedCosineQueryProjection(17, 7, seed=3)(value)
    right = FixedCosineQueryProjection(17, 7, seed=3)(value)
    torch.testing.assert_close(left, right)
    torch.testing.assert_close(left.norm(dim=1), torch.ones(5))
