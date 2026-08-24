import torch

from radio_gs.interfaces.query_packet import QueryPacket
from radio_gs.models.query_native_gaussian_memory import (
    ModalityQueryAdapter,
    QueryNativeGaussianPosteriorDecoder,
    QuerySetCategoricalDecoder,
    QuerySetEligibilityGate,
)


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
