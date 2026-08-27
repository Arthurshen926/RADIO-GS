import torch

from radio_gs.v3.memory.structured_memory import ExtraInstanceCodeOracle, StructuredMemoryHeads
from radio_gs.v3.training.phases import LowRankLatentResidual


def test_heads_derive_views_without_adding_gaussian_state():
    heads = StructuredMemoryHeads()
    latent = torch.randn(7, 512)
    assert heads.visual_view(latent).shape == (7, 256)
    assert heads.instance_view(latent, 0.25).shape == (7, 32)
    assert heads.boundary_view(latent).shape == (7, 16)
    assert heads.orthogonality_loss().ndim == 0


def test_oracle_is_explicitly_not_deployment_eligible():
    oracle = ExtraInstanceCodeOracle(11)
    assert oracle().shape == (11, 16)
    assert oracle.deployment_eligible is False


def test_low_rank_residual_starts_as_identity():
    residual = LowRankLatentResidual(rank=4)
    latent = torch.randn(5, 512)
    torch.testing.assert_close(residual(latent), latent)
