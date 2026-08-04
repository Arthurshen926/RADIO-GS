import torch

from radio_gs.scripts.build_nvos_evidence_to_support_v2 import (
    diagonal_shrinkage_lda,
)


def test_all_scribble_diagonal_lda_uses_every_weighted_row_and_separates_signs():
    generator = torch.Generator().manual_seed(7)
    positive = torch.randn((40, 8), generator=generator) * 0.03
    negative = torch.randn((40, 8), generator=generator) * 0.03
    positive[:, 0] += 1.0
    negative[:, 0] -= 1.0
    features = torch.cat([positive, negative])
    signed = torch.cat([torch.linspace(0.1, 1.0, 40), -torch.linspace(0.1, 1.0, 40)])
    responsibility = torch.linspace(0.5, 2.0, 80)
    probability, diagnostics = diagonal_shrinkage_lda(
        features,
        signed,
        responsibility,
        device=torch.device("cpu"),
        chunk_size=13,
    )
    assert diagnostics["positive_training_rows"] == 40
    assert diagnostics["negative_training_rows"] == 40
    assert probability[:40].min() > 0.5
    assert probability[40:].max() < 0.5


def test_all_scribble_diagonal_lda_fails_closed_on_sparse_sign():
    features = torch.eye(64)
    signed = torch.cat([torch.ones(31), -torch.ones(33)])
    responsibility = torch.ones(64)
    try:
        diagonal_shrinkage_lda(
            features,
            signed,
            responsibility,
            device=torch.device("cpu"),
            chunk_size=16,
        )
    except ValueError as error:
        assert "at least 32" in str(error)
    else:
        raise AssertionError("sparse signed population did not fail closed")
