import torch

from radio_gs.models.object_aware_universal_field_v2 import (
    ObjectAwareFieldHead,
    object_aware_proper_loss,
    scale_bins,
    sparse_proposal_pool,
)
from radio_gs.scripts.train_lerf_object_aware_field_v2_pilot import (
    fixed_view_split,
    relation_metrics,
)


def test_sparse_pool_is_exact_weighted_mean() -> None:
    codes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pooled, mass = sparse_proposal_pool(
        codes,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 3.0, 2.0]),
        2,
    )
    assert torch.allclose(pooled, torch.tensor([[0.25, 0.75], [1.0, 1.0]]))
    assert torch.equal(mass, torch.tensor([4.0, 2.0]))


def test_unknown_relation_has_no_effect_on_proper_loss() -> None:
    embedding = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    decoded = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
    teacher = decoded.clone()
    base = object_aware_proper_loss(
        embedding, decoded, teacher,
        torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([1, 0]),
    )
    with_unknown = object_aware_proper_loss(
        embedding, decoded, teacher,
        torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]), torch.tensor([1, 0, -1]),
    )
    assert torch.equal(base.total, with_unknown.total)


def test_scale_bins_and_source_split_are_fixed() -> None:
    assert scale_bins(torch.tensor([0.001, 0.01, 0.03, 0.2])).tolist() == [0, 1, 2, 3]
    train, heldout = fixed_view_split(torch.arange(8))
    assert heldout.tolist() == [False, False, False, True, False, False, False, True]
    assert torch.equal(train, ~heldout)


def test_relation_auc_is_threshold_free() -> None:
    result = relation_metrics(torch.tensor([0.9, 0.8, -0.2, -0.4]), torch.tensor([1, 1, 0, 0]))
    assert result["auc"] == 1.0


def test_head_persists_only_compact_object_codes() -> None:
    head = ObjectAwareFieldHead(7, object_dim=16, language_dim=32)
    assert head.object_codes.shape == (7, 16)
    assert head.scale_log_gates.shape == (4, 16)
