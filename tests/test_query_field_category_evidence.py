import pytest
import torch

from radio_gs.query_field.category_evidence import corroborated_category_scores


def test_no_member_evidence_preserves_native_competition_exactly():
    native = torch.tensor([[.21, .22, .1], [.3, .2, .1]])
    result = corroborated_category_scores(native, torch.zeros_like(native))
    assert torch.equal(result, (100 * native).softmax(1))
    assert torch.equal(result.argmax(1), native.argmax(1))


def test_member_evidence_is_bounded_and_can_resolve_low_margin_competition():
    native = torch.tensor([[.2, .199, .1], [.2, .1, .0]])
    support = torch.tensor([[0., 1., 0.], [0., 1., 1.]])
    prior = (100 * native).softmax(1)
    result = corroborated_category_scores(native, support)
    assert (result >= prior).all() and (result <= 2 * prior).all()
    assert result[0].argmax() == 1  # Structure actually influences the decision.
    assert result[1].argmax() == 0  # Strong native evidence is preserved.


def test_complete_menu_scores_are_computed_before_reported_subsets():
    native = torch.tensor([[.201, .2, .199]])
    support = torch.tensor([[.1, .8, .4]])
    full = corroborated_category_scores(native, support)
    assert not torch.equal(full[:, :2], corroborated_category_scores(native[:, :2], support[:, :2]))


def test_category_support_is_not_an_unbounded_logit():
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        corroborated_category_scores(torch.ones(2, 3), torch.full((2, 3), 2.))
