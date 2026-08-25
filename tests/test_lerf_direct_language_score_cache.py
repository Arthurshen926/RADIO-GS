import torch

from radio_gs.scripts.materialize_lerf_direct_language_score_cache import update_rowwise_topk


def test_rowwise_topk_keeps_strongest_observed_view_responses_per_query() -> None:
    topk = torch.full((2, 2, 2), -torch.inf)
    update_rowwise_topk(topk, torch.tensor([[0.2, 0.8], [9.0, 9.0]]), torch.tensor([True, False]))
    update_rowwise_topk(topk, torch.tensor([[0.5, 0.4], [0.7, 0.6]]), torch.tensor([True, True]))
    update_rowwise_topk(topk, torch.tensor([[0.3, 0.9], [0.8, 0.1]]), torch.tensor([True, True]))
    torch.testing.assert_close(topk[0], torch.tensor([[0.5, 0.3], [0.9, 0.8]]))
    torch.testing.assert_close(topk[1], torch.tensor([[0.8, 0.7], [0.6, 0.1]]))


def test_rowwise_topk_rejects_mismatched_domains() -> None:
    try:
        update_rowwise_topk(torch.zeros(2, 3, 1), torch.zeros(2, 4), torch.ones(2, dtype=torch.bool))
    except ValueError as error:
        assert "domain differs" in str(error)
    else:
        raise AssertionError("mismatched robust aggregation domain was accepted")
