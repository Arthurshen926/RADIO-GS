import torch
from radio_gs.v4.query.overlapping_candidates import compose_overlapping_candidates
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import compose_object_queries


def test_unselected_candidates_cannot_suppress_selected_object():
    membership = torch.ones(1, 1)
    query = torch.tensor([[.8]])
    expected, _ = compose_overlapping_candidates(membership, query)
    expanded = torch.ones(1, 6)
    support = torch.tensor([[.8, 0, 0, 0, 0, 0]])
    actual, _ = compose_overlapping_candidates(expanded, support, chunk_size=2)
    assert torch.equal(actual, expected)
    old, _ = compose_object_queries(expanded, support)
    assert old.item() < expected.item()


def test_duplicate_evidence_multiple_objects_and_no_match():
    membership = torch.tensor([[1., 0], [0, 1.], [.5, .5]])
    queries = torch.tensor([[.8, .7], [0., 0.]])
    result, _ = compose_overlapping_candidates(membership, queries)
    torch.testing.assert_close(result, torch.tensor([[.8, 0], [.7, 0], [.4, 0]]))
    duplicated, _ = compose_overlapping_candidates(membership.repeat(1, 3), queries.repeat(1, 3))
    assert torch.equal(duplicated, result)
