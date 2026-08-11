import torch

from radio_gs.querying.view_support_carrier import (
    build_compact_view_support,
    dense_support_to_csr,
    edge_covisibility_from_support,
)


def test_view_support_mapping_is_duplicate_invariant_and_sparse_stable() -> None:
    global_rows = torch.tensor([2, 5, 9, 12])
    support = build_compact_view_support(
        global_rows=global_rows,
        view_global_ids=[
            torch.tensor([2, 2, 9, 100]),
            torch.tensor([5, 9]),
            torch.tensor([2, 12, 12]),
        ],
    )
    expected = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.bool
    )
    assert torch.equal(support, expected)
    crow, col = dense_support_to_csr(support)
    assert torch.equal(crow, torch.tensor([0, 2, 3, 5, 6]))
    assert torch.equal(col.long(), torch.tensor([0, 2, 1, 0, 1, 2]))


def test_edge_covisibility_has_exact_jaccard_and_conditionals() -> None:
    support = torch.tensor(
        [[1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=torch.bool
    )
    edges = torch.tensor([[0, 0, 0], [1, 2, 3]])
    result = edge_covisibility_from_support(
        support=support, edge_index=edges, chunk_size=2
    )
    assert torch.equal(result["shared_view_count"], torch.tensor([1, 2, 0], dtype=torch.uint8))
    assert torch.allclose(result["jaccard"].float(), torch.tensor([1 / 3, 2 / 3, 0.0]), atol=5e-4)
    assert torch.allclose(result["source_given_target"].float(), torch.tensor([0.5, 2 / 3, 0.0]), atol=5e-4)
    assert torch.allclose(result["target_given_source"].float(), torch.tensor([0.5, 1.0, 0.0]), atol=5e-4)
    assert torch.allclose(result["overlap_coefficient"].float(), torch.tensor([0.5, 1.0, 0.0]), atol=5e-4)


def test_unobserved_endpoint_is_exact_zero_not_false_covisibility() -> None:
    result = edge_covisibility_from_support(
        support=torch.tensor([[0, 0], [1, 0]], dtype=torch.bool),
        edge_index=torch.tensor([[0], [1]]),
    )
    for name in [
        "jaccard",
        "source_given_target",
        "target_given_source",
        "overlap_coefficient",
    ]:
        assert result[name].item() == 0.0
