import torch

from radio_gs.scripts.materialize_source_same_axis_o0_query_subset import (
    select_dominant_absolute_queries,
)


def test_select_dominant_absolute_queries_is_fixed_and_sorted():
    region = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    positive = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=torch.float32
    )
    negative = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    result = select_dominant_absolute_queries(
        region_descriptors=region,
        positive_embeddings=positive,
        negative_embeddings=negative,
    )
    assert result["absolute_region_mask"].tolist() == [True, False, False]
    assert result["dominant_global_index"].tolist() == [0, 1, 1]
    assert result["selected_global_indices"].tolist() == [0]
    assert result["dominant_positive_subset_index"].tolist() == [0, -1, -1]


def test_select_dominant_absolute_queries_uses_lower_index_tie_break():
    region = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    positive = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    negative = torch.tensor(
        [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    result = select_dominant_absolute_queries(
        region_descriptors=region,
        positive_embeddings=positive,
        negative_embeddings=negative,
    )
    assert result["dominant_global_index"].item() == 0
    assert result["selected_global_indices"].tolist() == [0]


def test_select_dominant_absolute_queries_rejects_axis_mismatch():
    try:
        select_dominant_absolute_queries(
            region_descriptors=torch.ones(2, 3),
            positive_embeddings=torch.ones(4, 2),
            negative_embeddings=torch.ones(4, 3),
        )
    except ValueError as error:
        assert "inputs differ" in str(error)
    else:
        raise AssertionError("axis mismatch must fail closed")
