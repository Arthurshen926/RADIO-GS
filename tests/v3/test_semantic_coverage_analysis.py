import torch

from radio_gs.v3.evaluation.analyze_semantic_coverage import (
    _exclusive_coverage,
    _intersect_sorted,
)


def test_coverage_partition_is_first_failure_and_exhaustive() -> None:
    categories = _exclusive_coverage(
        visible=torch.tensor([0, 1, 1, 1, 1, 1]),
        sam_supported=torch.tensor([0, 0, 1, 1, 1, 1]),
        retained=torch.tensor([0, 0, 0, 1, 1, 1]),
        mixed=torch.tensor([0, 0, 0, 1, 0, 0]),
        semantic_valid=torch.tensor([0, 0, 0, 1, 0, 1]),
    )
    assert [int(value.sum()) for value in categories.values()] == [1, 1, 1, 1, 1, 1]
    assert torch.stack(tuple(categories.values())).sum(0).eq(1).all()


def test_sorted_intersection_preserves_common_rows() -> None:
    assert torch.equal(
        _intersect_sorted(torch.tensor([1, 4, 8]), torch.tensor([0, 1, 8, 9])),
        torch.tensor([1, 8]),
    )
