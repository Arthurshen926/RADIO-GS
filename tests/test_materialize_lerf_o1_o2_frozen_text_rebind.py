from __future__ import annotations

import pytest
import torch

from radio_gs.scripts import materialize_lerf_o1_o2_frozen_text_rebind as rebind


def _bank(queries: list[str], embeddings: torch.Tensor) -> dict[str, object]:
    return {"queries": queries, "embeddings": embeddings.float()}


def test_frozen_selection_preserves_requested_order() -> None:
    frozen = _bank(
        ["b", "a", "c"],
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]),
    )
    selected = rebind.select_frozen_embeddings(frozen, ["c", "a"])
    assert torch.equal(selected, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))


def test_binding_detects_one_stale_query_without_rejecting_roundoff() -> None:
    frozen = _bank(
        ["a", "dall-e brand"],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    source = _bank(
        ["a", "dall-e brand"],
        torch.tensor([[1.0, 1e-8, 0.0], [0.0, 0.8, 0.6]]),
    )
    report = rebind.compare_source_to_frozen(
        source, frozen, ["a", "dall-e brand"]
    )
    assert report["equivalent"] == [True, False]
    assert report["mismatched_query_ids"] == ["dall-e brand"]
    assert report["all_equivalent"] is False


def test_binding_fails_closed_on_axis_low_norm_and_missing_query() -> None:
    frozen = _bank(["a"], torch.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="query axis"):
        rebind.compare_source_to_frozen(
            _bank(["b"], torch.tensor([[1.0, 0.0]])), frozen, ["a"]
        )
    with pytest.raises(ValueError, match="low-norm"):
        rebind.select_frozen_embeddings(
            _bank(["a"], torch.zeros(1, 2)), ["a"]
        )
    with pytest.raises(ValueError, match="missing"):
        rebind.select_frozen_embeddings(frozen, ["b"])


def test_access_audit_is_cpu_only_and_metric_closed() -> None:
    audit = rebind.access_audit()
    assert audit["gpu_used"] is False
    assert audit["target_metrics_opened"] is False
    assert audit["target_metrics_computed"] is False


@pytest.mark.parametrize("oracle", ["O1", "O2"])
def test_recompute_changes_only_requested_query_columns(oracle: str) -> None:
    parent = torch.full((3, 3, 2), 0.125, dtype=torch.float32)
    rows = torch.tensor([0, 2])
    base = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = rebind._recompute_mismatched_columns(
        parent,
        embeddings=embeddings,
        mismatched_indices=[1],
        oracle=oracle,
        rows=rows,
        base_features=base,
        teacher_mean=teacher,
        teacher_valid=torch.tensor([True, False]),
    )
    assert torch.equal(result[:, :, 0], parent[:, :, 0])
    assert torch.equal(result[1, :, 1], parent[1, :, 1])
    assert not torch.equal(result[rows, :, 1], parent[rows, :, 1])
