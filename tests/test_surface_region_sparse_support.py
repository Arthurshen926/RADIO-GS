import pytest
import torch

from radio_gs.training.surface_region_sparse_support import (
    deterministic_sparse_token_support,
)


def _selection(
    region_ids: list[str],
    *,
    seed: int = 17,
    epoch: int = 4,
):
    mask = torch.ones(len(region_ids), 64, dtype=torch.bool)
    return deterministic_sparse_token_support(
        mask,
        anchor_index=torch.zeros(len(region_ids), dtype=torch.long),
        region_ids=region_ids,
        minimum_tokens=8,
        seed=seed,
        epoch=epoch,
    )


def test_sparse_support_has_stable_log_uniform_count_regression() -> None:
    selection = _selection(["region-a", "region-b", "场景/区域-c"])
    assert selection.kept_counts.tolist() == [31, 29, 44]
    assert bool((selection.kept_counts >= 8).all())
    assert bool((selection.kept_counts <= 64).all())
    for row, count in enumerate(selection.kept_counts.tolist()):
        assert selection.token_mask[row, :count].all()
        assert not selection.token_mask[row, count:].any()


def test_sparse_support_is_independent_of_batch_order_and_cache_sharding() -> None:
    region_ids = [f"scene-x/region-{index}" for index in range(12)]
    together = _selection(region_ids)
    order = torch.tensor([8, 2, 10, 0, 11, 4, 1, 9, 5, 7, 3, 6])
    shuffled_ids = [region_ids[index] for index in order.tolist()]
    shuffled = _selection(shuffled_ids)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    assert torch.equal(shuffled.token_mask[inverse], together.token_mask)
    assert torch.equal(shuffled.kept_counts[inverse], together.kept_counts)

    first = _selection(region_ids[:5])
    second = _selection(region_ids[5:])
    assert torch.equal(
        torch.cat((first.token_mask, second.token_mask)), together.token_mask
    )
    assert torch.equal(
        torch.cat((first.kept_counts, second.kept_counts)), together.kept_counts
    )


def test_sparse_support_keeps_anchor_and_nearest_valid_token_order() -> None:
    mask = torch.tensor(
        [
            [True, True, False, True, True, False, True, True],
            [True, True, True, True, True, True, True, True],
        ]
    )
    selection = deterministic_sparse_token_support(
        mask,
        anchor_index=torch.tensor([7, 0]),
        region_ids=["far-anchor", "nearest-anchor"],
        minimum_tokens=3,
        seed=9,
        epoch=2,
    )
    assert selection.token_mask[0, 7]
    assert int(selection.token_mask[0].sum()) == int(selection.kept_counts[0])
    # A far anchor replaces only the farthest otherwise-selected valid token.
    count = int(selection.kept_counts[0])
    expected = torch.tensor([0, 1, 3, 4, 6])[:count].tolist()
    if 7 not in expected:
        expected[-1] = 7
    assert torch.nonzero(selection.token_mask[0]).flatten().tolist() == sorted(expected)
    # With the anchor already first, selection is exactly the valid prefix.
    nearest_count = int(selection.kept_counts[1])
    assert torch.nonzero(selection.token_mask[1]).flatten().tolist() == list(
        range(nearest_count)
    )


def test_sparse_support_changes_across_epoch_without_global_rng_state() -> None:
    region_ids = [f"region-{index}" for index in range(64)]
    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state().clone()
    first = _selection(region_ids, epoch=1)
    rng_after = torch.random.get_rng_state()
    second = _selection(region_ids, epoch=2)
    assert torch.equal(rng_before, rng_after)
    assert not torch.equal(first.kept_counts, second.kept_counts)


def test_sparse_support_zeroes_named_tensors_and_preserves_gradients() -> None:
    mask = torch.ones(2, 12, dtype=torch.bool)
    selection = deterministic_sparse_token_support(
        mask,
        anchor_index=torch.tensor([0, 0]),
        region_ids=["a", "b"],
        minimum_tokens=3,
        seed=3,
        epoch=1,
    )
    features = torch.arange(48.0).reshape(2, 12, 2).requires_grad_()
    reliability = torch.ones(2, 12, 1)
    zeroed = selection.zero_tensors(
        {"features": features, "reliability": reliability}
    )
    expanded_mask = selection.token_mask[..., None]
    assert torch.equal(zeroed["features"][~expanded_mask.expand_as(features)], torch.zeros_like(zeroed["features"][~expanded_mask.expand_as(features)]))
    assert torch.equal(
        zeroed["features"][expanded_mask.expand_as(features)],
        features[expanded_mask.expand_as(features)],
    )
    assert torch.equal(
        zeroed["reliability"], selection.token_mask[..., None].float()
    )
    zeroed["features"].sum().backward()
    assert torch.equal(
        features.grad != 0, expanded_mask.expand_as(features)
    )


def test_sparse_support_fails_closed_on_invalid_regions() -> None:
    with pytest.raises(ValueError, match="fewer valid tokens"):
        deterministic_sparse_token_support(
            torch.tensor([[True, True, False]]),
            anchor_index=[0],
            region_ids=["too-small"],
            minimum_tokens=3,
            seed=0,
            epoch=0,
        )
    with pytest.raises(ValueError, match="anchor.*valid"):
        deterministic_sparse_token_support(
            torch.tensor([[True, False, True]]),
            anchor_index=[1],
            region_ids=["bad-anchor"],
            minimum_tokens=2,
            seed=0,
            epoch=0,
        )
    selection = _selection(["one"])
    with pytest.raises(ValueError, match=r"\[batch, token\]"):
        selection.zero_tensor(torch.ones(1, 63, 4))
