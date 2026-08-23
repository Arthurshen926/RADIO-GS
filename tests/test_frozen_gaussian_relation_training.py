import torch

from radio_gs.scripts.train_evaluate_frozen_gaussian_relation_decoder import (
    sample_gaussian_relation_pairs,
)


def test_relation_pair_sampling_excludes_unknown_and_heldout_proposals():
    left, right, label = sample_gaussian_relation_pairs(
        edge_left=torch.tensor([0, 0, 1]),
        edge_right=torch.tensor([1, 2, 2]),
        edge_relation=torch.tensor([1, -1, 0]),
        supports=[torch.tensor([3]), torch.tensor([4]), torch.tensor([5])],
        selected_proposals=torch.tensor([True, True, True]),
        samples_per_edge=2,
        seed=1,
    )
    assert left.tolist() == [3, 3, 4, 4]
    assert right.tolist() == [4, 4, 5, 5]
    assert label.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_relation_pair_sampling_requires_both_known_classes():
    try:
        sample_gaussian_relation_pairs(
            edge_left=torch.tensor([0]),
            edge_right=torch.tensor([1]),
            edge_relation=torch.tensor([1]),
            supports=[torch.tensor([3]), torch.tensor([4])],
            selected_proposals=torch.tensor([True, True]),
            samples_per_edge=1,
            seed=1,
        )
    except ValueError as error:
        assert "same and different" in str(error)
    else:
        raise AssertionError("one-class relation authority must fail closed")
