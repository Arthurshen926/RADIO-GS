from __future__ import annotations

import torch

from radio_gs.v4.completion.oracle import PartialObjectMembership
from radio_gs.v4.training.train_scannet_spatial_slots import (
    _sample_training_indices,
    build_observed_pca_geometry,
)


def _runtime():
    positive = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
            [False, False],
        ]
    )
    unknown = torch.tensor(
        [
            [False, False],
            [False, False],
            [False, False],
            [False, False],
            [True, True],
            [True, True],
            [True, True],
            [True, True],
        ]
    )
    negative = ~(positive | unknown)
    partial = PartialObjectMembership(
        positive=positive,
        negative=negative,
        unknown=unknown,
        eligible_elements=torch.ones(8, dtype=torch.bool),
    )
    return {
        "partial": partial,
        "centres": torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [5.0, 5.0, 0.0],
                [6.0, 5.0, 0.0],
            ]
        ),
        "labels": torch.tensor([0, 0, 1, 1, 0, 1, -1, -1]),
        "minimum_scale": 0.04,
    }


def test_observed_pca_geometry_uses_only_observed_token_points():
    runtime = _runtime()
    centres, frames, scales = build_observed_pca_geometry(runtime)

    torch.testing.assert_close(centres, torch.zeros(2, 3))
    torch.testing.assert_close(
        frames.transpose(-1, -2) @ frames,
        torch.eye(3).expand(2, 3, 3),
        atol=1e-6,
        rtol=0,
    )
    assert scales.shape == (2, 3)
    assert bool((scales >= 0.04).all())


def test_spatial_slot_sampler_uses_unknown_object_and_null_rows_only():
    indices, targets = _sample_training_indices(
        _runtime(), maximum_positive_per_token=2, maximum_null=2, seed=7
    )

    assert set(indices.tolist()) == {4, 5, 6, 7}
    assert set(targets.tolist()) == {0, 1, 2}
