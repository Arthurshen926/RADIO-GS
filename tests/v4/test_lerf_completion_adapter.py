from __future__ import annotations

import torch

from radio_gs.v4.carrier import Camera, SurfaceVoxelCarrier
from radio_gs.v4.completion.lerf_adapter import build_real_token_runtime


def test_real_token_runtime_hardens_only_observed_source_rows():
    carrier = SurfaceVoxelCarrier(
        torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]
        ),
        0.1,
        maximum_splat_radius=0,
        surface_band_voxels=1.0,
        maximum_contributors_per_pixel=1,
    )
    camera = Camera("source", torch.eye(3), torch.eye(4), 2, 2)
    membership = torch.tensor(
        [[0.9, 0.2], [0.7, 0.8], [0.0, 0.0], [0.0, 0.6]]
    )
    local_features = torch.randn(4, 71)

    runtime, audit = build_real_token_runtime(
        carrier=carrier,
        local_features=local_features,
        source_visible=torch.tensor([True, True, False, True]),
        observed_membership=membership,
        observation_cameras=[camera],
        view_token_ids=[torch.tensor([0, 1])],
        observed_threshold=0.5,
    )

    expected_positive = torch.tensor(
        [[True, False], [False, True], [False, False], [False, True]]
    )
    assert torch.equal(runtime["partial"].positive, expected_positive)
    assert bool(runtime["partial"].unknown[2].all())
    assert int(runtime["partial"].positive.sum(-1).max()) == 1
    assert audit["overlap_discarded_element_count"] == 1
    assert audit["complete_target_labels_read"] is False


def test_real_token_runtime_compacts_tokens_without_categorical_seed():
    carrier = SurfaceVoxelCarrier(
        torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]
        ),
        0.1,
        maximum_splat_radius=0,
        surface_band_voxels=1.0,
        maximum_contributors_per_pixel=1,
    )
    camera = Camera("source", torch.eye(3), torch.eye(4), 2, 2)
    # Token 1 has genuine soft source evidence and a source proposal receipt,
    # but it never wins an element-wise categorical assignment.
    membership = torch.tensor(
        [[0.9, 0.8, 0.0], [0.7, 0.6, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, 0.0]]
    )

    runtime, audit = build_real_token_runtime(
        carrier=carrier,
        local_features=torch.randn(4, 71),
        source_visible=torch.tensor([True, True, True, False]),
        observed_membership=membership,
        observation_cameras=[camera],
        view_token_ids=[torch.tensor([0, 1, 2])],
        observed_threshold=0.5,
    )

    assert torch.equal(runtime["active_token_ids"], torch.tensor([0, 2]))
    assert runtime["payload"]["object_ids"] == [0, 2]
    assert runtime["partial"].positive.shape == (4, 2)
    assert audit["active_completion_token_count"] == 2
    assert audit["inactive_unseeded_token_ids"] == [1]
