import numpy as np
import torch

from radio_gs.scripts.build_sam3_mpr_confirmed_mask_cache import SourceMask
from radio_gs.scripts.build_sam3_mpr_cache_reuse_confirmation import (
    containing_mask_indices,
)
from radio_gs.scripts.build_scannet_mpr_confirmed_relation_cache import (
    _adjoint_provenance,
    _batch_virtual_tracks,
    _cached_automatic_target_memberships,
    _select_prompted_mask_shard,
)


def test_cache_reuse_keeps_all_official_masks_at_mpr_anchor() -> None:
    masks = np.zeros((3, 3, 4), dtype=bool)
    masks[0, 1, 2] = True
    masks[1, 1, 2] = True
    masks[2, 0, 0] = True
    # The target point is rounded to its nearest raster centre.  Both valid
    # hierarchical candidates are retained; they are not ranked or collapsed.
    assert containing_mask_indices(masks, x=1.6, y=1.4).tolist() == [0, 1]


def test_cache_reuse_anchor_is_clamped_to_official_raster() -> None:
    masks = np.zeros((1, 2, 2), dtype=bool)
    masks[0, 1, 0] = True
    assert containing_mask_indices(masks, x=-20.0, y=20.0).tolist() == [0]


def test_cache_reuse_relation_reuses_exact_frozen_target_adjoint_rows() -> None:
    observed = torch.tensor([True, True, False])
    target_rows = {
        (40, 2): SourceMask(40, 2, torch.tensor([0.9, 0.8, 0.1]), observed, 0.9, 1.0),
        (40, 5): SourceMask(40, 5, torch.tensor([0.2, 0.95, 0.0]), observed, 0.8, 1.0),
    }
    payload = {
        "candidate_index": torch.tensor([5, 2]),
        "metadata": {
            "source": "official_sam3_mpr_confirmed_cached_multimask_teacher_control",
            "confirmation_mode": "reuse_existing_official_automatic_mask_at_exact_mpr_anchor",
        },
    }
    memberships, reused_observed = _cached_automatic_target_memberships(
        payload, target_frame=40, count=2, source_by_identity=target_rows,
    )
    assert torch.equal(memberships, torch.tensor([[0.2, 0.95, 0.0], [0.9, 0.8, 0.1]]))
    assert torch.equal(reused_observed, observed)


def test_real_redecoded_cache_cannot_take_cache_reuse_shortcut() -> None:
    payload = {
        "candidate_index": torch.tensor([0]),
        "metadata": {"source": "official_sam3_mpr_confirmed_cross_view_multimask_teacher"},
    }
    assert _cached_automatic_target_memberships(
        payload, target_frame=40, count=1, source_by_identity={},
    ) is None


def test_virtual_track_batching_retains_one_observation_row_per_track() -> None:
    memberships = [torch.tensor([[0.9, 0.1]]), torch.tensor([[0.2, 0.8]])]
    observations = [torch.tensor([True, False]), torch.tensor([False, True])]
    radii = [torch.tensor([0.2]), torch.tensor([0.5])]
    qualities = [torch.tensor([0.8]), torch.tensor([0.9])]
    stabilities = [torch.tensor([1.0]), torch.tensor([0.7])]
    packed = _batch_virtual_tracks(
        memberships, observations, radii, qualities, stabilities, batch_size=8,
    )
    assert len(packed[0]) == 1
    assert torch.equal(packed[0][0], torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
    assert torch.equal(packed[1][0], torch.tensor([[True, False], [False, True]]))
    assert torch.equal(packed[2][0], torch.tensor([0.2, 0.5]))


def test_cache_reuse_provenance_comes_from_the_frozen_mpr_contract() -> None:
    metadata = {
        "config": "/tmp/config.yaml", "checkpoint": "/tmp/model.pth",
        "xyz_sha256": "xyz", "gaussian_state_sha256": "state", "pose_sha256": "pose",
        "feature_height": 60, "feature_width": 81, "alpha_threshold": 0.02,
    }
    provenance = _adjoint_provenance(
        responsibility_metadata=metadata, adjoint_context=None, channel_chunk_size=32,
    )
    assert provenance == {**metadata, "channel_chunk_size": 32}


def test_relation_target_shards_are_disjoint_by_numeric_frame() -> None:
    from pathlib import Path

    paths = [Path(name) for name in ("100.pt", "0.pt", "40.pt", "80.pt")]
    assert _select_prompted_mask_shard(paths, shard_index=0, shard_count=2) == [Path("0.pt"), Path("80.pt")]
    assert _select_prompted_mask_shard(paths, shard_index=1, shard_count=2) == [Path("40.pt"), Path("100.pt")]
