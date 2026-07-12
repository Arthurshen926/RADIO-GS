import json

import torch

from radio_gs.losses.samclip_mask_loss import (
    compute_samclip_mask_losses,
    load_samclip_mask_manifest,
)


def _prototype_image(prototypes: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    height, width = segments.shape
    out = torch.zeros(prototypes.shape[1], height, width)
    valid = (segments >= 0) & (segments < prototypes.shape[0])
    out.permute(1, 2, 0)[valid] = prototypes[segments[valid]]
    return out


def test_mask_prototype_loss_is_near_zero_for_matching_segment_features() -> None:
    prototypes = torch.eye(3)
    segments = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, -1, -1],
            [2, 2, 99, 99],
        ],
        dtype=torch.long,
    )
    pred = _prototype_image(prototypes, segments)

    losses = compute_samclip_mask_losses(
        pred,
        prototypes,
        segments,
        min_pixels=2,
        contrastive_temperature=0.1,
    )

    assert losses["valid_regions"].item() == 3
    assert losses["prototype_loss"].item() < 1e-6
    assert losses["contrastive_loss"].item() < 1e-3


def test_mask_prototype_loss_increases_when_regions_are_swapped() -> None:
    prototypes = torch.eye(3)
    target_segments = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        dtype=torch.long,
    )
    swapped_segments = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        dtype=torch.long,
    )
    aligned = compute_samclip_mask_losses(
        _prototype_image(prototypes, target_segments),
        prototypes,
        target_segments,
        min_pixels=2,
        contrastive_temperature=0.1,
    )
    swapped = compute_samclip_mask_losses(
        _prototype_image(prototypes, swapped_segments),
        prototypes,
        target_segments,
        min_pixels=2,
        contrastive_temperature=0.1,
    )

    assert swapped["prototype_loss"] > aligned["prototype_loss"] + 0.5
    assert swapped["contrastive_loss"] > aligned["contrastive_loss"] + 1.0


def test_mask_loss_returns_zero_when_no_region_has_enough_pixels() -> None:
    prototypes = torch.eye(2)
    segments = torch.tensor([[0, 1], [1, -1]], dtype=torch.long)
    pred = _prototype_image(prototypes, segments)

    losses = compute_samclip_mask_losses(pred, prototypes, segments, min_pixels=4)

    assert losses["valid_regions"].item() == 0
    assert losses["prototype_loss"].item() == 0.0
    assert losses["contrastive_loss"].item() == 0.0
    assert losses["total_loss"].item() == 0.0


def test_mask_loss_penalizes_nonzero_invalid_background() -> None:
    prototypes = torch.eye(3)[:2]
    segments = torch.tensor([[0, -1], [1, -1]], dtype=torch.long)
    pred = torch.zeros(3, 2, 2)
    pred[:, 0, 0] = prototypes[0]
    pred[:, 1, 0] = prototypes[1]

    zero_losses = compute_samclip_mask_losses(
        pred,
        prototypes,
        segments,
        min_pixels=1,
        contrastive_weight=0.0,
        background_weight=1.0,
    )

    assert torch.isclose(zero_losses["background_loss"], torch.tensor(0.0))
    assert torch.isclose(zero_losses["total_loss"], zero_losses["prototype_loss"])

    pred[:, 0, 1] = torch.tensor([0.0, 0.0, 2.0])
    pred[:, 1, 1] = torch.tensor([0.0, 0.0, 1.0])
    losses = compute_samclip_mask_losses(
        pred,
        prototypes,
        segments,
        min_pixels=1,
        contrastive_weight=0.0,
        background_weight=1.0,
    )

    assert losses["background_loss"] > 0.0
    assert losses["total_loss"] > losses["prototype_loss"]


def test_load_samclip_mask_manifest_indexes_entries_by_frame_id(tmp_path) -> None:
    level_root = tmp_path / "scene" / "l1"
    level_root.mkdir(parents=True)
    feature = tmp_path / "frame_00041_f.npy"
    segments = tmp_path / "frame_00041_s.npy"
    manifest = {
        "outputs": [
            {
                "frame_id": 41,
                "stem": "frame_00041",
                "feature": str(feature),
                "segments": str(segments),
                "tensor": "backbone/rgb_41.pt",
            }
        ]
    }
    (level_root / "samclip_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    entries = load_samclip_mask_manifest(level_root)

    assert sorted(entries) == [41]
    assert entries[41].feature_path == feature
    assert entries[41].segments_path == segments
