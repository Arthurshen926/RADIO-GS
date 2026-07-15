from pathlib import Path
import random

import pytest

import torch
import torch.nn.functional as F

from radio_gs.scripts.build_generic_region_summary_cache import (
    _crop,
    _parse_crop_scales,
    _pool_spatial_region,
    _select_shard,
)
from radio_gs.scripts.merge_generic_region_summary_caches import (
    _validate_disjoint_shard_images,
)
from radio_gs.scripts.train_global_region_summary_bridge import (
    _parse_token_grid_sizes,
    _pool_region_tokens,
    _split_rows,
)


def test_deterministic_region_cache_shards_are_complete_and_disjoint() -> None:
    paths = [Path(f"image_{index}.jpg") for index in range(11)]
    shards = [_select_shard(paths, 4, index) for index in range(4)]

    assert {path for shard in shards for path in shard} == set(paths)
    assert sum(len(set(shard)) for shard in shards) == len(paths)
    assert all(set(left).isdisjoint(right) for i, left in enumerate(shards) for right in shards[i + 1 :])


def test_region_cache_shard_contract_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError):
        _select_shard([Path("image.jpg")], 0, 0)
    with pytest.raises(ValueError):
        _select_shard([Path("image.jpg")], 2, 2)
    with pytest.raises(ValueError):
        _select_shard([Path("image.jpg")], 2, 1)


def test_image_level_bridge_split_keeps_crops_grouped_and_prior_images_unseen() -> None:
    metadata = {
        "crop_records": [
            {"image": image, "crop_box_tl_hw": [0, 0, 8, 8]}
            for image in ("old.jpg", "old.jpg", "new_a.jpg", "new_a.jpg", "new_b.jpg", "new_b.jpg")
        ]
    }
    train_rows, validation_rows, manifest = _split_rows(
        metadata,
        6,
        validation_fraction=1.0 / 3.0,
        seed=0,
        split_unit="image",
        validation_excluded_images={"old.jpg"},
    )

    train_images = {metadata["crop_records"][index]["image"] for index in train_rows.tolist()}
    validation_images = {
        metadata["crop_records"][index]["image"] for index in validation_rows.tolist()
    }
    assert train_images.isdisjoint(validation_images)
    assert "old.jpg" not in validation_images
    assert validation_rows.numel() == 2
    assert manifest["unit"] == "image"
    assert manifest["validation_images"] == 1


def test_merge_allows_repeated_crop_scale_but_rejects_cross_shard_image_overlap() -> None:
    shard_a = {
        "num_images": 1,
        "num_crops": 2,
        "crop_records": [
            {"image": "a.jpg", "crop_box_tl_hw": [0, 0, 8, 8]},
            {"image": "a.jpg", "crop_box_tl_hw": [0, 0, 8, 8]},
        ],
    }
    shard_b = {
        "num_images": 1,
        "num_crops": 1,
        "crop_records": [
            {"image": "b.jpg", "crop_box_tl_hw": [0, 0, 8, 8]},
        ],
    }
    assert _validate_disjoint_shard_images([shard_a, shard_b]) == [
        {"a.jpg"},
        {"b.jpg"},
    ]

    shard_b["crop_records"][0]["image"] = "a.jpg"
    with pytest.raises(ValueError, match="duplicate source images"):
        _validate_disjoint_shard_images([shard_a, shard_b])


def test_full_image_context_region_pooling_matches_spatial_slices() -> None:
    spatial = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    full = _pool_spatial_region(
        spatial,
        [0, 0, 4, 4],
        image_height=4,
        image_width=4,
        token_grid=2,
    )
    expected_full = F.adaptive_avg_pool2d(spatial, (2, 2)).flatten(1).T
    assert torch.equal(full, expected_full)

    quarter = _pool_spatial_region(
        spatial,
        [0, 0, 2, 2],
        image_height=4,
        image_width=4,
        token_grid=2,
    )
    assert torch.equal(quarter[:, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))


def test_local_region_crop_scales_cycle_deterministically() -> None:
    image = torch.zeros(3, 100, 200)
    rng = random.Random(0)
    scales = _parse_crop_scales("0.06, 0.13 0.28")
    boxes = [
        _crop(
            image,
            rng,
            index,
            scales=scales,
            scale_policy="cycle",
        )[1]
        for index in range(3)
    ]

    assert scales == (0.06, 0.13, 0.28)
    assert [(box[2], box[3]) for box in boxes] == [(16, 16), (16, 26), (28, 56)]
    with pytest.raises(ValueError, match="crop scales"):
        _parse_crop_scales("0,1.2")


def test_bridge_token_density_pooling_matches_square_query_grids() -> None:
    metadata = {"region_token_grid": [8, 8]}
    source_grid, grids = _parse_token_grid_sizes(metadata, "3,7,8")
    tokens = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1)

    pooled = _pool_region_tokens(
        tokens, source_grid=source_grid, target_grid=3
    )
    expected = F.adaptive_avg_pool2d(
        tokens.transpose(1, 2).reshape(1, 1, 8, 8), (3, 3)
    ).flatten(2).transpose(1, 2)

    assert grids == (3, 7, 8)
    assert torch.equal(pooled, expected)
    assert _pool_region_tokens(
        tokens, source_grid=8, target_grid=8
    ).data_ptr() == tokens.data_ptr()
    with pytest.raises(ValueError, match="token grid sizes"):
        _parse_token_grid_sizes(metadata, "3,9")
