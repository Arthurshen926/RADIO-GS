import pytest

from radio_gs.benchmarks.scannet_uqis.ludvig_mapping_plan import (
    select_mapping_frame_ids,
)


def test_ludvig_mapping_sampler_is_deterministic_and_endpoint_preserving() -> None:
    frames = [f"{index:06d}" for index in range(500)]
    selected = select_mapping_frame_ids(frames, maximum_views=120)
    assert len(selected) == 120
    assert len(set(selected)) == 120
    assert selected[0] == frames[0]
    assert selected[-1] == frames[-1]
    assert selected == select_mapping_frame_ids(frames, maximum_views=120)


def test_ludvig_mapping_sampler_uses_every_legal_short_inventory() -> None:
    frames = ("000001", "000002", "000004")
    assert select_mapping_frame_ids(frames, maximum_views=120) == frames


def test_ludvig_mapping_sampler_rejects_unordered_or_duplicate_inventory() -> None:
    with pytest.raises(ValueError, match="unique, and sorted"):
        select_mapping_frame_ids(("000002", "000001"))
    with pytest.raises(ValueError, match="unique, and sorted"):
        select_mapping_frame_ids(("000001", "000001"))
