from pathlib import Path

import pytest

from radio_gs.data.view_split import ViewSplitError
from radio_gs.scripts.extract_official_crop_summary_teacher import (
    _generic_frame_records,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"test")


def test_source_rank_is_assigned_after_exact_target_exclusion(tmp_path: Path) -> None:
    _touch(tmp_path / "IMG_4028.JPG")
    _touch(tmp_path / "IMG_4026.JPG")
    _touch(tmp_path / "IMG_4027.JPG")

    records, excluded = _generic_frame_records(
        tmp_path,
        frame_id_mode="source_rank",
        excluded_image_stems=("IMG_4027",),
    )

    assert [(frame_id, path.stem) for frame_id, path in records] == [
        (0, "IMG_4026"),
        (1, "IMG_4028"),
    ]
    assert excluded == ["IMG_4027.JPG"]


def test_numeric_suffix_mode_preserves_legacy_ids(tmp_path: Path) -> None:
    _touch(tmp_path / "frame_20.jpg")
    _touch(tmp_path / "frame_5.jpg")

    records, excluded = _generic_frame_records(
        tmp_path,
        frame_id_mode="numeric_suffix",
    )

    assert [(frame_id, path.stem) for frame_id, path in records] == [
        (5, "frame_5"),
        (20, "frame_20"),
    ]
    assert excluded == []


def test_unknown_exact_exclusion_fails_closed(tmp_path: Path) -> None:
    _touch(tmp_path / "IMG_4026.JPG")

    with pytest.raises(ViewSplitError, match="not found by exact match"):
        _generic_frame_records(
            tmp_path,
            frame_id_mode="source_rank",
            excluded_image_stems=("IMG_9999",),
        )
