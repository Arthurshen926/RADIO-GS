from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.data.view_split import (
    ViewSplitError,
    load_excluded_image_stems,
    select_image_indices,
)


def test_json_and_explicit_exclusions_are_merged_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "excluded.json"
    source.write_text(
        json.dumps({"excluded_image_stems": ["TARGET_01", "TARGET_02"]}),
        encoding="utf-8",
    )

    assert load_excluded_image_stems(["TARGET_00", "TARGET_01"], source) == (
        "TARGET_00",
        "TARGET_01",
        "TARGET_02",
    )


def test_exact_stem_filter_returns_parallel_indices_and_names() -> None:
    paths = [Path("images/IMG_0001.JPG"), Path("images/IMG_0002.JPG"), Path("images/IMG_0003.JPG")]

    retained, excluded = select_image_indices(
        paths,
        ["IMG_0002"],
        min_remaining=2,
    )

    assert retained == [0, 2]
    assert excluded == ["IMG_0002.JPG"]


def test_unknown_exclusion_fails_closed() -> None:
    with pytest.raises(ViewSplitError, match="not found by exact match"):
        select_image_indices([Path("IMG_0001.JPG")], ["IMG_0010"])


def test_duplicate_source_stems_and_over_exclusion_fail_closed() -> None:
    with pytest.raises(ViewSplitError, match="Duplicate input image stem"):
        select_image_indices([Path("a/one.jpg"), Path("b/one.png")], [])
    with pytest.raises(ViewSplitError, match="at least 2"):
        select_image_indices(
            [Path("one.jpg"), Path("two.jpg")],
            ["one"],
            min_remaining=2,
        )


def test_stems_must_not_hide_paths_or_extensions() -> None:
    with pytest.raises(ViewSplitError, match="must not contain a path"):
        load_excluded_image_stems(["images/target"])
    with pytest.raises(ViewSplitError, match="without an extension"):
        load_excluded_image_stems(["target.jpg"])
