from pathlib import Path

import pytest

from radio_gs.data.benchmark_paths import list_feature_paths


def test_explicit_feature_ids_are_complete_and_keep_frozen_order(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    for frame_id in (0, 1, 2):
        (backbone / f"rgb_{frame_id}.pt").write_bytes(b"fixture")

    paths = list_feature_paths(tmp_path, frame_ids=[2, 0])

    assert [path.name for path in paths] == ["rgb_2.pt", "rgb_0.pt"]


def test_missing_explicit_feature_id_is_fatal(tmp_path: Path) -> None:
    (tmp_path / "rgb_0.pt").write_bytes(b"fixture")

    with pytest.raises(FileNotFoundError, match=r"Missing requested.*\[1\]"):
        list_feature_paths(tmp_path, frame_ids=[0, 1])


def test_duplicate_feature_index_is_fatal(tmp_path: Path) -> None:
    (tmp_path / "rgb_1.pt").write_bytes(b"fixture")
    (tmp_path / "rgb_01.pt").write_bytes(b"fixture")

    with pytest.raises(ValueError, match="Duplicate feature frame id 1"):
        list_feature_paths(tmp_path)

