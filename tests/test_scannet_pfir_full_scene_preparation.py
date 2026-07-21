import zipfile
from pathlib import Path

from radio_gs.benchmarks.scannet_pfir.preparation.prepare_full_scene import (
    remap_raw_labels,
    zip_frame_members,
)
import numpy as np


def test_projection_zip_frame_ids_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "instances.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("instance-filt/frame-000020.png", b"x")
        archive.writestr("instance-filt/100.png", b"y")
    with zipfile.ZipFile(path) as archive:
        assert zip_frame_members(archive) == {
            20: "instance-filt/frame-000020.png",
            100: "instance-filt/100.png",
        }


def test_projection_zip_rejects_duplicate_numeric_frames(tmp_path: Path) -> None:
    path = tmp_path / "instances.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/20.png", b"x")
        archive.writestr("b/frame-000020.png", b"y")
    with zipfile.ZipFile(path) as archive:
        try:
            zip_frame_members(archive)
        except ValueError as error:
            assert "duplicate projected frame 20" in str(error)
        else:
            raise AssertionError("duplicate numeric frame should fail")


def test_raw_scannet_labels_are_remapped_to_nyu40() -> None:
    raw = np.array([[0, 2, 1303, 9999]], dtype=np.int64)
    mapped = remap_raw_labels(raw, {2: 5, 1303: 40})
    np.testing.assert_array_equal(mapped, [[0, 5, 40, 0]])
