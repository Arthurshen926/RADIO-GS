from pathlib import Path

import pytest

from radio_gs.scripts.audit_vala_lerf3d_split import validate_vala_test_file


def test_vala_split_accepts_exact_extensionless_stems(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    path.write_text(
        "frame_00041\nframe_00105\nframe_00152\nframe_00195\n",
        encoding="utf-8",
    )
    result = validate_vala_test_file(path, "figurines")
    assert result["status"] == "exact_extensionless_vala_holdout"
    assert result["test_frame_count"] == 4


def test_vala_split_rejects_occam_style_suffixes(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    path.write_text(
        "frame_00041.jpg\nframe_00105.jpg\nframe_00152.jpg\nframe_00195.jpg\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extensionless"):
        validate_vala_test_file(path, "figurines")


def test_vala_split_rejects_missing_label_frame(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    path.write_text("frame_00041\nframe_00105\nframe_00152\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split mismatch"):
        validate_vala_test_file(path, "figurines")
