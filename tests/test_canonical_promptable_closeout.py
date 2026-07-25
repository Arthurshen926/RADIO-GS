from pathlib import Path

import pytest

from radio_gs.scripts.run_canonical_promptable_closeout import _ply_vertex_count


def test_ply_vertex_count_reads_binary_header_without_payload(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 123\nproperty float x\nend_header\n"
        b"\x00\x01\x02"
    )

    assert _ply_vertex_count(path) == 123


def test_ply_vertex_count_fails_closed_without_vertex_declaration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene.ply"
    path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="ascii")

    with pytest.raises(ValueError, match="lacks an element vertex"):
        _ply_vertex_count(path)
