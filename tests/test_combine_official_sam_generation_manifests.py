from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.scripts.combine_official_sam_generation_manifests import combine


def _shard(root: Path, name: str, stems: list[str]) -> Path:
    reports = []
    for stem in stems:
        image = root / f"{stem}.jpg"
        output = root / f"{stem}.pt"
        image.write_bytes(b"rgb")
        output.write_bytes(b"mask")
        reports.append({"image": str(image), "output": str(output), "masks": 1})
    path = root / name
    path.write_text(
        json.dumps({"output_root": str(root.resolve()), "images": reports}),
        encoding="utf-8",
    )
    return path


def test_combine_orders_disjoint_numeric_shards(tmp_path: Path) -> None:
    left = _shard(tmp_path, "left.json", ["20", "3"])
    right = _shard(tmp_path, "right.json", ["11"])
    payload = combine([left, right], output_root=tmp_path)
    assert [Path(value["image"]).stem for value in payload["images"]] == ["3", "11", "20"]


def test_combine_rejects_duplicate_frame_identity(tmp_path: Path) -> None:
    left = _shard(tmp_path, "left.json", ["3"])
    right = tmp_path / "right.json"
    right.write_text(left.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        combine([left, right], output_root=tmp_path)
