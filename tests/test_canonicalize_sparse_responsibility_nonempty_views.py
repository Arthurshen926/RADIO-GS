from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.canonicalize_sparse_responsibility_nonempty_views import (
    SCHEMA,
    VIEW_SCHEMA,
    canonicalize,
)
from radio_gs.utils.immutable_artifacts import sha256_file


def _empty_view(path: Path, *, entries: int = 0) -> str:
    torch.save(
        {
            "schema": VIEW_SCHEMA,
            "schema_version": 1,
            "frame_index": 20,
            "view_index": 1,
            "gaussian_ids": torch.arange(entries, dtype=torch.long),
            "pixel_ids": torch.arange(entries, dtype=torch.long),
            "base_weights": torch.ones(entries),
        },
        path,
    )
    return sha256_file(path)


def _authority(tmp_path: Path, *, empty_entries: int = 0, total_hits: int = 2):
    view_root = tmp_path / "views"
    view_root.mkdir()
    empty_path = view_root / "empty.pt"
    empty_sha = _empty_view(empty_path, entries=empty_entries)
    source = tmp_path / "authority.json"
    source.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "schema_version": 1,
                "frame_indices": [10, 20],
                "total_hits": total_hits,
                "views": [
                    {
                        "frame_index": 10,
                        "view_index": 0,
                        "num_hits": 2,
                        "relative_path": "views/nonempty.pt",
                        "sha256": "a" * 64,
                    },
                    {
                        "frame_index": 20,
                        "view_index": 1,
                        "num_hits": 0,
                        "relative_path": "views/empty.pt",
                        "sha256": empty_sha,
                    },
                ],
            }
        )
    )
    return source


def _args(source: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(source),
        expected_input_sha256=sha256_file(source),
        expected_removed_views=1,
        output=str(output),
    )


def test_removes_only_verified_empty_view_and_preserves_total_hits(tmp_path: Path):
    source = _authority(tmp_path)
    output = tmp_path / "canonical.json"
    report = canonicalize(_args(source, output))
    result = json.loads(output.read_text())
    assert report["retained_views"] == 1
    assert report["total_hits"] == 2
    assert result["total_hits"] == 2
    assert result["frame_indices"] == [10]
    assert [view["view_index"] for view in result["views"]] == [0]


def test_declared_zero_view_with_nonempty_payload_is_rejected(tmp_path: Path):
    source = _authority(tmp_path, empty_entries=1)
    with pytest.raises(ValueError, match="does not bind an empty view payload"):
        canonicalize(_args(source, tmp_path / "canonical.json"))


def test_retained_hits_must_equal_original_total(tmp_path: Path):
    source = _authority(tmp_path, total_hits=3)
    with pytest.raises(ValueError, match="retained nonempty hits differ"):
        canonicalize(_args(source, tmp_path / "canonical.json"))


def test_output_is_no_clobber(tmp_path: Path):
    source = _authority(tmp_path)
    output = tmp_path / "canonical.json"
    args = _args(source, output)
    canonicalize(args)
    with pytest.raises(FileExistsError):
        canonicalize(args)
