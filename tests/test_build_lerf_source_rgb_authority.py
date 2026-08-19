import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.scripts.build_lerf_source_rgb_authority import (
    build_authority,
    uniform_indices,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    frames = []
    for index in range(1, 11):
        path = images / f"frame_{index:05d}.jpg"
        path.write_bytes(f"image-{index}".encode())
        frames.append(
            {
                "frame_idx": index,
                "source_file": path.name,
                "source_sha256": _sha(path),
                "source_rank": index - 1,
                "saved_stem": f"rgb_{index}",
            }
        )
    manifest = {
        "scene": "toy",
        "image_dir": str(images),
        "frames": frames,
        "excluded_image_names": [],
        "excluded_image_stems": [],
    }
    manifest_path = tmp_path / "frames.json"
    manifest_path.write_text(json.dumps(manifest))
    indices = [1, 3, 5, 7, 9]
    exact = {
        "schema": "radio_gs.sparse_exact_marginal_responsibility_authority.v1",
        "schema_version": 1,
        "frame_indices": indices,
        "views": [{"frame_index": index} for index in indices],
        "metadata": {
            "assignment_mode": "exact_front_to_back_sparse_marginal",
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
    }
    exact_path = tmp_path / "exact.json"
    exact_path.write_text(json.dumps(exact))
    return manifest_path, exact_path


def test_uniform_indices_preserve_edges_and_count():
    assert uniform_indices(120, 4) == (0, 40, 79, 119)
    assert uniform_indices(5, 5) == (0, 1, 2, 3, 4)
    assert uniform_indices(5, 1) == (2,)


def test_build_authority_intersects_and_subsamples(tmp_path):
    manifest, exact = _fixture(tmp_path)
    payload = build_authority(
        scene="toy",
        frame_manifest_path=manifest,
        exact_mpr_authority_path=exact,
        maximum_images=3,
    )
    assert [record["image_id"] for record in payload["images"]] == [
        "frame_00001",
        "frame_00005",
        "frame_00009",
    ]
    assert payload["information_policy"]["target_or_evaluation_rgb_used"] is False


def test_build_authority_rejects_forbidden_exact_mpr(tmp_path):
    manifest, exact = _fixture(tmp_path)
    payload = json.loads(exact.read_text())
    payload["metadata"]["benchmark_masks_opened"] = True
    exact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="forbidden"):
        build_authority(
            scene="toy",
            frame_manifest_path=manifest,
            exact_mpr_authority_path=exact,
            maximum_images=3,
        )
