from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from radio_gs.scripts import train_colmap_gs as train_gs


def _rgb(path: Path, size: tuple[int, int] = (20, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path.resolve()


def test_geometry_uses_locked_camera_rgb_map_and_scales_intrinsics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_root = (tmp_path / "scene").resolve()
    scene_root.mkdir()
    prefixed = _rgb(tmp_path / "rgb/0_00001.png")
    exact = _rgb(tmp_path / "rgb/IMG_A.png")
    c2w_a = np.eye(4, dtype=np.float32)
    c2w_b = np.eye(4, dtype=np.float32)
    c2w_b[0, 3] = 2.0
    monkeypatch.setattr(
        train_gs,
        "_parse_colmap_sparse",
        lambda _root: {
            "file_paths": ["images/00001.jpg", "images/IMG_A.jpg"],
            "c2w_list": [c2w_a, c2w_b],
            "w": 100,
            "h": 50,
            "fl_x": 80.0,
            "fl_y": 90.0,
            "cx": 50.0,
            "cy": 25.0,
            "calibration_source": "unit-test",
        },
    )
    mapping = {
        "schema_version": 1,
        "scene_id": "toy",
        "scene_root": str(scene_root),
        "records": [
            {
                "rgb_camera_name": "0_00001",
                "rgb_path": str(prefixed),
                "colmap_camera_name": "00001",
                "colmap_file_path": "images/00001.jpg",
                "match_rule": "strip_official_0_or_1_split_prefix_then_exact_stem",
            },
            {
                "rgb_camera_name": "IMG_A",
                "rgb_path": str(exact),
                "colmap_camera_name": "IMG_A",
                "colmap_file_path": "images/IMG_A.jpg",
                "match_rule": "exact_case_sensitive_basename_stem",
            },
        ],
    }
    mapping_path = tmp_path / "camera_map.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    images, w2cs, K, width, height, _extent, metadata = train_gs.load_scene(
        str(scene_root),
        torch.device("cpu"),
        image_map_json=str(mapping_path),
        return_view_metadata=True,
    )

    assert len(images) == len(w2cs) == 2
    assert all(image.dtype == torch.uint8 for image in images)
    assert images[0][0, 0].tolist() == [10, 20, 30]
    assert (width, height) == (20, 10)
    assert torch.allclose(
        K,
        torch.tensor([[16.0, 0.0, 10.0], [0.0, 18.0, 5.0], [0.0, 0.0, 1.0]]),
    )
    assert metadata["selection"] == "locked_explicit_colmap_camera_to_rgb_path_map"
    assert metadata["intrinsics_scale_xy"] == [0.2, 0.2]
    assert metadata["training_image_paths"] == [str(prefixed), str(exact)]
    assert metadata["image_map"]["record_count"] == 2
    assert metadata["image_map"]["nearest_or_fuzzy_matching"] == "forbidden"


def test_geometry_rejects_image_directory_plus_locked_map(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either image_dir or image_map_json"):
        train_gs.load_scene(
            str(tmp_path),
            torch.device("cpu"),
            image_dir=str(tmp_path),
            image_map_json=str(tmp_path / "map.json"),
        )
