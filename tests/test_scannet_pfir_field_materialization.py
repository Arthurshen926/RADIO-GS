import json
from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfir.preparation.materialize_field_scenes import (
    materialize,
)
from radio_gs.benchmarks.scannet_pfir.protocol import canonical_json_sha256
from radio_gs.data.benchmark_paths import load_w2c_from_pose_dir
from radio_gs.scripts.train_scannet_gs import load_scannet_data


def test_materialization_excludes_query_assets_and_loads_padded_frames(tmp_path):
    dense = tmp_path / "dense" / "scene0001_00"
    for name in ("color", "depth", "pose", "instance", "label"):
        (dense / name).mkdir(parents=True)
    for frame_id in ("000000", "000020", "000040"):
        Image.new("RGB", (4, 3)).save(dense / "color" / f"{frame_id}.jpg")
        Image.fromarray(np.full((3, 4), 1000, np.uint16)).save(
            dense / "depth" / f"{frame_id}.png"
        )
        np.savetxt(dense / "pose" / f"{frame_id}.txt", np.eye(4))
        Image.fromarray(np.ones((3, 4), np.uint16)).save(
            dense / "instance" / f"{frame_id}.png"
        )
    intrinsic = np.eye(4)
    intrinsic[0, 0] = intrinsic[1, 1] = 2.0
    intrinsic[0, 2], intrinsic[1, 2] = 1.5, 1.0
    for name in (
        "intrinsics_color.txt",
        "intrinsics_depth.txt",
        "extrinsics_color.txt",
        "extrinsics_depth.txt",
    ):
        np.savetxt(dense / name, intrinsic)

    field_ids = ["000000", "000040"]
    record = {
        "scene_id": "scene0001_00",
        "field_frame_ids": field_ids,
        "query_exclusion_frames": ["000020"],
        "field_frame_manifest_sha256": canonical_json_sha256(field_ids),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"benchmark_version": "test", "queries": [record]}),
        encoding="utf-8",
    )
    output = tmp_path / "field"
    report = materialize(manifest, tmp_path / "dense", output)

    scene = output / "scene0001_00"
    assert report["valid"]
    assert sorted(path.stem for path in (scene / "color").iterdir()) == field_ids
    assert not (scene / "instance").exists()
    assert not (scene / "label").exists()
    assert (scene / "intrinsic" / "intrinsic_depth.txt").is_symlink()
    images, depths, poses, frame_ids = load_scannet_data(scene, 1, None)
    assert len(images) == len(depths) == len(poses) == 2
    assert frame_ids == [0, 40]
    w2c = load_w2c_from_pose_dir(scene / "pose", frame_ids)
    assert w2c.shape == (2, 4, 4)
