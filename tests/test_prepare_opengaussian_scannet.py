import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.scripts.prepare_opengaussian_scannet_scene import prepare_scene


def _write_tiny_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 255 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_scene_unpacks_color_and_writes_pose_lists(tmp_path):
    scene = "scene0000_00"
    src_scene = tmp_path / scene
    src_scene.mkdir()
    color_dir = src_scene / "color"
    color_dir.mkdir()
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(color_dir / "0.jpg")
    Image.fromarray(np.ones((4, 5, 3), dtype=np.uint8) * 255).save(color_dir / "1.jpg")

    transforms = {
        "w": 5,
        "h": 4,
        "fl_x": 4.0,
        "fl_y": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "frames": [
            {"file_path": "color/0", "transform_matrix": np.eye(4).tolist()},
            {"file_path": "color/1", "transform_matrix": np.eye(4).tolist()},
        ],
    }
    (src_scene / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")
    (src_scene / "transforms_test.json").write_text(json.dumps(transforms), encoding="utf-8")
    _write_tiny_ply(src_scene / "points3d.ply")
    _write_tiny_ply(src_scene / f"{scene}_vh_clean_2.labels.ply")

    zip_path = tmp_path / f"{scene}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(src_scene.rglob("*")):
            zf.write(path, path.relative_to(tmp_path))

    prepared = prepare_scene(
        scene=scene,
        data_root=tmp_path,
        output_root=tmp_path / "prepared",
        copy_mode="copy",
    )

    assert (prepared.scene_root / "color" / "0.jpg").exists()
    assert (prepared.scene_root / "transforms.json").exists()
    assert (prepared.scene_root / "traj_w_c.txt").exists()
    assert (prepared.scene_root / "splits" / "train_frames.txt").read_text().split() == ["0", "1"]
    assert prepared.num_train_frames == 2
    assert prepared.width == 5
    assert prepared.height == 4


def test_prepare_scene_drops_nonfinite_poses(tmp_path):
    scene = "scene0200_00"
    src_scene = tmp_path / scene
    src_scene.mkdir()
    color_dir = src_scene / "color"
    color_dir.mkdir()
    for frame_id in (0, 1):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(color_dir / f"{frame_id}.jpg")

    bad = np.eye(4).tolist()
    bad[0][0] = float("nan")
    transforms = {
        "w": 5,
        "h": 4,
        "fl_x": 4.0,
        "fl_y": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "frames": [
            {"file_path": "color/0", "transform_matrix": np.eye(4).tolist()},
            {"file_path": "color/1", "transform_matrix": bad},
        ],
    }
    (src_scene / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")
    (src_scene / "transforms_test.json").write_text(json.dumps(transforms), encoding="utf-8")
    _write_tiny_ply(src_scene / "points3d.ply")
    _write_tiny_ply(src_scene / f"{scene}_vh_clean_2.labels.ply")

    zip_path = tmp_path / f"{scene}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(src_scene.rglob("*")):
            zf.write(path, path.relative_to(tmp_path))

    prepared = prepare_scene(
        scene=scene,
        data_root=tmp_path,
        output_root=tmp_path / "prepared",
        copy_mode="copy",
    )

    assert prepared.num_train_frames == 1
    assert (prepared.scene_root / "splits" / "train_frames.txt").read_text().split() == ["0"]
    traj = np.loadtxt(prepared.scene_root / "traj_w_c.txt")
    assert np.isfinite(traj).all()


def test_prepare_scene_from_existing_dir_does_not_mutate_source_transforms(tmp_path):
    scene = "scene0400_00"
    src_scene = tmp_path / "source" / scene
    src_scene.mkdir(parents=True)
    color_dir = src_scene / "color"
    color_dir.mkdir()
    for frame_id in (0, 1):
        Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(color_dir / f"{frame_id}.jpg")

    bad = np.eye(4).tolist()
    bad[0][0] = float("nan")
    transforms = {
        "w": 5,
        "h": 4,
        "fl_x": 4.0,
        "fl_y": 4.0,
        "cx": 2.0,
        "cy": 2.0,
        "frames": [
            {"file_path": "color/0", "transform_matrix": np.eye(4).tolist()},
            {"file_path": "color/1", "transform_matrix": bad},
        ],
    }
    raw_text = json.dumps(transforms)
    (src_scene / "transforms_train.json").write_text(raw_text, encoding="utf-8")
    (src_scene / "transforms_test.json").write_text(raw_text, encoding="utf-8")
    _write_tiny_ply(src_scene / "points3d.ply")
    _write_tiny_ply(src_scene / f"{scene}_vh_clean_2.labels.ply")

    prepared = prepare_scene(
        scene=scene,
        data_root=tmp_path / "source",
        output_root=tmp_path / "prepared",
        copy_mode="symlink",
    )

    assert (src_scene / "transforms_train.json").read_text(encoding="utf-8") == raw_text
    prepared_train = json.loads(
        (prepared.scene_root / "transforms_train.json").read_text(encoding="utf-8")
    )
    assert len(prepared_train["frames"]) == 1
    assert prepared.num_train_frames == 1
