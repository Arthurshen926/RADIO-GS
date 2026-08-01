import json
from pathlib import Path

from radio_gs.scripts.run_vala_scannet_baseline import _stage_scene


def _write_transforms(path: Path, frame_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "w": 640,
                "h": 480,
                "fl_x": 500.0,
                "fl_y": 500.0,
                "cx": 320.0,
                "cy": 240.0,
                "frames": [
                    {
                        "file_path": f"color/{frame_id}",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                    for frame_id in frame_ids
                ],
            }
        ),
        encoding="utf-8",
    )


def test_stage_scene_preserves_train_frames_and_can_limit_loader_only_test(tmp_path: Path):
    scene = "scene0000_00"
    data_root = tmp_path / "data"
    model_root = tmp_path / "models"
    source = data_root / scene
    model = model_root / scene / "og_rgb_3dgs"
    (source / "color").mkdir(parents=True)
    (source / "language_features").mkdir()
    (model / "point_cloud" / "iteration_30000").mkdir(parents=True)
    _write_transforms(source / "transforms_train.json", ["0", "20", "40"])
    _write_transforms(source / "transforms_test.json", ["0", "20"])
    (source / f"{scene}_vh_clean_2.labels.ply").touch()
    (model / "point_cloud" / "iteration_30000" / "point_cloud.ply").touch()

    paths = _stage_scene(
        scene,
        data_root=data_root,
        model_root=model_root,
        staging_root=tmp_path / "staged",
        test_loader_limit=1,
    )

    train = json.loads((paths.staged / "transforms_train.json").read_text())
    test = json.loads((paths.staged / "transforms_test.json").read_text())
    manifest = json.loads((paths.staged / "staging_manifest.json").read_text())
    assert [Path(frame["file_path"]).stem for frame in train["frames"]] == ["0", "20", "40"]
    assert [Path(frame["file_path"]).stem for frame in test["frames"]] == ["0"]
    assert all("language_features_path" in frame for frame in train["frames"])
    assert manifest["splits"]["train"]["source_num_frames"] == 3
    assert manifest["splits"]["train"]["staged_num_frames"] == 3
    assert manifest["splits"]["test"]["source_num_frames"] == 2
    assert manifest["splits"]["test"]["staged_num_frames"] == 1
