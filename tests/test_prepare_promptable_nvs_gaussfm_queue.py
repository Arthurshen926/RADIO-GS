from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from radio_gs.config import load_config
from radio_gs.scripts import prepare_promptable_nvs_gaussfm_queue as queue
from radio_gs.scripts import render_promptable_nvs_features as render
from radio_gs.scripts.extract_radio_features import (
    _collect_image_paths,
    _saved_frame_indices,
)


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(17, 23, 31)).save(path)
    return path


def _toy_inputs(tmp_path: Path, *, benchmark: str) -> tuple[Path, Path, Path, dict]:
    scene_root = tmp_path / "scene"
    rgb_dir = scene_root / "images"
    aux = _image(rgb_dir / "aux.jpg")
    reference = _image(rgb_dir / "reference.jpg")
    target = _image(rgb_dir / "target.jpg")
    sparse = scene_root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(b"fixture")
    (sparse / "images.bin").write_bytes(b"fixture")

    if benchmark == "nvos":
        training_paths = (aux, reference)
        prompt_type = "fixed_positive_negative_scribbles"
    else:
        training_paths = (aux, reference, target)
        prompt_type = "single_reference_binary_mask"
    raw_scene = {
        "scene_id": "toy",
        "rgb_directory": str(rgb_dir),
        "prompt_frame_ids": ["ref"],
        "calibration_frame_ids": [],
        "evaluation_frame_ids": ["eval"],
        "frames": [
            {"frame_id": "ref", "camera_name": "reference"},
            {"frame_id": "eval", "camera_name": "target"},
        ],
        "training_frames": [
            {
                "frame_id": path.stem,
                "camera_name": path.stem,
                "rgb_path": str(path),
            }
            for path in training_paths
        ],
    }
    manifest = {
        "schema_version": 1,
        "benchmark": benchmark,
        "protocol": {
            "prompt_type": prompt_type,
            "threshold": {"mode": "fixed", "value": 0.0},
            "aggregation": "per_frame_then_per_scene_then_dataset_scene_macro",
            "metrics": ["foreground_iou", "pixel_accuracy"],
            "prediction_representation": "continuous_margin",
            "threshold_comparison": "greater_or_equal",
        },
        "scenes": [raw_scene],
        "protocol_hash": "a" * 64,
    }
    manifest_path = tmp_path / f"{benchmark}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "architecture": "explicit",
                "latent_dim": 8,
                "codec_type": "direct",
                "radio_repo": str(tmp_path / "RADIO"),
                "radio_version": "toy-radio",
                # Deliberately unsafe base values must be overridden.
                "use_refiner": True,
                "refiner_rgb_guide": True,
                "self_guided": True,
                "rgb_dir": "/would/leak/rgb",
                "seg_loss_weight": 1.0,
                "samclip_mask_loss_weight": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "RADIO").mkdir()
    return manifest_path, base_config, scene_root, manifest


def _normalized(manifest: dict) -> dict:
    scene = manifest["scenes"][0]
    return {
        "protocol_hash": manifest["protocol_hash"],
        "protocol": manifest["protocol"],
        "scenes": [
            {
                **scene,
                "frames": {frame["frame_id"]: frame for frame in scene["frames"]},
            }
        ],
    }


def _patch_protocol_and_colmap(monkeypatch, manifest: dict, scene_root: Path) -> None:
    monkeypatch.setattr(queue, "validate_dataset_manifest", lambda *_args, **_kwargs: _normalized(manifest))
    parsed = {
        "file_paths": ["images/aux.jpg", "images/reference.jpg", "images/target.jpg"],
        "c2w_list": [
            np.eye(4, dtype=np.float32),
            np.diag([1, 1, 1, 1]).astype(np.float32),
            np.array(
                [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            ),
        ],
        "w": 64,
        "h": 48,
        "fl_x": 50.0,
        "fl_y": 51.0,
        "cx": 32.0,
        "cy": 24.0,
    }
    monkeypatch.setattr(queue, "_parse_colmap_sparse", lambda root: parsed)


def test_nvos_queue_excludes_target_everywhere_and_does_not_run_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, base_config, scene_root, manifest = _toy_inputs(
        tmp_path, benchmark="nvos"
    )
    _patch_protocol_and_colmap(monkeypatch, manifest, scene_root)

    plan = queue.prepare_queue(
        manifest_path,
        tmp_path / "queue",
        base_config=base_config,
        scene_root_map={"toy": str(scene_root)},
        python_executable="python",
    )

    assert plan["status"] == "prepared_not_run"
    assert plan["execution"]["commands_executed_during_prepare"] == 0
    assert plan["execution"]["gpu_used_during_prepare"] is False
    assert plan["track"]["id"] == queue.NVOS_TRACK
    scene = plan["scenes"][0]
    assert scene["excluded_image_stems"] == ["target"]
    assert scene["safety"]["target_rgb_excluded_from_geometry_rgb_loss"] is True
    assert scene["safety"]["sparse_points_with_target_observations_removed"] is True
    assert scene["safety"]["upstream_camera_calibration_shared_exception"] is True
    assert plan["protocol_guards"]["fully_target_pixel_independent_camera_calibration"] is False
    assert "upstream joint reconstruction" in plan["protocol_guards"]["nvos_geometry_provenance"]
    assert "--exclude-image-stems-file" in scene["commands"]["geometry"]
    assert "--image-map-json" in scene["commands"]["geometry"]
    assert "--exclude-image-stems-file" in scene["commands"]["feature_extraction"]
    assert "--camera-map" in scene["commands"]["render"]
    mapping = json.loads(
        (tmp_path / "queue/scenes/toy/feature_pose_mapping.json").read_text()
    )
    assert {item["camera_name"] for item in mapping["records"]} == {"aux", "reference"}
    assert all("target" not in item["rgb_path"] for item in mapping["records"])
    camera_map = json.loads(
        (tmp_path / "queue/scenes/toy/rgb_to_colmap_camera_mapping.json").read_text()
    )
    assert camera_map["nearest_or_fuzzy_matching"] == "forbidden"
    assert camera_map["colmap_camera_to_rgb_path"]["target"].endswith("/target.jpg")
    assert {
        (item["rgb_camera_name"], item["colmap_camera_name"])
        for item in camera_map["records"]
    } == {("aux", "aux"), ("reference", "reference"), ("target", "target")}

    generated = load_config(scene["config"])
    assert generated.use_refiner is False
    assert generated.refiner_rgb_guide is False
    assert generated.self_guided is False
    assert generated.rgb_dir == ""
    assert generated.val_rgb_dir == ""
    assert generated.seg_loss_weight == 0.0
    assert generated.samclip_mask_loss_weight == 0.0
    assert scene["artifacts"]["feature_field_checkpoint"]["sha256"] == queue.PENDING_SHA256
    runner = Path(plan["runner_path"]).read_text(encoding="utf-8")
    assert "ALLOW_GPU" in runner
    assert "Refusing to launch" in runner


def test_spin_full_mask_queue_is_not_labeled_saga_same_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, base_config, scene_root, manifest = _toy_inputs(
        tmp_path, benchmark="spin_nerf"
    )
    _patch_protocol_and_colmap(monkeypatch, manifest, scene_root)

    plan = queue.prepare_queue(
        manifest_path,
        tmp_path / "queue",
        base_config=base_config,
        scene_root_map={"toy": str(scene_root)},
        python_executable="python",
    )

    assert plan["track"]["id"] == queue.SPIN_FULL_MASK_TRACK
    assert plan["track"]["saga_same_prompt_main_table_eligible"] is False
    assert "2D point prompts" in plan["track"]["comparison_note"]
    assert "--exclude-image-stems-file" not in plan["scenes"][0]["commands"]["geometry"]
    mapping = json.loads(
        (tmp_path / "queue/scenes/toy/feature_pose_mapping.json").read_text()
    )
    assert {item["camera_name"] for item in mapping["records"]} == {
        "aux",
        "reference",
        "target",
    }


def test_prepare_rejects_any_threshold_calibration_before_commands(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path, base_config, scene_root, manifest = _toy_inputs(
        tmp_path, benchmark="nvos"
    )
    manifest["protocol"]["threshold"] = {
        "mode": "calibrated",
        "source": "target",
        "scope": "scene",
        "candidates": [0.0],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _patch_protocol_and_colmap(monkeypatch, manifest, scene_root)

    with pytest.raises(queue.QueuePreparationError, match="target/test calibration"):
        queue.prepare_queue(
            manifest_path,
            tmp_path / "queue",
            base_config=base_config,
            scene_root_map={"toy": str(scene_root)},
        )


def test_renderer_rejects_rgb_refiner_and_resolves_exact_camera(monkeypatch) -> None:
    with pytest.raises(render.PromptableRenderError, match="use_refiner"):
        render.validate_feature_only_config(
            SimpleNamespace(
                use_refiner=True,
                refiner_rgb_guide=False,
                self_guided=False,
                train_sh=False,
                rgb_loss_weight=0.0,
                rgb_dir="",
                val_rgb_dir="",
            )
        )

    manifest = {
        "scenes": [
            {
                "scene_id": "toy",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["eval"],
                "frames": [
                    {"frame_id": "ref", "camera_name": "reference"},
                    {"frame_id": "eval", "camera_name": "target"},
                ],
            }
        ]
    }
    normalized = {
        "protocol_hash": "a" * 64,
        "scenes": [
            {
                "scene_id": "toy",
                "prompt_frame_ids": ["ref"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["eval"],
                "frames": {},
            }
        ],
    }
    monkeypatch.setattr(render, "validate_dataset_manifest", lambda *_a, **_k: normalized)
    monkeypatch.setattr(
        render,
        "_parse_colmap_sparse",
        lambda _root: {
            "file_paths": ["images/reference.jpg", "images/target.jpg"],
            "c2w_list": [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        },
    )

    camera_mapping = {
        "schema_version": 1,
        "scene_id": "toy",
        "records": [
            {
                "rgb_camera_name": name,
                "rgb_path": f"/rgb/{name}.png",
                "colmap_camera_name": name,
                "colmap_file_path": f"images/{name}.jpg",
                "match_rule": "exact_case_sensitive_basename_stem",
            }
            for name in ("reference", "target")
        ],
    }
    views = render.resolve_protocol_views(
        manifest,
        scene_id="toy",
        scene_root="/unused",
        camera_mapping=camera_mapping,
    )
    assert [(view["camera_name"], view["role"]) for view in views] == [
        ("reference", "prompt"),
        ("target", "evaluation"),
    ]


def test_feature_extractor_source_rank_prevents_duplicate_numeric_suffix_overwrite() -> None:
    paths = [
        Path("DJI_20200223_163023_427.jpg"),
        Path("DJI_20200223_163105_427.jpg"),
    ]
    with pytest.raises(ValueError, match=r"collision at rgb_427\.pt"):
        _saved_frame_indices(paths, mode="auto")
    assert _saved_frame_indices(paths, mode="source_rank") == [0, 1]


def test_source_rank_numeric_ties_use_exact_filename_tiebreaker(tmp_path: Path) -> None:
    second = _image(tmp_path / "DJI_20200223_163105_427.jpg")
    first = _image(tmp_path / "DJI_20200223_163023_427.jpg")

    paths, mode = _collect_image_paths(str(tmp_path))

    assert mode == "numeric_then_exact_filename"
    assert paths == [first, second]


def test_real_style_rgb_to_colmap_mapping_has_only_three_locked_rules(tmp_path: Path) -> None:
    exact = _image(tmp_path / "rgb/IMG_7238.png")
    prefixed = _image(tmp_path / "rgb/0_00001.png")
    indexed = _image(tmp_path / "rgb/image002.png")
    records = render.build_rgb_to_colmap_mapping(
        [exact, prefixed, indexed],
        [
            "images/00001.png",
            "images/IMG_4026.JPG",
            "images/IMG_4027.JPG",
            "images/IMG_7238.JPG",
        ],
        scene_id="real_style",
    )
    by_rgb = {record["rgb_camera_name"]: record for record in records}
    assert by_rgb["IMG_7238"]["colmap_camera_name"] == "IMG_7238"
    assert by_rgb["IMG_7238"]["match_rule"].startswith("exact_")
    assert by_rgb["0_00001"]["colmap_camera_name"] == "00001"
    assert by_rgb["0_00001"]["match_rule"].startswith("strip_official_")
    # Lexicographic order is 00001, IMG_4026, IMG_4027, IMG_7238.
    assert by_rgb["image002"]["colmap_camera_name"] == "IMG_4027"
    assert by_rgb["image002"]["canonical_index"] == 2

    unknown = _image(tmp_path / "rgb/almost_IMG_7238.png")
    with pytest.raises(render.PromptableRenderError, match="No nearest-name guessing"):
        render.build_rgb_to_colmap_mapping(
            [unknown],
            ["images/IMG_7238.JPG"],
            scene_id="no_guess",
        )
