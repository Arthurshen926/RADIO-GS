from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radio_gs.data.promptable_nvs_manifest import (
    ManifestError,
    NVOS_TASKS,
    SPIN_SCENE_FOLDERS,
    SPIN_DIAGNOSTIC_SCENES,
    SPIN_SCENES,
    build_nvos_manifest,
    build_spin_manifest,
    validate_manifest,
    write_manifest,
)
from radio_gs.evaluation.promptable_segmentation import (
    compute_protocol_hash,
    validate_manifest as validate_evaluation_manifest,
)
from radio_gs.scripts import eval_promptable_nvs_segmentation as eval_cli
from radio_gs.scripts import predict_promptable_nvs_feature_readout as predict_cli


def _image(path: Path, value: int = 255) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((3, 4), value, dtype=np.uint8)).save(path)
    return path


def _make_nvos(tmp_path: Path) -> tuple[Path, Path]:
    annotations = tmp_path / "nvos_annotations"
    rgb_root = tmp_path / "nvos_rgb"
    for index, task_id in enumerate(NVOS_TASKS):
        base = "horns" if task_id.startswith("horns_") else task_id
        reference_id = f"REF_{index:02d}"
        target_id = f"TARGET_{index:02d}"
        _image(annotations / "masks" / task_id / f"{target_id}.JPG", 80)
        _image(annotations / "masks" / task_id / f"{target_id}_mask.png")
        _image(annotations / "reference_image" / task_id / f"{reference_id}.JPG", 90)
        _image(annotations / "scribbles" / task_id / f"pos_0_{base}.png")
        _image(annotations / "scribbles" / task_id / f"neg_0_{base}.png")
        _image(annotations / "scribbles" / task_id / f"vis_0_{base}.png")
        scene_dir = rgb_root / f"{base}_undistort" / "images"
        _image(scene_dir / f"AUX_{index:02d}.JPG", 70)
        _image(scene_dir / f"{reference_id}.JPG", 90)
        _image(scene_dir / f"{target_id}.JPG", 80)
    return annotations, rgb_root


def _spin_frame_ids(scene_id: str) -> tuple[list[str], list[str]]:
    if scene_id in {"fern", "fortress", "leaves"}:
        return ["image000", "image001"], ["CAM000", "CAM001"]
    if scene_id == "orchids":
        # Explicitly exercise the official sparse-index rule. image015 must
        # map to RGB index 15, not annotation-list position 1.
        return ["image000", "image015"], [f"CAM{index:03d}" for index in range(16)]
    if scene_id == "truck":
        return ["0_000001", "1_000002"], ["000001", "000002"]
    if scene_id == "lego":
        return ["0_00001", "1_00002"], ["00001", "00002"]
    return [f"{scene_id}_000", f"{scene_id}_001"], [f"{scene_id}_000", f"{scene_id}_001"]


def _make_spin(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    annotations = tmp_path / "spin_annotations"
    rgb_map: dict[str, str] = {}
    for scene_id, folder in SPIN_SCENE_FOLDERS:
        annotation_ids, camera_ids = _spin_frame_ids(scene_id)
        for index, frame_id in enumerate(annotation_ids):
            _image(annotations / folder / f"{frame_id}.png", 255 - index)
            _image(annotations / folder / f"{frame_id}_pseudo.png", 100)
            _image(annotations / folder / f"{frame_id}_cutout.png", 100)
        rgb_dir = tmp_path / "spin_rgb" / scene_id
        for index, camera_id in enumerate(camera_ids):
            _image(rgb_dir / f"{camera_id}.jpg", 40 + index)
        rgb_map[scene_id] = str(rgb_dir)
    return annotations, rgb_map


def test_nvos_manifest_matches_evaluator_schema_and_excludes_targets(tmp_path: Path) -> None:
    annotations, rgb_root = _make_nvos(tmp_path)
    manifest = build_nvos_manifest(annotations, rgb_root)

    assert manifest["schema_version"] == 1
    assert manifest["protocol"]["threshold"] == {"mode": "fixed", "value": 0.0}
    assert set(manifest) >= {"schema_version", "protocol", "scenes", "protocol_hash"}
    assert "tasks" not in manifest
    assert len(manifest["scenes"]) == 8
    assert validate_evaluation_manifest(manifest)["protocol_hash"] == manifest["protocol_hash"]

    for scene in manifest["scenes"]:
        by_id = {frame["frame_id"]: frame for frame in scene["frames"]}
        prompt_id = scene["prompt_frame_ids"][0]
        target_id = scene["evaluation_frame_ids"][0]
        assert by_id[prompt_id]["ground_truth"] is None
        assert by_id[prompt_id]["feature_path"] is None
        assert by_id[target_id]["ground_truth"].endswith("_mask.png")
        assert scene["prompt"]["type"] == "positive_negative_scribbles"
        assert target_id not in {frame["frame_id"] for frame in scene["training_frames"]}


def test_nvos_missing_annotation_camera_fails_closed(tmp_path: Path) -> None:
    annotations, rgb_root = _make_nvos(tmp_path)
    (rgb_root / "fern_undistort" / "images" / "TARGET_00.JPG").unlink()
    with pytest.raises(ManifestError, match="target=TARGET_00"):
        build_nvos_manifest(annotations, rgb_root)


def test_spin_manifest_uses_sparse_canonical_index_and_reference_not_scored(
    tmp_path: Path,
) -> None:
    annotations, rgb_map = _make_spin(tmp_path)
    manifest = build_spin_manifest(
        annotations,
        rgb_map,
        enforce_official_counts=False,
    )

    assert [scene["scene_id"] for scene in manifest["scenes"]] == list(SPIN_SCENES)
    assert validate_evaluation_manifest(manifest)["protocol_hash"] == manifest["protocol_hash"]
    orchids = next(scene for scene in manifest["scenes"] if scene["scene_id"] == "orchids")
    by_id = {frame["frame_id"]: frame for frame in orchids["frames"]}
    assert by_id["image015"]["canonical_index"] == 15
    assert by_id["image015"]["rgb_sorted_index"] == 15
    assert by_id["image015"]["camera_name"] == "CAM015"
    assert orchids["prompt_frame_ids"] == ["image000"]
    assert "image000" not in orchids["evaluation_frame_ids"]
    assert orchids["prompt"]["mask_path"] == by_id["image000"]["ground_truth"]
    assert all(frame["feature_path"] is None for frame in orchids["frames"])

    truck = next(scene for scene in manifest["scenes"] if scene["scene_id"] == "truck")
    truck_frames = {frame["frame_id"]: frame for frame in truck["frames"]}
    assert truck_frames["0_000001"]["camera_name"] == "000001"
    assert truck_frames["1_000002"]["camera_name"] == "000002"


def test_spin_requires_explicit_complete_rgb_map(tmp_path: Path) -> None:
    annotations, rgb_map = _make_spin(tmp_path)
    del rgb_map["lego"]
    with pytest.raises(ManifestError, match=r"missing=\['lego'\]"):
        build_spin_manifest(annotations, rgb_map, enforce_official_counts=False)


def test_spin_missing_fork_can_only_build_labelled_nine_scene_diagnostic(tmp_path: Path) -> None:
    annotations, rgb_map = _make_spin(tmp_path)
    del rgb_map["fork"]
    manifest = build_spin_manifest(
        annotations,
        rgb_map,
        enforce_official_counts=False,
        diagnostic_missing_fork=True,
    )
    assert manifest["benchmark"] == "spin_nerf_diagnostic_9scene"
    assert [scene["scene_id"] for scene in manifest["scenes"]] == list(
        SPIN_DIAGNOSTIC_SCENES
    )
    assert manifest["protocol"]["formal_10scene_eligible"] is False
    assert manifest["protocol"]["missing_scenes"] == ["fork"]
    validate_manifest(manifest, check_files=True)


def test_spin_exact_mapping_failure_names_scene_and_mask(tmp_path: Path) -> None:
    annotations, rgb_map = _make_spin(tmp_path)
    (Path(rgb_map["fork"]) / "fork_001.jpg").rename(
        Path(rgb_map["fork"]) / "wrong_camera.jpg"
    )
    with pytest.raises(ManifestError, match=r"fork/fork_001\.png"):
        build_spin_manifest(annotations, rgb_map, enforce_official_counts=False)


def test_write_manifest_round_trips_and_remains_evaluator_valid(tmp_path: Path) -> None:
    annotations, rgb_root = _make_nvos(tmp_path)
    manifest = build_nvos_manifest(annotations, rgb_root)
    destination = write_manifest(manifest, tmp_path / "out" / "nvos.json")
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    normalized = validate_manifest(loaded, check_files=True)
    assert normalized["protocol_hash"] == manifest["protocol_hash"]


def test_official_spin_count_check_is_on_by_default(tmp_path: Path) -> None:
    annotations, rgb_map = _make_spin(tmp_path)
    with pytest.raises(ManifestError, match="official cohort has 24 masks; found 2"):
        build_spin_manifest(annotations, rgb_map)


def test_nvos_prompt_cannot_be_replaced_by_target_gt_without_hash_change(
    tmp_path: Path,
) -> None:
    annotations, rgb_root = _make_nvos(tmp_path)
    manifest = build_nvos_manifest(annotations, rgb_root)
    fern = manifest["scenes"][0]
    target_id = fern["evaluation_frame_ids"][0]
    target = next(frame for frame in fern["frames"] if frame["frame_id"] == target_id)
    fern["prompt"]["positive_path"] = target["ground_truth"]

    # The generic protocol hash is path-independent, but the dataset adapter
    # must bind prompt roles to the official annotation tree and content hash.
    with pytest.raises(ManifestError, match="official scribble directory|aliases"):
        validate_manifest(manifest, check_files=True)


def test_spin_prompt_mask_must_equal_reference_gt(tmp_path: Path) -> None:
    annotations, rgb_map = _make_spin(tmp_path)
    manifest = build_spin_manifest(
        annotations,
        rgb_map,
        enforce_official_counts=False,
    )
    room = next(scene for scene in manifest["scenes"] if scene["scene_id"] == "room")
    target_id = room["evaluation_frame_ids"][0]
    target = next(frame for frame in room["frames"] if frame["frame_id"] == target_id)
    room["prompt"]["mask_path"] = target["ground_truth"]

    with pytest.raises(ManifestError, match="reference-frame GT"):
        validate_manifest(manifest, check_files=True)


def test_dataset_validator_rejects_protocol_prompt_type_mismatch(tmp_path: Path) -> None:
    annotations, rgb_root = _make_nvos(tmp_path)
    manifest = build_nvos_manifest(annotations, rgb_root)
    manifest["protocol"]["prompt_type"] = "saga_style_positive_negative_points"
    manifest["protocol_hash"] = None

    with pytest.raises(ManifestError, match="separately implemented frozen sampler"):
        validate_manifest(manifest, check_files=True)


def test_benchmark_clis_reject_truncated_cohort_before_prediction_or_scoring(
    tmp_path: Path,
) -> None:
    annotations, rgb_root = _make_nvos(tmp_path)
    manifest = build_nvos_manifest(annotations, rgb_root)
    manifest["scenes"] = manifest["scenes"][:-1]
    manifest["protocol_hash"] = compute_protocol_hash(manifest)
    source = tmp_path / "truncated.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="cohort/order mismatch"):
        predict_cli.main(
            [
                "--manifest",
                str(source),
                "--output-dir",
                str(tmp_path / "predictions"),
            ]
        )
    with pytest.raises(ManifestError, match="cohort/order mismatch"):
        eval_cli.main(
            [
                "--manifest",
                str(source),
                "--prediction-manifest",
                str(tmp_path / "missing_predictions.json"),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
