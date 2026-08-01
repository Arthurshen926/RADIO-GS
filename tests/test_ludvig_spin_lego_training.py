import argparse
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import struct

from PIL import Image
import pytest

import reproductions.ludvig.run_ludvig_sam as runner
import reproductions.ludvig.stage_spin_lego_official_undistortion as lego_stage
import reproductions.ludvig.train_nvos_all_view_3dgs as common_training
import reproductions.ludvig.train_spin_lego_all_view_3dgs as lego_training


def _args(tmp_path: Path, attempt_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        attempt_id=attempt_id,
        upstream=tmp_path / "upstream",
        ludvig_upstream=tmp_path / "ludvig",
        dependency_root=tmp_path / "dependencies",
        python=tmp_path / "python",
        driver_library_dir=tmp_path / "driver",
        benchmark_root=tmp_path / "benchmarks",
        output_root=tmp_path / "output",
        dry_run=True,
    )


def _mock_cpu_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        common_training,
        "_validate_source",
        lambda *_args, **_kwargs: {"validated": "source"},
    )
    monkeypatch.setattr(
        common_training,
        "_validate_dependencies",
        lambda *_args, **_kwargs: {"validated": "dependencies"},
    )


def _write_camera_bin(
    path: Path,
    *,
    model_id: int,
    width: int,
    height: int,
    params: tuple[float, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<QiiQQ", 1, 1, model_id, width, height)
        + struct.pack(f"<{len(params)}d", *params)
    )


def _write_images_bin(path: Path, *, qvec_offset: float = 0.0) -> None:
    payload = bytearray(struct.pack("<Q", lego_stage.LEGO_REGISTERED_VIEWS))
    for index in range(lego_stage.LEGO_REGISTERED_VIEWS):
        payload.extend(
            struct.pack(
                "<i4d3di",
                index + 1,
                1.0 + qvec_offset,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            )
        )
        payload.extend(f"{index:05d}.png".encode("utf-8") + b"\x00")
        payload.extend(struct.pack("<Q", 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_lego_asset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "lego_real_night_radial"
    raw_sparse = source / "sparse" / "0"
    converted_sparse = source / "sparse"
    _write_camera_bin(
        raw_sparse / "cameras.bin",
        model_id=2,
        width=1020,
        height=768,
        params=lego_stage.LEGO_RAW_CAMERA_PARAMS,
    )
    _write_camera_bin(
        converted_sparse / "cameras.bin",
        model_id=1,
        width=1015,
        height=764,
        params=lego_stage.LEGO_UNDISTORTED_CAMERA_PARAMS,
    )
    _write_images_bin(raw_sparse / "images.bin")
    _write_images_bin(converted_sparse / "images.bin")
    (raw_sparse / "points3D.bin").write_bytes(b"raw points")
    (converted_sparse / "points3D.bin").write_bytes(b"converted points")

    images = source / "images"
    images_resized = source / "images_resized"
    images.mkdir(parents=True)
    images_resized.mkdir(parents=True)
    for index in range(lego_stage.LEGO_REGISTERED_VIEWS):
        prefix = "1" if index % 16 == 0 else "0"
        name = f"{prefix}_{index:05d}.png"
        color = (index % 256, (index * 3) % 256, (index * 7) % 256)
        Image.new("RGB", (1015, 764), color=color).save(images / name)
        Image.new("RGB", (1020, 768), color=color).save(images_resized / name)

    annotation = tmp_path / "lego_annotations"
    annotation.mkdir()
    annotation_indices = [*range(1, 44), 48]
    for index in annotation_indices:
        prefix = "1" if index % 16 == 0 else "0"
        stem = f"{prefix}_{index:05d}"
        mask = Image.new("L", (1015, 764), color=0)
        mask.putpixel((0, 0), 1)
        mask.save(annotation / f"{stem}.png")
        Image.new("RGBA", (1015, 764), color=(1, 2, 3, 255)).save(
            annotation / f"{stem}_cutout.png"
        )
        Image.new("RGB", (1015, 764), color=(4, 5, 6)).save(
            annotation / f"{stem}_pseudo.png"
        )
    return source, annotation


def _lego_audit(*, complete_annotation_hashes: bool = True) -> dict:
    digest = "a" * 64
    annotation_count = 132 if complete_annotation_hashes else 131
    mapping = {
        f"{'1' if index % 16 == 0 else '0'}_{index:05d}.png": (
            f"{index:05d}.png"
        )
        for index in range(102)
    }
    return {
        "strategy": "stage_native_spin_lego_official_undistortion",
        "camera_model": "PINHOLE",
        "camera_metadata_dimensions": [1015, 764],
        "registered_images": 102,
        "rgb_images": 102,
        "rgb_dimensions": [1015, 764],
        "raw_sparse_sha256": {
            "cameras.bin": digest,
            "images.bin": digest,
            "points3D.bin": digest,
        },
        "source_sha256": {
            "cameras.bin": digest,
            "images.bin": digest,
            "points3D.bin": digest,
        },
        "rgb_sha256": {name: digest for name in mapping},
        "raw_resized_rgb_sha256": {name: digest for name in mapping},
        "annotation_sha256": {
            f"annotation_{index:03d}.png": digest
            for index in range(annotation_count)
        },
        "image_name_mapping": {
            "rule": "remove_exact_0_or_1_split_prefix",
            "entries": mapping,
            "mapping_sha256": digest,
            "bijective": True,
        },
        "annotation_roles": {
            "reference_mask": "0_00001.png",
            "reference_camera": "00001",
            "reference_scored": False,
            "target_masks": 43,
            "target_masks_scoring_only": True,
            "annotation_rgb_used_for_training": False,
            "all_masks": 44,
            "mask_values": [0, 1],
            "all_mapped_to_registered_cameras": True,
        },
        "raw_dataset_modified": False,
    }


def test_lego_spec_freezes_official_undistortion_contract() -> None:
    spec = lego_training.SPIN_LEGO_SPEC

    assert spec.benchmark == "SPIn-NeRF"
    assert spec.scene == spec.geometry_scene == "lego"
    assert spec.expected_registered_images == 102
    assert spec.evaluation_render_resolution == (1015, 764)
    assert spec.converted_source_relative == Path(
        "SPIn-NeRF/source_images/lego_real_night_radial/"
        "lego_real_night_radial"
    )
    assert spec.raw_identity_source_relative is None
    assert spec.source_asset_contract == (
        common_training.NATIVE_SPIN_LEGO_PINHOLE_CONTRACT
    )
    assert common_training._automatic_training_resolution(1015, 764) == (
        1015,
        764,
    )
    with pytest.raises(FrozenInstanceError):
        spec.scene = "pinecone"


def test_lego_staging_proves_pose_prefix_and_annotation_roles(
    tmp_path: Path,
) -> None:
    source, annotation = _make_lego_asset(tmp_path)
    staging = tmp_path / "staging"
    audit = lego_stage._stage_spin_lego_pinhole_colmap(
        source,
        annotation,
        staging,
        1015,
        764,
    )

    assert audit["strategy"] == "stage_native_spin_lego_official_undistortion"
    assert audit["raw_camera_model"] == "SIMPLE_RADIAL"
    assert audit["camera_model"] == "PINHOLE"
    assert audit["registered_images"] == audit["rgb_images"] == 102
    assert audit["max_qvec_delta_vs_raw_sparse"] == 0.0
    assert audit["max_tvec_delta_vs_raw_sparse"] == 0.0
    assert len(audit["image_name_mapping"]["entries"]) == 102
    assert audit["image_name_mapping"]["bijective"] is True
    assert audit["annotation_roles"]["reference_mask"] == "0_00001.png"
    assert audit["annotation_roles"]["reference_scored"] is False
    assert audit["annotation_roles"]["target_masks"] == 43
    assert audit["annotation_roles"]["annotation_rgb_used_for_training"] is False
    assert len(audit["annotation_sha256"]) == 132
    assert audit["raw_dataset_modified"] is False
    assert (staging / "images" / "00000.png").resolve() == (
        source / "images" / "1_00000.png"
    ).resolve()
    assert (staging / "images" / "00001.png").resolve() == (
        source / "images" / "0_00001.png"
    ).resolve()
    assert (staging / "sparse" / "0" / "cameras.bin").resolve() == (
        source / "sparse" / "cameras.bin"
    ).resolve()

    renamed = source / "images" / "0_00001.png"
    renamed.rename(source / "images" / "0_99999.png")
    failed_staging = tmp_path / "failed_staging"
    with pytest.raises(runner.ProtocolError, match="frozen 0_/1_ split cohort"):
        lego_stage._stage_spin_lego_pinhole_colmap(
            source,
            annotation,
            failed_staging,
            1015,
            764,
        )
    assert not failed_staging.exists()


def test_lego_cpu_dry_run_records_full_source_and_annotation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "lego_official_undistortion_cpu_preflight")
    _mock_cpu_preflight(monkeypatch)

    def stage(source, annotation, staging, width, height):
        benchmark_root = args.benchmark_root.resolve()
        assert source == (
            benchmark_root / lego_training.SPIN_LEGO_SPEC.converted_source_relative
        )
        assert annotation == benchmark_root / lego_stage.LEGO_ANNOTATION_RELATIVE
        assert staging.parts[-2:] == ("staging", "colmap_pinhole_undistorted")
        assert (width, height) == (1015, 764)
        return _lego_audit()

    monkeypatch.setattr(common_training, "_stage_spin_lego_pinhole_colmap", stage)
    manifest_path = lego_training.launch(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "dry_run"
    assert manifest["benchmark"] == "SPIn-NeRF"
    assert manifest["scene"] == manifest["geometry_scene"] == "lego"
    protocol = manifest["effective_training_protocol"]
    assert protocol["registered_training_views"] == 102
    assert protocol["source_camera_resolution"] == [1015, 764]
    assert protocol["effective_resolution"] == [1015, 764]
    provenance = manifest["geometry_input_provenance"]
    assert provenance["source_asset_contract"] == (
        common_training.NATIVE_SPIN_LEGO_PINHOLE_CONTRACT
    )
    assert provenance["strategy"] == (
        "stage_native_spin_lego_official_undistortion"
    )
    assert provenance["source_rgb_images"] == 102
    assert len(provenance["source_rgb_sha256"]) == 102
    assert len(provenance["raw_resized_rgb_sha256"]) == 102
    assert len(provenance["annotation_sha256"]) == 132
    assert provenance["image_name_mapping"]["bijective"] is True
    assert provenance["annotation_roles"]["target_masks"] == 43
    assert provenance["raw_dataset_modified"] is False


def test_lego_training_fails_closed_on_incomplete_annotation_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "lego_incomplete_annotation_hashes")
    _mock_cpu_preflight(monkeypatch)
    monkeypatch.setattr(
        common_training,
        "_stage_spin_lego_pinhole_colmap",
        lambda *_args: _lego_audit(complete_annotation_hashes=False),
    )

    with pytest.raises(
        common_training.TrainingProtocolError,
        match="Incomplete Lego annotation hash audit",
    ):
        lego_training.launch(args)

    manifest_path = (
        args.output_root
        / "attempts"
        / args.attempt_id
        / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_preflight"


def test_lego_cli_has_no_mutable_scene_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train_spin_lego_all_view_3dgs.py", "--attempt-id", "cpu"],
    )
    args = lego_training.parse_args()

    assert not hasattr(args, "scene")
    assert args.attempt_id == "cpu"
