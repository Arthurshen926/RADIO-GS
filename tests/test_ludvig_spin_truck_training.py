import argparse
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import struct

from PIL import Image
import pytest

import reproductions.ludvig.run_ludvig_sam as runner
import reproductions.ludvig.recover_spin_truck_training_postflight as recovery
import reproductions.ludvig.train_nvos_all_view_3dgs as common_training
import reproductions.ludvig.train_spin_truck_all_view_3dgs as truck_training


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


def _truck_audit(*, complete_rgb_hashes: bool = True) -> dict:
    digest = "a" * 64
    rgb_count = 251 if complete_rgb_hashes else 250
    return {
        "strategy": "stage_native_graphdeco_truck_pinhole",
        "camera_model": "PINHOLE",
        "camera_metadata_dimensions": [1957, 1091],
        "registered_images": 251,
        "rgb_images": 251,
        "rgb_dimensions": [979, 546],
        "source_sha256": {
            "cameras.bin": digest,
            "images.bin": digest,
            "points3D.bin": digest,
        },
        "rgb_sha256": {
            f"{index:06d}.jpg": digest for index in range(1, rgb_count + 1)
        },
        "raw_dataset_modified": False,
    }


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


def _write_images_bin(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack(
            "<Qi4d3di",
            1,
            1,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
        )
        + name.encode("utf-8")
        + b"\x00"
        + struct.pack("<Q", 0)
    )


def test_truck_spec_freezes_graphdeco_source_and_resolution() -> None:
    spec = truck_training.SPIN_TRUCK_SPEC

    assert spec.benchmark == "SPIn-NeRF"
    assert spec.scene == spec.geometry_scene == "truck"
    assert spec.expected_registered_images == 251
    assert spec.evaluation_render_resolution == (979, 546)
    assert spec.converted_source_relative == Path(
        "SPIn-NeRF/source_images/tandt/extracted/tandt/truck"
    )
    assert spec.raw_identity_source_relative is None
    assert spec.source_asset_contract == (
        common_training.NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT
    )
    assert common_training._automatic_training_resolution(979, 546) == (
        979,
        546,
    )
    with pytest.raises(FrozenInstanceError):
        spec.scene = "pinecone"


def test_truck_native_staging_preserves_released_half_resolution_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "truck"
    sparse = source / "sparse" / "0"
    images = source / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (979, 546), color=(1, 2, 3)).save(
        images / "000001.jpg"
    )
    _write_camera_bin(
        sparse / "cameras.bin",
        model_id=1,
        width=1957,
        height=1091,
        params=(1163.25, 1156.28, 978.5, 545.5),
    )
    _write_images_bin(sparse / "images.bin", "000001.jpg")
    (sparse / "points3D.bin").write_bytes(b"points")

    staging = tmp_path / "staging"
    audit = runner._stage_spin_truck_pinhole_colmap(
        source,
        staging,
        979,
        546,
    )

    assert audit["strategy"] == "stage_native_graphdeco_truck_pinhole"
    assert audit["camera_model"] == "PINHOLE"
    assert audit["camera_metadata_dimensions"] == [1957, 1091]
    assert audit["rgb_dimensions"] == [979, 546]
    assert audit["registered_images"] == audit["rgb_images"] == 1
    assert set(audit["rgb_sha256"]) == {"000001.jpg"}
    assert audit["raw_dataset_modified"] is False
    assert (staging / "images").resolve() == images.resolve()
    assert (staging / "sparse" / "0" / "images.bin").resolve() == (
        sparse / "images.bin"
    ).resolve()

    _write_camera_bin(
        sparse / "cameras.bin",
        model_id=2,
        width=1957,
        height=1091,
        params=(1163.25, 978.5, 545.5, 0.01),
    )
    with pytest.raises(runner.ProtocolError, match="native PINHOLE"):
        runner._stage_spin_truck_pinhole_colmap(
            source,
            tmp_path / "unused",
            979,
            546,
        )


def test_truck_cpu_dry_run_records_native_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "truck_cpu_dry_run")
    _mock_cpu_preflight(monkeypatch)

    def stage(source, staging, width, height):
        assert source == (
            args.benchmark_root.resolve()
            / truck_training.SPIN_TRUCK_SPEC.converted_source_relative
        )
        assert staging.parts[-2:] == (
            "staging",
            "colmap_pinhole_undistorted",
        )
        assert (width, height) == (979, 546)
        return _truck_audit()

    monkeypatch.setattr(
        common_training,
        "_stage_spin_truck_pinhole_colmap",
        stage,
    )

    manifest_path = truck_training.launch(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "dry_run"
    assert manifest["benchmark"] == "SPIn-NeRF"
    assert manifest["scene"] == manifest["geometry_scene"] == "truck"
    protocol = manifest["effective_training_protocol"]
    assert protocol["registered_training_views"] == 251
    assert protocol["source_rgb_resolution"] == [979, 546]
    assert protocol["camera_metadata_resolution"] == [1957, 1091]
    assert protocol["source_camera_resolution"] == [979, 546]
    assert protocol["effective_resolution"] == [979, 546]
    assert protocol["evaluation_render_resolution"] == [979, 546]
    provenance = manifest["geometry_input_provenance"]
    assert provenance["source_asset_contract"] == (
        common_training.NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT
    )
    assert provenance["strategy"] == "stage_native_graphdeco_truck_pinhole"
    assert provenance["source_rgb_images"] == 251
    assert len(provenance["source_rgb_sha256"]) == 251
    assert provenance["raw_dataset_modified"] is False


def test_truck_output_validation_distinguishes_rgb_from_colmap_metadata(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    model = run_dir / "model"
    point_cloud = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    point_cloud.parent.mkdir(parents=True)
    point_cloud.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 12\nproperty float x\nend_header\npayload"
    )
    (model / "cfg_args").write_text(
        "Namespace(eval=False, resolution=-1)",
        encoding="utf-8",
    )
    (model / "cameras.json").write_text(
        json.dumps(
            [
                {"width": 1957, "height": 1091, "id": index}
                for index in range(251)
            ]
        ),
        encoding="utf-8",
    )

    output = common_training._validate_training_output(
        run_dir,
        model,
        expected_registered_images=251,
        expected_source_resolution=(979, 546),
        expected_camera_metadata_resolution=(1957, 1091),
    )

    assert output["source_rgb_resolution"] == [979, 546]
    assert output["camera_metadata_resolution"] == [1957, 1091]
    assert output["source_camera_resolution"] == [979, 546]
    assert output["effective_training_resolution"] == [979, 546]

    with pytest.raises(
        common_training.TrainingProtocolError,
        match="staged COLMAP metadata resolution",
    ):
        common_training._validate_training_output(
            run_dir,
            model,
            expected_registered_images=251,
            expected_source_resolution=(979, 546),
            expected_camera_metadata_resolution=(979, 546),
        )


def test_truck_postflight_recovery_is_pinned_and_check_only_by_default(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "truck_recovery"
    staged = run_dir / "staging" / "colmap_pinhole_undistorted"
    sparse = staged / "sparse" / "0"
    images = staged / "images"
    model = run_dir / "model"
    point_cloud = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    point_cloud.parent.mkdir(parents=True)

    sparse_hashes = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = sparse / name
        path.write_bytes(f"fake-{name}".encode("ascii"))
        sparse_hashes[name] = common_training._sha256(path)
    rgb_hashes = {}
    cameras = []
    for index in range(1, 252):
        name = f"{index:06d}.jpg"
        path = images / name
        path.write_bytes(f"fake-rgb-{index}".encode("ascii"))
        rgb_hashes[name] = common_training._sha256(path)
        cameras.append(
            {
                "id": index - 1,
                "img_name": path.stem,
                "width": 1957,
                "height": 1091,
            }
        )
    point_cloud.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 12\nproperty float x\nend_header\npayload"
    )
    (model / "cfg_args").write_text(
        "Namespace(eval=False, resolution=-1, "
        f"source_path={str(staged.resolve())!r}, "
        f"model_path={str(model.resolve())!r})",
        encoding="utf-8",
    )
    (model / "cameras.json").write_text(
        json.dumps(cameras),
        encoding="utf-8",
    )
    log = run_dir / "stdout_stderr.log"
    log.write_text("completed original 3DGS training", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "failed_validation",
        "method": "original-3DGS",
        "benchmark": "SPIn-NeRF",
        "scene": "truck",
        "geometry_scene": "truck",
        "geometry_protocol": "released_all_view",
        "expected_registered_images": 251,
        "evaluation_render_resolution": [979, 546],
        "source_asset_contract": (
            common_training.NATIVE_SPIN_TRUCK_PINHOLE_CONTRACT
        ),
        "attempt_id": run_dir.name,
        "returncode": 0,
        "log": str(log.resolve()),
        "log_sha256": common_training._sha256(log),
        "error_type": "TrainingProtocolError",
        "error": recovery.LEGACY_VALIDATION_ERROR,
        "camera_audit": {
            "strategy": "stage_native_graphdeco_truck_pinhole",
            "staged_scene": str(staged.resolve()),
            "camera_model": "PINHOLE",
            "camera_metadata_dimensions": [1957, 1091],
            "registered_images": 251,
            "rgb_images": 251,
            "rgb_dimensions": [979, 546],
            "released_rgb_downsampling": "ceil_half_from_colmap_metadata",
            "source_sha256": sparse_hashes,
            "rgb_sha256": rgb_hashes,
            "raw_dataset_modified": False,
        },
        "effective_training_protocol": {
            "registered_training_views": 251,
            "held_out_training_views": 0,
            "eval_split_enabled": False,
            "iterations": 30000,
            "resolution_argument": -1,
            "source_camera_resolution": [979, 546],
            "effective_resolution": [979, 546],
            "evaluation_render_resolution": [979, 546],
            "save_iterations": [30000],
        },
    }
    manifest_path = run_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = common_training._sha256(manifest_path)
    point_cloud_sha256 = common_training._sha256(point_cloud)
    original_manifest = manifest_path.read_bytes()

    candidate = recovery.recover(
        run_dir,
        expected_manifest_sha256=manifest_sha256,
        expected_point_cloud_sha256=point_cloud_sha256,
        apply=False,
    )
    assert manifest_path.read_bytes() == original_manifest
    assert candidate["status"] == "complete"
    assert candidate["training_output"]["source_rgb_resolution"] == [979, 546]
    assert candidate["training_output"]["camera_metadata_resolution"] == [
        1957,
        1091,
    ]
    assert candidate["postflight_recovery"]["gpu_work_performed"] is False

    with pytest.raises(
        recovery.PostflightRecoveryError,
        match="point cloud hash",
    ):
        recovery.recover(
            run_dir,
            expected_manifest_sha256=manifest_sha256,
            expected_point_cloud_sha256="0" * 64,
            apply=False,
        )
    assert manifest_path.read_bytes() == original_manifest

    recovery.recover(
        run_dir,
        expected_manifest_sha256=manifest_sha256,
        expected_point_cloud_sha256=point_cloud_sha256,
        apply=True,
    )
    recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recovered["status"] == "complete"
    assert "error" not in recovered
    assert recovered["training_output"]["point_cloud_sha256"] == (
        point_cloud_sha256
    )


def test_truck_training_fails_closed_on_incomplete_rgb_hash_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "truck_incomplete_hashes")
    _mock_cpu_preflight(monkeypatch)
    monkeypatch.setattr(
        common_training,
        "_stage_spin_truck_pinhole_colmap",
        lambda *_args: _truck_audit(complete_rgb_hashes=False),
    )

    with pytest.raises(
        common_training.TrainingProtocolError,
        match="Incomplete Truck RGB hash audit",
    ):
        truck_training.launch(args)

    manifest_path = (
        args.output_root
        / "attempts"
        / args.attempt_id
        / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_preflight"


def test_truck_training_fails_closed_without_colmap_metadata_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "truck_missing_camera_metadata")
    _mock_cpu_preflight(monkeypatch)
    audit = _truck_audit()
    audit.pop("camera_metadata_dimensions")
    monkeypatch.setattr(
        common_training,
        "_stage_spin_truck_pinhole_colmap",
        lambda *_args: audit,
    )

    with pytest.raises(
        common_training.TrainingProtocolError,
        match="1957x1091 COLMAP metadata",
    ):
        truck_training.launch(args)

    manifest_path = (
        args.output_root
        / "attempts"
        / args.attempt_id
        / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_preflight"


def test_truck_evaluation_requires_audited_native_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    source = (
        benchmark_root
        / "SPIn-NeRF"
        / "source_images"
        / "tandt"
        / "extracted"
        / "tandt"
        / "truck"
    )
    source.mkdir(parents=True)
    annotation = (
        benchmark_root
        / "SPIn-NeRF"
        / "multiview_annotations"
        / "Truck (Tanks & Temples)"
    )
    annotation.mkdir(parents=True)
    base = dict(
        benchmark="spin",
        scene="truck",
        benchmark_root=benchmark_root,
        stage_nvos_pinhole=False,
        stage_spin_llff_pinhole_from=None,
        gs_source=tmp_path / "point_cloud.ply",
        geometry_protocol="released_all_view",
    )

    with pytest.raises(runner.ProtocolError, match="Truck requires"):
        runner._resolve_inputs(
            argparse.Namespace(**base, stage_spin_truck_pinhole=False),
            tmp_path / "run_without_audit",
        )

    monkeypatch.setattr(
        runner,
        "_stage_spin_truck_pinhole_colmap",
        lambda *_args: _truck_audit(),
    )
    inputs = runner._resolve_inputs(
        argparse.Namespace(**base, stage_spin_truck_pinhole=True),
        tmp_path / "run_with_audit",
    )
    assert inputs["width"] == 979
    assert inputs["height"] == 546
    assert inputs["colmap_staging"]["registered_images"] == 251
    assert inputs["colmap_dir"].parts[-2:] == (
        "staging",
        "colmap_native_truck_pinhole",
    )


def test_truck_cli_has_no_mutable_scene_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train_spin_truck_all_view_3dgs.py", "--attempt-id", "cpu"],
    )
    args = truck_training.parse_args()

    assert not hasattr(args, "scene")
    assert args.attempt_id == "cpu"
