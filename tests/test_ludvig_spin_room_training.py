import argparse
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import reproductions.ludvig.train_nvos_all_view_3dgs as common_training
import reproductions.ludvig.train_spin_llff_room_all_view_3dgs as room_training


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


def _identity_audit(*, identity_proven: bool = True) -> dict:
    digest = "a" * 64
    return {
        "strategy": "reuse_verified_identical_llff_colmap_undistortion",
        "raw_scene_identity_proven": identity_proven,
        "raw_dataset_modified": False,
        "raw_rgb_images": 41,
        "raw_sparse_sha256": {
            "cameras.bin": digest,
            "images.bin": digest,
            "points3D.bin": digest,
        },
        "raw_rgb_sha256": {
            f"IMG_{index:03d}.JPG": digest for index in range(41)
        },
        "pinhole": {
            "strategy": "stage_existing_colmap_undistortion_output",
            "camera_model": "PINHOLE",
            "registered_images": 41,
            "rgb_images": 41,
            "rgb_dimensions": [4005, 3003],
            "raw_dataset_modified": False,
        },
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


def test_room_spec_freezes_exact_dataset_and_resolution_contract() -> None:
    spec = room_training.SPIN_LLFF_ROOM_SPEC

    assert spec.benchmark == "SPIn-NeRF"
    assert spec.scene == spec.geometry_scene == "room"
    assert spec.expected_registered_images == 41
    assert spec.evaluation_render_resolution == (1600, 1200)
    assert spec.converted_source_relative == Path(
        "NVOS/llff_undistorted/room_undistort"
    )
    assert spec.raw_identity_source_relative == Path(
        "SPIn-NeRF/source_images/llff_google_drive/extracted/"
        "nerf_llff_data/room"
    )
    assert common_training._automatic_training_resolution(4005, 3003) == (
        1600,
        1199,
    )
    with pytest.raises(FrozenInstanceError):
        spec.scene = "fern"


def test_room_cpu_dry_run_records_cross_dataset_identity_and_both_resolutions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "room_cpu_dry_run")
    _mock_cpu_preflight(monkeypatch)

    def stage(raw_source, converted_source, staging, width, height):
        benchmark_root = args.benchmark_root.resolve()
        assert raw_source == (
            benchmark_root
            / room_training.SPIN_LLFF_ROOM_SPEC.raw_identity_source_relative
        )
        assert converted_source == (
            benchmark_root
            / room_training.SPIN_LLFF_ROOM_SPEC.converted_source_relative
        )
        assert staging.parts[-2:] == (
            "staging",
            "colmap_pinhole_undistorted",
        )
        assert (width, height) == (1600, 1200)
        return _identity_audit()

    monkeypatch.setattr(
        common_training,
        "_stage_spin_llff_pinhole_colmap",
        stage,
    )

    manifest_path = room_training.launch(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "dry_run"
    assert manifest["benchmark"] == "SPIn-NeRF"
    assert manifest["scene"] == manifest["geometry_scene"] == "room"
    assert manifest["expected_registered_images"] == 41
    protocol = manifest["effective_training_protocol"]
    assert protocol["registered_training_views"] == 41
    assert protocol["source_camera_resolution"] == [4005, 3003]
    assert protocol["effective_resolution"] == [1600, 1199]
    assert protocol["evaluation_render_resolution"] == [1600, 1200]
    provenance = manifest["geometry_input_provenance"]
    assert provenance["source_asset_contract"] == (
        common_training.VERIFIED_SPIN_NVOS_PINHOLE_REUSE_CONTRACT
    )
    assert provenance["raw_scene_identity_proven"] is True
    assert provenance["raw_rgb_images"] == 41
    assert len(provenance["raw_rgb_sha256"]) == 41


def test_room_dry_run_fails_closed_when_raw_identity_is_not_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, "identity_failure")
    _mock_cpu_preflight(monkeypatch)
    monkeypatch.setattr(
        common_training,
        "_stage_spin_llff_pinhole_colmap",
        lambda *_args: _identity_audit(identity_proven=False),
    )

    with pytest.raises(
        common_training.TrainingProtocolError,
        match="identity was not proven",
    ):
        room_training.launch(args)

    manifest_path = (
        args.output_root
        / "attempts"
        / args.attempt_id
        / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_preflight"
    assert manifest["benchmark"] == "SPIn-NeRF"
    assert manifest["scene"] == manifest["geometry_scene"] == "room"


def test_nvos_launch_retains_scene_argument_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def launch_common(args, spec):
        captured["args"] = args
        captured["spec"] = spec
        return tmp_path / "manifest.json"

    monkeypatch.setattr(common_training, "launch_all_view_training", launch_common)
    args = argparse.Namespace(scene="fern")

    assert common_training.launch(args) == tmp_path / "manifest.json"
    assert captured["args"] is args
    assert captured["spec"] == common_training._nvos_training_spec("fern")


def test_room_cli_has_no_mutable_scene_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train_spin_llff_room_all_view_3dgs.py", "--attempt-id", "cpu"],
    )
    args = room_training.parse_args()

    assert not hasattr(args, "scene")
    assert args.attempt_id == "cpu"
