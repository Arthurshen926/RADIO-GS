import ast
import json
import hashlib
from pathlib import Path
import re
import struct

from PIL import Image
import pytest

import reproductions.ludvig.aggregate_results as aggregator_module
import reproductions.ludvig.run_ludvig_sam as launcher
from reproductions.ludvig.aggregate_results import AggregationError, aggregate
from reproductions.ludvig.run_nvos_hybrid_cohort import (
    ATTEMPT_PREFIX,
    _completed_attempt,
    _next_attempt_id,
    _task_grid,
)
from reproductions.ludvig.run_ludvig_sam import (
    NVOS_GEOMETRY_REGISTERED_IMAGES as RUNNER_NVOS_GEOMETRY_REGISTERED_IMAGES,
    NVOS_SCENES,
    NVOS_TASK_TO_GEOMETRY_SCENE,
    ProtocolError,
    SAM_VIT_H_CHECKPOINT_SHA256,
    SPIN_GEOMETRY_REGISTERED_IMAGES,
    _runtime_environment,
    _scaled_pinhole_audit,
    _resolve_inputs,
    _stage_nvos_pinhole_colmap,
    _stage_spin_llff_pinhole_colmap,
    _validate_pythonpath,
    _validate_sam_checkpoint,
    _validate_upstream,
)
from reproductions.ludvig.train_nvos_all_view_3dgs import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_NVOS_TRAINING_OUTPUT_ROOT,
    LOCK_FILE as OFFICIAL_3DGS_LOCK_FILE,
    NVOS_GEOMETRY_REGISTERED_IMAGES,
    NVOS_GEOMETRY_SCENES,
    OFFICIAL_3DGS_COMMIT,
    TrainingProtocolError,
    _automatic_training_resolution,
    _literal_class_defaults,
    _parse_namespace,
    _parse_ply_vertex_count,
    _training_output_root,
    _training_command,
    _validate_training_output,
)


FAKE_SAM_CHECKPOINT_BYTES = b"fake audited SAM checkpoint for CPU tests"
FAKE_SAM_CHECKPOINT_SHA256 = hashlib.sha256(
    FAKE_SAM_CHECKPOINT_BYTES
).hexdigest()
UPSTREAM_CHECKOUT = Path("/root/baselines/LUDVIG")


@pytest.fixture(autouse=True)
def _pin_fake_sam_hash_for_aggregator(monkeypatch) -> None:
    monkeypatch.setattr(
        aggregator_module,
        "SAM_VIT_H_CHECKPOINT_SHA256",
        FAKE_SAM_CHECKPOINT_SHA256,
    )


def _args(tmp_path: Path, **overrides):
    values = {
        "benchmark": "nvos",
        "scene": "fern",
        "benchmark_root": tmp_path,
        "geometry_protocol": "released_all_view",
        "gs_source": None,
        "stage_nvos_pinhole": False,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_released_nvos_fails_without_all_view_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="requires --gs-source"):
        _resolve_inputs(_args(tmp_path), tmp_path / "run")


def test_pythonpath_preflight_rejects_missing_rasterizer(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="--pythonpath is required"):
        _validate_pythonpath(None)
    with pytest.raises(ProtocolError, match="exactly one compiled"):
        _validate_pythonpath(tmp_path)

    package = tmp_path / "diff_gaussian_rasterization"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    extension = package / "_C.cpython-39-x86_64-linux-gnu.so"
    extension.write_bytes(b"test extension")
    provenance = _validate_pythonpath(tmp_path)
    assert provenance["root"] == str(tmp_path)
    assert provenance["extension"] == str(extension)
    assert len(provenance["extension_sha256"]) == 64


@pytest.mark.parametrize(
    ("task_scene", "geometry_scene"),
    [
        ("fern", "fern"),
        ("flower", "flower"),
        ("fortress", "fortress"),
        ("horns_center", "horns"),
        ("horns_left", "horns"),
        ("leaves", "leaves"),
        ("orchids", "orchids"),
        ("trex", "trex"),
    ],
)
def test_nvos_task_resolves_to_fail_closed_geometry_scene(
    tmp_path: Path,
    task_scene: str,
    geometry_scene: str,
) -> None:
    checkpoint = tmp_path / "point_cloud.ply"
    resolved = _resolve_inputs(
        _args(tmp_path, scene=task_scene, gs_source=checkpoint),
        tmp_path / "run",
    )

    assert NVOS_TASK_TO_GEOMETRY_SCENE[task_scene] == geometry_scene
    assert resolved["geometry_scene"] == geometry_scene
    assert resolved["colmap_dir"] == (
        tmp_path
        / "NVOS"
        / "llff_undistorted"
        / f"{geometry_scene}_undistort"
    )


def test_strict_geometry_mode_is_explicitly_hybrid_and_not_exact(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path
    scene_root = benchmark / "NVOS" / "llff_undistorted" / "fern_undistort"
    scene_root.mkdir(parents=True)
    checkpoint = (
        benchmark
        / "gaussfm_jobs"
        / "nvos_strict_unseen_v1"
        / "scenes"
        / "fern"
        / "geometry"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    resolved = _resolve_inputs(
        _args(
            tmp_path,
            geometry_protocol="strict_geometry_hybrid_diagnostic",
        ),
        tmp_path / "run",
    )

    assert resolved["cohort"] == list(NVOS_SCENES)
    assert resolved["strict_unseen_exact_match"] is False
    assert (
        resolved["target_rgb_visible_during_gaussian_splatting_training"] is False
    )
    assert "hybrid diagnostic" in resolved["geometry_note"]
    assert resolved["gs_source"] == checkpoint


@pytest.mark.parametrize(
    ("scene", "expected_rgb_preprocessing"),
    (
        ("room", launcher.SPIN_ROOM_RGB_PREPROCESSING),
        ("fern", None),
    ),
)
def test_spin_room_alone_enables_exact_rgb_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scene: str,
    expected_rgb_preprocessing,
) -> None:
    annotation = (
        tmp_path
        / "SPIn-NeRF"
        / "multiview_annotations"
        / launcher.SPIN_ANNOTATION_FOLDERS[scene]
    )
    annotation.mkdir(parents=True)
    source_scene = tmp_path / launcher.SPIN_SOURCE_RELATIVE[scene]
    source_scene.mkdir(parents=True)
    converted_source = tmp_path / f"{scene}_converted"
    converted_source.mkdir()
    checkpoint = tmp_path / f"{scene}.ply"
    captured = {}

    def fake_stage(
        spin_source_scene,
        converted_source_scene,
        staging_scene,
        target_width,
        target_height,
        rgb_preprocessing=None,
    ):
        captured.update(
            {
                "spin_source_scene": spin_source_scene,
                "converted_source_scene": converted_source_scene,
                "staging_scene": staging_scene,
                "target": (target_width, target_height),
                "rgb_preprocessing": rgb_preprocessing,
            }
        )
        return {
            "pinhole": {
                "registered_images": (
                    launcher.SPIN_GEOMETRY_REGISTERED_IMAGES[scene]
                ),
                "rgb_dimensions": [target_width, target_height],
            }
        }

    monkeypatch.setattr(
        launcher,
        "_stage_spin_llff_pinhole_colmap",
        fake_stage,
    )
    run_dir = tmp_path / f"run_{scene}"
    resolved = _resolve_inputs(
        _args(
            tmp_path,
            benchmark="spin",
            scene=scene,
            gs_source=checkpoint,
            stage_spin_llff_pinhole_from=converted_source,
            stage_spin_truck_pinhole=False,
        ),
        run_dir,
    )

    assert captured["rgb_preprocessing"] == expected_rgb_preprocessing
    assert captured["target"] == launcher.SPIN_IMAGE_SIZE[scene]
    assert resolved["colmap_dir"] == (
        run_dir / "staging" / "colmap_pinhole_undistorted"
    )


def test_upstream_patch_validation_does_not_import_upstream_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    checkout = tmp_path / "ludvig"
    checkout.mkdir()
    (checkout / "ludvig_uplift.py").write_text(
        'parser.add_argument("--seed")\nreproducibility(args.seed)\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_git_head", lambda _checkout: launcher.UPSTREAM_COMMIT)
    monkeypatch.setattr(
        launcher,
        "_upstream_patch_provenance",
        lambda _checkout: {"patch_sha256": launcher.UPSTREAM_PATCH_SHA256},
    )

    _validate_upstream(checkout)
    assert hashlib.sha256(launcher.UPSTREAM_PATCH.read_bytes()).hexdigest() == (
        launcher.UPSTREAM_PATCH_SHA256
    )


def _load_pure_upstream_function(relative_path: str, function_name: str):
    source_path = UPSTREAM_CHECKOUT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[function_name]


def test_upstream_v3_resolution_and_cuda_shape_guards() -> None:
    resolution = _load_pure_upstream_function(
        "utils/image.py", "_max_dimension_downsample_resolution"
    )
    require_sam_match = _load_pure_upstream_function(
        "predictors/sam.py", "_require_sam_resolution_match"
    )
    require_feature_match = _load_pure_upstream_function(
        "utils/solver.py", "_require_feature_camera_resolution"
    )

    assert resolution(4002, 3001, 1600) == (1600, 1199)
    assert resolution(4005, 3003, 1600) == (1600, 1199)
    assert resolution(1600, 1200, 1600) == (1600, 1200)
    for dimensions, expected in (
        ((3982, 2986), (1600, 1199)),
        ((3985, 2988), (1600, 1199)),
        ((4014, 3010), (1600, 1199)),
        ((3949, 2961), (1600, 1199)),
        ((4032, 3024), (1600, 1200)),
        ((979, 546), (979, 546)),
    ):
        assert resolution(*dimensions, 1600) == expected

    matching = (1199, 1600)
    require_sam_match(
        prompt_shape=matching,
        image_shape=matching,
        camera_shape=matching,
        mask_shape=matching,
    )
    require_feature_match(matching, matching)
    with pytest.raises(RuntimeError, match="before prompt sampling"):
        require_sam_match(
            prompt_shape=(1199, 1600),
            image_shape=(1199, 1599),
            camera_shape=(1199, 1600),
        )
    with pytest.raises(RuntimeError, match="after prediction"):
        require_sam_match(
            prompt_shape=matching,
            image_shape=matching,
            camera_shape=matching,
            mask_shape=(1199, 1599),
        )
    with pytest.raises(RuntimeError, match="before apply_weights"):
        require_feature_match((1199, 1599), (1199, 1600))


def test_ludvig_sam_rejects_unpinned_checkpoint_before_gpu(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "sam_vit_h.pth"
    checkpoint.write_bytes(b"not the audited SAM checkpoint")

    with pytest.raises(ProtocolError, match="audited ViT-H SAM"):
        _validate_sam_checkpoint(checkpoint)

    assert (
        SAM_VIT_H_CHECKPOINT_SHA256
        == "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
    )


def _released_training_payload(checkpoint: Path, scene: str = "fern") -> dict:
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "status": "complete",
        "method": "original-3DGS",
        "benchmark": "NVOS",
        "scene": scene,
        "geometry_scene": scene,
        "geometry_protocol": "released_all_view",
        "source_provenance": {
            "commit": "f7a116fb1397d9842239127d39dc212f93171f70",
            "ludvig_commit": "4461fc515439bb498a75d71738a1e73cf7a452ed",
        },
        "effective_training_protocol": {
            "registered_training_views": 20,
            "held_out_training_views": 0,
            "eval_split_enabled": False,
            "iterations": 30000,
            "resolution_argument": -1,
            "algorithm_source_modified": False,
        },
        "training_output": {
            "point_cloud": str(checkpoint.resolve()),
            "point_cloud_sha256": checkpoint_sha256,
            "cfg_args": {"eval": False, "resolution": -1},
            "registered_all_view_cameras": 20,
            "target_rgb_visible_during_training": True,
        },
    }


def test_released_training_manifest_is_checkpoint_bound_and_type_strict(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "point_cloud.ply"
    checkpoint.write_bytes(b"audited point cloud")
    manifest = tmp_path / "training_manifest.json"
    payload = _released_training_payload(checkpoint)
    payload.pop("geometry_scene")  # Compatibility with the in-flight fern run.
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    verified = launcher._validate_released_all_view_training_manifest(
        manifest,
        checkpoint,
        "fern",
    )

    assert verified["verified"] is True
    assert verified["point_cloud_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert verified["registered_training_views"] == 20
    assert verified["legacy_geometry_scene_fallback"] is True

    payload["effective_training_protocol"]["registered_training_views"] = True
    payload["training_output"]["registered_all_view_cameras"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="frozen asset contract"):
        launcher._validate_released_all_view_training_manifest(
            manifest,
            checkpoint,
            "fern",
        )


def test_released_training_manifest_enforces_geometry_view_count(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "point_cloud.ply"
    checkpoint.write_bytes(b"audited flower point cloud")
    manifest = tmp_path / "training_manifest.json"
    payload = _released_training_payload(checkpoint, scene="flower")
    # The helper defaults to fern's 20 views; flower's frozen asset has 34.
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProtocolError, match="expected 34, found 20"):
        launcher._validate_released_all_view_training_manifest(
            manifest,
            checkpoint,
            "flower",
        )


def test_released_training_manifest_rejects_wrong_scene_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "point_cloud.ply"
    checkpoint.write_bytes(b"audited point cloud")
    manifest = tmp_path / "training_manifest.json"
    payload = _released_training_payload(checkpoint, scene="flower")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProtocolError, match="does not match fern"):
        launcher._validate_released_all_view_training_manifest(
            manifest,
            checkpoint,
            "fern",
        )

    payload["scene"] = "fern"
    payload["geometry_scene"] = "fern"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint.write_bytes(b"changed after training")
    with pytest.raises(ProtocolError, match="hash does not match"):
        launcher._validate_released_all_view_training_manifest(
            manifest,
            checkpoint,
            "fern",
        )


def test_released_training_manifest_rejects_contradictory_geometry_scene(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "point_cloud.ply"
    checkpoint.write_bytes(b"audited horns point cloud")
    manifest = tmp_path / "training_manifest.json"
    payload = _released_training_payload(checkpoint, scene="horns")
    payload["geometry_scene"] = "horns_center"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProtocolError, match="does not match horns"):
        launcher._validate_released_all_view_training_manifest(
            manifest,
            checkpoint,
            "horns",
        )


@pytest.mark.parametrize(
    ("benchmark", "scene"),
    (("SPIn-NeRF", "room"), ("NVOS", "flower"), ("SPIn-NeRF", "fern")),
)
def test_missing_geometry_scene_fallback_is_only_for_legacy_nvos_fern(
    tmp_path: Path,
    benchmark: str,
    scene: str,
) -> None:
    checkpoint = tmp_path / "point_cloud.ply"
    checkpoint.write_bytes(b"audited point cloud")
    manifest = tmp_path / "training_manifest.json"
    payload = _released_training_payload(checkpoint, scene=scene)
    payload["benchmark"] = benchmark
    payload.pop("geometry_scene")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProtocolError, match="must explicitly record geometry_scene"):
        launcher._validate_released_all_view_training_manifest(
            manifest,
            checkpoint,
            scene,
        )

def test_runtime_environment_pins_gpu0_and_prepends_compatible_driver(
    tmp_path: Path, monkeypatch
) -> None:
    driver_dir = tmp_path / "driver"
    driver_dir.mkdir()
    (driver_dir / "libcuda.so.1").touch()
    pythonpath = tmp_path / "site"
    pythonpath.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setenv("PYTHONPATH", "/existing/python")
    args = type(
        "Args",
        (),
        {
            "driver_library_dir": driver_dir,
            "pythonpath": pythonpath,
        },
    )()

    environment, driver_library = _runtime_environment(args)

    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["LD_LIBRARY_PATH"] == f"{driver_dir}:/existing/lib"
    assert environment["PYTHONPATH"] == f"{pythonpath}:/existing/python"
    assert driver_library == driver_dir / "libcuda.so.1"


def _write_camera_bin(
    path: Path, model_id: int, width: int, height: int, params: tuple[float, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<QiiQQ", 1, 1, model_id, width, height)
        + struct.pack(f"<{len(params)}d", *params)
    )


def _write_images_bin(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<Qi4d3di", 1, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
        + name.encode("utf-8")
        + b"\x00"
        + struct.pack("<Q", 0)
    )


def test_nvos_pinhole_staging_validates_rgb_poses_and_scaled_intrinsics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fern"
    original = source / "sparse" / "0"
    converted = source / "dense" / "sparse"
    images = source / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(images / "frame.JPG")
    _write_camera_bin(
        original / "cameras.bin",
        model_id=2,
        width=42,
        height=32,
        params=(32.0, 21.0, 16.0, 0.01),
    )
    _write_camera_bin(
        converted / "cameras.bin",
        model_id=1,
        width=40,
        height=30,
        params=(32.0, 32.0, 20.0, 15.0),
    )
    _write_images_bin(original / "images.bin", "frame.JPG")
    _write_images_bin(converted / "images.bin", "frame.JPG")
    (converted / "points3D.bin").write_bytes(b"points")

    staging = tmp_path / "output" / "colmap"
    audit = _stage_nvos_pinhole_colmap(source, staging, 16, 12)

    assert audit["camera_model"] == "PINHOLE"
    assert audit["rgb_dimensions"] == [40, 30]
    assert audit["max_qvec_delta_vs_original_sparse"] == 0.0
    assert audit["max_tvec_delta_vs_original_sparse"] == 0.0
    assert audit["intrinsics"]["effective_target_fx"] == pytest.approx(12.8)
    assert audit["intrinsics"]["effective_target_fy"] == pytest.approx(12.8)
    assert audit["raw_dataset_modified"] is False
    assert (staging / "images").resolve() == images.resolve()
    assert (
        staging / "sparse" / "0" / "cameras.bin"
    ).resolve() == (converted / "cameras.bin").resolve()


def test_spin_llff_staging_requires_byte_identical_raw_scene(
    tmp_path: Path,
) -> None:
    spin = tmp_path / "spin"
    converted_source = tmp_path / "converted"
    spin_sparse = spin / "sparse" / "0"
    converted_raw_sparse = converted_source / "sparse" / "0"
    converted_sparse = converted_source / "dense" / "sparse"
    spin_images = spin / "images"
    raw_images = converted_source / "images_distort"
    undistorted_images = converted_source / "images"
    for directory in (spin_images, raw_images, undistorted_images):
        directory.mkdir(parents=True)
    Image.new("RGB", (42, 32), color=(1, 2, 3)).save(
        spin_images / "frame.JPG"
    )
    (raw_images / "frame.JPG").write_bytes(
        (spin_images / "frame.JPG").read_bytes()
    )
    Image.new("RGB", (40, 30), color=(1, 2, 3)).save(
        undistorted_images / "frame.JPG"
    )
    _write_camera_bin(
        spin_sparse / "cameras.bin",
        model_id=2,
        width=42,
        height=32,
        params=(32.0, 21.0, 16.0, 0.01),
    )
    _write_images_bin(spin_sparse / "images.bin", "frame.JPG")
    (spin_sparse / "points3D.bin").write_bytes(b"raw-points")
    converted_raw_sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (converted_raw_sparse / name).write_bytes(
            (spin_sparse / name).read_bytes()
        )
    _write_camera_bin(
        converted_sparse / "cameras.bin",
        model_id=1,
        width=40,
        height=30,
        params=(32.0, 32.0, 20.0, 15.0),
    )
    _write_images_bin(converted_sparse / "images.bin", "frame.JPG")
    (converted_sparse / "points3D.bin").write_bytes(b"undistorted-points")

    audit = _stage_spin_llff_pinhole_colmap(
        spin,
        converted_source,
        tmp_path / "staging",
        16,
        12,
    )

    assert audit["raw_scene_identity_proven"] is True
    assert audit["raw_rgb_images"] == 1
    assert audit["pinhole"]["camera_model"] == "PINHOLE"
    assert audit["pinhole"]["registered_images"] == 1
    assert audit["raw_dataset_modified"] is False
    assert (tmp_path / "staging" / "images").is_symlink()

    (converted_raw_sparse / "points3D.bin").write_bytes(b"changed")
    with pytest.raises(ProtocolError, match="same raw points3D.bin"):
        _stage_spin_llff_pinhole_colmap(
            spin,
            converted_source,
            tmp_path / "unused",
            16,
            12,
        )


def test_spin_room_staging_materializes_exact_centered_1600x1200_rgb(
    tmp_path: Path,
) -> None:
    spin = tmp_path / "spin_room"
    converted_source = tmp_path / "converted_room"
    spin_sparse = spin / "sparse" / "0"
    converted_raw_sparse = converted_source / "sparse" / "0"
    converted_sparse = converted_source / "dense" / "sparse"
    spin_images = spin / "images"
    raw_images = converted_source / "images_distort"
    undistorted_images = converted_source / "images"
    for directory in (spin_images, raw_images, undistorted_images):
        directory.mkdir(parents=True)

    Image.new("RGB", (42, 32), color=(1, 2, 3)).save(
        spin_images / "frame.JPG"
    )
    (raw_images / "frame.JPG").write_bytes(
        (spin_images / "frame.JPG").read_bytes()
    )
    source_rgb = undistorted_images / "frame.JPG"
    Image.new("RGB", (4005, 3003), color=(17, 53, 101)).save(
        source_rgb,
        quality=91,
        subsampling=2,
    )
    source_rgb_sha256 = hashlib.sha256(source_rgb.read_bytes()).hexdigest()

    _write_camera_bin(
        spin_sparse / "cameras.bin",
        model_id=2,
        width=42,
        height=32,
        params=(32.0, 21.0, 16.0, 0.01),
    )
    _write_images_bin(spin_sparse / "images.bin", "frame.JPG")
    (spin_sparse / "points3D.bin").write_bytes(b"raw-points")
    converted_raw_sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (converted_raw_sparse / name).write_bytes(
            (spin_sparse / name).read_bytes()
        )
    _write_camera_bin(
        converted_sparse / "cameras.bin",
        model_id=1,
        width=4005,
        height=3003,
        params=(3070.0, 3070.0, 2002.5, 1501.5),
    )
    _write_images_bin(converted_sparse / "images.bin", "frame.JPG")
    (converted_sparse / "points3D.bin").write_bytes(b"undistorted-points")

    staging = tmp_path / "staging_room"
    audit = _stage_spin_llff_pinhole_colmap(
        spin,
        converted_source,
        staging,
        1600,
        1200,
        rgb_preprocessing=launcher.SPIN_ROOM_RGB_PREPROCESSING,
    )

    pinhole = audit["pinhole"]
    preprocessing = pinhole["rgb_preprocessing"]
    assert pinhole["source_rgb_dimensions"] == [4005, 3003]
    assert pinhole["rgb_dimensions"] == [1600, 1200]
    assert preprocessing["center_crop_dimensions"] == [4004.0, 3003.0]
    assert preprocessing["center_crop_box"] == [0.5, 0.0, 4004.5, 3003.0]
    assert preprocessing["center_crop_coordinate_policy"] == (
        "fractional_geometric_center"
    )
    assert preprocessing["resize_resampling"] == "BICUBIC"
    assert preprocessing["output_encoding"] == "JPEG"
    assert preprocessing["source_dataset_modified"] is False
    assert preprocessing["source_rgb_sha256"]["frame.JPG"] == (
        source_rgb_sha256
    )
    assert hashlib.sha256(source_rgb.read_bytes()).hexdigest() == (
        source_rgb_sha256
    )
    staged_rgb = staging / "images" / "frame.JPG"
    assert staged_rgb.is_file()
    assert not (staging / "images").is_symlink()
    with Image.open(staged_rgb) as image:
        assert image.size == (1600, 1200)
        assert image.format == "JPEG"
    assert (staging / "sparse" / "0" / "cameras.bin").is_symlink()
    assert (staging / "sparse" / "0" / "images.bin").is_symlink()


def test_scaled_pinhole_audit_rejects_off_center_principal_point() -> None:
    with pytest.raises(ProtocolError, match="principal point is not centered"):
        _scaled_pinhole_audit(
            {
                "model": "PINHOLE",
                "width": 40,
                "height": 30,
                "params": [32.0, 32.0, 19.0, 15.0],
            },
            16,
            12,
        )


def _write_fake_run(
    root: Path,
    *,
    scene: str,
    seed: int,
    score: float,
    exact: bool = False,
    geometry_protocol: str = "strict_geometry_hybrid_diagnostic",
    benchmark: str = "NVOS",
    attempt=None,
) -> None:
    geometry_scene = (
        NVOS_TASK_TO_GEOMETRY_SCENE[scene]
        if benchmark == "NVOS"
        else scene
    )
    checkpoint = root / "_checkpoints" / geometry_scene / "point_cloud.ply"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(f"fake-{geometry_scene}-checkpoint".encode())
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sam_checkpoint = root / "_checkpoints" / "sam_vit_h.pth"
    if not sam_checkpoint.exists():
        sam_checkpoint.write_bytes(FAKE_SAM_CHECKPOINT_BYTES)

    training_provenance = None
    if geometry_protocol == "released_all_view":
        registered_views = (
            RUNNER_NVOS_GEOMETRY_REGISTERED_IMAGES[geometry_scene]
            if benchmark == "NVOS"
            else SPIN_GEOMETRY_REGISTERED_IMAGES[geometry_scene]
        )
        training_manifest = (
            root / "_training" / geometry_scene / "training_manifest.json"
        )
        training_manifest.parent.mkdir(parents=True, exist_ok=True)
        training_manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "method": "original-3DGS",
                    "benchmark": benchmark,
                    "scene": geometry_scene,
                    "geometry_scene": geometry_scene,
                    "geometry_protocol": "released_all_view",
                    "source_provenance": {
                        "commit": "f7a116fb1397d9842239127d39dc212f93171f70",
                        "ludvig_commit": (
                            "4461fc515439bb498a75d71738a1e73cf7a452ed"
                        ),
                    },
                    "effective_training_protocol": {
                        "registered_training_views": registered_views,
                        "held_out_training_views": 0,
                        "eval_split_enabled": False,
                        "iterations": 30000,
                        "resolution_argument": -1,
                        "algorithm_source_modified": False,
                    },
                    "training_output": {
                        "point_cloud": str(checkpoint.resolve()),
                        "point_cloud_sha256": checkpoint_sha256,
                        "cfg_args": {"eval": False, "resolution": -1},
                        "registered_all_view_cameras": registered_views,
                        "target_rgb_visible_during_training": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        training_provenance = {
            "verified": True,
            "training_manifest": str(training_manifest.resolve()),
            "training_manifest_sha256": hashlib.sha256(
                training_manifest.read_bytes()
            ).hexdigest(),
            "point_cloud_sha256": checkpoint_sha256,
        }

    run = root / scene / f"seed_{seed}"
    if attempt is not None:
        run = run / "attempts" / attempt
    result = run / scene / "sam"
    result.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "benchmark": benchmark,
                "scene": scene,
                "geometry_scene": geometry_scene,
                "seed": seed,
                "method": "LUDVIG-SAM",
                "protocol_id": "ludvig_official_online_multiview_v1",
                "upstream_commit": (
                    "4461fc515439bb498a75d71738a1e73cf7a452ed"
                ),
                "upstream_patch_provenance": {
                    "patch_sha256": aggregator_module.UPSTREAM_PATCH_SHA256,
                    "tracked_diff_sha256": (
                        aggregator_module.UPSTREAM_PATCH_SHA256
                    ),
                    "patched_file_sha256": (
                        aggregator_module.UPSTREAM_PATCHED_FILE_SHA256
                    ),
                    "staged_tracked_changes": False,
                    "other_tracked_changes": False,
                },
                "strict_unseen_exact_match": exact,
                "geometry_protocol": geometry_protocol,
                "target_rgb_visible_during_gaussian_splatting_training": (
                    geometry_protocol == "released_all_view"
                ),
                "target_rgb_visible_during_uplifting": True,
                "target_view_2d_foundation_model_calls": True,
                "target_masks_scoring_only": True,
                "reference_mask_calibration": benchmark == "SPIn-NeRF",
                "aggregation": (
                    "frame_mean_then_equal_weight_macro_over_10_scenes"
                    if benchmark == "SPIn-NeRF"
                    else "equal_weight_macro_over_8_tasks"
                ),
                "cohort": (
                    list(launcher.SPIN_SCENES)
                    if benchmark == "SPIn-NeRF"
                    else list(NVOS_SCENES)
                ),
                "gs_source": str(checkpoint.resolve()),
                "gs_source_sha256": checkpoint_sha256,
                "released_all_view_training_provenance": training_provenance,
                "sam_checkpoint": str(sam_checkpoint.resolve()),
                "sam_checkpoint_sha256": (
                    aggregator_module.SAM_VIT_H_CHECKPOINT_SHA256
                ),
            }
        ),
        encoding="utf-8",
    )
    protocol_result = (
        {
            "schema_version": 1,
            "benchmark": "SPIn-NeRF",
            "scene": scene,
            "reference_frame": "image000.png",
            "metric": "foreground_iou",
            "selected_threshold_parameter": 0.4,
            "selected_sam_candidate": 0,
            "threshold_and_candidate_policy": (
                "maximize_reference_mask_iou_per_scene"
            ),
            "reference_scored": False,
            "frame_results": [
                {
                    "frame": "image000.png",
                    "role": "reference",
                    "foreground_iou": 0.99,
                },
                {
                    "frame": "image001.png",
                    "role": "target",
                    "foreground_iou": score,
                },
            ],
            "scene_mean_iou": score,
        }
        if benchmark == "SPIn-NeRF"
        else {
            "schema_version": 1,
            "benchmark": "NVOS",
            "scene": scene,
            "reference_frame": "reference.png",
            "target_frame": "target.png",
            "metric": "foreground_iou",
            "oracle_target_iou": 0.999,
            "oracle_target_threshold_parameter": 54.0,
            "selected_iou": score,
            "selected_threshold_parameter": 75.0,
            "threshold_policy": "fixed_across_nvos_scenes",
            "reference_scored": False,
        }
    )
    (result / "protocol_result.json").write_text(
        json.dumps(protocol_result),
        encoding="utf-8",
    )


def test_aggregation_averages_runs_then_scenes_and_stays_non_strict(
    tmp_path: Path,
) -> None:
    _write_fake_run(tmp_path, scene="fern", seed=0, score=0.80)
    _write_fake_run(tmp_path, scene="fern", seed=1, score=0.90)
    _write_fake_run(tmp_path, scene="flower", seed=0, score=0.95)
    _write_fake_run(tmp_path, scene="flower", seed=1, score=0.97)

    summary = aggregate(tmp_path, "nvos")

    assert summary["schema_version"] == 2
    assert summary["per_scene"]["fern"]["local_mean_iou_percent"] == pytest.approx(85.0)
    assert summary["per_scene"]["flower"]["local_mean_iou_percent"] == pytest.approx(
        96.0
    )
    assert summary["local_scene_macro_iou_percent"] == pytest.approx(90.5)
    assert summary["strict_unseen_exact_match"] is False
    assert summary["eligible_for_full_cohort_single_run_report"] is False
    assert summary["eligible_for_paper_protocol_comparison"] is False
    assert summary["all_scenes_have_three_runs"] is False
    assert summary["all_scenes_have_required_seeds"] is False
    assert summary["oracle_values_aggregated"] is False
    assert summary["metric_source"] == "selected_iou_fixed_threshold"


def test_aggregation_rejects_unsafe_strict_label(tmp_path: Path) -> None:
    _write_fake_run(tmp_path, scene="fern", seed=0, score=0.8, exact=True)
    with pytest.raises(AggregationError, match="Unsafe strict-unseen"):
        aggregate(tmp_path, "nvos")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("upstream_commit", "wrong", "Unpinned LUDVIG-SAM"),
        ("sam_checkpoint_sha256", "wrong", "Unpinned LUDVIG-SAM"),
        (
            "target_view_2d_foundation_model_calls",
            False,
            "online-multiview visibility mismatch",
        ),
        (
            "aggregation",
            "query micro",
            "calibration/aggregation mismatch",
        ),
    ),
)
def test_aggregation_rejects_unpinned_online_protocol_fields(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    _write_fake_run(tmp_path, scene="fern", seed=0, score=0.8)
    manifest_path = tmp_path / "fern" / "seed_0" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match=message):
        aggregate(tmp_path, "nvos")


@pytest.mark.parametrize(
    ("score", "message"),
    (
        (float("nan"), "must be finite"),
        (-0.01, r"outside \[0, 1\]"),
        (1.01, r"outside \[0, 1\]"),
        (True, "must be a JSON number"),
        ("0.8", "must be a JSON number"),
    ),
)
def test_aggregation_rejects_invalid_iou(
    tmp_path: Path,
    score,
    message: str,
) -> None:
    _write_fake_run(tmp_path, scene="fern", seed=0, score=score)

    with pytest.raises(AggregationError, match=message):
        aggregate(tmp_path, "nvos")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("benchmark", "SPIn-NeRF", "protocol identity mismatch"),
        ("scene", "flower", "protocol identity mismatch"),
        ("metric", "pixel_accuracy", "protocol identity mismatch"),
        ("reference_scored", True, "protocol identity mismatch"),
        ("selected_threshold_parameter", 13.0, "fixed threshold must be 75"),
        ("threshold_policy", "oracle", "frame/threshold policy mismatch"),
    ),
)
def test_aggregation_binds_nvos_score_to_result_protocol(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    _write_fake_run(tmp_path, scene="fern", seed=0, score=0.8)
    result_path = (
        tmp_path
        / "fern"
        / "seed_0"
        / "fern"
        / "sam"
        / "protocol_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(AggregationError, match=message):
        aggregate(tmp_path, "nvos")


def test_aggregation_rehashes_live_sam_checkpoint(tmp_path: Path) -> None:
    _write_fake_run(tmp_path, scene="fern", seed=0, score=0.8)
    sam_checkpoint = tmp_path / "_checkpoints" / "sam_vit_h.pth"
    sam_checkpoint.write_bytes(b"changed after launch")

    with pytest.raises(AggregationError, match="SAM checkpoint is missing or changed"):
        aggregate(tmp_path, "nvos")


def test_spin_aggregation_recomputes_target_frame_mean(tmp_path: Path) -> None:
    _write_fake_run(
        tmp_path,
        scene="fern",
        seed=0,
        score=0.8,
        benchmark="SPIn-NeRF",
    )
    result_path = (
        tmp_path
        / "fern"
        / "seed_0"
        / "fern"
        / "sam"
        / "protocol_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["scene_mean_iou"] = 0.9
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(AggregationError, match="does not match target frame_results"):
        aggregate(tmp_path, "spin")


def test_aggregation_rejects_mixed_geometry_protocols(tmp_path: Path) -> None:
    _write_fake_run(
        tmp_path,
        scene="fern",
        seed=0,
        score=0.8,
        geometry_protocol="strict_geometry_hybrid_diagnostic",
    )
    _write_fake_run(
        tmp_path,
        scene="flower",
        seed=0,
        score=0.9,
        geometry_protocol="released_all_view",
    )
    with pytest.raises(AggregationError, match="identical geometry_protocol"):
        aggregate(tmp_path, "nvos")


def test_aggregation_rejects_duplicate_completed_scene_seed(
    tmp_path: Path,
) -> None:
    _write_fake_run(
        tmp_path, scene="fern", seed=0, score=0.8, attempt="attempt_1"
    )
    _write_fake_run(
        tmp_path, scene="fern", seed=0, score=0.9, attempt="attempt_2"
    )

    with pytest.raises(AggregationError, match="Duplicate completed scene/seed"):
        aggregate(tmp_path, "nvos")


def test_aggregation_requires_exact_paper_seed_set(tmp_path: Path) -> None:
    for seed, score in ((1, 0.8), (2, 0.9), (3, 0.85)):
        _write_fake_run(tmp_path, scene="fern", seed=seed, score=score)

    summary = aggregate(tmp_path, "nvos")

    assert summary["all_scenes_have_three_runs"] is True
    assert summary["all_scenes_have_required_seeds"] is False
    assert summary["per_scene"]["fern"]["seeds"] == [1, 2, 3]


def test_released_all_view_fern_three_seeds_enable_per_scene_paper_check(
    tmp_path: Path,
) -> None:
    for seed, score in ((0, 0.84), (1, 0.86), (2, 0.855)):
        _write_fake_run(
            tmp_path,
            scene="fern",
            seed=seed,
            score=score,
            geometry_protocol="released_all_view",
        )

    summary = aggregate(tmp_path, "nvos")

    assert summary["complete_requested_cohort"] is False
    assert summary["eligible_for_paper_protocol_comparison"] is False
    assert (
        summary["eligible_for_per_scene_three_seed_paper_protocol_check"]
        is True
    )
    assert summary["per_scene"]["fern"][
        "eligible_for_three_seed_paper_protocol_check"
    ] is True
    assert summary["per_scene"]["fern"]["local_mean_iou_percent"] == pytest.approx(
        85.16666666666667
    )
    assert summary["oracle_values_aggregated"] is False
    assert summary["released_all_view_training_provenance_verified"] is True
    assert summary["all_runs_have_verified_upstream_patch"] is True


def test_released_horns_tasks_share_one_geometry_manifest_fail_closed(
    tmp_path: Path,
) -> None:
    for task_scene in ("horns_center", "horns_left"):
        for seed in (0, 1, 2):
            _write_fake_run(
                tmp_path,
                scene=task_scene,
                seed=seed,
                score=0.9,
                geometry_protocol="released_all_view",
            )

    summary = aggregate(tmp_path, "nvos")

    center = summary["per_scene"]["horns_center"]
    left = summary["per_scene"]["horns_left"]
    assert center["gs_source_sha256"] == left["gs_source_sha256"]
    assert center["training_manifest_sha256"] == left["training_manifest_sha256"]
    assert center["eligible_for_three_seed_paper_protocol_check"] is True
    assert left["eligible_for_three_seed_paper_protocol_check"] is True


def test_released_horns_tasks_reject_different_geometry_checkpoints(
    tmp_path: Path,
) -> None:
    for task_scene in ("horns_center", "horns_left"):
        _write_fake_run(
            tmp_path,
            scene=task_scene,
            seed=0,
            score=0.9,
            geometry_protocol="released_all_view",
        )

    original_training = tmp_path / "_training" / "horns" / "training_manifest.json"
    alternate_checkpoint = (
        tmp_path / "_checkpoints" / "horns_alternate" / "point_cloud.ply"
    )
    alternate_checkpoint.parent.mkdir(parents=True)
    alternate_checkpoint.write_bytes(b"different but individually valid horns checkpoint")
    alternate_checkpoint_hash = hashlib.sha256(
        alternate_checkpoint.read_bytes()
    ).hexdigest()
    alternate_training = (
        tmp_path / "_training" / "horns_alternate" / "training_manifest.json"
    )
    alternate_training.parent.mkdir(parents=True)
    training_payload = json.loads(original_training.read_text(encoding="utf-8"))
    training_payload["training_output"]["point_cloud"] = str(
        alternate_checkpoint.resolve()
    )
    training_payload["training_output"][
        "point_cloud_sha256"
    ] = alternate_checkpoint_hash
    alternate_training.write_text(json.dumps(training_payload), encoding="utf-8")

    left_manifest_path = tmp_path / "horns_left" / "seed_0" / "run_manifest.json"
    left_manifest = json.loads(left_manifest_path.read_text(encoding="utf-8"))
    left_manifest["gs_source"] = str(alternate_checkpoint.resolve())
    left_manifest["gs_source_sha256"] = alternate_checkpoint_hash
    left_provenance = left_manifest["released_all_view_training_provenance"]
    left_provenance["training_manifest"] = str(alternate_training.resolve())
    left_provenance["training_manifest_sha256"] = hashlib.sha256(
        alternate_training.read_bytes()
    ).hexdigest()
    left_provenance["point_cloud_sha256"] = alternate_checkpoint_hash
    left_manifest_path.write_text(json.dumps(left_manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match="sharing a released-all-view geometry"):
        aggregate(tmp_path, "nvos")


def test_released_horns_tasks_reject_different_training_manifests(
    tmp_path: Path,
) -> None:
    for task_scene in ("horns_center", "horns_left"):
        _write_fake_run(
            tmp_path,
            scene=task_scene,
            seed=0,
            score=0.9,
            geometry_protocol="released_all_view",
        )

    original_training = tmp_path / "_training" / "horns" / "training_manifest.json"
    alternate_training = (
        tmp_path / "_training" / "horns_alternate" / "training_manifest.json"
    )
    alternate_training.parent.mkdir(parents=True)
    training_payload = json.loads(original_training.read_text(encoding="utf-8"))
    training_payload["audit_variant"] = "distinct manifest for the same checkpoint"
    alternate_training.write_text(json.dumps(training_payload), encoding="utf-8")

    left_manifest_path = tmp_path / "horns_left" / "seed_0" / "run_manifest.json"
    left_manifest = json.loads(left_manifest_path.read_text(encoding="utf-8"))
    left_provenance = left_manifest["released_all_view_training_provenance"]
    left_provenance["training_manifest"] = str(alternate_training.resolve())
    left_provenance["training_manifest_sha256"] = hashlib.sha256(
        alternate_training.read_bytes()
    ).hexdigest()
    left_manifest_path.write_text(json.dumps(left_manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match="sharing a released-all-view geometry"):
        aggregate(tmp_path, "nvos")


def test_released_aggregation_enforces_geometry_view_count(tmp_path: Path) -> None:
    _write_fake_run(
        tmp_path,
        scene="flower",
        seed=0,
        score=0.9,
        geometry_protocol="released_all_view",
    )
    training_path = tmp_path / "_training" / "flower" / "training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["effective_training_protocol"]["registered_training_views"] = 20
    training["training_output"]["registered_all_view_cameras"] = 20
    training_path.write_text(json.dumps(training), encoding="utf-8")
    run_manifest_path = tmp_path / "flower" / "seed_0" / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["released_all_view_training_provenance"][
        "training_manifest_sha256"
    ] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match="expected 34, found 20"):
        aggregate(tmp_path, "nvos")


def test_released_horns_task_rejects_wrong_geometry_scene_mapping(
    tmp_path: Path,
) -> None:
    _write_fake_run(
        tmp_path,
        scene="horns_center",
        seed=0,
        score=0.9,
        geometry_protocol="released_all_view",
    )
    manifest_path = tmp_path / "horns_center" / "seed_0" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["geometry_scene"] = "horns_center"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match="mapping mismatch"):
        aggregate(tmp_path, "nvos")


def test_released_all_view_requires_patch_provenance_for_paper_check(
    tmp_path: Path,
) -> None:
    for seed, score in ((0, 0.84), (1, 0.86), (2, 0.855)):
        _write_fake_run(
            tmp_path,
            scene="fern",
            seed=seed,
            score=score,
            geometry_protocol="released_all_view",
        )
        manifest_path = tmp_path / "fern" / f"seed_{seed}" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["upstream_patch_provenance"] = None
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = aggregate(tmp_path, "nvos")

    assert summary["released_all_view_training_provenance_verified"] is True
    assert summary["all_runs_have_verified_upstream_patch"] is False
    assert (
        summary["eligible_for_per_scene_three_seed_paper_protocol_check"]
        is False
    )


def test_full_released_nvos_is_not_mislabeled_as_hybrid_report(
    tmp_path: Path,
) -> None:
    for scene in NVOS_SCENES:
        for seed in (0, 1, 2):
            _write_fake_run(
                tmp_path,
                scene=scene,
                seed=seed,
                score=0.9,
                geometry_protocol="released_all_view",
            )

    summary = aggregate(tmp_path, "nvos")

    assert summary["eligible_for_paper_protocol_comparison"] is True
    assert summary["eligible_for_full_cohort_three_seed_hybrid_report"] is False


def test_full_released_nvos_accepts_mixed_approved_patch_provenance(
    tmp_path: Path,
) -> None:
    for scene in NVOS_SCENES:
        for seed in (0, 1, 2):
            _write_fake_run(
                tmp_path,
                scene=scene,
                seed=seed,
                score=0.9,
                geometry_protocol="released_all_view",
            )
            if scene == "trex":
                continue
            manifest_path = (
                tmp_path / scene / f"seed_{seed}" / "run_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance = manifest["upstream_patch_provenance"]
            if seed == 2:
                patch_sha256 = aggregator_module.UPSTREAM_PATCH_V2_SHA256
                patched_files = (
                    aggregator_module.UPSTREAM_PATCHED_FILE_V2_SHA256
                )
            else:
                patch_sha256 = aggregator_module.UPSTREAM_PATCH_V1_SHA256
                patched_files = (
                    aggregator_module.UPSTREAM_PATCHED_FILE_V1_SHA256
                )
            provenance["patch_sha256"] = patch_sha256
            provenance["tracked_diff_sha256"] = patch_sha256
            provenance["patched_file_sha256"] = patched_files
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = aggregate(tmp_path, "nvos")

    assert summary["complete_requested_cohort"] is True
    assert summary["all_runs_have_verified_upstream_patch"] is True
    assert summary["eligible_for_paper_protocol_comparison"] is True


def test_aggregation_rejects_unapproved_patch_provenance(
    tmp_path: Path,
) -> None:
    _write_fake_run(
        tmp_path,
        scene="fern",
        seed=0,
        score=0.84,
        geometry_protocol="released_all_view",
    )
    manifest_path = tmp_path / "fern" / "seed_0" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["upstream_patch_provenance"]
    provenance["patch_sha256"] = "0" * 64
    provenance["tracked_diff_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match="patch provenance mismatch"):
        aggregate(tmp_path, "nvos")


def test_released_all_view_aggregation_rejects_unbound_training_provenance(
    tmp_path: Path,
) -> None:
    _write_fake_run(
        tmp_path,
        scene="fern",
        seed=0,
        score=0.84,
        geometry_protocol="released_all_view",
    )
    manifest_path = tmp_path / "fern" / "seed_0" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["released_all_view_training_provenance"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AggregationError, match="verified training provenance"):
        aggregate(tmp_path, "nvos")


def test_spin_nine_scene_diagnostic_never_becomes_full_paper_comparison(
    tmp_path: Path,
) -> None:
    available_scenes = (
        "orchids",
        "leaves",
        "fern",
        "room",
        "horns",
        "fortress",
        "pinecone",
        "truck",
        "lego",
    )
    for scene in available_scenes:
        for seed in (0, 1, 2):
            _write_fake_run(
                tmp_path,
                scene=scene,
                seed=seed,
                score=0.9,
                geometry_protocol="released_all_view",
                benchmark="SPIn-NeRF",
            )

    summary = aggregate(tmp_path, "spin")

    assert summary["complete_requested_cohort"] is True
    assert summary["all_scenes_have_required_seeds"] is True
    assert summary["released_all_view_training_provenance_verified"] is True
    assert summary["eligible_for_paper_protocol_comparison"] is False
    assert summary["eligible_for_full_cohort_three_seed_hybrid_report"] is False


def test_spin_accepts_only_hashed_verified_nvos_llff_geometry_reuse(
    tmp_path: Path,
) -> None:
    _write_fake_run(
        tmp_path,
        scene="fern",
        seed=0,
        score=0.9,
        geometry_protocol="released_all_view",
        benchmark="SPIn-NeRF",
    )
    training_path = tmp_path / "_training" / "fern" / "training_manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["benchmark"] = "NVOS"
    training.pop("geometry_scene")
    training_path.write_text(json.dumps(training), encoding="utf-8")

    manifest_path = tmp_path / "fern" / "seed_0" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staging = {
        "strategy": "reuse_verified_identical_llff_colmap_undistortion",
        "raw_scene_identity_proven": True,
        "raw_sparse_sha256": {"cameras.bin": "a", "images.bin": "b"},
    }
    staging_sha256 = hashlib.sha256(
        json.dumps(staging, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["colmap_staging"] = staging
    provenance = manifest["released_all_view_training_provenance"]
    provenance["training_manifest_sha256"] = hashlib.sha256(
        training_path.read_bytes()
    ).hexdigest()
    provenance["legacy_geometry_scene_fallback"] = True
    provenance["cross_benchmark_asset_reuse"] = {
        "verified": True,
        "training_benchmark": "NVOS",
        "evaluation_benchmark": "SPIn-NeRF",
        "geometry_scene": "fern",
        "colmap_staging_sha256": staging_sha256,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = aggregate(tmp_path, "spin")
    assert summary["per_scene"]["fern"]["local_mean_iou_percent"] == 90.0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["released_all_view_training_provenance"][
        "cross_benchmark_asset_reuse"
    ]["colmap_staging_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AggregationError, match="legacy fallback"):
        aggregate(tmp_path, "spin")


def test_hybrid_cohort_grid_is_eight_scenes_by_three_unique_seeds() -> None:
    tasks = _task_grid()

    assert len(tasks) == 24
    assert len(set(tasks)) == 24
    assert {scene for scene, _seed in tasks} == set(NVOS_SCENES)
    assert {seed for _scene, seed in tasks} == {0, 1, 2}


def test_hybrid_cohort_reuses_one_exact_completed_scene_seed(
    tmp_path: Path,
) -> None:
    _write_fake_run(
        tmp_path,
        scene="fern",
        seed=0,
        score=0.8401535329822478,
        attempt="pilot",
    )

    completed = _completed_attempt(
        tmp_path / "fern" / "seed_0", "fern", 0
    )

    assert completed is not None
    assert completed["selected_iou"] == pytest.approx(0.8401535329822478)
    assert completed["attempt_id"] is None


def test_hybrid_cohort_attempt_ids_never_reuse_existing_directory(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    (attempts / f"{ATTEMPT_PREFIX}1").mkdir(parents=True)
    (attempts / f"{ATTEMPT_PREFIX}3").mkdir()
    (attempts / "unrelated_pilot").mkdir()

    assert _next_attempt_id(tmp_path) == f"{ATTEMPT_PREFIX}2"


def test_full_hybrid_cohort_requires_8_scenes_x_exact_3_seeds_and_never_oracle(
    tmp_path: Path,
) -> None:
    for scene_index, scene in enumerate(NVOS_SCENES):
        for seed in (0, 1, 2):
            _write_fake_run(
                tmp_path,
                scene=scene,
                seed=seed,
                score=0.80 + 0.001 * scene_index + 0.0001 * seed,
            )

    summary = aggregate(tmp_path, "nvos")

    assert summary["complete_requested_cohort"] is True
    assert summary["all_scenes_have_required_seeds"] is True
    assert summary["eligible_for_full_cohort_three_seed_hybrid_report"] is True
    assert summary["eligible_for_paper_protocol_comparison"] is False
    assert summary["oracle_values_aggregated"] is False
    assert all(row["num_runs"] == 3 for row in summary["per_scene"].values())
    assert all(row["seeds"] == [0, 1, 2] for row in summary["per_scene"].values())
    assert summary["per_scene"]["fern"]["local_mean_iou_percent"] == pytest.approx(
        80.01
    )


def test_official_3dgs_lock_has_complete_hashes_and_original_defaults() -> None:
    lock = json.loads(OFFICIAL_3DGS_LOCK_FILE.read_text(encoding="utf-8"))

    assert lock["commit"] == OFFICIAL_3DGS_COMMIT
    assert lock["selection"]["ludvig_gaussiansplatting_is_gitlink"] is False
    assert lock["selection"]["common_tracked_files_compared"] == 30
    assert lock["selection"]["byte_identical_files"] == 20
    assert lock["training_defaults"]["iterations"] == 30_000
    assert lock["training_defaults"]["position_lr_init"] == pytest.approx(0.00016)
    assert lock["training_defaults"]["position_lr_final"] == pytest.approx(
        0.0000016
    )
    assert lock["training_defaults"]["feature_lr"] == pytest.approx(0.0025)
    assert lock["training_defaults"]["densify_until_iter"] == 15_000
    assert lock["model_defaults"]["resolution"] == -1
    assert lock["model_defaults"]["eval"] is False
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in lock["source_sha256"].values()
    )


def test_literal_default_extraction_and_all_view_training_command(
    tmp_path: Path,
) -> None:
    source = """
class OptimizationParams:
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        super().__init__()
"""
    defaults = _literal_class_defaults(source, "OptimizationParams")
    command = _training_command(
        tmp_path / "python",
        tmp_path / "upstream",
        tmp_path / "staged",
        tmp_path / "model",
    )

    assert defaults == {"iterations": 30_000, "position_lr_init": 0.00016}
    assert command[command.index("--iterations") + 1] == "30000"
    assert command[command.index("--test_iterations") + 1] == "-1"
    assert command[command.index("--save_iterations") + 1] == "30000"
    assert "--eval" not in command
    assert "--resolution" not in command


def test_nvos_geometry_training_specs_and_output_roots_are_scene_safe(
    tmp_path: Path,
) -> None:
    assert NVOS_GEOMETRY_SCENES == (
        "fern",
        "flower",
        "fortress",
        "horns",
        "leaves",
        "orchids",
        "trex",
    )
    assert NVOS_GEOMETRY_REGISTERED_IMAGES == {
        "fern": 20,
        "flower": 34,
        "fortress": 42,
        "horns": 62,
        "leaves": 26,
        "orchids": 25,
        "trex": 55,
    }
    assert _training_output_root("fern", None) == (
        DEFAULT_NVOS_TRAINING_OUTPUT_ROOT.resolve()
    )
    assert _training_output_root("flower", None).parts[-2:] == (
        "flower",
        "training",
    )
    explicit_fern_root = tmp_path / "existing-layout" / "fern" / "training"
    assert _training_output_root("fern", explicit_fern_root) == (
        explicit_fern_root.resolve()
    )
    with pytest.raises(TrainingProtocolError, match="Unknown NVOS geometry"):
        _training_output_root("horns_center", None)


def test_original_3dgs_auto_resolution_preserves_trex_float_edge() -> None:
    assert _automatic_training_resolution(3985, 2988) == (1600, 1199)
    assert _automatic_training_resolution(4002, 3001) == (1599, 1199)
    with pytest.raises(TrainingProtocolError, match="Invalid source resolution"):
        _automatic_training_resolution(0, 3001)


def test_training_output_validation_uses_scene_view_count_and_source_size(
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
    cameras = [
        {"width": 3982, "height": 2986, "id": index}
        for index in range(NVOS_GEOMETRY_REGISTERED_IMAGES["flower"])
    ]
    (model / "cameras.json").write_text(
        json.dumps(cameras),
        encoding="utf-8",
    )

    output = _validate_training_output(
        run_dir,
        model,
        expected_registered_images=NVOS_GEOMETRY_REGISTERED_IMAGES["flower"],
        expected_source_resolution=(3982, 2986),
    )
    assert output["registered_all_view_cameras"] == 34
    assert output["source_camera_resolution"] == [3982, 2986]
    assert output["effective_training_resolution"] == [1600, 1199]

    with pytest.raises(TrainingProtocolError, match="Expected 35 all-view cameras"):
        _validate_training_output(
            run_dir,
            model,
            expected_registered_images=35,
            expected_source_resolution=(3982, 2986),
        )
    with pytest.raises(
        TrainingProtocolError,
        match="staged COLMAP metadata resolution",
    ):
        _validate_training_output(
            run_dir,
            model,
            expected_registered_images=34,
            expected_source_resolution=(4003, 3002),
        )


def test_training_output_parsers_are_fail_closed(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg_args"
    cfg.write_text(
        "Namespace(eval=False, resolution=-1, source_path='/scene')",
        encoding="utf-8",
    )
    ply = tmp_path / "point_cloud.ply"
    ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 123\nproperty float x\nend_header\npayload"
    )

    assert _parse_namespace(cfg) == {
        "eval": False,
        "resolution": -1,
        "source_path": "/scene",
    }
    assert _parse_ply_vertex_count(ply) == 123

    invalid = tmp_path / "invalid.ply"
    invalid.write_bytes(b"ply\nend_header\n")
    with pytest.raises(TrainingProtocolError, match="Invalid or empty 3DGS PLY"):
        _parse_ply_vertex_count(invalid)
