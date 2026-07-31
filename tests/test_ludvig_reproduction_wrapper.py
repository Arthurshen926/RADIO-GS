import json
from pathlib import Path
import re
import struct

from PIL import Image
import pytest

import reproductions.ludvig.run_ludvig_sam as launcher
from reproductions.ludvig.aggregate_results import AggregationError, aggregate
from reproductions.ludvig.run_nvos_hybrid_cohort import (
    ATTEMPT_PREFIX,
    _completed_attempt,
    _next_attempt_id,
    _task_grid,
)
from reproductions.ludvig.run_ludvig_sam import (
    NVOS_SCENES,
    ProtocolError,
    _runtime_environment,
    _scaled_pinhole_audit,
    _resolve_inputs,
    _stage_nvos_pinhole_colmap,
    _stage_spin_llff_pinhole_colmap,
    _validate_upstream,
)
from reproductions.ludvig.train_nvos_all_view_3dgs import (
    LOCK_FILE as OFFICIAL_3DGS_LOCK_FILE,
    OFFICIAL_3DGS_COMMIT,
    TrainingProtocolError,
    _literal_class_defaults,
    _parse_namespace,
    _parse_ply_vertex_count,
    _training_command,
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

    _validate_upstream(checkout)


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

    (converted_raw_sparse / "points3D.bin").write_bytes(b"changed")
    with pytest.raises(ProtocolError, match="same raw points3D.bin"):
        _stage_spin_llff_pinhole_colmap(
            spin,
            converted_source,
            tmp_path / "unused",
            16,
            12,
        )


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
    attempt=None,
) -> None:
    run = root / scene / f"seed_{seed}"
    if attempt is not None:
        run = run / "attempts" / attempt
    result = run / scene / "sam"
    result.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "benchmark": "NVOS",
                "scene": scene,
                "seed": seed,
                "protocol_id": "ludvig_official_online_multiview_v1",
                "strict_unseen_exact_match": exact,
                "geometry_protocol": geometry_protocol,
                "target_rgb_visible_during_gaussian_splatting_training": (
                    geometry_protocol == "released_all_view"
                ),
                "target_rgb_visible_during_uplifting": True,
                "gs_source_sha256": f"{scene}-checkpoint",
            }
        ),
        encoding="utf-8",
    )
    (result / "protocol_result.json").write_text(
        json.dumps(
            {
                "selected_iou": score,
                "oracle_target_iou": 0.999,
            }
        ),
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
