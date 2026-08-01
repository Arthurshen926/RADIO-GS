from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import (
    CAMERA_AXES,
    FIELD_CONTRACT_VERSION,
    FIELD_SOURCE_POLICY,
    FROZEN_SOURCE_ADAPTER_LEDGER_FILENAME_BY_SCENE,
    FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE,
    INTRINSICS_MODEL,
    LudvigPFPRPhaseAError,
    METHOD_QUERY_KEYS,
    OFFICIAL_DINO_CHECKPOINT_NAME,
    POSE_CONVENTION,
    POSE_FIXTURE_CAMERA_XYZ,
    PhaseAConfig,
    SOURCE_ADAPTER_AUDIT_SCOPE,
    SOURCE_ADAPTER_LEDGER_VERSION,
    SOURCE_ADAPTER_TRUST_MODEL,
    SOURCE_ADAPTER_TRUST_STATEMENT,
    SOURCE_INVENTORY_CANONICALIZATION,
    SOURCE_INVENTORY_RECORD_SCHEMA,
    UPSTREAM_AUDIT_FILES,
    audit_benchmark,
    audit_checkpoint,
    audit_field_contract,
    reject_unimplemented_phase,
    run_phase_a,
    sha256_file,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import (
    PFPR_V2_BENCHMARK_VERSION,
    ProtocolConfig,
    canonical_json_sha256,
)


SCENE = "scene0050_02"
VIEW_COUNT = 120
FIELD_FRAME_COUNT = 123
CHECKPOINT_BYTES = b"synthetic DINOv2 ViT-g-reg checkpoint"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_ply(path: Path, vertices: int = 2) -> None:
    properties = (
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    lines = ["ply", "format ascii 1.0", f"element vertex {vertices}"]
    lines.extend(f"property float {name}" for name in properties)
    lines.append("end_header")
    lines.extend(" ".join("0" for _ in properties) for _ in range(vertices))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _make_upstream(path: Path) -> str:
    for relative in UPSTREAM_AUDIT_FILES:
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"audited {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=PFPR Test",
            "-c",
            "user.email=pfpr@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _binding(path: Path, relative_path: str) -> dict:
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_source_inventory(source: Path, frame_ids: list[int]) -> dict:
    records = []
    modality_records = {"color": [], "depth": [], "pose": []}
    for frame_id in frame_ids:
        stem = f"{frame_id:06d}"
        relative_paths = {
            "color": f"color/{stem}.jpg",
            "depth": f"depth/{stem}.png",
            "pose": f"pose/{stem}.txt",
        }
        modalities = {}
        for role, relative in relative_paths.items():
            item = _binding(source / relative, relative)
            modalities[role] = item
            modality_records[role].append({"frame_id": frame_id, **item})
        records.append({"frame_id": frame_id, "modalities": modalities})
    return {
        "canonicalization": SOURCE_INVENTORY_CANONICALIZATION,
        "record_schema": SOURCE_INVENTORY_RECORD_SCHEMA,
        "record_count": len(records),
        "canonical_records_sha256": canonical_json_sha256(records),
        "modality_records_sha256": {
            role: canonical_json_sha256(items)
            for role, items in modality_records.items()
        },
    }


def _pose_fixture(source: Path, frame_id: int, depth_k: np.ndarray) -> dict:
    c2w = np.loadtxt(source / "pose" / f"{frame_id:06d}.txt").reshape(4, 4)
    camera_xyz = np.asarray(POSE_FIXTURE_CAMERA_XYZ, dtype=np.float64)
    world_xyz = (c2w @ np.append(camera_xyz, 1.0))[:3]
    pixel_uv = np.array(
        [
            depth_k[0, 0] * camera_xyz[0] / camera_xyz[2] + depth_k[0, 2],
            depth_k[1, 1] * camera_xyz[1] / camera_xyz[2] + depth_k[1, 2],
        ]
    )
    return {
        "frame_id": frame_id,
        "camera_center_world": c2w[:3, 3].tolist(),
        "camera_xyz": camera_xyz.tolist(),
        "world_xyz": world_xyz.tolist(),
        "pixel_uv": pixel_uv.tolist(),
    }


def _write_source_adapter_ledger(
    ledger_path: Path,
    *,
    source: Path,
    field_contract_path: Path,
    frame_ids: list[int],
) -> None:
    contract = json.loads(field_contract_path.read_text())
    depth_k = np.loadtxt(source / "intrinsics_depth.txt").reshape(4, 4)
    fixture_ids = [frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]]
    ledger = {
        "schema_version": SOURCE_ADAPTER_LEDGER_VERSION,
        "scene_id": SCENE,
        "trust_model": SOURCE_ADAPTER_TRUST_MODEL,
        "provenance": {
            "audit_date_utc": "2026-08-01",
            "audit_scope": SOURCE_ADAPTER_AUDIT_SCOPE,
            "evaluator_private_manifest_opened": False,
            "field_contract_sha256": sha256_file(field_contract_path),
            "materialization_mode": contract["materialization_mode"],
            "query_private_anchor_pose_depth_used": False,
            "source_sens_sha256": contract["source_sens_sha256"],
            "statement": SOURCE_ADAPTER_TRUST_STATEMENT,
        },
        "coverage_prefix": {
            "count": len(frame_ids),
            "ordered_frame_ids_sha256": canonical_json_sha256(frame_ids),
        },
        "selected_source_inventory": _canonical_source_inventory(source, frame_ids),
        "camera_contract": {
            "pose_convention": POSE_CONVENTION,
            "camera_axes": CAMERA_AXES,
            "intrinsics_model": INTRINSICS_MODEL,
            "image_dimensions": contract["color_size"],
            "intrinsics_bindings": {
                "depth": _binding(
                    source / "intrinsics_depth.txt", "intrinsics_depth.txt"
                ),
                "color": _binding(
                    source / "intrinsics_color.txt", "intrinsics_color.txt"
                ),
            },
            "pose_fixtures": [
                _pose_fixture(source, frame_id, depth_k) for frame_id in fixture_ids
            ],
        },
    }
    _write_json(ledger_path, ledger)


def _refresh_source_adapter_inventory(
    config: PhaseAConfig, frame_ids: list[int]
) -> PhaseAConfig:
    ledger = json.loads(config.source_adapter_ledger.read_text())
    ledger["selected_source_inventory"] = _canonical_source_inventory(
        config.source_scene, frame_ids
    )
    _write_json(config.source_adapter_ledger, ledger)
    return replace(
        config,
        expected_source_adapter_ledger_sha256=sha256_file(
            config.source_adapter_ledger
        ),
    )


def _make_asset(tmp_path: Path) -> tuple[PhaseAConfig, list[int]]:
    benchmark = tmp_path / "benchmark"
    source = tmp_path / "source" / SCENE
    geometry = tmp_path / "geometry" / "point_cloud.ply"
    checkpoint = tmp_path / "weights" / OFFICIAL_DINO_CHECKPOINT_NAME
    upstream = tmp_path / "LUDVIG"
    output = tmp_path / "attempt"
    benchmark.mkdir(parents=True)
    source.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(CHECKPOINT_BYTES)
    commit = _make_upstream(upstream)

    candidate = benchmark / "candidates" / f"{SCENE}.npy"
    candidate.parent.mkdir()
    np.save(
        candidate,
        np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float32,
        ),
    )
    candidate_hash = sha256_file(candidate)
    queries = []
    public_queries = []
    for index in range(10):
        query_id = f"{SCENE}_pfpr_{index:03d}"
        crop = benchmark / "queries" / "rgb" / f"{query_id}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), color=(index, 2 * index, 3 * index)).save(crop)
        crop_hash = sha256_file(crop)
        query = {
            "available_method_inputs": ["scene_id", "crop_rgb"],
            "benchmark_version": PFPR_V2_BENCHMARK_VERSION,
            "crop_rgb_path": str(crop.resolve()),
            "crop_rgb_sha256": crop_hash,
            "query_id": query_id,
            "scene_id": SCENE,
        }
        assert set(query) == METHOD_QUERY_KEYS
        queries.append(query)
        public_queries.append(
            {
                "crop_rgb_sha256": crop_hash,
                "query_depth_used_by_method": False,
                "query_id": query_id,
                "query_mask_used_by_method": False,
                "query_pose_used_by_method": False,
                "scene_id": SCENE,
            }
        )
    exclusion_digest = "e" * 64
    domain = {
        "candidate_points": 3,
        "candidate_xyz_path": str(candidate.resolve()),
        "candidate_xyz_sha256": candidate_hash,
        "excluded_query_source_frame_ids_sha256": exclusion_digest,
        "geometry_only": True,
        "scene_id": SCENE,
    }
    protocol_config = asdict(ProtocolConfig())
    common = {
        "benchmark_version": PFPR_V2_BENCHMARK_VERSION,
        "protocol_config": protocol_config,
        "scene_domains": [domain],
        "source_pfir_public_manifest": "/immutable/source.json",
        "source_pfir_public_manifest_sha256": "f" * 64,
    }
    method = {**common, "queries": queries}
    public = {
        **common,
        "queries": public_queries,
        "scene_reports": [
            {
                "eligible_anchor_count": 100,
                "field_frame_count": FIELD_FRAME_COUNT,
                "held_out_frame_count": 10,
                "scene_id": SCENE,
                "selected_anchor_count": 10,
            }
        ],
    }
    method_path = benchmark / "manifest.method.json"
    public_path = benchmark / "manifest.public.json"
    _write_json(method_path, method)
    _write_json(public_path, public)

    # Deliberately reverse frame IDs so the test detects accidental sorted
    # selection instead of the required coverage prefix.
    coverage_order = list(reversed(range(1000, 1000 + FIELD_FRAME_COUNT)))
    selected = sorted(coverage_order)
    contract = {
        "field_contract_version": FIELD_CONTRACT_VERSION,
        "scene_id": SCENE,
        "source_policy": FIELD_SOURCE_POLICY,
        "source_sens_sha256": "a" * 64,
        "materialization_mode": "decoded_sens",
        "field_frame_count": FIELD_FRAME_COUNT,
        "max_field_frames": FIELD_FRAME_COUNT,
        "frame_selection_policy": "depth_voxel_coverage",
        "selection_order_frame_indices": coverage_order,
        "selected_frame_indices": selected,
        "field_frame_manifest_sha256": canonical_json_sha256(selected),
        "excluded_query_source_frame_ids_sha256": exclusion_digest,
        "color_size": [8, 6],
        "source_color_size": [16, 12],
        "uses_private_anchor": False,
        "uses_private_depth_pixel": False,
        "uses_instances_or_semantic_labels": False,
        "contains_instance_or_label_directories": False,
    }
    contract_path = source / "field_source_contract.json"
    _write_json(contract_path, contract)
    depth_k = np.array(
        [[4.0, 0.0, 3.5, 0.0], [0.0, 4.5, 2.5, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    color_k = np.array(
        [[9.0, 0.0, 7.5, 0.0], [0.0, 9.5, 5.5, 0.0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    np.savetxt(source / "intrinsics_depth.txt", depth_k, fmt="%.12f")
    np.savetxt(source / "intrinsics_color.txt", color_k, fmt="%.12f")
    for folder in ("color", "depth", "pose"):
        (source / folder).mkdir()
    for frame_id in selected:
        stem = f"{frame_id:06d}"
        Image.new("RGB", (8, 6), color=(frame_id % 255, 20, 40)).save(
            source / "color" / f"{stem}.jpg"
        )
        Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(
            source / "depth" / f"{stem}.png"
        )
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [frame_id / 1000.0, -0.2, 1.0]
        np.savetxt(source / "pose" / f"{stem}.txt", c2w, fmt="%.12f")

    prefix = coverage_order[:VIEW_COUNT]
    source_adapter_ledger = tmp_path / "trusted" / "source_adapter_ledger.json"
    _write_source_adapter_ledger(
        source_adapter_ledger,
        source=source,
        field_contract_path=contract_path,
        frame_ids=prefix,
    )
    _write_ply(geometry, vertices=2)
    return (
        PhaseAConfig(
            scene_id=SCENE,
            benchmark_dir=benchmark,
            source_scene=source,
            field_contract_sha256=sha256_file(contract_path),
            source_adapter_ledger=source_adapter_ledger,
            expected_source_adapter_ledger_sha256=sha256_file(
                source_adapter_ledger
            ),
            geometry_ply=geometry,
            geometry_sha256=sha256_file(geometry),
            dino_checkpoint=checkpoint,
            ludvig_upstream=upstream,
            output_dir=output,
            view_count=VIEW_COUNT,
            expected_field_frame_count=FIELD_FRAME_COUNT,
            expected_gaussian_count=2,
            expected_method_manifest_sha256=sha256_file(method_path),
            expected_public_manifest_sha256=sha256_file(public_path),
            expected_checkpoint_size=len(CHECKPOINT_BYTES),
            expected_checkpoint_sha256=hashlib.sha256(CHECKPOINT_BYTES).hexdigest(),
            expected_ludvig_commit=commit,
        ),
        prefix,
    )


def test_phase_a_stages_coverage_prefix_with_depth_intrinsics_and_manifest(
    tmp_path: Path,
) -> None:
    config, coverage_order = _make_asset(tmp_path)

    manifest = run_phase_a(config, argv=["run_pfpr_dinov2.py", "--phase", "phase-a"])

    saved = json.loads((config.output_dir / "run_manifest.json").read_text())
    assert saved == manifest
    assert manifest["status"] == "phase_a_complete_phase_b_available_not_run"
    assert manifest["result_eligible"] is False
    assert manifest["gpu_work_started"] is False
    assert manifest["torch_imported_by_phase_a"] is False
    assert manifest["evaluator_private_manifest_opened"] is False
    assert manifest["view_selection"]["ordered_frame_ids"] == coverage_order
    assert manifest["view_selection"]["ordered_frame_ids_sha256"] == (
        canonical_json_sha256(coverage_order)
    )
    assert (
        manifest["camera_intrinsics"]["selected_role"]
        == "depth_intrinsics_for_8x6_depth_aligned_RGB"
    )
    assert manifest["camera_intrinsics"]["fx"] == 4.0
    assert manifest["camera_intrinsics"]["rejected_original_color_sha256"] != (
        manifest["camera_intrinsics"]["selected_sha256"]
    )
    assert manifest["colmap_staging"]["roundtrip"]["passed"] is True
    assert manifest["source_adapter_ledger"]["pose_convention"] == POSE_CONVENTION
    assert manifest["source_adapter_ledger"]["camera_axes"] == CAMERA_AXES
    assert manifest["source_adapter_ledger"]["pose_fixtures_passed"] is True
    assert manifest["phase_status"]["phase_b_dino_scene_features_and_pca"] == (
        "available_separate_not_run"
    )

    staged_images = sorted((config.output_dir / "staging" / "colmap" / "images").iterdir())
    assert len(staged_images) == VIEW_COUNT
    assert staged_images[0].name == f"000000_{coverage_order[0]:06d}.jpg"
    assert staged_images[0].is_symlink()
    cameras = (
        config.output_dir / "staging" / "colmap" / "sparse" / "0" / "cameras.txt"
    ).read_text()
    assert "1 PINHOLE 8 6 4 4.5 3.5 2.5" in cameras
    assert " 9 " not in cameras


def test_checkpoint_validation_is_size_and_sha_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / OFFICIAL_DINO_CHECKPOINT_NAME
    checkpoint.write_bytes(b"same-size")
    expected = hashlib.sha256(b"different").hexdigest()

    with pytest.raises(LudvigPFPRPhaseAError, match="SHA-256 mismatch"):
        audit_checkpoint(
            checkpoint,
            expected_size=len(b"same-size"),
            expected_sha256=expected,
        )
    with pytest.raises(LudvigPFPRPhaseAError, match="size mismatch"):
        audit_checkpoint(
            checkpoint,
            expected_size=len(b"same-size") + 1,
            expected_sha256=hashlib.sha256(b"same-size").hexdigest(),
        )


def test_repo_frozen_scene0050_source_adapter_ledger_hash() -> None:
    filename = FROZEN_SOURCE_ADAPTER_LEDGER_FILENAME_BY_SCENE[SCENE]
    ledger = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "ludvig"
        / "receipts"
        / filename
    )

    assert sha256_file(ledger) == FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE[SCENE]


def test_wrapper_injects_repo_frozen_source_adapter_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper_path = (
        Path(__file__).resolve().parents[1]
        / "reproductions"
        / "ludvig"
        / "run_pfpr_dinov2.py"
    )
    spec = importlib.util.spec_from_file_location("ludvig_pfpr_wrapper_test", wrapper_path)
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    args = wrapper.parser().parse_args(
        [
            "--scene",
            SCENE,
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
            "--source-scene",
            str(tmp_path / "source"),
            "--field-contract-sha256",
            "a" * 64,
            "--geometry-ply",
            str(tmp_path / "geometry.ply"),
            "--geometry-sha256",
            "b" * 64,
            "--dino-checkpoint",
            str(tmp_path / OFFICIAL_DINO_CHECKPOINT_NAME),
            "--output-dir",
            str(tmp_path / "attempt"),
        ]
    )
    monkeypatch.setattr(wrapper, "run_phase_a", lambda config, argv=None: config)

    config = wrapper.run(args, argv=[])

    assert config.expected_source_adapter_ledger_sha256 == (
        FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE[SCENE]
    )
    assert config.source_adapter_ledger == (
        wrapper_path.parent
        / "receipts"
        / FROZEN_SOURCE_ADAPTER_LEDGER_FILENAME_BY_SCENE[SCENE]
    )


def test_field_contract_requires_public_query_exclusion_digest(tmp_path: Path) -> None:
    config, _coverage = _make_asset(tmp_path)

    with pytest.raises(LudvigPFPRPhaseAError, match="exclusion commitment mismatch"):
        audit_field_contract(
            config.source_scene,
            SCENE,
            "a" * 64,
            expected_sha256=config.field_contract_sha256,
            expected_frame_count=FIELD_FRAME_COUNT,
            view_count=VIEW_COUNT,
        )


def test_method_manifest_rejects_private_anchor_even_when_rehashed(tmp_path: Path) -> None:
    config, _coverage = _make_asset(tmp_path)
    path = config.benchmark_dir / "manifest.method.json"
    payload = json.loads(path.read_text())
    payload["queries"][0]["anchor_world_xyz"] = [0.0, 0.0, 0.0]
    _write_json(path, payload)

    with pytest.raises(LudvigPFPRPhaseAError, match="prohibited key"):
        audit_benchmark(
            config.benchmark_dir,
            SCENE,
            expected_method_sha256=sha256_file(path),
            expected_public_sha256=config.expected_public_manifest_sha256,
        )


@pytest.mark.parametrize("phase", ["uplift", "score", "evaluate", "all"])
def test_unimplemented_phases_are_explicitly_fail_closed(phase: str) -> None:
    with pytest.raises(LudvigPFPRPhaseAError, match="not implemented"):
        reject_unimplemented_phase(phase)


def test_dino_pca_phase_is_registered_as_implemented() -> None:
    reject_unimplemented_phase("dino-pca")


def test_existing_attempt_is_never_overwritten(tmp_path: Path) -> None:
    config, _coverage = _make_asset(tmp_path)
    config.output_dir.mkdir()
    sentinel = config.output_dir / "owned_by_user.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(LudvigPFPRPhaseAError, match="Refusing to overwrite"):
        run_phase_a(config)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_non_rigid_c2w_fails_before_staging(tmp_path: Path) -> None:
    config, coverage = _make_asset(tmp_path)
    pose = config.source_scene / "pose" / f"{coverage[0]:06d}.txt"
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    np.savetxt(pose, invalid)

    with pytest.raises(LudvigPFPRPhaseAError, match="not a rigid c2w"):
        run_phase_a(config)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("role", ["color", "depth", "pose", "intrinsics"])
def test_trusted_source_adapter_ledger_rejects_post_audit_mutation(
    tmp_path: Path, role: str
) -> None:
    config, coverage = _make_asset(tmp_path)
    stem = f"{coverage[0]:06d}"
    if role == "color":
        Image.new("RGB", (8, 6), color=(255, 0, 255)).save(
            config.source_scene / "color" / f"{stem}.jpg"
        )
    elif role == "depth":
        Image.fromarray(np.full((6, 8), 2000, dtype=np.uint16)).save(
            config.source_scene / "depth" / f"{stem}.png"
        )
    elif role == "pose":
        path = config.source_scene / "pose" / f"{stem}.txt"
        pose = np.loadtxt(path).reshape(4, 4)
        pose[0, 3] += 0.5
        np.savetxt(path, pose, fmt="%.12f")
    else:
        path = config.source_scene / "intrinsics_depth.txt"
        intrinsics = np.loadtxt(path).reshape(4, 4)
        intrinsics[0, 0] += 0.25
        np.savetxt(path, intrinsics, fmt="%.12f")

    with pytest.raises(
        LudvigPFPRPhaseAError, match="trusted (?:adapter )?ledger"
    ):
        run_phase_a(config)
    assert not config.output_dir.exists()


def test_selected_source_symlink_must_resolve_inside_scene(tmp_path: Path) -> None:
    config, coverage = _make_asset(tmp_path)
    external = tmp_path / "private_query_source.jpg"
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(external)
    selected = config.source_scene / "color" / f"{coverage[0]:06d}.jpg"
    selected.unlink()
    selected.symlink_to(external)

    with pytest.raises(LudvigPFPRPhaseAError, match="outside source_scene"):
        run_phase_a(config)
    assert not config.output_dir.exists()


def test_private_evaluator_manifest_is_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _coverage = _make_asset(tmp_path)
    evaluator = config.benchmark_dir / "manifest.evaluator.json"
    evaluator.write_text("private sentinel: invalid JSON", encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.name == "manifest.evaluator.json":
            raise AssertionError("private evaluator manifest was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest = run_phase_a(config)

    assert manifest["evaluator_private_manifest_opened"] is False
    assert manifest["source_adapter_ledger"]["provenance"][
        "evaluator_private_manifest_opened"
    ] is False


def test_w2c_pose_substitution_fails_independent_frozen_fixture(
    tmp_path: Path,
) -> None:
    config, coverage = _make_asset(tmp_path)
    path = config.source_scene / "pose" / f"{coverage[0]:06d}.txt"
    c2w = np.loadtxt(path).reshape(4, 4)
    np.savetxt(path, np.linalg.inv(c2w), fmt="%.12f")
    # Rebind the byte inventory to prove that the separate semantic fixture,
    # rather than only the pose-file hash, rejects a w2c-as-c2w substitution.
    config = _refresh_source_adapter_inventory(config, coverage)

    with pytest.raises(
        LudvigPFPRPhaseAError, match="Trusted c2w/OpenCV pose fixture failed"
    ):
        run_phase_a(config)
    assert not config.output_dir.exists()


def test_depth_intrinsics_projection_offsets_are_rejected(tmp_path: Path) -> None:
    config, _coverage = _make_asset(tmp_path)
    path = config.source_scene / "intrinsics_depth.txt"
    intrinsics = np.loadtxt(path).reshape(4, 4)
    intrinsics[0, 3] = 123.0
    intrinsics[1, 3] = -99.0
    np.savetxt(path, intrinsics, fmt="%.12f")

    with pytest.raises(LudvigPFPRPhaseAError, match="not PINHOLE-compatible"):
        run_phase_a(config)
    assert not config.output_dir.exists()
