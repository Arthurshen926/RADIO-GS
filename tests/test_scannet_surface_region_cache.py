from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import radio_gs.scripts.build_scannet_surface_region_cache as cache_builder
from radio_gs.scripts.build_scannet_surface_region_cache import (
    FIXED_CORE_TEACHER_SEMANTICS,
    TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE,
    _excluded_spaces,
    _json_sha256,
    _lift_observation,
    _load_teacher_replay_cache,
    _physical_space,
    _project_region_box,
    _scene_names,
    _sha256,
    _surface_region_id,
    _teacher_medoid,
    _teacher_region_contract,
    _teacher_support_sha256,
    _teacher_target_sha256,
    _thermal_pause,
    _voxel_fuse,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_scene_intermediate import (
    EXPECTED_IMPLEMENTATION_ROLES,
    SourceFileBinding,
    SurfaceSceneFrameBinding,
    SurfaceSceneIntermediate,
    SurfaceSceneIntermediateContract,
    assert_exact_surface_scene_replay,
    default_graph_config_dict,
)
from radio_gs.querying.support_solver import PrimitiveSupportGraph


def test_voxel_fusion_is_deterministic() -> None:
    xyz = torch.tensor([[0.00, 0, 0], [0.01, 0, 0], [0.05, 0, 0], [0.10, 0, 0]])
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    fused_xyz, fused_features, footprint, count = _voxel_fuse(
        xyz, features, torch.full((4,), 0.02), 0.04
    )
    assert len(fused_xyz) == 3
    assert count.max() == 2
    assert torch.isfinite(fused_features).all() and torch.isfinite(footprint).all()


def test_thermal_pause_synchronizes_cuda_before_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        cache_builder.torch.cuda,
        "synchronize",
        lambda device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        cache_builder.time,
        "sleep",
        lambda seconds: calls.append(("sleep", seconds)),
    )

    device = torch.device("cuda:0")
    _thermal_pause(device, 1.5, image_count=3)

    assert calls == [("synchronize", device), ("sleep", 4.5)]
    calls.clear()
    _thermal_pause(device, 0.0, image_count=3)
    assert calls == []
    with pytest.raises(ValueError, match="finite and non-negative"):
        _thermal_pause(device, -1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _thermal_pause(device, float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        _thermal_pause(device, 1.0, image_count=0)


def test_teacher_medoid_selects_consensus_view() -> None:
    tokens = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    assert _teacher_medoid(tokens) in {0, 1}


def test_lifting_samples_pixel_centres_with_align_corners_false() -> None:
    depth = torch.ones(2, 3)
    intrinsic = torch.eye(4)
    spatial = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    _xyz, sampled, _footprint = _lift_observation(
        depth, intrinsic, intrinsic, torch.eye(4), spatial,
        stride=1, color_size=(3, 2),
    )
    torch.testing.assert_close(sampled[:, 0], spatial.flatten(), atol=1e-6, rtol=0)


def test_singleton_surface_region_gets_a_valid_teacher_crop() -> None:
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = intrinsic[1, 1] = 100.0
    intrinsic[0, 2] = intrinsic[1, 2] = 50.0
    box = _project_region_box(
        torch.tensor([[0.0, 0.0, 1.0]]), torch.ones(100, 100),
        intrinsic, intrinsic, torch.eye(4), (100, 100), min_visible=1,
        context_pad=0.0,
    )
    assert box is not None
    top, left, bottom, right = box
    assert bottom - top == 24 and right - left == 24


def test_teacher_region_is_fixed_core_only_and_input_sampling_independent() -> None:
    input_contract = SurfaceRegionContractV2(
        token_subsampling="core_context_radial_stratified_v1",
        token_candidate_limit=1024,
        reliability_semantics="uniform_valid",
    )

    teacher = _teacher_region_contract(input_contract, 4096)

    assert teacher.context_ratio == 1.0
    assert teacher.maximum_tokens == 4096
    assert teacher.token_candidate_limit == 4096
    assert teacher.token_subsampling == "nearest_geodesic_then_node_index"
    assert teacher.reliability_semantics == "uniform_valid"
    assert teacher.minimum_tokens == 1


def test_teacher_contract_ignores_student_reliability_ablation() -> None:
    geometric = SurfaceRegionContractV2(
        token_candidate_limit=1024,
        reliability_semantics="geometric_mean_observation_agreement",
    )
    uniform = SurfaceRegionContractV2(
        token_candidate_limit=1024,
        reliability_semantics="uniform_valid",
    )

    assert (
        _teacher_region_contract(geometric, 4096).digest
        == _teacher_region_contract(uniform, 4096).digest
    )


def test_surface_region_id_binds_ordered_physical_teacher_support() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    support_a = _teacher_support_sha256(xyz, torch.tensor([0, 1]))
    support_b = _teacher_support_sha256(xyz, torch.tensor([0, 2]))
    region_a = _surface_region_id(
        "scene0001_00", 0, 0.25, "contract", support_a
    )
    region_b = _surface_region_id(
        "scene0001_00", 0, 0.25, "contract", support_b
    )

    assert support_a != support_b
    assert region_a != region_b
    assert region_a == _surface_region_id(
        "scene0001_00", 0, 0.25, "contract", support_a
    )


def _write_teacher_replay_cache(path: Path) -> dict:
    views = [
        {"frame": "0.jpg", "crop_box_tlbr": (0, 0, 24, 24)},
        {"frame": "1.jpg", "crop_box_tlbr": (1, 1, 25, 25)},
    ]
    summary_tokens = torch.zeros(1, 3, 1280, dtype=torch.float16)
    crop_summaries = torch.zeros(1, 3, 32, dtype=torch.float16)
    teacher_mask = torch.tensor([[True, True, False]])
    support_sha256 = "a" * 64
    teacher_contract = _teacher_region_contract(
        SurfaceRegionContractV2(), 4096
    )
    protocol = {
        "schema_version": 1,
        "support_semantics": FIXED_CORE_TEACHER_SEMANTICS,
    }
    target_sha256 = _teacher_target_sha256(
        summary_tokens[0],
        crop_summaries[0],
        teacher_mask[0],
    )
    record = {
        "region_id": _surface_region_id(
            "scene0001_00",
            7,
            0.25,
            teacher_contract.digest,
            support_sha256,
        ),
        "scene": "scene0001_00",
        "seed": 7,
        "physical_radius_m": 0.25,
        "teacher_support_sha256": support_sha256,
        "teacher_region_tokens": 2,
        "teacher_region_saturated": False,
        "teacher_target_source": "fresh_official_runtime",
        "teacher_target_sha256": target_sha256,
        "teacher_views": views,
        "teacher_medoid": 0,
    }
    metadata = {
        "schema_version": 3,
        "split_role": "train",
        "split_file_sha256": "split",
        "dataset_root": "/dataset",
        "teacher_region_contract": teacher_contract.to_dict(),
        "teacher_region_contract_sha256": teacher_contract.digest,
        "teacher_region_semantics": FIXED_CORE_TEACHER_SEMANTICS,
        "teacher_target_schema_version": 1,
        "teacher_crop_protocol": (
            "core_support_defined_unmasked_bbox_min24_context_pad0_v1"
        ),
        "teacher_target_protocol": protocol,
        "teacher_target_protocol_sha256": _json_sha256(protocol),
        "teacher_target_source": "fresh_official_runtime",
        "teacher_regions_saturated": 0,
        "regions_per_scene_requested": 1,
        "teacher_views_requested": 3,
        "builder_script_sha256": _sha256(
            Path(cache_builder.__file__).resolve()
        ),
        "radio_checkpoint_sha256": "radio",
        "scene_names": ["scene0001_00"],
        "scene_region_counts": {"scene0001_00": 1},
        "failed_scenes": {},
        "excluded_physical_spaces": ["scene0097"],
        "physical_space_disjoint": True,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "annotations_opened": False,
        "labels_opened": False,
        "instances_opened": False,
        "masks_opened": False,
        "text_opened": False,
        "region_records": [record],
    }
    payload = {
        "official_summary_tokens": summary_tokens,
        "official_crop_summaries": crop_summaries,
        "teacher_mask": teacher_mask,
        "metadata": metadata,
    }
    torch.save(payload, path)
    return payload


def test_teacher_replay_cache_round_trips_and_rejects_target_tampering(
    tmp_path,
) -> None:
    path = tmp_path / "control.pt"
    payload = _write_teacher_replay_cache(path)
    metadata = payload["metadata"]

    loaded = _load_teacher_replay_cache(
        str(path),
        expected_split_role="train",
        expected_split_file_sha256="split",
        expected_dataset_root="/dataset",
        expected_teacher_contract_sha256=(
            metadata["teacher_region_contract_sha256"]
        ),
        expected_teacher_target_protocol_sha256=(
            metadata["teacher_target_protocol_sha256"]
        ),
        expected_radio_checkpoint_sha256="radio",
        expected_excluded_physical_spaces={"scene0097"},
        expected_regions_per_scene=1,
        expected_teacher_views=3,
    )

    assert loaded is not None
    assert list(loaded[1]) == ["scene0001_00"]
    assert loaded[2]["sha256"] == _sha256(path)

    payload["official_summary_tokens"][0, 0, 0] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="target digest"):
        _load_teacher_replay_cache(
            str(path),
            expected_split_role="train",
            expected_split_file_sha256="split",
            expected_dataset_root="/dataset",
            expected_teacher_contract_sha256=(
                metadata["teacher_region_contract_sha256"]
            ),
            expected_teacher_target_protocol_sha256=(
                metadata["teacher_target_protocol_sha256"]
            ),
            expected_radio_checkpoint_sha256="radio",
            expected_excluded_physical_spaces={"scene0097"},
            expected_regions_per_scene=1,
            expected_teacher_views=3,
        )


def test_historical_teacher_replay_requires_exact_external_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical_control.pt"
    payload = _write_teacher_replay_cache(path)
    metadata = payload["metadata"]
    historical_builder = "b" * 64
    metadata["builder_script_sha256"] = historical_builder
    torch.save(payload, path)
    manifest = tmp_path / "run_manifest.json"
    write_frozen_json(manifest, {"screen": "unit-test"})
    authority = tmp_path / "replay_authority.json"
    authority_payload = {
        "artifact_type": TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE,
        "schema_version": 1,
        "authorization_scope": (
            "exact_historical_cache_fixed_teacher_replay_only"
        ),
        "run_manifest": file_record(manifest),
        "cache": file_record(path),
        "split_role": "train",
        "split_file_sha256": "split",
        "scene_names": ["scene0001_00"],
        "teacher_region_contract_sha256": metadata[
            "teacher_region_contract_sha256"
        ],
        "teacher_target_protocol_sha256": metadata[
            "teacher_target_protocol_sha256"
        ],
        "radio_checkpoint_sha256": "radio",
        "source_builder_script_sha256": historical_builder,
    }
    write_frozen_json(authority, authority_payload)

    with pytest.raises(ValueError, match="external replay authority"):
        _load_teacher_replay_cache(
            str(path),
            expected_split_role="train",
            expected_split_file_sha256="split",
            expected_dataset_root="/dataset",
            expected_teacher_contract_sha256=metadata[
                "teacher_region_contract_sha256"
            ],
            expected_teacher_target_protocol_sha256=metadata[
                "teacher_target_protocol_sha256"
            ],
            expected_radio_checkpoint_sha256="radio",
            expected_excluded_physical_spaces={"scene0097"},
            expected_regions_per_scene=1,
            expected_teacher_views=3,
        )

    loaded = _load_teacher_replay_cache(
        str(path),
        authority_path=str(authority),
        authority_sha256=_sha256(authority),
        expected_split_role="train",
        expected_split_file_sha256="split",
        expected_dataset_root="/dataset",
        expected_teacher_contract_sha256=metadata[
            "teacher_region_contract_sha256"
        ],
        expected_teacher_target_protocol_sha256=metadata[
            "teacher_target_protocol_sha256"
        ],
        expected_radio_checkpoint_sha256="radio",
        expected_excluded_physical_spaces={"scene0097"},
        expected_regions_per_scene=1,
        expected_teacher_views=3,
    )
    assert loaded is not None
    assert loaded[3] == file_record(authority)


def test_exclusion_is_applied_to_every_rescan_of_a_physical_space(tmp_path) -> None:
    root = tmp_path / "frames"
    for scene in ("scene0012_00", "scene0012_02", "scene0013_00"):
        (root / scene).mkdir(parents=True)
    split = tmp_path / "split.txt"
    split.write_text("scene0012_00\nscene0012_02\nscene0013_00\n")
    exclusion = tmp_path / "pfir_dev.txt"
    exclusion.write_text("# comment\nscene0012_02\n")

    spaces, records = _excluded_spaces(str(exclusion), "")
    assert spaces == {"scene0012"}
    assert records[0]["sha256"]
    assert _physical_space("scene0012_02") == "scene0012"
    assert _scene_names(
        split,
        root,
        excluded_physical_spaces=spaces,
    ) == ["scene0013_00"]


def _fastpath_source(path: Path, text: str) -> SourceFileBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return SourceFileBinding.from_path(path)


def _fastpath_contract(tmp_path: Path) -> SurfaceSceneIntermediateContract:
    frames = []
    for index in range(2):
        frames.append(
            SurfaceSceneFrameBinding(
                frame_id=f"{index:06d}",
                color=_fastpath_source(
                    tmp_path / "inputs" / f"{index:06d}.jpg",
                    f"color-{index}",
                ),
                depth=_fastpath_source(
                    tmp_path / "inputs" / f"{index:06d}.png",
                    f"depth-{index}",
                ),
                pose=_fastpath_source(
                    tmp_path / "inputs" / f"{index:06d}.txt",
                    f"pose-{index}",
                ),
            )
        )
    implementations = {
        role: _fastpath_source(
            tmp_path / "implementation" / f"{role}.py",
            f"implementation-{role}",
        )
        for role in EXPECTED_IMPLEMENTATION_ROLES
    }
    return SurfaceSceneIntermediateContract(
        scene="scene0024_00",
        source_frames=tuple(frames),
        depth_intrinsic=_fastpath_source(
            tmp_path / "inputs" / "intrinsics_depth.txt",
            "depth-intrinsic",
        ),
        color_intrinsic=_fastpath_source(
            tmp_path / "inputs" / "intrinsics_color.txt",
            "color-intrinsic",
        ),
        radio_checkpoint=_fastpath_source(
            tmp_path / "checkpoint" / "radio.pt",
            "checkpoint",
        ),
        radio_version="c-radio_v4-h",
        radio_resolution=384,
        depth_stride=8,
        voxel_size=0.04,
        adaptor_names={
            "appearance": "dino_v3_7b",
            "boundary": "sam3",
        },
        adaptor_batch_size=64,
        affinity_dimension=256,
        graph_config=default_graph_config_dict(),
        implementation_sources=implementations,
    )


def _fastpath_intermediate(tmp_path: Path) -> SurfaceSceneIntermediate:
    features = torch.zeros(3, 1280, dtype=torch.float32)
    features[0, 0] = 1
    features[1, 1] = 1
    features[2, 2] = 1
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.int64,
    )
    raw_affinity = torch.tensor(
        [0.8, 0.8, 0.6, 0.6],
        dtype=torch.float32,
    )
    row_sum = torch.zeros(3, dtype=torch.float32)
    row_sum.index_add_(0, edge_index[0], raw_affinity)
    graph = PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=(
            raw_affinity / row_sum[edge_index[0]].clamp_min(1e-12)
        ),
        raw_affinity=raw_affinity,
        local_sigma=torch.full((3,), 0.1, dtype=torch.float32),
        num_nodes=3,
        edge_channels={
            "geometry": torch.tensor(
                [0.9, 0.9, 0.7, 0.7],
                dtype=torch.float32,
            ),
            "appearance": torch.tensor(
                [0.8, 0.8, 0.6, 0.6],
                dtype=torch.float32,
            ),
            "boundary": torch.ones(4, dtype=torch.float32),
        },
    )
    return SurfaceSceneIntermediate(
        contract=_fastpath_contract(tmp_path),
        xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        radio_features=features,
        geometric_reliability=torch.tensor(
            [0.4, 0.7, 0.9],
            dtype=torch.float32,
        ),
        graph=graph,
    )


def _publish_fastpath_fixture(tmp_path: Path):
    value = _fastpath_intermediate(tmp_path)
    root = cache_builder._resolved_intermediate_root(
        tmp_path / "scene-cache",
        create=True,
    )
    published, scene_record = cache_builder._publish_scene_intermediate(
        root,
        value,
    )
    run_contract = {
        "dataset_root": "/dataset",
        "split_role": "train",
        "selection": {"shard_count": 4, "shard_index": 0},
    }
    manifest = cache_builder._publish_scene_intermediate_manifest(
        root,
        scenes=[value.contract.scene],
        contracts={value.contract.scene: value.contract},
        run_contract=run_contract,
    )
    return value, root, published, scene_record, manifest, run_contract


def test_scene_intermediate_fastpath_atomic_manifest_round_trip_and_recovery(
    tmp_path,
) -> None:
    value, root, published, scene_record, manifest, run_contract = (
        _publish_fastpath_fixture(tmp_path)
    )
    assert_exact_surface_scene_replay(value, published)
    assert scene_record["contract_sha256"] == value.contract.digest
    assert not (root / f".{value.contract.scene}.pending").exists()

    replay_root, by_scene, replay_provenance = (
        cache_builder._load_scene_intermediate_manifest(
            manifest["manifest"]["path"],
            expected_sha256=manifest["manifest"]["sha256"],
            expected_scenes=[value.contract.scene],
            expected_run_contract=run_contract,
        )
    )
    replay, replay_record = cache_builder._load_published_scene_intermediate(
        replay_root / value.contract.scene,
        root=replay_root,
        expected_contract=value.contract,
        expected_authority_sha256=by_scene[value.contract.scene][
            "authority_sha256"
        ],
        expected_manifest_record=by_scene[value.contract.scene],
    )
    assert_exact_surface_scene_replay(value, replay)
    assert replay_record == by_scene[value.contract.scene]
    assert replay_provenance["mode"] == "exact_replay"

    (root / cache_builder.SCENE_INTERMEDIATE_MANIFEST_NAME).unlink()
    final_directory = root / value.contract.scene
    pending = root / f".{value.contract.scene}.pending"
    final_directory.rename(pending)
    recovered = cache_builder._recover_scene_intermediate_if_available(
        root,
        scene_name=value.contract.scene,
        expected_contract=value.contract,
    )
    assert recovered is not None
    assert_exact_surface_scene_replay(value, recovered[0])
    assert final_directory.is_dir() and not pending.exists()


def test_scene_intermediate_fastpath_fails_closed_on_contract_and_source_change(
    tmp_path,
) -> None:
    value, root, _published, _scene_record, _manifest, _run_contract = (
        _publish_fastpath_fixture(tmp_path)
    )
    directory = root / value.contract.scene
    changed_graph = dict(value.contract.graph_config)
    changed_graph["neighbors"] = int(changed_graph["neighbors"]) + 1
    wrong_contract = replace(value.contract, graph_config=changed_graph)
    with pytest.raises(ValueError, match="external contract differs"):
        cache_builder._load_published_scene_intermediate(
            directory,
            root=root,
            expected_contract=wrong_contract,
        )

    changed_source = Path(value.contract.source_frames[0].color.path)
    changed_source.write_text("changed-color", encoding="utf-8")
    with pytest.raises(ValueError, match="bound source file changed"):
        cache_builder._load_published_scene_intermediate(
            directory,
            root=root,
            expected_contract=value.contract,
        )


def test_scene_intermediate_fastpath_rejects_symlink_and_wrong_manifest_sha(
    tmp_path,
) -> None:
    value, root, _published, _scene_record, manifest, run_contract = (
        _publish_fastpath_fixture(tmp_path)
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        cache_builder._load_scene_intermediate_manifest(
            manifest["manifest"]["path"],
            expected_sha256="0" * 64,
            expected_scenes=[value.contract.scene],
            expected_run_contract=run_contract,
        )

    wrong_shard = {
        **run_contract,
        "selection": {"shard_count": 4, "shard_index": 1},
    }
    wrong_role = {**run_contract, "split_role": "validation"}
    for wrong_run_contract in (wrong_shard, wrong_role):
        with pytest.raises(ValueError, match="manifest contract differs"):
            cache_builder._load_scene_intermediate_manifest(
                manifest["manifest"]["path"],
                expected_sha256=manifest["manifest"]["sha256"],
                expected_scenes=[value.contract.scene],
                expected_run_contract=wrong_run_contract,
            )

    directory = root / value.contract.scene
    real_directory = root / f"{value.contract.scene}.real"
    directory.rename(real_directory)
    directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        cache_builder._load_scene_intermediate_manifest(
            manifest["manifest"]["path"],
            expected_sha256=manifest["manifest"]["sha256"],
            expected_scenes=[value.contract.scene],
            expected_run_contract=run_contract,
        )


def test_candidate_uniform_reliability_uses_geometric_scene_domain() -> None:
    geometric = torch.tensor([0.2, 0.7, 1.0], dtype=torch.float32)
    uniform = cache_builder._candidate_reliability_from_geometric(
        geometric,
        mode="uniform_valid",
        num_rows=3,
    )
    matched = cache_builder._candidate_reliability_from_geometric(
        geometric,
        mode="geometric_mean_observation_agreement",
        num_rows=3,
    )

    assert uniform.dtype == geometric.dtype
    assert uniform.shape == geometric.shape
    assert torch.equal(uniform, torch.ones_like(geometric))
    assert matched is geometric


def test_empty_scene_intermediate_options_preserve_legacy_resume_cli(
    tmp_path,
) -> None:
    common = {
        "dataset_root": str(tmp_path / "dataset"),
        "split_file": str(tmp_path / "split.txt"),
        "output": str(tmp_path / "cache.pt"),
        "radio_repo": str(tmp_path / "radio"),
        "radio_checkpoint": str(tmp_path / "radio.pt"),
        "scene_graph_output_root": "",
        "teacher_replay_cache": "",
        "seed": 7,
    }
    legacy = cache_builder._normalize_resume_cli(
        Namespace(**common),
        tmp_path / "resume",
    )
    extended = cache_builder._normalize_resume_cli(
        Namespace(
            **common,
            scene_intermediate_output_root="",
            scene_intermediate_manifest="",
            scene_intermediate_manifest_sha256="",
        ),
        tmp_path / "resume",
    )

    assert extended == legacy
    assert not any(key.startswith("scene_intermediate") for key in extended)


def test_scene_intermediate_run_contract_binds_role_shard_and_snapshot(
    tmp_path,
) -> None:
    split = tmp_path / "split.txt"
    split.write_text("scene0024_00\n", encoding="utf-8")
    args = Namespace(
        scene_names="",
        shard_count=4,
        shard_index=2,
        max_scenes=8,
    )
    run_contract = cache_builder._scene_intermediate_run_contract(
        args,
        dataset_root="/dataset",
        split_file=split,
        split_file_sha256=_sha256(split),
        split_role="train",
        scenes=["scene0024_00"],
        excluded_physical_spaces={"scene0097"},
        exclusion_files=[],
    )

    assert run_contract["builder"]["path"] == str(
        Path(cache_builder.__file__).resolve()
    )
    assert run_contract["builder"]["sha256"] == _sha256(
        Path(cache_builder.__file__).resolve()
    )
    assert run_contract["split_role"] == "train"
    assert run_contract["selection"] == {
        "mode": "deterministic_shard",
        "scene_names_argument": [],
        "shard_count": 4,
        "shard_index": 2,
        "max_scenes": 8,
        "selected_scenes": ["scene0024_00"],
    }
