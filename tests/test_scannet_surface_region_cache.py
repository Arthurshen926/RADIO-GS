from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import radio_gs.scripts.build_scannet_surface_region_cache as cache_builder
from radio_gs.scripts.build_scannet_surface_region_cache import (
    FIXED_CORE_TEACHER_SEMANTICS,
    TEACHER_REPLAY_AUTHORITY_ARTIFACT_TYPE,
    TEACHER_VIEW_SELECTION_COVERAGE_DIVERSITY,
    TEACHER_VIEW_SELECTION_LEGACY,
    _excluded_spaces,
    _candidate_region_contract,
    _json_sha256,
    _lift_observation,
    _load_teacher_replay_cache,
    _physical_space,
    _project_region_box,
    _project_region_observation,
    _scene_names,
    _sha256,
    _surface_region_id,
    _teacher_medoid,
    _teacher_region_contract,
    _teacher_target_protocol,
    _teacher_view_statistics,
    _select_teacher_views_coverage_diversity,
    _materialize_region_student_row,
    _teacher_support_sha256,
    _teacher_target_sha256,
    _thermal_pause,
    _voxel_fuse,
    _v3_teacher_contract_from_replay,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json
from radio_gs.interfaces.surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
    SurfaceRegionContractV4,
    SurfaceRegionExpansionV3,
)
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


def test_voxel_fusion_is_bit_exact_across_caller_thread_counts() -> None:
    generator = torch.Generator().manual_seed(29)
    xyz = torch.randn(2048, 3, generator=generator) * 0.1
    features = torch.randn(2048, 64, generator=generator)
    footprint = torch.rand(2048, generator=generator)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        single = _voxel_fuse(xyz, features, footprint, 0.04)
        assert torch.get_num_threads() == 1
        caller_threads = min(4, max(2, original_threads))
        torch.set_num_threads(caller_threads)
        parallel_caller = _voxel_fuse(
            xyz, features, footprint, 0.04
        )
        assert torch.get_num_threads() == caller_threads
    finally:
        torch.set_num_threads(original_threads)
    for expected, actual in zip(single, parallel_caller):
        assert torch.equal(expected, actual)


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


def test_projected_teacher_observation_retains_support_visibility() -> None:
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = intrinsic[1, 1] = 100.0
    intrinsic[0, 2] = intrinsic[1, 2] = 50.0
    depth = torch.ones(100, 100)
    depth[50, 60] = 2.0
    observation = _project_region_observation(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ]
        ),
        depth,
        intrinsic,
        intrinsic,
        torch.eye(4),
        (100, 100),
        min_visible=1,
        context_pad=0.0,
    )

    assert observation is not None
    assert observation["projected_support_mask"].tolist() == [True, True, False]
    assert observation["visible_support_mask"].tolist() == [True, False, False]
    assert int(observation["crop_projected_support_mask"].sum()) == 1
    assert observation["coverage"] == pytest.approx(1 / 3)
    assert observation["visibility_purity"] == pytest.approx(1 / 2)
    assert _project_region_box(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ]
        ),
        depth,
        intrinsic,
        intrinsic,
        torch.eye(4),
        (100, 100),
        min_visible=1,
        context_pad=0.0,
    ) == observation["crop_box_tlbr"]


def _view_candidate(
    frame: str,
    visible: list[bool],
    direction: list[float],
) -> dict:
    mask = torch.tensor(visible, dtype=torch.bool)
    return {
        "frame": frame,
        "crop_box_tlbr": [0, 0, 24, 24],
        "projected_support_mask": torch.ones_like(mask),
        "visible_support_mask": mask,
        "crop_projected_support_mask": mask.reshape(1, -1),
        "coverage": float(mask.float().mean()),
        "visibility_purity": float(mask.float().mean()),
        "view_direction": torch.tensor(direction),
    }


def test_teacher_view_selection_prefers_coverage_then_diversity() -> None:
    candidates = [
        _view_candidate("b.jpg", [True, True, True, False], [0.99, 0.1, 0]),
        _view_candidate("c.jpg", [False, False, True, True], [-1, 0, 0]),
        _view_candidate("a.jpg", [True, True, True, False], [1, 0, 0]),
    ]

    selected = _select_teacher_views_coverage_diversity(candidates, 2)

    assert [value["frame"] for value in selected] == ["a.jpg", "c.jpg"]
    assert [
        value["frame"]
        for value in _select_teacher_views_coverage_diversity(candidates, 2)
    ] == ["a.jpg", "c.jpg"]


def test_teacher_view_statistics_are_sufficient_and_query_free() -> None:
    selected = [
        _view_candidate("a.jpg", [True, True, False, False], [1, 0, 0]),
        _view_candidate("c.jpg", [False, True, True, False], [-1, 0, 0]),
    ]
    statistics = _teacher_view_statistics(
        selected,
        summary_tokens=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        crop_descriptors=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )

    assert statistics["support_tokens"] == 4
    assert statistics["union_visible_support_fraction"] == pytest.approx(0.75)
    assert statistics["view_angular_dispersion_mean_pi"] == pytest.approx(1.0)
    assert statistics["official_summary_token_cosine_dispersion"] == pytest.approx(1.0)
    assert statistics["official_crop_descriptor_cosine_dispersion"] == pytest.approx(0.0)
    assert len(statistics["views"][0]["projected_support_mask"]) == 2
    assert len(statistics["views"][0]["visible_support_mask"]) == 2
    assert statistics["views"][0][
        "crop_projected_support_mask_shape_hw"
    ] == [1, 4]
    assert len(statistics["views"][0]["crop_projected_support_mask"]) == 2


def test_teacher_view_selection_is_an_explicit_target_schema() -> None:
    contract = _teacher_region_contract(SurfaceRegionContractV2(), 4096)
    legacy = _teacher_target_protocol(
        Namespace(
            frames_per_scene=8,
            min_visible_tokens=12,
            teacher_views=3,
            radio_resolution=384,
        ),
        contract,
        None,
        radio_version="test-radio",
        radio_checkpoint_sha256="a" * 64,
    )
    candidate = _teacher_target_protocol(
        Namespace(
            frames_per_scene=8,
            min_visible_tokens=12,
            teacher_views=3,
            radio_resolution=384,
            teacher_view_selection=(
                TEACHER_VIEW_SELECTION_COVERAGE_DIVERSITY
            ),
        ),
        contract,
        None,
        radio_version="test-radio",
        radio_checkpoint_sha256="a" * 64,
    )

    assert legacy["schema_version"] == 1
    assert legacy["frame_selection"] == TEACHER_VIEW_SELECTION_LEGACY
    assert "view_statistics_schema_version" not in legacy
    assert candidate["schema_version"] == 2
    assert candidate["frame_selection"] == (
        TEACHER_VIEW_SELECTION_COVERAGE_DIVERSITY
    )
    assert candidate["view_statistics_query_free"] is True


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


def _region_contract_args(**overrides: object) -> Namespace:
    values = {
        "region_contract_version": "v2",
        "region_radii": "0.25,0.45,0.70",
        "context_ratio": 1.2,
        "graph_neighbors": 16,
        "max_tokens": 256,
        "min_tokens": 24,
        "path_cost_mode": "euclidean",
        "path_affinity_floor": 1e-4,
        "token_subsampling": "core_context_radial_stratified_v1",
        "token_candidate_limit": 1024,
        "core_token_fraction": 0.6,
        "region_reliability_mode": "uniform_valid",
    }
    values.update(overrides)
    return Namespace(**values)


def test_candidate_contract_mode_is_explicit_and_v2_defaults_are_preserved() -> None:
    contract = _candidate_region_contract(_region_contract_args())
    expected = SurfaceRegionContractV2(
        radii_m=(0.25, 0.45, 0.70),
        context_ratio=1.2,
        neighbors=16,
        maximum_tokens=256,
        minimum_tokens=24,
        path_cost_mode="euclidean",
        path_affinity_floor=1e-4,
        token_subsampling="core_context_radial_stratified_v1",
        token_candidate_limit=1024,
        core_token_fraction=0.6,
        reliability_semantics="uniform_valid",
    )
    assert type(contract) is SurfaceRegionContractV2
    assert contract.to_dict() == expected.to_dict()
    assert contract.digest == expected.digest

    v3 = _candidate_region_contract(
        _region_contract_args(
            region_contract_version="v3",
            token_subsampling="nearest_geodesic_then_node_index",
        )
    )
    assert type(v3) is SurfaceRegionContractV3
    assert v3.path_cost_mode == "euclidean"
    v4 = _candidate_region_contract(
        _region_contract_args(
            region_contract_version="v4",
            token_subsampling=(
                "complete_core_then_typed_context_deterministic_backfill_v1"
            ),
        )
    )
    assert type(v4) is SurfaceRegionContractV4
    assert v4.token_candidate_limit == 1024
    assert v4.core_token_fraction == 0.6
    with pytest.raises(ValueError, match="nearest_geodesic"):
        _candidate_region_contract(
            _region_contract_args(region_contract_version="v3")
        )
    with pytest.raises(ValueError, match="complete_core_then_typed_context"):
        _candidate_region_contract(
            _region_contract_args(region_contract_version="v4")
        )
    with pytest.raises(ValueError, match="euclidean"):
        _candidate_region_contract(
            _region_contract_args(
                region_contract_version="v3",
                token_subsampling="nearest_geodesic_then_node_index",
                path_cost_mode="appearance_boundary_geometric",
            )
        )


def test_v3_rejects_v2_scene_intermediate_before_runtime_loading(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "scene0001_00").mkdir(parents=True)
    split = tmp_path / "split.txt"
    split.write_text("scene0001_00\n", encoding="utf-8")
    args = _region_contract_args(
        region_contract_version="v3",
        token_subsampling="nearest_geodesic_then_node_index",
    )
    args.__dict__.update(
        {
            "dataset_root": str(dataset),
            "split_file": str(split),
            "output": str(tmp_path / "output.pt"),
            "overwrite_output": False,
            "split_role": "train",
            "exclude_scene_files": "",
            "exclude_scene_names": "",
            "scene_names": "",
            "shard_index": 0,
            "shard_count": 1,
            "max_scenes": 1,
            "scene_intermediate_output_root": str(
                tmp_path / "intermediate"
            ),
            "scene_intermediate_manifest": "",
            "scene_intermediate_manifest_sha256": "",
        }
    )
    with pytest.raises(ValueError, match="forbids scene-intermediate"):
        cache_builder.build(args)


def test_v3_teacher_is_the_independent_frozen_v2_authority() -> None:
    candidate = SurfaceRegionContractV3(
        maximum_tokens=256,
        minimum_tokens=24,
        token_candidate_limit=1024,
        reliability_semantics="uniform_valid",
    )
    teacher = _teacher_region_contract(candidate, 4096)
    assert type(teacher) is SurfaceRegionContractV2
    assert teacher.feature_normalization == "l2_direction"
    assert teacher.path_cost_mode == "appearance_boundary_geometric"
    assert teacher.digest == (
        "a96e8721e6c9ed12c7da7528273b9af7d03da6bc709002e1e49639b9ee6b2f82"
    )


def test_v3_student_row_preserves_direction_amplitude_and_support_tiers() -> None:
    contract = SurfaceRegionContractV3(
        maximum_tokens=4,
        minimum_tokens=3,
        token_candidate_limit=4,
        reliability_semantics="uniform_valid",
    )
    raw_features = torch.zeros(4, 1280)
    raw_features[0, 0] = 2.0
    raw_features[1, 1] = 4.0
    raw_features[2, 2] = 8.0
    raw_features[3, 3] = 16.0
    raw_norm = torch.linalg.vector_norm(raw_features, dim=-1)
    directions = torch.nn.functional.normalize(raw_features, dim=-1)
    expansion = SurfaceRegionExpansionV3(
        rows=torch.tensor([0, 1, 2]),
        core_mask=torch.tensor([True, False, False]),
        context_mask=torch.tensor([False, True, False]),
        support_fill_mask=torch.tensor([False, False, True]),
        semantic_geodesic_distance=torch.tensor([0.0, 0.2, float("inf")]),
        recovery_distance=torch.tensor([float("inf"), float("inf"), 0.5]),
        anchor_index=0,
    )
    row, selection, counts = _materialize_region_student_row(
        contract=contract,
        expansion=expansion,
        anchor_row=0,
        xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ),
        radio_features=directions,
        raw_radio_l2_norm=raw_norm,
        local_sigma=torch.full((4,), 0.1),
        primitive_reliability=torch.tensor([1.0, 0.8, 0.6, 0.4]),
        radius=1.0,
    )

    assert selection.selected_count == 3
    assert counts == {
        "tokens": 3,
        "core_tokens": 1,
        "context_tokens": 1,
        "semantic_tokens": 2,
        "support_fill_tokens": 1,
        "minimum_satisfied": True,
    }
    assert torch.equal(
        row["support_fill_mask"],
        torch.tensor([False, False, True, False]),
    )
    assert row["geometry"].shape == (4, 16)
    torch.testing.assert_close(
        row["geometry"][:3, 15].float(),
        raw_norm[:3].log(),
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        row["reliability"][2, 0].float(),
        0.6 * torch.exp(torch.tensor(-0.5)),
        atol=5e-4,
        rtol=5e-4,
    )
    assert bool(row["token_mask"][:3].all())
    assert not bool(row["token_mask"][3])
    assert not bool(row["radio_features"][3].count_nonzero())
    assert not bool(row["geometry"][3].count_nonzero())
    assert not bool(row["reliability"][3].count_nonzero())


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


def test_v3_teacher_replay_adopts_its_independent_v2_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "teacher.pt"
    payload = _write_teacher_replay_cache(path)
    contract = _v3_teacher_contract_from_replay(
        path,
        expected_candidate_limit=4096,
    )
    assert type(contract) is SurfaceRegionContractV2
    assert contract.digest == payload["metadata"][
        "teacher_region_contract_sha256"
    ]
    with pytest.raises(ValueError, match="fixed-core authority"):
        _v3_teacher_contract_from_replay(
            path,
            expected_candidate_limit=2048,
        )


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


def test_paired_schema4_teacher_replay_uses_only_exact_full_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paired.pt"
    payload = _write_teacher_replay_cache(path)
    metadata = payload["metadata"]
    full = {
        **metadata["region_records"][0],
        "row_role": "full_support",
        "paired_full_region_id": metadata["region_records"][0]["region_id"],
    }
    completion = {
        **full,
        "region_id": "completion-row-id",
        "row_role": "eligibility_completion",
        "paired_full_region_id": full["region_id"],
    }
    metadata["schema_version"] = 4
    metadata["region_records"] = [full, completion]
    metadata["scene_region_counts"] = {"scene0001_00": 2}
    metadata["scene_teacher_region_counts"] = {"scene0001_00": 1}
    metadata["eligibility_completion"] = {
        "schema_version": 1,
        "validation_checkpoint_selection": "full_support_rows_only",
    }
    for key in (
        "official_summary_tokens",
        "official_crop_summaries",
        "teacher_mask",
    ):
        payload[key] = payload[key].repeat(2, *([1] * (payload[key].ndim - 1)))
    torch.save(payload, path)

    loaded = _load_teacher_replay_cache(
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
    assert loaded is not None
    assert len(loaded[1]["scene0001_00"]) == 1
    assert loaded[1]["scene0001_00"][0][0] == 0

    payload["official_summary_tokens"][1, 0, 0] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="share exact teacher tensors"):
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


def test_paired_schema4_replay_filters_only_newly_excluded_source_scenes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paired_with_new_exclusion.pt"
    payload = _write_teacher_replay_cache(path)
    metadata = payload["metadata"]
    first_full = {
        **metadata["region_records"][0],
        "row_role": "full_support",
        "paired_full_region_id": metadata["region_records"][0]["region_id"],
    }
    first_completion = {
        **first_full,
        "region_id": "first-completion-row-id",
        "row_role": "eligibility_completion",
        "paired_full_region_id": first_full["region_id"],
    }
    excluded_support_sha256 = "b" * 64
    excluded_full_id = _surface_region_id(
        "scene0002_00",
        int(first_full["seed"]),
        float(first_full["physical_radius_m"]),
        metadata["teacher_region_contract_sha256"],
        excluded_support_sha256,
    )
    excluded_full = {
        **first_full,
        "region_id": excluded_full_id,
        "scene": "scene0002_00",
        "teacher_support_sha256": excluded_support_sha256,
        "paired_full_region_id": excluded_full_id,
    }
    excluded_completion = {
        **excluded_full,
        "region_id": "excluded-completion-row-id",
        "row_role": "eligibility_completion",
        "paired_full_region_id": excluded_full_id,
    }
    metadata["schema_version"] = 4
    metadata["region_records"] = [
        first_full,
        first_completion,
        excluded_full,
        excluded_completion,
    ]
    metadata["scene_names"] = ["scene0001_00", "scene0002_00"]
    metadata["scene_region_counts"] = {
        "scene0001_00": 2,
        "scene0002_00": 2,
    }
    metadata["scene_teacher_region_counts"] = {
        "scene0001_00": 1,
        "scene0002_00": 1,
    }
    metadata["eligibility_completion"] = {
        "schema_version": 1,
        "validation_checkpoint_selection": "full_support_rows_only",
    }
    for key in (
        "official_summary_tokens",
        "official_crop_summaries",
        "teacher_mask",
    ):
        payload[key] = payload[key].repeat(
            4, *([1] * (payload[key].ndim - 1))
        )
    torch.save(payload, path)

    loaded = _load_teacher_replay_cache(
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
        expected_excluded_physical_spaces={"scene0097", "scene0002"},
        expected_regions_per_scene=1,
        expected_teacher_views=3,
    )
    assert loaded is not None
    assert list(loaded[1]) == ["scene0001_00"]
    assert loaded[1]["scene0001_00"][0][0] == 0

    payload["official_summary_tokens"][3, 0, 0] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="share exact teacher tensors"):
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
            expected_excluded_physical_spaces={"scene0097", "scene0002"},
            expected_regions_per_scene=1,
            expected_teacher_views=3,
        )


def test_schema2_teacher_replay_requires_bound_view_statistics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "view_statistics.pt"
    payload = _write_teacher_replay_cache(path)
    metadata = payload["metadata"]
    record = metadata["region_records"][0]
    selected = [
        _view_candidate("0.jpg", [True, False], [1, 0, 0]),
        _view_candidate("1.jpg", [False, True], [-1, 0, 0]),
    ]
    for index, value in enumerate(selected):
        projected_crop = torch.zeros(24, 24, dtype=torch.bool)
        projected_crop[index, index] = True
        value["crop_projected_support_mask"] = projected_crop
    statistics = _teacher_view_statistics(
        selected,
        summary_tokens=payload["official_summary_tokens"][0, :2],
        crop_descriptors=payload["official_crop_summaries"][0, :2],
    )
    record["teacher_view_statistics"] = statistics
    protocol = {
        "schema_version": 2,
        "support_semantics": FIXED_CORE_TEACHER_SEMANTICS,
        "frame_selection": TEACHER_VIEW_SELECTION_COVERAGE_DIVERSITY,
    }
    metadata["teacher_target_schema_version"] = 2
    metadata["teacher_target_protocol"] = protocol
    metadata["teacher_target_protocol_sha256"] = _json_sha256(protocol)
    torch.save(payload, path)

    loaded = _load_teacher_replay_cache(
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
        expected_teacher_target_schema_version=2,
    )
    assert loaded is not None

    record["teacher_view_statistics"]["views"][0][
        "visible_support_mask"
    ] = "zz"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="view statistics"):
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
            expected_teacher_target_schema_version=2,
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
