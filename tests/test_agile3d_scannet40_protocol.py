import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from radio_gs.benchmarks.agile3d_scannet40.protocol import (
    Click,
    aggregate_official_metrics,
    evaluate_interactive_predictions,
    interaction_health_metrics,
    quantize_scannet_points,
    select_next_click,
)
from radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache import (
    CanonicalPointPredictor,
    _load_feature_cache,
)
import radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field as canonical_evaluator
from radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field import (
    CanonicalFieldPointPredictor,
    constrain_released_click_scores,
    _gaussian_covariances,
    _read_official_geometry,
    evaluate,
    observation_source_from_render_contract,
    select_object_shard,
    stable_support_record,
    validate_capability_teacher_fidelity,
    validate_continuous_support_threshold,
    validate_full_observation_mpr_contract,
    validate_observation_contract,
)
from radio_gs.benchmarks.agile3d_scannet40.merge_results import merge
from radio_gs.field import FeatureSpaceSignature
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    canonical_observation_contract,
    observation_contract_sha256,
)
from radio_gs.querying.query_spec import SelectionMode
from radio_gs.querying.support_solver import SupportGraphConfig, SupportSolverConfig
from radio_gs.querying.support_solver import build_primitive_support_graph
from radio_gs.scripts.export_canonical_field_to_agile3d_mesh import (
    interpolate_feature_matrix,
)


def _canonical_signature(name: str, dim: int) -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio-hash",
        raw_feature_dim=1280,
        adaptor_name=name,
        adaptor_sha256="radio-hash",
        adaptor_output_dim=dim,
        token_type="primitive",
        field_checkpoint_sha256="field-hash",
    )


def test_released_click_score_constraint_is_opt_in_and_local() -> None:
    scores = torch.tensor([0.2, 0.3, 0.8, 0.7])
    unchanged = constrain_released_click_scores(
        scores, positive_indices=[1], negative_indices=[2], mode="none"
    )
    constrained = constrain_released_click_scores(
        scores,
        positive_indices=[1],
        negative_indices=[2],
        mode="click_score_clamp",
    )
    assert torch.equal(unchanged, scores)
    assert torch.equal(constrained, torch.tensor([0.2, 1.0, 0.0, 0.7]))
    assert torch.equal(scores, torch.tensor([0.2, 0.3, 0.8, 0.7]))


def test_released_click_score_constraint_rejects_conflicting_signs() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        constrain_released_click_scores(
            torch.zeros(3),
            positive_indices=[1],
            negative_indices=[1],
            mode="click_score_clamp",
        )


def test_quantization_reconstructs_full_rows() -> None:
    xyz = np.array([[0.10, 0, 0], [0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    labels = np.array([0, 1, 1])
    scene = quantize_scannet_points(xyz, np.zeros((3, 3)), labels, voxel_size=0.05)
    assert len(scene.coordinates) == 2
    np.testing.assert_array_equal(scene.unique_map, [0, 1])
    np.testing.assert_array_equal(scene.labels[scene.inverse_map], labels)


def test_object_sharding_is_a_deterministic_execution_partition() -> None:
    objects = [SimpleNamespace(object_id=index) for index in range(7)]
    shards = [
        select_object_shard(objects, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    assert [[item.object_id for item in shard] for shard in shards] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    assert sorted(item.object_id for shard in shards for item in shard) == list(range(7))
    with pytest.raises(ValueError, match="shard"):
        select_object_shard(objects, shard_index=3, shard_count=3)


def test_label_free_geometry_reader_never_invokes_the_label_ply_parser(
    tmp_path, monkeypatch
) -> None:
    """Support preflight must not materialize the PLY ``label`` property.

    AGILE packs geometry and evaluator labels into one binary PLY.  The
    full-observation gate is allowed to read coordinates/RGB but must not use
    the generic PLY loader, because that loader eagerly creates the label
    column before the field has passed its label-free quality audit.
    """

    vertices = np.empty(
        2,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("R", "u1"),
            ("G", "u1"),
            ("B", "u1"),
            ("label", "f8"),
        ],
    )
    vertices["x"] = [1.0, 4.0]
    vertices["y"] = [2.0, 5.0]
    vertices["z"] = [3.0, 6.0]
    vertices["R"] = [10, 40]
    vertices["G"] = [20, 50]
    vertices["B"] = [30, 60]
    vertices["label"] = [123.0, 456.0]
    path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))

    def _forbidden_parser(*_args, **_kwargs):
        raise AssertionError("label-free geometry reader used PlyData.read")

    monkeypatch.setattr(PlyData, "read", _forbidden_parser)
    xyz, rgb = _read_official_geometry(path)

    np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(
        rgb,
        np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.float32) / 255.0,
    )


def test_support_only_preflight_never_opens_objects_or_ply_labels(tmp_path, monkeypatch) -> None:
    """A failed/full support audit remains label-free by construction."""

    benchmark = tmp_path / "benchmark"
    (benchmark / "scans").mkdir(parents=True)
    (benchmark / "scans" / "scene_a.ply").touch()
    fields = tmp_path / "fields"
    (fields / "canonical_fields" / "scene_a").mkdir(parents=True)
    xyz = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    class _Predictor:
        def protocol_report(self):
            return {
                "continuous_support_fraction": 0.97,
                "continuous_support_quantiles": {"p50": 1.0},
                "labels_opened": False,
            }

    monkeypatch.setattr(
        canonical_evaluator,
        "_read_official_geometry",
        lambda _path: (xyz, np.zeros((1, 3), dtype=np.float32)),
    )
    monkeypatch.setattr(
        canonical_evaluator,
        "quantize_scannet_points",
        lambda *_args, **_kwargs: SimpleNamespace(
            raw_coordinates=np.zeros((1, 3), dtype=np.float32),
            unique_map=np.asarray([0], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        canonical_evaluator,
        "_load_scene_predictor",
        lambda *_args, **_kwargs: (
            _Predictor(),
            {
                "source_observation_root": str(tmp_path / "rgbd"),
                "declared_source_contract": "field_only_dense_rgbd_v1",
                "field_source_contract_sha256": "",
                "field_source_contract_version": "",
            },
        ),
    )
    monkeypatch.setattr(
        canonical_evaluator,
        "_read_official_labels",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("support-only read PLY labels")
        ),
    )
    monkeypatch.setattr(
        canonical_evaluator,
        "load_official_object_list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("support-only opened AGILE objects")
        ),
    )
    output = tmp_path / "support.json"
    report = evaluate(
        argparse.Namespace(
            benchmark_root=str(benchmark),
            field_root=str(fields),
            geometry_cache_root=str(tmp_path / "geometry"),
            output=str(output),
            field_checkpoint_name="canonical_mpr_v2.pt",
            capability_cache_name="official_dino_sam3_views.pt",
            support_graph_name="shared_support_graph_k16.pt",
            reliability_cache_name="",
            scene_names="scene_a",
            device="cpu",
            voxel_size=0.05,
            evaluation_voxel_size_m=0.05,
            click_seed_kernel="native_gaussian",
            max_clicks=20,
            click_workers=1,
            observation_contract="dense_overlap_pilot",
            require_support_gate=True,
            support_only=True,
            minimum_support_fraction=0.95,
            readout_candidate_k=64,
            readout_support_threshold=1e-6,
            seed_candidate_k=64,
            seed_topk=0,
            seed_temperature=1.0,
            world_point_prototype_mode="per_click_local",
            world_point_max_prototypes=0,
            world_point_prototype_weighting="support_mass",
            solver_type="confidence_random_walker",
            laplacian_weight=1.0,
            cg_iterations=64,
            support_threshold=0.5,
            hard_seed_threshold=0.2,
            hard_seed_conflict_policy="positive_priority",
            hard_seed_conflict_margin=0.0,
            unary_edge_contrast=0.0,
            feature_calibration="none",
            background_centroids=0,
            background_negative_policy="pooled_mean",
            calibration_sample_size=8192,
            centroid_iterations=4,
            score_calibration="none",
            require_official_extracted_capability_teachers=False,
        )
    )
    assert report["mode"] == "label_free_field_support_preflight"
    assert report["support_gate_passed"] is True
    assert report["protocol"]["labels_opened"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["mode"] == report["mode"]


def test_quantized_callback_coordinates_restore_the_public_world_origin() -> None:
    xyz = np.array(
        [[2.20, -0.05, 0.06], [2.24, -0.05, 0.06], [2.31, 0.00, 0.10]],
        dtype=np.float32,
    )
    scene = quantize_scannet_points(
        xyz, np.zeros((len(xyz), 3), dtype=np.float32), np.zeros(len(xyz), dtype=np.int32)
    )
    # The released sparse callback uses origin-shifted coordinates for voxel
    # processing.  A continuous field built from ScanNet RGB-D poses lives in
    # the original public world frame, so the evaluator must restore exactly
    # this label-free scene translation before invoking its world-query path.
    np.testing.assert_allclose(
        scene.raw_coordinates + xyz.min(axis=0, keepdims=True),
        xyz[scene.unique_map],
        atol=1e-6,
    )


def test_first_click_is_deepest_target_point() -> None:
    xyz = np.column_stack([np.arange(7), np.zeros((7, 2))]).astype(np.float32)
    target = np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool)
    click = select_next_click(xyz, np.zeros(7, dtype=bool), target, order=0)
    assert click == Click(point_index=3, is_positive=True, order=0)


def test_corrective_click_uses_larger_fp_or_fn_error_radius() -> None:
    xyz = np.column_stack([np.arange(10), np.zeros((10, 2))]).astype(np.float32)
    target = np.zeros(10, dtype=bool)
    target[2:7] = True
    prediction = np.zeros(10, dtype=bool)
    prediction[2] = True
    prediction[8] = True
    click = select_next_click(xyz, prediction, target, order=1)
    assert click is not None
    assert click.is_positive
    assert click.point_index in {4, 5}


def test_click_search_worker_count_is_scheduling_only() -> None:
    rng = np.random.default_rng(7)
    xyz = rng.normal(size=(128, 3)).astype(np.float32)
    target = np.zeros(128, dtype=bool)
    target[15:91] = True
    prediction = np.zeros(128, dtype=bool)
    prediction[20:30] = True
    prediction[100:111] = True
    assert select_next_click(
        xyz, prediction, target, order=1, workers=1
    ) == select_next_click(xyz, prediction, target, order=1, workers=-1)


def test_interaction_forces_clicked_labels_and_aggregates_metrics() -> None:
    xyz = np.column_stack([np.arange(5), np.zeros((5, 2))]).astype(np.float32)
    target = np.array([0, 1, 1, 1, 0], dtype=bool)

    def predictor(_xyz, previous, _clicks):
        return previous

    result = evaluate_interactive_predictions(
        xyz,
        target,
        target,
        np.arange(5),
        predictor,
        max_clicks=20,
    )
    assert result["trajectory"][1] > 0
    assert set(result["seed_satisfaction"][1]) == {
        "positive",
        "negative",
        "all",
    }
    # The inert predictor misses each newly placed click before the official
    # evaluator overwrite, so this exposes method-side constraint failures
    # without changing the benchmark trajectory.
    assert result["seed_satisfaction"][1]["positive"] == 0.0
    metrics = aggregate_official_metrics([result["trajectory"]])
    assert set(("IoU@1", "IoU@15", "NoC@50", "NoC@90")) <= set(metrics)


def test_interaction_records_label_free_stages_before_protocol_overwrite() -> None:
    xyz = np.column_stack([np.arange(5), np.zeros((5, 2))]).astype(np.float32)
    target = np.array([0, 1, 1, 1, 0], dtype=bool)

    class DiagnosticPredictor:
        last_seed_satisfaction_stages = {}

        def __call__(self, _xyz, previous, clicks):
            self.last_seed_satisfaction_stages = {
                "primitive_solver": {
                    "positive": 0.5,
                    "negative": None,
                    "all": 0.5,
                },
                "official_continuous_readout": {
                    "positive": 0.0,
                    "negative": None,
                    "all": 0.0,
                },
                "post_selection_pre_overwrite": {
                    "positive": 0.0,
                    "negative": None,
                    "all": 0.0,
                },
            }
            return previous

    result = evaluate_interactive_predictions(
        xyz,
        target,
        target,
        np.arange(5),
        DiagnosticPredictor(),
        max_clicks=1,
    )
    stages = result["seed_satisfaction_stages"][1]
    assert stages["primitive_solver"]["positive"] == 0.5
    assert stages["official_continuous_readout"]["positive"] == 0.0
    assert stages["post_selection_pre_overwrite"]["positive"] == 0.0
    assert stages["protocol_overwrite"]["positive"] == 1.0


def test_interaction_health_reports_click_monotonicity_and_regression_size() -> None:
    healthy = {step: 0.1 + 0.01 * step for step in range(1, 21)}
    regressing = dict(healthy)
    regressing[6] = regressing[5] - 0.20
    report = interaction_health_metrics([healthy, regressing], max_clicks=20)
    assert report["trajectory_count"] == 2
    assert report["transition_count"] == 38
    assert report["monotonic_trajectory_fraction"] == pytest.approx(0.5)
    assert report["monotonic_transition_fraction"] == pytest.approx(37 / 38)
    assert report["mean_regression_magnitude"] == pytest.approx(0.20)


def test_interaction_health_aggregates_pre_overwrite_seed_satisfaction() -> None:
    trajectory = {step: 0.1 + 0.01 * step for step in range(1, 21)}
    satisfaction = {
        step: {
            "positive": 1.0,
            "negative": None if step == 1 else 0.75,
            "all": 1.0 if step == 1 else 0.875,
        }
        for step in range(1, 21)
    }
    report = interaction_health_metrics(
        [trajectory],
        seed_satisfaction=[satisfaction],
        max_clicks=20,
    )
    assert report["positive_seed_satisfaction"] == 1.0
    assert report["negative_seed_satisfaction"] == pytest.approx(0.75)
    assert report["all_seed_satisfaction"] == pytest.approx(
        (1.0 + 19 * 0.875) / 20
    )


def test_canonical_predictor_hard_clamps_positive_and_negative_clicks() -> None:
    xyz = np.column_stack([np.arange(6) * 0.05, np.zeros((6, 2))]).astype(np.float32)
    features = np.array(
        [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
        dtype=np.float32,
    )
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=None,
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=2),
        solver_config=SupportSolverConfig(
            solver_type="random_walker",
            cg_iterations=16,
        ),
    )
    prediction = predictor(
        xyz,
        np.zeros(6, dtype=bool),
        (
            Click(point_index=1, is_positive=True, order=0),
            Click(point_index=4, is_positive=False, order=1),
        ),
    )
    assert prediction[1]
    assert not prediction[4]


def test_canonical_field_predictor_uses_standard_world_query_without_observation_lift() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_primitive_support_graph(
        torch.from_numpy(xyz),
        appearance_features=features,
        boundary_features=features,
        config=SupportGraphConfig(neighbors=2),
    )
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=torch.from_numpy(xyz),
        gaussian_covariance=torch.eye(3).repeat(3, 1, 1) * 0.0001,
        gaussian_precision=torch.eye(3).repeat(3, 1, 1) * 10000.0,
        gaussian_opacity=torch.ones(3),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_canonical_signature("dino", 2),
        boundary_signature=_canonical_signature("sam3", 2),
        graph=graph,
        official_xyz=xyz,
        device="cpu",
        solver_config=SupportSolverConfig(
            solver_type="random_walker", cg_iterations=16
        ),
        readout_candidate_k=3,
    )

    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (
            Click(point_index=0, is_positive=True, order=0),
            Click(point_index=2, is_positive=False, order=1),
        ),
    )

    assert prediction[0]
    assert not prediction[2]
    assert set(predictor.last_seed_satisfaction_stages) == {
        "primitive_solver",
        "official_continuous_readout",
        "post_selection_pre_overwrite",
    }
    for stage in predictor.last_seed_satisfaction_stages.values():
        assert set(stage) == {"positive", "negative", "all"}
    report = predictor.protocol_report()
    assert report["observation_lift"] == "none"
    assert report["labels_opened"] is False
    assert report["primitive_reliability_applied"] is False


def test_canonical_field_click_constraint_survives_readout_support_gate() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_primitive_support_graph(
        torch.from_numpy(xyz),
        appearance_features=features,
        boundary_features=features,
        config=SupportGraphConfig(neighbors=2),
    )
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=torch.from_numpy(xyz),
        gaussian_covariance=torch.eye(3).repeat(3, 1, 1) * 0.0001,
        gaussian_precision=torch.eye(3).repeat(3, 1, 1) * 10000.0,
        gaussian_opacity=torch.ones(3),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_canonical_signature("dino", 2),
        boundary_signature=_canonical_signature("sam3", 2),
        graph=graph,
        official_xyz=xyz,
        device="cpu",
        solver_config=SupportSolverConfig(
            solver_type="random_walker", cg_iterations=16
        ),
        readout_candidate_k=3,
        readout_support_threshold=1e9,
        point_readout_constraint="click_score_clamp",
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=0, is_positive=True, order=0),),
    )
    assert prediction[0]
    assert predictor.last_seed_satisfaction_stages[
        "post_selection_pre_overwrite"
    ]["all"] == 1.0
    assert predictor.protocol_report()["point_readout_constraint"] == "click_score_clamp"
    with pytest.raises(ValueError, match="conflicting"):
        predictor(
            xyz,
            prediction,
            (
                Click(point_index=1, is_positive=True, order=0),
                Click(point_index=1, is_positive=False, order=1),
            ),
        )


def test_full_observation_contract_rejects_sparse_frames25k_source() -> None:
    with pytest.raises(ValueError, match="scannet_frames_25k"):
        validate_observation_contract(
            "scannet_full_observation_v1",
            "/mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k/scene0011_01",
            require_support_gate=True,
        )


def test_render_contract_exposes_the_actual_observation_root(tmp_path) -> None:
    contract_path = tmp_path / "scene.yaml"
    contract_path.write_text(
        "scene_root: /mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k/scene0011_01\n"
        "field_source_excluded_query_frame_count: 3\n"
        "field_source_excluded_query_frame_ids_sha256: digest\n",
        encoding="utf-8",
    )
    source = observation_source_from_render_contract(contract_path)
    assert source["source_observation_root"].endswith("scannet_frames_25k/scene0011_01")
    assert source["field_source_excluded_query_frame_count"] == "3"
    assert source["field_source_excluded_query_frame_ids_sha256"] == "digest"
    with pytest.raises(ValueError, match="scannet_frames_25k"):
        validate_observation_contract(
            "scannet_full_observation_v1",
            source["source_observation_root"],
            require_support_gate=True,
            declared_source_contract=source["declared_source_contract"],
        )


def test_full_observation_contract_requires_explicit_full_sens_declaration() -> None:
    with pytest.raises(ValueError, match="explicitly declare"):
        validate_observation_contract(
            "scannet_full_observation_v1",
            "/mnt/pool/Datasets/ScanNet/scans/scene0011_01",
            require_support_gate=True,
            declared_source_contract="field_only_dense_rgbd_v1",
        )


def test_full_observation_contract_requires_an_auditable_source_digest() -> None:
    with pytest.raises(ValueError, match="field-source contract digest"):
        validate_observation_contract(
            "scannet_full_observation_v1",
            "/mnt/pool/Datasets/ScanNet/scans/scene0011_01",
            require_support_gate=True,
            declared_source_contract="scannet_full_observation_v1",
        )


def test_full_observation_contract_requires_matching_source_version() -> None:
    with pytest.raises(ValueError, match="matching source-contract version"):
        validate_observation_contract(
            "scannet_full_observation_v1",
            "/mnt/pool/Datasets/ScanNet/scans/scene0011_01",
            require_support_gate=True,
            declared_source_contract="scannet_full_observation_v1",
            field_source_contract_sha256="a" * 64,
            field_source_contract_version="scannet-agile-dense-observation-field-v1",
        )


def test_full_observation_pilot_requires_full_sens_provenance_and_support_gate() -> None:
    kwargs = {
        "source_observation_root": "/mnt/pool/Datasets/ScanNet/scans/scene0011_01",
        "declared_source_contract": "scannet_full_observation_v1",
        "field_source_contract_sha256": "a" * 64,
        "field_source_contract_version": "scannet_full_observation_v1",
    }
    with pytest.raises(ValueError, match="requires the support gate"):
        validate_observation_contract(
            "scannet_full_observation_pilot",
            require_support_gate=False,
            **kwargs,
        )
    validate_observation_contract(
        "scannet_full_observation_pilot",
        require_support_gate=True,
        **kwargs,
    )


def test_ungated_full_observation_diagnostic_requires_explicit_opt_in() -> None:
    """A fast score may not silently masquerade as a formal AGILE result."""

    kwargs = {
        "source_observation_root": "/mnt/pool/Datasets/ScanNet/scans/scene0011_01",
        "declared_source_contract": "scannet_full_observation_v1",
        "field_source_contract_sha256": "a" * 64,
        "field_source_contract_version": "scannet_full_observation_v1",
    }
    with pytest.raises(ValueError, match="explicit"):
        validate_observation_contract(
            "scannet_full_observation_diagnostic_v1",
            require_support_gate=False,
            **kwargs,
        )
    validate_observation_contract(
        "scannet_full_observation_diagnostic_v1",
        require_support_gate=False,
        allow_ungated_diagnostic=True,
        **kwargs,
    )
    with pytest.raises(ValueError, match="diagnostic"):
        validate_observation_contract(
            "scannet_full_observation_diagnostic_v1",
            require_support_gate=True,
            allow_ungated_diagnostic=True,
            **kwargs,
        )


def test_full_observation_support_gate_rejects_gaussian_tail_only_coverage() -> None:
    with pytest.raises(ValueError, match="meaningful continuous support"):
        validate_continuous_support_threshold(
            "scannet_full_observation_pilot", 1e-6
        )
    validate_continuous_support_threshold(
        "scannet_full_observation_pilot", 1e-2
    )
    # Legacy/pilot source contracts retain their historical output threshold;
    # the stricter requirement is only the full-observation quality gate.
    validate_continuous_support_threshold("dense_overlap_pilot", 1e-6)


def test_full_observation_requires_coverage_ranked_mpr_from_the_same_source() -> None:
    contract = canonical_observation_contract(
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME
    )
    metadata = {
        "observation_lifting_contract": contract,
        "observation_lifting_contract_sha256": observation_contract_sha256(contract),
        "num_declared_views": 240,
        "aggregation_mode": contract["aggregation_mode"],
        "registration_weight_mode": contract["registration_weight_mode"],
        "raster_view_fusion": contract["raster_view_fusion"],
        "normalize_each_view": contract["normalize_each_view"],
        "per_view_normalization_applied": True,
        "depth_tolerance": contract["depth_tolerance"],
        "relative_depth_tolerance": contract["relative_depth_tolerance"],
        "alpha_threshold": contract["alpha_threshold"],
        "robust_mpr": False,
        "full_observation_coverage_order_applied": True,
        "full_observation_source_view_count": 480,
        "full_observation_source_contract_sha256": "a" * 64,
        "full_observation_source_contract_version": "scannet_full_observation_v1",
    }
    validate_full_observation_mpr_contract(
        "scannet_full_observation_pilot",
        metadata,
        expected_source_contract_sha256="a" * 64,
        expected_source_contract_version="scannet_full_observation_v1",
    )

    legacy = dict(metadata)
    legacy["observation_lifting_contract"] = canonical_observation_contract()
    legacy["observation_lifting_contract_sha256"] = observation_contract_sha256(
        legacy["observation_lifting_contract"]
    )
    legacy["num_declared_views"] = 120
    with pytest.raises(ValueError, match="canonical-full-observation-mpr-v1"):
        validate_full_observation_mpr_contract(
            "scannet_full_observation_pilot",
            legacy,
            expected_source_contract_sha256="a" * 64,
            expected_source_contract_version="scannet_full_observation_v1",
        )

    wrong_source = dict(metadata)
    wrong_source["full_observation_source_contract_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source-contract digest"):
        validate_full_observation_mpr_contract(
            "scannet_full_observation_pilot",
            wrong_source,
            expected_source_contract_sha256="a" * 64,
            expected_source_contract_version="scannet_full_observation_v1",
        )


def test_full_observation_accepts_the_explicit_480_view_mpr_v2() -> None:
    contract = canonical_observation_contract(
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME
    )
    metadata = {
        "observation_lifting_contract": contract,
        "observation_lifting_contract_sha256": observation_contract_sha256(contract),
        "num_declared_views": 476,
        "aggregation_mode": contract["aggregation_mode"],
        "registration_weight_mode": contract["registration_weight_mode"],
        "raster_view_fusion": contract["raster_view_fusion"],
        "normalize_each_view": contract["normalize_each_view"],
        "per_view_normalization_applied": True,
        "depth_tolerance": contract["depth_tolerance"],
        "relative_depth_tolerance": contract["relative_depth_tolerance"],
        "alpha_threshold": contract["alpha_threshold"],
        "robust_mpr": False,
        "full_observation_coverage_order_applied": True,
        "full_observation_source_view_count": 480,
        "full_observation_source_contract_sha256": "a" * 64,
        "full_observation_source_contract_version": "scannet_full_observation_v1",
    }
    validate_full_observation_mpr_contract(
        "scannet_full_observation_pilot",
        metadata,
        expected_source_contract_sha256="a" * 64,
        expected_source_contract_version="scannet_full_observation_v1",
    )


def test_full_observation_accepts_the_explicit_960_view_mpr_v3() -> None:
    contract = canonical_observation_contract(
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME
    )
    metadata = {
        "observation_lifting_contract": contract,
        "observation_lifting_contract_sha256": observation_contract_sha256(contract),
        "num_declared_views": 956,
        "aggregation_mode": contract["aggregation_mode"],
        "registration_weight_mode": contract["registration_weight_mode"],
        "raster_view_fusion": contract["raster_view_fusion"],
        "normalize_each_view": contract["normalize_each_view"],
        "per_view_normalization_applied": True,
        "depth_tolerance": contract["depth_tolerance"],
        "relative_depth_tolerance": contract["relative_depth_tolerance"],
        "alpha_threshold": contract["alpha_threshold"],
        "robust_mpr": False,
        "full_observation_coverage_order_applied": True,
        "full_observation_source_view_count": 960,
        "full_observation_source_contract_sha256": "a" * 64,
        "full_observation_source_contract_version": "scannet_full_observation_v1",
    }
    validate_full_observation_mpr_contract(
        "scannet_full_observation_pilot",
        metadata,
        expected_source_contract_sha256="a" * 64,
        expected_source_contract_version="scannet_full_observation_v1",
    )


def test_official_spatial_teacher_provenance_gate_is_artifact_only() -> None:
    metadata = {
        "capability_training_mpr_sources": {
            "appearance": {
                "capability_map_source": "official_extracted",
                "capability_adaptor_execution": "official_c_radio_runtime_adaptor_output",
            },
            "boundary": {
                "capability_map_source": "official_extracted",
                "capability_adaptor_execution": "official_c_radio_runtime_adaptor_output",
            },
        },
        "render_capability_teacher_source": "official_extracted",
    }
    report = validate_capability_teacher_fidelity(
        metadata, require_official_extracted=True
    )
    assert report["capability_training_mpr_sources"] == {
        "appearance": "official_extracted",
        "boundary": "official_extracted",
    }
    assert report["requires_official_extracted_capability_teachers"] is True

    with pytest.raises(ValueError, match="native official C-RADIO"):
        validate_capability_teacher_fidelity(
            {
                **metadata,
                "render_capability_teacher_source": "project_raw",
            },
            require_official_extracted=True,
        )

def test_support_identity_ignores_runtime_geometry_cache_reuse_flag() -> None:
    first = {"scene_id": "scene0011_01", "continuous_support_fraction": 0.97, "geometry_cache_reused": False}
    second = {**first, "geometry_cache_reused": True}
    assert stable_support_record(first) == stable_support_record(second)


def test_named_pilot_contracts_reject_a_mismatched_field_source() -> None:
    with pytest.raises(ValueError, match="dense_pfpr_queryheldout_v1"):
        validate_observation_contract(
            "dense_pfpr_queryheldout_pilot",
            "/tmp/scene0011_01",
            require_support_gate=False,
            declared_source_contract="field_only_dense_rgbd_v1",
        )


def test_named_pilot_contract_requires_a_source_digest_after_matching_name() -> None:
    with pytest.raises(ValueError, match="field-source contract digest"):
        validate_observation_contract(
            "dense_pfpr_queryheldout_pilot",
            "/tmp/scene0011_01",
            require_support_gate=False,
            declared_source_contract="dense_pfpr_queryheldout_v1",
        )
    with pytest.raises(ValueError, match="dense_agile_all_observations_pilot"):
        validate_observation_contract(
            "dense_agile_all_observations_pilot",
            "/tmp/scene0011_01",
            require_support_gate=False,
            declared_source_contract="dense_pfpr_queryheldout_v1",
        )


def test_gaussian_covariances_support_explicit_geometry_accessors() -> None:
    class ExplicitGeometry:
        def get_rotation(self):
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        def get_scaling(self):
            return torch.tensor([[0.1, 0.2, 0.3]])

    covariance = _gaussian_covariances(ExplicitGeometry())
    assert torch.allclose(
        covariance,
        torch.diag_embed(torch.tensor([[0.01, 0.04, 0.09]])),
    )


def test_canonical_reader_convolves_with_the_fixed_evaluator_voxel_cell() -> None:
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    features = torch.tensor([[1.0, 0.0]])
    graph = build_primitive_support_graph(
        torch.from_numpy(xyz),
        appearance_features=features,
        boundary_features=features,
        config=SupportGraphConfig(neighbors=1),
    )
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=torch.from_numpy(xyz),
        gaussian_covariance=torch.eye(3)[None] * 1e-8,
        gaussian_precision=torch.eye(3)[None] * 1e8,
        gaussian_opacity=torch.ones(1),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_canonical_signature("dino", 2),
        boundary_signature=_canonical_signature("sam3", 2),
        graph=graph,
        official_xyz=np.array([[0.04, 0.0, 0.0]], dtype=np.float32),
        device="cpu",
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=8),
        readout_candidate_k=1,
        evaluation_voxel_size_m=0.05,
        node_reliability=torch.ones(1),
    )
    assert bool(predictor.readout_valid[0])
    report = predictor.protocol_report()
    assert report["readout_kernel"] == "gaussian_convolved_with_evaluator_voxel_cell"
    assert report["voxel_cell_variance_m2"] == pytest.approx(0.05**2 / 12.0)
    assert report["primitive_reliability_applied"] is True


def test_canonical_reader_keeps_exact_click_seeds_separate_from_voxel_readout() -> None:
    """The released callback point is exact even though its output is a cell."""

    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    features = torch.tensor([[1.0, 0.0]])
    graph = build_primitive_support_graph(
        torch.from_numpy(xyz),
        appearance_features=features,
        boundary_features=features,
        config=SupportGraphConfig(neighbors=1),
    )
    native_precision = torch.eye(3)[None] * 1e8
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=torch.from_numpy(xyz),
        gaussian_covariance=torch.eye(3)[None] * 1e-8,
        gaussian_precision=native_precision,
        gaussian_opacity=torch.ones(1),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_canonical_signature("dino", 2),
        boundary_signature=_canonical_signature("sam3", 2),
        graph=graph,
        official_xyz=np.array([[0.04, 0.0, 0.0]], dtype=np.float32),
        device="cpu",
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=8),
        readout_candidate_k=1,
        evaluation_voxel_size_m=0.05,
        click_seed_kernel="native_gaussian",
    )
    assert torch.allclose(predictor.seed_precision, native_precision)
    assert not torch.allclose(predictor.seed_precision, predictor.readout_precision)
    report = predictor.protocol_report()
    assert report["click_seed_kernel"] == "native_gaussian"


def test_canonical_reader_can_hard_anchor_only_the_best_gaussian() -> None:
    """A point click may keep soft local evidence while hard-clamping one row.

    This is a generic continuous-field interaction contract: covariance-aware
    responsibilities still construct the query descriptor, but the exact hard
    click constraint cannot accidentally cover every broad Gaussian near the
    coordinate.
    """

    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.03, 0.0, 0.0], [0.06, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_primitive_support_graph(
        torch.from_numpy(xyz),
        appearance_features=features,
        boundary_features=features,
        config=SupportGraphConfig(neighbors=2),
    )
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=torch.from_numpy(xyz),
        gaussian_covariance=torch.eye(3).repeat(3, 1, 1) * 0.001,
        gaussian_precision=torch.eye(3).repeat(3, 1, 1) * 1000.0,
        gaussian_opacity=torch.ones(3),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_canonical_signature("dino", 2),
        boundary_signature=_canonical_signature("sam3", 2),
        graph=graph,
        official_xyz=xyz,
        device="cpu",
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
        readout_candidate_k=3,
        seed_topk=1,
    )
    report = predictor.protocol_report()
    assert report["hard_seed_topk"] == 1
    assert report["hard_seed_conflict_policy"] == "positive_priority"
    assert report["hard_seed_conflict_margin"] == 0.0
    assert report["world_point_prototype_mode"] == "per_click_local"


def test_canonical_reader_reports_explicit_min_seed_cover_variant() -> None:
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    features = torch.tensor([[1.0, 0.0]])
    graph = build_primitive_support_graph(
        torch.from_numpy(xyz),
        appearance_features=features,
        boundary_features=features,
        config=SupportGraphConfig(neighbors=1),
    )
    predictor = CanonicalFieldPointPredictor(
        gaussian_xyz=torch.from_numpy(xyz),
        gaussian_covariance=torch.eye(3)[None] * 0.001,
        gaussian_precision=torch.eye(3)[None] * 1000.0,
        gaussian_opacity=torch.ones(1),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=_canonical_signature("dino", 2),
        boundary_signature=_canonical_signature("sam3", 2),
        graph=graph,
        official_xyz=xyz,
        device="cpu",
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=8),
        readout_candidate_k=1,
        selection_mode=SelectionMode.MIN_SEED_COVER,
    )
    assert predictor.protocol_report()["selection_mode"] == "min_seed_cover"


def test_observed_domain_is_exactly_legacy_when_every_row_is_observed() -> None:
    xyz = np.column_stack([np.arange(6) * 0.05, np.zeros((6, 2))]).astype(np.float32)
    features = np.array(
        [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
        dtype=np.float32,
    )
    common = dict(
        appearance_features=features,
        boundary_features=features,
        observation_valid=np.ones(len(xyz), dtype=bool),
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=2),
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
    )
    observed = CanonicalPointPredictor(
        xyz, features, observation_lift_mode="observed_domain", **common
    )
    legacy = CanonicalPointPredictor(
        xyz, features, observation_lift_mode="full_cache_legacy", **common
    )
    clicks = (
        Click(point_index=1, is_positive=True, order=0),
        Click(point_index=4, is_positive=False, order=1),
    )
    np.testing.assert_array_equal(
        observed(xyz, np.zeros(len(xyz), dtype=bool), clicks),
        legacy(xyz, np.zeros(len(xyz), dtype=bool), clicks),
    )


def test_canonical_predictor_keeps_only_the_positive_seeded_component() -> None:
    xyz = np.array(
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [4.00, 0.0, 0.0],
            [4.05, 0.0, 0.0],
            [4.10, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Every point has the same visual unary. The instance query must still
    # return only the physical component reached by its registered click.
    features = np.array([[1, 0]] * len(xyz), dtype=np.float32)
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=None,
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=2),
        solver_config=SupportSolverConfig(
            solver_type="random_walker", cg_iterations=32
        ),
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=1, is_positive=True, order=0),),
    )
    np.testing.assert_array_equal(prediction, [True, True, True, False, False, False])


def test_observed_domain_lifts_nearby_unobserved_click_without_zero_feature_seed() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.float32)
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=np.array([True, True, False]),
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=1),
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
        observation_lift_mode="observed_domain",
        observation_lift_neighbors=2,
        observation_lift_maximum_distance_m=0.10,
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=2, is_positive=True, order=0),),
    )
    assert prediction[2]
    report = predictor.observation_lift_report()
    assert report["solver_nodes"] == 2
    assert report["projectable_fraction"] == 1.0


def test_observed_domain_refuses_to_hallucinate_remote_unobserved_click() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [1.00, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.float32)
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=np.array([True, True, False]),
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=1),
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
        observation_lift_mode="observed_domain",
        observation_lift_neighbors=2,
        observation_lift_maximum_distance_m=0.10,
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=2, is_positive=True, order=0),),
    )
    assert not prediction.any()


def test_vector_feature_interpolation_is_query_and_label_free() -> None:
    source_xyz = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    source_features = np.eye(2, dtype=np.float32)
    target_xyz = np.array([[0, 0, 0], [0.5, 0, 0], [3, 0, 0]], dtype=np.float32)
    mapped, valid = interpolate_feature_matrix(
        source_xyz,
        source_features,
        target_xyz,
        neighbors=2,
        maximum_distance_m=1.0,
    )
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(mapped[0], [1, 0], atol=2e-4)
    np.testing.assert_allclose(mapped[1], [2 ** -0.5, 2 ** -0.5], atol=1e-4)


def test_quantized_feature_cache_requires_exact_unique_map(tmp_path) -> None:
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    unique = np.array([0, 2, 5], dtype=np.int64)
    path = tmp_path / "scene.npz"
    np.savez(
        path,
        xyz=xyz[unique],
        unique_map=unique,
        radio_features=np.ones((3, 4), dtype=np.float16),
        appearance_features=np.ones((3, 2), dtype=np.float16),
        boundary_features=np.ones((3, 2), dtype=np.float16),
    )
    cache = _load_feature_cache(path, xyz, quantized_unique_map=unique)
    assert bool(cache["_is_quantized"])


def test_feature_cache_rejects_misaligned_observation_validity(tmp_path) -> None:
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    path = tmp_path / "scene.npz"
    np.savez(
        path,
        xyz=xyz,
        radio_features=np.ones((6, 4), dtype=np.float16),
        valid=np.ones(5, dtype=bool),
    )
    try:
        _load_feature_cache(path, xyz, quantized_unique_map=np.arange(6))
    except ValueError as error:
        assert "valid does not align" in str(error)
    else:
        raise AssertionError("misaligned feature validity must fail closed")


def test_merge_requires_exact_official_coverage_and_preserves_trajectories(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(
        root / "single" / "object_ids.npy",
        np.asarray([["scene_a", "1"], ["scene_b", "2"]]),
    )
    (root / "single" / "object_classes.txt").write_text("chair\ntable\n")
    protocol = {
        "voxel_size_m": 0.05,
        "max_clicks": 20,
        "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
        "clicked_labels_forced": True,
        "test_set_calibration": False,
        "selection_mode": "seeded_component",
        "unary_mode": "shared_capability",
        "appearance_unary_weight": 1.0,
        "boundary_unary_weight": 0.35,
        "observation_lift_mode": "observed_domain",
        "observation_lift_neighbors": 3,
        "observation_lift_maximum_distance_m": 0.10,
    }

    def shard(scene, object_id, semantic, coverage, value):
        return {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "protocol": protocol,
            "scene_coverage": [
                {
                    "scene_id": scene,
                    "quantized_points": 4,
                    "valid_feature_points": 3,
                    "feature_coverage": coverage,
                    "observation_lift": {
                        "mode": "observed_domain",
                        "solver_nodes": 3,
                        "projectable_points": 4,
                        "projectable_fraction": 1.0,
                        "neighbors": 3,
                        "maximum_distance_m": 0.10,
                    },
                }
            ],
            "rows": [
                {
                    "scene_id": scene,
                    "object_id": object_id,
                    "semantic_class": semantic,
                    "trajectory": {str(step): value for step in range(1, 21)},
                }
            ],
        }

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(shard("scene_a", 1, "chair", 0.75, 0.25)))
    second.write_text(json.dumps(shard("scene_b", 2, "table", 1.0, 0.75)))
    output = tmp_path / "merged.json"
    report = merge(root, [second, first], output)
    assert report["metrics"]["IoU@1"] == 0.5
    assert report["coverage_summary"]["mean_feature_coverage"] == 0.875
    assert report["coverage_summary"]["mean_projectable_fraction"] == 1.0
    assert [row["scene_id"] for row in report["rows"]] == ["scene_a", "scene_b"]
