import json
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import radio_gs.scripts.eval_nvos_gaussian_first as nvos_eval
from radio_gs.field import FeatureSpaceSignature
from radio_gs.querying.query_compilers import (
    _deterministic_prototypes,
    compile_registered_primitive_seeds,
)
from radio_gs.querying.reliability_fusion import (
    DUAL_PROTOTYPE_SEED_PROVENANCE,
    DUAL_SOLVER_SEED_PROVENANCE,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _FROZEN_LEGACY_PROTOTYPE_ALPHA_THRESHOLD,
    _PROBABILITY_PRESERVING_SOURCE_UNARY,
    _REGISTERED_QUERY_LIKELIHOOD_SOURCE_RECONSTRUCTION,
    _SOURCE_ONLY_CORRESPONDENCE_COMPLETION,
    _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION,
    _apply_hierarchical_source_completion_trust,
    _dataset_protocol_contract,
    _joint_signed_observation_seeds,
    _load_training_poses,
    _registered_solver_masses,
    _registered_query_likelihood_contract,
    _prompt_cycle_fixed_ranking,
    _prompt_cycle_reconstruction_metrics,
    _probability_preserving_registration_maps,
    _require_bipolar_solver_support,
    _render_registered_stage_maps,
    _registered_strong_unary_method_contract,
    _registered_posterior_consensus_method_contract,
    _requires_legacy_prototype_observation,
    _resolve_observed_feature_path,
    _resolve_scene_carrier_assets,
    _scaled_raster_shape,
    _source_completion_unary_contract,
    _validate_direct_raster_adjoint_args,
    _validate_hard_seed_anchor_only_probability_args,
    _validate_object_multiview_region_memory_args,
    _validate_registered_disjoint_domain_composition_args,
    _validate_registered_prototype_seed_construction_args,
    _validate_registered_query_likelihood_args,
    _validate_registered_reference_threshold_calibration_args,
    _validate_source_completion_unary_args,
    _validate_source_only_correspondence_args,
    _valid_normalized_score_map,
    _weighted_spherical_prototypes,
)


def _source_correspondence_args(**updates: object) -> Namespace:
    values = {
        "source_only_correspondence_completion": (
            _SOURCE_ONLY_CORRESPONDENCE_COMPLETION
        ),
        "source_correspondence_support_graph": "/frozen/graph.pt",
        "source_correspondence_support_graph_sha256": "a" * 64,
        "source_multiview_responsibility_cache": "/frozen/views.pt",
        "source_multiview_responsibility_cache_sha256": "b" * 64,
        "support_mode": "canonical_support",
        "disable_registered_graph": True,
        "registered_readout_stage": "unary_prior",
        "registered_observation_fusion": "probability_mixture",
        "registered_query_likelihood_calibration": "none",
        "registered_forward_unary": "none",
        "query_conditioned_diffusion_kernel": "none",
        "negative_spatial_mode": "none",
        "require_asset_hashes": True,
    }
    values.update(updates)
    return Namespace(**values)


def test_source_correspondence_contract_requires_frozen_unary_only_path() -> None:
    _validate_source_only_correspondence_args(_source_correspondence_args())
    with pytest.raises(ValueError, match="disable-registered-graph"):
        _validate_source_only_correspondence_args(
            _source_correspondence_args(disable_registered_graph=False)
        )
    with pytest.raises(ValueError, match="valid SHA256"):
        _validate_source_only_correspondence_args(
            _source_correspondence_args(
                source_multiview_responsibility_cache_sha256="bad"
            )
        )


def test_source_correspondence_assets_rejected_when_mode_is_disabled() -> None:
    with pytest.raises(ValueError, match="assets require"):
        _validate_source_only_correspondence_args(
            _source_correspondence_args(
                source_only_correspondence_completion="none"
            )
        )


def _object_region_memory_args(**updates: object) -> Namespace:
    values = {
        "object_multiview_region_memory": (
            nvos_eval.OBJECT_MULTIVIEW_REGION_MEMORY
        ),
        "object_region_memory": "/frozen/memory.pt",
        "object_region_memory_sha256": "c" * 64,
        "support_mode": "canonical_support",
        "disable_registered_graph": True,
        "registered_readout_stage": "unary_prior",
        "registered_observation_fusion": "probability_mixture",
        "registered_query_likelihood_calibration": "none",
        "registered_forward_unary": "none",
        "source_only_correspondence_completion": "none",
        "query_conditioned_diffusion_kernel": "none",
        "negative_spatial_mode": "none",
        "require_asset_hashes": True,
    }
    values.update(updates)
    return Namespace(**values)


def test_object_region_memory_requires_frozen_unary_only_path() -> None:
    _validate_object_multiview_region_memory_args(_object_region_memory_args())
    with pytest.raises(ValueError, match="source-only-correspondence"):
        _validate_object_multiview_region_memory_args(
            _object_region_memory_args(
                source_only_correspondence_completion=(
                    _SOURCE_ONLY_CORRESPONDENCE_COMPLETION
                )
            )
        )
    with pytest.raises(ValueError, match="valid SHA256"):
        _validate_object_multiview_region_memory_args(
            _object_region_memory_args(object_region_memory_sha256="bad")
        )


def test_object_region_memory_assets_rejected_when_mode_is_disabled() -> None:
    with pytest.raises(ValueError, match="assets require"):
        _validate_object_multiview_region_memory_args(
            _object_region_memory_args(object_multiview_region_memory="none")
        )


def _disjoint_composition_args(**updates: object) -> Namespace:
    values = vars(_object_region_memory_args()).copy()
    values.update(
        {
            "registered_disjoint_domain_composition": (
                nvos_eval.DISJOINT_DOMAIN_COMPOSITION
            ),
            "registered_query_likelihood_calibration": (
                _REGISTERED_QUERY_LIKELIHOOD_SOURCE_RECONSTRUCTION
            ),
            "solver_support_threshold": 0.5,
        }
    )
    values.update(updates)
    return Namespace(**values)


def test_disjoint_composition_requires_both_frozen_inputs_and_threshold():
    args = _disjoint_composition_args()
    _validate_object_multiview_region_memory_args(args)
    _validate_registered_disjoint_domain_composition_args(args)
    with pytest.raises(ValueError, match="fixed 0.5"):
        _validate_registered_disjoint_domain_composition_args(
            _disjoint_composition_args(solver_support_threshold=0.4)
        )
    with pytest.raises(ValueError, match="object multiview"):
        _validate_registered_disjoint_domain_composition_args(
            _disjoint_composition_args(object_multiview_region_memory="none")
        )


def _registered_likelihood_args(**updates: object) -> Namespace:
    values = {
        "registered_query_likelihood_calibration": (
            _REGISTERED_QUERY_LIKELIHOOD_SOURCE_RECONSTRUCTION
        ),
        "support_mode": "canonical_support",
        "prompt_registration_mode": "raster_adjoint",
        "registered_observation_fusion": "probability_mixture",
        "registered_seed_construction": "joint_signed",
        "registered_forward_unary": "none",
        "registered_readout_stage": "unary_prior",
        "disable_registered_graph": True,
    }
    values.update(updates)
    return Namespace(**values)


def test_registered_query_likelihood_contract_is_source_only_and_fixed() -> None:
    args = _registered_likelihood_args()
    _validate_registered_query_likelihood_args(args)
    contract = _registered_query_likelihood_contract(args)
    assert contract is not None
    assert contract["target_rgb_mask_or_metric_used"] is False
    assert contract["parameter_sweep"] is False
    assert contract["unobserved_policy"] == "coverage=0_exact_abstention"


def test_registered_query_likelihood_rejects_graph_or_nonprobability_fusion() -> None:
    with pytest.raises(ValueError, match="probability_mixture"):
        _validate_registered_query_likelihood_args(
            _registered_likelihood_args(registered_observation_fusion="additive")
        )
    with pytest.raises(ValueError, match="disable-registered-graph"):
        _validate_registered_query_likelihood_args(
            _registered_likelihood_args(disable_registered_graph=False)
        )


def test_training_pose_loader_filters_target_view_overlap(tmp_path) -> None:
    pose = tmp_path / "pose.txt"
    np.savetxt(pose, np.eye(4, dtype=np.float32))
    (tmp_path / "train_frame_ids.json").write_text(json.dumps({"frame_ids": [7, 8]}))
    (tmp_path / "feature_pose_mapping.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "feature_frame_id": 7,
                        "camera_name": "target",
                        "pose_path": str(pose),
                    },
                    {
                        "feature_frame_id": 8,
                        "camera_name": "reference",
                        "pose_path": str(pose),
                    },
                ]
            }
        )
    )
    poses = _load_training_poses(tmp_path, ["target"])
    assert len(poses) == 1
    torch.testing.assert_close(poses[0], torch.eye(4))


def test_training_pose_loader_uses_resolved_camera_name_not_annotation_alias(tmp_path) -> None:
    pose = tmp_path / "pose.txt"
    np.savetxt(pose, np.eye(4, dtype=np.float32))
    (tmp_path / "train_frame_ids.json").write_text(json.dumps({"frame_ids": [1, 2]}))
    (tmp_path / "feature_pose_mapping.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "feature_frame_id": 1,
                        "camera_name": "IMG_4027",
                        "pose_path": str(pose),
                    },
                    {
                        "feature_frame_id": 2,
                        "camera_name": "IMG_4026",
                        "pose_path": str(pose),
                    },
                ]
            }
        )
    )

    # The annotation alias is image001, but the frozen protocol resolver maps
    # it to IMG_4027 before calling the loader.
    poses = _load_training_poses(tmp_path, ["IMG_4027"])

    assert len(poses) == 1


def test_observed_feature_path_uses_frozen_camera_mapping(tmp_path) -> None:
    feature_dir = tmp_path / "radio_features" / "backbone"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "rgb_7.pt"
    torch.save(torch.zeros(2, 3, 4), feature_path)
    (tmp_path / "feature_pose_mapping.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "feature_frame_id": 7,
                        "camera_name": "IMG_4027",
                        "colmap_camera_name": "IMG_4027",
                    }
                ]
            }
        )
    )
    (tmp_path / "radio_features" / "frame_manifest.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"source_rank": 7, "frame_idx": 7, "saved_stem": "rgb_7"}
                ]
            }
        )
    )

    assert _resolve_observed_feature_path(tmp_path, "IMG_4027") == feature_path


def test_scene_carrier_assets_allow_complete_explicit_frozen_bundle(
    tmp_path,
) -> None:
    queue_scene = tmp_path / "queue_scene"
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "carrier.pth"
    camera_map = tmp_path / "camera.json"
    for path in (config, checkpoint, camera_map):
        path.write_text("frozen", encoding="utf-8")

    actual = _resolve_scene_carrier_assets(
        queue_scene,
        scene_config=str(config),
        scene_checkpoint=str(checkpoint),
        camera_map=str(camera_map),
    )

    assert actual == (config.resolve(), checkpoint.resolve(), camera_map.resolve())


def test_scene_carrier_assets_reject_partial_explicit_bundle(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --scene-config"):
        _resolve_scene_carrier_assets(
            tmp_path,
            scene_config=str(tmp_path / "config.yaml"),
        )


def test_single_prototype_matches_weighted_mean() -> None:
    rows = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    weights = torch.tensor([3.0, 1.0])
    actual = _weighted_spherical_prototypes(rows, weights, 1)
    expected = F.normalize(torch.tensor([3.0, 1.0]), dim=0)
    torch.testing.assert_close(actual[0], expected)


def test_multiple_prototypes_preserve_distinct_prompt_appearances() -> None:
    rows = F.normalize(
        torch.tensor([[1.0, 0.02], [1.0, -0.02], [0.02, 1.0], [-0.02, 1.0]]),
        dim=-1,
    )
    centers = _weighted_spherical_prototypes(rows, torch.ones(4), 2)
    similarities = rows @ centers.T
    assert torch.all(similarities.amax(dim=1) > 0.99)
    assert abs(float(centers[0] @ centers[1])) < 0.1


def test_sparse_registered_compiler_uses_continuous_primitive_seeds() -> None:
    signature = FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio",
        raw_feature_dim=1280,
        adaptor_name="official",
        adaptor_output_dim=2,
        token_type="primitive",
    )
    query = compile_registered_primitive_seeds(
        torch.tensor([0.1, 0.8, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        appearance_features=torch.eye(3, 2),
        boundary_features=torch.eye(3, 2),
        appearance_signature=signature,
        boundary_signature=signature,
        prototype_count=1,
    )
    torch.testing.assert_close(
        query.positive_seeds.weights, torch.tensor([0.125, 1.0, 0.0])
    )
    assert query.negative_seeds is not None


def test_registered_compiler_default_prototype_seed_path_is_unchanged() -> None:
    signature = FeatureSpaceSignature(
        radio_version="unit",
        radio_checkpoint_sha256="radio",
        raw_feature_dim=3,
        adaptor_name="official",
        adaptor_output_dim=3,
        token_type="primitive",
    )
    features = torch.eye(3)
    positive = torch.tensor([0.8, 0.2, 0.0])
    negative = torch.tensor([0.0, 0.1, 0.9])
    baseline = compile_registered_primitive_seeds(
        positive,
        negative,
        appearance_features=features,
        boundary_features=features,
        appearance_signature=signature,
        boundary_signature=signature,
        prototype_count=1,
    )
    explicit_same = compile_registered_primitive_seeds(
        positive,
        negative,
        appearance_features=features,
        boundary_features=features,
        appearance_signature=signature,
        boundary_signature=signature,
        prototype_count=1,
        prototype_positive_seeds=positive,
        prototype_negative_seeds=negative,
    )

    assert "prototype_seed_decoupled" not in baseline.metadata
    torch.testing.assert_close(
        baseline.appearance_evidence.features,
        explicit_same.appearance_evidence.features,
    )
    torch.testing.assert_close(
        baseline.appearance_evidence.negatives,
        explicit_same.appearance_evidence.negatives,
    )
    torch.testing.assert_close(
        baseline.positive_seeds.weights,
        explicit_same.positive_seeds.weights,
    )
    torch.testing.assert_close(
        baseline.primitive_unary_evidence.values,
        explicit_same.primitive_unary_evidence.values,
    )


def test_registered_compiler_decouples_prototypes_from_solver_seeds() -> None:
    signature = FeatureSpaceSignature(
        radio_version="unit",
        radio_checkpoint_sha256="radio",
        raw_feature_dim=3,
        adaptor_name="official",
        adaptor_output_dim=3,
        token_type="primitive",
    )
    features = torch.eye(3)
    query = compile_registered_primitive_seeds(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=signature,
        boundary_signature=signature,
        prototype_count=1,
        prototype_positive_seeds=torch.tensor([0.0, 1.0, 0.0]),
        prototype_negative_seeds=torch.tensor([1.0, 0.0, 0.0]),
    )

    torch.testing.assert_close(
        query.appearance_evidence.features,
        features[1:2],
    )
    torch.testing.assert_close(
        query.appearance_evidence.negatives,
        features[0:1],
    )
    torch.testing.assert_close(
        query.positive_seeds.weights,
        torch.tensor([1.0, 0.0, 0.0]),
    )
    assert query.metadata["prototype_seed_decoupled"] is True
    assert query.metadata["prototype_seed_provenance"] == (
        "explicit_override_unspecified"
    )


def test_registered_compiler_default_all_zero_negative_remains_supported() -> None:
    signature = FeatureSpaceSignature(
        radio_version="unit",
        radio_checkpoint_sha256="radio",
        raw_feature_dim=3,
        adaptor_name="official",
        adaptor_output_dim=3,
        token_type="primitive",
    )
    features = torch.eye(3)
    query = compile_registered_primitive_seeds(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.zeros(3),
        appearance_features=features,
        boundary_features=features,
        appearance_signature=signature,
        boundary_signature=signature,
        prototype_count=1,
    )

    assert query.appearance_evidence.negatives is None
    assert query.negative_seeds is not None
    assert torch.equal(query.negative_seeds.weights, torch.zeros(3))
    assert query.negative_seed_groups is None


@pytest.mark.parametrize(
    "prototype_positive,prototype_negative",
    [
        (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])),
        (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])),
        (torch.tensor([0.0, float("nan"), 0.0]), torch.tensor([1.0, 0.0, 0.0])),
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),
    ],
)
def test_registered_compiler_rejects_invalid_decoupled_prototype_seeds(
    prototype_positive: torch.Tensor,
    prototype_negative: torch.Tensor,
) -> None:
    signature = FeatureSpaceSignature(
        radio_version="unit",
        radio_checkpoint_sha256="radio",
        raw_feature_dim=3,
        adaptor_name="official",
        adaptor_output_dim=3,
        token_type="primitive",
    )
    features = torch.eye(3)

    with pytest.raises(ValueError, match="prototype (positive|negative) seeds"):
        compile_registered_primitive_seeds(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            appearance_features=features,
            boundary_features=features,
            appearance_signature=signature,
            boundary_signature=signature,
            prototype_count=1,
            prototype_positive_seeds=prototype_positive,
            prototype_negative_seeds=prototype_negative,
        )


def test_native_prompt_raster_shape_is_not_tied_to_feature_resolution() -> None:
    assert _scaled_raster_shape(756, 1008, 1.0) == (756, 1008)
    assert _scaled_raster_shape(756, 1008, 0.5) == (378, 504)


def test_frozen_legacy_prototype_helper_keeps_alpha_and_cpu_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Renderer:
        def render_features(self, model: object, viewmat: torch.Tensor) -> dict:
            del model, viewmat
            return {
                "depth_map": torch.ones(1, 2, 2),
                "alpha_map": torch.ones(1, 2, 2),
            }

    def _fake_rasterize(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        return torch.ones(1, 2), torch.ones(1)

    monkeypatch.setattr(
        nvos_eval,
        "rasterize_registered_view_features",
        _fake_rasterize,
    )
    nvos_eval._rasterize_frozen_legacy_prototype_support(
        model=object(),
        renderer=_Renderer(),
        viewmat=torch.eye(4),
        prompt_maps=torch.ones(1, 2, 2, 2),
        depth_tolerance=0.08,
        relative_depth_tolerance=0.02,
    )

    assert captured["registration_alpha_threshold"] == 0.02
    assert captured["deterministic_cpu_accumulation"] is True
    assert captured["registration_weight_mode"] == "alpha_depth"


def test_prompt_cycle_metrics_and_ranking_are_fixed_in_prompt_space() -> None:
    mask = np.array([[True, False], [True, False]])
    visibility = np.ones((2, 2), dtype=np.float32)
    prototype = _prompt_cycle_reconstruction_metrics(
        np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        mask,
        visibility,
    )
    exact = _prompt_cycle_reconstruction_metrics(
        np.array([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32),
        mask,
        visibility,
    )
    ranking = _prompt_cycle_fixed_ranking(
        {"prototype_expert": prototype, "exact_expert": exact}
    )

    assert prototype["soft_iou"] > exact["soft_iou"]
    assert prototype["balanced_bce"] < exact["balanced_bce"]
    assert ranking["consensus_choice"] == "prototype_expert"
    assert ranking["uses_target_rgb_or_mask"] is False
    assert ranking["learned_or_scene_tuned_constants"] is False


def _direct_raster_adjoint_args(**updates: object) -> Namespace:
    values = {
        "registered_observation_fusion": "direct_raster_adjoint",
        "support_mode": "canonical_support",
        "prompt_registration_mode": "raster_adjoint",
        "prompt_registration_scale": 1.0,
        "alpha_threshold": 0.0,
        "registered_seed_unary_weight": 0.0,
        "registered_seed_construction": "joint_signed",
        "registered_prototype_seed_construction": "winner_take_all",
        "registered_forward_unary": "none",
    }
    values.update(updates)
    return Namespace(**values)


def test_direct_raster_adjoint_contract_accepts_exact_native_path() -> None:
    _validate_direct_raster_adjoint_args(_direct_raster_adjoint_args())


def test_direct_raster_adjoint_contract_fails_closed_before_model_loading() -> None:
    with pytest.raises(
        ValueError,
        match="--prompt-registration-mode raster_adjoint.*--alpha-threshold 0",
    ):
        _validate_direct_raster_adjoint_args(
            _direct_raster_adjoint_args(
                prompt_registration_mode="legacy_alpha_depth",
                alpha_threshold=0.1,
            )
        )


def test_raster_adjoint_bernoulli_poe_contract_is_exact_and_target_blind() -> None:
    args = _direct_raster_adjoint_args(
        registered_observation_fusion="raster_adjoint_bernoulli_poe"
    )
    _validate_direct_raster_adjoint_args(args)
    contract = _registered_posterior_consensus_method_contract(
        "raster_adjoint_bernoulli_poe"
    )
    assert contract is not None
    assert contract["neutral_expert_probability"] == 0.5
    assert contract["neutral_policy"] == "exact_field_unary_preservation"
    assert contract["certain_conflict_policy"] == "neutral_probability_0.5"
    assert contract["uses_target_rgb_or_mask"] is False
    assert contract["uses_scene_specific_constants"] is False


def test_raster_adjoint_bernoulli_poe_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="raster_adjoint_bernoulli_poe requires"):
        _validate_direct_raster_adjoint_args(
            _direct_raster_adjoint_args(
                registered_observation_fusion="raster_adjoint_bernoulli_poe",
                prompt_registration_scale=0.5,
            )
        )


def _dual_registration_args(**updates: object) -> Namespace:
    values = vars(
        _direct_raster_adjoint_args(
            registered_observation_fusion="dual_registration_bernoulli_poe"
        )
    )
    values.update(
        {
            "depth_tolerance": 0.08,
            "relative_depth_tolerance": 0.02,
            "support_threshold": 0.0,
            "prototype_count": 4,
            "prototype_strategy": "spherical_mean_fps",
            "appearance_weight": 1.0,
            "boundary_weight": 0.35,
            "prototype_temperature": 0.07,
            "feature_calibration": "none",
            "score_calibration": "none",
        }
    )
    values.update(updates)
    return Namespace(**values)


def test_dual_registration_contract_requires_frozen_legacy_expert() -> None:
    _validate_direct_raster_adjoint_args(_dual_registration_args())
    contract = _registered_posterior_consensus_method_contract(
        "dual_registration_bernoulli_poe"
    )
    assert contract is not None
    assert contract["observation_operator_coupling"] == (
        "independent_legacy_prototype_and_native_exact_adjoint"
    )
    assert contract["prototype_operator_contract"] == {
        "mode": "legacy_alpha_depth",
        "alpha_threshold": _FROZEN_LEGACY_PROTOTYPE_ALPHA_THRESHOLD,
        "deterministic_cpu_accumulation": True,
        "seed_provenance": DUAL_PROTOTYPE_SEED_PROVENANCE,
    }
    assert contract["exact_operator_contract"] == {
        "mode": "native_front_to_back_raster_adjoint",
        "alpha_threshold": 0.0,
        "seed_provenance": DUAL_SOLVER_SEED_PROVENANCE,
    }
    shared = _registered_posterior_consensus_method_contract(
        "raster_adjoint_bernoulli_poe"
    )
    assert shared is not None
    assert shared["observation_operator_coupling"] == (
        "shared_native_exact_adjoint_negative_ablation"
    )
    assert _requires_legacy_prototype_observation(
        "dual_registration_bernoulli_poe"
    )
    assert not _requires_legacy_prototype_observation(
        "raster_adjoint_bernoulli_poe"
    )


def test_dual_registration_contract_fails_on_legacy_operator_drift() -> None:
    with pytest.raises(
        ValueError,
        match="--depth-tolerance 0.08.*--prototype-count 4",
    ):
        _validate_direct_raster_adjoint_args(
            _dual_registration_args(
                depth_tolerance=0.1,
                prototype_count=8,
            )
        )


def _anchor_only_args(**updates: object) -> Namespace:
    values = {
        "registered_observation_fusion": (
            "hard_seed_anchor_only_probability"
        ),
        "support_mode": "canonical_support",
        "prompt_registration_mode": "raster_adjoint",
        "prompt_registration_scale": 1.0,
        "alpha_threshold": 0.0,
        "registered_seed_unary_weight": 0.0,
        "registered_seed_construction": "joint_signed",
        "registered_observation_confidence": "poisson_mass_coverage",
        "registered_observation_mass_scale": 1.0,
        "registered_observation_coverage_power": 1.0,
        "hard_seed_threshold": 0.20,
        "hard_seed_conflict_policy": "exclusive_relative",
        "hard_seed_conflict_margin": 0.0,
        "registered_forward_unary": "none",
        "registered_prototype_seed_construction": "shared",
    }
    values.update(updates)
    return Namespace(**values)


def test_anchor_only_contract_accepts_only_the_fixed_target_blind_path() -> None:
    _validate_hard_seed_anchor_only_probability_args(_anchor_only_args())


def test_anchor_only_contract_fails_closed_before_model_loading() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "--registered-observation-confidence poisson_mass_coverage.*"
            "--registered-observation-mass-scale 1.*"
            "--hard-seed-threshold 0.2.*--registered-forward-unary none"
        ),
    ):
        _validate_hard_seed_anchor_only_probability_args(
            _anchor_only_args(
                registered_observation_confidence="poisson_mass",
                registered_observation_mass_scale=2.0,
                hard_seed_threshold=0.3,
                registered_forward_unary="beta_coverage_v1",
            )
        )


def test_anchor_only_method_contract_discloses_bitwise_fallback() -> None:
    assert _registered_strong_unary_method_contract(
        "hard_seed_anchor_only_probability",
        anchor_threshold=0.20,
    ) == {
        "policy": "anchor_only_on_shared_hard_seed_rows",
        "anchor_threshold_source": "solver.hard_seed_threshold",
        "anchor_threshold": 0.20,
        "formula": (
            "a=1[c>0 and abs(s)>=tau]; c_eff=a; "
            "p=(1-a)p_field+a*q"
        ),
        "non_anchor_policy": "bitwise_field_unary_preservation",
        "new_numeric_constant": False,
    }


def test_decoupled_prototype_seed_contract_accepts_exact_dual_path() -> None:
    _validate_registered_prototype_seed_construction_args(
        _anchor_only_args(
            registered_prototype_seed_construction="winner_take_all"
        )
    )


def test_decoupled_prototype_seed_contract_rejects_coupled_solver_mode() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "decoupled registered prototype seeds require.*"
            "--registered-seed-construction joint_signed.*"
            "--registered-observation-fusion hard_seed_anchor_only_probability"
        ),
    ):
        _validate_registered_prototype_seed_construction_args(
            _anchor_only_args(
                registered_prototype_seed_construction="winner_take_all",
                registered_seed_construction="winner_take_all",
                registered_observation_fusion="additive",
            )
        )


def test_reference_threshold_calibration_contract_accepts_unary_only() -> None:
    _validate_registered_reference_threshold_calibration_args(
        Namespace(
            registered_reference_threshold_calibration=True,
            support_mode="canonical_support",
            registered_readout_stage="unary_prior",
            query_conditioned_diffusion_kernel="none",
            registered_forward_unary="none",
        )
    )


def test_reference_threshold_calibration_rejects_query_diffusion() -> None:
    with pytest.raises(
        ValueError,
        match="--query-conditioned-diffusion-kernel none",
    ):
        _validate_registered_reference_threshold_calibration_args(
            Namespace(
                registered_reference_threshold_calibration=True,
                support_mode="canonical_support",
                registered_readout_stage="unary_prior",
                query_conditioned_diffusion_kernel="ludvig_release_compat",
                registered_forward_unary="none",
            )
        )


def _source_completion_args(**updates: object) -> Namespace:
    values = {
        "source_completion_unary": _PROBABILITY_PRESERVING_SOURCE_UNARY,
        "source_completion": "/frozen/source.pt",
        "source_completion_sha256": "a" * 64,
        "source_completion_receipt": "/frozen/source.json",
        "source_completion_receipt_sha256": "b" * 64,
        "source_completion_calibration": "none",
        "source_completion_calibration_gate": "",
        "source_completion_calibration_gate_sha256": "",
        "support_mode": "canonical_support",
        "disable_registered_graph": True,
        "registered_readout_stage": "unary_prior",
        "query_conditioned_diffusion_kernel": "none",
        "registered_forward_unary": "none",
        "registered_observation_fusion": "probability_mixture",
        "registered_seed_unary_weight": 0.0,
        "prompt_registration_mode": "raster_adjoint",
        "prompt_registration_scale": 1.0,
        "alpha_threshold": 0.0,
        "feature_contribution_gamma": 1.0,
        "registered_reference_threshold_calibration": False,
    }
    values.update(updates)
    return Namespace(**values)


def test_probability_preserving_source_unary_contract_is_graph_off_and_parameter_free():
    args = _source_completion_args()
    _validate_source_completion_unary_args(args)
    _validate_registered_prototype_seed_construction_args(args)
    contract = _source_completion_unary_contract(args)
    assert contract is not None
    assert contract["graph"] == "disabled_zero_edge_unary_prior_only"
    assert contract["scene_specific_numeric_constants"] is False
    assert contract["target_rgb_or_mask_used"] is False
    assert contract["exact_adjoint_channels"] == ["c*q", "c*(1-q)"]
    assert contract["source_only_calibration"] is None


def test_source_completion_loo_gate_requires_both_immutable_assets() -> None:
    args = _source_completion_args(
        source_completion_calibration="all_trial_loo_majority_iou_v1",
        source_completion_calibration_gate="/frozen/gate.json",
        source_completion_calibration_gate_sha256="c" * 64,
    )
    _validate_source_completion_unary_args(args)
    contract = _source_completion_unary_contract(args)
    assert contract is not None
    assert contract["source_only_calibration"]["uses_target_rgb_mask_or_metric"] is False
    assert contract["calibration_reject_fallback"] == (
        "frozen_compact_hard_seed_anchor_only_probability_base"
    )

    with pytest.raises(ValueError, match="calibration-gate-sha256"):
        _validate_source_completion_unary_args(
            _source_completion_args(
                source_completion_calibration="all_trial_loo_majority_iou_v1",
                source_completion_calibration_gate="/frozen/gate.json",
            )
        )


def test_hierarchical_source_completion_contract_is_fixed_local_positive_v2() -> None:
    args = _source_completion_args(
        source_completion_calibration=(
            _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION
        ),
        source_completion_calibration_gate="/frozen/gate.json",
        source_completion_calibration_gate_sha256="c" * 64,
    )
    _validate_source_completion_unary_args(args)
    contract = _source_completion_unary_contract(args)
    assert contract is not None
    assert contract["calibration_reject_fallback"] == (
        "v2_local_majority_positive_completion"
    )
    local = contract["hierarchical_local_positive_contract"]
    assert local["proposal_region"] == "q>0.5"
    assert local["proposal_reliability"] == "c=2*q-1"
    assert local["nonproposal_policy"] == "c=0_no_negative_evidence"
    assert local["parameter_sweep"] is False


def test_hierarchical_reject_keeps_only_local_positive_completion() -> None:
    probability = torch.tensor([[0.2, 0.5, 0.6], [0.9, 0.1, 1.0]])
    entropy_reliability = torch.full_like(probability, 0.75)
    raw_positive = np.zeros((2, 3), dtype=bool)
    raw_negative = np.zeros((2, 3), dtype=bool)
    raw_positive[1, 1] = True
    raw_negative[1, 2] = True
    local_probability, local_reliability, evidence = (
        _apply_hierarchical_source_completion_trust(
            probability,
            entropy_reliability,
            accept_full_completion=False,
            raw_positive=raw_positive,
            raw_negative=raw_negative,
        )
    )
    torch.testing.assert_close(
        local_probability,
        torch.tensor([[0.5, 0.5, 1.0], [1.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        local_reliability,
        torch.tensor([[0.0, 0.0, 0.2], [0.8, 1.0, 1.0]]),
    )
    assert evidence["branch"] == "v2_local_majority_positive_completion"
    assert evidence["v1_global_gate_accepted"] is False


def test_hierarchical_accept_preserves_full_v1_tensors_exactly() -> None:
    probability = torch.tensor([[0.2, 0.8]])
    reliability = torch.tensor([[0.3, 0.7]])
    result_probability, result_reliability, evidence = (
        _apply_hierarchical_source_completion_trust(
            probability,
            reliability,
            accept_full_completion=True,
            raw_positive=np.zeros((1, 2), dtype=bool),
            raw_negative=np.zeros((1, 2), dtype=bool),
        )
    )
    assert result_probability is probability
    assert result_reliability is reliability
    assert evidence["branch"] == "v1_full_probability_preserving_completion"
    assert evidence["local_positive_contract"] is None


def test_source_completion_gate_assets_are_rejected_when_calibration_is_disabled():
    with pytest.raises(ValueError, match="calibration gate assets require"):
        _validate_source_completion_unary_args(
            _source_completion_args(
                source_completion_calibration_gate="/frozen/gate.json",
                source_completion_calibration_gate_sha256="c" * 64,
            )
        )


def test_probability_preserving_registration_uses_one_four_channel_adjoint_input():
    raw = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    probability = torch.tensor([[0.75, 0.25]])
    reliability = torch.tensor([[0.8, 0.4]])

    packed = _probability_preserving_registration_maps(
        raw, probability, reliability
    )

    assert packed.shape == (1, 4, 1, 2)
    torch.testing.assert_close(packed[:, :2], raw)
    torch.testing.assert_close(
        packed[0, 2], torch.tensor([[0.6, 0.1]])
    )
    torch.testing.assert_close(
        packed[0, 3], torch.tensor([[0.2, 0.3]])
    )
    torch.testing.assert_close(packed[0, 2] + packed[0, 3], reliability)


def test_probability_preserving_registration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        _probability_preserving_registration_maps(
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
        )
    invalid = torch.tensor([[0.0, 1.1]])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _probability_preserving_registration_maps(
            torch.zeros(1, 2, 1, 2), invalid, torch.ones(1, 2)
        )


@pytest.mark.parametrize(
    "updates, expected",
    [
        ({"disable_registered_graph": False}, "--disable-registered-graph"),
        ({"registered_readout_stage": "propagated"}, "unary_prior"),
        ({"registered_observation_fusion": "additive"}, "probability_mixture"),
        ({"prompt_registration_scale": 0.5}, "registration-scale 1"),
        ({"source_completion_sha256": "bad"}, "valid SHA256"),
    ],
)
def test_probability_preserving_source_unary_fails_closed(updates, expected):
    with pytest.raises(ValueError, match=expected):
        _validate_source_completion_unary_args(_source_completion_args(**updates))


def test_source_completion_assets_are_rejected_when_mode_is_disabled():
    args = _source_completion_args(source_completion_unary="none")
    with pytest.raises(ValueError, match="source completion assets require"):
        _validate_source_completion_unary_args(args)


def test_joint_signed_registered_seeds_leave_conflicting_mass_neutral() -> None:
    positive, negative = _joint_signed_observation_seeds(
        torch.tensor([0.7, 0.0, -0.5, 0.0]),
        torch.tensor([0.9, 0.8, 0.7, 0.0]),
        support_threshold=0.0,
    )

    torch.testing.assert_close(positive, torch.tensor([0.7, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(negative, torch.tensor([0.0, 0.0, 0.5, 0.0]))


def test_historical_registered_seed_construction_preserves_positive_tie() -> None:
    positive, negative = _registered_solver_masses(
        torch.tensor([0.4]),
        torch.tensor([0.4]),
        support_threshold=0.0,
        construction="winner_take_all",
    )

    torch.testing.assert_close(positive, torch.tensor([0.4]))
    torch.testing.assert_close(negative, torch.tensor([0.0]))


def test_capability_filter_must_preserve_both_prompt_signs() -> None:
    with pytest.raises(RuntimeError, match="Capability-valid.*neg=0"):
        _require_bipolar_solver_support(
            torch.tensor([0.5, 0.0]),
            torch.zeros(2),
            label="Capability-valid",
        )


def test_registered_stage_renderer_reuses_only_the_actual_final_stage() -> None:
    values = {
        "unary_prior": torch.tensor([1.0]),
        "propagated": torch.tensor([2.0]),
        "connected": torch.tensor([3.0]),
    }
    rendered = _render_registered_stage_maps(
        values,
        final_stage="propagated",
        final_rendered=np.array([20.0], dtype=np.float32),
        render=lambda tensor: np.array([float(tensor.item() * 10.0)]),
    )

    np.testing.assert_array_equal(rendered["unary_prior"], np.array([10.0]))
    np.testing.assert_array_equal(rendered["propagated"], np.array([20.0]))
    np.testing.assert_array_equal(rendered["connected"], np.array([30.0]))


def test_dataset_protocol_contract_excludes_method_score_semantics() -> None:
    manifest = {
        "benchmark": "nvos",
        "protocol": {
            "cohort": ["scene"],
            "dataset_version": "v1",
            "task": "segmentation",
            "prompt_type": "fixed_scribble",
            "prompt_support": "complete",
            "prompt_asset_sha256": {
                "scene": {"positive": "p", "negative": "n"}
            },
            "prediction_representation": "continuous_margin",
            "score_semantics": "cosine_margin",
            "threshold": {"value": 0.0},
        },
        "scenes": [
            {
                "scene_id": "scene",
                "prompt": {
                    "type": "positive_negative_scribbles",
                    "frame_id": "prompt",
                },
                "prompt_frame_ids": ["prompt"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "excluded_training_frame_ids": ["target"],
                "training_frames": [{"frame_id": "prompt"}],
                "target_rgb_policy": "forbidden",
                "frames": [
                    {
                        "frame_id": "target",
                        "ground_truth_sha256": "ground-truth",
                    }
                ],
            }
        ],
    }

    original = _dataset_protocol_contract(manifest)
    manifest["protocol"]["prediction_representation"] = "posterior"
    manifest["protocol"]["score_semantics"] = "foreground_probability"
    manifest["protocol"]["threshold"] = {"value": 0.5}

    assert _dataset_protocol_contract(manifest) == original
    manifest["protocol"]["prompt_asset_sha256"]["scene"]["positive"] = "changed"
    assert _dataset_protocol_contract(manifest) != original


def test_dataset_contract_reuses_frozen_reference_frame_hash(tmp_path) -> None:
    prompt_path = tmp_path / "reference.png"
    digest = "a" * 64
    manifest = {
        "benchmark": "spin",
        "protocol_hash": "legacy-frozen-hash",
        "protocol": {"cohort": ["scene"]},
        "scenes": [
            {
                "scene_id": "scene",
                "prompt": {
                    "type": "reference_binary_mask",
                    "frame_id": "reference",
                    "mask_path": str(prompt_path),
                },
                "prompt_frame_ids": ["reference"],
                "calibration_frame_ids": [],
                "evaluation_frame_ids": ["target"],
                "excluded_training_frame_ids": [],
                "training_frames": [],
                "frames": [
                    {
                        "frame_id": "reference",
                        "ground_truth": str(prompt_path),
                        "ground_truth_sha256": digest,
                    },
                    {
                        "frame_id": "target",
                        "ground_truth_sha256": "b" * 64,
                    },
                ],
            }
        ],
    }

    contract = _dataset_protocol_contract(manifest)

    assert contract["legacy_protocol_hash"] == "legacy-frozen-hash"
    assert contract["scenes"][0]["prompt"]["asset_sha256"] == {
        "reference_binary_mask": digest
    }


def test_dataset_contract_does_not_reuse_hash_for_a_different_prompt_path(tmp_path) -> None:
    manifest = {
        "benchmark": "spin",
        "protocol": {"cohort": ["scene"]},
        "scenes": [
            {
                "scene_id": "scene",
                "prompt": {
                    "type": "reference_binary_mask",
                    "frame_id": "reference",
                    "mask_path": str(tmp_path / "different.png"),
                },
                "evaluation_frame_ids": [],
                "frames": [
                    {
                        "frame_id": "reference",
                        "ground_truth": str(tmp_path / "reference.png"),
                        "ground_truth_sha256": "a" * 64,
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="prompt asset hashes are undeclared"):
        _dataset_protocol_contract(manifest)


def test_valid_normalized_score_map_uses_only_supported_compositing_mass() -> None:
    rendered = torch.tensor(
        [
            [[0.20, 0.00], [0.45, 0.10]],
            [[0.25, 0.00], [0.50, 0.20]],
        ]
    )
    actual = _valid_normalized_score_map(rendered)
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.8, 0.0], [0.9, 0.5]]),
    )


def test_valid_normalized_score_map_interpolates_to_total_alpha_score() -> None:
    rendered = torch.tensor(
        [
            [[0.20, 0.45]],
            [[0.25, 0.50]],
        ]
    )

    total_alpha = _valid_normalized_score_map(rendered, coverage_power=1.0)

    torch.testing.assert_close(total_alpha, rendered[0])


def test_sparse_prototypes_match_prefiltered_reference_for_half_bank() -> None:
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=torch.float16,
        ).float(),
        dim=-1,
    ).half()
    weights = torch.tensor([0.0, 0.7, 0.0, 0.2, 0.0])

    actual_features, actual_masses = _deterministic_prototypes(
        features, weights, count=2, chunk_size=1
    )
    active = weights > 0
    expected_features, expected_masses = _deterministic_prototypes(
        features[active].float(), weights[active], count=2
    )

    torch.testing.assert_close(actual_features, expected_features)
    torch.testing.assert_close(actual_masses, expected_masses)


def test_spherical_mean_fps_anchors_with_weighted_mean() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    weights = torch.tensor([0.5, 0.4, 0.1])
    prototypes, masses = _deterministic_prototypes(
        features,
        weights,
        count=2,
        chunk_size=1,
        strategy="spherical_mean_fps",
    )
    expected_mean = F.normalize(
        (F.normalize(features, dim=-1) * weights[:, None]).sum(dim=0), dim=0
    )
    torch.testing.assert_close(prototypes[0], expected_mean)
    torch.testing.assert_close(masses.sum(), torch.tensor(1.0))
