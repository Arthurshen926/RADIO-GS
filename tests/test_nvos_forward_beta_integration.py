from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import pytest
import torch

from radio_gs.scripts import eval_nvos_gaussian_first as nvos_evaluator
from radio_gs.scripts import nvos_forward_beta_scene_authority as scene_authority
from radio_gs.field import FeatureSpaceSignature
from radio_gs.querying.evidence_scorer import (
    EvidenceScoringConfig,
    fuse_registered_observation_unary,
    registered_forward_beta_balanced_residual_observation,
)
from radio_gs.querying.query_compilers import compile_registered_primitive_seeds
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_spec import PrimitiveUnaryEvidence
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _candidate_method_manifest_contract,
    _center_registered_forward_score_map,
    _compact_registered_forward_beta_diagnostics,
    _execute_registered_forward_beta,
    _json_sha256,
    _load_registered_forward_protocol_authority,
    _registered_forward_scoring_contract,
    _registered_forward_unary_contract,
    _resize_nvos_score_for_evaluation,
    _validate_registered_forward_unary_args,
    run as run_nvos,
)
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    build_authority,
)


def _legacy_contract_args(**overrides) -> argparse.Namespace:
    values = {
        "support_mode": "canonical_support",
        "region_space": "sam3",
        "prompt_registration_mode": "raster_adjoint",
        "prompt_registration_scale": 1.0,
        "alpha_threshold": 0.0,
        "depth_tolerance": 0.08,
        "relative_depth_tolerance": 0.02,
        "registered_seed_construction": "joint_signed",
        "registered_observation_fusion": "probability_mixture",
        "registered_seed_unary_weight": 0.0,
        "registered_observation_confidence": "relative_joint_max",
        "registered_observation_mass_scale": 1.0,
        "support_threshold": 0.0,
        "prototype_count": 1,
        "prototype_strategy": "spherical_mean_fps",
        "appearance_weight": 1.0,
        "boundary_weight": 0.35,
        "prototype_temperature": 0.07,
        "feature_calibration": "none",
        "background_centroids": 0,
        "score_calibration": "none",
        "negative_spatial_mode": "none",
        "registered_selection_mode": "seeded_component",
        "registered_readout_stage": "propagated",
        "graph_policy": "legacy",
        "component_graph_policy": "same",
        "graph_legacy_residual": 0.0,
        "channel_confidence_mode": "none",
        "score_render_resolution": "scaled_renderer",
        "score_render_scale": 1.0,
        "valid_support_normalization": False,
        "valid_support_coverage_power": 0.0,
        "feature_contribution_gamma": 1.0,
        "score_chunk_size": 1024,
        "solver_support_threshold": 0.5,
        "solver_type": "diffusion",
        "solver_iterations": 2,
        "solver_residual": 0.3,
        "solver_unary_temperature": 0.5,
        "laplacian_weight": 1.0,
        "cg_iterations": 8,
        "cg_tolerance": 1e-5,
        "hard_seed_threshold": 0.2,
        "hard_seed_conflict_policy": "positive_priority",
        "hard_seed_conflict_margin": 0.0,
        "component_edge_threshold": 1e-5,
        "seeded_component_min_weight": 0.2,
        "canonical_reliability_cache": "",
        "diagnostic_graph_affinity_override": "",
        "require_asset_hashes": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _signature(name: str) -> FeatureSpaceSignature:
    return FeatureSpaceSignature(
        radio_version="c-radio_v4-h",
        radio_checkpoint_sha256="radio-hash",
        raw_feature_dim=1280,
        adaptor_name=name,
        adaptor_sha256=f"{name}-hash",
        adaptor_output_dim=2,
        token_type="primitive",
        field_checkpoint_sha256="field-hash",
    )


def _graph() -> PrimitiveSupportGraph:
    return PrimitiveSupportGraph(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_weight=torch.tensor([0.6, 0.6, 0.4, 0.4]),
        raw_affinity=torch.tensor([0.8, 0.8, 0.5, 0.5]),
        local_sigma=torch.ones(3),
        num_nodes=3,
    )


def _authority_manifest(method_sha: str = "a" * 64) -> dict[str, object]:
    authority = build_authority(
        candidate_method_sha256=method_sha,
        scoring_contract={
            "score_semantics": "beta_centered_posterior",
            "prediction_representation": "continuous_beta_centered_posterior",
            "threshold": {"comparison": "greater_or_equal", "value": 0.0},
            "resize": "nearest",
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    return {
        "registered_forward_protocol_authority": authority,
        "registered_forward_protocol_authority_sha256": _json_sha256(authority),
    }


def test_none_is_bit_compatible_with_missing_legacy_switch() -> None:
    absent = _legacy_contract_args()
    explicit_none = _legacy_contract_args(registered_forward_unary="none")

    absent_contract = _candidate_method_manifest_contract(absent)
    explicit_contract = _candidate_method_manifest_contract(explicit_none)

    assert absent_contract == explicit_contract
    assert "registered_forward_unary" not in absent_contract
    assert _registered_forward_unary_contract(absent) is None
    assert _registered_forward_scoring_contract(absent) is None
    assert _load_registered_forward_protocol_authority(absent, None, "") is None


def test_beta_contract_is_authority_bound_non_exact_and_compact() -> None:
    args = _legacy_contract_args(registered_forward_unary="beta_coverage_v1")
    contract = _registered_forward_unary_contract(args)

    assert contract is not None
    assert contract["status"] == "protocol_authority_bound_non_exact_diagnostic"
    assert contract["strict_unseen_eligible"] is False
    assert contract["selection_applied_to_main_output"] is False
    assert contract["required_final_readout"] == "propagated"
    assert contract["prompt_registration_scale"] == 1.0
    assert contract["compositor_alpha_threshold"] == 0.0
    assert contract["scoring_adapter"] == {
        "score_semantics": "beta_centered_posterior",
        "prediction_representation": "continuous_beta_centered_posterior",
        "threshold": {"comparison": "greater_or_equal", "value": 0.0},
        "resize": "nearest",
    }
    assert len(json.dumps(contract)) < 4000


def test_beta_v2_contract_records_balancing_precision_residual_and_anchor() -> None:
    args = _legacy_contract_args(
        registered_forward_unary="beta_balanced_residual_v2",
        canonical_reliability_cache="canonical_reliability.pt",
    )

    contract = _registered_forward_unary_contract(args)

    assert contract is not None
    assert contract["mode"] == "beta_balanced_residual_v2"
    assert contract["class_balance"]["scope"] == "global_expected_counts"
    assert contract["class_balance"]["class_prior_from_scribble_area"] is False
    assert contract["field_prior_concentration_bounds"] == {
        "minimum": 1.0,
        "maximum": 2.0,
    }
    assert contract["semantic_precision_is_primary_for_nonanchors"] is True
    assert contract["anchor"]["threshold_source"] == (
        "solver.hard_seed_threshold"
    )
    assert contract["uses_target_calibration"] is False
    assert contract["uses_scene_id_branching"] is False
    assert contract["scoring_adapter"] == _registered_forward_scoring_contract(args)
    assert len(json.dumps(contract)) < 6000

    method = _candidate_method_manifest_contract(args)
    assert method["canonical_reliability_cache"] == (
        "per_scene_source_artifact:canonical_primitive_reliability_v1.pt"
    )


def test_beta_v2_configuration_requires_reliability_and_joint_signed_seeds() -> None:
    missing_reliability = _legacy_contract_args(
        registered_forward_unary="beta_balanced_residual_v2",
        canonical_reliability_cache="",
    )
    wrong_seed_construction = _legacy_contract_args(
        registered_forward_unary="beta_balanced_residual_v2",
        canonical_reliability_cache="canonical_reliability.pt",
        registered_seed_construction="winner_take_all",
    )

    with pytest.raises(ValueError, match="canonical-reliability-cache"):
        _validate_registered_forward_unary_args(missing_reliability)
    with pytest.raises(ValueError, match="joint_signed"):
        _validate_registered_forward_unary_args(wrong_seed_construction)


def test_beta_v2_compact_diagnostics_expose_balancing_precision_and_anchors() -> None:
    _, diagnostics = registered_forward_beta_balanced_residual_observation(
        gaussian_ids=torch.tensor([0, 1, 1]),
        pixel_ids=torch.tensor([0, 1, 2]),
        contribution_weights=torch.tensor([0.8, 0.8, 0.8]),
        capability_valid=torch.tensor([True, True]),
        field_prior=torch.tensor([0.6, 0.4]),
        primitive_reliability=torch.tensor([0.8, 0.5]),
        primitive_coverage=torch.tensor([0.75, 0.5]),
        positive_pixel_mask=torch.tensor([True, False, False]),
        negative_pixel_mask=torch.tensor([False, True, True]),
        labeled_pixel_mask=torch.tensor([True, True, True]),
        all_pixel_mask=torch.tensor([True, True, True]),
        anchor_threshold=0.2,
    )

    compact = _compact_registered_forward_beta_diagnostics(
        diagnostics, torch.tensor([True, True])
    )

    assert compact["v2"]["class_balance"]["balanced_positive_sum"] == (
        pytest.approx(compact["v2"]["class_balance"]["balanced_negative_sum"])
    )
    assert compact["v2"]["anchors"] == {
        "positive": 1,
        "negative": 1,
        "conflicting": 0,
    }
    assert compact["v2"]["distributions"][
        "field_prior_concentration_valid"
    ]["max"] <= 2.0
    assert compact["vectors_persisted"] is False
    assert compact["v2"]["vectors_persisted"] is False


def test_beta_candidate_contract_records_zero_threshold_and_nearest() -> None:
    args = _legacy_contract_args(registered_forward_unary="beta_coverage_v1")

    contract = _candidate_method_manifest_contract(args)

    assert contract["score_render"]["pixel_threshold"] == 0.0
    assert contract["score_render"]["resize_to_ground_truth"] == (
        "cv2.INTER_NEAREST"
    )


def test_centered_posterior_adapter_keeps_zero_tie_positive() -> None:
    posterior = torch.tensor([[0.0, 0.25], [0.5, 1.0]]).numpy()

    centered = _center_registered_forward_score_map(posterior)

    assert centered.dtype.name == "float32"
    torch.testing.assert_close(
        torch.from_numpy(centered),
        torch.tensor([[-1.0, -0.5], [0.0, 1.0]]),
    )
    assert (centered >= 0.0).tolist() == [[False, False], [True, True]]


def test_beta_resize_is_nearest_while_legacy_remains_linear() -> None:
    score = torch.tensor([[-1.0, 1.0], [1.0, -1.0]]).numpy()

    beta = _resize_nvos_score_for_evaluation(
        score,
        (4, 4),
        registered_forward_unary="beta_coverage_v1",
    )
    legacy = _resize_nvos_score_for_evaluation(
        score,
        (4, 4),
        registered_forward_unary="none",
    )
    legacy_reference = cv2.resize(
        score,
        (4, 4),
        interpolation=cv2.INTER_LINEAR,
    )

    torch.testing.assert_close(
        torch.from_numpy(beta),
        torch.tensor(
            [
                [-1.0, -1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
            ]
        ),
    )
    assert torch.equal(
        torch.from_numpy(legacy),
        torch.from_numpy(legacy_reference),
    )
    assert not torch.equal(torch.from_numpy(beta), torch.from_numpy(legacy))


def test_beta_authority_requires_method_sha_and_derives_non_exactness() -> None:
    args = _legacy_contract_args(registered_forward_unary="beta_coverage_v1")
    with pytest.raises(ValueError, match="candidate method contract SHA256"):
        _load_registered_forward_protocol_authority(args, None, "")

    authority = _load_registered_forward_protocol_authority(
        args,
        _authority_manifest(),
        "a" * 64,
    )

    assert authority is not None
    assert authority["candidate"]["method_contract_sha256"] == "a" * 64
    assert authority["scoring_contract"] == _registered_forward_scoring_contract(
        args
    )
    assert authority["strict_unseen_protocol_exact_match"] is False
    assert authority["strict_unseen_exact_match_blockers"] == [
        "score_semantics_differs",
        "prediction_representation_differs",
    ]


def test_inline_authority_validation_does_not_open_paper_runtime(
    monkeypatch,
) -> None:
    args = _legacy_contract_args(registered_forward_unary="beta_coverage_v1")
    manifest = _authority_manifest()
    original_open = Path.open

    def reject_paper_open(path, *open_args, **open_kwargs):
        if "paper" in Path(path).parts:
            raise AssertionError("snapshot runtime must not open paper authority")
        return original_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr(Path, "open", reject_paper_open)

    authority = _load_registered_forward_protocol_authority(
        args,
        manifest,
        "a" * 64,
    )

    assert authority is not None
    assert authority["strict_unseen_protocol_exact_match"] is False


@pytest.mark.parametrize("tamper", ["digest", "scoring", "candidate", "extra"])
def test_inline_authority_tampering_fails_closed(tamper: str) -> None:
    args = _legacy_contract_args(registered_forward_unary="beta_coverage_v1")
    manifest = _authority_manifest()
    authority = deepcopy(manifest["registered_forward_protocol_authority"])
    if tamper == "digest":
        manifest["registered_forward_protocol_authority_sha256"] = "b" * 64
    elif tamper == "scoring":
        authority["scoring_contract"]["score_semantics"] = (
            "continuous_cosine_margin"
        )
        manifest["registered_forward_protocol_authority"] = authority
        manifest["registered_forward_protocol_authority_sha256"] = (
            _json_sha256(authority)
        )
    elif tamper == "candidate":
        authority["candidate"]["method_contract_sha256"] = "b" * 64
        manifest["registered_forward_protocol_authority"] = authority
        manifest["registered_forward_protocol_authority_sha256"] = (
            _json_sha256(authority)
        )
    else:
        authority["caller_exact_override"] = True
        manifest["registered_forward_protocol_authority"] = authority
        manifest["registered_forward_protocol_authority_sha256"] = (
            _json_sha256(authority)
        )

    with pytest.raises(ValueError):
        _load_registered_forward_protocol_authority(
            args,
            manifest,
            "a" * 64,
        )


def test_beta_run_without_candidate_manifest_fails_before_model_load(
    tmp_path,
) -> None:
    manifest = tmp_path / "benchmark.json"
    manifest.write_text("{}\n", encoding="utf-8")
    args = _legacy_contract_args(
        registered_forward_unary="beta_coverage_v1",
        manifest=str(manifest),
        run_manifest="",
        scene_id="synthetic",
        device="cpu",
    )

    with pytest.raises(ValueError, match="candidate method contract SHA256"):
        run_nvos(args)


@pytest.mark.parametrize("physical_index", [0, 1])
def test_beta_cli_dispatches_dynamic_physical_gpu_attestation(
    monkeypatch,
    physical_index: int,
) -> None:
    calls = []
    monkeypatch.setattr(
        scene_authority,
        "write_forward_beta_cuda_child_attestation",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        nvos_evaluator,
        "write_cuda_child_attestation",
        lambda **_kwargs: pytest.fail("beta must not use the v3 GPU1 writer"),
    )
    monkeypatch.setattr(nvos_evaluator, "run", lambda _args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_nvos_gaussian_first.py",
            "--manifest", "unused.json",
            "--queue-root", "unused",
            "--scene-id", "fern",
            "--output-dir", "unused",
            "--support-mode", "canonical_support",
            "--prompt-registration-mode", "raster_adjoint",
            "--alpha-threshold", "0",
            "--registered-observation-fusion", "probability_mixture",
            "--registered-readout-stage", "propagated",
            "--registered-forward-unary", "beta_coverage_v1",
            "--gpu-attestation-output", "attestation.json",
            "--expected-gpu-physical-index", str(physical_index),
            "--expected-gpu-uuid", (
                "GPU-11111111-1111-1111-1111-111111111111"
            ),
            "--expected-gpu-bus-id", "00000000:01:00.0",
        ],
    )

    nvos_evaluator.main()

    assert calls == [
        {
            "output": "attestation.json",
            "scene": "fern",
            "physical_index": physical_index,
            "expected_uuid": "GPU-11111111-1111-1111-1111-111111111111",
            "expected_bus_id": "00000000:01:00.0",
        }
    ]


def test_legacy_cli_keeps_original_gpu1_attestation_writer(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        nvos_evaluator,
        "write_cuda_child_attestation",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        scene_authority,
        "write_forward_beta_cuda_child_attestation",
        lambda **_kwargs: pytest.fail("legacy must not use beta writer"),
    )
    monkeypatch.setattr(nvos_evaluator, "run", lambda _args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_nvos_gaussian_first.py",
            "--manifest", "unused.json",
            "--queue-root", "unused",
            "--scene-id", "fern",
            "--output-dir", "unused",
            "--gpu-attestation-output", "attestation.json",
            "--expected-gpu-uuid", "GPU-legacy",
            "--expected-gpu-bus-id", "00000000:82:00.0",
        ],
    )

    nvos_evaluator.main()

    assert calls == [
        {
            "output": "attestation.json",
            "scene": "fern",
            "expected_uuid": "GPU-legacy",
            "expected_bus_id": "00000000:82:00.0",
        }
    ]


@pytest.mark.parametrize(
    ("override", "required_fragment"),
    [
        ({"support_mode": "prompt_gaussian"}, "canonical_support"),
        ({"registered_observation_fusion": "additive"}, "probability_mixture"),
        ({"registered_seed_unary_weight": 0.1}, "unary-weight 0"),
        ({"registered_readout_stage": "connected"}, "propagated"),
        ({"prompt_registration_mode": "legacy_alpha_depth"}, "raster_adjoint"),
        ({"prompt_registration_scale": 0.5}, "registration-scale 1"),
        ({"alpha_threshold": 0.02}, "alpha-threshold 0"),
        ({"feature_contribution_gamma": 2.0}, "gamma 1"),
    ],
)
def test_beta_configuration_fails_closed(override, required_fragment) -> None:
    args = _legacy_contract_args(
        registered_forward_unary="beta_coverage_v1",
        **override,
    )

    with pytest.raises(ValueError, match=required_fragment):
        _validate_registered_forward_unary_args(args)


def test_cpu_two_pass_preserves_unobserved_rows_and_graph_topology() -> None:
    graph = _graph()
    graph_snapshot = {
        "edge_index": graph.edge_index.clone(),
        "edge_weight": graph.edge_weight.clone(),
        "raw_affinity": graph.raw_affinity.clone(),
        "local_sigma": graph.local_sigma.clone(),
    }
    appearance_signature = _signature("appearance")
    boundary_signature = _signature("boundary")
    features = {
        "appearance": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
        "boundary": torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]]),
    }
    signatures = {
        "appearance": appearance_signature,
        "boundary": boundary_signature,
    }
    historical_observation = PrimitiveUnaryEvidence(
        torch.tensor([0.9, -0.8, 0.7]),
        "historical_direct_observation",
        confidence=torch.tensor([0.9, 0.8, 0.7]),
    )
    query = compile_registered_primitive_seeds(
        torch.tensor([0.8, 0.0, 0.0]),
        torch.tensor([0.0, 0.8, 0.0]),
        appearance_features=features["appearance"],
        boundary_features=features["boundary"],
        appearance_signature=appearance_signature,
        boundary_signature=boundary_signature,
        primitive_unary_evidence=historical_observation,
        seed_normalization="none",
    )
    base_engine = CanonicalQueryEngine(
        graph,
        scoring_config=EvidenceScoringConfig(
            appearance_weight=1.0,
            boundary_weight=0.35,
            registered_observation_fusion="probability_mixture",
        ),
        solver_config=SupportSolverConfig(
            iterations=2,
            residual=0.3,
            unary_temperature=0.5,
            support_threshold=0.5,
        ),
        node_reliability=torch.tensor([0.8, 0.7, 0.6]),
    )

    class RecordingEngine:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.queries = []

        def execute(self, query_value, banks, *, feature_signatures=None):
            self.queries.append(query_value)
            return self.wrapped.execute(
                query_value,
                banks,
                feature_signatures=feature_signatures,
            )

    engine = RecordingEngine(base_engine)
    result, field_result, observation, diagnostics = (
        _execute_registered_forward_beta(
            engine,
            query,
            features,
            signatures,
            gaussian_ids=torch.tensor([0, 1, 2, 1, 2, 3]),
            pixel_ids=torch.tensor([0, 0, 0, 1, 1, 2]),
            contribution_weights=torch.tensor([0.5, 0.3, 0.1, 0.6, 0.2, 0.7]),
            capability_valid=torch.tensor([True, True, False, True]),
            valid_rows=torch.tensor([0, 1, 3]),
            positive_pixels=torch.tensor([True, False, False]),
            negative_pixels=torch.tensor([False, True, False]),
            unary_temperature=0.5,
        )
    )

    assert len(engine.queries) == 2
    assert engine.queries[0].primitive_unary_evidence is None
    assert engine.queries[1].primitive_unary_evidence is not None
    assert engine.queries[1].primitive_unary_evidence.source == (
        "forward_likelihood_beta_coverage_v1"
    )
    torch.testing.assert_close(
        engine.queries[0].positive_seeds.weights,
        query.positive_seeds.weights,
    )
    torch.testing.assert_close(
        engine.queries[1].negative_seeds.weights,
        query.negative_seeds.weights,
    )

    assert observation.confidence[3].item() == 0.0
    assert diagnostics.labeled_expected_count[3].item() == 0.0
    assert result.unary[2].item() == field_result.unary[2].item()
    expected = fuse_registered_observation_unary(
        field_result.unary,
        PrimitiveUnaryEvidence(
            observation.values[torch.tensor([0, 1, 3])],
            observation.source,
            observation.confidence[torch.tensor([0, 1, 3])],
        ),
        unary_temperature=0.5,
    )
    torch.testing.assert_close(result.unary, expected)

    for name, before in graph_snapshot.items():
        torch.testing.assert_close(getattr(graph, name), before)

    compact = _compact_registered_forward_beta_diagnostics(
        diagnostics,
        torch.tensor([True, True, False, True]),
    )
    assert compact["vectors_persisted"] is False
    assert compact["row_counts"]["observed_valid"] == 2
    assert len(json.dumps(compact)) < 8000

    def assert_scalar_tree(value) -> None:
        assert not isinstance(value, torch.Tensor)
        assert not isinstance(value, (list, tuple))
        if isinstance(value, dict):
            for child in value.values():
                assert_scalar_tree(child)

    assert_scalar_tree(compact)


def test_cpu_v2_two_pass_promotes_direct_evidence_to_solver_anchors() -> None:
    appearance_signature = _signature("appearance")
    boundary_signature = _signature("boundary")
    features = {
        "appearance": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
        "boundary": torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]]),
    }
    signatures = {
        "appearance": appearance_signature,
        "boundary": boundary_signature,
    }
    query = compile_registered_primitive_seeds(
        torch.tensor([0.1, 0.0, 0.0]),
        torch.tensor([0.0, 0.1, 0.0]),
        appearance_features=features["appearance"],
        boundary_features=features["boundary"],
        appearance_signature=appearance_signature,
        boundary_signature=boundary_signature,
        seed_normalization="none",
    )
    base_engine = CanonicalQueryEngine(
        _graph(),
        scoring_config=EvidenceScoringConfig(
            appearance_weight=1.0,
            boundary_weight=0.35,
            registered_observation_fusion="probability_mixture",
        ),
        solver_config=SupportSolverConfig(
            iterations=2,
            residual=0.3,
            unary_temperature=0.5,
            support_threshold=0.5,
            hard_seed_threshold=0.2,
        ),
    )

    class RecordingEngine:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.queries = []

        def execute(self, query_value, banks, *, feature_signatures=None):
            self.queries.append(query_value)
            return self.wrapped.execute(
                query_value,
                banks,
                feature_signatures=feature_signatures,
            )

    engine = RecordingEngine(base_engine)
    result, _, observation, diagnostics = _execute_registered_forward_beta(
        engine,
        query,
        features,
        signatures,
        gaussian_ids=torch.tensor([0, 1]),
        pixel_ids=torch.tensor([0, 1]),
        contribution_weights=torch.tensor([0.8, 0.8]),
        capability_valid=torch.tensor([True, True, False, True]),
        valid_rows=torch.tensor([0, 1, 3]),
        positive_pixels=torch.tensor([True, False]),
        negative_pixels=torch.tensor([False, True]),
        unary_temperature=0.5,
        mode="beta_balanced_residual_v2",
        primitive_reliability=torch.tensor([0.8, 0.7, 0.0, 0.6]),
        primitive_coverage=torch.tensor([0.9, 0.8, 0.0, 0.7]),
        anchor_threshold=0.2,
    )

    assert len(engine.queries) == 2
    assert observation.source == "forward_likelihood_beta_balanced_residual_v2"
    assert diagnostics.positive_anchor_mask.tolist() == [True, False, False, False]
    assert diagnostics.negative_anchor_mask.tolist() == [False, True, False, False]
    torch.testing.assert_close(
        engine.queries[0].positive_seeds.weights,
        torch.tensor([0.1, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        engine.queries[1].positive_seeds.weights,
        torch.tensor([1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        engine.queries[1].negative_seeds.weights,
        torch.tensor([0.0, 1.0, 0.0]),
    )
    # A unit seed remains exact after the graph solver, so propagation cannot
    # erase the accepted direct positive/negative observations.
    assert result.probabilities[0].item() == pytest.approx(1.0)
    assert result.probabilities[1].item() == pytest.approx(0.0)
