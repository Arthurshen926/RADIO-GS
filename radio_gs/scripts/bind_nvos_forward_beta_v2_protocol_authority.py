#!/usr/bin/env python3
"""Bind the Beta-v2 candidate and reliability cohort to protocol authority.

The published authority intentionally keeps the evaluator's existing, exact
protocol-authority schema.  The query-independent reliability manifest is a
mandatory precondition here and is bound separately (path, bytes and SHA-256)
by the v2 snapshot and run manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from radio_gs.scripts.bind_evaluation_protocol_freeze import (
    BindingError,
    write_binding_receipt,
)
from radio_gs.scripts.bind_nvos_beta_v2_reliability_manifest import (
    ARTIFACT_TYPE as RELIABILITY_ARTIFACT_TYPE,
    ORDERED_SCENES,
    validate_manifest,
)
from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    AuthorityError,
    build_authority,
    canonical_json_sha256,
    validate_authority_payload,
    _read_stable_bytes,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _candidate_method_manifest_contract,
)


CANDIDATE_ID = "nvos-forward-beta-balanced-residual-v2"
FORWARD_MODE = "beta_balanced_residual_v2"
RELIABILITY_MARKER = (
    "per_scene_source_artifact:canonical_primitive_reliability_v1.pt"
)
DEFAULT_CANDIDATE = Path(
    "paper/artifacts/nvos_forward_beta_balanced_residual_v2_candidate_20260802.yaml"
)
DEFAULT_RELIABILITY_MANIFEST = Path(
    "paper/artifacts/nvos_forward_beta_balanced_residual_v2_reliability_manifest_20260802.json"
)
EXPECTED_METHOD_NAMESPACE: dict[str, Any] = {
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
    "registered_observation_confidence": "poisson_mass_coverage",
    "registered_observation_mass_scale": 1.0,
    "registered_observation_coverage_power": 1.0,
    "support_threshold": 0.0,
    "prototype_count": 4,
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
    "registered_forward_unary": FORWARD_MODE,
    "graph_policy": "legacy",
    "component_graph_policy": "same",
    "graph_legacy_residual": 0.0,
    "channel_confidence_mode": "none",
    "score_render_resolution": "prompt_native",
    "score_render_scale": 1.0,
    "valid_support_normalization": True,
    "valid_support_coverage_power": 1.0,
    "feature_contribution_gamma": 1.0,
    "score_chunk_size": 8192,
    "solver_type": "confidence_random_walker",
    "solver_iterations": 12,
    "solver_residual": 0.30,
    "solver_unary_temperature": 0.10,
    "solver_support_threshold": 0.50,
    "laplacian_weight": 1.0,
    "cg_iterations": 64,
    "cg_tolerance": 1e-5,
    "hard_seed_threshold": 0.20,
    "hard_seed_conflict_policy": "exclusive_relative",
    "hard_seed_conflict_margin": 0.0,
    "component_edge_threshold": 1e-5,
    "seeded_component_min_weight": 0.20,
    "canonical_reliability_cache": RELIABILITY_MARKER,
    "diagnostic_graph_affinity_override": "",
    "require_asset_hashes": True,
}


class BetaV2AuthorityError(ValueError):
    """Raised when the Beta-v2 candidate or reliability binding drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BetaV2AuthorityError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BetaV2AuthorityError(f"{label} must be a mapping")
    return value


def _stable(path: str | Path, *, label: str) -> tuple[Path, bytes]:
    lexical = Path(path).absolute()
    try:
        encoded = _read_stable_bytes(lexical, label=label)
    except AuthorityError as error:
        raise BetaV2AuthorityError(str(error)) from error
    return lexical, encoded


def file_record(path: str | Path, *, label: str) -> dict[str, object]:
    lexical, encoded = _stable(path, label=label)
    return {
        "path": str(lexical),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def load_candidate_contract(
    path: str | Path,
) -> tuple[Mapping[str, Any], dict[str, Any], str, bytes]:
    """Load the declarative namespace and rederive the complete method contract."""

    _, encoded = _stable(path, label="Beta-v2 candidate YAML")
    try:
        payload = yaml.safe_load(encoded.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise BetaV2AuthorityError("Beta-v2 candidate is not UTF-8 YAML") from error
    candidate = _mapping(payload, "Beta-v2 candidate")
    _require(candidate.get("schema_version") == 1, "candidate schema differs")
    _require(candidate.get("candidate_id") == CANDIDATE_ID, "candidate id differs")
    _require(
        candidate.get("status")
        == "protocol_authority_bound_non_exact_diagnostic",
        "candidate status differs",
    )
    _require(candidate.get("promoted") is False, "candidate cannot be promoted")
    _require(
        candidate.get("v1_result_or_receipt_reuse_permitted") is False,
        "candidate must forbid v1 result or receipt reuse",
    )
    cohort = _mapping(candidate.get("cohort"), "candidate cohort")
    _require(
        cohort.get("ordered_tasks") == list(ORDERED_SCENES),
        "candidate cohort differs",
    )
    eligibility = _mapping(candidate.get("eligibility"), "candidate eligibility")
    _require(
        eligibility.get("strict_unseen_protocol_exact_match") is False
        and eligibility.get("frozen_diagnostic_eligible") is False
        and eligibility.get("main_result_eligible") is False
        and eligibility.get("target_metric_may_select_method_stage_or_continuation")
        is False,
        "candidate eligibility differs",
    )
    execution = _mapping(candidate.get("execution_contract"), "execution contract")
    _require(
        execution.get("v1_result_or_receipt_reuse_permitted") is False,
        "execution contract must forbid v1 result or receipt reuse",
    )
    namespace = dict(_mapping(candidate.get("method_namespace"), "method namespace"))
    _require(
        namespace == EXPECTED_METHOD_NAMESPACE,
        "candidate method namespace differs from the frozen v2 invocation",
    )
    _require(
        namespace.get("registered_forward_unary") == FORWARD_MODE,
        "candidate forward mode differs",
    )
    _require(
        namespace.get("canonical_reliability_cache") == RELIABILITY_MARKER,
        "candidate reliability marker differs",
    )
    try:
        method = _candidate_method_manifest_contract(argparse.Namespace(**namespace))
    except (AttributeError, TypeError, ValueError) as error:
        raise BetaV2AuthorityError(
            "candidate method namespace cannot derive evaluator contract"
        ) from error
    forward = _mapping(method.get("registered_forward_unary"), "forward contract")
    _require(forward.get("mode") == FORWARD_MODE, "derived forward mode differs")
    _require(
        method.get("canonical_reliability_cache") == RELIABILITY_MARKER,
        "derived reliability marker differs",
    )
    _require(
        forward.get("semantic_precision_is_primary_for_nonanchors") is True,
        "derived semantic precision contract differs",
    )
    scoring = _mapping(forward.get("scoring_adapter"), "scoring adapter")
    _require(
        scoring
        == {
            "score_semantics": "beta_centered_posterior",
            "prediction_representation": "continuous_beta_centered_posterior",
            "threshold": {"comparison": "greater_or_equal", "value": 0.0},
            "resize": "nearest",
        },
        "derived scoring adapter differs",
    )
    return candidate, method, canonical_json_sha256(method), encoded


def validate_reliability_binding(path: str | Path) -> tuple[dict[str, Any], dict[str, object]]:
    """Fully validate the immutable per-scene reliability cohort."""

    try:
        payload = validate_manifest(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise BetaV2AuthorityError(str(error)) from error
    _require(
        payload.get("artifact_type") == RELIABILITY_ARTIFACT_TYPE,
        "reliability manifest type differs",
    )
    _require(
        payload.get("ordered_scenes") == list(ORDERED_SCENES),
        "reliability manifest cohort differs",
    )
    safety = _mapping(payload.get("safety_contract"), "reliability safety")
    _require(
        safety.get("query_independent") is True
        and all(
            safety.get(key) is False
            for key in (
                "uses_query",
                "uses_text",
                "uses_target_labels",
                "uses_target_masks",
                "uses_metric_feedback",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        ),
        "reliability manifest is not query-independent",
    )
    return payload, file_record(path, label="Beta-v2 reliability manifest")


def build_v2_authority(
    *,
    repo_root: str | Path,
    candidate: str | Path,
    reliability_manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any], str]:
    """Validate both v2 inputs and build the unchanged evaluator authority schema."""

    _, method, method_sha256, _ = load_candidate_contract(candidate)
    _, reliability_record = validate_reliability_binding(reliability_manifest)
    forward = _mapping(method.get("registered_forward_unary"), "forward contract")
    scoring = _mapping(forward.get("scoring_adapter"), "scoring adapter")
    try:
        authority = build_authority(
            candidate_method_sha256=method_sha256,
            scoring_contract=scoring,
            repo_root=repo_root,
        )
        validate_authority_payload(authority)
    except AuthorityError as error:
        raise BetaV2AuthorityError(str(error)) from error
    _require(
        authority.get("strict_unseen_protocol_exact_match") is False,
        "Beta-v2 authority must remain non-exact",
    )
    _require(
        _mapping(
            authority.get("external_comparator_provenance"),
            "external comparator provenance",
        ).get("candidate_binding")
        == {
            "canonical_task_id": None,
            "registry_row": None,
            "promptable_registry_row": None,
        },
        "external comparator cannot become the v2 candidate binding",
    )
    return authority, reliability_record, method, method_sha256


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--reliability-manifest", type=Path, default=DEFAULT_RELIABILITY_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    candidate = args.candidate
    if not candidate.is_absolute():
        candidate = args.repo_root / candidate
    reliability = args.reliability_manifest
    if not reliability.is_absolute():
        reliability = args.repo_root / reliability
    authority, _, _, _ = build_v2_authority(
        repo_root=args.repo_root,
        candidate=candidate,
        reliability_manifest=reliability,
    )
    try:
        write_binding_receipt(args.output, authority)
    except BindingError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(authority, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
