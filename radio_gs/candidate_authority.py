"""Fail-closed preflight for the five-contract Candidate Authority Bundle.

The bundle is the immutable boundary between method construction and any
scored lifecycle work.  It binds one shared Method Contract, the five current
Evaluation Contracts, and the execution identities needed to attribute later
evidence.  Benchmark identity appears only in the Evaluation Adapter bindings;
it cannot select algorithmic behavior.  The registered-2D SAM3 identity below
belongs to the current LUDVIG-online Primary method contract from issue #22;
it is not the historical LUDVIG comparator path scoped by ADR 0002.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radio_gs.utils.immutable_artifacts import (
    canonical_json_bytes as _canonical_json_bytes,
    load_json_object,
    write_frozen_json,
)


CANDIDATE_AUTHORITY_SCHEMA = "radio_gs.candidate_authority.v1"
METHOD_CONTRACT_SCHEMA = "radio_gs.method_contract.v1"
EVALUATION_CONTRACT_SCHEMA = "radio_gs.evaluation_contract.v1"
RUNTIME_COMPLIANCE_PROOF_SCHEMA = "runtime-compliance-proof-v1"
ATTEMPT_LEDGER_SCHEMA = "attempt-ledger-v1"

EXPECTED_EVALUATION_CONTRACT_IDS = (
    "lerf2d-field-only-four-scene-v1",
    "lerf3d-field-only-four-scene-v1",
    "nvos-ludvig-online-all-view-eight-v1",
    "spin-ludvig-online-full-mask-available-nine-v1",
    "scannet-ovs-paper8-v1",
)

_METHOD_KEYS = {
    "method_contract_schema",
    "field_schema",
    "joint_mapping_objective",
    "mapping_checkpoint_rule",
    "global_method_parameters",
    "method_specific_global_parameters",
    "modality_compilers",
    "capability_views",
    "field_derived_support_topology",
    "solvers_calibrations",
    "output_domain_operators",
    "precision_determinism_policy",
    "implementation_identity",
    "environment_identity",
}
_AUTHORITY_KEYS = {
    "schema_version",
    "candidate_id",
    "method_contract",
    "evaluation_contracts",
    "execution_matrix",
    "seed_policy",
    "retry_policy",
    "runtime_compliance_proof_schema",
    "attempt_ledger_schema",
}
_EVALUATION_KEYS = {
    "contract_schema",
    "contract_id",
    "adapter_id",
    "cohort_identity",
    "information_boundary",
    "authorized_query_input",
    "output_domain",
    "evaluator_identity",
    "metric_aggregation_identity",
    "comparator_identity",
    "target_identity",
    "mandatory_floor_identity",
}
_FIELD_KEYS = {
    "family",
    "local_code_dimension",
    "persistent_semantic_fields",
    "canonical_capability_feature",
    "deployment_support_state",
    "decoder_identity",
    "fusion_identity",
    "sidecar_policy",
}
_MAPPING_OBJECTIVE_KEYS = {
    "identity",
    "calibration_authority",
    "normalization_policy",
    "primitive_formula",
    "render_formula",
    "source_scope",
    "benchmark_independent",
}
_CHECKPOINT_KEYS = {
    "identity",
    "selection_scope",
    "validation_objective",
    "validation_split",
    "check_frequency",
    "patience",
    "maximum_budget",
    "observation_budget_schedule",
    "deterministic_tie_break",
    "earliest_best",
    "min_delta",
    "benchmark_independent",
}
_GLOBAL_PARAMETER_KEYS = {"identity", "frozen", "scene_bound"}
_METHOD_GLOBAL_PARAMETER_KEYS = {
    "identity",
    "frozen",
    "scene_bound",
    "soft_limit_bytes",
    "hard_limit_bytes",
}
_COMPILER_KEYS = {
    "identity",
    "benchmark_independent",
    "query_workspace_only",
    "query_time_vision_model_identity",
}
_VIEW_KEYS = {"identity", "source", "persistent"}
_TOPOLOGY_KEYS = {
    "identity",
    "source",
    "persistent",
    "query_independent",
    "rebuildable",
}
_SOLVER_KEYS = {
    "solver_identity",
    "calibration_identity",
    "frozen",
    "benchmark_independent",
}
_OUTPUT_KEYS = {"identity", "inputs", "benchmark_independent"}
_PRECISION_KEYS = {
    "deployed_dtype",
    "compute_dtype",
    "binary_posterior_boundary",
    "deterministic_reduction",
    "tie_policy",
}
_IMPLEMENTATION_KEYS = {"repository", "commit", "dirty_patch_sha256"}
_ENVIRONMENT_KEYS = {"runtime", "container_or_environment", "dependency_lock_sha256"}
_EVALUATION_BOUNDARY_KEYS = {
    "mapping_captured_rgb",
    "query_captured_rgb",
    "rendered_rgb",
    "targets",
    "labels",
}
_QUERY_INPUT_KEYS = {"modality", "shape", "private_siblings"}
_EXECUTION_MATRIX_KEYS = {
    "contract_ids",
    "required_stage_order",
    "requires_all_contracts",
}
_SEED_POLICY_KEYS = {
    "stochastic_seeds",
    "deterministic_seed",
    "paired_across_contracts",
}
_RETRY_POLICY_KEYS = {"max_retries", "allowed_failure_class", "identity_preserving"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_CONTRACT_FACTS = {
    "lerf2d-field-only-four-scene-v1": {
        "cohort_identity": "lerf-four-scene-22-camera-208-query-v1",
        "query_modality": "category_text",
        "query_shape": "official_object_category_string",
        "query_captured_rgb": "forbidden",
        "mapping_captured_rgb": "authorized",
        "rendered_rgb": "forbidden",
        "targets": "evaluator_private",
        "labels": "evaluator_private",
        "private_siblings": ["target_mask", "target_box", "metric"],
        "output_domain": "camera_raster",
        "evaluator_identity": "lerf-evaluator-v1",
        "metric_aggregation_identity": "lerf2d-object-scene-four-scene-macro-v1",
        "comparator_identity": "sad-gs-vpa-occamlgs-v1",
        "target_identity": "sad-gs-68.8-locacc-88.7-round-half-up-v1",
        "mandatory_floor_identity": "occamlgs-63.6200435-82.8487101-v1",
    },
    "lerf3d-field-only-four-scene-v1": {
        "cohort_identity": "lerf-four-scene-208-query-v1",
        "query_modality": "category_text",
        "query_shape": "official_object_category_string",
        "query_captured_rgb": "forbidden",
        "mapping_captured_rgb": "authorized",
        "rendered_rgb": "forbidden",
        "targets": "evaluator_private",
        "labels": "evaluator_private",
        "private_siblings": ["target_mask", "metric"],
        "output_domain": "world_sample",
        "evaluator_identity": "lerf-evaluator-v1",
        "metric_aggregation_identity": "lerf3d-object-scene-four-scene-macro-v1",
        "comparator_identity": "pairgs-raw-3d-vala-v1",
        "target_identity": "pairgs-60.4-79.6-68.2-round-half-up-v1",
        "mandatory_floor_identity": "vala-54.1248877-79.3526471-56.6114038-v1",
    },
    "nvos-ludvig-online-all-view-eight-v1": {
        "cohort_identity": "nvos-official-eight-task-v1",
        "query_modality": "positive_scribble_raster",
        "query_shape": "official_reference_identity_and_calibration",
        "query_captured_rgb": "all_registered_views",
        "mapping_captured_rgb": "all_registered_views",
        "rendered_rgb": "forbidden",
        "targets": "evaluator_private",
        "labels": "evaluator_private",
        "private_siblings": ["target_mask", "metric"],
        "output_domain": "camera_raster",
        "evaluator_identity": "nvos-evaluator-v1",
        "metric_aggregation_identity": "nvos-eight-task-foreground-iou-macro-v1",
        "comparator_identity": "ludvig-dinov2-v1",
        "target_identity": "ludvig-dinov2-92.4-round-half-up-v1",
        "mandatory_floor_identity": "ludvig-sam-91.25768502741802-v1",
    },
    "spin-ludvig-online-full-mask-available-nine-v1": {
        "cohort_identity": "spin-available-nine-excluding-fork-v1",
        "query_modality": "complete_binary_mask",
        "query_shape": "canonical_reference_identity_and_calibration",
        "query_captured_rgb": "all_registered_views",
        "mapping_captured_rgb": "all_registered_views",
        "rendered_rgb": "forbidden",
        "targets": "evaluator_private",
        "labels": "evaluator_private",
        "private_siblings": ["target_mask", "metric"],
        "output_domain": "camera_raster",
        "evaluator_identity": "spin-evaluator-v1",
        "metric_aggregation_identity": "spin-frame-scene-nine-scene-macro-v1",
        "comparator_identity": "ludvig-sam-available9-v1",
        "target_identity": "ludvig-sam-93.7200449592385-v1",
        "mandatory_floor_identity": "ludvig-sam-93.7200449592385-v1",
    },
    "scannet-ovs-paper8-v1": {
        "cohort_identity": "scannet-paper8-eight-scene-v1",
        "query_modality": "category_text",
        "query_shape": "official_19_class_bank",
        "query_captured_rgb": "forbidden",
        "mapping_captured_rgb": "registered_rgbd_and_poses",
        "rendered_rgb": "forbidden",
        "targets": "evaluator_private",
        "labels": "evaluator_private",
        "private_siblings": ["mesh_labels", "pseudo_gt", "metric"],
        "output_domain": "world_sample",
        "evaluator_identity": "scannet-ovs-evaluator-v1",
        "metric_aggregation_identity": "scannet-eight-scene-19-15-10-macro-v1",
        "comparator_identity": "vala-paper8-v1",
        "target_identity": "scannet-six-component-zero-tolerance-v1",
        "mandatory_floor_identity": "scannet-six-component-zero-tolerance-v1",
    },
}


class CandidateAuthorityError(ValueError):
    """Raised when Candidate Authority preflight cannot prove a valid bundle."""


def _fail(message: str) -> None:
    raise CandidateAuthorityError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if any(
            "benchmark" in key or "task" in key or "contract" in key for key in extra
        ):
            _fail(f"{label} is benchmark-conditioned; extra={extra}")
        _fail(f"{label} has unexpected fields; missing={missing}, extra={extra}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value


def _bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        _fail(f"{label} must be {expected}")


def _finite_json(value: Any, label: str) -> None:
    try:
        _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        _fail(f"{label} must be finite canonical JSON: {error}")


def _validate_method_contract(method: Any) -> None:
    method_map = _mapping(method, "method_contract")
    _exact_keys(method_map, _METHOD_KEYS, "method_contract")
    _nonempty_string(method_map["method_contract_schema"], "method_contract_schema")
    if method_map["method_contract_schema"] != METHOD_CONTRACT_SCHEMA:
        _fail("method_contract_schema is not the current Method Contract")

    field = _mapping(method_map["field_schema"], "field_schema")
    _exact_keys(field, _FIELD_KEYS, "field_schema")
    if field["family"] != "sidecar_free_d512_l512_single_feature":
        _fail("field_schema must bind the sidecar-free D512/L512 field family")
    if field["local_code_dimension"] != 512:
        _fail("field_schema.local_code_dimension must equal 512")
    if field["persistent_semantic_fields"] != 1:
        _fail("field_schema.persistent_semantic_fields must equal one")
    if field["canonical_capability_feature"] != "Canonical Capability Feature":
        _fail("field_schema must name the Canonical Capability Feature")
    support = _mapping(field["deployment_support_state"], "deployment_support_state")
    _exact_keys(
        support, {"validity_bits", "quality_scalars"}, "deployment_support_state"
    )
    if support["validity_bits"] != 1 or support["quality_scalars"] != 5:
        _fail("deployment support state exceeds the admitted schema")
    if field["sidecar_policy"] != "forbid_second_semantic_field_and_cache":
        _fail("field_schema sidecar policy is not the current fail-closed policy")
    _nonempty_string(field["decoder_identity"], "field_schema.decoder_identity")
    _nonempty_string(field["fusion_identity"], "field_schema.fusion_identity")

    objective = _mapping(
        method_map["joint_mapping_objective"], "joint_mapping_objective"
    )
    _exact_keys(objective, _MAPPING_OBJECTIVE_KEYS, "joint_mapping_objective")
    _nonempty_string(objective["identity"], "joint_mapping_objective.identity")
    if objective["identity"] != "joint-mapping-objective-v1":
        _fail("joint_mapping_objective identity differs from the frozen method")
    if objective["primitive_formula"] != "R + 0.5 C + 0.25 S + Omega":
        _fail("joint_mapping_objective primitive formula differs")
    if objective["render_formula"] != "R + 0.5 C + 0.25 S + 0.5 G + 0.5 B + Omega":
        _fail("joint_mapping_objective render formula differs")
    if objective["calibration_authority"] != "scene-disjoint-source-calibration-v1":
        _fail("joint_mapping_objective calibration authority differs")
    if objective["normalization_policy"] != "b0-median-nonzero-family-scale-v1":
        _fail("joint_mapping_objective normalization policy differs")
    _nonempty_string(
        objective["primitive_formula"], "joint_mapping_objective.primitive_formula"
    )
    _nonempty_string(
        objective["render_formula"], "joint_mapping_objective.render_formula"
    )
    _nonempty_string(objective["source_scope"], "joint_mapping_objective.source_scope")
    _bool(
        objective["benchmark_independent"],
        True,
        "joint_mapping_objective.benchmark_independent",
    )

    checkpoint = _mapping(
        method_map["mapping_checkpoint_rule"], "mapping_checkpoint_rule"
    )
    _exact_keys(checkpoint, _CHECKPOINT_KEYS, "mapping_checkpoint_rule")
    _nonempty_string(checkpoint["identity"], "mapping_checkpoint_rule.identity")
    if checkpoint["identity"] != "mapping-only-checkpoint-rule-v1":
        _fail("mapping_checkpoint_rule identity differs from the frozen method")
    if checkpoint["selection_scope"] != "mapping_only":
        _fail("mapping_checkpoint_rule must be mapping-only")
    _nonempty_string(
        checkpoint["validation_objective"],
        "mapping_checkpoint_rule.validation_objective",
    )
    if checkpoint["validation_objective"] != "joint_mapping_objective":
        _fail("mapping_checkpoint_rule validation objective differs")
    if checkpoint["validation_split"] != "scene-disjoint-held-out-mapping-v1":
        _fail("mapping_checkpoint_rule validation split differs")
    if checkpoint["check_frequency"] != "every_observation_equivalent_pass":
        _fail("mapping_checkpoint_rule check frequency differs")
    if checkpoint["patience"] != "none":
        _fail("mapping_checkpoint_rule patience differs")
    if checkpoint["maximum_budget"] != 16:
        _fail("mapping_checkpoint_rule maximum budget differs")
    if checkpoint["observation_budget_schedule"] != [1, 2, 4, 8, 16]:
        _fail("mapping_checkpoint_rule observation budget schedule differs")
    if (
        checkpoint["deterministic_tie_break"]
        != "earliest_checkpoint_then_canonical_digest_order"
    ):
        _fail("mapping_checkpoint_rule deterministic tie-break differs")
    _bool(checkpoint["earliest_best"], True, "mapping_checkpoint_rule.earliest_best")
    if checkpoint["min_delta"] != 0:
        _fail("mapping_checkpoint_rule.min_delta must equal zero")
    _bool(
        checkpoint["benchmark_independent"],
        True,
        "mapping_checkpoint_rule.benchmark_independent",
    )

    global_parameters = _mapping(
        method_map["global_method_parameters"], "global_method_parameters"
    )
    _exact_keys(global_parameters, _GLOBAL_PARAMETER_KEYS, "global_method_parameters")
    _nonempty_string(global_parameters["identity"], "global_method_parameters.identity")
    if global_parameters["identity"] != "global-method-parameters-v1":
        _fail("global_method_parameters identity differs from the frozen method")
    _bool(global_parameters["frozen"], True, "global_method_parameters.frozen")
    _bool(
        global_parameters["scene_bound"], False, "global_method_parameters.scene_bound"
    )

    method_parameters = _mapping(
        method_map["method_specific_global_parameters"],
        "method_specific_global_parameters",
    )
    _exact_keys(
        method_parameters,
        _METHOD_GLOBAL_PARAMETER_KEYS,
        "method_specific_global_parameters",
    )
    _nonempty_string(
        method_parameters["identity"], "method_specific_global_parameters.identity"
    )
    if method_parameters["identity"] != "method-specific-global-parameters-v1":
        _fail(
            "method_specific_global_parameters identity differs from the frozen method"
        )
    _bool(method_parameters["frozen"], True, "method_specific_global_parameters.frozen")
    _bool(
        method_parameters["scene_bound"],
        False,
        "method_specific_global_parameters.scene_bound",
    )
    if (
        not isinstance(method_parameters["soft_limit_bytes"], int)
        or method_parameters["soft_limit_bytes"] <= 0
    ):
        _fail("method_specific_global_parameters.soft_limit_bytes must be positive")
    if (
        not isinstance(method_parameters["hard_limit_bytes"], int)
        or method_parameters["hard_limit_bytes"]
        <= method_parameters["soft_limit_bytes"]
    ):
        _fail(
            "method_specific_global_parameters.hard_limit_bytes must exceed soft limit"
        )

    compilers = _mapping(method_map["modality_compilers"], "modality_compilers")
    expected_modalities = {
        "category_text",
        "positive_scribble_raster",
        "complete_binary_mask",
    }
    _exact_keys(compilers, expected_modalities, "modality_compilers")
    for modality, raw_compiler in compilers.items():
        compiler = _mapping(raw_compiler, f"modality_compilers.{modality}")
        _exact_keys(compiler, _COMPILER_KEYS, f"modality_compilers.{modality}")
        _nonempty_string(
            compiler["identity"], f"modality_compilers.{modality}.identity"
        )
        _nonempty_string(
            compiler["query_time_vision_model_identity"],
            f"modality_compilers.{modality}.query_time_vision_model_identity",
        )
        expected_compiler_identity = {
            "category_text": "category-query-modality-compiler-v1",
            "positive_scribble_raster": "registered2d-sam3-multiview-consensus-v1",
            "complete_binary_mask": "registered2d-sam3-multiview-consensus-v1",
        }[modality]
        if compiler["identity"] != expected_compiler_identity:
            _fail(
                f"modality_compilers.{modality} identity differs from the frozen method"
            )
        expected_vision_identity = (
            "not_applicable"
            if modality == "category_text"
            else "official-sam3-query-time-vision-model-v1"
        )
        if compiler["query_time_vision_model_identity"] != expected_vision_identity:
            _fail(f"modality_compilers.{modality} vision model identity differs")
        _bool(
            compiler["benchmark_independent"],
            True,
            f"modality_compilers.{modality}.benchmark_independent",
        )
        _bool(
            compiler["query_workspace_only"],
            True,
            f"modality_compilers.{modality}.query_workspace_only",
        )

    views = _mapping(method_map["capability_views"], "capability_views")
    _exact_keys(views, {"semantic", "appearance", "boundary"}, "capability_views")
    for name, raw_view in views.items():
        view = _mapping(raw_view, f"capability_views.{name}")
        _exact_keys(view, _VIEW_KEYS, f"capability_views.{name}")
        _nonempty_string(view["identity"], f"capability_views.{name}.identity")
        expected_view_identity = f"{name}-capability-view-v1"
        if view["identity"] != expected_view_identity:
            _fail(f"capability_views.{name} identity differs from the frozen method")
        if view["source"] != "Canonical Capability Feature":
            _fail(f"capability_views.{name} must derive from the sole field")
        _bool(view["persistent"], False, f"capability_views.{name}.persistent")

    topology = _mapping(
        method_map["field_derived_support_topology"],
        "field_derived_support_topology",
    )
    _exact_keys(topology, _TOPOLOGY_KEYS, "field_derived_support_topology")
    _nonempty_string(topology["identity"], "field_derived_support_topology.identity")
    if topology["identity"] != "boundary-calibrated-field-region-hierarchy-v1":
        _fail("field_derived_support_topology identity differs from the frozen method")
    if topology["source"] != "Deployment Scene State and Global Method Parameters":
        _fail("field_derived_support_topology must derive from field state")
    _bool(topology["persistent"], False, "field_derived_support_topology.persistent")
    _bool(
        topology["query_independent"],
        True,
        "field_derived_support_topology.query_independent",
    )
    _bool(topology["rebuildable"], True, "field_derived_support_topology.rebuildable")

    solvers = _mapping(method_map["solvers_calibrations"], "solvers_calibrations")
    _exact_keys(
        solvers, {"category_retrieval", "instance_selection"}, "solvers_calibrations"
    )
    for name, raw_solver in solvers.items():
        solver = _mapping(raw_solver, f"solvers_calibrations.{name}")
        _exact_keys(solver, _SOLVER_KEYS, f"solvers_calibrations.{name}")
        _nonempty_string(
            solver["solver_identity"], f"solvers_calibrations.{name}.solver_identity"
        )
        _nonempty_string(
            solver["calibration_identity"],
            f"solvers_calibrations.{name}.calibration_identity",
        )
        expected_solver_identity = {
            "category_retrieval": "semantic-region-hypothesis-marginal-v1",
            "instance_selection": "candidate-and-hierarchy-marginal-v1",
        }[name]
        expected_calibration_identity = {
            "category_retrieval": "category-calibration-v1",
            "instance_selection": "instance-calibration-v1",
        }[name]
        if solver["solver_identity"] != expected_solver_identity:
            _fail(f"solvers_calibrations.{name} solver identity differs")
        if solver["calibration_identity"] != expected_calibration_identity:
            _fail(f"solvers_calibrations.{name} calibration identity differs")
        _bool(solver["frozen"], True, f"solvers_calibrations.{name}.frozen")
        _bool(
            solver["benchmark_independent"],
            True,
            f"solvers_calibrations.{name}.benchmark_independent",
        )

    operators = _mapping(
        method_map["output_domain_operators"], "output_domain_operators"
    )
    _exact_keys(operators, {"camera_raster", "world_sample"}, "output_domain_operators")
    for name, raw_operator in operators.items():
        operator = _mapping(raw_operator, f"output_domain_operators.{name}")
        _exact_keys(operator, _OUTPUT_KEYS, f"output_domain_operators.{name}")
        _nonempty_string(
            operator["identity"], f"output_domain_operators.{name}.identity"
        )
        if operator["identity"] != f"{name.replace('_', '-')}-output-v1":
            _fail(
                f"output_domain_operators.{name} identity differs from the frozen method"
            )
        if not isinstance(operator["inputs"], list) or not operator["inputs"]:
            _fail(f"output_domain_operators.{name}.inputs must be non-empty")
        _bool(
            operator["benchmark_independent"],
            True,
            f"output_domain_operators.{name}.benchmark_independent",
        )

    precision = _mapping(
        method_map["precision_determinism_policy"], "precision_determinism_policy"
    )
    _exact_keys(precision, _PRECISION_KEYS, "precision_determinism_policy")
    _nonempty_string(
        precision["deployed_dtype"], "precision_determinism_policy.deployed_dtype"
    )
    _nonempty_string(
        precision["compute_dtype"], "precision_determinism_policy.compute_dtype"
    )
    if precision["binary_posterior_boundary"] != 0.5:
        _fail("precision_determinism_policy binary boundary must equal 0.5")
    _bool(
        precision["deterministic_reduction"],
        True,
        "precision_determinism_policy.deterministic_reduction",
    )
    _nonempty_string(precision["tie_policy"], "precision_determinism_policy.tie_policy")

    implementation = _mapping(
        method_map["implementation_identity"], "implementation_identity"
    )
    _exact_keys(implementation, _IMPLEMENTATION_KEYS, "implementation_identity")
    for key in _IMPLEMENTATION_KEYS:
        _nonempty_string(implementation[key], f"implementation_identity.{key}")
    _require_sha256(
        implementation["dirty_patch_sha256"],
        "implementation_identity.dirty_patch_sha256",
    )

    environment = _mapping(method_map["environment_identity"], "environment_identity")
    _exact_keys(environment, _ENVIRONMENT_KEYS, "environment_identity")
    for key in _ENVIRONMENT_KEYS:
        _nonempty_string(environment[key], f"environment_identity.{key}")
    _require_sha256(
        environment["dependency_lock_sha256"],
        "environment_identity.dependency_lock_sha256",
    )


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")


def _validate_evaluation_contracts(contracts: Any) -> None:
    if not isinstance(contracts, list):
        _fail("evaluation_contracts must be a list")
    if (
        tuple(
            contract.get("contract_id") if isinstance(contract, Mapping) else None
            for contract in contracts
        )
        != EXPECTED_EVALUATION_CONTRACT_IDS
    ):
        _fail("evaluation contract identity or frozen order differs")
    adapter_ids = [
        contract.get("adapter_id") if isinstance(contract, Mapping) else None
        for contract in contracts
    ]
    if len(set(adapter_ids)) != len(adapter_ids):
        _fail("evaluation adapter identities must be unique")
    for index, raw_contract in enumerate(contracts):
        contract = _mapping(raw_contract, f"evaluation_contracts[{index}]")
        _exact_keys(contract, _EVALUATION_KEYS, f"evaluation_contracts[{index}]")
        if contract["contract_schema"] != EVALUATION_CONTRACT_SCHEMA:
            _fail(f"evaluation_contracts[{index}] contract identity is not current")
        _nonempty_string(
            contract["contract_id"], f"evaluation_contracts[{index}].contract_id"
        )
        expected_facts = _EXPECTED_CONTRACT_FACTS[contract["contract_id"]]
        for fact in ("cohort_identity", "output_domain"):
            expected = expected_facts[fact]
            if contract.get(fact) != expected:
                _fail(
                    f"evaluation_contracts[{index}] {fact} differs from the current contract"
                )
        _nonempty_string(
            contract["adapter_id"], f"evaluation_contracts[{index}].adapter_id"
        )
        _nonempty_string(
            contract["cohort_identity"],
            f"evaluation_contracts[{index}].cohort_identity",
        )
        _nonempty_string(
            contract["output_domain"], f"evaluation_contracts[{index}].output_domain"
        )
        for key in (
            "evaluator_identity",
            "metric_aggregation_identity",
            "comparator_identity",
            "target_identity",
            "mandatory_floor_identity",
        ):
            _nonempty_string(contract[key], f"evaluation_contracts[{index}].{key}")
            if contract[key] != expected_facts[key]:
                _fail(
                    f"evaluation_contracts[{index}] {key} differs from the current authority"
                )
        boundary = _mapping(
            contract["information_boundary"],
            f"evaluation_contracts[{index}].information_boundary",
        )
        _exact_keys(
            boundary,
            _EVALUATION_BOUNDARY_KEYS,
            f"evaluation_contracts[{index}].information_boundary",
        )
        query_input = _mapping(
            contract["authorized_query_input"],
            f"evaluation_contracts[{index}].authorized_query_input",
        )
        _exact_keys(
            query_input,
            _QUERY_INPUT_KEYS,
            f"evaluation_contracts[{index}].authorized_query_input",
        )
        if query_input["modality"] != expected_facts["query_modality"]:
            _fail(
                f"evaluation_contracts[{index}] query modality differs from the current contract"
            )
        if query_input["shape"] != expected_facts["query_shape"]:
            _fail(
                f"evaluation_contracts[{index}] query shape differs from the current contract"
            )
        _nonempty_string(
            query_input["modality"],
            f"evaluation_contracts[{index}].authorized_query_input.modality",
        )
        _nonempty_string(
            query_input["shape"],
            f"evaluation_contracts[{index}].authorized_query_input.shape",
        )
        if not isinstance(query_input["private_siblings"], list):
            _fail(
                f"evaluation_contracts[{index}].authorized_query_input.private_siblings must be a list"
            )
        for boundary_key in (
            "mapping_captured_rgb",
            "query_captured_rgb",
            "rendered_rgb",
            "targets",
            "labels",
        ):
            if boundary[boundary_key] != expected_facts[boundary_key]:
                if (
                    boundary_key == "query_captured_rgb"
                    and boundary[boundary_key] == "forbidden"
                ):
                    _fail(
                        f"evaluation_contracts[{index}] is an obsolete RGB-free authority"
                    )
                _fail(
                    f"evaluation_contracts[{index}] {boundary_key} differs from the current information boundary"
                )
        if query_input["private_siblings"] != expected_facts["private_siblings"]:
            _fail(f"evaluation_contracts[{index}] private sibling grants differ")


def _validate_execution_policy(authority: Mapping[str, Any]) -> None:
    matrix = _mapping(authority["execution_matrix"], "execution_matrix")
    _exact_keys(matrix, _EXECUTION_MATRIX_KEYS, "execution_matrix")
    if tuple(matrix["contract_ids"]) != EXPECTED_EVALUATION_CONTRACT_IDS:
        _fail("execution_matrix contract order differs")
    if tuple(matrix["required_stage_order"]) != (
        "mapping_training",
        "deployment_sealing",
        "warm_cache_compilation",
        "query_prediction_sealing",
        "evaluation",
    ):
        _fail("execution_matrix stage order differs")
    _bool(
        matrix["requires_all_contracts"],
        True,
        "execution_matrix.requires_all_contracts",
    )

    seeds = _mapping(authority["seed_policy"], "seed_policy")
    _exact_keys(seeds, _SEED_POLICY_KEYS, "seed_policy")
    if seeds["stochastic_seeds"] != [0, 1, 2]:
        _fail("seed_policy must bind paired stochastic seeds [0, 1, 2]")
    if seeds["deterministic_seed"] != "not_applicable":
        _fail("seed_policy deterministic seed differs")
    _bool(seeds["paired_across_contracts"], True, "seed_policy.paired_across_contracts")

    retry = _mapping(authority["retry_policy"], "retry_policy")
    _exact_keys(retry, _RETRY_POLICY_KEYS, "retry_policy")
    if retry["max_retries"] != 1:
        _fail("retry_policy must allow at most one retry")
    if retry["allowed_failure_class"] != "predeclared_infrastructure_failure":
        _fail("retry_policy allows an undeclared failure class")
    _bool(retry["identity_preserving"], True, "retry_policy.identity_preserving")

    if authority["runtime_compliance_proof_schema"] != RUNTIME_COMPLIANCE_PROOF_SCHEMA:
        _fail("runtime compliance proof schema differs")
    if authority["attempt_ledger_schema"] != ATTEMPT_LEDGER_SCHEMA:
        _fail("attempt ledger schema differs")


def _authority_body(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in authority.items()
        if key != "candidate_id"
    }


def _validate_authority_mapping(
    authority: Any, *, check_identity: bool
) -> dict[str, Any]:
    value = _mapping(authority, "candidate authority")
    _exact_keys(value, _AUTHORITY_KEYS, "candidate authority")
    if value["schema_version"] != CANDIDATE_AUTHORITY_SCHEMA:
        _fail("candidate authority schema differs")
    _validate_method_contract(value["method_contract"])
    _validate_evaluation_contracts(value["evaluation_contracts"])
    _validate_execution_policy(value)
    body = _authority_body(value)
    _finite_json(body, "candidate authority")
    expected_id = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    if check_identity:
        _require_sha256(value["candidate_id"], "candidate_id")
        if value["candidate_id"] != expected_id:
            _fail("candidate_id does not match the content-addressed authority")
    return copy.deepcopy(dict(value))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class CandidateAuthorityBundle(Mapping[str, Any]):
    """An in-memory, recursively immutable Candidate Authority Bundle."""

    _payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def as_dict(self) -> dict[str, Any]:
        return _thaw(self._payload)

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict()) + b"\n"


def build_candidate_authority(
    *,
    method_contract: Mapping[str, Any],
    evaluation_contracts: list[Mapping[str, Any]],
    execution_matrix: Mapping[str, Any],
    seed_policy: Mapping[str, Any],
    retry_policy: Mapping[str, Any],
    runtime_compliance_proof_schema: str,
    attempt_ledger_schema: str,
) -> CandidateAuthorityBundle:
    """Validate and content-address one complete candidate preflight input."""

    authority = {
        "schema_version": CANDIDATE_AUTHORITY_SCHEMA,
        "candidate_id": "0" * 64,
        "method_contract": copy.deepcopy(method_contract),
        "evaluation_contracts": copy.deepcopy(evaluation_contracts),
        "execution_matrix": copy.deepcopy(execution_matrix),
        "seed_policy": copy.deepcopy(seed_policy),
        "retry_policy": copy.deepcopy(retry_policy),
        "runtime_compliance_proof_schema": runtime_compliance_proof_schema,
        "attempt_ledger_schema": attempt_ledger_schema,
    }
    _validate_authority_mapping(authority, check_identity=False)
    authority["candidate_id"] = hashlib.sha256(
        _canonical_json_bytes(_authority_body(authority))
    ).hexdigest()
    validated = _validate_authority_mapping(authority, check_identity=True)
    return CandidateAuthorityBundle(_freeze(validated))


def validate_candidate_authority(
    authority: Mapping[str, Any] | CandidateAuthorityBundle,
) -> CandidateAuthorityBundle:
    """Revalidate a candidate bundle without accepting metadata-only repair."""

    value = (
        authority.as_dict()
        if isinstance(authority, CandidateAuthorityBundle)
        else authority
    )
    validated = _validate_authority_mapping(value, check_identity=True)
    return CandidateAuthorityBundle(_freeze(validated))


def write_candidate_authority(
    path: str | Path,
    authority: Mapping[str, Any] | CandidateAuthorityBundle,
) -> Path:
    """Publish one exact bundle without replacing a different existing artifact."""

    bundle = validate_candidate_authority(authority)
    return write_frozen_json(path, bundle.as_dict())


def load_candidate_authority(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> CandidateAuthorityBundle:
    """Load and revalidate an immutable bundle through the artifact seam."""

    payload, _, _ = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="candidate authority",
    )
    return validate_candidate_authority(payload)


def audit_candidate_authority(path: str | Path) -> dict[str, Any]:
    """Return a fail-closed validation report for a persisted authority."""

    try:
        bundle = load_candidate_authority(path)
    except (OSError, TypeError, ValueError) as error:
        return {"valid": False, "candidate_id": None, "errors": [str(error)]}
    return {"valid": True, "candidate_id": bundle["candidate_id"], "errors": []}


def reference_candidate_authority_inputs() -> dict[str, Any]:
    """Return a complete small fixture for the public Candidate Authority seam."""

    method_contract = {
        "method_contract_schema": METHOD_CONTRACT_SCHEMA,
        "field_schema": {
            "family": "sidecar_free_d512_l512_single_feature",
            "local_code_dimension": 512,
            "persistent_semantic_fields": 1,
            "canonical_capability_feature": "Canonical Capability Feature",
            "deployment_support_state": {"validity_bits": 1, "quality_scalars": 5},
            "decoder_identity": "radio-d512-decoder-v1",
            "fusion_identity": "radio-l512-fusion-v1",
            "sidecar_policy": "forbid_second_semantic_field_and_cache",
        },
        "joint_mapping_objective": {
            "identity": "joint-mapping-objective-v1",
            "calibration_authority": "scene-disjoint-source-calibration-v1",
            "normalization_policy": "b0-median-nonzero-family-scale-v1",
            "primitive_formula": "R + 0.5 C + 0.25 S + Omega",
            "render_formula": "R + 0.5 C + 0.25 S + 0.5 G + 0.5 B + Omega",
            "source_scope": "mapping_observations_and_mapping_artifacts_only",
            "benchmark_independent": True,
        },
        "mapping_checkpoint_rule": {
            "identity": "mapping-only-checkpoint-rule-v1",
            "selection_scope": "mapping_only",
            "validation_objective": "joint_mapping_objective",
            "validation_split": "scene-disjoint-held-out-mapping-v1",
            "check_frequency": "every_observation_equivalent_pass",
            "patience": "none",
            "maximum_budget": 16,
            "observation_budget_schedule": [1, 2, 4, 8, 16],
            "deterministic_tie_break": "earliest_checkpoint_then_canonical_digest_order",
            "earliest_best": True,
            "min_delta": 0,
            "benchmark_independent": True,
        },
        "global_method_parameters": {
            "identity": "global-method-parameters-v1",
            "frozen": True,
            "scene_bound": False,
        },
        "method_specific_global_parameters": {
            "identity": "method-specific-global-parameters-v1",
            "frozen": True,
            "scene_bound": False,
            "soft_limit_bytes": 8 * 1024 * 1024,
            "hard_limit_bytes": 128 * 1024 * 1024,
        },
        "modality_compilers": {
            "category_text": {
                "identity": "category-query-modality-compiler-v1",
                "benchmark_independent": True,
                "query_workspace_only": True,
                "query_time_vision_model_identity": "not_applicable",
            },
            "positive_scribble_raster": {
                "identity": "registered2d-sam3-multiview-consensus-v1",
                "benchmark_independent": True,
                "query_workspace_only": True,
                "query_time_vision_model_identity": "official-sam3-query-time-vision-model-v1",
            },
            "complete_binary_mask": {
                "identity": "registered2d-sam3-multiview-consensus-v1",
                "benchmark_independent": True,
                "query_workspace_only": True,
                "query_time_vision_model_identity": "official-sam3-query-time-vision-model-v1",
            },
        },
        "capability_views": {
            "semantic": {
                "identity": "semantic-capability-view-v1",
                "source": "Canonical Capability Feature",
                "persistent": False,
            },
            "appearance": {
                "identity": "appearance-capability-view-v1",
                "source": "Canonical Capability Feature",
                "persistent": False,
            },
            "boundary": {
                "identity": "boundary-capability-view-v1",
                "source": "Canonical Capability Feature",
                "persistent": False,
            },
        },
        "field_derived_support_topology": {
            "identity": "boundary-calibrated-field-region-hierarchy-v1",
            "source": "Deployment Scene State and Global Method Parameters",
            "persistent": False,
            "query_independent": True,
            "rebuildable": True,
        },
        "solvers_calibrations": {
            "category_retrieval": {
                "solver_identity": "semantic-region-hypothesis-marginal-v1",
                "calibration_identity": "category-calibration-v1",
                "frozen": True,
                "benchmark_independent": True,
            },
            "instance_selection": {
                "solver_identity": "candidate-and-hierarchy-marginal-v1",
                "calibration_identity": "instance-calibration-v1",
                "frozen": True,
                "benchmark_independent": True,
            },
        },
        "output_domain_operators": {
            "camera_raster": {
                "identity": "camera-raster-output-v1",
                "inputs": [
                    "geometry",
                    "opacity",
                    "Gaussian Query Posterior",
                    "Output Request Metadata",
                ],
                "benchmark_independent": True,
            },
            "world_sample": {
                "identity": "world-sample-output-v1",
                "inputs": ["geometry", "opacity", "Gaussian Query Posterior"],
                "benchmark_independent": True,
            },
        },
        "precision_determinism_policy": {
            "deployed_dtype": "float32",
            "compute_dtype": "float32",
            "binary_posterior_boundary": 0.5,
            "deterministic_reduction": True,
            "tie_policy": "canonical_digest_order",
        },
        "implementation_identity": {
            "repository": "Arthurshen926/RADIO-GS",
            "commit": "candidate-authority-fixture",
            "dirty_patch_sha256": "0" * 64,
        },
        "environment_identity": {
            "runtime": "python3.9",
            "container_or_environment": "radio-gs-ci",
            "dependency_lock_sha256": "1" * 64,
        },
    }
    contracts = [
        {
            "contract_schema": EVALUATION_CONTRACT_SCHEMA,
            "contract_id": "lerf2d-field-only-four-scene-v1",
            "adapter_id": "evaluation-adapter/lerf2d-v1",
            "cohort_identity": "lerf-four-scene-22-camera-208-query-v1",
            "information_boundary": {
                "mapping_captured_rgb": "authorized",
                "query_captured_rgb": "forbidden",
                "rendered_rgb": "forbidden",
                "targets": "evaluator_private",
                "labels": "evaluator_private",
            },
            "authorized_query_input": {
                "modality": "category_text",
                "shape": "official_object_category_string",
                "private_siblings": ["target_mask", "target_box", "metric"],
            },
            "output_domain": "camera_raster",
            "evaluator_identity": "lerf-evaluator-v1",
            "metric_aggregation_identity": "lerf2d-object-scene-four-scene-macro-v1",
            "comparator_identity": "sad-gs-vpa-occamlgs-v1",
            "target_identity": "sad-gs-68.8-locacc-88.7-round-half-up-v1",
            "mandatory_floor_identity": "occamlgs-63.6200435-82.8487101-v1",
        },
        {
            "contract_schema": EVALUATION_CONTRACT_SCHEMA,
            "contract_id": "lerf3d-field-only-four-scene-v1",
            "adapter_id": "evaluation-adapter/lerf3d-v1",
            "cohort_identity": "lerf-four-scene-208-query-v1",
            "information_boundary": {
                "mapping_captured_rgb": "authorized",
                "query_captured_rgb": "forbidden",
                "rendered_rgb": "forbidden",
                "targets": "evaluator_private",
                "labels": "evaluator_private",
            },
            "authorized_query_input": {
                "modality": "category_text",
                "shape": "official_object_category_string",
                "private_siblings": ["target_mask", "metric"],
            },
            "output_domain": "world_sample",
            "evaluator_identity": "lerf-evaluator-v1",
            "metric_aggregation_identity": "lerf3d-object-scene-four-scene-macro-v1",
            "comparator_identity": "pairgs-raw-3d-vala-v1",
            "target_identity": "pairgs-60.4-79.6-68.2-round-half-up-v1",
            "mandatory_floor_identity": "vala-54.1248877-79.3526471-56.6114038-v1",
        },
        {
            "contract_schema": EVALUATION_CONTRACT_SCHEMA,
            "contract_id": "nvos-ludvig-online-all-view-eight-v1",
            "adapter_id": "evaluation-adapter/nvos-v1",
            "cohort_identity": "nvos-official-eight-task-v1",
            "information_boundary": {
                "mapping_captured_rgb": "all_registered_views",
                "query_captured_rgb": "all_registered_views",
                "rendered_rgb": "forbidden",
                "targets": "evaluator_private",
                "labels": "evaluator_private",
            },
            "authorized_query_input": {
                "modality": "positive_scribble_raster",
                "shape": "official_reference_identity_and_calibration",
                "private_siblings": ["target_mask", "metric"],
            },
            "output_domain": "camera_raster",
            "evaluator_identity": "nvos-evaluator-v1",
            "metric_aggregation_identity": "nvos-eight-task-foreground-iou-macro-v1",
            "comparator_identity": "ludvig-dinov2-v1",
            "target_identity": "ludvig-dinov2-92.4-round-half-up-v1",
            "mandatory_floor_identity": "ludvig-sam-91.25768502741802-v1",
        },
        {
            "contract_schema": EVALUATION_CONTRACT_SCHEMA,
            "contract_id": "spin-ludvig-online-full-mask-available-nine-v1",
            "adapter_id": "evaluation-adapter/spin-available9-v1",
            "cohort_identity": "spin-available-nine-excluding-fork-v1",
            "information_boundary": {
                "mapping_captured_rgb": "all_registered_views",
                "query_captured_rgb": "all_registered_views",
                "rendered_rgb": "forbidden",
                "targets": "evaluator_private",
                "labels": "evaluator_private",
            },
            "authorized_query_input": {
                "modality": "complete_binary_mask",
                "shape": "canonical_reference_identity_and_calibration",
                "private_siblings": ["target_mask", "metric"],
            },
            "output_domain": "camera_raster",
            "evaluator_identity": "spin-evaluator-v1",
            "metric_aggregation_identity": "spin-frame-scene-nine-scene-macro-v1",
            "comparator_identity": "ludvig-sam-available9-v1",
            "target_identity": "ludvig-sam-93.7200449592385-v1",
            "mandatory_floor_identity": "ludvig-sam-93.7200449592385-v1",
        },
        {
            "contract_schema": EVALUATION_CONTRACT_SCHEMA,
            "contract_id": "scannet-ovs-paper8-v1",
            "adapter_id": "evaluation-adapter/scannet-ovs-paper8-v1",
            "cohort_identity": "scannet-paper8-eight-scene-v1",
            "information_boundary": {
                "mapping_captured_rgb": "registered_rgbd_and_poses",
                "query_captured_rgb": "forbidden",
                "rendered_rgb": "forbidden",
                "targets": "evaluator_private",
                "labels": "evaluator_private",
            },
            "authorized_query_input": {
                "modality": "category_text",
                "shape": "official_19_class_bank",
                "private_siblings": ["mesh_labels", "pseudo_gt", "metric"],
            },
            "output_domain": "world_sample",
            "evaluator_identity": "scannet-ovs-evaluator-v1",
            "metric_aggregation_identity": "scannet-eight-scene-19-15-10-macro-v1",
            "comparator_identity": "vala-paper8-v1",
            "target_identity": "scannet-six-component-zero-tolerance-v1",
            "mandatory_floor_identity": "scannet-six-component-zero-tolerance-v1",
        },
    ]
    return {
        "method_contract": method_contract,
        "evaluation_contracts": contracts,
        "execution_matrix": {
            "contract_ids": list(EXPECTED_EVALUATION_CONTRACT_IDS),
            "required_stage_order": [
                "mapping_training",
                "deployment_sealing",
                "warm_cache_compilation",
                "query_prediction_sealing",
                "evaluation",
            ],
            "requires_all_contracts": True,
        },
        "seed_policy": {
            "stochastic_seeds": [0, 1, 2],
            "deterministic_seed": "not_applicable",
            "paired_across_contracts": True,
        },
        "retry_policy": {
            "max_retries": 1,
            "allowed_failure_class": "predeclared_infrastructure_failure",
            "identity_preserving": True,
        },
        "runtime_compliance_proof_schema": RUNTIME_COMPLIANCE_PROOF_SCHEMA,
        "attempt_ledger_schema": ATTEMPT_LEDGER_SCHEMA,
    }
