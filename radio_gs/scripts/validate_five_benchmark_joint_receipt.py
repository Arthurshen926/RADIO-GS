#!/usr/bin/env python3
"""Validate a fail-closed Five-Benchmark Program readiness receipt.

This validator deliberately separates a complete Joint Development Baseline
from a prospectively frozen paper row.  Missing assets are readiness failures,
not malformed-receipt errors; inconsistent identities and false claims are
rejected.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radio_gs.universal_field_v1 import (
    UNIVERSAL_FIELD_ID,
    validate_universal_field_authority,
    validate_universal_field_payload,
)
from radio_gs.five_benchmark_method_v1 import validate_method_authority
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


SCHEMA = "radio_gs.five_benchmark_joint_evaluation_receipt.v1"
TASKS = ("lerf2d", "lerf3d", "scannet_ovs", "nvos", "spin_nerf_available9")
EXPECTED_PHYSICAL_FIELDS = 29
EXPECTED_TASK_FIELD_REFERENCES = 33
PAPER_SEEDS = [0, 1, 2]


class JointReceiptError(ValueError):
    """Raised when a receipt is internally inconsistent or overclaims."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointReceiptError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _gate(status: bool, finding: str) -> dict[str, Any]:
    return {"status": "pass" if status else "fail", "finding": finding}


def _all_pass(gates: Mapping[str, Any]) -> bool:
    return all(
        isinstance(value, Mapping) and value.get("status") == "pass"
        for value in gates.values()
    )


def _verify_authorities(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    authorities = _mapping(receipt.get("authorities"), "authorities")
    universal_record = authorities.get("universal_field")
    universal_path = validate_file_record(
        universal_record, label="Universal Field authority"
    )
    universal, universal_sha, _ = load_json_object(
        universal_path,
        expected_sha256=str(_mapping(universal_record, "Universal authority")["sha256"]),
        label="Universal Field authority",
    )
    validate_universal_field_authority(universal)
    _require(
        universal.get("universal_field_id") == UNIVERSAL_FIELD_ID,
        "Universal Field identity differs",
    )
    construction_record = authorities.get("construction")
    construction_path = validate_file_record(
        construction_record, label="construction authority"
    )
    construction, _, _ = load_json_object(
        construction_path,
        expected_sha256=str(_mapping(construction_record, "construction authority")["sha256"]),
        label="construction authority",
    )
    validate_method_authority(construction)
    validate_file_record(authorities.get("evaluation_freeze"), label="evaluation freeze")
    inventory_record = receipt.get("source_inventory")
    inventory_path = validate_file_record(
        inventory_record, label="Universal Field inventory"
    )
    inventory, _, _ = load_json_object(
        inventory_path,
        expected_sha256=str(_mapping(inventory_record, "source inventory")["sha256"]),
        label="Universal Field inventory",
    )
    _require(
        inventory.get("artifact_type")
        in {
            "radio_gs_universal_field_v1_asset_inventory",
            "radio_gs_universal_field_v1_live_asset_inventory",
        }
        and inventory.get("universal_field_id") == UNIVERSAL_FIELD_ID,
        "source inventory identity differs",
    )
    inventory_authority = _mapping(inventory.get("authority"), "inventory authority")
    _require(
        inventory_authority.get("sha256") == universal_sha,
        "source inventory authority binding differs",
    )
    return universal, universal_sha


def _candidate(receipt: Mapping[str, Any], universal_sha: str) -> Mapping[str, Any]:
    candidate = _mapping(receipt.get("joint_candidate"), "joint_candidate")
    identity = _mapping(candidate.get("identity"), "joint candidate identity")
    _require(
        identity.get("universal_field_id") == UNIVERSAL_FIELD_ID,
        "candidate field identity differs",
    )
    _require(
        identity.get("universal_field_authority_sha256") == universal_sha,
        "candidate Universal Field authority binding differs",
    )
    readouts = _mapping(identity.get("typed_readout_map"), "typed_readout_map")
    _require(set(readouts) == set(TASKS), "typed readout task membership differs")
    expected = canonical_json_sha256(identity)
    _require(candidate.get("candidate_sha256") == expected, "candidate SHA-256 differs")
    return candidate


def _field_gates(
    receipt: Mapping[str, Any], *, verify_field_files: bool
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    deployment = _mapping(receipt.get("deployment_fields"), "deployment_fields")
    instances_raw = deployment.get("instances")
    _require(isinstance(instances_raw, list), "deployment field instances must be a list")
    instances: dict[str, Mapping[str, Any]] = {}
    duplicate = False
    verified = True
    complete = True
    for raw in instances_raw:
        row = _mapping(raw, "deployment field instance")
        instance_id = str(row.get("dataset_instance_id", ""))
        if not instance_id or instance_id in instances:
            duplicate = True
            continue
        instances[instance_id] = row
        complete = complete and row.get("status") == "complete"
        field_record = row.get("universal_field")
        if verify_field_files and row.get("status") == "complete":
            field_path = validate_file_record(
                field_record, label=f"{instance_id} Universal Field"
            )
            payload, _, _ = load_torch_mapping(
                field_path,
                expected_sha256=str(_mapping(field_record, "field record")["sha256"]),
                map_location="cpu",
                label=f"{instance_id} Universal Field",
            )
            validate_universal_field_payload(payload)
        else:
            verified = verified and row.get("content_verified") is True

    task_refs = _mapping(deployment.get("task_references"), "task field references")
    exact_tasks = set(task_refs) == set(TASKS)
    refs: dict[str, list[str]] = {}
    for task in TASKS:
        values = task_refs.get(task, [])
        _require(isinstance(values, list), f"{task} field references must be a list")
        refs[task] = [str(value) for value in values]
    reference_count = sum(map(len, refs.values()))
    references_exist = all(value in instances for values in refs.values() for value in values)
    lerf_shared = refs["lerf2d"] == refs["lerf3d"] and len(refs["lerf2d"]) == 4
    namespace_safe = not (
        set(refs["nvos"]) & set(refs["spin_nerf_available9"])
    )
    gates = {
        "physical_field_accounting": _gate(
            not duplicate and len(instances) == EXPECTED_PHYSICAL_FIELDS,
            f"{len(instances)}/{EXPECTED_PHYSICAL_FIELDS} unique physical fields",
        ),
        "task_reference_accounting": _gate(
            exact_tasks
            and reference_count == EXPECTED_TASK_FIELD_REFERENCES
            and references_exist,
            f"{reference_count}/{EXPECTED_TASK_FIELD_REFERENCES} task field references",
        ),
        "lerf_shared_field_identity": _gate(
            lerf_shared, "LERF2D and LERF3D must share exactly four field instances"
        ),
        "dataset_instance_namespace": _gate(
            namespace_safe,
            "NVOS and SPIn names are distinct dataset instances",
        ),
        "all_fields_complete": _gate(
            complete and len(instances) == EXPECTED_PHYSICAL_FIELDS,
            "every required instance must be a complete Universal Field v1",
        ),
        "field_content_verified": _gate(
            verify_field_files or verified,
            "field bytes and schema must be verified, not inventory-trusted only",
        ),
    }
    return gates, instances


def _task_gates(
    receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    instances: Mapping[str, Mapping[str, Any]],
    task_references: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = _mapping(receipt.get("task_results"), "task_results")
    _require(set(results) == set(TASKS), "task result membership differs")
    candidate_sha = str(candidate["candidate_sha256"])
    readouts = _mapping(
        _mapping(candidate["identity"], "candidate identity").get("typed_readout_map"),
        "candidate readouts",
    )
    development: dict[str, Any] = {}
    paper: dict[str, Any] = {}
    for task in TASKS:
        row = _mapping(results[task], f"{task} result")
        evidence_ok = False
        if row.get("status") == "complete":
            evidence = row.get("evidence")
            evidence_path = validate_file_record(
                evidence, label=f"{task} evaluation evidence"
            )
            evidence_payload, _, _ = load_json_object(
                evidence_path,
                expected_sha256=str(_mapping(evidence, "task evidence")["sha256"]),
                label=f"{task} evaluation evidence",
            )
            receipt_payload = dict(row)
            receipt_payload.pop("evidence", None)
            evidence_ok = (
                evidence_payload == receipt_payload
                and evidence_payload.get("schema_version") == 1
                and evidence_payload.get("artifact_type")
                == "radio_gs_five_benchmark_task_evaluation_receipt"
                and evidence_payload.get("task_id") == task
            )
        scene_order = row.get("scene_order")
        field_bindings = row.get("field_bindings")
        bindings_ok = (
            isinstance(field_bindings, Mapping)
            and isinstance(scene_order, list)
            and scene_order == task_references.get(task)
        )
        if bindings_ok:
            bindings_ok = set(field_bindings) == set(scene_order)
            for instance_id, digest in field_bindings.items():
                instance = instances.get(str(instance_id))
                record = instance.get("universal_field") if instance is not None else None
                bindings_ok = bindings_ok and isinstance(record, Mapping) and record.get("sha256") == digest
        development_flags = {
            "complete_evidence": evidence_ok,
            "same_candidate": row.get("candidate_sha256") == candidate_sha,
            "same_readout": row.get("readout_id") == readouts[task],
            "contract_identified": bool(str(row.get("evaluation_contract_id", ""))),
            "exact_contract": row.get("exact_evaluation_contract") is True,
            "complete_cohort": row.get("complete_frozen_cohort") is True,
            "field_bindings": bindings_ok,
            "evaluator_bound": row.get("evaluator_content_bound") is True,
            "authorized_target_access": row.get("authorized_target_access") is True,
            "prediction_barrier": row.get("prediction_barrier_required") is not True
            or row.get("prediction_barrier_passed") is True,
            "no_within_run_target_metric_selection": row.get(
                "target_metrics_used_for_current_run_selection"
            )
            is False,
        }
        dev_ok = all(development_flags.values())
        development[task] = _gate(dev_ok, ", ".join(
            key for key, value in development_flags.items() if not value
        ) or "complete same-candidate frozen-contract result")

        paper_flags = {
            "development_eligible": dev_ok,
            "prospectively_preregistered": row.get("prospectively_preregistered") is True,
            "prospectively_blind": row.get("prospectively_blind") is True,
            "no_prior_target_metric_selection": row.get(
                "target_metrics_used_for_candidate_selection"
            )
            is False,
            "paper_seed_panel": row.get("stochastic") is not True
            or row.get("seed_panel") == PAPER_SEEDS,
        }
        paper[task] = _gate(
            all(paper_flags.values()),
            ", ".join(key for key, value in paper_flags.items() if not value)
            or "prospectively frozen paper row",
        )
    return development, paper


def validate(
    receipt: Mapping[str, Any], *, verify_field_files: bool = False
) -> dict[str, Any]:
    _require(receipt.get("schema") == SCHEMA, "joint receipt schema differs")
    _require(receipt.get("schema_version") == 1, "joint receipt schema version differs")
    _, universal_sha = _verify_authorities(receipt)
    candidate = _candidate(receipt, universal_sha)
    field_gates, instances = _field_gates(
        receipt, verify_field_files=verify_field_files
    )
    task_references = _mapping(
        _mapping(receipt["deployment_fields"], "deployment_fields").get(
            "task_references"
        ),
        "task field references",
    )
    task_development, task_paper = _task_gates(
        receipt, candidate, instances, task_references
    )
    development_gates = {
        "field_readiness": _gate(_all_pass(field_gates), "all field gates must pass"),
        "five_complete_task_rows": _gate(
            _all_pass(task_development), "all five development task rows must pass"
        ),
        "historical_peak_stitching_forbidden": _gate(
            receipt.get("historical_peak_stitching") is False,
            "historical per-task peak stitching is forbidden",
        ),
    }
    development_eligible = _all_pass(development_gates)
    paper_gates = {
        "joint_development_baseline": _gate(
            development_eligible, "joint development baseline must first be complete"
        ),
        "five_paper_task_rows": _gate(
            _all_pass(task_paper), "all five prospective paper task rows must pass"
        ),
    }
    paper_eligible = _all_pass(paper_gates)
    derived = {
        "field_gates": field_gates,
        "task_development_gates": task_development,
        "task_paper_gates": task_paper,
        "joint_development_gates": development_gates,
        "paper_gates": paper_gates,
        "joint_development_baseline_eligible": development_eligible,
        "paper_row_eligible": paper_eligible,
        "sota_claim_supported": paper_eligible
        and receipt.get("meets_same_contract_sota_target") is True,
    }
    declared = _mapping(receipt.get("eligibility"), "declared eligibility")
    for key in (
        "joint_development_baseline_eligible",
        "paper_row_eligible",
        "sota_claim_supported",
    ):
        _require(declared.get(key) == derived[key], f"declared {key} overclaims")
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt")
    parser.add_argument("--verify-field-files", action="store_true")
    args = parser.parse_args()
    receipt, digest, source = load_json_object(args.receipt, label="joint receipt")
    report = validate(receipt, verify_field_files=args.verify_field_files)
    print(json.dumps({"receipt": str(source), "receipt_sha256": digest, **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
