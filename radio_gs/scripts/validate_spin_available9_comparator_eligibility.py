#!/usr/bin/env python3
"""Validate the fail-closed Available-Nine comparator eligibility audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = (
    REPO_ROOT
    / "paper/artifacts/spin_available9_comparator_eligibility_audit_20260812.json"
)


class AuditError(ValueError):
    """Raised when the audit or one of its evidence bindings has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_yaml(path: Path) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _resolve_repository_evidence(audit: Mapping[str, Any]) -> dict[str, bytes]:
    resolved: dict[str, bytes] = {}
    rows = audit.get("repository_evidence")
    _require(isinstance(rows, list) and rows, "repository_evidence must be non-empty")
    for raw in rows:
        row = _mapping(raw, "repository evidence row")
        evidence_id = str(row.get("id", ""))
        _require(evidence_id and evidence_id not in resolved, "evidence ids must be unique")
        if "path" in row:
            path = (REPO_ROOT / str(row["path"])).resolve()
            _require(path.is_file(), f"missing evidence: {path}")
            payload = path.read_bytes()
        else:
            commit = str(row.get("git_commit", ""))
            git_path = str(row.get("git_path", ""))
            blob = str(row.get("git_blob", ""))
            _require(len(commit) == 40 and git_path and len(blob) == 40, "invalid git evidence")
            actual_blob = subprocess.run(
                ["git", "rev-parse", f"{commit}:{git_path}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            _require(actual_blob == blob, f"git blob drifted: {evidence_id}")
            payload = subprocess.run(
                ["git", "show", f"{commit}:{git_path}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        _require(_sha256_bytes(payload) == row.get("sha256"), f"SHA-256 drift: {evidence_id}")
        resolved[evidence_id] = payload
    return resolved


def _validate_contract(audit: Mapping[str, Any]) -> None:
    scope = _mapping(audit.get("scope"), "scope")
    _require(
        scope.get("evaluation_contract_id") == "spin-full-mask-field-only-available-nine-v1",
        "wrong Evaluation Contract",
    )
    contract = _mapping(audit.get("contract"), "contract")
    _require(
        contract.get("cohort")
        == ["orchids", "leaves", "fern", "room", "horns", "fortress", "pinecone", "truck", "lego"],
        "Available-Nine cohort/order drifted",
    )
    _require("no captured or rendered RGB" in str(contract.get("query_boundary")), "RGB-free query boundary absent")
    _require("non-reference" in str(contract.get("scored_frames")), "scored-frame rule absent")
    _require(
        contract.get("aggregation")
        == "unweighted frame mean within each scene, then unweighted macro over all nine scenes",
        "aggregation drifted",
    )
    _require(contract.get("required_proof") == "runtime-compliance-proof-v1 PASS over the complete Exact Row Authority", "proof gate drifted")
    _require(contract.get("comparator_role") == "external comparator rather than the current RADIO-GS method", "comparator role drifted")
    _require(contract.get("fail_closed") is True, "contract is not fail closed")


def _validate_bound_facts(evidence: Mapping[str, bytes]) -> None:
    registry = _mapping(yaml.safe_load(evidence["promptable_nvs_protocol_registry"]), "registry")
    local9 = _mapping(
        _mapping(registry["protocols"]["ludvig_spin_horns_released_all_view_exact_3seed_v1"], "LUDVIG row")["current_nine_scene_diagnostic"],
        "LUDVIG local9",
    )
    _require(math.isclose(float(local9["local_scene_macro_iou_percent"]), 93.7200449592385), "LUDVIG local9 metric drifted")
    _require(math.isclose(float(local9["paper_same_scene_macro_iou_percent"]), 94.57777777777778), "paper matching-nine metric drifted")

    ludvig = _mapping(json.loads(evidence["ludvig_sam_query_interface_audit"]), "LUDVIG audit")
    rgb_access = str(_mapping(ludvig["sam_candidate_generation"], "SAM generation")["rgb_access"])
    _require("evaluation/target RGB" in rgb_access, "LUDVIG target-RGB evidence absent")
    mapping = _mapping(ludvig["candidate_to_gaussian_mapping"], "candidate mapping")
    _require(mapping.get("query_independent_canonical_field") is False, "LUDVIG unexpectedly became query independent")
    _require(mapping.get("target_view_independent") is False, "LUDVIG unexpectedly became target-view independent")

    current = _mapping(json.loads(evidence["radio_gs_reference_selected_result"]), "current RADIO-GS result")
    metrics = _mapping(current["macro_foreground_iou"], "current metrics")
    _require(math.isclose(float(metrics["previous_canonical_mainline"]), 0.8771615046217027), "RADIO-GS current metric drifted")
    _require(math.isclose(float(metrics["reproduced_ludvig_sam"]), 0.9372004495923849), "reproduced LUDVIG metric drifted")

    legacy = _mapping(json.loads(evidence["radio_gs_legacy_exact_local9_authority"]), "legacy authority")
    claim_scope = _mapping(legacy["claim_scope"], "legacy claim scope")
    _require(claim_scope.get("cohort") == "local9_full_reference_mask_diagnostic", "legacy authority scope drifted")
    _require(math.isclose(float(_mapping(legacy["metrics"], "legacy metrics")["foreground_iou"]), 0.6252262843520126), "legacy metric drifted")
    _require("runtime_compliance_proof" not in legacy, "legacy authority unexpectedly claims a Runtime Compliance Proof")


def validate(audit_path: Path = DEFAULT_AUDIT) -> dict[str, Any]:
    audit = _read_json(audit_path)
    _require(audit.get("schema") == "radio_gs.spin_available9_comparator_eligibility_audit.v1", "wrong audit schema")
    _require(audit.get("schema_version") == 1, "wrong audit schema version")
    _require(audit.get("status") == "complete_no_eligible_existing_comparator", "audit is incomplete")
    _validate_contract(audit)
    evidence = _resolve_repository_evidence(audit)
    _validate_bound_facts(evidence)

    expected_count = int(_mapping(audit.get("reproducible_checks"), "checks")["expected_candidate_count"])
    candidates = audit.get("candidate_audit")
    _require(isinstance(candidates, list) and len(candidates) == expected_count, "candidate inventory is incomplete")
    ids: set[str] = set()
    for raw in candidates:
        candidate = _mapping(raw, "candidate")
        candidate_id = str(candidate.get("candidate_id", ""))
        _require(candidate_id and candidate_id not in ids, "candidate ids must be unique")
        ids.add(candidate_id)
        failures = candidate.get("blocking_failures")
        _require(isinstance(failures, list) and failures, f"{candidate_id}: no blocking failure")
        _require(candidate.get("eligible") is False, f"{candidate_id}: eligibility is not fail closed")

    resolution = _mapping(audit.get("resolution"), "resolution")
    _require(resolution.get("verdict") == "no_eligible_existing_comparator", "verdict drifted")
    _require(resolution.get("numeric_sota_target") is None, "numeric target must remain absent")
    _require(resolution.get("target_status") == "no_eligible_target", "target status drifted")
    _require(resolution.get("metadata_only_retrofit_permitted") is False, "metadata retrofit was enabled")
    _require(resolution.get("historical_metric_reuse_permitted") is False, "historical metric reuse was enabled")
    return {
        "audit": str(audit_path.resolve()),
        "audit_sha256": _sha256_bytes(audit_path.read_bytes()),
        "candidate_count": len(candidates),
        "evidence_count": len(evidence),
        "numeric_sota_target": None,
        "verdict": "no_eligible_existing_comparator",
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
