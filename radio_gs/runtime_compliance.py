"""Independent verification for one runtime-compliance-proof-v1 row.

This module is the audit boundary between the producing lifecycle and an
authority row.  Producers may submit declarations, Stage Receipts, lineage,
and runtime observations, but this verifier recomputes the identities and
awards compliance only after the complete row closes.  In particular, a
missing observation is not treated as an empty observation and a private
target cannot be made legal by moving or renaming it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radio_gs.candidate_authority import (
    CandidateAuthorityBundle,
    validate_candidate_authority,
)
from radio_gs.stage_receipts import (
    STAGE_ORDER,
    StageReceiptError,
    directory_merkle,
    validate_stage_receipt,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_bytes,
    load_json_object,
    stable_descriptor_load,
    write_frozen_json,
)


RUNTIME_COMPLIANCE_AUDIT_SCHEMA = "radio_gs.runtime_compliance_audit.v1"
RUNTIME_COMPLIANCE_PROOF_SCHEMA = "runtime-compliance-proof-v1"
RUNTIME_OBSERVATION_SCHEMA = "radio_gs.runtime_observation.v1"
EVIDENCE_NODE_SCHEMA = "radio_gs.evidence_node.v1"
LINEAGE_EDGE_SCHEMA = "radio_gs.lineage_edge.v1"
ACTIVITY_SCHEMA = "radio_gs.activity_record.v1"
ROW_IDENTITY_SCHEMA = "radio_gs.row_identity.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRACE_CHANNELS = (
    "file",
    "metadata",
    "mmap",
    "executable",
    "library",
    "model",
    "descendant",
    "ipc",
    "shared_memory",
    "inherited_descriptor",
    "network",
)
_EDGE_TYPES = {"read", "derive", "emit", "evaluate"}
_ACTIVITY_KINDS = {
    "executable",
    "file",
    "metadata",
    "mmap",
    "library",
    "model",
    "descendant",
}
_DISALLOWED_ACTIVITY_KINDS = {"ipc", "shared_memory", "inherited_descriptor", "network"}
_KNOWN_GRANTS = {
    "method_contract",
    "evaluation_contract",
    "mapping_observation",
    "global_method_parameters",
    "deployment_scene_state",
    "warm_cache",
    "authorized_query_input",
    "output_request_metadata",
    "query_workspace",
    "prediction",
    "evaluator_private_target",
    "evaluator_result",
}
_FORBIDDEN_TOKENS = (
    "capability_bank",
    "descriptor",
    "query_score",
    "score_cache",
    "teacher",
    "mpr",
    "ground_truth",
    "ground-truth",
    "target_mask",
    "private_target",
    "labels",
    "metric",
    "rendered_rgb",
)
_ALLOWED_ROOT_GRANTS = {
    "method_contract",
    "evaluation_contract",
    "mapping_observation",
    "global_method_parameters",
    "authorized_query_input",
    "output_request_metadata",
    "evaluator_private_target",
}
_REQUIRED_STORAGE_KEYS = {
    "schema",
    "field_family",
    "local_code_dimension",
    "persistent_semantic_fields",
    "deployment_support_state",
    "scene_gaussian_count",
    "persistent_scene_storage_increment_bytes",
    "scene_soft_target_bytes",
    "scene_hard_limit_bytes",
    "method_specific_global_bytes",
    "method_specific_global_soft_target_bytes",
    "method_specific_global_hard_limit_bytes",
    "serialization_overhead_bytes",
    "forbidden_member_types",
    "cold_start_query_executed",
    "warm_cache_rebuilds_bitwise_identical",
}
_REQUIRED_EXECUTION_KEYS = {
    "schema",
    "schema_version",
    "auditor_started_before_first_instruction",
    "process_tree_complete",
    "network_disabled",
    "network_attempts",
    "inherited_descriptors_cleared",
    "shared_memory_cleared",
    "trace_channels",
    "unknown_activity",
    "allowlist_mounts",
    "stage_workspaces",
    "process_tree",
    "stage_executions",
    "declared_activity",
    "observed_activity",
}
_CONTENT_KINDS = {
    "opaque_file",
    "directory_merkle",
    "canonical_manifest",
    "tensor_container",
}


class RuntimeComplianceError(ValueError):
    """Raised by strict helper constructors for malformed audit records."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _digest(value: Any) -> str:
    try:
        return hashlib.sha256(canonical_json_bytes(_plain(value))).hexdigest()
    except (TypeError, ValueError) as error:
        raise RuntimeComplianceError("value is not finite canonical JSON") from error


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeComplianceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeComplianceError(f"{label} must be a non-empty string")
    return value


def _node_body(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _plain(value)
        for key, value in node.items()
        if key != "node_id"
    }


def _edge_body(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _plain(value)
        for key, value in edge.items()
        if key != "edge_id"
    }


def _activity_body(activity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _plain(value)
        for key, value in activity.items()
        if key != "activity_id"
    }


def row_identity(contract_id: str, execution_ids: Sequence[str]) -> str:
    """Return the content identity for one complete authority-row cohort."""

    _require_string(contract_id, label="contract_id")
    ids = list(execution_ids)
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise RuntimeComplianceError("execution_ids must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise RuntimeComplianceError("execution_ids must be unique")
    return _digest(
        {
            "schema": ROW_IDENTITY_SCHEMA,
            "contract_id": contract_id,
            "execution_ids": ids,
        }
    )


def evidence_node(
    *,
    content_type: str,
    lifecycle_class: str,
    content_sha256: str,
    size_bytes: int,
    schema_identity: str,
    locator: str,
    information_grant: str,
    stage: str,
    row_id: str,
    contract_id: str,
    content_kind: str = "opaque_file",
    query_id: str | None = None,
    producer_receipt: Mapping[str, Any] | None = None,
    logical_bytes: int = 0,
    members: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create one typed Evidence Lineage node with a stable node identity."""

    _require_string(content_type, label="content_type")
    _require_string(lifecycle_class, label="lifecycle_class")
    _require_sha256(content_sha256, label="content_sha256")
    _require_string(schema_identity, label="schema_identity")
    _require_string(locator, label="locator")
    _require_string(information_grant, label="information_grant")
    _require_string(stage, label="stage")
    _require_sha256(row_id, label="row_id")
    _require_string(contract_id, label="contract_id")
    if content_kind not in _CONTENT_KINDS:
        raise RuntimeComplianceError("content_kind differs")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise RuntimeComplianceError("size_bytes must be a non-negative integer")
    if not isinstance(logical_bytes, int) or isinstance(logical_bytes, bool) or logical_bytes < 0:
        raise RuntimeComplianceError("logical_bytes must be a non-negative integer")
    if query_id is not None:
        _require_string(query_id, label="query_id")
    producer = None if producer_receipt is None else _plain(producer_receipt)
    if producer is not None:
        if set(producer) != {"stage", "receipt_id"}:
            raise RuntimeComplianceError("producer_receipt fields differ")
        _require_string(producer["stage"], label="producer_receipt.stage")
        _require_sha256(producer["receipt_id"], label="producer_receipt.receipt_id")
    member_list = [] if members is None else [_plain(member) for member in members]
    body = {
        "schema": EVIDENCE_NODE_SCHEMA,
        "schema_version": 1,
        "content_type": content_type,
        "lifecycle_class": lifecycle_class,
        "content_identity": {
            "sha256": content_sha256,
            "size_bytes": size_bytes,
        },
        "schema_identity": schema_identity,
        "locator": locator,
        "information_grant": information_grant,
        "stage": stage,
        "row_id": row_id,
        "contract_id": contract_id,
        "content_kind": content_kind,
        "query_id": query_id,
        "producer_receipt": producer,
        "logical_bytes": logical_bytes,
        "members": member_list,
        "root": producer is None,
    }
    return {"node_id": _digest(body), **body}


def lineage_edge(
    *,
    source: str,
    target: str,
    edge_type: str,
    stage: str,
    purpose: str,
    information_grant: str,
) -> dict[str, Any]:
    """Create one typed read/derive/emit/evaluate Evidence Lineage edge."""

    for value, label in (
        (source, "source"),
        (target, "target"),
        (stage, "stage"),
        (purpose, "purpose"),
        (information_grant, "information_grant"),
    ):
        _require_string(value, label=label)
    if edge_type not in _EDGE_TYPES:
        raise RuntimeComplianceError("edge_type is not part of the typed lineage schema")
    body = {
        "schema": LINEAGE_EDGE_SCHEMA,
        "schema_version": 1,
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "stage": stage,
        "purpose": purpose,
        "information_grant": information_grant,
    }
    return {"edge_id": _digest(body), **body}


def activity_record(
    *,
    kind: str,
    stage: str,
    process_id: str,
    operation: str,
    node_id: str,
    identity_sha256: str,
) -> dict[str, Any]:
    """Create one declared or observed runtime activity record."""

    for value, label in (
        (kind, "kind"),
        (stage, "stage"),
        (process_id, "process_id"),
        (operation, "operation"),
        (node_id, "node_id"),
    ):
        _require_string(value, label=label)
    _require_sha256(identity_sha256, label="identity_sha256")
    body = {
        "schema": ACTIVITY_SCHEMA,
        "schema_version": 1,
        "kind": kind,
        "stage": stage,
        "process_id": process_id,
        "operation": operation,
        "node_id": node_id,
        "identity_sha256": identity_sha256,
    }
    return {"activity_id": _digest(body), **body}


def _file_identity(path: str | Path) -> dict[str, Any]:
    def size(handle: Any) -> int:
        return int(os.fstat(handle.fileno()).st_size)

    try:
        value, digest, _source = stable_descriptor_load(
            path,
            size,
            label="lineage node artifact",
        )
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeComplianceError(f"cannot read lineage artifact {path}: {error}") from error
    return {"sha256": digest, "size_bytes": int(value)}


def _artifact_identity(path: str | Path, content_kind: str) -> dict[str, Any]:
    """Recompute one supported content identity without trusting its locator."""

    if content_kind not in _CONTENT_KINDS:
        raise RuntimeComplianceError("lineage content kind is unknown")

    if content_kind == "canonical_manifest":
        try:
            value, _raw_digest, _source = load_json_object(
                path,
                label="lineage canonical manifest",
            )
            canonical = canonical_json_bytes(value)
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeComplianceError(
                f"cannot read lineage manifest {path}: {error}"
            ) from error
        return {
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "size_bytes": len(canonical),
        }

    if content_kind != "directory_merkle":
        return _file_identity(path)
    try:
        manifest = directory_merkle(path)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeComplianceError(
            f"cannot read lineage directory {path}: {error}"
        ) from error
    return {
        "sha256": manifest["merkle_root_sha256"],
        "size_bytes": sum(entry["size_bytes"] for entry in manifest["entries"]),
    }


def _receipt_artifacts(receipt: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield every typed artifact exported by one validated Stage Receipt."""

    for section_name in ("inputs", "outputs"):
        section = receipt.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for artifact in section.values():
            if isinstance(artifact, Mapping):
                yield artifact


def _tensor_member_signature(member: Mapping[str, Any]) -> tuple[Any, ...]:
    """Project a logical tensor member onto identity-bearing fields."""

    return (
        member.get("name"),
        member.get("dtype"),
        tuple(member.get("shape", ()))
        if isinstance(member.get("shape"), list)
        else member.get("shape"),
        member.get("sha256"),
    )


def _member_forbidden(member: Mapping[str, Any]) -> bool:
    values = (
        str(member.get("name", "")).lower(),
        str(member.get("content_type", "")).lower(),
    )
    return any(token in value for value in values for token in _FORBIDDEN_TOKENS)


def _node_forbidden(node: Mapping[str, Any]) -> bool:
    values = (
        str(node.get("content_type", "")).lower(),
        str(node.get("lifecycle_class", "")).lower(),
        str(node.get("information_grant", "")).lower(),
    )
    if node.get("information_grant") == "evaluator_private_target":
        return True
    return any(token in value for value in values for token in _FORBIDDEN_TOKENS)


def _authorized_private_target(node: Mapping[str, Any]) -> bool:
    """Return whether a private target is legal at the evaluation boundary."""

    return (
        node.get("stage") == "evaluation"
        and node.get("information_grant") == "evaluator_private_target"
        and node.get("lifecycle_class") == "evaluator_private_target"
    )


def _canonical_activity_key(value: Any) -> str:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeComplianceError("activity record is not finite canonical JSON") from error


class _Findings:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.incomplete: list[str] = []

    def fail(self, message: str) -> None:
        if message not in self.failures:
            self.failures.append(message)

    def missing(self, message: str) -> None:
        if message not in self.incomplete:
            self.incomplete.append(message)


_VERIFIED_PROOF_TOKEN = object()


class RuntimeComplianceProof(Mapping[str, Any]):
    """Recursively immutable, content-addressed PASS proof.

    The private verification token distinguishes a proof emitted by
    ``RuntimeComplianceVerifier`` from a merely self-consistent mapping read
    from disk.  Persistence accepts only the former; loading still checks the
    serialized content identity but never upgrades trust by itself.
    """

    __slots__ = ("_payload", "_verified")

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        _token: object | None = None,
    ) -> None:
        object.__setattr__(self, "_payload", _freeze(payload))
        object.__setattr__(self, "_verified", _token is _VERIFIED_PROOF_TOKEN)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("RuntimeComplianceProof is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def as_dict(self) -> dict[str, Any]:
        return _plain(self._payload)


_REQUIRED_PROOF_KEYS = {
    "schema",
    "schema_version",
    "status",
    "candidate_id",
    "contract_id",
    "row_id",
    "stage_receipt_ids",
    "lineage_root_ids",
    "lineage_node_ids",
    "lineage_edge_ids",
    "prediction_node_id",
    "evaluator_result_node_id",
    "prediction_merkle_root_sha256",
    "storage_assertion_sha256",
    "environment_identity_sha256",
    "verifier_identity",
    "proof_id",
}


def validate_runtime_compliance_proof(
    proof: Mapping[str, Any] | RuntimeComplianceProof,
) -> RuntimeComplianceProof:
    """Recompute and validate one persisted independent PASS proof."""

    value = proof.as_dict() if isinstance(proof, RuntimeComplianceProof) else _plain(proof)
    if not isinstance(value, Mapping):
        raise ValueError("runtime compliance proof must be a mapping")
    value = dict(value)
    if set(value) != _REQUIRED_PROOF_KEYS:
        raise ValueError("runtime compliance proof fields differ")
    if value["schema"] != RUNTIME_COMPLIANCE_PROOF_SCHEMA or value["schema_version"] != 1:
        raise ValueError("runtime compliance proof schema differs")
    if value["status"] != "PASS":
        raise ValueError("only PASS runtime compliance proofs may be persisted")
    for key in (
        "candidate_id",
        "row_id",
        "prediction_node_id",
        "evaluator_result_node_id",
        "prediction_merkle_root_sha256",
        "storage_assertion_sha256",
        "environment_identity_sha256",
        "proof_id",
    ):
        _require_sha256(value[key], label=key)
    for key in ("contract_id", "verifier_identity"):
        _require_string(value[key], label=key)
    for key in ("stage_receipt_ids", "lineage_root_ids", "lineage_node_ids", "lineage_edge_ids"):
        values = value[key]
        if not isinstance(values, list) or not values or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in values
        ) or len(values) != len(set(values)):
            raise ValueError(f"runtime compliance proof {key} is incomplete")
        if key != "stage_receipt_ids" and values != sorted(values):
            raise ValueError(f"runtime compliance proof {key} is not canonical")
    if len(value["stage_receipt_ids"]) != len(STAGE_ORDER):
        raise ValueError("runtime compliance proof Stage Receipt chain is incomplete")
    body = {key: item for key, item in value.items() if key != "proof_id"}
    expected = _digest(body)
    if value["proof_id"] != expected:
        raise ValueError("runtime compliance proof content identity differs")
    verified = (
        isinstance(proof, RuntimeComplianceProof)
        and proof._verified
    )
    return RuntimeComplianceProof(
        value,
        _token=_VERIFIED_PROOF_TOKEN if verified else None,
    )


def write_runtime_compliance_proof(
    path: str | Path,
    proof: Mapping[str, Any] | RuntimeComplianceProof,
) -> Path:
    """Publish one PASS proof without replacing a different proof."""

    if not isinstance(proof, RuntimeComplianceProof) or not proof._verified:
        raise ValueError("runtime compliance proof must come from the independent verifier")
    validated = validate_runtime_compliance_proof(proof)
    return write_frozen_json(path, validated.as_dict())


def load_runtime_compliance_proof(path: str | Path) -> RuntimeComplianceProof:
    """Load a proof through the immutable artifact and content-identity seam."""

    try:
        payload, _digest_value, _source = load_json_object(
            path,
            label="runtime compliance proof",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot load runtime compliance proof: {error}") from error
    return validate_runtime_compliance_proof(payload)


class RuntimeComplianceVerifier:
    """Recompute one row-wide Runtime Compliance Proof independently."""

    verifier_identity = "radio_gs.runtime_compliance_verifier.v1"

    def verify(
        self,
        audit: Mapping[str, Any],
        candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle,
    ) -> dict[str, Any]:
        findings = _Findings()
        audit_value = _plain(audit) if isinstance(audit, Mapping) else audit
        candidate: CandidateAuthorityBundle | None = None
        try:
            candidate = validate_candidate_authority(candidate_authority)
        except (TypeError, ValueError) as error:
            findings.fail(f"candidate authority is invalid: {error}")

        if not isinstance(audit_value, Mapping):
            findings.missing("runtime compliance audit is missing")
            return self._report(findings, None, None)

        audit_map = dict(audit_value)
        try:
            self._validate_audit_header(audit_map, candidate, findings)
            receipts = self._validate_receipts(audit_map, candidate, findings)
            contract = self._contract(audit_map, candidate, findings)
            nodes, edges, roots = self._validate_lineage(
                audit_map,
                contract,
                receipts,
                findings,
            )
            row = self._validate_row(audit_map, nodes, receipts, findings)
            if row is not None:
                self._validate_prediction_binding(
                    audit_map,
                    contract,
                    nodes,
                    edges,
                    receipts,
                    row,
                    findings,
                )
            self._validate_storage(audit_map, nodes, candidate, findings)
            self._validate_environment(audit_map, receipts, candidate, findings)
            self._validate_execution(audit_map, nodes, roots, receipts, findings)
        except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as error:
            findings.fail(f"runtime compliance verifier rejected malformed evidence: {error}")
            row = None

        if findings.failures:
            status = "FAIL"
        elif findings.incomplete:
            status = "INCOMPLETE"
        else:
            status = "PASS"

        report: dict[str, Any] = {
            "schema": RUNTIME_COMPLIANCE_AUDIT_SCHEMA,
            "status": status,
            "failures": sorted(findings.failures + findings.incomplete),
            "verifier_identity": self.verifier_identity,
        }
        if status == "PASS" and candidate is not None and row is not None:
            try:
                report["proof"] = self._proof(
                    audit_map,
                    candidate,
                    nodes,
                    edges,
                    roots,
                    receipts,
                    row,
                )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                findings.fail(f"runtime compliance proof could not be sealed: {error}")
                report["status"] = "FAIL"
                report["failures"] = sorted(findings.failures + findings.incomplete)
        return report

    def _report(
        self,
        findings: _Findings,
        _audit: Mapping[str, Any] | None,
        _candidate: CandidateAuthorityBundle | None,
    ) -> dict[str, Any]:
        status = "FAIL" if findings.failures else "INCOMPLETE"
        return {
            "schema": RUNTIME_COMPLIANCE_AUDIT_SCHEMA,
            "status": status,
            "failures": sorted(findings.failures + findings.incomplete),
            "verifier_identity": self.verifier_identity,
        }

    @staticmethod
    def _validate_audit_header(
        audit: Mapping[str, Any],
        candidate: CandidateAuthorityBundle | None,
        findings: _Findings,
    ) -> None:
        required = {
            "schema",
            "schema_version",
            "candidate_id",
            "contract_id",
            "producer_identity",
            "row",
            "stage_receipts",
            "lineage",
            "execution",
            "environment",
            "storage",
        }
        if set(audit) != required:
            findings.missing("runtime compliance audit fields are incomplete")
        if audit.get("schema") != RUNTIME_COMPLIANCE_AUDIT_SCHEMA:
            findings.fail("runtime compliance audit schema differs")
        if audit.get("schema_version") != 1:
            findings.fail("runtime compliance audit version differs")
        try:
            _require_string(audit.get("producer_identity"), label="producer_identity")
        except RuntimeComplianceError as error:
            findings.missing(str(error))
        if audit.get("producer_identity") == RuntimeComplianceVerifier.verifier_identity:
            findings.fail("producer and independent verifier identities are not separated")
        if candidate is not None:
            if audit.get("candidate_id") != candidate["candidate_id"]:
                findings.fail("audit candidate identity differs")
            contract_ids = {
                contract["contract_id"] for contract in candidate["evaluation_contracts"]
            }
            if audit.get("contract_id") not in contract_ids:
                findings.fail("audit contract identity is not in the Candidate Authority Bundle")
        else:
            findings.missing("candidate authority identity is unavailable")

    @staticmethod
    def _validate_receipts(
        audit: Mapping[str, Any],
        candidate: CandidateAuthorityBundle | None,
        findings: _Findings,
    ) -> list[dict[str, Any]]:
        value = audit.get("stage_receipts")
        if not isinstance(value, list) or len(value) != len(STAGE_ORDER):
            findings.missing("stage receipt chain is incomplete")
            return []
        if candidate is None:
            findings.missing("stage receipts cannot be attributed without a candidate")
            return []
        receipts: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        for index, raw in enumerate(value):
            try:
                receipt = validate_stage_receipt(raw, candidate)
                mapping = receipt.as_dict()
            except (StageReceiptError, TypeError, ValueError) as error:
                findings.fail(f"stage receipt {index} is invalid: {error}")
                continue
            if mapping["stage"] != STAGE_ORDER[index]:
                findings.fail("stage receipt order differs")
            if previous is None:
                expected_predecessor = None
            else:
                expected_predecessor = {
                    "stage": previous["stage"],
                    "stage_index": previous["stage_index"],
                    "receipt_id": previous["receipt_id"],
                }
            if mapping["predecessor"] != expected_predecessor:
                findings.fail("stage receipt predecessor chain differs")
            receipts.append(mapping)
            previous = mapping
        if len(receipts) != len(STAGE_ORDER):
            findings.missing("not every Stage Receipt could be independently validated")
        return receipts

    @staticmethod
    def _contract(
        audit: Mapping[str, Any],
        candidate: CandidateAuthorityBundle | None,
        findings: _Findings,
    ) -> Mapping[str, Any] | None:
        if candidate is None:
            return None
        contract_id = audit.get("contract_id")
        for contract in candidate["evaluation_contracts"]:
            if contract["contract_id"] == contract_id:
                return contract
        findings.fail("evaluation contract is not bound by the candidate")
        return None

    def _validate_lineage(
        self,
        audit: Mapping[str, Any],
        contract: Mapping[str, Any] | None,
        receipts: Sequence[Mapping[str, Any]],
        findings: _Findings,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
        lineage = audit.get("lineage")
        if not isinstance(lineage, Mapping):
            findings.missing("Evidence Lineage is missing")
            return {}, {}, set()
        if set(lineage) != {"nodes", "edges"}:
            findings.missing("Evidence Lineage nodes or edges are incomplete")
        raw_nodes = lineage.get("nodes")
        raw_edges = lineage.get("edges")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            findings.missing("Evidence Lineage nodes are incomplete")
            return {}, {}, set()
        nodes: dict[str, dict[str, Any]] = {}
        receipt_by_id = {receipt.get("receipt_id"): receipt for receipt in receipts}
        expected_contract_id = audit.get("contract_id")
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, Mapping):
                findings.missing(f"lineage node {index} is not typed")
                continue
            node = _plain(raw)
            node_id = node.get("node_id")
            if not isinstance(node_id, str) or _SHA256.fullmatch(node_id) is None:
                findings.fail(f"lineage node {index} has no valid node identity")
                continue
            if node_id in nodes:
                findings.fail(f"lineage node identity is duplicated: {node_id}")
                continue
            nodes[node_id] = node
            row = audit.get("row")
            if isinstance(row, Mapping) and node.get("row_id") != row.get("row_id"):
                findings.fail(f"lineage node {index} row identity differs")
            self._validate_node(
                node,
                index,
                expected_contract_id,
                receipt_by_id,
                contract,
                findings,
            )

        if not isinstance(raw_edges, list) or not raw_edges:
            findings.missing("Evidence Lineage edges are incomplete")
            return nodes, {}, {node_id for node_id, node in nodes.items() if node.get("root") is True}
        edges: dict[str, dict[str, Any]] = {}
        incoming: dict[str, list[str]] = defaultdict(list)
        adjacency: dict[str, list[str]] = defaultdict(list)
        for index, raw in enumerate(raw_edges):
            if not isinstance(raw, Mapping):
                findings.missing(f"lineage edge {index} is not typed")
                continue
            edge = _plain(raw)
            edge_id = edge.get("edge_id")
            if not isinstance(edge_id, str) or _SHA256.fullmatch(edge_id) is None:
                findings.fail(f"lineage edge {index} has no valid edge identity")
                continue
            if edge_id in edges:
                findings.fail(f"lineage edge identity is duplicated: {edge_id}")
                continue
            edges[edge_id] = edge
            self._validate_edge(edge, index, nodes, findings)
            source = edge.get("source")
            target = edge.get("target")
            if source in nodes and target in nodes:
                if source == target:
                    findings.fail("Evidence Lineage contains a self-cycle")
                adjacency[source].append(target)
                incoming[target].append(source)

        roots = {node_id for node_id, node in nodes.items() if node.get("root") is True}
        if not roots:
            findings.missing("Evidence Lineage has no authorized roots")
        for node_id, node in nodes.items():
            producer = node.get("producer_receipt")
            if producer is None and node.get("root") is not True:
                findings.fail(f"lineage node {node_id} has an unbound root status")
            if producer is not None and not incoming.get(node_id):
                findings.missing(f"lineage node {node_id} has no incoming derivation")
            if node.get("root") is True and incoming.get(node_id):
                findings.fail(f"lineage root {node_id} has an ancestor edge")
            if node.get("root") is True and node.get("information_grant") not in _ALLOWED_ROOT_GRANTS:
                findings.fail(f"lineage root {node_id} has no Information Grant")

        reachable: set[str] = set()
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            queue.extend(adjacency.get(current, ()))
        for node_id in nodes:
            if node_id not in reachable:
                findings.missing(f"lineage node {node_id} is not closed to an authorized root")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                findings.fail("Evidence Lineage contains a cycle")
                return
            if node_id in visited:
                return
            visiting.add(node_id)
            for child in adjacency.get(node_id, ()):
                visit(child)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in nodes:
            visit(node_id)

        self._validate_direct_stage_closure(nodes, incoming, contract, findings)
        return nodes, edges, roots

    @staticmethod
    def _validate_node(
        node: Mapping[str, Any],
        index: int,
        contract_id: Any,
        receipts: Mapping[str, Mapping[str, Any]],
        contract: Mapping[str, Any] | None,
        findings: _Findings,
    ) -> None:
        required = {
            "schema",
            "schema_version",
            "node_id",
            "content_type",
            "lifecycle_class",
            "content_identity",
            "schema_identity",
            "locator",
            "information_grant",
            "stage",
            "row_id",
            "contract_id",
            "content_kind",
            "query_id",
            "producer_receipt",
            "logical_bytes",
            "members",
            "root",
        }
        if set(node) != required:
            findings.missing(f"lineage node {index} fields are incomplete")
        if node.get("schema") != EVIDENCE_NODE_SCHEMA or node.get("schema_version") != 1:
            findings.fail(f"lineage node {index} schema differs")
        if node.get("contract_id") != contract_id:
            findings.fail(f"lineage node {index} contract identity differs")
        if node.get("stage") not in STAGE_ORDER:
            findings.fail(f"lineage node {index} stage differs")
        if node.get("content_kind") not in _CONTENT_KINDS:
            findings.fail(f"lineage node {index} content kind differs")
        if node.get("information_grant") not in _KNOWN_GRANTS:
            findings.fail(f"lineage node {index} Information Grant is unknown")
        if not isinstance(node.get("root"), bool):
            findings.fail(f"lineage node {index} root status is not boolean")
        content_type = str(node.get("content_type", "")).lower()
        if _node_forbidden(node) and not _authorized_private_target(node):
            findings.fail(f"lineage node {index} has a forbidden artifact type")
        logical_bytes = node.get("logical_bytes")
        if (
            not isinstance(logical_bytes, int)
            or isinstance(logical_bytes, bool)
            or logical_bytes < 0
        ):
            findings.fail(f"lineage node {index} logical byte inventory is invalid")
        try:
            _require_string(node.get("content_type"), label=f"lineage node {index}.content_type")
            _require_string(node.get("lifecycle_class"), label=f"lineage node {index}.lifecycle_class")
            _require_string(node.get("schema_identity"), label=f"lineage node {index}.schema_identity")
            _require_string(node.get("locator"), label=f"lineage node {index}.locator")
            _require_sha256(node.get("row_id"), label=f"lineage node {index}.row_id")
            if node.get("query_id") is not None:
                _require_string(node["query_id"], label=f"lineage node {index}.query_id")
        except RuntimeComplianceError as error:
            findings.fail(str(error))

        content_identity = node.get("content_identity")
        if not isinstance(content_identity, Mapping) or set(content_identity) != {"sha256", "size_bytes"}:
            findings.missing(f"lineage node {index} content identity is incomplete")
        else:
            try:
                _require_sha256(content_identity["sha256"], label=f"lineage node {index}.content_identity.sha256")
                if not isinstance(content_identity["size_bytes"], int) or isinstance(content_identity["size_bytes"], bool) or content_identity["size_bytes"] < 0:
                    raise RuntimeComplianceError("size_bytes is invalid")
                locator = node.get("locator")
                content_kind = node.get("content_kind")
                if isinstance(locator, str) and content_kind in _CONTENT_KINDS:
                    actual = _artifact_identity(locator, content_kind)
                    if actual != dict(content_identity):
                        findings.fail(f"lineage node {index} content identity differs from bytes")
                else:
                    findings.missing(f"lineage node {index} artifact locator or kind is incomplete")
            except RuntimeComplianceError as error:
                findings.missing(str(error))

        if node.get("node_id") != _digest(_node_body(node)):
            findings.fail(f"lineage node {index} content identity is not self-consistent")

        members = node.get("members")
        if not isinstance(members, list):
            findings.missing(f"lineage node {index} structured members are incomplete")
        else:
            names: list[str] = []
            member_bytes = 0
            for member_index, raw_member in enumerate(members):
                if not isinstance(raw_member, Mapping):
                    findings.fail(f"lineage node {index} member {member_index} is malformed")
                    continue
                member = _plain(raw_member)
                required_member = {"name", "content_type", "dtype", "shape", "logical_bytes", "sha256"}
                if set(member) != required_member:
                    findings.missing(f"lineage node {index} member {member_index} is incomplete")
                name = member.get("name")
                if not isinstance(name, str) or not name:
                    findings.fail(f"lineage node {index} member name is invalid")
                else:
                    names.append(name)
                try:
                    _require_string(member.get("content_type"), label="structured member content_type")
                    _require_string(member.get("dtype"), label="structured member dtype")
                    _require_sha256(member.get("sha256"), label="structured member sha256")
                    shape = member.get("shape")
                    if not isinstance(shape, list) or any(
                        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                        for dim in shape
                    ):
                        raise RuntimeComplianceError("structured member shape is invalid")
                    value = member.get("logical_bytes")
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise RuntimeComplianceError("structured member logical bytes are invalid")
                    member_bytes += value
                except RuntimeComplianceError as error:
                    findings.fail(f"lineage node {index} member is invalid: {error}")
                if _member_forbidden(member):
                    findings.fail(f"lineage node {index} contains forbidden member {name}")
            if names != sorted(names) or len(names) != len(set(names)):
                findings.fail(f"lineage node {index} members are not canonically ordered")
            if isinstance(logical_bytes, int) and logical_bytes != member_bytes:
                findings.fail(f"lineage node {index} logical byte inventory differs from members")
            if node.get("content_kind") == "canonical_manifest" and members:
                findings.fail(f"lineage node {index} mixes a manifest with structured members")
            if node.get("content_kind") == "directory_merkle" and members:
                findings.fail(f"lineage node {index} mixes a directory with structured members")
            if node.get("content_kind") == "tensor_container" and not members:
                findings.fail(f"lineage node {index} tensor container has no structured members")

        producer = node.get("producer_receipt")
        if producer is None:
            if node.get("root") is not True:
                findings.fail(f"lineage node {index} root status differs")
        elif isinstance(producer, Mapping):
            if set(producer) != {"stage", "receipt_id"}:
                findings.missing(f"lineage node {index} producer receipt is incomplete")
            else:
                if producer.get("stage") != node.get("stage"):
                    findings.fail(f"lineage node {index} producer stage differs")
                if producer.get("receipt_id") not in receipts:
                    findings.missing(f"lineage node {index} producer receipt is unavailable")
                elif receipts[producer["receipt_id"]].get("stage") != producer.get("stage"):
                    findings.fail(f"lineage node {index} producer receipt stage differs")
            if node.get("root") is not False:
                findings.fail(f"lineage node {index} root status differs")
        else:
            findings.missing(f"lineage node {index} producer receipt is malformed")

        if node.get("content_kind") == "tensor_container" and isinstance(producer, Mapping):
            receipt = receipts.get(producer.get("receipt_id"))
            tensor_records = (
                [
                    artifact
                    for artifact in _receipt_artifacts(receipt)
                    if artifact.get("schema") == "radio_gs.tensor_container.v1"
                ]
                if receipt is not None
                else []
            )
            content_identity = node.get("content_identity")
            matching_record = next(
                (
                    artifact
                    for artifact in tensor_records
                    if isinstance(artifact.get("container"), Mapping)
                    and isinstance(content_identity, Mapping)
                    and {
                        "sha256": artifact["container"].get("sha256"),
                        "size_bytes": artifact["container"].get("size_bytes"),
                    }
                    == dict(content_identity)
                ),
                None,
            )
            if matching_record is None:
                findings.fail(
                    f"lineage node {index} tensor container is not bound to its Stage Receipt"
                )
            else:
                expected_members = matching_record.get("members")
                actual_members = node.get("members")
                if not isinstance(expected_members, list) or not isinstance(actual_members, list):
                    findings.missing(
                        f"lineage node {index} tensor member inventory is incomplete"
                    )
                elif [
                    _tensor_member_signature(member)
                    for member in expected_members
                    if isinstance(member, Mapping)
                ] != [
                    _tensor_member_signature(member)
                    for member in actual_members
                    if isinstance(member, Mapping)
                ]:
                    findings.fail(
                        f"lineage node {index} tensor logical members differ from its Stage Receipt"
                    )

        content_type = str(node.get("content_type", "")).lower()
        if node.get("stage") != "evaluation" and (
            node.get("information_grant") == "evaluator_private_target"
            or "ground_truth" in content_type
            or "target" in content_type
        ):
            findings.fail(f"ground truth or target was accessed before evaluation at lineage node {index}")
        if _node_forbidden(node) and not _authorized_private_target(node):
            findings.fail(f"forbidden or disguised lineage evidence appears at lineage node {index}")
        if contract is not None and "captured_rgb" in content_type:
            boundary = contract["information_boundary"]["query_captured_rgb"]
            if node.get("stage") == "query_prediction_sealing" and boundary == "forbidden":
                findings.fail(f"query captured RGB is forbidden at lineage node {index}")

    @staticmethod
    def _validate_edge(
        edge: Mapping[str, Any],
        index: int,
        nodes: Mapping[str, Mapping[str, Any]],
        findings: _Findings,
    ) -> None:
        required = {
            "schema",
            "schema_version",
            "edge_id",
            "source",
            "target",
            "edge_type",
            "stage",
            "purpose",
            "information_grant",
        }
        if set(edge) != required:
            findings.missing(f"lineage edge {index} fields are incomplete")
        if edge.get("schema") != LINEAGE_EDGE_SCHEMA or edge.get("schema_version") != 1:
            findings.fail(f"lineage edge {index} schema differs")
        if edge.get("edge_id") != _digest(_edge_body(edge)):
            findings.fail(f"lineage edge {index} identity differs")
        for key in ("source", "target"):
            if edge.get(key) not in nodes:
                findings.missing(f"lineage edge {index} references an unknown {key}")
        if edge.get("edge_type") not in _EDGE_TYPES:
            findings.fail(f"lineage edge {index} type differs")
        if edge.get("stage") not in STAGE_ORDER:
            findings.fail(f"lineage edge {index} stage differs")
        try:
            _require_string(edge.get("purpose"), label=f"lineage edge {index}.purpose")
            _require_string(edge.get("information_grant"), label=f"lineage edge {index}.information_grant")
        except RuntimeComplianceError as error:
            findings.fail(str(error))
        source = nodes.get(edge.get("source"))
        target = nodes.get(edge.get("target"))
        if source is not None:
            if edge.get("information_grant") != source.get("information_grant"):
                findings.fail(f"lineage edge {index} Information Grant differs from its source")
        if target is not None and edge.get("stage") != target.get("stage"):
            findings.fail(f"lineage edge {index} stage does not match its output")
        if (
            source is not None
            and target is not None
            and source.get("stage") in STAGE_ORDER
            and target.get("stage") in STAGE_ORDER
            and STAGE_ORDER.index(source["stage"]) > STAGE_ORDER.index(target["stage"])
        ):
            findings.fail(f"lineage edge {index} crosses lifecycle stages backwards")

    @staticmethod
    def _validate_direct_stage_closure(
        nodes: Mapping[str, Mapping[str, Any]],
        incoming: Mapping[str, Sequence[str]],
        contract: Mapping[str, Any] | None,
        findings: _Findings,
    ) -> None:
        for target_id, target in nodes.items():
            source_nodes = [nodes[source_id] for source_id in incoming.get(target_id, ()) if source_id in nodes]
            if target.get("stage") == "warm_cache_compilation":
                allowed = {"deployment_scene_state", "global_method_parameters"}
                for source in source_nodes:
                    if source.get("information_grant") not in allowed:
                        findings.fail("warm-cache compilation reads a Training Artifact or undeclared state")
            if target.get("stage") == "query_prediction_sealing":
                allowed = {
                    "deployment_scene_state",
                    "global_method_parameters",
                    "warm_cache",
                    "authorized_query_input",
                    "output_request_metadata",
                }
                for source in source_nodes:
                    grant = source.get("information_grant")
                    if grant not in allowed and source.get("content_type") != "query_workspace":
                        findings.fail("query execution reads undeclared or evaluator-private evidence")
                    if _node_forbidden(source):
                        if (
                            contract is None
                            or contract["information_boundary"]["query_captured_rgb"] == "forbidden"
                            or "captured_rgb" not in str(source.get("content_type", "")).lower()
                        ):
                            findings.fail("query execution reads forbidden evidence")
            if target.get("stage") == "evaluation":
                allowed = {
                    "evaluation_contract",
                    "prediction",
                    "evaluator_private_target",
                }
                for source in source_nodes:
                    if source.get("information_grant") not in allowed:
                        findings.fail("evaluation reads undeclared or pre-seal evidence")

    @staticmethod
    def _validate_row(
        audit: Mapping[str, Any],
        nodes: Mapping[str, Mapping[str, Any]],
        receipts: Sequence[Mapping[str, Any]],
        findings: _Findings,
    ) -> dict[str, Any] | None:
        row = audit.get("row")
        if not isinstance(row, Mapping):
            findings.missing("Exact Row Authority execution inventory is missing")
            return None
        row_value = _plain(row)
        required = {
            "row_id",
            "contract_id",
            "required_execution_ids",
            "execution_children",
            "prediction_node_id",
            "evaluator_result_node_id",
        }
        if set(row_value) != required:
            findings.missing("Exact Row Authority execution inventory is incomplete")
        if row_value.get("contract_id") != audit.get("contract_id"):
            findings.fail("Exact Row Authority contract identity differs from the audit contract")
        ids = row_value.get("required_execution_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != len(set(ids))
        ):
            findings.missing("required execution cohort is incomplete")
            return row_value
        try:
            expected_row_id = row_identity(row_value.get("contract_id", ""), ids)
        except RuntimeComplianceError as error:
            findings.fail(f"row execution identity is invalid: {error}")
        else:
            if row_value.get("row_id") != expected_row_id:
                findings.fail("row identity differs from its execution cohort")
        children = row_value.get("execution_children")
        if not isinstance(children, list) or not children:
            findings.missing("execution children are incomplete")
            return row_value
        child_ids: list[str] = []
        query_receipt_id = receipts[3].get("receipt_id") if len(receipts) == len(STAGE_ORDER) else None
        for index, child in enumerate(children):
            if not isinstance(child, Mapping):
                findings.missing(f"execution child {index} is incomplete")
                continue
            child_value = _plain(child)
            expected_child = {
                "execution_id",
                "scene_id",
                "query_id",
                "status",
                "complete",
                "stage_receipt_id",
                "prediction_node_ids",
                "evaluator_result_node_id",
            }
            if set(child_value) != expected_child:
                findings.missing(f"execution child {index} fields are incomplete")
            execution_id = child_value.get("execution_id")
            if not isinstance(execution_id, str) or not execution_id:
                findings.missing(f"execution child {index} identity is missing")
                continue
            child_ids.append(execution_id)
            if child_value.get("status") != "succeeded" or child_value.get("complete") is not True:
                findings.missing(f"execution child {execution_id} did not complete")
            if query_receipt_id is not None and child_value.get("stage_receipt_id") != query_receipt_id:
                findings.fail(f"execution child {execution_id} is not bound to query sealing")
            predictions = child_value.get("prediction_node_ids")
            if not isinstance(predictions, list) or not predictions:
                findings.missing(f"execution child {execution_id} prediction inventory is incomplete")
            else:
                for node_id in predictions:
                    if node_id not in nodes:
                        findings.missing(f"execution child {execution_id} references an unknown prediction")
            evaluator_id = child_value.get("evaluator_result_node_id")
            if evaluator_id not in nodes:
                findings.missing(f"execution child {execution_id} references an unknown evaluator result")
            elif evaluator_id != row_value.get("evaluator_result_node_id"):
                findings.fail(f"execution child {execution_id} is not bound to the row evaluator result")
            if isinstance(predictions, list) and row_value.get("prediction_node_id") not in predictions:
                findings.fail(f"execution child {execution_id} is not bound to the row prediction inventory")
        if child_ids != ids:
            findings.missing("execution children do not cover the required cohort")
        for key in ("prediction_node_id", "evaluator_result_node_id"):
            if row_value.get(key) not in nodes:
                findings.missing(f"row references an unknown {key}")
        prediction = nodes.get(row_value.get("prediction_node_id"))
        if prediction is not None:
            if prediction.get("content_type") not in {"prediction", "prediction_inventory"}:
                findings.fail("row prediction reference is not a prediction inventory")
            if prediction.get("stage") != "query_prediction_sealing":
                findings.fail("row prediction was not sealed during query prediction sealing")
            producer = prediction.get("producer_receipt")
            if (
                not isinstance(producer, Mapping)
                or len(receipts) != len(STAGE_ORDER)
                or producer.get("receipt_id") != receipts[3].get("receipt_id")
            ):
                findings.fail("row prediction is not bound to the query Stage Receipt")
        evaluator = nodes.get(row_value.get("evaluator_result_node_id"))
        if evaluator is not None:
            if evaluator.get("stage") != "evaluation":
                findings.fail("row evaluator result was emitted before evaluation")
            producer = evaluator.get("producer_receipt")
            if (
                not isinstance(producer, Mapping)
                or len(receipts) != len(STAGE_ORDER)
                or producer.get("receipt_id") != receipts[4].get("receipt_id")
            ):
                findings.fail("row evaluator result is not bound to the evaluation Stage Receipt")
        return row_value

    @staticmethod
    def _artifact_record_identity(record: Mapping[str, Any]) -> dict[str, Any] | None:
        """Project a Stage Receipt artifact onto the node identity it exports."""

        schema = record.get("schema")
        if schema == "radio_gs.opaque_file.v1":
            return {
                "sha256": record.get("sha256"),
                "size_bytes": record.get("size_bytes"),
            }
        if schema == "radio_gs.canonical_manifest.v1":
            value = record.get("value")
            if not isinstance(value, Mapping):
                return None
            canonical = canonical_json_bytes(value)
            return {
                "sha256": record.get("sha256"),
                "size_bytes": len(canonical),
            }
        if schema == "radio_gs.directory_merkle.v1":
            entries = record.get("entries")
            if not isinstance(entries, list):
                return None
            return {
                "sha256": record.get("merkle_root_sha256"),
                "size_bytes": sum(
                    entry.get("size_bytes", 0)
                    for entry in entries
                    if isinstance(entry, Mapping)
                ),
            }
        if schema == "radio_gs.tensor_container.v1":
            container = record.get("container")
            if not isinstance(container, Mapping):
                return None
            return {
                "sha256": container.get("sha256"),
                "size_bytes": container.get("size_bytes"),
            }
        if schema == "radio_gs.prediction_inventory.v1":
            directory = record.get("directory")
            if not isinstance(directory, Mapping):
                return None
            entries = directory.get("entries")
            if not isinstance(entries, list):
                return None
            return {
                "sha256": record.get("merkle_root_sha256"),
                "size_bytes": sum(
                    entry.get("size_bytes", 0)
                    for entry in entries
                    if isinstance(entry, Mapping)
                ),
            }
        return None

    @classmethod
    def _validate_prediction_binding(
        cls,
        audit: Mapping[str, Any],
        contract: Mapping[str, Any] | None,
        nodes: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
        receipts: Sequence[Mapping[str, Any]],
        row: Mapping[str, Any],
        findings: _Findings,
    ) -> None:
        """Close the query barrier and bind the evaluator to sealed outputs."""

        if len(receipts) != len(STAGE_ORDER):
            findings.missing("prediction and evaluator receipts are unavailable")
            return

        query_receipt = receipts[STAGE_ORDER.index("query_prediction_sealing")]
        evaluation_receipt = receipts[STAGE_ORDER.index("evaluation")]
        prediction_id = row.get("prediction_node_id")
        evaluator_id = row.get("evaluator_result_node_id")
        prediction = nodes.get(prediction_id)
        evaluator = nodes.get(evaluator_id)
        if prediction is None or evaluator is None:
            return

        if prediction.get("content_type") not in {"prediction", "prediction_inventory"}:
            findings.fail("prediction binding has a non-inventory artifact type")
        if prediction.get("stage") != "query_prediction_sealing":
            findings.fail("prediction binding is outside query prediction sealing")
        inventory = query_receipt.get("prediction_inventory")
        if not isinstance(inventory, Mapping):
            findings.missing("query Stage Receipt prediction inventory is missing")
        else:
            expected_identity = cls._artifact_record_identity(inventory)
            if prediction.get("content_kind") != "directory_merkle":
                findings.fail("prediction inventory uses a mixed container kind")
            if prediction.get("content_identity") != expected_identity:
                findings.fail("prediction inventory identity differs from the query Stage Receipt")

        producer = prediction.get("producer_receipt")
        if not isinstance(producer, Mapping) or producer.get("stage") != "query_prediction_sealing":
            findings.fail("prediction inventory producer stage is not query sealing")
        elif producer.get("receipt_id") != query_receipt.get("receipt_id"):
            findings.fail("prediction inventory producer receipt differs from query sealing")

        prediction_edges = [
            edge
            for edge in edges.values()
            if edge.get("target") == prediction_id
            and edge.get("edge_type") == "emit"
        ]
        if not any(
            nodes.get(edge.get("source"), {}).get("content_type") == "query_workspace"
            for edge in prediction_edges
        ):
            findings.fail("prediction inventory is not emitted by the query workspace")

        if evaluator.get("content_type") != "evaluator_result":
            findings.fail("row evaluator binding has a non-evaluator artifact type")
        if evaluator.get("stage") != "evaluation":
            findings.fail("evaluator result is outside evaluation")
        evaluator_query_id = evaluator.get("query_id")
        prediction_query_id = prediction.get("query_id")
        if evaluator_query_id is not None and prediction_query_id != evaluator_query_id:
            findings.fail("prediction and evaluator query identities differ")

        producer = evaluator.get("producer_receipt")
        if not isinstance(producer, Mapping) or producer.get("stage") != "evaluation":
            findings.fail("evaluator result producer stage is not evaluation")
        elif producer.get("receipt_id") != evaluation_receipt.get("receipt_id"):
            findings.fail("evaluator result producer receipt differs from evaluation")

        output_identities = [
            identity
            for raw in evaluation_receipt.get("outputs", {}).values()
            if isinstance(raw, Mapping)
            for identity in [cls._artifact_record_identity(raw)]
            if identity is not None
        ]
        evaluator_identity = evaluator.get("content_identity")
        if not any(evaluator_identity == identity for identity in output_identities):
            findings.fail("evaluator result identity is not an evaluation output")

        incoming = [
            edge
            for edge in edges.values()
            if edge.get("target") == evaluator_id
            and edge.get("edge_type") == "evaluate"
        ]
        incoming_sources = {
            edge.get("source")
            for edge in incoming
            if edge.get("source") in nodes
        }
        if prediction_id not in incoming_sources:
            findings.fail("evaluator result is not bound to sealed predictions")
        if not any(
            nodes[source_id].get("content_type") == "evaluation_contract"
            for source_id in incoming_sources
        ):
            findings.fail("evaluator result is not bound to its Evaluation Contract")
        if not any(
            nodes[source_id].get("information_grant") == "evaluator_private_target"
            and nodes[source_id].get("stage") == "evaluation"
            for source_id in incoming_sources
        ):
            findings.fail("evaluator result is not bound to an evaluation-private target")
        evaluation_input_identities = [
            identity
            for raw in evaluation_receipt.get("inputs", {}).values()
            if isinstance(raw, Mapping)
            for identity in [cls._artifact_record_identity(raw)]
            if identity is not None
        ]
        for source_id in incoming_sources:
            source = nodes[source_id]
            if source.get("information_grant") != "evaluator_private_target":
                continue
            # A private target may be an evaluator-owned root, in which case
            # its bytes are intentionally outside the producer lifecycle.  If
            # it claims a Stage Receipt producer, however, that receipt must
            # expose the exact target identity as an evaluation input.
            if source.get("producer_receipt") is not None and not any(
                source.get("content_identity") == identity
                for identity in evaluation_input_identities
            ):
                findings.fail("evaluation-private target is not an evaluation input")

        if contract is not None and evaluator.get("contract_id") != audit.get("contract_id"):
            findings.fail("evaluator result contract identity differs")

    @staticmethod
    def _validate_storage(
        audit: Mapping[str, Any],
        nodes: Mapping[str, Mapping[str, Any]],
        candidate: CandidateAuthorityBundle | None,
        findings: _Findings,
    ) -> None:
        storage = audit.get("storage")
        if not isinstance(storage, Mapping):
            findings.missing("cold storage assertion is missing")
            return
        value = _plain(storage)
        if set(value) != _REQUIRED_STORAGE_KEYS:
            findings.missing("cold storage assertion is incomplete")
        if value.get("schema") != "radio_gs.storage_assertion.v1":
            findings.fail("cold storage assertion schema differs")
        if candidate is not None:
            field = candidate["method_contract"]["field_schema"]
            if value.get("field_family") != field["family"]:
                findings.fail("cold storage field family differs from the Candidate Authority")
            if value.get("local_code_dimension") != field["local_code_dimension"]:
                findings.fail("cold storage local code dimension differs")
            if value.get("persistent_semantic_fields") != field["persistent_semantic_fields"]:
                findings.fail("cold storage contains more than the sole Canonical Capability Feature")
            if value.get("deployment_support_state") != field["deployment_support_state"]:
                findings.fail("Deployment Support State schema differs")

        n = value.get("scene_gaussian_count")
        numeric_keys = {
            "scene_gaussian_count",
            "persistent_scene_storage_increment_bytes",
            "scene_soft_target_bytes",
            "scene_hard_limit_bytes",
            "method_specific_global_bytes",
            "method_specific_global_soft_target_bytes",
            "method_specific_global_hard_limit_bytes",
            "serialization_overhead_bytes",
        }
        for key in numeric_keys:
            if not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0:
                findings.fail(f"cold storage value {key} is invalid")
        if isinstance(n, int) and not isinstance(n, bool) and n > 0:
            expected_soft = 2048 * n + 8 * 1024 * 1024
            expected_hard = min(2304 * n + 16 * 1024 * 1024, 2560 * n - 1)
            if value.get("scene_soft_target_bytes") != expected_soft:
                findings.fail("scene soft storage target differs")
            if value.get("scene_hard_limit_bytes") != expected_hard:
                findings.fail("scene hard storage limit differs")
        if value.get("method_specific_global_soft_target_bytes") != 8 * 1024 * 1024:
            findings.fail("method-specific global soft target differs")
        if value.get("method_specific_global_hard_limit_bytes") != 128 * 1024 * 1024:
            findings.fail("method-specific global hard limit differs")
        if (
            isinstance(value.get("persistent_scene_storage_increment_bytes"), int)
            and isinstance(value.get("scene_hard_limit_bytes"), int)
            and value["persistent_scene_storage_increment_bytes"] > value["scene_hard_limit_bytes"]
        ):
            findings.fail("Persistent Scene-Storage Increment exceeds the cold hard limit")
        if (
            isinstance(value.get("method_specific_global_bytes"), int)
            and isinstance(value.get("method_specific_global_hard_limit_bytes"), int)
            and value["method_specific_global_bytes"] > value["method_specific_global_hard_limit_bytes"]
        ):
            findings.fail("Method-Specific Global Parameters exceed the hard limit")
        if (
            isinstance(value.get("persistent_scene_storage_increment_bytes"), int)
            and isinstance(value.get("serialization_overhead_bytes"), int)
            and value["serialization_overhead_bytes"] > value["persistent_scene_storage_increment_bytes"] * 0.01
        ):
            findings.fail("serialization overhead exceeds the one-percent allowance")
        if value.get("forbidden_member_types") != []:
            findings.fail("cold storage inventory contains forbidden members")
        if value.get("cold_start_query_executed") is not True:
            findings.fail("cold-start query execution was not proven")
        if value.get("warm_cache_rebuilds_bitwise_identical") is not True:
            findings.fail("warm-cache rebuild was not bitwise identical")

        global_nodes = [
            node
            for node in nodes.values()
            if node.get("content_type") == "global_method_parameters"
        ]
        if len(global_nodes) != 1:
            findings.missing("Global Method Parameters inventory is missing or ambiguous")
        elif isinstance(value.get("method_specific_global_bytes"), int):
            global_logical_bytes = global_nodes[0].get("logical_bytes")
            if (
                isinstance(global_logical_bytes, int)
                and global_logical_bytes > value["method_specific_global_bytes"]
            ):
                findings.fail("Global Method Parameters are understated")

        deployment_nodes = [node for node in nodes.values() if node.get("content_type") == "deployment_scene_state"]
        if not deployment_nodes:
            findings.missing("Deployment Scene State inventory is missing")
            return
        if len(deployment_nodes) != 1:
            findings.fail("Deployment Scene State inventory is ambiguous")
            return
        members = deployment_nodes[0].get("members")
        if not isinstance(members, list):
            findings.missing("Deployment Scene State members are missing")
            return
        canonical = [member for member in members if isinstance(member, Mapping) and member.get("content_type") == "canonical_capability_feature"]
        if len(canonical) != 1:
            findings.fail("Deployment Scene State must contain exactly one Canonical Capability Feature")
        for member in members:
            if not isinstance(member, Mapping):
                continue
            content_type = str(member.get("content_type", "")).lower()
            allowed_types = {
                "canonical_capability_feature",
                "conventional_rendering_state",
                "deployment_support_state",
                "quality_scalar",
                "validity_bit",
            }
            if (
                _member_forbidden(member)
                or content_type not in allowed_types
            ):
                findings.fail("Deployment Scene State contains a forbidden semantic or query member")
        if canonical:
            shape = canonical[0].get("shape")
            if not isinstance(shape, list) or not shape or shape[-1] != 512:
                findings.fail("Canonical Capability Feature dimension differs from D512")
            if isinstance(n, int) and n > 0 and shape and shape[0] != n:
                findings.fail("Canonical Capability Feature row count differs from the scene")
        deployment_logical_bytes = deployment_nodes[0].get("logical_bytes")
        if (
            isinstance(deployment_logical_bytes, int)
            and isinstance(value.get("scene_hard_limit_bytes"), int)
            and deployment_logical_bytes > value["scene_hard_limit_bytes"]
        ):
            findings.fail("Deployment Scene State exceeds the cold hard limit")

    @staticmethod
    def _validate_environment(
        audit: Mapping[str, Any],
        receipts: Sequence[Mapping[str, Any]],
        candidate: CandidateAuthorityBundle | None,
        findings: _Findings,
    ) -> None:
        environment = audit.get("environment")
        if not isinstance(environment, Mapping) or set(environment) != {"declared", "observed", "identity_sha256"}:
            findings.missing("execution environment identity is incomplete")
            return
        declared = environment.get("declared")
        observed = environment.get("observed")
        if not isinstance(declared, Mapping) or not isinstance(observed, Mapping):
            findings.missing("declared or observed environment is missing")
            return
        declared_value = _plain(declared)
        observed_value = _plain(observed)
        if environment.get("identity_sha256") != _digest(declared_value):
            findings.fail("declared environment identity digest differs")
        if observed_value != declared_value:
            findings.fail("observed environment identity differs from declared environment")
        if candidate is not None:
            expected = candidate["method_contract"]["environment_identity"]
            for key, expected_value in expected.items():
                if declared_value.get(key) != expected_value:
                    findings.fail(f"environment identity differs at {key}")
            implementation = candidate["method_contract"]["implementation_identity"]
        else:
            implementation = None
        for receipt in receipts:
            execution = receipt.get("execution", {})
            env_manifest = execution.get("environment")
            code_manifest = execution.get("code_identity")
            if isinstance(env_manifest, Mapping) and env_manifest.get("value") != declared_value:
                findings.fail("Stage Receipt environment does not match observed environment")
            if implementation is not None and isinstance(code_manifest, Mapping):
                for key in ("repository", "commit", "dirty_patch_sha256"):
                    if code_manifest.get(key) != implementation.get(key):
                        findings.fail(f"Stage Receipt code identity differs at {key}")

    @staticmethod
    def _validate_execution(
        audit: Mapping[str, Any],
        nodes: Mapping[str, Mapping[str, Any]],
        roots: set[str],
        receipts: Sequence[Mapping[str, Any]],
        findings: _Findings,
    ) -> None:
        execution = audit.get("execution")
        if not isinstance(execution, Mapping):
            findings.missing("runtime observation is missing")
            return
        value = _plain(execution)
        if set(value) != _REQUIRED_EXECUTION_KEYS:
            findings.missing("runtime observation fields are incomplete")
        if value.get("schema") != RUNTIME_OBSERVATION_SCHEMA or value.get("schema_version") != 1:
            findings.fail("runtime observation schema differs")
        if value.get("auditor_started_before_first_instruction") is not True:
            findings.fail("auditor did not start before the first instruction")
        if value.get("process_tree_complete") is not True:
            findings.missing("complete process tree was not observed")
        if value.get("network_disabled") is not True:
            findings.fail("network was not disabled")
        if value.get("inherited_descriptors_cleared") is not True:
            findings.missing("inherited descriptors were not proven cleared")
        if value.get("shared_memory_cleared") is not True:
            findings.missing("shared memory was not proven cleared")
        if not isinstance(value.get("network_attempts"), list):
            findings.missing("network activity inventory is incomplete")
        elif value.get("network_attempts"):
            findings.fail("network activity was observed")
        if not isinstance(value.get("unknown_activity"), list):
            findings.missing("unknown activity inventory is incomplete")
        elif value.get("unknown_activity"):
            findings.fail("unknown runtime activity was observed")

        channels = value.get("trace_channels")
        if not isinstance(channels, Mapping) or set(channels) != set(_TRACE_CHANNELS):
            findings.missing("runtime trace channels are incomplete")
        elif any(channels.get(channel) is not True for channel in _TRACE_CHANNELS):
            findings.missing("runtime trace channels are incomplete")

        mounts = value.get("allowlist_mounts")
        if not isinstance(mounts, list) or not mounts:
            findings.missing("allowlist mounts are incomplete")
        else:
            mounted: set[str] = set()
            mount_paths: set[str] = set()
            for mount in mounts:
                if not isinstance(mount, Mapping) or set(mount) != {"path", "mode", "node_id"}:
                    findings.missing("allowlist mount record is incomplete")
                    continue
                path = mount.get("path")
                if not isinstance(path, str) or not path.startswith("/") or path in {"/", "/root", "/home", "/tmp", "/etc"}:
                    findings.fail("allowlist exposes a broad or invalid mount")
                elif path in mount_paths:
                    findings.fail("allowlist mount path is duplicated")
                mount_paths.add(path)
                node_id = mount.get("node_id")
                if mount.get("mode") != "ro":
                    findings.fail("allowlist exposes a non-read-only mount")
                if node_id not in roots:
                    findings.fail("allowlist exposes a non-root artifact")
                if node_id in mounted:
                    findings.fail("allowlist mount is duplicated")
                mounted.add(node_id)
            if mounted != roots:
                findings.missing("allowlist mounts do not cover exactly the evidence roots")

        workspaces = value.get("stage_workspaces")
        if not isinstance(workspaces, list) or len(workspaces) != len(STAGE_ORDER):
            findings.missing("stage workspaces are incomplete")
        else:
            seen_stages: set[str] = set()
            workspace_paths: set[str] = set()
            for workspace in workspaces:
                if not isinstance(workspace, Mapping):
                    findings.missing("stage workspace record is incomplete")
                    continue
                stage = workspace.get("stage")
                seen_stages.add(stage)
                path = workspace.get("path")
                if not isinstance(path, str) or not path.startswith("/"):
                    findings.missing("stage workspace path is incomplete")
                elif path in workspace_paths:
                    findings.fail("stage workspace path is duplicated")
                workspace_paths.add(path)
                if stage not in STAGE_ORDER:
                    findings.fail("stage workspace names an unknown stage")
                if any(workspace.get(key) is not True for key in ("empty_at_start", "empty_at_end", "caches_empty")):
                    findings.fail("stage workspace was not empty and cache-free")
            if seen_stages != set(STAGE_ORDER):
                findings.missing("stage workspaces do not cover the lifecycle")

        process_tree = value.get("process_tree")
        process_by_stage: dict[str, dict[str, Any]] = {}
        process_by_id: dict[str, dict[str, Any]] = {}
        process_ids: set[str] = set()
        if not isinstance(process_tree, list) or not process_tree:
            findings.missing("process tree is incomplete")
        else:
            for process in process_tree:
                if not isinstance(process, Mapping):
                    findings.missing("process tree record is incomplete")
                    continue
                process_value = _plain(process)
                required = {"process_id", "parent_id", "stage", "entrypoint_node_id", "children_complete"}
                if set(process_value) != required:
                    findings.missing("process tree record is incomplete")
                    continue
                process_id = process_value.get("process_id")
                stage = process_value.get("stage")
                if not isinstance(process_id, str) or not process_id:
                    findings.fail("process tree identity is duplicated or invalid")
                    continue
                if process_id in process_ids:
                    findings.fail("process tree identity is duplicated or invalid")
                    continue
                process_ids.add(process_id)
                process_by_id[process_id] = process_value
                parent = process_value.get("parent_id")
                if parent is not None and (not isinstance(parent, str) or not parent):
                    findings.fail("process tree parent identity is invalid")
                if parent == process_id:
                    findings.fail("process tree contains a self-parent")
                if not isinstance(stage, str) or not stage:
                    findings.fail("process tree stage is invalid")
                    continue
                if stage in process_by_stage:
                    findings.fail("process tree has multiple roots for one stage")
                process_by_stage[stage] = process_value
                if stage not in STAGE_ORDER or process_value.get("entrypoint_node_id") not in nodes:
                    findings.fail("process tree entrypoint is not bound to the lineage")
                if process_value.get("children_complete") is not True:
                    findings.missing("process tree children are incomplete")
            if set(process_by_stage) != set(STAGE_ORDER):
                findings.missing("process tree does not cover the five stages")
            for process in process_by_stage.values():
                parent = process.get("parent_id")
                if parent is not None and parent not in process_ids:
                    findings.fail("process tree references an unknown parent")
            for process_id in process_ids:
                seen_parents: set[str] = set()
                current = process_id
                while True:
                    parent = process_by_id[current].get("parent_id")
                    if parent is None:
                        break
                    if parent in seen_parents or parent == process_id:
                        findings.fail("process tree contains a cycle")
                        break
                    seen_parents.add(parent)
                    if parent not in process_by_id:
                        break
                    current = parent

        stage_executions = value.get("stage_executions")
        if not isinstance(stage_executions, list) or len(stage_executions) != len(STAGE_ORDER):
            findings.missing("stage execution observations are incomplete")
        else:
            observed_stages: list[str] = []
            receipt_by_id = {receipt.get("receipt_id"): receipt for receipt in receipts}
            for stage_execution in stage_executions:
                if not isinstance(stage_execution, Mapping):
                    findings.missing("stage execution record is incomplete")
                    continue
                item = _plain(stage_execution)
                stage = item.get("stage")
                observed_stages.append(stage)
                if stage not in STAGE_ORDER:
                    findings.fail("stage execution names an unknown stage")
                receipt = receipt_by_id.get(item.get("receipt_id"))
                if receipt is None:
                    findings.fail("stage execution is not bound to a Stage Receipt")
                elif receipt.get("stage") != stage:
                    findings.fail("stage execution is bound to the wrong Stage Receipt")
                if item.get("complete") is not True:
                    findings.missing("stage execution is incomplete")
                ids = item.get("process_ids")
                if not isinstance(ids, list) or not ids or any(process_id not in process_ids for process_id in ids):
                    findings.missing("stage execution process coverage is incomplete")
                elif len(ids) != len(set(ids)):
                    findings.fail("stage execution process coverage is duplicated")
                elif any(process_by_id[process_id].get("stage") != stage for process_id in ids):
                    findings.fail("stage execution is bound to a process from another stage")
            if observed_stages != list(STAGE_ORDER):
                findings.missing("stage execution observations do not cover the lifecycle")

        declared = value.get("declared_activity")
        observed = value.get("observed_activity")
        if not isinstance(declared, list) or not isinstance(observed, list):
            findings.missing("declared or observed activity is incomplete")
            return
        if declared != observed:
            findings.fail("observed runtime activity differs from declared activity")
        key = _canonical_activity_key
        if declared != sorted(declared, key=key) or observed != sorted(observed, key=key):
            findings.fail("runtime activity ordering evidence differs")
        declared_keys = [key(item) for item in declared]
        if len(declared_keys) != len(set(declared_keys)):
            findings.fail("runtime activity contains a duplicate dependency record")
        activity_nodes: set[str] = set()
        for index, raw in enumerate(declared):
            if not isinstance(raw, Mapping):
                findings.fail(f"activity record {index} is malformed")
                continue
            activity = _plain(raw)
            required = {"activity_id", "schema", "schema_version", "kind", "stage", "process_id", "operation", "node_id", "identity_sha256"}
            if set(activity) != required:
                findings.missing(f"activity record {index} is incomplete")
                continue
            if activity.get("schema") != ACTIVITY_SCHEMA or activity.get("schema_version") != 1:
                findings.fail(f"activity record {index} schema differs")
            if activity.get("activity_id") != _digest(_activity_body(activity)):
                findings.fail(f"activity record {index} identity differs")
            if activity.get("kind") in _DISALLOWED_ACTIVITY_KINDS:
                findings.fail(f"activity record {index} exposes a forbidden process or IPC path")
            elif activity.get("kind") not in _ACTIVITY_KINDS:
                findings.fail(f"activity record {index} kind is unknown")
            if activity.get("stage") not in STAGE_ORDER:
                findings.fail(f"activity record {index} stage differs")
            if activity.get("process_id") not in process_ids:
                findings.fail(f"activity record {index} process is outside the observed tree")
            elif process_by_id[activity["process_id"]].get("stage") != activity.get("stage"):
                findings.fail(f"activity record {index} process stage differs")
            node_id = activity.get("node_id")
            if node_id not in nodes:
                findings.missing(f"activity record {index} references an unknown artifact")
                continue
            activity_nodes.add(node_id)
            content_identity = nodes[node_id].get("content_identity")
            if (
                not isinstance(content_identity, Mapping)
                or activity.get("identity_sha256") != content_identity.get("sha256")
            ):
                findings.fail(f"activity record {index} identity differs from its artifact")
            # An executable/library/model observation may reuse the same
            # content identity in each lifecycle process.  File ownership is
            # stage-bound, so keep the stronger stage check for file reads and
            # writes while allowing process-level observations to bind the
            # implementation identity across stages.
            if activity.get("kind") == "file" and activity.get("stage") != nodes[node_id].get("stage"):
                findings.fail(f"activity record {index} stage differs from its artifact")
            if activity.get("kind") == "file":
                expected_operation = "read" if nodes[node_id].get("root") is True else "write"
                if activity.get("operation") != expected_operation:
                    findings.fail(f"activity record {index} file operation differs from lifecycle ownership")
        missing_nodes = set(nodes) - activity_nodes
        if missing_nodes:
            findings.missing("runtime activity does not cover every lineage artifact")

        for index, receipt in enumerate(receipts):
            trace = receipt.get("execution", {}).get("runtime_trace", {})
            if isinstance(trace, Mapping):
                trace_value = trace.get("value")
                if (
                    not isinstance(trace_value, Mapping)
                    or trace_value.get("trace_schema") != "runtime-compliance-trace-v1"
                    or not isinstance(trace_value.get("trace_id"), str)
                    or not trace_value.get("trace_id")
                    or trace_value.get("complete") is not True
                ):
                    findings.missing("a Stage Receipt lacks complete runtime trace evidence")
            else:
                findings.missing(f"stage {STAGE_ORDER[index]} runtime trace is not a manifest")

    @staticmethod
    def _proof(
        audit: Mapping[str, Any],
        candidate: CandidateAuthorityBundle,
        nodes: Mapping[str, Mapping[str, Any]],
        edges: Mapping[str, Mapping[str, Any]],
        roots: set[str],
        receipts: Sequence[Mapping[str, Any]],
        row: Mapping[str, Any],
    ) -> RuntimeComplianceProof:
        storage = _plain(audit["storage"])
        environment = _plain(audit["environment"])
        query_receipt = receipts[3]
        body = {
            "schema": RUNTIME_COMPLIANCE_PROOF_SCHEMA,
            "schema_version": 1,
            "status": "PASS",
            "candidate_id": candidate["candidate_id"],
            "contract_id": audit["contract_id"],
            "row_id": row["row_id"],
            "stage_receipt_ids": [receipt["receipt_id"] for receipt in receipts],
            "lineage_root_ids": sorted(roots),
            "lineage_node_ids": sorted(nodes),
            "lineage_edge_ids": sorted(edges),
            "prediction_node_id": row["prediction_node_id"],
            "evaluator_result_node_id": row["evaluator_result_node_id"],
            "prediction_merkle_root_sha256": query_receipt["prediction_inventory"]["merkle_root_sha256"],
            "storage_assertion_sha256": _digest(storage),
            "environment_identity_sha256": environment["identity_sha256"],
            "verifier_identity": RuntimeComplianceVerifier.verifier_identity,
        }
        proof = {**body, "proof_id": _digest(body)}
        validated = validate_runtime_compliance_proof(proof)
        return RuntimeComplianceProof(
            validated.as_dict(),
            _token=_VERIFIED_PROOF_TOKEN,
        )


def verify_runtime_compliance(
    audit: Mapping[str, Any],
    candidate_authority: Mapping[str, Any] | CandidateAuthorityBundle,
) -> dict[str, Any]:
    """Convenience wrapper for the independent verifier seam."""

    return RuntimeComplianceVerifier().verify(audit, candidate_authority)


__all__ = [
    "ACTIVITY_SCHEMA",
    "EVIDENCE_NODE_SCHEMA",
    "LINEAGE_EDGE_SCHEMA",
    "RUNTIME_COMPLIANCE_AUDIT_SCHEMA",
    "RUNTIME_COMPLIANCE_PROOF_SCHEMA",
    "RUNTIME_OBSERVATION_SCHEMA",
    "RuntimeComplianceProof",
    "RuntimeComplianceError",
    "RuntimeComplianceVerifier",
    "activity_record",
    "evidence_node",
    "lineage_edge",
    "load_runtime_compliance_proof",
    "row_identity",
    "validate_runtime_compliance_proof",
    "verify_runtime_compliance",
    "write_runtime_compliance_proof",
]
