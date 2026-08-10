#!/usr/bin/env python3
"""One-shot reserved-bank confirmation for descriptor-first source responses.

This sibling leaves the completed 101-query development materializer byte-for-
byte frozen.  It reuses that exact renderer and descriptor loader while
replacing only the sealed query-bank loader and response shape contract with
the preregistered 90-query ``audit`` split.  No target asset is authorized.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

import radio_gs.scripts.materialize_lerf_source_text_response_summaries as _base
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_source_text_response_confirmation_execution.v1"
AUTHORITY_STATUS = "authorized_reserved_audit_source_only_confirmation"
QUERY_COUNT = 90
QUERY_SPLIT = "audit"
AUDIT_BANK = {
    "path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_audit_embeddings.pt",
    "sha256": "46dd338340a310e2b59997d1b6ea4882590c76f8aca389d4aa0abc2b3c5c2721",
    "manifest_path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_audit_embeddings.manifest.json",
    "manifest_sha256": "1f19257393b45d713fb80be707e962a9680b84b793328a7364ba78bdd57b46b4",
    "query_split": QUERY_SPLIT,
    "queries": QUERY_COUNT,
    "embedding_tensor_sha256": "8cc8e9ca3903488ef7558ce19d9e4bec9cb6378570c8c12d72ac96219c8dd415",
}
DEV_MANIFEST = {
    "path": "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260731/target_blind_siglip2_text_bank_v1/target_blind_siglip2_dev_embeddings.manifest.json",
    "sha256": "50335f0f7f1a0f47388b600844bbfeba9b6a8a8290f3f74a88d3814b13b671d3",
    "split": "dev",
    "queries": 101,
    "synsets": 101,
    "ordered_records_sha256": "824b2d7f8115db39fe379ec2f8cb834f1c78da5e7c1d3051fb52bb3464887711",
    "split_synset_tab_query_lf_sha256": "26b814d872a6455961097d1ab1390d951d69e69c2a88d4c58e9bd003217f6544",
}
BASE_IMPLEMENTATION = _base.IMPLEMENTATION
IMPLEMENTATION = file_record(Path(__file__).resolve())


def _record(value: object, *, label: str) -> dict[str, str]:
    return _base._record(value, label=label)


def _validate_confirmation_fields(authority: Mapping[str, Any]) -> None:
    confirmation = authority.get("confirmation")
    if not isinstance(confirmation, Mapping) or set(confirmation) != {
        "candidate_locked_after_dev_gate",
        "dev_gate_result",
        "reserved_bank_opened_once",
        "dev_audit_disjointness_proof",
        "end_to_end_interface_ab_not_capacity_matched",
    }:
        raise ValueError("reserved confirmation declaration differs")
    if (
        confirmation.get("candidate_locked_after_dev_gate") is not True
        or confirmation.get("reserved_bank_opened_once") is not True
        or confirmation.get("end_to_end_interface_ab_not_capacity_matched") is not True
    ):
        raise ValueError("reserved confirmation is not candidate locked")
    dev_gate_record = _record(
        confirmation.get("dev_gate_result"), label="development gate result"
    )
    proof_record = _record(
        confirmation.get("dev_audit_disjointness_proof"),
        label="dev/audit disjointness proof",
    )
    validate_file_record(dev_gate_record, label="development gate result")
    validate_file_record(proof_record, label="dev/audit disjointness proof")
    dev_gate, _, _ = load_json_object(
        dev_gate_record["path"],
        expected_sha256=dev_gate_record["sha256"],
        label="development source text-response gate result",
    )
    candidate_methods = [
        str(method.get("method_id"))
        for method in authority.get("methods", [])
        if isinstance(method, Mapping) and method.get("role") == "candidate"
    ]
    decision = dev_gate.get("decision")
    if (
        dev_gate.get("schema")
        != "radio_gs.lerf_source_text_response_ranking_gate.v1"
        or dev_gate.get("status") != "passed"
        or not isinstance(decision, Mapping)
        or decision.get("candidate_eligible_for_next_source_gate") is not True
        or dev_gate.get("metric_execution_authorized") is not False
        or dev_gate.get("metric_executed") is not False
        or candidate_methods != [dev_gate.get("candidate_method_id")]
    ):
        raise ValueError("development source gate did not lock this candidate")
    proof, _, _ = load_json_object(
        proof_record["path"],
        expected_sha256=proof_record["sha256"],
        label="dev/audit query-bank disjointness proof",
    )
    expected_audit_manifest = {
        "path": AUDIT_BANK["manifest_path"],
        "sha256": AUDIT_BANK["manifest_sha256"],
        "split": QUERY_SPLIT,
        "queries": QUERY_COUNT,
        "synsets": QUERY_COUNT,
        "ordered_records_sha256": "0bfe94aeb6b5e0fecc978c6c66d77bba0fc0b5b7be59d922a801915843bd748f",
        "split_synset_tab_query_lf_sha256": "3b78a2e81e2750dd7314d6431ac44ddea05dd505948d775e9d1e33e87ae0bc7b",
    }
    proof_value = proof.get("proof")
    proof_access = proof.get("access_audit")
    if (
        proof.get("schema")
        != "radio_gs.lerf_source_text_response_dev_audit_disjointness_proof.v1"
        or proof.get("schema_version") != 1
        or proof.get("status")
        != "sealed_disjoint_before_reserved_audit_confirmation"
        or proof.get("dev_manifest") != DEV_MANIFEST
        or proof.get("audit_manifest") != expected_audit_manifest
        or not isinstance(proof_value, Mapping)
        or proof_value.get("query_intersection_count") != 0
        or proof_value.get("synset_intersection_count") != 0
        or proof_value.get("dev_audit_disjoint") is not True
        or proof_value.get("benchmark_vocabulary_opened_by_banks") is not False
        or not isinstance(proof_access, Mapping)
        or proof_access.get("reserved_audit_embedding_tensor_opened") is not False
        or proof_access.get("benchmark_queries_masks_or_labels_opened") is not False
        or proof_access.get("target_metric_execution_authorized") is not False
    ):
        raise ValueError("dev/audit query-bank disjointness proof differs")
    dev_manifest_record = {
        "path": DEV_MANIFEST["path"],
        "sha256": DEV_MANIFEST["sha256"],
    }
    audit_manifest_record = {
        "path": AUDIT_BANK["manifest_path"],
        "sha256": AUDIT_BANK["manifest_sha256"],
    }
    validate_file_record(dev_manifest_record, label="development query-bank manifest")
    validate_file_record(audit_manifest_record, label="audit query-bank manifest")
    dev_manifest, _, _ = load_json_object(
        dev_manifest_record["path"],
        expected_sha256=dev_manifest_record["sha256"],
        label="development query-bank manifest",
    )
    audit_manifest, _, _ = load_json_object(
        audit_manifest_record["path"],
        expected_sha256=audit_manifest_record["sha256"],
        label="audit query-bank manifest",
    )
    dev_queries = dev_manifest.get("queries")
    audit_queries = audit_manifest.get("queries")
    dev_synsets = dev_manifest.get("synsets")
    audit_synsets = audit_manifest.get("synsets")
    if (
        dev_manifest.get("split") != DEV_MANIFEST["split"]
        or audit_manifest.get("split") != QUERY_SPLIT
        or dev_manifest.get("benchmark_vocabulary_opened") is not False
        or audit_manifest.get("benchmark_vocabulary_opened") is not False
        or not isinstance(dev_queries, list)
        or not isinstance(audit_queries, list)
        or not isinstance(dev_synsets, list)
        or not isinstance(audit_synsets, list)
        or len(dev_queries) != DEV_MANIFEST["queries"]
        or len(audit_queries) != QUERY_COUNT
        or len(dev_synsets) != DEV_MANIFEST["synsets"]
        or len(audit_synsets) != QUERY_COUNT
        or dev_manifest.get("ordered_records_sha256")
        != DEV_MANIFEST["ordered_records_sha256"]
        or audit_manifest.get("ordered_records_sha256")
        != expected_audit_manifest["ordered_records_sha256"]
        or set(dev_queries).intersection(audit_queries)
        or set(dev_synsets).intersection(audit_synsets)
    ):
        raise ValueError("dev/audit query-bank manifests are not exactly disjoint")


def validate_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("confirmation execution authority must be an object")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("implementation") != IMPLEMENTATION
        or authority.get("base_materializer") != BASE_IMPLEMENTATION
    ):
        raise ValueError("confirmation execution authority schema differs")
    _validate_confirmation_fields(authority)
    if authority.get("query_bank") != AUDIT_BANK:
        raise ValueError("reserved audit query-bank authority differs")
    inputs = authority.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("confirmation input contract differs")
    if inputs.get("query_bank_artifact") != {
        "path": AUDIT_BANK["path"],
        "sha256": AUDIT_BANK["sha256"],
    } or inputs.get("query_bank_manifest") != {
        "path": AUDIT_BANK["manifest_path"],
        "sha256": AUDIT_BANK["manifest_sha256"],
    }:
        raise ValueError("reserved audit query-bank input differs")
    _record(
        inputs.get("source_gate_preregistration"),
        label="candidate-specific confirmation preregistration",
    )

    # Reuse the development materializer's complete structural validator for
    # geometry, methods, provenance, reseal, outputs, execution, and access.
    # The adapted values exist only in memory and do not open the dev bank.
    adapted = copy.deepcopy(authority)
    adapted["schema"] = _base.AUTHORITY_SCHEMA
    adapted["status"] = "authorized_source_only_compact_summary"
    adapted["implementation"] = BASE_IMPLEMENTATION
    adapted_bank = dict(AUDIT_BANK)
    adapted_bank["query_split"] = "dev"
    adapted_bank["queries"] = 101
    adapted["query_bank"] = adapted_bank
    _base.validate_authority(adapted)
    return authority


def load_authority(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, str]]:
    authority, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="reserved audit confirmation execution authority",
    )
    return validate_authority(authority), {"path": str(source), "sha256": digest}


def _load_query_bank(
    authority: Mapping[str, Any],
) -> tuple[torch.Tensor, list[str], dict[str, object]]:
    inputs = authority["inputs"]
    artifact = _record(inputs["query_bank_artifact"], label="audit query bank artifact")
    manifest_record = _record(
        inputs["query_bank_manifest"], label="audit query bank manifest"
    )
    payload, _, source = load_torch_mapping(
        artifact["path"],
        expected_sha256=artifact["sha256"],
        map_location="cpu",
        label="reserved target-blind audit query bank",
    )
    manifest, _, manifest_source = load_json_object(
        manifest_record["path"],
        expected_sha256=manifest_record["sha256"],
        label="reserved target-blind audit query bank manifest",
    )
    embeddings = torch.as_tensor(payload.get("embeddings"))
    queries = payload.get("synsets")
    if (
        payload.get("split") != QUERY_SPLIT
        or payload.get("benchmark_vocabulary_opened") is not False
        or payload.get("uses_benchmark_vocabulary_for_construction") is not False
        or manifest.get("split") != QUERY_SPLIT
        or manifest.get("benchmark_vocabulary_opened") is not False
        or not isinstance(queries, list)
        or len(queries) != QUERY_COUNT
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.shape != (QUERY_COUNT, 1536)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("reserved audit query bank contract differs")
    if payload.get("embedding_tensor_sha256") != AUDIT_BANK["embedding_tensor_sha256"]:
        raise ValueError("reserved audit embedding tensor authority differs")
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=5e-5, rtol=5e-5)):
        raise ValueError("reserved audit query embeddings are not normalized")
    query_record = dict(authority["query_bank"])
    if (
        query_record != AUDIT_BANK
        or query_record["path"] != str(source)
        or query_record["manifest_path"] != str(manifest_source)
    ):
        raise ValueError("reserved audit query-bank record differs")
    return embeddings.contiguous(), [str(value) for value in queries], query_record


def descriptor_map_to_text_responses(
    descriptor_map: torch.Tensor, text_embeddings: torch.Tensor
) -> torch.Tensor:
    descriptors = torch.as_tensor(descriptor_map).float()
    text = F.normalize(torch.as_tensor(text_embeddings).float(), dim=-1)
    if descriptors.ndim != 3 or descriptors.shape[0] != 1536:
        raise ValueError("rendered descriptor map contract differs")
    if text.shape != (QUERY_COUNT, 1536):
        raise ValueError("reserved audit text embedding contract differs")
    unit = F.normalize(descriptors, dim=0, eps=1e-12)
    return torch.einsum("qd,dhw->qhw", text.to(unit.device), unit)


def materialize(
    authority_path: str | Path, expected_authority_sha256: str
) -> dict[str, Any]:
    original_load_authority = _base.load_authority
    original_load_query_bank = _base._load_query_bank
    original_response = _base.descriptor_map_to_text_responses
    _base.load_authority = load_authority
    _base._load_query_bank = _load_query_bank
    _base.descriptor_map_to_text_responses = descriptor_map_to_text_responses
    try:
        result = _base.materialize(authority_path, expected_authority_sha256)
    finally:
        _base.load_authority = original_load_authority
        _base._load_query_bank = original_load_query_bank
        _base.descriptor_map_to_text_responses = original_response
    if result.get("query_count") != QUERY_COUNT:
        raise RuntimeError("reserved audit confirmation query count differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--execution-authority-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.execution_authority, args.execution_authority_sha256),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_BANK",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "BASE_IMPLEMENTATION",
    "IMPLEMENTATION",
    "QUERY_COUNT",
    "QUERY_SPLIT",
    "descriptor_map_to_text_responses",
    "load_authority",
    "materialize",
    "validate_authority",
]
