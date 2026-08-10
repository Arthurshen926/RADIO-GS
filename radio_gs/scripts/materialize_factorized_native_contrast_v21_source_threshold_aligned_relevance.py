#!/usr/bin/env python3
"""Materialize source-threshold-aligned contrast-V2.1 target relevance."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_contrast_v21_source_threshold_aligned_relevance as formal
from radio_gs.scripts import materialize_factorized_native_contrast_v21_lerf_exact_relevance as raw_script
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
TEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests/test_factorized_native_contrast_v21_source_threshold_aligned_relevance.py"
)
IMPLEMENTATION_DEPENDENCIES = {
    "aligned_relevance_formal": Path(formal.__file__).resolve(),
    "aligned_relevance_tests": TEST_PATH,
    "source_threshold_formal": Path(formal.threshold_formal.__file__).resolve(),
    "raw_relevance_formal": Path(formal.raw_formal.__file__).resolve(),
    "raw_relevance_materializer": Path(raw_script.__file__).resolve(),
}
AUTHORITY_STATUS = "authorized_by_promoted_source_global_threshold_for_target_relevance"


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be canonical absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def _argument_record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    return formal.record({"path": canonical, "sha256": str(digest)}, label=label)


def _load_raw_lineage(record: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, digest, source = load_torch_mapping(
        record["path"], expected_sha256=record["sha256"], map_location="cpu",
        label="raw contrast V2.1 exact target relevance",
    )
    if dict(record) != {"path": str(source), "sha256": digest}:
        raise ValueError("raw exact relevance record differs")
    payload = formal.raw_formal.validate_query_relevance(raw)
    execution = raw_script.validate_authority(
        payload["query_execution_authority"]["path"],
        expected_sha256=payload["query_execution_authority"]["sha256"],
        expected_output=record["path"],
    )
    expected_inputs = {
        "source_result": execution["source_result"],
        "target_descriptor": execution["target_descriptor"],
        "health_v4_audit": execution["health_v4_audit"],
        "health_v4_preregistration": execution["health_v4_preregistration"],
        "query_preregistration": execution["query_preregistration"],
        "exact_query_manifest": execution["exact_query_manifest"],
        "positive_text_cache": execution["positive_text_cache"],
        "all_query_text_cache": execution["all_query_text_cache"],
        "canonical_negative_bank": execution["canonical_negative_bank"],
    }
    descriptor = execution["verified_prequery_gate"]["descriptor_view"]
    if (
        payload["query_execution_authority"] != execution["verified_record"]
        or payload["producer"] != execution["implementation"]
        or payload["input_authority"] != expected_inputs
        or payload["scene_id"] != descriptor["scene_id"]
        or payload["physical_space_id"] != descriptor["physical_space_id"]
        or payload["region_row_ids"] != descriptor["region_row_ids"]
        or payload["region_fingerprints"] != descriptor["region_fingerprints"]
        or not torch.equal(
            payload["canonical_region_indices"], descriptor["canonical_region_indices"]
        )
        or payload["query_ids"] != list(execution["verified_positive"].query_ids)
    ):
        raise ValueError("raw exact relevance source/health/query lineage differs")
    return payload, execution


def _validate_source_and_raw(
    *, threshold_record: Mapping[str, str], raw_record: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Source promotion must be validated before any target/query artifact opens.
    threshold, _ = formal.load_promoted_source_threshold_envelope(threshold_record)
    raw, execution = _load_raw_lineage(raw_record)
    source = threshold["input_authority"]["source_contrast_v21_result"]
    checkpoint = threshold["input_authority"]["source_contrast_v21_checkpoint"]
    raw_source_gate = execution["verified_prequery_gate"]["source_gate"]
    if (
        execution["source_result"] != source
        or raw["input_authority"]["source_result"] != source
        or raw_source_gate["result"]["checkpoint"] != checkpoint
    ):
        raise ValueError("source threshold and target relevance lineage differ")
    return threshold, raw, execution


def _dependency_records() -> dict[str, dict[str, str]]:
    return {name: file_record(path) for name, path in IMPLEMENTATION_DEPENDENCIES.items()}


def _validate_dependencies(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(IMPLEMENTATION_DEPENDENCIES):
        raise ValueError("aligned relevance implementation dependencies differ")
    result: dict[str, dict[str, str]] = {}
    for name, expected in IMPLEMENTATION_DEPENDENCIES.items():
        observed = validate_file_record(value[name], label=f"aligned relevance dependency {name}")
        if observed != expected:
            raise ValueError(f"aligned relevance dependency differs: {name}")
        result[name] = formal.record(value[name], label=f"aligned relevance dependency {name}")
    return result


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.output_authority, label="aligned relevance authority")
    relevance_output = _new(args.aligned_relevance_output, label="aligned relevance output")
    threshold_record = _argument_record(
        args.source_threshold_envelope, args.expected_source_threshold_envelope_sha256,
        label="source threshold envelope",
    )
    raw_record = _argument_record(
        args.raw_query_relevance, args.expected_raw_query_relevance_sha256,
        label="raw exact query relevance",
    )
    threshold, raw, execution = _validate_source_and_raw(
        threshold_record=threshold_record, raw_record=raw_record
    )
    source_gate = execution["verified_prequery_gate"]["source_gate"]
    authority = {
        "schema": formal.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": AUTHORITY_STATUS,
        "scene_id": raw["scene_id"],
        "physical_space_id": raw["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "implementation_dependencies": _dependency_records(),
        "relevance_contract_sha256": formal.RELEVANCE_CONTRACT_SHA256,
        "source_threshold_envelope": threshold_record,
        "raw_query_relevance": raw_record,
        "raw_query_execution_authority": dict(raw["query_execution_authority"]),
        "source_result": dict(execution["source_result"]),
        "source_checkpoint": dict(source_gate["result"]["checkpoint"]),
        "target_descriptor": dict(execution["target_descriptor"]),
        "health_v4_audit": dict(execution["health_v4_audit"]),
        "exact_query_manifest": dict(execution["exact_query_manifest"]),
        "positive_text_cache": dict(execution["positive_text_cache"]),
        "all_query_text_cache": dict(execution["all_query_text_cache"]),
        "canonical_negative_bank": dict(execution["canonical_negative_bank"]),
        "source_global_margin_threshold": float(
            threshold["thresholds"]["train_selected_candidate"]
        ),
        "aligned_relevance_output": str(relevance_output),
        "target_relevance_execution_authorized": True,
        "metric_execution_authorized": False,
        "frozen_relative_candidate_execution_authorized": False,
        "access_audit": formal.access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {
        "status": "source_threshold_aligned_relevance_authority_built",
        "authority": file_record(authority_output),
        "threshold": authority["source_global_margin_threshold"],
    }


def validate_authority(
    path: str | Path, *, expected_sha256: str, expected_output: str | Path | None = None
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path, expected_sha256=expected_sha256,
        label="source-threshold aligned relevance execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "physical_space_id",
        "implementation", "implementation_dependencies", "relevance_contract_sha256",
        "source_threshold_envelope", "raw_query_relevance",
        "raw_query_execution_authority", "source_result", "source_checkpoint",
        "target_descriptor", "health_v4_audit", "exact_query_manifest",
        "positive_text_cache", "all_query_text_cache", "canonical_negative_bank",
        "source_global_margin_threshold", "aligned_relevance_output",
        "target_relevance_execution_authorized", "metric_execution_authorized",
        "frozen_relative_candidate_execution_authorized", "access_audit",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("source-threshold aligned relevance authority fields differ")
    authority = dict(raw)
    if (
        authority["schema"] != formal.EXECUTION_AUTHORITY_SCHEMA
        or authority["schema_version"] != formal.SCHEMA_VERSION
        or authority["status"] != AUTHORITY_STATUS
        or authority["relevance_contract_sha256"] != formal.RELEVANCE_CONTRACT_SHA256
        or authority["target_relevance_execution_authorized"] is not True
        or authority["metric_execution_authorized"] is not False
        or authority["frozen_relative_candidate_execution_authorized"] is not False
        or authority["access_audit"] != formal.access_audit()
    ):
        raise ValueError("source-threshold aligned relevance authority header differs")
    threshold_record = formal.record(
        authority["source_threshold_envelope"], label="source threshold envelope"
    )
    raw_record = formal.record(authority["raw_query_relevance"], label="raw relevance")
    threshold, raw_payload, raw_execution = _validate_source_and_raw(
        threshold_record=threshold_record, raw_record=raw_record
    )
    if validate_file_record(authority["implementation"], label="aligned implementation") != IMPLEMENTATION:
        raise ValueError("source-threshold aligned implementation differs")
    dependencies = _validate_dependencies(authority["implementation_dependencies"])
    source_gate = raw_execution["verified_prequery_gate"]["source_gate"]
    expected_records = {
        "raw_query_execution_authority": raw_payload["query_execution_authority"],
        "source_result": raw_execution["source_result"],
        "source_checkpoint": source_gate["result"]["checkpoint"],
        "target_descriptor": raw_execution["target_descriptor"],
        "health_v4_audit": raw_execution["health_v4_audit"],
        "exact_query_manifest": raw_execution["exact_query_manifest"],
        "positive_text_cache": raw_execution["positive_text_cache"],
        "all_query_text_cache": raw_execution["all_query_text_cache"],
        "canonical_negative_bank": raw_execution["canonical_negative_bank"],
    }
    for name, expected in expected_records.items():
        if formal.record(authority[name], label=f"aligned authority {name}") != expected:
            raise ValueError(f"source-threshold aligned authority {name} differs")
    threshold_value = float(threshold["thresholds"]["train_selected_candidate"])
    if (
        authority["scene_id"] != raw_payload["scene_id"]
        or authority["physical_space_id"] != raw_payload["physical_space_id"]
        or abs(float(authority["source_global_margin_threshold"]) - threshold_value) > 1e-15
    ):
        raise ValueError("source-threshold aligned target identity differs")
    output = str(Path(authority["aligned_relevance_output"]).expanduser().resolve())
    if output != authority["aligned_relevance_output"]:
        raise ValueError("source-threshold aligned output is not canonical")
    if expected_output is not None and output != str(Path(expected_output).expanduser().resolve()):
        raise ValueError("source-threshold aligned output differs")
    authority.update(
        {
            "implementation": formal.record(authority["implementation"], label="aligned implementation"),
            "implementation_dependencies": dependencies,
            "source_threshold_envelope": threshold_record,
            "raw_query_relevance": raw_record,
            "verified_threshold": threshold,
            "verified_raw_payload": raw_payload,
            "verified_raw_execution": raw_execution,
            "verified_record": {"path": str(source), "sha256": digest},
            "aligned_relevance_output": output,
        }
    )
    return authority


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output, label="aligned relevance output")
    execution = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        expected_output=output,
    )
    raw = execution["verified_raw_payload"]
    threshold_result = execution["verified_threshold"]
    threshold = float(threshold_result["thresholds"]["train_selected_candidate"])
    margin, aligned = formal.boundary_align(
        raw["region_absolute_relevance"], threshold=threshold
    )
    payload = {
        "schema": formal.RELEVANCE_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.relevance_contract(),
        "contract_sha256": formal.RELEVANCE_CONTRACT_SHA256,
        "scene_id": raw["scene_id"],
        "physical_space_id": raw["physical_space_id"],
        "producer": file_record(IMPLEMENTATION),
        "execution_authority": dict(execution["verified_record"]),
        "input_authority": {
            "source_threshold_envelope": execution["source_threshold_envelope"],
            "raw_query_relevance": execution["raw_query_relevance"],
            "raw_query_execution_authority": execution["raw_query_execution_authority"],
            "source_result": execution["source_result"],
            "source_checkpoint": execution["source_checkpoint"],
            "target_descriptor": execution["target_descriptor"],
            "health_v4_audit": execution["health_v4_audit"],
            "exact_query_manifest": execution["exact_query_manifest"],
            "positive_text_cache": execution["positive_text_cache"],
            "all_query_text_cache": execution["all_query_text_cache"],
            "canonical_negative_bank": execution["canonical_negative_bank"],
        },
        "source_global_margin_threshold": threshold,
        "raw_probability_boundary": float(
            torch.sigmoid(torch.tensor(formal.LOGIT_SCALE * threshold, dtype=torch.float64))
        ),
        "region_row_ids": list(raw["region_row_ids"]),
        "canonical_region_indices": raw["canonical_region_indices"].clone(),
        "region_fingerprints": list(raw["region_fingerprints"]),
        "query_ids": list(raw["query_ids"]),
        "recovered_margin": margin,
        "region_boundary_aligned_relevance": aligned,
        "coverage_audit": formal.query_coverage_audit(
            query_ids=list(raw["query_ids"]),
            raw_probability=raw["region_absolute_relevance"],
            aligned_probability=aligned,
        ),
        "rank_invariance_audit": {
            "per_query_strict_order_preserved": True,
            "queries_checked": len(raw["query_ids"]),
            "regions_checked": len(raw["region_row_ids"]),
            "ranking_normalization": False,
        },
        "access_audit": formal.access_audit(),
    }
    payload["channel_sha256"] = formal.channel_sha256(payload)
    payload = formal.validate_relevance(
        payload, raw_payload=raw, source_threshold_result=threshold_result
    )
    write_torch_noclobber(output, payload)
    return {
        "status": "source_threshold_aligned_target_relevance_complete",
        "shape": list(aligned.shape),
        "threshold": threshold,
        "coverage_audit": payload["coverage_audit"],
        "output": file_record(output),
        "metric_computed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--source-threshold-envelope", required=True)
    build.add_argument("--expected-source-threshold-envelope-sha256", required=True)
    build.add_argument("--raw-query-relevance", required=True)
    build.add_argument("--expected-raw-query-relevance-sha256", required=True)
    build.add_argument("--aligned-relevance-output", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_authority)
    validate = commands.add_parser("validate-authority")
    validate.add_argument("--execution-authority", required=True)
    validate.add_argument("--expected-execution-authority-sha256", required=True)
    validate.add_argument("--expected-output")
    validate.set_defaults(handler=lambda args: {
        "status": "source_threshold_aligned_relevance_authority_valid",
        "authority": validate_authority(
            args.execution_authority,
            expected_sha256=args.expected_execution_authority_sha256,
            expected_output=args.expected_output,
        )["verified_record"],
    })
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_STATUS", "IMPLEMENTATION", "IMPLEMENTATION_DEPENDENCIES",
    "build_authority", "build_parser", "materialize", "validate_authority",
]
