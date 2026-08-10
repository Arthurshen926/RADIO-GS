#!/usr/bin/env python3
"""Materialize a no-GT exact-query cache from a sealed full O0 lift.

This command has no metric-authority or benchmark-quality entry point.  A
fixed safety gate compares the candidate only with the frozen O0 unary before
an evaluator-compatible external score cache can be emitted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import rank256_o0_full_lift_premetric as formal
from radio_gs.querying.v21_absolute_relevance_adapter import (
    load_v21_positive_text_bank,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_lerf_exact_relevance
    as exact_query,
)
from radio_gs.scripts import materialize_factorized_native_dba_v2_figurines as dba
from radio_gs.scripts import materialize_rank256_o0_primitive_lifting_dryrun as lift
from radio_gs.scripts.build_lerf_o0_anchored_graph_residual_cache import (
    exact_o0_readout,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
INTERFACE = Path(formal.__file__).resolve()
TESTS = Path(__file__).resolve().parents[2] / "tests/test_rank256_o0_full_lift_premetric.py"
EXECUTION_SCHEMA = "radio_gs.rank256_o0_full_lift_premetric_execution.v1"
EXECUTION_STATUS = "authorized_no_GT_exact_query_premetric_cache_only"
AUDIT_SCHEMA = "radio_gs.rank256_o0_full_lift_premetric_audit.v1"
EXPECTED_CONFIGURATION = {
    "max_angle_radians": 0.15,
    "minimum_region_reliability": 0.0,
}


def authority_access_audit() -> dict[str, bool]:
    return {
        "source_promotion_and_query_free_lift_validated_first": True,
        "sealed_full_lift_record_opened": True,
        "frozen_exact_query_protocol_opened_after_lift_gate": True,
        "frozen_O0_score_records_opened": True,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "target_metrics_computed": False,
        "metric_execution_authorized": False,
    }


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical absolute path")
    return path


def _record(path: object, digest: object, *, label: str) -> dict[str, str]:
    record = {"path": str(Path(str(path)).expanduser().resolve()), "sha256": str(digest)}
    if str(path) != record["path"]:
        raise ValueError(f"{label} path must be canonical and absolute")
    validate_file_record(record, label=label)
    return record


def _mapping_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    return _record(value["path"], value["sha256"], label=label)


def _validate_exact_query_protocol_authority(
    record: Mapping[str, str], *, scene_id: str, physical_space_id: str
) -> dict[str, Any]:
    """Validate only the frozen text protocol, not an unrelated target model."""

    raw, digest, source = load_json_object(
        record["path"],
        expected_sha256=record["sha256"],
        label="full-lift exact query protocol authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "implementation",
        "implementation_dependencies",
        "source_result",
        "target_descriptor",
        "health_v4_audit",
        "health_v4_preregistration",
        "query_preregistration",
        "exact_query_manifest",
        "positive_text_cache",
        "all_query_text_cache",
        "canonical_negative_bank",
        "query_relevance_output",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema") != exact_query.formal.QUERY_EXECUTION_SCHEMA
        or raw.get("schema_version") != exact_query.formal.SCHEMA_VERSION
        or raw.get("scene_id") != scene_id
        or raw.get("physical_space_id") != physical_space_id
        or raw.get("query_execution_authorized") is not True
        or raw.get("metric_execution_authorized") is not False
    ):
        raise ValueError("full-lift exact query protocol authority differs")
    manifest_record = _mapping_record(
        raw["exact_query_manifest"], label="full-lift exact query manifest"
    )
    positive_record = _mapping_record(
        raw["positive_text_cache"], label="full-lift positive text cache"
    )
    all_query_record = _mapping_record(
        raw["all_query_text_cache"], label="full-lift all-query text cache"
    )
    negative_record = _mapping_record(
        raw["canonical_negative_bank"], label="full-lift negative text cache"
    )
    if (
        all_query_record != exact_query.formal.FROZEN_ALL_QUERY_CACHE
        or negative_record != exact_query.formal.FROZEN_CANONICAL_NEGATIVE_BANK
    ):
        raise ValueError("full-lift frozen text singleton differs")
    manifest, positive, negative = exact_query._validate_exact_text_protocol(
        scene_id=scene_id,
        manifest_record=manifest_record,
        positive_record=positive_record,
    )
    if len(positive.query_ids) != 21:
        raise ValueError("full-lift exact query count differs")
    return {
        "record": {"path": str(source), "sha256": digest},
        "manifest_record": manifest_record,
        "positive_record": positive_record,
        "all_query_record": all_query_record,
        "negative_record": negative_record,
        "manifest": manifest,
        "positive": positive,
        "negative": negative,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    # This validation is deliberately first and cannot receive query records.
    parent = lift.validate_authority(
        args.full_lift_execution_authority,
        expected_sha256=args.expected_full_lift_execution_authority_sha256,
        expected_output=args.full_lift_descriptor,
    )
    if (
        parent["scene_id"] != "figurines"
        or parent["configuration_object"].to_dict() != EXPECTED_CONFIGURATION
        or parent["scope"]["prefix_order"]
        != "o0_global_rows_ascending_storage_order"
        or parent["scope"]["valid_row_prefix_limit"] <= 0
    ):
        raise ValueError("full-lift frozen parent configuration differs")
    lift_record = _record(
        args.full_lift_descriptor,
        args.expected_full_lift_descriptor_sha256,
        label="sealed full-lift descriptor",
    )
    query_record = _record(
        args.exact_query_authority,
        args.expected_exact_query_authority_sha256,
        label="exact query protocol authority",
    )
    query = _validate_exact_query_protocol_authority(
        query_record,
        scene_id=parent["scene_id"],
        physical_space_id=(
            f"lerf:{parent['scene_id']}:geometry-checkpoint-sha256:"
            f"{parent['input_authority']['renderer_geometry_checkpoint']['sha256']}"
        ),
    )
    positive = _record(
        args.positive_o0_scores,
        args.expected_positive_o0_scores_sha256,
        label="positive exact O0 raw scores",
    )
    negative = _record(
        args.negative_o0_scores,
        args.expected_negative_o0_scores_sha256,
        label="negative exact O0 raw scores",
    )
    output = _new(args.output_cache, label="full-lift external cache")
    audit = _new(args.output_audit, label="full-lift premetric audit")
    authority_output = _new(args.output_authority, label="full-lift premetric authority")
    authority = {
        "schema": EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": EXECUTION_STATUS,
        "scene_id": parent["scene_id"],
        "physical_space_id": (
            f"lerf:{parent['scene_id']}:geometry-checkpoint-sha256:"
            f"{parent['input_authority']['renderer_geometry_checkpoint']['sha256']}"
        ),
        "implementation": file_record(IMPLEMENTATION),
        "interface": file_record(INTERFACE),
        "tests": file_record(TESTS),
        "premetric_contract": formal.premetric_contract(),
        "premetric_contract_sha256": formal.CONTRACT_SHA256,
        "input_authority": {
            "full_lift_execution_authority": dict(parent["verified_record"]),
            "full_lift_descriptor": lift_record,
            "exact_query_authority": dict(query["record"]),
            "positive_o0_scores": positive,
            "negative_o0_scores": negative,
        },
        "frozen_lift_configuration": EXPECTED_CONFIGURATION,
        "full_valid_domain_required": True,
        "output_cache": str(output),
        "output_audit": str(audit),
        "materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": authority_access_audit(),
    }
    write_frozen_json(authority_output, authority)
    return {
        "status": "rank256_O0_full_lift_premetric_authority_built",
        "authority": file_record(authority_output),
        "output_cache": str(output),
        "output_audit": str(audit),
    }


def validate_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="rank256 O0 full-lift premetric authority",
    )
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "implementation",
        "interface",
        "tests",
        "premetric_contract",
        "premetric_contract_sha256",
        "input_authority",
        "frozen_lift_configuration",
        "full_valid_domain_required",
        "output_cache",
        "output_audit",
        "materialization_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    authority = dict(raw) if isinstance(raw, Mapping) else {}
    if (
        set(authority) != required
        or authority.get("schema") != EXECUTION_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status") != EXECUTION_STATUS
        or authority.get("premetric_contract") != formal.premetric_contract()
        or authority.get("premetric_contract_sha256") != formal.CONTRACT_SHA256
        or authority.get("frozen_lift_configuration") != EXPECTED_CONFIGURATION
        or authority.get("full_valid_domain_required") is not True
        or authority.get("materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != authority_access_audit()
        or validate_file_record(authority.get("implementation"), label="full-lift materializer")
        != IMPLEMENTATION
        or validate_file_record(authority.get("interface"), label="full-lift interface")
        != INTERFACE
        or validate_file_record(authority.get("tests"), label="full-lift tests") != TESTS
    ):
        raise ValueError("full-lift premetric authority header differs")
    records = authority["input_authority"]
    expected_names = {
        "full_lift_execution_authority",
        "full_lift_descriptor",
        "exact_query_authority",
        "positive_o0_scores",
        "negative_o0_scores",
    }
    if not isinstance(records, Mapping) or set(records) != expected_names:
        raise ValueError("full-lift premetric input authority differs")
    records = {
        name: _mapping_record(records[name], label=f"full-lift input {name}")
        for name in sorted(expected_names)
    }
    parent = lift.validate_authority(
        records["full_lift_execution_authority"]["path"],
        expected_sha256=records["full_lift_execution_authority"]["sha256"],
        expected_output=records["full_lift_descriptor"]["path"],
    )
    if (
        parent["scene_id"] != authority["scene_id"]
        or parent["configuration_object"].to_dict() != EXPECTED_CONFIGURATION
        or authority["physical_space_id"]
        != f"lerf:{authority['scene_id']}:geometry-checkpoint-sha256:"
        f"{parent['input_authority']['renderer_geometry_checkpoint']['sha256']}"
    ):
        raise ValueError("full-lift parent/physical-space authority differs")
    query = _validate_exact_query_protocol_authority(
        records["exact_query_authority"],
        scene_id=authority["scene_id"],
        physical_space_id=authority["physical_space_id"],
    )
    for name in ("output_cache", "output_audit"):
        value = str(authority[name])
        if value != str(Path(value).expanduser().resolve()):
            raise ValueError(f"{name} path differs")
    authority.update(
        {
            "verified_record": {"path": str(source), "sha256": digest},
            "input_authority": records,
            "verified_parent": parent,
            "verified_query": query,
        }
    )
    return authority


def _validate_lift_output(
    payload: object,
    *,
    record: Mapping[str, str],
    parent: Mapping[str, Any],
    o0_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "physical_space_id",
        "source_variant",
        "producer",
        "lifting_implementation",
        "execution_authority",
        "input_authority",
        "lifting_contract",
        "lifting_contract_sha256",
        "configuration",
        "scope",
        "primitive_global_rows",
        "selected_region_positions",
        "lifted_descriptor",
        "region_contribution_mask",
        "coverage_count",
        "aggregate_reliability",
        "aggregate_residual_norm",
        "angular_step_radians",
        "updated_mask",
        "fallback_mask",
        "fallback_bitwise_o0",
        "tensor_sha256",
        "routing_audit",
        "formal_candidate_authorized",
        "query_execution_authorized",
        "metric_execution_authorized",
        "access_audit",
    }
    value = dict(payload) if isinstance(payload, Mapping) else {}
    rows = value.get("primitive_global_rows")
    descriptor = value.get("lifted_descriptor")
    fallback = value.get("fallback_mask")
    updated = value.get("updated_mask")
    coverage = value.get("coverage_count")
    angles = value.get("angular_step_radians")
    selected_regions = value.get("selected_region_positions")
    region_contribution = value.get("region_contribution_mask")
    aggregate_reliability = value.get("aggregate_reliability")
    aggregate_residual_norm = value.get("aggregate_residual_norm")
    o0_rows = torch.as_tensor(o0_descriptor["global_rows"]).long().cpu().contiguous()
    o0_features = torch.as_tensor(o0_descriptor["features_by_scale"]).detach().cpu().contiguous()
    expected_physical = (
        f"lerf:{parent['scene_id']}:geometry-checkpoint-sha256:"
        f"{parent['input_authority']['renderer_geometry_checkpoint']['sha256']}"
    )
    if (
        set(value) != required
        or value.get("schema") != lift.OUTPUT_SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != "complete_nonformal_query_free_prefix_dryrun"
        or value.get("scene_id") != parent["scene_id"]
        or value.get("physical_space_id") != expected_physical
        or value.get("source_variant") != parent["source_variant"]
        or value.get("producer") != parent["implementation"]
        or value.get("lifting_implementation") != parent["lifting_implementation"]
        or value.get("execution_authority") != parent["verified_record"]
        or value.get("input_authority") != parent["input_authority"]
        or value.get("lifting_contract") != parent["lifting_contract"]
        or value.get("lifting_contract_sha256")
        != parent["lifting_contract_sha256"]
        or value.get("configuration") != EXPECTED_CONFIGURATION
        or value.get("scope") != parent["scope"]
        or value.get("access_audit") != lift.access_audit()
        or value.get("formal_candidate_authorized") is not False
        or value.get("query_execution_authorized") is not False
        or value.get("metric_execution_authorized") is not False
        or value.get("fallback_bitwise_o0") is not True
        or not torch.is_tensor(rows)
        or rows.dtype != torch.int64
        or rows.device.type != "cpu"
        or not torch.equal(rows, o0_rows)
        or int(parent["scope"]["valid_row_prefix_limit"]) != int(o0_rows.numel())
        or not torch.is_tensor(descriptor)
        or descriptor.device.type != "cpu"
        or descriptor.shape != o0_features.shape
        or descriptor.dtype != o0_features.dtype
        or descriptor.shape[1:] != (3, 1536)
        or not bool(torch.isfinite(descriptor).all())
        or not torch.is_tensor(fallback)
        or fallback.dtype != torch.bool
        or fallback.shape != descriptor.shape[:2]
        or not torch.is_tensor(updated)
        or updated.dtype != torch.bool
        or updated.shape != fallback.shape
        or not torch.equal(fallback, ~updated)
        or not torch.is_tensor(coverage)
        or coverage.dtype != torch.int64
        or coverage.shape != (rows.numel(),)
        or bool((coverage < 0).any())
        or not torch.is_tensor(selected_regions)
        or selected_regions.dtype != torch.int64
        or selected_regions.device.type != "cpu"
        or selected_regions.ndim != 1
        or bool((selected_regions < 0).any())
        or int(torch.unique(selected_regions).numel()) != int(selected_regions.numel())
        or not torch.is_tensor(region_contribution)
        or region_contribution.dtype != torch.bool
        or region_contribution.shape != selected_regions.shape
        or not torch.is_tensor(aggregate_reliability)
        or aggregate_reliability.dtype != torch.float64
        or aggregate_reliability.shape != (rows.numel(),)
        or not bool(torch.isfinite(aggregate_reliability).all())
        or bool((aggregate_reliability < 0.0).any())
        or bool((aggregate_reliability > 1.0).any())
        or not torch.is_tensor(aggregate_residual_norm)
        or aggregate_residual_norm.dtype != torch.float64
        or aggregate_residual_norm.shape != (rows.numel(),)
        or not bool(torch.isfinite(aggregate_residual_norm).all())
        or bool((aggregate_residual_norm < 0.0).any())
        or not torch.is_tensor(angles)
        or angles.dtype != torch.float64
        or angles.shape != fallback.shape
        or not bool(torch.isfinite(angles).all())
        or bool((angles < 0.0).any())
        or float(angles.max()) > 0.15 + 1e-12
        or not torch.equal(
            descriptor[fallback].view(torch.uint8),
            o0_features[fallback].view(torch.uint8),
        )
    ):
        raise ValueError("sealed full-lift output/full-domain binding differs")
    expected_hashes = {
        "primitive_global_rows": tensor_sha256(rows),
        "o0_prefix_descriptor": tensor_sha256(o0_features),
        "lifted_descriptor": tensor_sha256(descriptor),
        "coverage_count": tensor_sha256(coverage),
        "angular_step_radians": tensor_sha256(angles),
    }
    if value.get("tensor_sha256") != expected_hashes:
        raise ValueError("sealed full-lift output tensor hashes differ")
    expected_routing = {
        "valid_prefix_primitives": int(rows.numel()),
        "intersecting_regions": int(selected_regions.numel()),
        "contributing_regions": int(region_contribution.sum()),
        "covered_primitives": int((coverage > 0).sum()),
        "updated_primitive_scales": int(updated.sum()),
        "fallback_primitive_scales": int(fallback.sum()),
        "maximum_angular_step_radians": float(angles.max()),
    }
    if value.get("routing_audit") != expected_routing:
        raise ValueError("sealed full-lift routing audit differs")
    value["verified_record"] = dict(record)
    return value


def execute(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    output_cache = _new(authority["output_cache"], label="full-lift external cache")
    output_audit = _new(authority["output_audit"], label="full-lift premetric audit")
    records = authority["input_authority"]
    parent = authority["verified_parent"]

    o0_descriptor_raw, _, _ = load_torch_mapping(
        parent["input_authority"]["o0_descriptor"]["path"],
        expected_sha256=parent["input_authority"]["o0_descriptor"]["sha256"],
        map_location="cpu",
        label="full-lift query-free O0 descriptor",
    )
    lift_raw, _, _ = load_torch_mapping(
        records["full_lift_descriptor"]["path"],
        expected_sha256=records["full_lift_descriptor"]["sha256"],
        map_location="cpu",
        label="sealed full-lift descriptor",
    )
    lifted = _validate_lift_output(
        lift_raw,
        record=records["full_lift_descriptor"],
        parent=parent,
        o0_descriptor=o0_descriptor_raw,
    )
    positive_raw, _, _ = load_torch_mapping(
        records["positive_o0_scores"]["path"],
        expected_sha256=records["positive_o0_scores"]["sha256"],
        map_location="cpu",
        label="positive frozen O0 raw scores",
    )
    negative_raw, _, _ = load_torch_mapping(
        records["negative_o0_scores"]["path"],
        expected_sha256=records["negative_o0_scores"]["sha256"],
        map_location="cpu",
        label="negative frozen O0 raw scores",
    )
    query = authority["verified_query"]
    query_ids = list(query["positive"].query_ids)
    dba._validate_o0_pair(
        positive_raw,
        negative_raw,
        query_ids=query_ids,
        physical_space_id=authority["physical_space_id"],
    )
    o0 = lift._validate_o0(
        o0_descriptor_raw,
        renderer_xyz=torch.as_tensor(positive_raw["xyz"]).float().cpu().contiguous(),
    )
    if (
        not torch.equal(o0["global_rows"], lifted["primitive_global_rows"])
        or not torch.equal(o0["valid"], torch.as_tensor(positive_raw["valid"]))
        or lifted["physical_space_id"] != authority["physical_space_id"]
        or query_ids != list(positive_raw["query_ids"])
    ):
        raise ValueError("full-lift O0/query/full-domain axes differ")

    raw = formal.exact_fp32_cosine_scores(
        lifted["lifted_descriptor"],
        positive_text=query["positive"].embeddings,
        negative_text=query["negative"].embeddings,
    )
    candidate_positive = formal.scatter_sparse_scores(
        raw.positive,
        global_rows=lifted["primitive_global_rows"],
        total_rows=int(o0["xyz"].shape[0]),
    )
    candidate_negative = formal.scatter_sparse_scores(
        raw.negative,
        global_rows=lifted["primitive_global_rows"],
        total_rows=int(o0["xyz"].shape[0]),
    )
    candidate = exact_o0_readout(
        positive_scores=candidate_positive,
        negative_scores=candidate_negative,
        xyz=o0["xyz"],
        valid=o0["valid"],
        chunk_size=formal.KNN_CHUNK_ROWS,
    )
    baseline = exact_o0_readout(
        positive_scores=torch.as_tensor(positive_raw["query_scores"]).float().cpu(),
        negative_scores=torch.as_tensor(negative_raw["query_scores"]).float().cpu(),
        xyz=o0["xyz"],
        valid=o0["valid"],
        chunk_size=formal.KNN_CHUNK_ROWS,
    )
    audit_core = formal.build_premetric_audit(
        candidate_scores=candidate.final_scores,
        o0_scores=baseline.final_scores,
        valid=o0["valid"],
        query_ids=query_ids,
        candidate_positive_raw_sparse=raw.positive,
        candidate_negative_raw_sparse=raw.negative,
        o0_positive_raw=positive_raw["query_scores"],
        o0_negative_raw=negative_raw["query_scores"],
        primitive_global_rows=lifted["primitive_global_rows"],
        fallback_mask=lifted["fallback_mask"],
        axes_exact=True,
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": audit_core["status"],
        "premetric_passed": audit_core["premetric_passed"],
        "execution_authority": dict(authority["verified_record"]),
        "input_authority": dict(records),
        "premetric_contract": formal.premetric_contract(),
        "premetric_contract_sha256": formal.CONTRACT_SHA256,
        "checks": audit_core["checks"],
        "aggregate": audit_core["aggregate"],
        "per_query": audit_core["per_query"],
        "readout_audit": {
            "primitive_rows": int(o0["valid"].numel()),
            "valid_primitive_rows": int(o0["valid"].sum()),
            "scale_count": 3,
            "query_count": len(query_ids),
            "candidate_selected_scale_indices": candidate.selected_scale_indices.tolist(),
            "o0_selected_scale_indices": baseline.selected_scale_indices.tolist(),
            "candidate_raw_smoothed_peaks": candidate.raw_smoothed_peaks.tolist(),
            "o0_raw_smoothed_peaks": baseline.raw_smoothed_peaks.tolist(),
        },
        "axis_invariants": {
            "full_O0_valid_domain": True,
            "primitive_global_rows_equal_where_O0_valid": True,
            "O0_descriptor_xyz_equals_score_xyz": True,
            "positive_negative_xyz_valid_scale_query_axes_exact": True,
            "fallback_descriptor_bitwise_O0": True,
            "all_score_tensors_FP32": True,
        },
        "metric_execution_authorized": False,
        "access_audit": formal.access_audit(),
    }
    if not audit_core["premetric_passed"]:
        write_frozen_json(output_audit, audit)
        raise RuntimeError(
            f"full-lift premetric gate rejected; audit={output_audit}"
        )
    external = formal.build_external_query_score_cache(
        query_scores=candidate.final_scores,
        valid=o0["valid"],
        xyz=o0["xyz"],
        query_ids=query_ids,
        scene_id=authority["scene_id"],
        physical_space_id=authority["physical_space_id"],
        input_authority=records,
    )
    write_torch_noclobber(output_cache, external)
    audit["output_cache"] = file_record(output_cache)
    write_frozen_json(output_audit, audit)
    return {
        "status": "rank256_O0_full_lift_premetric_PASS",
        "cache": file_record(output_cache),
        "audit": file_record(output_audit),
        "aggregate": audit["aggregate"],
        "metric_execution_authorized": False,
    }


def _add_record(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument(
        "--expected-" + name.replace("_", "-") + "-sha256", required=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    for name in (
        "full_lift_execution_authority",
        "full_lift_descriptor",
        "exact_query_authority",
        "positive_o0_scores",
        "negative_o0_scores",
    ):
        _add_record(build, name)
    build.add_argument("--output-cache", required=True)
    build.add_argument("--output-audit", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_authority)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--execution-authority", required=True)
    execute_parser.add_argument("--expected-execution-authority-sha256", required=True)
    execute_parser.set_defaults(handler=execute)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_SCHEMA",
    "EXECUTION_SCHEMA",
    "build_authority",
    "execute",
    "validate_authority",
]
