#!/usr/bin/env python3
"""Materialize and freeze the native-V3 LERF external-score metric bridge.

Neither subcommand opens the label root or runs the evaluator.  The exact
query axis is restored only after the contrast-V2.1 health-gated execution
authority has passed its complete validator.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import posixpath
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_lerf_exact as contrast_relevance,
)
from radio_gs.interfaces import (
    lerf_v21_native_v3_frozen_metric_bridge as formal,
)
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces import (
    surface_region_v21_native_v3_absolute_readout as readout_formal,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_lerf_exact_relevance
    as contrast_materializer,
)
from radio_gs.scripts import (
    materialize_surface_region_v21_native_v3_absolute_readout
    as readout_materializer,
)
from radio_gs.scripts.materialize_lerf_multiscale_query_score_cache import (
    _renderer_checkpoint_xyz,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


IMPLEMENTATION = Path(__file__).resolve()
LAUNCHER = (
    Path(__file__).resolve().parent
    / "run_lerf_v21_native_v3_frozen_metric.py"
)


def _argument_record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    return formal.record(
        {"path": canonical, "sha256": str(digest)}, label=label
    )


def _new_file(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical path")
    return path


def _canonical_unopened_path(value: object, *, label: str) -> str:
    """Validate an absolute path lexically without stat/open on the target."""

    raw = str(value)
    if not raw.startswith("/") or posixpath.normpath(raw) != raw:
        raise ValueError(f"{label} must be canonical absolute")
    return raw


def _load_health_gated_relevance_chain(
    relevance_record: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate health lineage before returning any query identifier."""

    raw, digest, source = load_torch_mapping(
        relevance_record["path"],
        expected_sha256=relevance_record["sha256"],
        map_location="cpu",
        label="health-gated contrast-V2.1 exact relevance",
    )
    if {"path": str(source), "sha256": digest} != dict(relevance_record):
        raise ValueError("exact relevance record differs")
    # Only dispatch/schema and the execution record are inspected before the
    # full health gate.  In particular, query_ids is not accessed here.
    schema = raw.get("schema")
    if schema != contrast_relevance.QUERY_RELEVANCE_SCHEMA:
        raise ValueError(
            "unsupported exact relevance schema; add a fail-closed global "
            "source-calibrated validator before promotion"
        )
    execution_record = formal.record(
        raw.get("query_execution_authority"),
        label="exact relevance query execution",
    )
    execution = contrast_materializer.validate_authority(
        execution_record["path"],
        expected_sha256=execution_record["sha256"],
        expected_output=relevance_record["path"],
    )
    # This validator is deliberately called only after validate_authority has
    # completed the source/descriptor/health-v4/preregistration gate.
    relevance = contrast_relevance.validate_query_relevance(raw)
    if relevance["query_execution_authority"] != execution["verified_record"]:
        raise ValueError("exact relevance binds another health-gated execution")
    return relevance, execution


def _load_readout(
    readout_record: Mapping[str, str],
) -> dict[str, Any]:
    raw, digest, source = load_torch_mapping(
        readout_record["path"],
        expected_sha256=readout_record["sha256"],
        map_location="cpu",
        label="native-V3 absolute readout",
    )
    if {"path": str(source), "sha256": digest} != dict(readout_record):
        raise ValueError("native-V3 readout record differs")
    readout = readout_formal.validate_readout_authority(raw)
    producer = formal.record(readout.get("producer"), label="readout producer")
    if (
        validate_file_record(producer, label="readout producer")
        != Path(readout_materializer.__file__).resolve()
    ):
        raise ValueError("native-V3 readout producer differs")
    inputs = readout.get("input_authority")
    if not isinstance(inputs, Mapping):
        raise ValueError("native-V3 readout inputs differ")
    for name, value in inputs.items():
        validate_file_record(value, label=f"native-V3 readout input {name}")
    return readout


def _materialize_inputs(args: argparse.Namespace) -> dict[str, Any]:
    relevance_record = _argument_record(
        args.relevance_authority,
        args.expected_relevance_authority_sha256,
        label="health-gated exact relevance",
    )
    readout_record = _argument_record(
        args.native_v3_readout_authority,
        args.expected_native_v3_readout_authority_sha256,
        label="native-V3 absolute readout",
    )
    renderer_record = _argument_record(
        args.renderer_geometry_checkpoint,
        args.expected_renderer_geometry_checkpoint_sha256,
        label="renderer geometry checkpoint",
    )
    manifest_record = _argument_record(
        args.exact_query_manifest,
        args.expected_exact_query_manifest_sha256,
        label="frozen exact query manifest",
    )
    all_record = _argument_record(
        args.all_query_text_cache,
        args.expected_all_query_text_cache_sha256,
        label="frozen all-query text cache",
    )
    negative_record = _argument_record(
        args.canonical_negative_text_cache,
        args.expected_canonical_negative_text_cache_sha256,
        label="frozen canonical-negative text cache",
    )
    # The health gate precedes query-axis restoration.  Independent supplied
    # protocol records are compared only after the gate has passed.
    relevance, execution = _load_health_gated_relevance_chain(relevance_record)
    readout = _load_readout(readout_record)
    readout_inputs = readout["input_authority"]
    state_record = formal.record(
        readout_inputs["factorized_primitive_state"],
        label="readout factorized primitive state",
    )
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    renderer_raw, renderer_sha, renderer_path = (
        load_sha_bound_project_checkpoint_mapping(
            renderer_record["path"],
            expected_sha256=renderer_record["sha256"],
            map_location="cpu",
            label="renderer geometry checkpoint",
        )
    )
    if {"path": str(renderer_path), "sha256": renderer_sha} != renderer_record:
        raise ValueError("renderer geometry checkpoint record differs")
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    cache = formal.build_external_query_score_cache(
        validated_relevance=relevance,
        verified_query_execution=execution,
        validated_readout=readout,
        relevance_record=relevance_record,
        readout_record=readout_record,
        renderer_geometry_record=renderer_record,
        exact_query_manifest_record=manifest_record,
        all_query_cache_record=all_record,
        canonical_negative_cache_record=negative_record,
        factorized_state_record=state_record,
        state_xyz=state.xyz.detach().cpu().float().contiguous(),
        state_valid=state.valid.detach().cpu().bool().contiguous(),
        renderer_xyz=renderer_xyz,
    )
    return {
        "cache": cache,
        "relevance": relevance,
        "execution": execution,
        "readout": readout,
        "records": {
            "relevance_authority": relevance_record,
            "native_v3_readout_authority": readout_record,
            "renderer_geometry_checkpoint": renderer_record,
            "exact_query_manifest": manifest_record,
            "all_query_text_cache": all_record,
            "canonical_negative_text_cache": negative_record,
            "factorized_primitive_state": state_record,
        },
    }


def materialize_cache(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_file(args.output_cache, label="external query-score cache")
    receipt_output = _new_file(args.output_receipt, label="cache receipt")
    chain = _materialize_inputs(args)
    write_torch_noclobber(output, chain["cache"])
    cache_record = file_record(output)
    receipt = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "native_v3_external_query_score_cache_complete",
        "scene_id": chain["cache"]["metadata"]["scene_id"],
        "physical_space_id": chain["cache"]["metadata"]["physical_space_id"],
        "query_axis_count": len(chain["cache"]["metadata"]["query_names"]),
        "query_axis_sha256": chain["cache"]["metadata"]["query_axis_sha256"],
        "primitive_count": int(chain["cache"]["query_scores"].shape[0]),
        "selected_memberships": int(chain["cache"]["query_scores"].sum()),
        "output_cache": cache_record,
        "input_authority": dict(chain["records"]),
        "implementation": file_record(IMPLEMENTATION),
        "access_audit": formal.cache_access_audit(),
    }
    write_frozen_json(receipt_output, receipt)
    return {
        "status": receipt["status"],
        "output_cache": cache_record,
        "receipt": file_record(receipt_output),
        "query_axis_count": receipt["query_axis_count"],
        "access_audit": receipt["access_audit"],
    }


def _validate_cache_file(record_value: Mapping[str, str]) -> dict[str, Any]:
    payload, digest, source = load_torch_mapping(
        record_value["path"],
        expected_sha256=record_value["sha256"],
        map_location="cpu",
        label="native-V3 external query-score cache",
    )
    if {"path": str(source), "sha256": digest} != dict(record_value):
        raise ValueError("external query-score cache record differs")
    return formal.validate_external_query_score_cache(payload)


def build_metric_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_file(args.output_authority, label="metric authority")
    cache_record = _argument_record(
        args.external_query_score_cache,
        args.expected_external_query_score_cache_sha256,
        label="external query-score cache",
    )
    cache = _validate_cache_file(cache_record)
    # Re-run the complete immutable relevance/readout/renderer chain.  A
    # self-consistent cache payload is not sufficient provenance on its own.
    rebuilt_chain = _materialize_inputs(args)
    rebuilt_cache = rebuilt_chain["cache"]
    if (
        cache["metadata"] != rebuilt_cache["metadata"]
        or cache["channel_sha256"] != rebuilt_cache["channel_sha256"]
        or not torch.equal(cache["query_scores"], rebuilt_cache["query_scores"])
        or not torch.equal(cache["valid"], rebuilt_cache["valid"])
        or not torch.equal(cache["xyz"], rebuilt_cache["xyz"])
    ):
        raise ValueError("metric cache is not the exact rebuilt validated bridge")
    records = {
        "relevance_authority": _argument_record(
            args.relevance_authority,
            args.expected_relevance_authority_sha256,
            label="health-gated exact relevance",
        ),
        "native_v3_readout_authority": _argument_record(
            args.native_v3_readout_authority,
            args.expected_native_v3_readout_authority_sha256,
            label="native-V3 absolute readout",
        ),
        "renderer_geometry_checkpoint": _argument_record(
            args.renderer_geometry_checkpoint,
            args.expected_renderer_geometry_checkpoint_sha256,
            label="renderer geometry checkpoint",
        ),
        "exact_query_manifest": _argument_record(
            args.exact_query_manifest,
            args.expected_exact_query_manifest_sha256,
            label="frozen exact query manifest",
        ),
        "all_query_text_cache": _argument_record(
            args.all_query_text_cache,
            args.expected_all_query_text_cache_sha256,
            label="frozen all-query text cache",
        ),
        "canonical_negative_text_cache": _argument_record(
            args.canonical_negative_text_cache,
            args.expected_canonical_negative_text_cache_sha256,
            label="frozen canonical-negative text cache",
        ),
        "config": _argument_record(
            args.config,
            args.expected_config_sha256,
            label="frozen LERF config",
        ),
    }
    for name, value in records.items():
        validate_file_record(value, label=f"metric input {name}")
    metadata = cache["metadata"]
    for name in (
        "relevance_authority",
        "native_v3_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        cache_name = (
            "readout_authority"
            if name == "native_v3_readout_authority"
            else name
        )
        if metadata[cache_name] != records[name]:
            raise ValueError(f"metric/cache lineage differs: {name}")
    validate_file_record(formal.FROZEN_EVALUATOR, label="frozen evaluator")
    validate_file_record(formal.FROZEN_SUMMARY_HEAD, label="frozen summary head")
    if not LAUNCHER.is_file() or LAUNCHER.is_symlink():
        raise ValueError("frozen metric launcher is unavailable")
    label_root = _canonical_unopened_path(args.label_root, label="label root")
    output_dir = _canonical_unopened_path(args.output_dir, label="metric output")
    authority = {
        "schema": formal.METRIC_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.metric_authority_contract(),
        "contract_sha256": formal.METRIC_AUTHORITY_CONTRACT_SHA256,
        "status": "authorized_single_frozen_native_v3_lerf_metric",
        "scene_id": metadata["scene_id"],
        "physical_space_id": metadata["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "launcher": file_record(LAUNCHER),
        "frozen_evaluator": dict(formal.FROZEN_EVALUATOR),
        "frozen_summary_head": dict(formal.FROZEN_SUMMARY_HEAD),
        "external_query_score_cache": cache_record,
        **records,
        "label_root": label_root,
        "output_dir": output_dir,
        "protocol": dict(formal.METRIC_PROTOCOL),
        "single_candidate_no_sweep": True,
        "scene_specific_parameters": False,
        "metric_execution_authorized": True,
        "access_audit": formal.metric_build_access_audit(),
    }
    formal.validate_metric_authority_payload(authority)
    write_frozen_json(output, authority)
    return {
        "status": "native_v3_frozen_metric_authority_built_without_gt_access",
        "authority": file_record(output),
        "scene_id": authority["scene_id"],
        "protocol": authority["protocol"],
        "access_audit": authority["access_audit"],
    }


def validate_metric_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="native-V3 frozen metric authority",
    )
    authority = formal.validate_metric_authority_payload(raw)
    expected_paths = {
        "implementation": IMPLEMENTATION,
        "launcher": LAUNCHER,
        "frozen_evaluator": Path(formal.FROZEN_EVALUATOR["path"]),
        "frozen_summary_head": Path(formal.FROZEN_SUMMARY_HEAD["path"]),
    }
    for name, expected in expected_paths.items():
        if validate_file_record(authority[name], label=name) != expected.resolve():
            raise ValueError(f"metric authority {name} differs")
    for name in (
        "relevance_authority",
        "native_v3_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
    ):
        validate_file_record(authority[name], label=f"metric input {name}")
    cache = _validate_cache_file(authority["external_query_score_cache"])
    metadata = cache["metadata"]
    bindings = {
        "relevance_authority": "relevance_authority",
        "native_v3_readout_authority": "readout_authority",
        "renderer_geometry_checkpoint": "renderer_geometry_checkpoint",
        "exact_query_manifest": "exact_query_manifest",
        "all_query_text_cache": "all_query_text_cache",
        "canonical_negative_text_cache": "canonical_negative_text_cache",
    }
    if (
        authority["scene_id"] != metadata["scene_id"]
        or authority["physical_space_id"] != metadata["physical_space_id"]
        or any(authority[name] != metadata[cache_name] for name, cache_name in bindings.items())
    ):
        raise ValueError("metric authority/cache binding differs")
    authority["verified_cache"] = cache
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def _add_bound_input(
    parser: argparse.ArgumentParser, name: str, *, digest_name: str | None = None
) -> None:
    option = "--" + name.replace("_", "-")
    parser.add_argument(option, required=True)
    digest = digest_name or f"expected_{name}_sha256"
    parser.add_argument("--" + digest.replace("_", "-"), required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("materialize-cache")
    for name in (
        "relevance_authority",
        "native_v3_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        _add_bound_input(cache, name)
    cache.add_argument("--output-cache", required=True)
    cache.add_argument("--output-receipt", required=True)
    cache.set_defaults(handler=materialize_cache)

    metric = subparsers.add_parser("build-metric-authority")
    for name in (
        "external_query_score_cache",
        "relevance_authority",
        "native_v3_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
    ):
        _add_bound_input(metric, name)
    metric.add_argument("--label-root", required=True)
    metric.add_argument("--output-dir", required=True)
    metric.add_argument("--output-authority", required=True)
    metric.set_defaults(handler=build_metric_authority)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "IMPLEMENTATION",
    "LAUNCHER",
    "build_metric_authority",
    "build_parser",
    "materialize_cache",
    "validate_metric_authority",
]
