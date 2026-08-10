#!/usr/bin/env python3
"""Build the explicit frozen-relative external cache and metric authority."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import posixpath
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_primitive_readout
    as primitive_formal,
)
from radio_gs.interfaces import (
    factorized_native_contrast_v21_frozen_relative_readout as relative_formal,
)
from radio_gs.interfaces import (
    lerf_v21_frozen_relative_frozen_metric_adapter as formal,
)
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.scripts import (
    build_lerf_v21_native_v3_frozen_metric_bridge as legacy_builder,
)
from radio_gs.scripts import (
    materialize_factorized_native_contrast_v21_frozen_relative_primitive_readout
    as primitive_script,
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
LAUNCHER = Path(__file__).resolve().parent / "run_lerf_v21_frozen_relative_metric.py"


def _argument_record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    record = formal.record({"path": canonical, "sha256": str(digest)}, label=label)
    validate_file_record(record, label=label)
    return record


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical path")
    return path


def _unopened(value: object, *, label: str) -> str:
    raw = str(value)
    if not raw.startswith("/") or posixpath.normpath(raw) != raw:
        raise ValueError(f"{label} must be canonical absolute")
    return raw


def _load_mapping(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    raw, digest, source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label=label,
    )
    if {"path": str(source), "sha256": digest} != dict(record):
        raise ValueError(f"{label} record differs")
    return raw


def _materialize_inputs(args: argparse.Namespace) -> dict[str, Any]:
    records = {
        "relevance_authority": _argument_record(
            args.relevance_authority,
            args.expected_relevance_authority_sha256,
            label="health-gated exact relevance",
        ),
        "frozen_relative_readout_authority": _argument_record(
            args.frozen_relative_readout_authority,
            args.expected_frozen_relative_readout_authority_sha256,
            label="frozen-relative readout",
        ),
        "primitive_readout_authority": _argument_record(
            args.primitive_readout_authority,
            args.expected_primitive_readout_authority_sha256,
            label="frozen-relative primitive readout",
        ),
        "renderer_geometry_checkpoint": _argument_record(
            args.renderer_geometry_checkpoint,
            args.expected_renderer_geometry_checkpoint_sha256,
            label="renderer geometry checkpoint",
        ),
        "exact_query_manifest": _argument_record(
            args.exact_query_manifest,
            args.expected_exact_query_manifest_sha256,
            label="exact query manifest",
        ),
        "all_query_text_cache": _argument_record(
            args.all_query_text_cache,
            args.expected_all_query_text_cache_sha256,
            label="all-query text cache",
        ),
        "canonical_negative_text_cache": _argument_record(
            args.canonical_negative_text_cache,
            args.expected_canonical_negative_text_cache_sha256,
            label="canonical-negative cache",
        ),
    }
    relevance, execution = legacy_builder._load_health_gated_relevance_chain(
        records["relevance_authority"]
    )
    relative = relative_formal.validate_readout_authority(
        _load_mapping(
            records["frozen_relative_readout_authority"],
            label="frozen-relative readout",
        )
    )
    primitive = primitive_formal.validate_readout_authority(
        _load_mapping(
            records["primitive_readout_authority"],
            label="frozen-relative primitive readout",
        )
    )
    primitive_execution_record = formal.record(
        primitive["execution_authority"], label="primitive execution"
    )
    primitive_execution = primitive_script.validate_authority(
        primitive_execution_record["path"],
        expected_sha256=primitive_execution_record["sha256"],
        expected_output=records["primitive_readout_authority"]["path"],
    )
    if (
        primitive_execution["input_authority"]["frozen_relative_readout"]
        != records["frozen_relative_readout_authority"]
    ):
        raise ValueError("primitive authority binds another relative readout")
    state_record = formal.record(
        primitive["input_authority"]["factorized_primitive_state"],
        label="factorized primitive state",
    )
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    records["factorized_primitive_state"] = state_record
    renderer_raw, renderer_sha, renderer_path = (
        load_sha_bound_project_checkpoint_mapping(
            records["renderer_geometry_checkpoint"]["path"],
            expected_sha256=records["renderer_geometry_checkpoint"]["sha256"],
            map_location="cpu",
            label="frozen-relative metric renderer",
        )
    )
    if {
        "path": str(renderer_path),
        "sha256": renderer_sha,
    } != records["renderer_geometry_checkpoint"]:
        raise ValueError("frozen-relative metric renderer record differs")
    renderer_xyz = _renderer_checkpoint_xyz(renderer_raw)
    cache = formal.build_external_query_score_cache(
        validated_relevance=relevance,
        verified_query_execution=execution,
        validated_relative=relative,
        validated_primitive=primitive,
        relevance_record=records["relevance_authority"],
        relative_record=records["frozen_relative_readout_authority"],
        primitive_record=records["primitive_readout_authority"],
        renderer_record=records["renderer_geometry_checkpoint"],
        manifest_record=records["exact_query_manifest"],
        all_query_record=records["all_query_text_cache"],
        negative_record=records["canonical_negative_text_cache"],
        state_record=state_record,
        state_xyz=state.xyz,
        state_valid=state.valid,
        renderer_xyz=renderer_xyz,
    )
    return {
        "records": records,
        "relevance": relevance,
        "execution": execution,
        "relative": relative,
        "primitive": primitive,
        "cache": cache,
    }


def materialize_cache(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output_cache, label="frozen-relative external cache")
    receipt_output = _new(args.output_receipt, label="external cache receipt")
    chain = _materialize_inputs(args)
    write_torch_noclobber(output, chain["cache"])
    cache_record = file_record(output)
    receipt = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": "frozen_relative_external_query_score_cache_complete",
        "scene_id": chain["cache"]["metadata"]["scene_id"],
        "query_axis_count": len(chain["cache"]["metadata"]["query_names"]),
        "primitive_memberships": int(chain["cache"]["query_scores"].sum()),
        "output_cache": cache_record,
        "input_authority": chain["records"],
        "implementation": file_record(IMPLEMENTATION),
        "access_audit": formal.cache_access_audit(),
    }
    write_frozen_json(receipt_output, receipt)
    return {
        "status": receipt["status"],
        "output_cache": cache_record,
        "receipt": file_record(receipt_output),
        "query_axis_count": receipt["query_axis_count"],
    }


def _load_cache(record: Mapping[str, str]) -> dict[str, Any]:
    return formal.validate_external_query_score_cache(
        _load_mapping(record, label="frozen-relative external cache")
    )


def build_metric_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output_authority, label="frozen-relative metric authority")
    cache_record = _argument_record(
        args.external_query_score_cache,
        args.expected_external_query_score_cache_sha256,
        label="frozen-relative external cache",
    )
    cache = _load_cache(cache_record)
    chain = _materialize_inputs(args)
    rebuilt = chain["cache"]
    if (
        cache["metadata"] != rebuilt["metadata"]
        or cache["channel_sha256"] != rebuilt["channel_sha256"]
        or not torch.equal(cache["query_scores"], rebuilt["query_scores"])
        or not torch.equal(cache["valid"], rebuilt["valid"])
        or not torch.equal(cache["xyz"], rebuilt["xyz"])
    ):
        raise ValueError("external cache is not the rebuilt frozen-relative chain")
    config_record = _argument_record(
        args.config, args.expected_config_sha256, label="frozen LERF config"
    )
    validate_file_record(formal.FROZEN_EVALUATOR, label="frozen evaluator")
    validate_file_record(formal.FROZEN_SUMMARY_HEAD, label="frozen summary head")
    if not LAUNCHER.is_file() or LAUNCHER.is_symlink():
        raise ValueError("frozen-relative metric launcher is unavailable")
    metadata = cache["metadata"]
    authority = {
        "schema": formal.METRIC_AUTHORITY_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "contract": formal.metric_authority_contract(),
        "contract_sha256": formal.METRIC_AUTHORITY_CONTRACT_SHA256,
        "status": "authorized_single_frozen_relative_lerf_metric",
        "scene_id": metadata["scene_id"],
        "physical_space_id": metadata["physical_space_id"],
        "implementation": file_record(IMPLEMENTATION),
        "launcher": file_record(LAUNCHER),
        "frozen_evaluator": formal.FROZEN_EVALUATOR,
        "frozen_summary_head": formal.FROZEN_SUMMARY_HEAD,
        "external_query_score_cache": cache_record,
        "relevance_authority": metadata["relevance_authority"],
        "frozen_relative_readout_authority": metadata[
            "frozen_relative_readout_authority"
        ],
        "primitive_readout_authority": metadata["primitive_readout_authority"],
        "renderer_geometry_checkpoint": metadata[
            "renderer_geometry_checkpoint"
        ],
        "exact_query_manifest": metadata["exact_query_manifest"],
        "all_query_text_cache": metadata["all_query_text_cache"],
        "canonical_negative_text_cache": metadata[
            "canonical_negative_text_cache"
        ],
        "config": config_record,
        "label_root": _unopened(args.label_root, label="label root"),
        "output_dir": _unopened(args.output_dir, label="metric output"),
        "protocol": formal.METRIC_PROTOCOL,
        "single_candidate_no_sweep": True,
        "scene_specific_parameters": False,
        "metric_execution_authorized": True,
        "access_audit": formal.metric_build_access_audit(),
    }
    formal.validate_metric_authority(authority)
    write_frozen_json(output, authority)
    return {
        "status": "frozen_relative_metric_authority_built_without_gt_access",
        "authority": file_record(output),
        "scene_id": authority["scene_id"],
        "protocol": authority["protocol"],
    }


def validate_metric_authority(
    path: str | Path, *, expected_sha256: str
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="frozen-relative metric authority",
    )
    authority = formal.validate_metric_authority(raw)
    for name in (
        "implementation",
        "launcher",
        "frozen_evaluator",
        "frozen_summary_head",
        "external_query_score_cache",
        "relevance_authority",
        "frozen_relative_readout_authority",
        "primitive_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
    ):
        validate_file_record(authority[name], label=f"metric {name}")
    cache = _load_cache(authority["external_query_score_cache"])
    metadata = cache["metadata"]
    bindings = {
        "relevance_authority": "relevance_authority",
        "frozen_relative_readout_authority": (
            "frozen_relative_readout_authority"
        ),
        "primitive_readout_authority": "primitive_readout_authority",
        "renderer_geometry_checkpoint": "renderer_geometry_checkpoint",
        "exact_query_manifest": "exact_query_manifest",
        "all_query_text_cache": "all_query_text_cache",
        "canonical_negative_text_cache": "canonical_negative_text_cache",
    }
    if (
        authority["scene_id"] != metadata["scene_id"]
        or authority["physical_space_id"] != metadata["physical_space_id"]
        or any(authority[key] != metadata[value] for key, value in bindings.items())
    ):
        raise ValueError("metric authority/cache binding differs")
    authority["verified_cache"] = cache
    authority["verified_record"] = {"path": str(source), "sha256": digest}
    return authority


def _add_input(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("materialize-cache")
    for name in (
        "relevance_authority",
        "frozen_relative_readout_authority",
        "primitive_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        _add_input(cache, name)
    cache.add_argument("--output-cache", required=True)
    cache.add_argument("--output-receipt", required=True)
    cache.set_defaults(handler=materialize_cache)
    metric = subparsers.add_parser("build-metric-authority")
    for name in (
        "external_query_score_cache",
        "relevance_authority",
        "frozen_relative_readout_authority",
        "primitive_readout_authority",
        "renderer_geometry_checkpoint",
        "exact_query_manifest",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "config",
    ):
        _add_input(metric, name)
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
