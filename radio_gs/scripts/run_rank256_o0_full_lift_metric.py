#!/usr/bin/env python3
"""Preregister and run one frozen LERF metric for a passed rank256 O0 lift.

The premetric materializer deliberately has no metric entry point.  This
separate launcher can authorize exactly one metric only after the sealed
cache has passed every fixed no-GT gate and has been bound to the frozen
renderer and evaluator inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import posixpath
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from radio_gs.interfaces import rank256_o0_full_lift_premetric as formal
from radio_gs.scripts import materialize_rank256_o0_full_lift_premetric_cache as premetric
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


SCHEMA = "radio_gs.rank256_o0_full_lift_frozen_metric_execution.v1"
STATUS = "authorized_single_rank256_O0_full_lift_frozen_LERF_metric"
IMPLEMENTATION = Path(__file__).resolve()
PROTOCOL = {
    "protocol_preset": "vala_paper_3d",
    "score_threshold": 0.6,
    "score_postprocess": "none",
    "selection_mode": "score_threshold",
    "projection_mode": "selected_only_alpha",
    "official_frames_only": True,
    "mask_refinement": "none",
    "alpha_binarization": "png_uint8_gt10",
    "silhouette_threshold": 10.0 / 255.0,
    "threshold_scan": False,
}
FROZEN_EVALUATOR = {
    "path": "/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py",
    "sha256": "8cb39acf08c4f90f6339002ef32022437f67e7cdaae80fc16206b49abbb917d5",
}
FROZEN_SUMMARY_HEAD = {
    "path": "/root/RADIO-GS/checkpoints/siglip2_summary_head.pth",
    "sha256": "41ccc47b2da9b1aed3ee1e80397dc721ec625e083054175c27698e8840b6263c",
}
FROZEN_FIGURINES_CONFIG = {
    "path": (
        "/root/RADIO-GS/radio_gs/configs/generated/query_consistency/"
        "lerf_figurines_radio_verified_pose.yaml"
    ),
    "sha256": "a17ada0f1d34cf043f04ddc2f6503c262845d1fed8b4550df8f5d79f2dbd8f11",
}


def _record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    value = {"path": canonical, "sha256": str(digest)}
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    validate_file_record(value, label=label)
    return value


def _mapping_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} record differs")
    return _record(value["path"], value["sha256"], label=label)


def _new(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical absolute path")
    return path


def _unopened(value: object, *, label: str) -> str:
    raw = str(value)
    if not raw.startswith("/") or posixpath.normpath(raw) != raw:
        raise ValueError(f"{label} must be canonical absolute")
    return raw


def _build_access_audit() -> dict[str, bool]:
    return {
        "passed_premetric_authority_cache_and_audit_validated": True,
        "frozen_evaluator_inputs_validated": True,
        "label_root_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "target_metrics_computed": False,
        "subprocess_started": False,
        "threshold_scan": False,
    }


def _load_passed_premetric(
    *,
    authority_record: Mapping[str, str],
    cache_record: Mapping[str, str],
    audit_record: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority = premetric.validate_authority(
        authority_record["path"], expected_sha256=authority_record["sha256"]
    )
    cache_raw, cache_digest, cache_path = load_torch_mapping(
        cache_record["path"],
        expected_sha256=cache_record["sha256"],
        map_location="cpu",
        label="rank256 full-lift external cache",
    )
    cache = formal.validate_external_query_score_cache(cache_raw)
    audit_raw, audit_digest, audit_path = load_json_object(
        audit_record["path"],
        expected_sha256=audit_record["sha256"],
        label="rank256 full-lift premetric audit",
    )
    audit = dict(audit_raw) if isinstance(audit_raw, Mapping) else {}
    required_audit = {
        "schema",
        "schema_version",
        "status",
        "premetric_passed",
        "execution_authority",
        "input_authority",
        "premetric_contract",
        "premetric_contract_sha256",
        "checks",
        "aggregate",
        "per_query",
        "readout_audit",
        "axis_invariants",
        "metric_execution_authorized",
        "access_audit",
        "output_cache",
    }
    if (
        set(audit) != required_audit
        or audit.get("schema") != premetric.AUDIT_SCHEMA
        or audit.get("schema_version") != 1
        or audit.get("status") != "PASS"
        or audit.get("premetric_passed") is not True
        or not isinstance(audit.get("checks"), Mapping)
        or not audit["checks"]
        or any(value is not True for value in audit["checks"].values())
        or audit.get("premetric_contract") != formal.premetric_contract()
        or audit.get("premetric_contract_sha256") != formal.CONTRACT_SHA256
        or audit.get("execution_authority") != dict(authority_record)
        or audit.get("input_authority") != authority["input_authority"]
        or audit.get("output_cache") != dict(cache_record)
        or audit.get("metric_execution_authorized") is not False
        or audit.get("access_audit") != formal.access_audit()
        or not isinstance(audit.get("axis_invariants"), Mapping)
        or not audit["axis_invariants"]
        or any(value is not True for value in audit["axis_invariants"].values())
        or authority["output_cache"] != cache_record["path"]
        or authority["output_audit"] != audit_record["path"]
        or cache["metadata"]["scene_id"] != authority["scene_id"]
        or cache["metadata"]["physical_space_id"] != authority["physical_space_id"]
        or cache["metadata"]["query_names"]
        != list(authority["verified_query"]["positive"].query_ids)
        or cache["metadata"]["metric_execution_authorized"] is not False
        or cache["metadata"]["input_authority"] != authority["input_authority"]
        or audit.get("aggregate", {}).get("query_count") != 21
        or audit.get("aggregate", {}).get("supported_queries") != 21
        or {"path": str(cache_path), "sha256": cache_digest} != dict(cache_record)
        or {"path": str(audit_path), "sha256": audit_digest} != dict(audit_record)
    ):
        raise ValueError("rank256 full-lift passed premetric binding differs")
    return authority, cache, audit


def _validate_frozen_bindings(
    *,
    frozen_inputs: Mapping[str, Mapping[str, str]],
    premetric_authority: Mapping[str, Any],
) -> None:
    """Fail closed unless evaluator and query inputs are the frozen records."""

    query = premetric_authority["verified_query"]
    parent_renderer = premetric_authority["verified_parent"]["input_authority"][
        "renderer_geometry_checkpoint"
    ]
    expected = {
        "frozen_evaluator": FROZEN_EVALUATOR,
        "frozen_summary_head": FROZEN_SUMMARY_HEAD,
        "config": FROZEN_FIGURINES_CONFIG,
        "renderer_geometry_checkpoint": parent_renderer,
        "all_query_text_cache": query["all_query_record"],
        "canonical_negative_text_cache": query["negative_record"],
    }
    if premetric_authority["scene_id"] != "figurines" or any(
        dict(frozen_inputs.get(name, {})) != dict(record)
        for name, record in expected.items()
    ):
        raise ValueError("rank256 full-lift frozen evaluator binding differs")


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new(args.output_authority, label="rank256 full-lift metric authority")
    premetric_record = _record(
        args.premetric_execution_authority,
        args.expected_premetric_execution_authority_sha256,
        label="rank256 full-lift premetric execution authority",
    )
    cache_record = _record(
        args.external_query_score_cache,
        args.expected_external_query_score_cache_sha256,
        label="rank256 full-lift external cache",
    )
    audit_record = _record(
        args.premetric_audit,
        args.expected_premetric_audit_sha256,
        label="rank256 full-lift premetric audit",
    )
    premetric_authority, cache, audit = _load_passed_premetric(
        authority_record=premetric_record,
        cache_record=cache_record,
        audit_record=audit_record,
    )
    frozen_inputs = {
        "frozen_evaluator": _record(
            args.frozen_evaluator,
            args.expected_frozen_evaluator_sha256,
            label="frozen evaluator",
        ),
        "frozen_summary_head": _record(
            args.frozen_summary_head,
            args.expected_frozen_summary_head_sha256,
            label="frozen summary head",
        ),
        "config": _record(
            args.config, args.expected_config_sha256, label="frozen scene config"
        ),
        "renderer_geometry_checkpoint": _record(
            args.renderer_geometry_checkpoint,
            args.expected_renderer_geometry_checkpoint_sha256,
            label="renderer geometry checkpoint",
        ),
        "all_query_text_cache": _record(
            args.all_query_text_cache,
            args.expected_all_query_text_cache_sha256,
            label="all-query text cache",
        ),
        "canonical_negative_text_cache": _record(
            args.canonical_negative_text_cache,
            args.expected_canonical_negative_text_cache_sha256,
            label="canonical-negative text cache",
        ),
    }
    _validate_frozen_bindings(
        frozen_inputs=frozen_inputs,
        premetric_authority=premetric_authority,
    )
    value = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": STATUS,
        "scene_id": premetric_authority["scene_id"],
        "implementation": file_record(IMPLEMENTATION),
        "premetric_implementation": file_record(premetric.IMPLEMENTATION),
        "premetric_interface": file_record(premetric.INTERFACE),
        "premetric_execution_authority": premetric_record,
        "external_query_score_cache": cache_record,
        "premetric_audit": audit_record,
        **frozen_inputs,
        "label_root": _unopened(args.label_root, label="label root"),
        "output_dir": _unopened(args.output_dir, label="metric output"),
        "protocol": PROTOCOL,
        "single_candidate_no_sweep": True,
        "metric_execution_authorized": True,
        "access_audit": _build_access_audit(),
    }
    write_frozen_json(output, value)
    return {
        "status": "single_rank256_full_lift_metric_preregistered_before_GT_access",
        "authority": file_record(output),
    }


def validate_authority(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="rank256 full-lift frozen metric authority",
    )
    value = dict(raw) if isinstance(raw, Mapping) else {}
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "premetric_implementation",
        "premetric_interface",
        "premetric_execution_authority",
        "external_query_score_cache",
        "premetric_audit",
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
        "label_root",
        "output_dir",
        "protocol",
        "single_candidate_no_sweep",
        "metric_execution_authorized",
        "access_audit",
    }
    if (
        set(value) != required
        or value.get("schema") != SCHEMA
        or value.get("schema_version") != 1
        or value.get("status") != STATUS
        or value.get("implementation") != file_record(IMPLEMENTATION)
        or value.get("premetric_implementation") != file_record(premetric.IMPLEMENTATION)
        or value.get("premetric_interface") != file_record(premetric.INTERFACE)
        or value.get("protocol") != PROTOCOL
        or value.get("single_candidate_no_sweep") is not True
        or value.get("metric_execution_authorized") is not True
        or value.get("access_audit") != _build_access_audit()
    ):
        raise ValueError("rank256 full-lift metric authority header differs")
    records = {}
    for name in (
        "premetric_execution_authority",
        "external_query_score_cache",
        "premetric_audit",
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        records[name] = _mapping_record(value[name], label=f"metric {name}")
    premetric_authority, _, _ = _load_passed_premetric(
        authority_record=records["premetric_execution_authority"],
        cache_record=records["external_query_score_cache"],
        audit_record=records["premetric_audit"],
    )
    frozen_inputs = {
        name: records[name]
        for name in (
            "frozen_evaluator",
            "frozen_summary_head",
            "config",
            "renderer_geometry_checkpoint",
            "all_query_text_cache",
            "canonical_negative_text_cache",
        )
    }
    _validate_frozen_bindings(
        frozen_inputs=frozen_inputs,
        premetric_authority=premetric_authority,
    )
    if (
        value["scene_id"] != premetric_authority["scene_id"]
    ):
        raise ValueError("rank256 full-lift metric scene/renderer differs")
    value["verified_record"] = {"path": str(source), "sha256": digest}
    return value


def build_command(authority: Mapping[str, Any], *, gpu: int) -> list[str]:
    if isinstance(gpu, bool) or int(gpu) < 0:
        raise ValueError("gpu must be non-negative")
    return [
        sys.executable,
        authority["frozen_evaluator"]["path"],
        "--config",
        authority["config"]["path"],
        "--checkpoint",
        authority["renderer_geometry_checkpoint"]["path"],
        "--scene",
        authority["scene_id"],
        "--protocol_preset",
        "vala_paper_3d",
        "--label_dir",
        authority["label_root"],
        "--output_dir",
        authority["output_dir"],
        "--summary_head_weights",
        authority["frozen_summary_head"]["path"],
        "--text_embedding_cache",
        authority["all_query_text_cache"]["path"],
        "--canonical_embedding_cache",
        authority["canonical_negative_text_cache"]["path"],
        "--external_query_score_cache",
        authority["external_query_score_cache"]["path"],
        "--gpu",
        str(int(gpu)),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = validate_authority(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
    )
    command = build_command(authority, gpu=args.gpu)
    if not args.execute:
        return {
            "status": "rank256_full_lift_metric_dry_run",
            "execution_authority": authority["verified_record"],
            "command": command,
        }
    output = Path(authority["output_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("rank256 full-lift metric output must be new")
    subprocess.run(command, check=True)
    result = output / authority["scene_id"] / "lerf_direct_3d_selection_results.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError("rank256 full-lift metric result is unavailable")
    return {
        "status": "rank256_full_lift_metric_complete",
        "execution_authority": authority["verified_record"],
        "result": file_record(result),
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
        "premetric_execution_authority",
        "external_query_score_cache",
        "premetric_audit",
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        _add_record(build, name)
    build.add_argument("--label-root", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_authority)
    execute = commands.add_parser("run")
    execute.add_argument("--execution-authority", required=True)
    execute.add_argument("--expected-execution-authority-sha256", required=True)
    execute.add_argument("--gpu", type=int, default=0)
    execute.add_argument("--execute", action="store_true")
    execute.set_defaults(handler=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "PROTOCOL",
    "SCHEMA",
    "build_authority",
    "build_command",
    "run",
    "validate_authority",
]
