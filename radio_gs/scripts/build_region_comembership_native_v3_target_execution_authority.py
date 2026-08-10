#!/usr/bin/env python3
"""Seal one query-free native-V3 target execution after source promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces import region_comembership_native_v3_target as formal
from radio_gs.scripts import infer_region_comembership_v2 as parent_inference
from radio_gs.scripts import materialize_region_comembership_features_v2 as parent_feature
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
)


def _new_output(value: str, *, label: str) -> str:
    raw = str(value)
    resolved = str(Path(raw).expanduser().resolve())
    if raw != resolved:
        raise ValueError(f"{label} must be a canonical absolute path")
    path = Path(resolved)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return resolved


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"native V3 target authority exists: {output}")

    source_result = {
        "path": str(Path(args.source_result).expanduser().resolve()),
        "sha256": str(args.expected_source_result_sha256),
    }
    checkpoint = {
        "path": str(Path(args.promoted_checkpoint).expanduser().resolve()),
        "sha256": str(args.expected_promoted_checkpoint_sha256),
    }
    # Fail closed on source before hashing or opening either target parent.
    source_gate = formal.validate_source_promotion(source_result, checkpoint)

    parent_feature_raw, parent_feature_sha, parent_feature_path = load_torch_mapping(
        args.parent_v2_feature_authority,
        expected_sha256=args.expected_parent_v2_feature_authority_sha256,
        map_location="cpu",
        label="native V3 parent V2 target feature authority",
    )
    parent_inference_raw, parent_inference_sha, parent_inference_path = load_torch_mapping(
        args.parent_v2_inference_authority,
        expected_sha256=args.expected_parent_v2_inference_authority_sha256,
        map_location="cpu",
        label="native V3 parent V2 target inference authority",
    )
    feature = parent_feature.validate_feature_authority(parent_feature_raw)
    inference = parent_inference.validate_inference_authority(parent_inference_raw)
    scene_id = str(args.scene_id)
    feature_record = {"path": str(parent_feature_path), "sha256": parent_feature_sha}
    inference_record = {
        "path": str(parent_inference_path),
        "sha256": parent_inference_sha,
    }
    if (
        feature["domain"] != "target"
        or inference["domain"] != "target"
        or feature["scene_id"] != scene_id
        or inference["scene_id"] != scene_id
        or inference["feature_authority"] != feature_record
        or feature["region_fingerprints"] != inference["region_fingerprints"]
    ):
        raise ValueError("native V3 parent V2 target chain differs")
    target_inputs = {
        "parent_v2_feature_authority": feature_record,
        "parent_v2_inference_authority": inference_record,
        "accepted_v2": dict(feature["input_authority"]["accepted_v2"]),
        "factorized_state": dict(feature["input_authority"]["factorized_state"]),
    }
    feature_output = _new_output(args.target_feature_output, label="target feature output")
    inference_output = _new_output(
        args.target_inference_output, label="target inference output"
    )
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": formal.SCHEMA_VERSION,
        "status": formal.TARGET_EXECUTION_STATUS,
        "scene_id": scene_id,
        "source_result": dict(source_gate["result_record"]),
        "promoted_checkpoint": dict(source_gate["checkpoint_record"]),
        "builder_implementation": file_record(Path(__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in formal.TARGET_IMPLEMENTATION_PATHS.items()
        },
        "target_inputs": target_inputs,
        "target_feature_output": feature_output,
        "target_inference_output": inference_output,
        "feature_materialization_authorized": True,
        "checkpoint_inference_authorized": True,
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
        "fallback_contract": formal.fallback_contract(),
        "access_audit": formal.access_audit(target_opened=True),
    }
    written = write_frozen_json(output, authority)
    record = file_record(written)
    formal.validate_target_execution_authority(
        written,
        expected_sha256=record["sha256"],
        scene_id=scene_id,
        expected_feature_output=feature_output,
        expected_inference_output=inference_output,
    )
    return {
        "status": "native_v3_query_free_target_execution_authority_complete",
        "scene_id": scene_id,
        "authority": record,
        "selected_rule": source_gate["checkpoint"]["selected_rule"],
        "exact_anchor_fallback": formal.fallback_contract(),
        "query_execution_authorized": False,
        "metric_execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--promoted-checkpoint", required=True)
    parser.add_argument("--expected-promoted-checkpoint-sha256", required=True)
    parser.add_argument("--parent-v2-feature-authority", required=True)
    parser.add_argument("--expected-parent-v2-feature-authority-sha256", required=True)
    parser.add_argument("--parent-v2-inference-authority", required=True)
    parser.add_argument("--expected-parent-v2-inference-authority-sha256", required=True)
    parser.add_argument("--target-feature-output", required=True)
    parser.add_argument("--target-inference-output", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
