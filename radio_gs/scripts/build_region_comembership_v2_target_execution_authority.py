#!/usr/bin/env python3
"""Seal one target execution only after formal source-only V2 promotion.

The ordering in this builder is part of the contract: the complete source
result and checkpoint are loaded and validated before any target input path is
hashed.  Consequently a failed source gate cannot accidentally open target
feature authorities while an execution JSON is being assembled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.interfaces import region_comembership_v2_formal as formal
from radio_gs.scripts import train_source_region_comembership_v2 as trainer
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
)


def _canonical_new_output(value: str, *, label: str) -> str:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path):
        raise ValueError(f"{label} must be an absolute canonical path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    return str(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"target execution authority exists: {output}")

    # Source promotion is deliberately validated before touching target paths.
    result_raw, result_sha, result_path = load_json_object(
        args.four_plus_two_result,
        expected_sha256=args.expected_four_plus_two_result_sha256,
        label="formal V2 source result",
    )
    result = formal.validate_result(result_raw, require_promotion=True)
    checkpoint_raw, checkpoint_sha, checkpoint_path = load_torch_mapping(
        args.promoted_checkpoint,
        expected_sha256=args.expected_promoted_checkpoint_sha256,
        map_location="cpu",
        label="formal V2 promoted checkpoint",
    )
    checkpoint = formal.validate_checkpoint(checkpoint_raw)
    checkpoint_record = {"path": str(checkpoint_path), "sha256": checkpoint_sha}
    if (
        result["checkpoint"] != checkpoint_record
        or checkpoint["selected_validation"] != result["selected_validation"]
        or checkpoint["promotion_gate"] != result["promotion_gate"]
        or checkpoint["promotion_gate"]["passed"] is not True
    ):
        raise ValueError("formal V2 result/checkpoint promotion chain differs")

    feature_output = _canonical_new_output(
        args.target_feature_output, label="target feature output"
    )
    inference_output = _canonical_new_output(
        args.target_inference_output, label="target inference output"
    )
    if feature_output == inference_output:
        raise ValueError("target feature and inference outputs must differ")

    target_paths = {
        "accepted_v2": args.accepted_v2,
        "typed_context": args.typed_context,
        "support_graph": args.support_graph,
        "factorized_state": args.factorized_state,
        "capability_descriptor": args.capability_descriptor,
    }
    target_inputs = {
        name: file_record(target_paths[name]) for name in formal.TARGET_INPUT_NAMES
    }
    root = Path(trainer.__file__).resolve().parents[2]
    authority = {
        "schema": formal.TARGET_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": formal.TARGET_EXECUTION_STATUS,
        "scene_id": str(args.scene_id),
        "preregistration": file_record(root / trainer.PREREGISTRATION),
        "efficiency_addendum": file_record(root / trainer.EFFICIENCY_ADDENDUM),
        "four_plus_two_result": {"path": str(result_path), "sha256": result_sha},
        "promoted_checkpoint": checkpoint_record,
        "target_feature_inputs": target_inputs,
        "target_feature_output": feature_output,
        "target_inference_output": inference_output,
        "target_feature_materialization_authorized": True,
        "target_checkpoint_inference_authorized": True,
        "target_metric_authorized": False,
        "access_audit": {
            "benchmark_images_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "target_metrics_computed": False,
        },
    }
    written = write_frozen_json(output, authority)
    record = file_record(written)
    formal.validate_target_execution_authority(
        written,
        expected_sha256=record["sha256"],
        scene_id=str(args.scene_id),
        expected_feature_output=feature_output,
        expected_inference_output=inference_output,
    )
    return {
        "status": "region_comembership_v2_target_execution_authority_complete",
        "scene_id": str(args.scene_id),
        "authority": record,
        "selected_validation": result["selected_validation"],
        "target_inputs_opened_after_source_promotion": True,
        "target_metric_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--four-plus-two-result", required=True)
    parser.add_argument("--expected-four-plus-two-result-sha256", required=True)
    parser.add_argument("--promoted-checkpoint", required=True)
    parser.add_argument("--expected-promoted-checkpoint-sha256", required=True)
    parser.add_argument("--accepted-v2", required=True)
    parser.add_argument("--typed-context", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--factorized-state", required=True)
    parser.add_argument("--capability-descriptor", required=True)
    parser.add_argument("--target-feature-output", required=True)
    parser.add_argument("--target-inference-output", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(build(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
