#!/usr/bin/env python3
"""Adapt frozen historical results into fail-closed five-benchmark task receipts.

The adapter does not make an old result more eligible than its source evidence.
In particular, result completeness, development-contract eligibility, and paper
eligibility are recorded separately.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    write_frozen_json,
)


TASKS = ("lerf2d", "lerf3d", "scannet_ovs", "nvos")
DEFAULT_SOURCES = {
    "lerf2d": Path("paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json"),
    "lerf3d": Path("paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json"),
    "scannet_ovs": Path("paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json"),
    "nvos": Path("paper/artifacts/nvos_method_v1_field_box_signed_points_sam3_candidate_result_20260816.json"),
}


class TaskResultAdapterError(ValueError):
    """Raised when source evidence cannot support the requested receipt."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskResultAdapterError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _joint_context(joint: Mapping[str, Any], task: str) -> tuple[str, str, list[str], dict[str, str]]:
    candidate = _mapping(joint.get("joint_candidate"), "joint_candidate")
    identity = _mapping(candidate.get("identity"), "joint candidate identity")
    readouts = _mapping(identity.get("typed_readout_map"), "typed_readout_map")
    deployment = _mapping(joint.get("deployment_fields"), "deployment_fields")
    references = _mapping(deployment.get("task_references"), "task_references")
    scene_order = references.get(task)
    _require(isinstance(scene_order, list), f"{task} scene order is unavailable")
    instance_index = {
        str(row.get("dataset_instance_id")): row
        for row in deployment.get("instances", [])
        if isinstance(row, Mapping)
    }
    bindings: dict[str, str] = {}
    for instance_id in scene_order:
        row = instance_index.get(str(instance_id))
        field = row.get("universal_field") if isinstance(row, Mapping) else None
        if isinstance(field, Mapping) and isinstance(field.get("sha256"), str):
            bindings[str(instance_id)] = str(field["sha256"])
    return (
        str(candidate.get("candidate_sha256", "")),
        str(readouts.get(task, "")),
        [str(value) for value in scene_order],
        bindings,
    )


def _lerf2d(source: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(source.get("lerf2d_frozen_full4"), "LERF2D full4 evidence")
    scenes = _mapping(row.get("scene_results"), "LERF2D scene results")
    _require(row.get("scene_count") == 4 and set(scenes) == {"figurines", "ramen", "teatime", "waldo_kitchen"}, "LERF2D source is not full4")
    return {
        "result_complete": True,
        "development_contract_eligible": False,
        "paper_contract_eligible": False,
        "contract_gaps": [
            "evaluator content is not bound by the historical report",
            "historical evaluation does not bind the Universal Field v1 bytes used by the joint candidate",
        ],
        "metrics": {
            "sample_micro_miou": row["sample_micro_miou"],
            "scene_macro_miou": row["scene_macro_miou"],
            "localization_accuracy": row["localization_accuracy"],
            "sample_count": row["sample_count"],
        },
        "contract_id": "lerf2d-vala-paper-2d-full4-historical-unsealed-v1",
    }


def _lerf3d(source: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(source.get("lerf3d_frozen_full4"), "LERF3D full4 evidence")
    scenes = _mapping(row.get("scene_results"), "LERF3D scene results")
    _require(row.get("scene_count") == 4 and set(scenes) == {"figurines", "ramen", "teatime", "waldo_kitchen"}, "LERF3D source is not full4")
    _require(row.get("readout") == "primitive_text_score_on_frozen_3d_support", "LERF3D source is not Primitive Readout-v0")
    return {
        "result_complete": True,
        "development_contract_eligible": False,
        "paper_contract_eligible": False,
        "contract_gaps": ["one semantic level was evaluated; the frozen LERF3D contract requires three"],
        "metrics": {
            "sample_micro_miou": row["sample_micro_miou"],
            "sample_micro_acc025": row["sample_micro_acc025"],
            "sample_micro_acc050": row["sample_micro_acc050"],
            "semantic_levels": 1,
        },
        "contract_id": "lerf3d-full4-single-level-diagnostic-v1",
    }


def _scannet(source: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(source.get("scannet_ovs_paper8"), "ScanNet paper8 evidence")
    macro = _mapping(_mapping(row.get("paper8_macro"), "ScanNet paper8 macro").get("vala_pseudo_volume"), "ScanNet metrics")
    _require(row.get("method_v1_complete") == 8 and len(row.get("scene_results", {})) == 8, "ScanNet source is not complete paper8")
    _require(row.get("evaluation_protocol") == "frozen_and_reproduced", "ScanNet protocol is not frozen")
    return {
        "result_complete": True,
        "development_contract_eligible": True,
        "paper_contract_eligible": False,
        "contract_gaps": ["the existing row is development evidence, not a new prospectively blind seed-panel run"],
        "metrics": {split: {"miou": macro[split]["miou"], "macc": macro[split]["macc"]} for split in ("19", "15", "10")},
        "contract_id": "scannet-ovs-vala-paper8-frozen-reproduced-v1",
    }


def _nvos(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _mapping(source.get("result"), "NVOS result")
    eligibility = _mapping(source.get("eligibility"), "NVOS eligibility")
    prediction = _mapping(source.get("prediction_batch"), "NVOS prediction batch")
    _require(source.get("status") == "development_candidate_promoted_by_preregistered_gate", "NVOS candidate was not promoted for development")
    _require(eligibility.get("target_rgb_assisted") is True, "NVOS adapter requires the RGB-assisted candidate")
    _require(prediction.get("all_eight_predictions_sealed_before_first_target_mask_open") is True, "NVOS prediction barrier did not pass")
    return {
        "result_complete": True,
        "development_contract_eligible": True,
        "paper_contract_eligible": False,
        "contract_gaps": [
            "target-RGB assistance is legal for Method-v1 development but not the legacy strict-unseen contract",
            "the candidate was selected with target metrics and is not prospectively blind",
        ],
        "metrics": {
            "macro_foreground_iou": result["macro_foreground_iou"],
            "pixel_accuracy": result["pixel_accuracy"],
            "scene_foreground_iou": result["scene_foreground_iou"],
        },
        "contract_id": "nvos-full8-method-v1-query-transient-rgb-sam-development-v1",
    }


_ADAPTERS = {"lerf2d": _lerf2d, "lerf3d": _lerf3d, "scannet_ovs": _scannet, "nvos": _nvos}


def adapt(*, task: str, joint_receipt: Mapping[str, Any], source: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    _require(task in TASKS, f"unsupported task: {task}")
    candidate_sha, readout_id, scene_order, field_bindings = _joint_context(joint_receipt, task)
    assessment = _ADAPTERS[task](source)
    dev = assessment["development_contract_eligible"]
    receipt = {
        "schema_version": 1,
        "artifact_type": "radio_gs_five_benchmark_task_evaluation_receipt",
        "task_id": task,
        "status": "complete" if assessment["result_complete"] else "incomplete",
        "candidate_sha256": candidate_sha,
        "readout_id": readout_id,
        "evaluation_contract_id": assessment["contract_id"],
        "scene_order": scene_order,
        "field_bindings": field_bindings,
        "exact_evaluation_contract": dev,
        "complete_frozen_cohort": assessment["result_complete"],
        "evaluator_content_bound": dev,
        "authorized_target_access": True,
        "prediction_barrier_required": task == "nvos",
        "prediction_barrier_passed": task != "nvos" or dev,
        "target_metrics_used_for_current_run_selection": False,
        "prospectively_preregistered": False,
        "prospectively_blind": False,
        "target_metrics_used_for_candidate_selection": True,
        "stochastic": False,
        "seed_panel": [],
        "adapter_assessment": {
            **assessment,
            "source_evidence": file_record(source_path),
            "all_joint_field_bindings_available": len(field_bindings) == len(scene_order),
        },
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--joint-receipt", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    joint, _, _ = load_json_object(args.joint_receipt, label="joint readiness receipt")
    source_path = args.source or DEFAULT_SOURCES[args.task]
    source, _, resolved = load_json_object(source_path, label=f"{args.task} source evidence")
    receipt = adapt(task=args.task, joint_receipt=joint, source=source, source_path=resolved)
    write_frozen_json(args.output, receipt)
    print(json.dumps({"output": str(args.output.resolve()), "assessment": receipt["adapter_assessment"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
