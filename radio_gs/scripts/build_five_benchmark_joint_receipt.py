#!/usr/bin/env python3
"""Build a no-clobber, fail-closed Five-Benchmark Program readiness receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radio_gs.scripts.validate_five_benchmark_joint_receipt import (
    SCHEMA,
    TASKS,
    validate,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    sha256_file,
    write_frozen_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNIVERSAL_AUTHORITY = REPO_ROOT / "paper/artifacts/universal_field_v1_authority_20260816.json"
DEFAULT_CONSTRUCTION_AUTHORITY = REPO_ROOT / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
DEFAULT_EVALUATION_FREEZE = REPO_ROOT / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
DEFAULT_INVENTORY = REPO_ROOT / "paper/artifacts/universal_field_v1_asset_inventory_20260816.json"


def _expected_cohorts(construction: dict[str, Any]) -> dict[str, list[str]]:
    frozen = construction["frozen_cohorts"]
    return {task: [str(value) for value in frozen[task]] for task in TASKS}


def _instance_id(task: str, scene: str) -> str:
    if task in ("lerf2d", "lerf3d"):
        return f"lerf:{scene}"
    return f"{task}:{scene}"


def build_receipt(
    *,
    universal_authority: Path = DEFAULT_UNIVERSAL_AUTHORITY,
    construction_authority: Path = DEFAULT_CONSTRUCTION_AUTHORITY,
    evaluation_freeze: Path = DEFAULT_EVALUATION_FREEZE,
    inventory_path: Path = DEFAULT_INVENTORY,
    task_receipts: dict[str, Path] | None = None,
) -> dict[str, Any]:
    universal, universal_sha, universal_path = load_json_object(
        universal_authority, label="Universal Field authority"
    )
    construction, construction_sha, construction_path = load_json_object(
        construction_authority, label="construction authority"
    )
    inventory, inventory_sha, inventory_source = load_json_object(
        inventory_path, label="Universal Field inventory"
    )
    cohorts = _expected_cohorts(construction)
    readouts = {
        "lerf2d": "radio-gs-primitive-readout-v0",
        "lerf3d": "radio-gs-primitive-readout-v0",
        "scannet_ovs": "radio-gs-primitive-readout-v0",
        "nvos": "radio-gs-transient-rgb-sam-adapter-v1",
        "spin_nerf_available9": "radio-gs-transient-rgb-sam-adapter-v1",
    }
    identity = {
        "universal_field_id": universal["universal_field_id"],
        "universal_field_authority_sha256": universal_sha,
        "construction_authority_sha256": construction_sha,
        "typed_readout_map": readouts,
        "target_access": construction["target_access"],
    }
    candidate_sha = canonical_json_sha256(identity)

    inventory_scenes = inventory.get("scenes", {})
    inventory_instances = inventory.get("instances", {})
    instances: list[dict[str, Any]] = []
    seen: set[str] = set()
    references: dict[str, list[str]] = {}
    for task in TASKS:
        task_refs: list[str] = []
        for scene in cohorts[task]:
            instance_id = _instance_id(task, scene)
            task_refs.append(instance_id)
            if instance_id in seen:
                continue
            seen.add(instance_id)
            # A full inventory must use namespaced instance ids. The legacy
            # checked-in inventory has only unambiguous LERF and ScanNet keys.
            source = (
                inventory_instances.get(instance_id)
                if isinstance(inventory_instances, dict)
                else None
            )
            if source is None and task in ("lerf2d", "lerf3d", "scannet_ovs"):
                source = inventory_scenes.get(scene)
            if source is None and task == "nvos":
                source = inventory_scenes.get(f"nvos/{scene}")
            if source is None and task == "spin_nerf_available9":
                source = inventory_scenes.get(f"spin_nerf_available9/{scene}")
            if isinstance(source, dict):
                instances.append(
                    {
                        "dataset_instance_id": instance_id,
                        "scene_id": scene,
                        "status": source.get("status"),
                        "content_verified": False,
                        "universal_field": source.get("universal_field"),
                        "source_field": source.get("source_field"),
                        "factorized_cache": source.get("factorized_cache"),
                        "migration_report": source.get("migration_report"),
                    }
                )
            else:
                instances.append(
                    {
                        "dataset_instance_id": instance_id,
                        "scene_id": scene,
                        "status": "missing_from_inventory",
                        "content_verified": False,
                        "universal_field": None,
                    }
                )
        references[task] = task_refs

    supplied = task_receipts or {}
    task_results: dict[str, Any] = {}
    for task in TASKS:
        path = supplied.get(task)
        if path is None:
            task_results[task] = {
                "status": "missing",
                "candidate_sha256": candidate_sha,
                "scene_order": references[task],
                "field_bindings": {},
            }
        else:
            payload, _, source = load_json_object(path, label=f"{task} task receipt")
            task_results[task] = {**payload, "evidence": file_record(source)}

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "artifact_type": "radio_gs_five_benchmark_joint_evaluation_receipt",
        "joint_candidate": {
            "candidate_sha256": candidate_sha,
            "identity": identity,
        },
        "authorities": {
            "universal_field": {"path": str(universal_path), "sha256": universal_sha},
            "construction": {"path": str(construction_path), "sha256": construction_sha},
            "evaluation_freeze": file_record(evaluation_freeze),
        },
        "source_inventory": {
            "path": str(inventory_source),
            "sha256": inventory_sha,
        },
        "deployment_fields": {
            "instances": instances,
            "task_references": references,
        },
        "task_results": task_results,
        "historical_peak_stitching": False,
        "meets_same_contract_sota_target": False,
        "eligibility": {
            "joint_development_baseline_eligible": False,
            "paper_row_eligible": False,
            "sota_claim_supported": False,
        },
    }
    # Readiness receipts are allowed to fail gates. The validator still rejects
    # malformed identities or an eligibility overclaim.
    validate(receipt, verify_field_files=False)
    return receipt


def _parse_task_receipt(value: str) -> tuple[str, Path]:
    task, separator, path = value.partition("=")
    if not separator or task not in TASKS or not path:
        raise argparse.ArgumentTypeError("expected TASK=PATH for a frozen task receipt")
    return task, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal-authority", type=Path, default=DEFAULT_UNIVERSAL_AUTHORITY)
    parser.add_argument("--construction-authority", type=Path, default=DEFAULT_CONSTRUCTION_AUTHORITY)
    parser.add_argument("--evaluation-freeze", type=Path, default=DEFAULT_EVALUATION_FREEZE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--task-receipt", action="append", default=[], type=_parse_task_receipt)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    task_receipts = dict(args.task_receipt)
    if len(task_receipts) != len(args.task_receipt):
        parser.error("task receipts must be unique")
    receipt = build_receipt(
        universal_authority=args.universal_authority,
        construction_authority=args.construction_authority,
        evaluation_freeze=args.evaluation_freeze,
        inventory_path=args.inventory,
        task_receipts=task_receipts,
    )
    if args.output is None:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        write_frozen_json(args.output, receipt)
        print(
            json.dumps(
                {"output": str(args.output.resolve()), "sha256": sha256_file(args.output)},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
