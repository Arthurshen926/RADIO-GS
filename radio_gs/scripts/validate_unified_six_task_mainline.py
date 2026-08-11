"""Fail-closed validation for the six-task unified mainline registry."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_TASKS = {
    "lerf2d",
    "lerf3d",
    "scannet_ovs",
    "nvos",
    "spin9",
    "agile3d",
}
TRACKS = {
    "strict_reusable_field",
    "target_rgb_assisted",
    "oracle_diagnostic",
    "external_reproduction",
    "single_radio_field",
    "query_rgb_assisted_diagnostic",
    "teacher_bypass_diagnostic",
}
FIELD_LINEAGES = {
    "field_decode_only",
    "field_plus_task_sidecar",
    "teacher_sidecar_bypass",
    "legacy_field_plus_task_head",
    "external_checkpoint_reproduction",
    "single_radio_field_decode",
}


def _mapping(value: Any, label: str, issues: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        issues.append(f"{label} must be a mapping")
        return {}
    return value


def _check_metrics(value: Any, label: str, issues: list[str]) -> None:
    metrics = _mapping(value, label, issues)
    if not metrics:
        issues.append(f"{label} must not be empty")
        return
    for name, raw in metrics.items():
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            issues.append(f"{label}.{name} must be numeric")
            continue
        number = float(raw)
        if not math.isfinite(number):
            issues.append(f"{label}.{name} must be finite")
        elif not 0.0 <= number <= 1.0:
            issues.append(f"{label}.{name} must be in [0, 1]")


def _check_authority(path_value: Any, root: Path, label: str, issues: list[str]) -> None:
    if not isinstance(path_value, str) or not path_value:
        issues.append(f"{label} must be a non-empty path")
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        issues.append(f"{label} does not exist: {path}")


def _check_row(
    row_value: Any,
    root: Path,
    label: str,
    issues: list[str],
    *,
    schema_version: int,
    primary: bool,
) -> None:
    row = _mapping(row_value, label, issues)
    if not row:
        return
    track = row.get("track")
    if track not in TRACKS:
        issues.append(f"{label}.track must be one of {sorted(TRACKS)}")
    lineage = row.get("field_lineage")
    if lineage not in FIELD_LINEAGES:
        issues.append(
            f"{label}.field_lineage must be one of {sorted(FIELD_LINEAGES)}"
        )
    if track == "oracle_diagnostic" and row.get("paper_role") in {
        "current_exact",
        "primary",
    }:
        issues.append(f"{label} cannot promote an oracle row")
    if track == "strict_reusable_field" and row.get("target_rgb_visible_to_method") is not False:
        issues.append(f"{label} strict rows must set target_rgb_visible_to_method=false")
    if schema_version == 2 and primary:
        if track != "single_radio_field":
            issues.append(f"{label} v2 primary rows must use single_radio_field")
        if lineage != "single_radio_field_decode":
            issues.append(f"{label} v2 primary rows must decode the single RADIO field")
        if row.get("persistent_semantic_feature") != "radio":
            issues.append(f"{label} v2 primary rows must store only RADIO semantics")
        if row.get("training_teacher_payload_in_checkpoint") is not False:
            issues.append(f"{label} v2 primary rows must exclude teacher payloads from checkpoints")
        if row.get("query_time_source_rgb_visible_to_method") is not False:
            issues.append(f"{label} v2 primary rows must prohibit query-time source RGB")
        if row.get("query_time_target_rgb_visible_to_method") is not False:
            issues.append(f"{label} v2 primary rows must prohibit query-time target RGB")
        if row.get("target_gt_visible_to_method") is not False:
            issues.append(f"{label} v2 primary rows must prohibit target GT")
    _check_authority(row.get("authority"), root, f"{label}.authority", issues)
    _check_metrics(row.get("metrics"), f"{label}.metrics", issues)


def validate_registry(path: str | Path, *, root: str | Path = ".") -> list[str]:
    registry_path = Path(path)
    root_path = Path(root)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    registry = _mapping(payload, "registry", issues)
    schema_version = registry.get("schema_version")
    if schema_version not in {1, 2}:
        issues.append("schema_version must equal 1 or 2")
    claim = _mapping(registry.get("claim_boundary"), "claim_boundary", issues)
    tasks = _mapping(registry.get("tasks"), "tasks", issues)
    if schema_version == 2:
        contract = _mapping(
            registry.get("mainline_contract"), "mainline_contract", issues
        )
        persistent = _mapping(
            contract.get("persistent_scene_representation"),
            "mainline_contract.persistent_scene_representation",
            issues,
        )
        mapping_time = _mapping(
            contract.get("mapping_time"), "mainline_contract.mapping_time", issues
        )
        query_time = _mapping(
            contract.get("query_time"), "mainline_contract.query_time", issues
        )
        if persistent.get("semantic_feature_family") != "radio" or persistent.get(
            "stored_semantic_feature_tensors"
        ) != ["canonical_radio"]:
            issues.append("v2 mainline must persist exactly one canonical RADIO feature")
        if mapping_time.get("source_rgb_allowed") is not True or mapping_time.get(
            "official_sam_allowed"
        ) is not True:
            issues.append("v2 mapping must allow source RGB and official SAM teachers")
        if mapping_time.get("teacher_payload_in_field_checkpoint") is not False:
            issues.append("v2 mapping teachers must be absent from the field checkpoint")
        if mapping_time.get("target_or_evaluation_rgb_allowed") is not False:
            issues.append("v2 mapping must prohibit target/evaluation RGB")
        for key in (
            "source_rgb_allowed",
            "target_rgb_allowed",
            "source_rgb_proposal_sidecar_allowed",
            "target_rgb_proposal_sidecar_allowed",
            "target_gt_visible_to_method",
        ):
            if query_time.get(key) is not False:
                issues.append(f"v2 query-time contract must set {key}=false")
        architecture = _mapping(
            registry.get("shared_architecture_target"),
            "shared_architecture_target",
            issues,
        )
        build_teachers = _mapping(
            architecture.get("build_time_teachers"),
            "shared_architecture_target.build_time_teachers",
            issues,
        )
        official_sam = _mapping(
            build_teachers.get("official_sam"),
            "shared_architecture_target.build_time_teachers.official_sam",
            issues,
        )
        if official_sam.get("objective") != (
            "scale_matched_relative_order_plus_control_relative_no_harm"
        ) or official_sam.get("persistence") != "absent_after_training":
            issues.append(
                "v2 official SAM must be a non-persistent scale-relative no-harm teacher"
            )
        upgrade = _mapping(
            registry.get("source_sam_field_upgrade"),
            "source_sam_field_upgrade",
            issues,
        )
        if (
            upgrade.get("persistent_semantic_feature") != "radio"
            or upgrade.get("field_architecture_changed") is not False
            or upgrade.get("source_rgb_or_official_sam_at_evaluation") is not False
        ):
            issues.append(
                "v2 source-SAM upgrade must preserve one RADIO field and RGB-free evaluation"
            )
        for key in ("source_gate_authority", "scannet_sentinel_authority"):
            _check_authority(
                upgrade.get(key), root_path, f"source_sam_field_upgrade.{key}", issues
            )
        policy = _mapping(
            registry.get("promotion_policy"), "promotion_policy", issues
        )
        for key in (
            "require_all_tasks_passed",
            "require_single_persistent_radio_feature",
            "prohibit_query_time_source_or_target_rgb",
            "prohibit_runtime_sam_clip_dino_feature_branches",
            "permit_source_rgb_official_sam_training_teacher",
            "require_training_teacher_payload_absent_from_checkpoint",
            "require_exact_frozen_protocol",
            "require_six_task_no_regression_before_promotion",
        ):
            if policy.get(key) is not True:
                issues.append(f"v2 promotion_policy must set {key}=true")
    task_names = set(tasks)
    if task_names != EXPECTED_TASKS:
        issues.append(
            "tasks must equal "
            f"{sorted(EXPECTED_TASKS)}; got {sorted(task_names)}"
        )

    all_gates_passed = True
    for task_name in sorted(EXPECTED_TASKS & task_names):
        task = _mapping(tasks[task_name], f"tasks.{task_name}", issues)
        rows = task.get("current_rows")
        if not isinstance(rows, list) or not rows:
            issues.append(f"tasks.{task_name}.current_rows must be a non-empty list")
        else:
            for index, row in enumerate(rows):
                _check_row(
                    row,
                    root_path,
                    f"tasks.{task_name}.current_rows[{index}]",
                    issues,
                    schema_version=int(schema_version or -1),
                    primary=True,
                )
        diagnostics = task.get("diagnostic_rows", [])
        if diagnostics is not None and not isinstance(diagnostics, list):
            issues.append(f"tasks.{task_name}.diagnostic_rows must be a list")
        elif isinstance(diagnostics, list):
            for index, row in enumerate(diagnostics):
                _check_row(
                    row,
                    root_path,
                    f"tasks.{task_name}.diagnostic_rows[{index}]",
                    issues,
                    schema_version=int(schema_version or -1),
                    primary=False,
                )
        if "external_reference" in task:
            _check_row(
                task["external_reference"],
                root_path,
                f"tasks.{task_name}.external_reference",
                issues,
                schema_version=int(schema_version or -1),
                primary=False,
            )
        if not isinstance(task.get("promotion_target"), Mapping):
            issues.append(f"tasks.{task_name}.promotion_target must be a mapping")
        gate = _mapping(task.get("unified_gate"), f"tasks.{task_name}.unified_gate", issues)
        if gate.get("passed") is not True:
            all_gates_passed = False

    claim_eligible = claim.get("unified_paper_claim_eligible")
    if claim_eligible is True and not all_gates_passed:
        issues.append("unified paper claim cannot be eligible while any task gate is open")
    if claim_eligible not in {True, False}:
        issues.append("claim_boundary.unified_paper_claim_eligible must be boolean")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        default="paper/artifacts/unified_six_task_mainline_v1.yaml",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    issues = validate_registry(args.registry, root=args.root)
    if issues:
        raise SystemExit("\n".join(f"- {issue}" for issue in issues))
    print("unified six-task registry ok")


if __name__ == "__main__":
    main()
