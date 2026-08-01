#!/usr/bin/env python3
"""Validate the non-destructive evaluation cleanup decision inventory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CATEGORIES = {"must_keep", "deprecated_keep", "archive_then_remove", "safe_remove"}


class CleanupInventoryError(ValueError):
    """Raised when the inventory could authorize or obscure an unsafe cleanup."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CleanupInventoryError(f"{path} must be a mapping")
    return value


def _authoritative_paths(freeze: Mapping[str, Any]) -> set[str]:
    paths: set[str] = set()
    tasks = _mapping(freeze.get("canonical_tasks"), "freeze.canonical_tasks")
    for task_id, raw_task in tasks.items():
        task = _mapping(raw_task, f"freeze.{task_id}")
        for artifact in task.get("authoritative_artifacts", []):
            paths.add(str(_mapping(artifact, f"freeze.{task_id}.artifact")["path"]))
        paths.update(str(path) for path in task.get("entrypoints", []))
        report = task.get("report")
        if report:
            paths.add(str(report))
    return paths


def validate_inventory(
    inventory: Mapping[str, Any], *, freeze: Mapping[str, Any]
) -> None:
    if inventory.get("schema_version") != 1:
        raise CleanupInventoryError("schema_version must equal 1")
    policy = _mapping(inventory.get("policy"), "policy")
    if policy.get("deletion_performed") is not False:
        raise CleanupInventoryError("inventory must record deletion_performed=false")
    if policy.get("user_approval_required_for_every_cleanup_wave") is not True:
        raise CleanupInventoryError("every cleanup wave must require user approval")
    if policy.get("archive_requires_checksum_manifest") is not True:
        raise CleanupInventoryError("archive cleanup must require a checksum manifest")
    categories = _mapping(inventory.get("categories"), "categories")
    if set(categories) != CATEGORIES:
        raise CleanupInventoryError("categories must define the four cleanup classes")
    for group_id, raw_group in _mapping(
        inventory.get("protected_groups"), "protected_groups"
    ).items():
        group = _mapping(raw_group, f"protected_groups.{group_id}")
        if group.get("category") != "must_keep":
            raise CleanupInventoryError(f"protected group {group_id} must be must_keep")

    protected = _authoritative_paths(freeze)
    candidates = _mapping(inventory.get("candidates"), "candidates")
    if not candidates:
        raise CleanupInventoryError("candidates must not be empty")
    for candidate_id, raw_candidate in candidates.items():
        candidate = _mapping(raw_candidate, f"candidates.{candidate_id}")
        category = candidate.get("category")
        if category not in CATEGORIES - {"must_keep"}:
            raise CleanupInventoryError(
                f"cleanup candidate {candidate_id} has invalid category {category!r}"
            )
        target = candidate.get("target")
        if not isinstance(target, str) or not target.strip():
            raise CleanupInventoryError(f"cleanup candidate {candidate_id} needs a target")
        if target in protected:
            raise CleanupInventoryError(
                f"cleanup candidate {candidate_id} directly targets a frozen artifact"
            )
        if category == "archive_then_remove" and not (
            candidate.get("retain_before_action") or candidate.get("blocker")
        ):
            raise CleanupInventoryError(
                f"archive candidate {candidate_id} needs retain_before_action or blocker"
            )


def load_and_validate(inventory_path: Path, freeze_path: Path) -> Mapping[str, Any]:
    inventory = _mapping(
        yaml.safe_load(inventory_path.read_text(encoding="utf-8")), str(inventory_path)
    )
    freeze = _mapping(
        yaml.safe_load(freeze_path.read_text(encoding="utf-8")), str(freeze_path)
    )
    validate_inventory(inventory, freeze=freeze)
    return inventory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inventory",
        nargs="?",
        type=Path,
        default=Path("paper/artifacts/evaluation_protocol_cleanup_inventory_20260801.yaml"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("paper/artifacts/evaluation_protocol_freeze_20260801.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inventory = load_and_validate(args.inventory, args.freeze)
    print(f"validated {len(inventory['candidates'])} cleanup decisions: {args.inventory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
