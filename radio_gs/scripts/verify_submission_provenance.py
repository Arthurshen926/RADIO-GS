"""Validate paper-facing provenance fields in the submission freeze manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("output/radio_gs/reports/submission_freeze_manifest.json")
REQUIRED_ROW_FIELDS = (
    "scene",
    "source",
    "config",
    "checkpoint",
    "selector_policy",
    "text_head",
    "teacher_model",
    "feature_manifest",
    "evaluator",
    "evaluator_script",
    "evaluator_sha256",
)
PATH_FIELDS = (
    "source",
    "config",
    "checkpoint",
    "feature_manifest",
    "text_embedding_cache",
    "score_cache",
    "evaluator_script",
)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _lookup(row: dict[str, Any], parent: dict[str, Any], manifest: dict[str, Any], key: str) -> Any:
    if _has_value(row.get(key)):
        return row.get(key)
    if _has_value(parent.get(key)):
        return parent.get(key)
    metadata = manifest.get("metadata", {})
    if isinstance(metadata, dict) and _has_value(metadata.get(key)):
        return metadata.get(key)
    return None


def _path_exists(root: Path, value: Any) -> bool:
    if not _has_value(value):
        return True
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.exists()


def _validate_rows(
    manifest: dict[str, Any],
    parent: dict[str, Any],
    rows_path: str,
    issues: list[str],
    *,
    root: Path,
    check_paths: bool,
) -> None:
    rows = parent.get("rows", [])
    if not isinstance(rows, list):
        issues.append(f"{rows_path} is not a list")
        return
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(f"{rows_path}[{index}] is not an object")
            continue
        for field in REQUIRED_ROW_FIELDS:
            if not _has_value(_lookup(row, parent, manifest, field)):
                issues.append(f"{rows_path}[{index}].{field} is missing")
        if check_paths:
            for field in PATH_FIELDS:
                value = _lookup(row, parent, manifest, field)
                if _has_value(value) and not _path_exists(root, value):
                    issues.append(f"{rows_path}[{index}].{field} path does not exist: {value}")


def validate_manifest(
    manifest: dict[str, Any],
    *,
    root: str | Path = ".",
    check_paths: bool = False,
) -> list[str]:
    issues: list[str] = []
    root_path = Path(root)
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict) or not _has_value(metadata.get("git_commit")):
        issues.append("metadata.git_commit is missing")

    for section_name in ("lerf", "scannet"):
        section = manifest.get(section_name, {})
        if not isinstance(section, dict):
            issues.append(f"{section_name} is not an object")
            continue
        _validate_rows(
            manifest,
            section,
            f"{section_name}.rows",
            issues,
            root=root_path,
            check_paths=check_paths,
        )

    direct_readouts = manifest.get("direct3d_readouts", [])
    if not isinstance(direct_readouts, list):
        issues.append("direct3d_readouts is not a list")
    else:
        for index, readout in enumerate(direct_readouts):
            if not isinstance(readout, dict):
                issues.append(f"direct3d_readouts[{index}] is not an object")
                continue
            _validate_rows(
                manifest,
                readout,
                f"direct3d_readouts[{index}].rows",
                issues,
                root=root_path,
                check_paths=check_paths,
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--check_paths", action="store_true")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    issues = validate_manifest(manifest, root=args.root, check_paths=args.check_paths)
    if issues:
        print(f"FAILED provenance verification for {path}")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"OK provenance verification for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
