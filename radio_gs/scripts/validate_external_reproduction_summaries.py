#!/usr/bin/env python3
"""Validate completed external baseline summary artifacts before registry sync."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCENES = {"figurines", "ramen", "teatime", "waldo_kitchen"}
EXPECTED_LERF_OBJECTS = 208
DEFAULT_GAGS_SUMMARY = Path("paper/artifacts/gags_lerf_summary.json")
DEFAULT_DRSPLAT_SUMMARY = Path("paper/artifacts/drsplat_lerf_summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_scene_set(method: str, scenes: set[str], issues: list[str]) -> None:
    if scenes != EXPECTED_SCENES:
        issues.append(
            f"{method} scenes mismatch: got {sorted(scenes)}, expected {sorted(EXPECTED_SCENES)}"
        )


def _validate_gags(path: Path, *, required: bool, issues: list[str]) -> None:
    if not path.exists():
        if required:
            issues.append("missing required GAGS summary")
        return
    try:
        summary = _read_json(path)
        scenes = {str(row["scene"]) for row in summary["completed_rows"]}
        query_count = int(summary["object_weighted"]["query_count"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"invalid GAGS summary {path}: {exc}")
        return
    _validate_scene_set("GAGS", scenes, issues)
    if query_count != EXPECTED_LERF_OBJECTS:
        issues.append(f"GAGS query_count mismatch: got {query_count}, expected {EXPECTED_LERF_OBJECTS}")


def _validate_drsplat(path: Path, *, required: bool, issues: list[str]) -> None:
    if not path.exists():
        if required:
            issues.append("missing required Dr. Splat summary")
        return
    try:
        summary = _read_json(path)
        scenes = {str(scene) for scene in summary["scenes"]}
        count = int(summary["macro"]["count"])
        missing = int(summary["macro"]["missing"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"invalid Dr. Splat summary {path}: {exc}")
        return
    _validate_scene_set("Dr. Splat", scenes, issues)
    if count != EXPECTED_LERF_OBJECTS:
        issues.append(f"Dr. Splat count mismatch: got {count}, expected {EXPECTED_LERF_OBJECTS}")
    if count > 0 and missing >= count:
        issues.append(f"Dr. Splat missing masks invalid: missing={missing}, count={count}")


def validate_summaries(
    *,
    gags_summary_path: str | Path = DEFAULT_GAGS_SUMMARY,
    drsplat_summary_path: str | Path = DEFAULT_DRSPLAT_SUMMARY,
    require_gags: bool = False,
    require_drsplat: bool = False,
) -> list[str]:
    issues: list[str] = []
    _validate_gags(Path(gags_summary_path), required=require_gags, issues=issues)
    _validate_drsplat(Path(drsplat_summary_path), required=require_drsplat, issues=issues)
    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gags-summary", type=Path, default=DEFAULT_GAGS_SUMMARY)
    parser.add_argument("--drsplat-summary", type=Path, default=DEFAULT_DRSPLAT_SUMMARY)
    parser.add_argument("--require-gags", action="store_true")
    parser.add_argument("--require-drsplat", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    issues = validate_summaries(
        gags_summary_path=args.gags_summary,
        drsplat_summary_path=args.drsplat_summary,
        require_gags=args.require_gags,
        require_drsplat=args.require_drsplat,
    )
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)
    print("external reproduction summaries ok")


if __name__ == "__main__":
    main()
