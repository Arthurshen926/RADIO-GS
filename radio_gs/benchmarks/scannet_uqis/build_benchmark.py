#!/usr/bin/env python3
"""Freeze a ScanNet-UQIS release from audited private construction records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .protocol import UQISProtocolConfig, freeze_release


def _records(path: str | Path, key: str) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get(key)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON array or object containing {key!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-records", required=True)
    parser.add_argument("--target-records", required=True)
    parser.add_argument("--query-id-salt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-role",
        choices=("pilot",),
        required=True,
        help="v0.1 currently exposes only the result-ineligible pilot constructor",
    )
    parser.add_argument("--allow-incomplete-pilot", action="store_true")
    args = parser.parse_args()
    salt = Path(args.query_id_salt_file).read_bytes()
    release = freeze_release(
        _records(args.scene_records, "scenes"),
        _records(args.target_records, "targets"),
        args.output_dir,
        split_role=args.split_role,
        query_id_salt=salt,
        config=UQISProtocolConfig(),
        allow_incomplete_pilot=args.allow_incomplete_pilot,
    )
    print(json.dumps(release, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
