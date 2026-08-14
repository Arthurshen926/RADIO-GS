#!/usr/bin/env python3
"""Audit a frozen ScanNet-UQIS release and fail closed on any drift."""

from __future__ import annotations

import argparse
import json

from .protocol import audit_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--skip-asset-hashes", action="store_true")
    args = parser.parse_args()
    report = audit_release(
        args.benchmark_dir, check_files=not args.skip_asset_hashes
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
