#!/usr/bin/env python3
"""Stage one isolated UQIS method workspace."""

from __future__ import annotations

import argparse
import json

from .workspace import stage_query_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument(
        "--modality", choices=("text", "image", "point_2d", "point_3d"), required=True
    )
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--workspace-dir", required=True)
    args = parser.parse_args()
    receipt = stage_query_workspace(
        args.benchmark_dir,
        modality=args.modality,
        query_id=args.query_id,
        workspace_dir=args.workspace_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
