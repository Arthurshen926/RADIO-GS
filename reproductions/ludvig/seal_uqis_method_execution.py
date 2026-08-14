#!/usr/bin/env python3
"""Seal the complete nine-scene LUDVIG field and query execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.benchmarks.scannet_uqis.ludvig_evaluation import (
    seal_ludvig_method_execution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = seal_ludvig_method_execution(
        args.benchmark_dir, args.mapping_root, args.evaluation_root, args.output_dir
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
