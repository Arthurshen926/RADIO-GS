#!/usr/bin/env python3
"""Seal official ScanNet-UQIS-9 construction inputs and receipts."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_uqis.construction_authority import (
    seal_construction_authority,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-root", required=True)
    parser.add_argument("--candidate-release-root", required=True)
    parser.add_argument("--cohort-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = seal_construction_authority(
        args.construction_root,
        args.candidate_release_root,
        args.cohort_ledger,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

