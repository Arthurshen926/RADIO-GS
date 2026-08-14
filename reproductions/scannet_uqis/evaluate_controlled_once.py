#!/usr/bin/env python3
"""Run one sealed evaluator-controlled UQIS construction validation."""

from __future__ import annotations

import argparse
import json

from radio_gs.benchmarks.scannet_uqis.controlled_evaluation import (
    evaluate_construction_authority_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-authority", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--sealed-prediction-batch", required=True)
    parser.add_argument("--one-shot-ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate_construction_authority_once(
        args.construction_authority,
        args.benchmark_dir,
        args.prediction_dir,
        args.sealed_prediction_batch,
        args.one_shot_ledger,
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
