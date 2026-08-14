#!/usr/bin/env python3
"""Seal public UQIS predictions before evaluator-private data is opened."""

from __future__ import annotations

import argparse
import json

from .evaluate_predictions import seal_prediction_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--method-run-manifest", required=True)
    parser.add_argument(
        "--row-scope",
        choices=("universal_complete", "modality_comparator"),
        required=True,
    )
    parser.add_argument(
        "--modality", choices=("text", "image", "point_2d", "point_3d")
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = seal_prediction_batch(
        args.benchmark_dir,
        args.prediction_dir,
        args.method_run_manifest,
        args.output,
        row_scope=args.row_scope,
        modality=args.modality,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
