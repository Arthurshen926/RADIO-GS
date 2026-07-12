#!/usr/bin/env python3
"""Score protocol-bound NVOS/SPIn-NeRF predictions in a separate GT stage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.evaluation.promptable_segmentation import evaluate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation report exists (use --overwrite): {output}")
    manifest_path = args.manifest.expanduser().resolve()
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Reject truncated cohorts, prompt/GT role swaps, and changed annotation
    # content before the generic metric implementation is entered.
    validate_dataset_manifest(raw_manifest, check_files=True)
    report = evaluate_manifest(
        manifest_path,
        prediction_manifest=args.prediction_manifest,
        ground_truth_root=args.ground_truth_root,
        prediction_root=args.prediction_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(output),
                "protocol_hash": report["protocol_hash"],
                "num_scenes": report["dataset"]["num_scenes"],
                "num_frames": report["dataset"]["num_frames"],
                "foreground_iou": report["dataset"]["foreground_iou"],
                "pixel_accuracy": report["dataset"]["pixel_accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
