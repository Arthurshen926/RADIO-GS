"""CLI for sealing pre-existing carrier geometry into a v4 receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry_receipt import GeometryReceipt, HashedInput


def _input(value: str) -> HashedInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be ROLE=PATH")
    role, path = value.split("=", 1)
    if not role or not path:
        raise argparse.ArgumentTypeError("input must contain a non-empty role and path")
    return HashedInput.seal(role, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--coordinate-convention", required=True)
    parser.add_argument("--input", action="append", type=_input, required=True)
    parser.add_argument("--source-rgb-opened", action="store_true")
    parser.add_argument("--target-rgb-opened", action="store_true")
    parser.add_argument("--benchmark-images-opened", action="store_true")
    parser.add_argument("--benchmark-masks-opened", action="store_true")
    parser.add_argument("--benchmark-labels-opened", action="store_true")
    parser.add_argument("--assisted-diagnostic", action="store_true")
    parser.add_argument("--model-family")
    parser.add_argument("--model-checkpoint-sha256")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata_json)
    if not isinstance(metadata, dict):
        parser.error("metadata JSON must encode an object")
    if args.assisted_diagnostic:
        metadata["assisted_diagnostic"] = True
    receipt = GeometryReceipt(
        carrier=args.carrier,
        coordinate_convention=args.coordinate_convention,
        inputs=tuple(args.input),
        source_rgb_opened=args.source_rgb_opened,
        target_rgb_opened=args.target_rgb_opened,
        benchmark_images_opened=args.benchmark_images_opened,
        benchmark_masks_opened=args.benchmark_masks_opened,
        benchmark_labels_opened=args.benchmark_labels_opened,
        model_family=args.model_family,
        model_checkpoint_sha256=args.model_checkpoint_sha256,
        metadata=metadata,
    )
    receipt.write(Path(args.output).resolve())
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
