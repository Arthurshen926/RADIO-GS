#!/usr/bin/env python3
"""Reuse one official SAM3 model across several sealed NVOS scene plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from radio_gs.scripts.build_nvos_synchronous_multiview_box_sam3_inventory import run
from radio_gs.scripts.build_nvos_synchronous_multiview_sam3_inventory import (
    FROZEN_SAM3_SHA256,
    _sha256,
)
from radio_gs.scripts.build_sam3_foundation_cache import (
    _load_sam3_model,
    set_requested_cuda_device,
)


def execute(args: argparse.Namespace) -> dict[str, object]:
    scenes = [str(value) for value in args.scenes]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("--scene must be a nonempty unique sequence")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if _sha256(checkpoint) != str(args.expected_checkpoint_sha256):
        raise ValueError("official SAM3 checkpoint SHA-256 differs")
    set_requested_cuda_device(args.device)
    processor = _load_sam3_model(
        checkpoint_path=str(checkpoint),
        device=args.device,
        confidence_threshold=0.0,
        dtype="float32",
        resolution=int(args.resolution),
        point_only=False,
        build_on_cpu=True,
    )
    completed: list[dict[str, object]] = []
    for scene in scenes:
        plan = Path(args.plan_root).expanduser().resolve() / scene / "plan/candidate_plan.json"
        plan = plan.resolve(strict=True)
        output = Path(args.output_root).expanduser().resolve() / scene / "sam3_inventory"
        inventory = output / "inventory.json"
        if inventory.is_file():
            completed.append(
                {"scene": scene, "inventory": str(inventory), "resumed": True}
            )
            continue
        result = run(
            SimpleNamespace(
                plan=str(plan),
                expected_plan_sha256=_sha256(plan),
                output_dir=str(output),
                checkpoint=str(checkpoint),
                expected_checkpoint_sha256=str(args.expected_checkpoint_sha256),
                resolution=int(args.resolution),
                box_padding_pixels=int(args.box_padding_pixels),
                device=str(args.device),
            ),
            processor=processor,
        )
        completed.append(
            {"scene": scene, "inventory": result["inventory_path"], "resumed": False}
        )
    return {"device": str(args.device), "completed": completed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", dest="scenes", required=True)
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=FROZEN_SAM3_SHA256)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--box-padding-pixels", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(execute(build_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
