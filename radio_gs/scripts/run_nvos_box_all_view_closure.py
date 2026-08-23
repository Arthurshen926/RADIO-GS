#!/usr/bin/env python3
"""Materialize and render one sealed NVOS all-view box posterior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from radio_gs.scripts.build_nvos_synchronous_multiview_sam3_inventory import _sha256
from radio_gs.scripts.materialize_nvos_synchronous_candidate_marginal import run
from radio_gs.scripts.render_nvos_synchronous_candidate_marginal import render


def close(args: argparse.Namespace) -> dict[str, str]:
    scene = str(args.scene)
    plan = (Path(args.plan_root).expanduser().resolve() / scene / "plan/candidate_plan.json").resolve(strict=True)
    inventory = (Path(args.inventory_root).expanduser().resolve() / scene / "sam3_inventory/inventory.json").resolve(strict=True)
    output = Path(args.output_root).expanduser().resolve() / scene
    marginal_dir = output / "marginal_positive_unknown"
    marginal = marginal_dir / "primitive_posterior.pt"
    marginal_receipt = marginal_dir / "receipt.json"
    if not marginal_receipt.is_file():
        run(
            SimpleNamespace(
                inventory=str(inventory),
                expected_inventory_sha256=_sha256(inventory),
                output_dir=str(marginal_dir),
                expected_candidates=1,
                view_huber_delta=2.0,
                view_fusion="positive_unknown_noisy_or",
                device=str(args.device),
            )
        )
    prediction_dir = output / "prediction_positive_unknown"
    prediction_receipt = prediction_dir / "prediction_receipt.json"
    if not prediction_receipt.is_file():
        render(
            SimpleNamespace(
                plan=str(plan),
                expected_plan_sha256=_sha256(plan),
                marginal=str(marginal),
                expected_marginal_sha256=_sha256(marginal),
                output_dir=str(prediction_dir),
                device=str(args.device),
            )
        )
    return {
        "scene": scene,
        "inventory": str(inventory),
        "marginal": str(marginal),
        "prediction_receipt": str(prediction_receipt),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--inventory-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(close(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
