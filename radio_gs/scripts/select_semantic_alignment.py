#!/usr/bin/env python3
"""Apply the frozen three-level semantic alignment gate without test calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.interfaces.semantic_alignment import (
    GlobalSemanticBridgeManifest,
    SemanticAlignmentPolicy,
    SemanticAlignmentStage,
    SemanticOracleResult,
)


def _load_oracle(path: str) -> SemanticOracleResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))["oracle"]
    return SemanticOracleResult(
        stage=SemanticAlignmentStage(payload["stage"]),
        dataset=str(payload["dataset"]),
        miou=float(payload["miou"]),
        localization_accuracy=float(payload["localization_accuracy"]),
        sample_count=int(payload["sample_count"]),
        protocol_hash=str(payload["protocol_hash"]),
        metadata=dict(payload.get("metadata", {})),
    )


def select(args: argparse.Namespace) -> dict:
    stage1 = _load_oracle(args.stage1)
    stage2 = _load_oracle(args.stage2) if args.stage2 else None
    bridge_manifest = None
    if args.bridge_manifest:
        bridge_manifest = GlobalSemanticBridgeManifest(
            **json.loads(Path(args.bridge_manifest).read_text(encoding="utf-8"))
        )
    policy = SemanticAlignmentPolicy(
        minimum_miou=float(args.minimum_miou),
        minimum_localization_accuracy=float(args.minimum_localization_accuracy),
    )
    try:
        decision = policy.decide(
            stage1, stage2=stage2, bridge_manifest=bridge_manifest
        )
        report = {
            "status": "selected",
            "selected_stage": decision.selected_stage.value,
            "reason": decision.reason,
        }
    except RuntimeError as exc:
        report = {
            "status": "requires_global_frozen_bridge",
            "selected_stage": None,
            "reason": str(exc),
        }
    report.update(
        {
            "quality_gate": {
                "minimum_miou": float(args.minimum_miou),
                "minimum_localization_accuracy": float(
                    args.minimum_localization_accuracy
                ),
                "set_before_bridge_training": True,
            },
            "stage1": {
                "miou": stage1.miou,
                "localization_accuracy": stage1.localization_accuracy,
                "protocol_hash": stage1.protocol_hash,
            },
            "stage2": (
                {
                    "miou": stage2.miou,
                    "localization_accuracy": stage2.localization_accuracy,
                    "protocol_hash": stage2.protocol_hash,
                }
                if stage2 is not None
                else None
            ),
            "test_set_calibration": False,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage2", default="")
    parser.add_argument("--bridge-manifest", default="")
    parser.add_argument("--minimum-miou", type=float, default=0.35)
    parser.add_argument("--minimum-localization-accuracy", type=float, default=0.60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(select(args), indent=2))


if __name__ == "__main__":
    main()
