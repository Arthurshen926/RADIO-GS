#!/usr/bin/env python3
"""Aggregate immutable per-scene LERF score-quality diagnostic receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.scripts.eval_lerf_adaptive_support_diagnostic import sha256_file


METRICS = (
    "average_precision",
    "auprc",
    "oracle_threshold_iou",
    "positive_negative_score_margin",
    "frozen_formal_miou",
    "frozen_formal_positive_coverage",
    "frozen_formal_selected_purity",
    "target_blind_otsu3_miou",
    "target_blind_otsu3_positive_coverage",
    "target_blind_otsu3_selected_purity",
)


def summarize(paths: list[Path]) -> dict:
    scenes = {}
    all_queries = []
    registration_hashes = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = payload.get("score_quality_diagnostic")
        if not isinstance(audit, dict):
            raise ValueError(f"missing score-quality diagnostic: {path}")
        scene = str(audit["scene"])
        if scene in scenes:
            raise ValueError(f"duplicate scene: {scene}")
        receipt = audit.get("method_receipt_frozen_before_labels", {})
        registration_hashes.add(receipt.get("experiment_registration_sha256"))
        queries = list(audit.get("queries", []))
        aggregate = dict(audit["aggregate_object_mean"])
        scenes[scene] = {
            "objects": len(queries),
            "aggregate_object_mean": aggregate,
            "result": str(path.resolve()),
            "result_sha256": sha256_file(path),
            "method_receipt": receipt,
        }
        all_queries.extend(queries)
    if len(registration_hashes) != 1 or None in registration_hashes:
        raise ValueError("scene diagnostics do not share one preregistration")
    scene_rows = [row["aggregate_object_mean"] for row in scenes.values()]
    return {
        "artifact_type": "radio_gs_lerf3d_score_quality_diagnostic_aggregate",
        "status": "label_aware_diagnostic_only_not_formal",
        "experiment_registration_sha256": next(iter(registration_hashes)),
        "scenes": scenes,
        "scene_equal_macro": {
            key: float(np.mean([float(row[key]) for row in scene_rows]))
            for key in METRICS
        },
        "object_micro": {
            "objects": len(all_queries),
            "average_precision": float(np.mean([row["average_precision"] for row in all_queries])),
            "oracle_threshold_iou": float(np.mean([row["oracle_threshold_iou"] for row in all_queries])),
        },
        "claim_boundary": (
            "AP/AUPRC and per-object oracle threshold IoU are label-aware upper-bound "
            "diagnostics and may not select a method or threshold. Formal and Otsu3 "
            "remain the separately identified frozen and target-blind comparisons."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    report = summarize([Path(path) for path in args.result])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["scene_equal_macro"], indent=2))


if __name__ == "__main__":
    main()
