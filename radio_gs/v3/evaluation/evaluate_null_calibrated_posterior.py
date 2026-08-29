"""Evaluate a source-trained null calibrator on a held-out source residue."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.semantic_mapping_error_ladder import _metrics
from radio_gs.v3.query.calibrated_posterior import load_null_calibrated_posterior
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--residue", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.residue not in (0, 3):
        raise ValueError("null calibrator evaluation is held-out source only")
    calibrator_path = Path(args.calibrator).resolve(strict=True)
    evidence_paths = [Path(value).resolve(strict=True) for value in args.evidence]
    model = load_null_calibrated_posterior(calibrator_path, device=args.device)
    reports = []
    for path in evidence_paths:
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if (
            payload.get("schema") not in (
                "radio_gs.sugm_v3.clean_posterior_evidence.v1",
                "radio_gs.sugm_v3.clean_posterior_evidence.v2",
            )
            or not metadata.get("source_only")
            or metadata.get("target_rgb_opened")
            or metadata.get("benchmark_metrics_opened")
        ):
            raise ValueError("null calibrator evaluation evidence lineage differs")
        views = torch.as_tensor(payload["proposal_view_indices"]).long()
        selected = views % 4 == args.residue
        states = torch.as_tensor(payload["query_state"])[selected].to(torch.int8)
        active = (states == 1).any(0) & (states == 0).any(0)
        if not bool(active.any()):
            reports.append(
                {
                    "scene": payload["scene"],
                    "evaluable_queries": 0,
                    "metrics": None,
                    "evidence": {"path": str(path), "sha256": sha256_file(path)},
                }
            )
            continue
        positive = torch.as_tensor(payload["positive_features"])[selected][:, active]
        negative = torch.as_tensor(payload["negative_features"])[selected][:, active]
        shape = positive.shape[:2]
        logit = model.logit_from_features(
            positive.reshape(-1, 7).to(args.device),
            negative.reshape(-1, 3).to(args.device),
        )
        score = torch.sigmoid(logit).reshape(shape).cpu()
        reports.append(
            {
                "scene": payload["scene"],
                "evaluable_queries": int(active.sum()),
                "metrics": _metrics(score, states[:, active]),
                "evidence": {"path": str(path), "sha256": sha256_file(path)},
            }
        )
    valid = [report["metrics"] for report in reports if report["metrics"] is not None]
    if not valid:
        raise ValueError("null calibrator split has no evaluable scene")
    macro = {
        key: sum(float(value[key]) for value in valid) / len(valid)
        for key in (
            "recall_at_1",
            "recall_at_5",
            "mrr",
            "positive_similarity",
            "hardest_negative_similarity",
            "margin",
        )
    }
    output = {
        "schema": "radio_gs.sugm_v3.null_calibrated_posterior_evaluation.v1",
        "residue": args.residue,
        "scene_macro": macro,
        "scenes": reports,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "calibrator": {
            "path": str(calibrator_path),
            "sha256": sha256_file(calibrator_path),
        },
    }
    write_frozen_json(Path(args.output).resolve(), output)
    print(output)


if __name__ == "__main__":
    main()
