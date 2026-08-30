"""Aggregate the frozen four-scene exact-render posterior gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _macro(reports: list[dict], cohort: str, variant: str) -> dict[str, float]:
    values = [
        report[cohort][variant]
        for report in reports
        if report[cohort][variant] is not None
    ]
    if not values:
        raise ValueError(f"exact-render posterior has no {cohort} authority")
    return {
        key: sum(float(value[key]) for value in values) / len(values)
        for key in values[0]
    }


def _gate_decision(
    reports: list[dict],
    positive: dict[str, dict[str, float]],
    empty: dict[str, dict[str, float]],
    *,
    iou_tolerance: float = 0.01,
) -> tuple[bool, list[str]]:
    failures = []
    positive_scenes = sum(report["positive"]["calibrated"] is not None for report in reports)
    empty_scenes = sum(report["empty"]["calibrated"] is not None for report in reports)
    if positive_scenes < 3:
        failures.append("fewer than three scenes have positive heldout authority")
    if empty_scenes < 3:
        failures.append("fewer than three scenes have empty heldout authority")
    if not all(bool(report["identity_bitwise_preserved"]) for report in reports):
        failures.append("clean D128 identity or signed-D16 parent changed")
    if positive["calibrated"]["brier"] >= positive["uncalibrated"]["brier"]:
        failures.append("positive exact-render Brier did not improve")
    if (
        positive["calibrated"]["mask_iou"]
        < positive["uncalibrated"]["mask_iou"] - iou_tolerance
    ):
        failures.append("positive exact-render IoU regressed beyond tolerance")
    if (
        empty["calibrated"]["foreground_probability"]
        >= empty["uncalibrated"]["foreground_probability"]
    ):
        failures.append("empty-target foreground probability did not decrease")
    if (
        empty["calibrated"]["foreground_fraction"]
        > empty["uncalibrated"]["foreground_fraction"]
    ):
        failures.append("empty-target foreground fraction increased")
    for report in reports:
        if (
            report["empty"]["calibrated"] is not None
            and report["empty"]["calibrated"]["foreground_probability"] >= 0.5
        ):
            failures.append(f"{report['scene']}: empty-target mean is not background by 0.5")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected-residue", type=int, required=True)
    parser.add_argument("--iou-tolerance", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_residue not in (0, 3) or args.iou_tolerance != 0.01:
        raise ValueError("exact-render posterior aggregate contract differs")
    paths = [Path(value).resolve(strict=True) for value in args.report]
    reports = [json.loads(path.read_text()) for path in paths]
    if (
        len(reports) != 4
        or len({report["scene"] for report in reports}) != 4
        or any(
            report.get("schema")
            != "radio_gs.sugm_v3.exact_render_posterior_source_evaluation.v1"
            or report.get("residue") != args.expected_residue
            or report.get("target_rgb_opened")
            or report.get("benchmark_metrics_opened")
            for report in reports
        )
    ):
        raise ValueError("exact-render posterior aggregate lineage differs")
    positive = {
        variant: _macro(reports, "positive", variant)
        for variant in ("uncalibrated", "calibrated")
    }
    empty = {
        variant: _macro(reports, "empty", variant)
        for variant in ("uncalibrated", "calibrated")
    }
    passed, failures = _gate_decision(
        reports, positive, empty, iou_tolerance=args.iou_tolerance
    )
    payload = {
        "schema": "radio_gs.sugm_v3.exact_render_posterior_gate.v1",
        "residue": args.expected_residue,
        "scene_macro": {"positive": positive, "empty": empty},
        "scenes": [{
            "scene": report["scene"],
            "positive_examples": report["positive_examples"],
            "empty_examples": report["empty_examples"],
            "positive": report["positive"],
            "empty": report["empty"],
            "cohort_availability": report.get("cohort_availability", {
                "positive": report["positive"]["calibrated"] is not None,
                "empty": report["empty"]["calibrated"] is not None,
                "missing_is_not_imputed": True,
            }),
        } for report in reports],
        "cohort_coverage": {
            "positive_scenes": sum(
                report["positive"]["calibrated"] is not None for report in reports
            ),
            "empty_scenes": sum(
                report["empty"]["calibrated"] is not None for report in reports
            ),
            "minimum_scenes_per_cohort": 3,
            "missing_authority_imputed": False,
        },
        "gate": {
            "passed": passed,
            "failures": failures,
            "iou_tolerance": args.iou_tolerance,
            "rule": "identity_exact; positive_Brier_down; positive_IoU_no_more_than_0.01_down; empty_mean_down_and_below_0.5_per_scene; empty_FPR_nonincrease",
        },
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()


__all__ = ["_gate_decision", "_macro"]
