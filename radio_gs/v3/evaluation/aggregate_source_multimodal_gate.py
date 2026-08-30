"""Close Gate 5 with frozen text and image source-dev/audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _image_macro(reports: list[dict], field: str) -> dict[str, float]:
    return {
        key: sum(float(report[field][key]) for report in reports) / len(reports)
        for key in ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")
    }


def _image_gate(
    reports: list[dict], *, iou_tolerance: float = 0.01
) -> tuple[bool, list[str], dict[str, dict[str, float]]]:
    baseline = _image_macro(reports, "uncalibrated_metrics")
    candidate = _image_macro(reports, "metrics")
    failures = []
    if not all(bool(report.get("identity_bitwise_preserved")) for report in reports):
        failures.append("image query changed a frozen parent")
    if candidate["brier"] >= baseline["brier"]:
        failures.append("image posterior Brier did not improve")
    if candidate["mask_iou"] < baseline["mask_iou"] - iou_tolerance:
        failures.append("image posterior IoU regressed beyond tolerance")
    return not failures, failures, {"uncalibrated": baseline, "calibrated": candidate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dev-gate", required=True)
    parser.add_argument("--text-audit-gate", required=True)
    parser.add_argument("--image-dev-report", action="append", required=True)
    parser.add_argument("--image-audit-report", action="append", required=True)
    parser.add_argument("--iou-tolerance", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iou_tolerance != 0.01:
        raise ValueError("multimodal source gate tolerance differs")
    text_paths = {
        "dev": Path(args.text_dev_gate).resolve(strict=True),
        "audit": Path(args.text_audit_gate).resolve(strict=True),
    }
    text = {name: json.loads(path.read_text()) for name, path in text_paths.items()}
    image_paths = {
        "dev": [Path(value).resolve(strict=True) for value in args.image_dev_report],
        "audit": [Path(value).resolve(strict=True) for value in args.image_audit_report],
    }
    image = {
        split: [json.loads(path.read_text()) for path in paths]
        for split, paths in image_paths.items()
    }
    if (
        text["dev"].get("residue") != 3
        or text["audit"].get("residue") != 0
        or any(not text[split].get("gate", {}).get("passed") for split in text)
    ):
        raise ValueError("multimodal source gate requires passed text dev and audit")
    for split, expected in (("dev", 3), ("audit", 0)):
        reports = image[split]
        if (
            len(reports) != 4
            or len({report.get("scene") for report in reports}) != 4
            or any(
                report.get("schema") != "radio_gs.sugm_v3.image_query_source_dev.v2"
                or report.get("evaluation_residue") != expected
                or report.get("target_rgb_opened")
                or report.get("benchmark_metrics_opened")
                for report in reports
            )
        ):
            raise ValueError(f"multimodal image {split} lineage differs")
    image_gate = {}
    failures = []
    for split in ("dev", "audit"):
        passed, local_failures, metrics = _image_gate(
            image[split], iou_tolerance=args.iou_tolerance
        )
        image_gate[split] = {"passed": passed, "failures": local_failures, "metrics": metrics}
        failures.extend(f"image_{split}: {value}" for value in local_failures)
    payload = {
        "schema": "radio_gs.sugm_v3.source_multimodal_gate.v1",
        "gate": {
            "passed": not failures,
            "failures": failures,
            "rule": "text exact-render dev+audit pass; image Brier down and IoU no more than 0.01 down on dev+audit; identity exact",
            "iou_tolerance": args.iou_tolerance,
        },
        "text": {
            split: text[split]["scene_macro"] for split in ("dev", "audit")
        },
        "image": image_gate,
        "same_gaussian_posterior_for_2d_and_3d": True,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "text": {
                split: {"path": str(path), "sha256": sha256_file(path)}
                for split, path in text_paths.items()
            },
            "image": {
                split: [
                    {"path": str(path), "sha256": sha256_file(path)} for path in paths
                ] for split, paths in image_paths.items()
            },
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()


__all__ = ["_image_gate", "_image_macro"]
