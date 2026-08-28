"""Freeze one shared identity--extent weight from source-only residues."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _load(names: list[str], residue: int) -> list[tuple[Path, dict]]:
    values = []
    for name in names:
        path = Path(name).resolve(strict=True)
        with path.open() as handle:
            report = json.load(handle)
        if (
            report.get("schema") != "radio_gs.sugm_v3.image_query_source_dev.v1"
            or not report.get("source_only")
            or report.get("target_rgb_opened")
            or report.get("benchmark_metrics_opened")
            or report.get("evaluation_residue") != residue
            or not report.get("center_semantic_identity")
            or float(report.get("identity_extent_weight", 0.0)) <= 0
        ):
            raise ValueError("identity--extent selection authority differs")
        values.append((path, report))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="append", required=True)
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dev = _load(args.dev, 3)
    audit = _load(args.audit, 0)
    grouped: dict[float, list[tuple[Path, dict]]] = defaultdict(list)
    for item in dev:
        grouped[float(item[1]["identity_extent_weight"])].append(item)
    scene_count = len({item[1]["scene"] for item in dev})
    complete = {
        weight: items for weight, items in grouped.items()
        if len({item[1]["scene"] for item in items}) == scene_count
    }
    if not complete:
        raise ValueError("identity--extent candidate grid is incomplete")
    macro = {
        weight: sum(float(item[1]["metrics"]["mask_iou"]) for item in items) / len(items)
        for weight, items in complete.items()
    }
    selected = max(macro, key=lambda weight: (macro[weight], -weight))
    selected_audit = [
        item for item in audit
        if float(item[1]["identity_extent_weight"]) == selected
    ]
    if {item[1]["scene"] for item in selected_audit} != {
        item[1]["scene"] for item in complete[selected]
    }:
        raise ValueError("selected identity--extent weight lacks source audit")
    audit_macro = sum(
        float(item[1]["metrics"]["mask_iou"]) for item in selected_audit
    ) / len(selected_audit)
    payload = {
        "schema": "radio_gs.sugm_v3.identity_extent_weight_selection.v1",
        "selected_weight": selected,
        "selected_dev_macro_iou": macro[selected],
        "selected_audit_macro_iou": audit_macro,
        "candidate_dev_macro_iou": {str(key): value for key, value in sorted(macro.items())},
        "selection_rule": "maximum_cross_scene_source_dev_macro_iou",
        "hard_gate": False,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "dev_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path, _ in dev
        ],
        "audit_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path, _ in selected_audit
        ],
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
