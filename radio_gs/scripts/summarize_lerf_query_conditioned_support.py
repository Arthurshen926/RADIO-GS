#!/usr/bin/env python3
"""Summarize the preregistered LERF text Evidence-to-Support expansion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.scripts.eval_lerf_adaptive_support_diagnostic import sha256_file
from radio_gs.scripts.eval_lerf_query_conditioned_support import (
    EXPECTED_REGISTRATION_SHA256,
)


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _telemetry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = [row for row in rows if row.get("event") == "sample"]
    if not samples:
        raise ValueError(f"telemetry contains no samples: {path}")
    events: dict[str, int] = {}
    for row in rows:
        event = str(row.get("event", ""))
        events[event] = events.get(event, 0) + 1
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_count": len(samples),
        # Release-verification rows are measurements too and can capture the
        # peak between two regular samples, so peaks cover every logged row.
        "max_temperature_c": max(float(row["temp_c"]) for row in rows),
        "max_power_w": max(float(row["power_w"]) for row in rows),
        "max_memory_mib": max(float(row["memory_mib"]) for row in rows),
        "events": events,
        "thermal_pause_count": sum(
            count for event, count in events.items() if "pause" in event
        ),
        "thermal_abort_count": sum(
            count for event, count in events.items() if "abort" in event
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.candidate_root)
    adaptive_root = Path(args.adaptive_root)
    score_quality = _load_json(Path(args.score_quality_aggregate))
    vala = _load_json(Path(args.vala_result))
    method_config_sha256: str | None = None
    per_scene: dict[str, Any] = {}
    metric_columns = {
        "candidate_miou": [],
        "frozen_formal_miou": [],
        "target_blind_otsu3_miou": [],
        "reproduced_vala_miou": [],
        "candidate_acc025": [],
        "candidate_acc050": [],
    }
    for scene in SCENES:
        candidate_path = root / scene / scene / "lerf_direct_3d_selection_results.json"
        receipt_path = root / scene / "method_receipt.prelabel.json"
        otsu_path = (
            adaptive_root
            / "recursive_upper_otsu3"
            / scene
            / scene
            / "lerf_direct_3d_selection_results.json"
        )
        candidate_payload = _load_json(candidate_path)
        receipt = _load_json(receipt_path)
        otsu_payload = _load_json(otsu_path)
        if receipt.get("experiment_registration_sha256") != EXPECTED_REGISTRATION_SHA256:
            raise ValueError(f"registration differs for {scene}")
        current_config_sha = str(receipt.get("method_config_sha256", ""))
        if not current_config_sha:
            raise ValueError(f"method config hash is missing for {scene}")
        if method_config_sha256 is None:
            method_config_sha256 = current_config_sha
        elif current_config_sha != method_config_sha256:
            raise ValueError("method config differs across expanded scenes")
        candidate = candidate_payload["scene"]["results"]["thr0p6"]
        otsu = otsu_payload["scene"]["results"]["thr0p6"]
        formal = score_quality["scenes"][scene]["aggregate_object_mean"]
        vala_scene = vala["per_scene"][scene]
        values = {
            "objects": int(candidate["n"]),
            "candidate_miou": float(candidate["miou"]),
            "candidate_acc025": float(candidate["acc025"]),
            "candidate_acc050": float(candidate["acc050"]),
            "candidate_boundary_f": float(candidate["boundary_f"]),
            "candidate_trimap_iou": float(candidate["trimap_iou"]),
            "frozen_formal_miou": float(formal["frozen_formal_miou"]),
            "target_blind_otsu3_miou": float(otsu["miou"]),
            "reproduced_vala_miou": float(vala_scene["mIoU"]),
        }
        values.update(
            {
                "delta_vs_frozen_formal": values["candidate_miou"]
                - values["frozen_formal_miou"],
                "delta_vs_target_blind_otsu3": values["candidate_miou"]
                - values["target_blind_otsu3_miou"],
                "delta_vs_reproduced_vala": values["candidate_miou"]
                - values["reproduced_vala_miou"],
                "candidate_result": str(candidate_path.resolve()),
                "candidate_result_sha256": sha256_file(candidate_path),
                "prelabel_receipt": str(receipt_path.resolve()),
                "prelabel_receipt_sha256": sha256_file(receipt_path),
                "telemetry": _telemetry(root / "logs" / f"{scene}.telemetry.csv"),
            }
        )
        per_scene[scene] = values
        for name in metric_columns:
            metric_columns[name].append(float(values[name]))

    macro = {
        name: sum(values) / len(values) for name, values in metric_columns.items()
    }
    macro.update(
        {
            "delta_vs_frozen_formal": macro["candidate_miou"]
            - macro["frozen_formal_miou"],
            "delta_vs_target_blind_otsu3": macro["candidate_miou"]
            - macro["target_blind_otsu3_miou"],
            "delta_vs_reproduced_vala": macro["candidate_miou"]
            - macro["reproduced_vala_miou"],
        }
    )
    total_objects = sum(int(row["objects"]) for row in per_scene.values())
    object_micro_candidate = sum(
        int(row["objects"]) * float(row["candidate_miou"])
        for row in per_scene.values()
    ) / total_objects
    return {
        "artifact_type": "radio_gs_lerf_text_evidence_to_support_v1_summary",
        "status": "preregistered_target_blind_method_scored_by_frozen_evaluator",
        "experiment_registration_sha256": EXPECTED_REGISTRATION_SHA256,
        "method_config_sha256": method_config_sha256,
        "development_and_confirmation_gate": {
            "required_scenes": ["figurines", "waldo_kitchen"],
            "passed": all(
                per_scene[scene]["delta_vs_target_blind_otsu3"] > 0
                for scene in ("figurines", "waldo_kitchen")
            ),
        },
        "scenes": per_scene,
        "scene_equal_macro": macro,
        "object_micro": {
            "objects": total_objects,
            "candidate_miou": object_micro_candidate,
        },
        "comparison_sources": {
            "score_quality_aggregate": str(
                Path(args.score_quality_aggregate).resolve()
            ),
            "score_quality_aggregate_sha256": sha256_file(
                args.score_quality_aggregate
            ),
            "reproduced_vala_result": str(Path(args.vala_result).resolve()),
            "reproduced_vala_result_sha256": sha256_file(args.vala_result),
        },
        "claim_boundary": (
            "The candidate and Otsu3 use the frozen RADIO-GS evaluator; the "
            "VALA row is the separately reproduced released semantic pipeline "
            "on compatible Occam RGB geometry."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        default="output/optimization_20260803/lerf_text_audit/query_conditioned_support_v1",
    )
    parser.add_argument(
        "--adaptive-root",
        default="output/optimization_20260803/lerf_text_audit/adaptive_support",
    )
    parser.add_argument(
        "--score-quality-aggregate",
        default="output/optimization_20260803/lerf_text_audit/score_quality_v1/aggregate.json",
    )
    parser.add_argument(
        "--vala-result",
        default=(
            "/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/"
            "vala/lerf3d_occam_geometry_v1/evaluation/all_metrics_30000_0.6.json"
        ),
    )
    parser.add_argument("--output", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = summarize(args)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(output.resolve())
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
