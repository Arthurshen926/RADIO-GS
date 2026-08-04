#!/usr/bin/env python3
"""Audit and summarize preregistered SPIn query-conditioned support results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_PROTOCOL_HASH = "d8a87284ddc2fde946a5d9de83aec190487e61c72259dc62656be603c2af6752"
EXPECTED_REGISTRATION_SHA256 = "7c539fb523c7152446bdc5f28325986a9162baa6c85a5608a66552023aa869c4"
DEV_SCENES = ("leaves", "lego", "orchids")
CONFIRMATION_SCENES = ("pinecone", "fern")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def first_reference_maximum(candidates: list[dict]) -> dict:
    if len(candidates) != 20:
        raise ValueError("reference calibration must contain exactly 20 f/g candidates")
    best = candidates[0]
    for candidate in candidates[1:]:
        if float(candidate["reference_iou"]) > float(best["reference_iou"]):
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--ludvig-summary", required=True)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registration = Path(args.experiment_registration).resolve()
    registration_sha = sha256_file(registration)
    if registration_sha != EXPECTED_REGISTRATION_SHA256:
        raise ValueError("experiment registration digest changed")
    ludvig = load(args.ludvig_summary)
    records: dict[str, dict] = {}
    for scene in args.scenes:
        report_path = Path(args.run_root) / "reference_calibrated" / scene / f"{scene}_evaluation.json"
        baseline_path = Path(args.baseline_root) / "evaluations" / scene / f"{scene}_evaluation.json"
        report = load(report_path)
        baseline = load(baseline_path)
        if report.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
            raise ValueError(f"{scene}: frozen protocol hash differs")
        query = report.get("registered_prompt_evidence", {}).get(
            "query_conditioned_diffusion", {}
        )
        if (
            query.get("reference_calibration") is not True
            or query.get("reference_only") is not True
            or query.get("target_masks_opened") is not False
            or query.get("target_metrics_opened") is not False
        ):
            raise ValueError(f"{scene}: calibration safety contract differs")
        candidates = query.get("reference_calibration_candidates", [])
        best = first_reference_maximum(candidates)
        selected = {
            "feature_bandwidth": float(query["feature_bandwidth"]),
            "regularizer_bandwidth": float(query["regularizer_bandwidth"]),
            "reference_iou": float(query["selected_reference_iou"]),
            "rendered_threshold": float(query["selected_rendered_threshold"]),
        }
        for key, value in selected.items():
            if not np.isclose(value, float(best[key]), atol=1e-12, rtol=0):
                raise ValueError(f"{scene}: selected {key} is not the first reference maximum")
        if not np.isclose(
            float(report["score_threshold"]),
            selected["rendered_threshold"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"{scene}: target threshold differs from frozen reference choice")
        if query.get("effective_knn_columns") != 201 or query.get("knn_includes_self") is not True:
            raise ValueError(f"{scene}: release kNN semantics differ")
        if query.get("native_ludvig_dinov2_pca40_exact") is not False:
            raise ValueError(f"{scene}: C-RADIO relation was mislabeled as native LUDVIG")
        relation_sidecar = load(str(query["relation_cache"]) + ".json")
        knn_sidecar = load(str(query["knn_cache"]) + ".json")
        for sidecar in (relation_sidecar, knn_sidecar):
            if sidecar["metadata"]["experiment_registration_sha256"] != registration_sha:
                raise ValueError(f"{scene}: cache registration binding differs")
        baseline_stages = baseline["stage_metrics"]
        unary = float(baseline_stages["unary_prior"]["foreground_iou"])
        graph = float(baseline_stages["propagated"]["foreground_iou"])
        connected = float(baseline_stages["connected"]["foreground_iou"])
        score = float(report["foreground_iou"])
        ludvig_score = float(ludvig["per_scene"][scene]["local_mean_iou_percent"]) / 100.0
        records[scene] = {
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "selected_by_reference_only": selected,
            "exact_adjoint_unary": unary,
            "fixed_graph": graph,
            "connected_report_only": connected,
            "query_conditioned_support": score,
            "local_ludvig_sam": ludvig_score,
            "delta_vs_exact_unary": score - unary,
            "delta_vs_fixed_graph": score - graph,
            "delta_vs_connected": score - connected,
            "delta_vs_local_ludvig": score - ludvig_score,
        }
    ordered = [records[scene] for scene in args.scenes]
    macro = {
        key: float(np.mean([record[key] for record in ordered]))
        for key in (
            "exact_adjoint_unary",
            "fixed_graph",
            "connected_report_only",
            "query_conditioned_support",
            "local_ludvig_sam",
            "delta_vs_exact_unary",
            "delta_vs_fixed_graph",
            "delta_vs_connected",
            "delta_vs_local_ludvig",
        )
    }
    scene_set = set(args.scenes)
    dev_complete = set(DEV_SCENES).issubset(scene_set)
    confirmation_complete = set(CONFIRMATION_SCENES).issubset(scene_set)
    dev_gate = None
    if dev_complete:
        dev = [records[scene] for scene in DEV_SCENES]
        dev_gate = {
            "mean_gain_vs_exact_unary": float(
                np.mean([value["delta_vs_exact_unary"] for value in dev])
            ),
            "improved_scene_count": int(
                sum(value["delta_vs_exact_unary"] > 0 for value in dev)
            ),
            "worst_scene_gain": float(
                min(value["delta_vs_exact_unary"] for value in dev)
            ),
        }
        dev_gate["passed"] = bool(
            dev_gate["mean_gain_vs_exact_unary"] >= 0.03
            and dev_gate["improved_scene_count"] >= 2
            and dev_gate["worst_scene_gain"] >= -0.01
        )
    confirmation_gate = None
    if confirmation_complete:
        confirmation = [records[scene] for scene in CONFIRMATION_SCENES]
        confirmation_gate = {
            "mean_gain_vs_exact_unary": float(
                np.mean([value["delta_vs_exact_unary"] for value in confirmation])
            ),
            "worst_scene_gain": float(
                min(value["delta_vs_exact_unary"] for value in confirmation)
            ),
        }
        confirmation_gate["passed"] = bool(
            confirmation_gate["mean_gain_vs_exact_unary"] >= 0.02
            and confirmation_gate["worst_scene_gain"] >= -0.02
        )
    output = {
        "schema_version": "spin9_query_conditioned_diffusion_summary_v1",
        "experiment_registration": str(registration),
        "experiment_registration_sha256": registration_sha,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "method_scope": (
            "LUDVIG-release-kernel-compatible with C-RADIO DINOv3 signed-hash256 relation; "
            "not native LUDVIG DINOv2 PCA40"
        ),
        "calibration_contract": (
            "20 preregistered f/g candidates and final rendered threshold selected only "
            "on the declared reference full mask before target masks are opened"
        ),
        "scenes": records,
        "macro_over_requested_scenes": macro,
        "development_gate": dev_gate,
        "confirmation_gate": confirmation_gate,
        "full9_expansion_allowed": bool(
            dev_gate is not None
            and confirmation_gate is not None
            and dev_gate["passed"]
            and confirmation_gate["passed"]
        ),
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
