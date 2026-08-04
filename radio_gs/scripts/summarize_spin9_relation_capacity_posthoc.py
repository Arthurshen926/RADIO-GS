#!/usr/bin/env python3
"""Audit the post-hoc SPIn uncompressed-relation causal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_PROTOCOL_HASH = "d8a87284ddc2fde946a5d9de83aec190487e61c72259dc62656be603c2af6752"
EXPECTED_DECLARATION_SHA256 = "5944a9f049786d28bc526c37c3a9ce0183c284ce75a9145ce633c865344e5af1"
EXPECTED_EVALUATOR_SHA256 = "3a9f781687bf61916e3ff139dfba769f415b811ec88de19d02d552ba4a647477"
EXPECTED_QUERY_KERNEL_SHA256 = "e46c624bd204d776afc5d6455df1f122a416e0cb24db83ca0d6860eb98c860f6"
SCENES = ("lego", "orchids")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def first_reference_maximum(candidates: list[dict]) -> dict:
    if len(candidates) != 20:
        raise ValueError("reference calibration must contain exactly 20 candidates")
    best = candidates[0]
    for candidate in candidates[1:]:
        if float(candidate["reference_iou"]) > float(best["reference_iou"]):
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--formal-run-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--ludvig-summary", required=True)
    parser.add_argument("--diagnostic-declaration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    declaration_path = Path(args.diagnostic_declaration).resolve()
    if sha256_file(declaration_path) != EXPECTED_DECLARATION_SHA256:
        raise ValueError("post-hoc declaration digest changed")
    declaration = load(declaration_path)
    if (
        declaration.get("status") != "post_hoc_causal_diagnostic"
        or declaration.get("formal_preregistered_result") is not False
        or declaration.get("declared_after_full9_results_were_available") is not True
    ):
        raise ValueError("diagnostic declaration status differs")
    ludvig = load(args.ludvig_summary)
    records: dict[str, dict] = {}
    completed_records: list[dict] = []
    resource_failed_scenes: list[str] = []
    implementation_receipts: set[tuple[str, str]] = set()

    for scene in SCENES:
        report_path = (
            Path(args.run_root)
            / "reference_calibrated"
            / scene
            / f"{scene}_evaluation.json"
        )
        formal_path = (
            Path(args.formal_run_root)
            / "reference_calibrated"
            / scene
            / f"{scene}_evaluation.json"
        )
        baseline_path = (
            Path(args.baseline_root)
            / "evaluations"
            / scene
            / f"{scene}_evaluation.json"
        )
        if not report_path.is_file():
            failure_path = (
                Path(args.run_root) / "resource_failures" / f"{scene}.json"
            )
            failure = load(failure_path)
            if (
                failure.get("scene_id") != scene
                or failure.get("status") != "resource_limit_host_oom"
                or failure.get("diagnostic_status")
                != "post_hoc_causal_diagnostic"
                or failure.get("formal_preregistered_result") is not False
                or failure.get("promotion_eligible") is not False
                or failure.get("diagnostic_declaration_sha256")
                != EXPECTED_DECLARATION_SHA256
                or failure.get("target_masks_opened") is not False
                or failure.get("target_metrics_opened") is not False
            ):
                raise ValueError(f"{scene}: resource failure receipt differs")
            records[scene] = {
                "status": "resource_limit_host_oom",
                "resource_failure_receipt": str(failure_path.resolve()),
                "resource_failure_receipt_sha256": sha256_file(failure_path),
                "failure_stage": failure.get("failure_stage"),
                "resource_observation": failure.get("resource_observation"),
                "relation_cache": failure.get("relation_cache"),
                "target_masks_opened": False,
                "target_metrics_opened": False,
            }
            resource_failed_scenes.append(scene)
            continue
        report, formal, baseline = load(report_path), load(formal_path), load(baseline_path)
        if report.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
            raise ValueError(f"{scene}: frozen protocol hash differs")
        query = report.get("registered_prompt_evidence", {}).get(
            "query_conditioned_diffusion", {}
        )
        required_receipt = {
            "relation_projection": "none_uncompressed",
            "relation_feature_dimension": 4096,
            "lossy_relation_compression": False,
            "diagnostic_status": "post_hoc_causal_diagnostic",
            "formal_preregistered_result": False,
            "scene_selection_after_full9": True,
            "reference_calibration": True,
            "reference_only": True,
            "target_masks_opened": False,
            "target_metrics_opened": False,
            "relation_distance_bank_reused_across_feature_bandwidths": True,
            "execution_optimization_changes_method_semantics": False,
        }
        if any(query.get(key) != value for key, value in required_receipt.items()):
            raise ValueError(f"{scene}: post-hoc diagnostic receipt differs")
        if query.get("diagnostic_declaration_sha256") != EXPECTED_DECLARATION_SHA256:
            raise ValueError(f"{scene}: declaration binding differs")
        if query.get("effective_knn_columns") != 201 or query.get("knn_includes_self") is not True:
            raise ValueError(f"{scene}: release kNN semantics differ")
        best = first_reference_maximum(query.get("reference_calibration_candidates", []))
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
            raise ValueError(f"{scene}: target threshold differs from reference freeze")

        relation_path = Path(query["relation_cache"])
        sidecar = load(str(relation_path) + ".json")
        metadata = sidecar.get("metadata", {})
        if (
            sidecar.get("feature_dimension") != 4096
            or sidecar.get("output_sha256") != query.get("relation_cache_sha256")
            or metadata.get("diagnostic_declaration_sha256")
            != EXPECTED_DECLARATION_SHA256
            or metadata.get("query_independent") is not True
            or metadata.get("target_masks_opened") is not False
            or metadata.get("target_metrics_opened") is not False
        ):
            raise ValueError(f"{scene}: relation cache provenance differs")

        method = report.get("method_contract", {})
        evaluator_sha = str(method.get("evaluator_sha256", ""))
        query_sha = str(
            method.get("implementation_sha256", {}).get(
                "radio_gs/querying/query_conditioned_diffusion.py", ""
            )
        )
        if evaluator_sha != EXPECTED_EVALUATOR_SHA256 or query_sha != EXPECTED_QUERY_KERNEL_SHA256:
            raise ValueError(f"{scene}: frozen diagnostic implementation differs")
        implementation_receipts.add((evaluator_sha, query_sha))

        score = float(report["foreground_iou"])
        hash256 = float(formal["foreground_iou"])
        stages = baseline["stage_metrics"]
        unary = float(stages["unary_prior"]["foreground_iou"])
        graph = float(stages["propagated"]["foreground_iou"])
        connected = float(stages["connected"]["foreground_iou"])
        ludvig_score = float(ludvig["per_scene"][scene]["local_mean_iou_percent"]) / 100.0
        records[scene] = {
            "status": "completed",
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "selected_by_reference_only": selected,
            "uncompressed_dino4096": score,
            "formal_signed_hash256": hash256,
            "exact_adjoint_unary": unary,
            "fixed_graph": graph,
            "connected_report_only": connected,
            "local_ludvig_sam": ludvig_score,
            "delta_vs_signed_hash256": score - hash256,
            "delta_vs_exact_unary": score - unary,
            "delta_vs_fixed_graph": score - graph,
            "delta_vs_connected": score - connected,
            "delta_vs_local_ludvig": score - ludvig_score,
        }
        completed_records.append(records[scene])
    if not completed_records or len(implementation_receipts) != 1:
        raise ValueError("diagnostic scenes used different implementations")

    metric_keys = (
        "uncompressed_dino4096",
        "formal_signed_hash256",
        "exact_adjoint_unary",
        "fixed_graph",
        "connected_report_only",
        "local_ludvig_sam",
        "delta_vs_signed_hash256",
        "delta_vs_exact_unary",
        "delta_vs_fixed_graph",
        "delta_vs_connected",
        "delta_vs_local_ludvig",
    )
    output = {
        "schema_version": "spin_relation_capacity_posthoc_diagnostic_summary_v1",
        "status": "post_hoc_causal_diagnostic",
        "formal_preregistered_result": False,
        "promotion_eligible": False,
        "diagnostic_complete": not resource_failed_scenes,
        "completed_scenes": [
            scene for scene in SCENES if records[scene]["status"] == "completed"
        ],
        "resource_failed_scenes": resource_failed_scenes,
        "interpretation": (
            "Matched solver/readout diagnostic of relation representation capacity; "
            "scene choice occurred after full9 and cannot establish formal promotion."
        ),
        "diagnostic_declaration": str(declaration_path),
        "diagnostic_declaration_sha256": EXPECTED_DECLARATION_SHA256,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "scenes": records,
        "macro_over_posthoc_scenes": {
            key: float(np.mean([record[key] for record in completed_records]))
            for key in metric_keys
        },
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
