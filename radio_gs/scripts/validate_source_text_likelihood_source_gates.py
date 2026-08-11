#!/usr/bin/env python3
"""Seal source-only gates before any held-out ScanNet development label opens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.data.scannet_source_region_semantics import (
    PREREGISTERED_SOURCE_FIT_SCENES,
    sha256_file,
    validate_source_region_semantic_sidecar,
)
from radio_gs.querying.source_text_query_likelihood import (
    SOURCE_TEXT_CHECKPOINT_SCHEMA,
)
from radio_gs.scripts.build_source_text_query_likelihood_dataset import (
    _write_json_noclobber,
    validate_dataset_manifest,
)
from radio_gs.scripts.train_source_text_query_likelihood_head import RECEIPT_SCHEMA


GATE_SCHEMA = "radio_gs.source_text_query_likelihood_source_gates.v1"


def validate_source_gates(
    *,
    dataset_manifest: str | Path,
    sidecars: Sequence[str | Path],
    checkpoint: str | Path,
    fit_receipt: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(dataset_manifest).expanduser().resolve(strict=True)
    manifest, _payloads = validate_dataset_manifest(manifest_path)
    if tuple(record["scene_id"] for record in manifest["records"]) != tuple(
        PREREGISTERED_SOURCE_FIT_SCENES
    ):
        raise ValueError("source gate dataset is not the preregistered fit cohort")
    sidecar_rows = []
    for raw_path in sidecars:
        path = Path(raw_path).expanduser().resolve(strict=True)
        payload = validate_source_region_semantic_sidecar(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        statistics = payload["statistics"]
        gates = {
            "global_fixed_radius_geometry_coverage_at_least_0p25": (
                float(statistics["global_geometry_coverage"]) >= 0.25
            ),
            "valid_region_fraction_at_least_0p25": (
                float(statistics["valid_region_fraction"]) >= 0.25
            ),
            "mixed_regions_retained": (
                int(statistics["mixed_valid_region_count"]) > 0
            ),
            "development_labels_closed": (
                payload["source_access"]["development_semantic_labels_opened"]
                is False
            ),
            "agile_instance_ids_closed": (
                payload["source_access"]["agile3d_instance_ids_opened"] is False
            ),
            "pseudo_labels_closed": (
                payload["source_access"]["pseudo_semantic_labels_opened"] is False
            ),
        }
        if not all(gates.values()):
            raise RuntimeError(f"source semantic gate failed for {payload['scene_id']}")
        sidecar_rows.append(
            {
                "scene_id": payload["scene_id"],
                "sidecar": {"path": str(path), "sha256": sha256_file(path)},
                "statistics": dict(statistics),
                "gates": gates,
            }
        )
    sidecar_rows.sort(key=lambda value: value["scene_id"])
    if tuple(row["scene_id"] for row in sidecar_rows) != tuple(
        PREREGISTERED_SOURCE_FIT_SCENES
    ):
        raise ValueError("source gates require exactly the three preregistered sidecars")

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint_payload.get("schema") != SOURCE_TEXT_CHECKPOINT_SCHEMA:
        raise ValueError("source text checkpoint schema differs")
    if checkpoint_payload.get("source_scene_ids") != list(
        PREREGISTERED_SOURCE_FIT_SCENES
    ):
        raise ValueError("source text checkpoint scene cohort differs")
    if checkpoint_payload.get("dataset_manifest", {}).get("sha256") != sha256_file(
        manifest_path
    ):
        raise ValueError("source text checkpoint dataset hash differs")
    receipt_path = Path(fit_receipt).expanduser().resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("source text fit receipt schema differs")
    diagnostics = receipt.get("diagnostics", {})
    initial = diagnostics.get("initial", {})
    final = diagnostics.get("final", {})
    evaluator = receipt.get("evaluator_integration", {})
    fit_gates = {
        "cpu_only": diagnostics.get("cuda_initialized") is False,
        "source_balanced_bce_improved": (
            float(final.get("macro_balanced_bce", float("inf")))
            < float(initial.get("macro_balanced_bce", float("-inf")))
        ),
        "source_probability_gap_improved": (
            float(final.get("macro_positive_minus_negative_probability", float("-inf")))
            > float(initial.get("macro_positive_minus_negative_probability", float("inf")))
        ),
        "source_probability_gap_positive": (
            float(final.get("macro_positive_minus_negative_probability", -1.0)) > 0
        ),
        "evaluator_integration_closed": evaluator.get("enabled") is False,
        "scannet_exact_default_unchanged": (
            evaluator.get("scannet_exact_default_changed") is False
        ),
        "no_lerf_metric": evaluator.get("lerf_metric_run") is False,
        "no_scannet_metric": evaluator.get("scannet_metric_run") is False,
    }
    if not all(fit_gates.values()):
        raise RuntimeError("source text likelihood fit gate failed")
    return {
        "schema": GATE_SCHEMA,
        "schema_version": 1,
        "status": "source_gates_passed_single_development_open_authorized",
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "sidecars": sidecar_rows,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "fit_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        "fit_gates": fit_gates,
        "development_authorization": {
            "scene_id": "scene0003_00",
            "maximum_opens": 1,
            "parameter_callback_allowed": False,
            "benchmark_metric_allowed": False,
        },
        "source_access": dict(manifest["source_access"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--sidecar", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fit-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = validate_source_gates(
        dataset_manifest=args.dataset_manifest,
        sidecars=args.sidecar,
        checkpoint=args.checkpoint,
        fit_receipt=args.fit_receipt,
    )
    output = _write_json_noclobber(args.output, payload)
    print(json.dumps({"output": str(output), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
