#!/usr/bin/env python3
"""Evaluate the immutable V4/accepted-V2 safe base on frozen generic data.

This is deliberately not a trainer.  It opens exactly the two SHA-bound
query-free validation shards, reconstructs the immutable accepted V2 adapter,
and evaluates only the preregistered generic teacher metrics.  It has no
optimizer, trainable residual, text/query input, or benchmark entry point.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from radio_gs.interfaces.surface_region_summary import (
    ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
    ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
    ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
    SurfaceRegionSummaryReadoutV4,
)
from radio_gs.models.siglip_projection import (
    OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
    SigLIP2SummaryHead,
)
from radio_gs.scripts.train_surface_region_summary_readout import (
    _completion_validation_views,
    _evaluate,
    _load,
    _selection_score,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    write_frozen_json,
)


STAGE2_REGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/"
    "surface_region_contract_v4_candidate_complete_preregistration_20260805.json"
)
STAGE2_REGISTRATION_SHA256 = (
    "0eed806dd630bc4f51f710b68f0db2b2bbb9e4b23c28aa3e4286e5cdba987746"
)
V4_CONTRACT_SHA256 = (
    "55d051772c24f0e27bc464c4ed55b90f519d0c919231b977006152ce1f03e147"
)
VALIDATION_SPLIT_SHA256 = (
    "2e71b30363f1c9268fd403c32139290e89070478fd0f5badbc54f2dc64665ec9"
)
VALIDATION_SCENES = (
    "scene0164_03",
    "scene0187_00",
    "scene0423_02",
    "scene0553_00",
    "scene0593_00",
    "scene0690_01",
    "scene0699_00",
    "scene0702_00",
)
BASE_DESCRIPTOR_COSINE_FLOOR = 0.9409590106010437


def _load_stage2_registration() -> dict:
    registration, _, _ = load_json_object(
        STAGE2_REGISTRATION,
        expected_sha256=STAGE2_REGISTRATION_SHA256,
        label="V4 stage-2 preregistration",
    )
    stage2 = registration.get("staged_validation", {}).get("stage_2")
    if (
        registration.get("registration")
        != "surface_region_contract_v4_candidate_complete_typed_budget_v1"
        or not isinstance(stage2, Mapping)
        or stage2.get("scope")
        != "all eight held-out query-free ScanNet validation scenes"
        or float(stage2.get("base_descriptor_cosine_floor", float("nan")))
        != BASE_DESCRIPTOR_COSINE_FLOOR
        or registration.get("single_change", {}).get("digest")
        != V4_CONTRACT_SHA256
    ):
        raise ValueError("V4 stage-2 preregistration authority differs")
    return registration


def _validate_stage2_validation_authority(meta: Mapping[str, object]) -> None:
    if meta.get("region_contract_version") != "surface-region-contract-v4":
        raise ValueError("stage-2 requires the exact V4 region contract")
    if meta.get("region_contract_sha256") != V4_CONTRACT_SHA256:
        raise ValueError("stage-2 V4 contract digest differs")
    if tuple(meta.get("split_hashes", ())) != (VALIDATION_SPLIT_SHA256,):
        raise ValueError("stage-2 validation split authority differs")
    if tuple(sorted(str(value) for value in meta.get("scenes", ()))) != tuple(
        sorted(VALIDATION_SCENES)
    ):
        raise ValueError("stage-2 must cover exactly the eight frozen scenes")
    if meta.get("radio_checkpoint_sha256") != OFFICIAL_C_RADIO_V4_H_HALF_SHA256:
        raise ValueError("stage-2 RADIO checkpoint provenance differs")
    cache_artifacts = meta.get("cache_artifacts")
    if not isinstance(cache_artifacts, list) or len(cache_artifacts) != 2:
        raise ValueError("stage-2 requires exactly two SHA-bound validation shards")
    resolved = [str(record.get("path", "")) for record in cache_artifacts]
    if len(set(resolved)) != 2 or any(not value for value in resolved):
        raise ValueError("stage-2 validation shard identities are not distinct")
    eligibility = meta.get("eligibility_completion")
    if (
        not isinstance(eligibility, Mapping)
        or eligibility.get("enabled") is not True
        or int(eligibility.get("variants_per_teacher_region", 0)) != 1
        or eligibility.get("validation_checkpoint_selection")
        != "full_support_rows_only"
    ):
        raise ValueError("stage-2 eligibility-completion authority differs")


def _stage2_gate(metrics: Mapping[str, float]) -> dict[str, object]:
    required = (
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    )
    finite = all(
        key in metrics and math.isfinite(float(metrics[key])) for key in required
    )
    descriptor = float(metrics.get("mean_descriptor_cosine", float("nan")))
    conditions = {
        "all_generic_metrics_finite": finite,
        "mean_descriptor_cosine_noninferiority": (
            finite and descriptor >= BASE_DESCRIPTOR_COSINE_FLOOR
        ),
    }
    return {
        "schema_version": 1,
        "authority": "surface_region_contract_v4_stage2_preregistered_v1",
        "thresholds": {
            "mean_descriptor_cosine_min": BASE_DESCRIPTOR_COSINE_FLOOR,
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
        "stage3_residual_experiment_authorized": all(conditions.values()),
        "benchmark_opening_authorized": False,
    }


def evaluate(args: argparse.Namespace) -> dict:
    registration = _load_stage2_registration()
    cache_paths = [Path(value) for value in args.validation_cache]
    cache_digests = [str(value) for value in args.validation_cache_sha256]
    if len(cache_paths) != 2 or len(cache_digests) != 2:
        raise ValueError("stage-2 requires exactly two cache paths and two SHA-256 values")

    validation, meta = _load(
        cache_paths,
        "validation",
        expected_sha256=cache_digests,
    )
    _validate_stage2_validation_authority(meta)
    full, completion = _completion_validation_views(validation)
    if completion is None:
        raise ValueError("stage-2 validation lacks eligibility-completion rows")
    for label, data in (("full", full), ("completion", completion)):
        scene_ids = [str(value) for value in data.get("scene_ids", ())]
        if set(scene_ids) != set(VALIDATION_SCENES):
            raise ValueError(f"stage-2 {label} rows do not cover every frozen scene")

    device = torch.device(str(args.device))
    model, _ = SurfaceRegionSummaryReadoutV4.from_accepted_v2_checkpoint(
        args.accepted_v2_checkpoint,
        map_location=device,
    )
    model = model.to(device).eval()
    architecture = model.architecture(V4_CONTRACT_SHA256)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable != 0 or architecture.get("trainable_parameter_count") != 0:
        raise ValueError("V4 stage-2 adapter must have zero trainable parameters")

    head = SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint)
    head = head.to(device).eval().requires_grad_(False)
    full_metrics = _evaluate(
        model,
        head,
        full,
        device,
        int(args.batch_size),
    )
    completion_metrics = _evaluate(
        model,
        head,
        completion,
        device,
        int(args.batch_size),
    )
    gate = _stage2_gate(full_metrics)
    report = {
        "schema_version": 1,
        "artifact_type": "surface_region_v4_query_free_safe_stage2_report",
        "protocol": {
            "registration": str(STAGE2_REGISTRATION.resolve()),
            "registration_sha256": STAGE2_REGISTRATION_SHA256,
            "scope": registration["staged_validation"]["stage_2"]["scope"],
            "checkpoint_selection_rows": "full_support_only",
            "completion_rows": "diagnostic_only",
            "training": False,
            "query_or_text_opened": False,
            "benchmark_opened": False,
        },
        "inputs": {
            "validation_caches": meta["cache_artifacts"],
            "validation_split_sha256": VALIDATION_SPLIT_SHA256,
            "validation_scenes": list(VALIDATION_SCENES),
            "region_contract_sha256": V4_CONTRACT_SHA256,
            "radio_checkpoint": {
                "path": str(Path(args.radio_checkpoint).resolve()),
                "sha256": OFFICIAL_C_RADIO_V4_H_HALF_SHA256,
            },
            "accepted_v2_checkpoint": {
                "path": str(Path(args.accepted_v2_checkpoint).resolve()),
                "sha256": ACCEPTED_SURFACE_REGION_V2_CHECKPOINT_SHA256,
                "architecture_sha256": (
                    ACCEPTED_SURFACE_REGION_V2_ARCHITECTURE_SHA256
                ),
                "state_dict_sha256": ACCEPTED_SURFACE_REGION_V2_STATE_DICT_SHA256,
                "provenance_sha256": ACCEPTED_SURFACE_REGION_V2_PROVENANCE_SHA256,
                "contract_sha256": ACCEPTED_SURFACE_REGION_V2_CONTRACT_SHA256,
            },
        },
        "readout": architecture,
        "validation": {
            "full_support_rows": len(full["radio_features"]),
            "completion_rows": len(completion["radio_features"]),
            "full_support": {
                **full_metrics,
                "selection_score": _selection_score(full_metrics),
            },
            "completion_diagnostic": {
                **completion_metrics,
                "selection_score": _selection_score(completion_metrics),
            },
        },
        "stage2_gate": gate,
    }
    write_frozen_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-cache",
        action="append",
        required=True,
        help="One frozen V4 validation shard; pass exactly twice.",
    )
    parser.add_argument(
        "--validation-cache-sha256",
        action="append",
        required=True,
        help="SHA-256 paired by order with --validation-cache; pass exactly twice.",
    )
    parser.add_argument("--accepted-v2-checkpoint", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
