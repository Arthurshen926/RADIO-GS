#!/usr/bin/env python3
"""Audit the preregistered Field-B cache and reproduce its training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.scripts.eval_field_a_label_free_gate import _atomic_json
from radio_gs.scripts.train_canonical_radio_field import (
    _consensus_from_cache,
    _load_field_b_relation_triplets,
)
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


def run(args: argparse.Namespace) -> dict:
    registration, registration_sha, registration_path = load_json_object(
        args.experiment_registration,
        expected_sha256=args.expected_experiment_registration_sha256,
        label="Field-B experiment registration",
    )
    if registration.get("schema_version") != (
        "canonical_field_b_boundary_relation_registration_v1"
    ):
        raise ValueError("Field-B registration schema differs")
    primary, primary_sha, primary_path = load_mpr_cache(
        args.mpr_cache,
        expected_sha256=args.expected_mpr_cache_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=False,
    )
    dino, dino_sha, dino_path = load_mpr_cache(
        args.dino_mpr_cache,
        expected_sha256=args.expected_dino_mpr_cache_sha256,
        expected_feature_space="dino_v3",
        require_reliability=True,
        require_formal_safety=True,
    )
    relation, provenance = _load_field_b_relation_triplets(
        args.relation_triplet_cache,
        expected_sha256=args.expected_relation_triplet_cache_sha256,
        num_rows=int(primary["xyz"].shape[0]),
        capability_valid=torch.as_tensor(dino["valid"]).bool(),
        geometry_fingerprint=dict(primary.get("geometry_fingerprint", {})),
        expected_dino_sha256=args.expected_dino_mpr_cache_sha256,
        expected_sam3_sha256=args.expected_sam3_mpr_cache_sha256,
    )
    consensus = _consensus_from_cache(primary)
    valid_rows = torch.where(consensus.valid)[0]
    training_contract = dict(registration.get("training_contract", {}))
    seed = int(training_contract["seed"])
    validation_fraction = float(training_contract["validation_fraction"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = valid_rows[torch.randperm(valid_rows.numel(), generator=generator)]
    validation_count = max(1, int(round(order.numel() * validation_fraction)))
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:]
    training_mask = torch.zeros(int(primary["xyz"].shape[0]), dtype=torch.bool)
    training_mask[training_rows] = True

    triplets = int(relation["teacher_margin"].numel())
    positive = relation["pair_index"][:, :triplets]
    negative = relation["pair_index"][:, triplets:]
    keep = training_mask[positive].all(dim=0) & training_mask[negative].all(dim=0)
    kept_positive = positive[:, keep]
    kept_negative = negative[:, keep]
    validation_mask = torch.zeros_like(training_mask)
    validation_mask[validation_rows] = True
    validation_endpoint_count = int(
        validation_mask[
            torch.cat([kept_positive.reshape(-1), kept_negative.reshape(-1)])
        ].sum()
    )

    metadata = dict(provenance["metadata"])
    acceptance = dict(registration.get("pretraining_cpu_acceptance_gate", {}))
    checks = {
        "retained_anchor_fraction": (
            float(metadata["retained_fraction"])
            >= float(acceptance["minimum_retained_anchor_fraction"])
        ),
        "zero_boundary_hardness_fraction": (
            float(metadata["zero_selected_boundary_hardness_fraction"])
            <= float(acceptance["maximum_zero_selected_boundary_hardness_fraction"])
        ),
        "both_boundary_channels_present": (
            int(metadata["sam_boundary_negatives"]) > 0
            and int(metadata["appearance_boundary_negatives"]) > 0
        ),
        "teacher_margins_finite_positive": (
            bool(torch.isfinite(relation["teacher_margin"]).all())
            and bool((relation["teacher_margin"] > 0.0).all())
            and bool((relation["teacher_margin"] <= 1.0).all())
        ),
        "training_triplets_remain": bool(keep.any()),
        "validation_endpoints_excluded": validation_endpoint_count == 0,
    }
    receipt = {
        "schema_version": "canonical_field_b_triplet_cpu_audit_receipt_v1",
        "experiment_registration": {
            "path": str(registration_path),
            "sha256": registration_sha,
        },
        "primary_mpr": {"path": str(primary_path), "sha256": primary_sha},
        "dino_mpr": {"path": str(dino_path), "sha256": dino_sha},
        "sam3_mpr_sha256": args.expected_sam3_mpr_cache_sha256,
        "relation_triplet_cache": provenance,
        "full_cache_audit": {
            "candidate_anchors": int(metadata["candidate_anchors"]),
            "triplets": triplets,
            "retained_fraction": float(metadata["retained_fraction"]),
            "dropped_nonpositive_teacher_gap": int(
                metadata["dropped_nonpositive_teacher_gap"]
            ),
            "zero_selected_boundary_hardness_fraction": float(
                metadata["zero_selected_boundary_hardness_fraction"]
            ),
            "sam_boundary_negatives": int(metadata["sam_boundary_negatives"]),
            "appearance_boundary_negatives": int(
                metadata["appearance_boundary_negatives"]
            ),
            "teacher_margin": metadata["teacher_margin"],
        },
        "split_audit": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "raw_valid_rows": int(valid_rows.numel()),
            "training_rows": int(training_rows.numel()),
            "validation_rows": int(validation_rows.numel()),
            "full_triplets": triplets,
            "training_triplets_after_validation_exclusion": int(keep.sum()),
            "removed_triplets_touching_validation": int((~keep).sum()),
            "validation_endpoint_count_after_exclusion": validation_endpoint_count,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "gpu_used": False,
        "source_hashes": {
            "audit": sha256_file(Path(__file__).resolve()),
            "trainer": sha256_file(
                Path(__file__).with_name("train_canonical_radio_field.py")
            ),
        },
    }
    if not receipt["passed"]:
        raise ValueError(f"Field-B CPU audit failed: {checks}")
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Field-B audit output exists: {output}")
    _atomic_json(output, receipt)
    return {**receipt, "output": str(output), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--expected-experiment-registration-sha256", required=True)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--expected-mpr-cache-sha256", required=True)
    parser.add_argument("--dino-mpr-cache", required=True)
    parser.add_argument("--expected-dino-mpr-cache-sha256", required=True)
    parser.add_argument("--expected-sam3-mpr-cache-sha256", required=True)
    parser.add_argument("--relation-triplet-cache", required=True)
    parser.add_argument("--expected-relation-triplet-cache-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
