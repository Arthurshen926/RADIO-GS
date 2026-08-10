#!/usr/bin/env python3
"""Audit contrast V2.1 against the preregistered source-teacher envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_contrast_v21_target_descriptor as descriptor_formal
from radio_gs.interfaces import factorized_native_contrast_v21_target_health_v3 as formal
from radio_gs.interfaces import factorized_native_target_health_v2 as health_v2_formal
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts.audit_factorized_native_target_descriptor_health_v2 import (
    descriptor_statistics,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber health V3 audit: {output}")

    prereg_record = {
        "path": str(Path(args.preregistration).expanduser().resolve()),
        "sha256": str(args.expected_preregistration_sha256),
    }
    source_record = {
        "path": str(Path(args.source_contrast_v21_result).expanduser().resolve()),
        "sha256": str(args.expected_source_contrast_v21_result_sha256),
    }
    descriptor_record = {
        "path": str(Path(args.target_descriptor).expanduser().resolve()),
        "sha256": str(args.expected_target_descriptor_sha256),
    }
    baseline_record = {
        "path": str(Path(args.accepted_v2_baseline).expanduser().resolve()),
        "sha256": str(args.expected_accepted_v2_baseline_sha256),
    }
    health_v2_record = {
        "path": str(Path(args.accepted_relative_health_v2).expanduser().resolve()),
        "sha256": str(args.expected_accepted_relative_health_v2_sha256),
    }

    prereg = formal.validate_preregistration(
        prereg_record["path"], expected_sha256=prereg_record["sha256"]
    )
    prereg_record = dict(prereg["verified_record"])
    if source_record != formal.SOURCE_RESULT_RECORD:
        raise ValueError("health V3 source result is not the preregistered singleton")
    source = descriptor_formal.validate_source_contrast_v21_result(source_record)
    if source["source_only_passed"] is not True:
        raise ValueError("health V3 source promotion did not pass")

    health_v2_raw, health_v2_sha, health_v2_path = load_json_object(
        health_v2_record["path"],
        expected_sha256=health_v2_record["sha256"],
        label="Accepted-relative health V2 diagnostic",
    )
    health_v2 = health_v2_formal.validate_health_audit(health_v2_raw)
    health_v2_producer = validate_file_record(
        health_v2["producer"], label="health V2 diagnostic producer"
    )
    if health_v2_producer != health_v2_formal.HEALTH_AUDIT_IMPLEMENTATION_PATH:
        raise ValueError("health V2 diagnostic implementation differs")
    health_v2_record = {"path": str(health_v2_path), "sha256": health_v2_sha}

    descriptor_raw, descriptor_sha, descriptor_path = load_torch_mapping(
        descriptor_record["path"],
        expected_sha256=descriptor_record["sha256"],
        map_location="cpu",
        label="contrast V2.1 health V3 target descriptor",
    )
    descriptor = descriptor_formal.validate_target_descriptor_authority(
        descriptor_raw
    )
    descriptor_record = {"path": str(descriptor_path), "sha256": descriptor_sha}
    baseline_raw, baseline_sha, baseline_path = load_torch_mapping(
        baseline_record["path"],
        expected_sha256=baseline_record["sha256"],
        map_location="cpu",
        label="contrast V2.1 health V3 AcceptedV2 baseline",
    )
    baseline = validate_target_accepted_v2_authority(baseline_raw)
    baseline_record = {"path": str(baseline_path), "sha256": baseline_sha}

    candidate_values = descriptor["semantic_descriptor"]
    baseline_values = baseline["accepted_v2_e0"]
    active = descriptor["active_update_mask"]
    exact = descriptor["exact_state_anchor_mask"]
    fallback = descriptor["immutable_fallback_mask"]
    candidate_stats = descriptor_statistics(candidate_values)
    baseline_stats = descriptor_statistics(baseline_values)
    unit_finite = all(
        bool(torch.isfinite(values).all())
        and torch.allclose(
            torch.linalg.vector_norm(values, dim=-1),
            torch.ones(values.shape[0]),
            rtol=0.0,
            atol=2e-4,
        )
        for values in (candidate_values, baseline_values)
    )
    alignment = {
        "scene_and_physical_space_equal": (
            descriptor["scene_id"] == baseline["scene_id"]
            and descriptor["physical_space_id"] == baseline["physical_space_id"]
        ),
        "canonical_region_indices_equal": torch.equal(
            descriptor["canonical_region_indices"],
            baseline["canonical_region_indices"],
        ),
        "region_fingerprints_equal": (
            descriptor["region_fingerprints"] == baseline["region_fingerprints"]
        ),
        "accepted_input_record_equal": (
            descriptor["input_authority"]["target_accepted_v2"] == baseline_record
        ),
        "source_result_record_equal": (
            descriptor["input_authority"]["source_contrast_v21_result"]
            == source_record
        ),
        "health_v2_descriptor_and_baseline_records_equal": (
            health_v2["input_authority"]["target_descriptor"] == descriptor_record
            and health_v2["input_authority"]["accepted_v2_baseline"]
            == baseline_record
        ),
        "health_v2_candidate_statistics_equal": (
            health_v2["candidate_statistics"] == candidate_stats
            and health_v2["baseline_statistics"] == baseline_stats
        ),
        "exact_active_masks_equal": torch.equal(exact, active),
        "fallback_mask_complement_active": torch.equal(fallback, ~active),
        "fallback_descriptor_bitwise_equal": torch.equal(
            candidate_values[fallback], baseline_values[fallback]
        ),
        "candidate_and_accepted_unit_l2_finite": unit_finite,
    }
    if not all(alignment.values()):
        raise ValueError("contrast V2.1 health V3 lineage/alignment differs")
    checks = formal.expected_checks(candidate_stats)
    eligible = all(checks.values())
    payload = {
        "schema": formal.HEALTH_AUDIT_SCHEMA,
        "schema_version": formal.HEALTH_AUDIT_SCHEMA_VERSION,
        "contract": formal.health_contract(),
        "contract_sha256": formal.HEALTH_CONTRACT_SHA256,
        "status": "pass" if eligible else "reject_outside_source_teacher_envelope",
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "producer": file_record(Path(__file__).resolve()),
        "input_authority": {
            "preregistration": prereg_record,
            "contrast_v21_source_result": source_record,
            "target_descriptor": descriptor_record,
            "accepted_v2_baseline": baseline_record,
            "accepted_relative_health_v2_diagnostic": health_v2_record,
        },
        "descriptor_channel_sha256": dict(descriptor["channel_sha256"]),
        "accepted_v2_channel_sha256": dict(baseline["channel_sha256"]),
        "alignment_audit": alignment,
        "accepted_relative_health_v2_status": health_v2["status"],
        "candidate_statistics": candidate_stats,
        "accepted_v2_diagnostic_statistics": baseline_stats,
        "frozen_global_gate": formal.frozen_gate(),
        "checks": checks,
        "query_authority_eligible": eligible,
        "access_audit": formal.health_access_audit(),
    }
    payload = formal.validate_health_audit(payload)
    write_frozen_json(output, payload)
    return {
        "status": payload["status"],
        "query_authority_eligible": eligible,
        "accepted_relative_health_v2_status": health_v2["status"],
        "candidate_statistics": candidate_stats,
        "accepted_v2_diagnostic_statistics": baseline_stats,
        "frozen_global_gate": formal.frozen_gate(),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "output": file_record(output),
        "access_audit": formal.health_access_audit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--source-contrast-v21-result", required=True)
    parser.add_argument("--expected-source-contrast-v21-result-sha256", required=True)
    parser.add_argument("--target-descriptor", required=True)
    parser.add_argument("--expected-target-descriptor-sha256", required=True)
    parser.add_argument("--accepted-v2-baseline", required=True)
    parser.add_argument("--expected-accepted-v2-baseline-sha256", required=True)
    parser.add_argument("--accepted-relative-health-v2", required=True)
    parser.add_argument("--expected-accepted-relative-health-v2-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(audit(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["audit", "build_parser"]
