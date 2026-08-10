#!/usr/bin/env python3
"""Materialize a query-free anti-collapse audit against AcceptedV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import factorized_native_target_descriptor as descriptor_formal
from radio_gs.interfaces import factorized_native_target_health as formal
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    write_frozen_json,
)


_PAIR_CHUNK = 2048
_ROW_CHUNK = 512


def deterministic_pair_axis(regions: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(regions)
    if count < 2:
        raise ValueError("anti-collapse audit requires at least two regions")
    samples = min(formal.PAIR_SAMPLE_CAP, count * (count - 1))
    axis = torch.arange(samples, dtype=torch.long)
    first = (axis * 104_729 + 17) % count
    second = (axis * 130_363 + 7_919) % count
    second = torch.where(second == first, (second + 1) % count, second)
    if bool((first == second).any()):
        raise RuntimeError("anti-collapse deterministic pair axis contains self-pairs")
    return first, second


def deterministic_gram_axis(regions: int) -> torch.Tensor:
    count = int(regions)
    samples = min(formal.GRAM_REGION_SAMPLE_CAP, count)
    if samples < 2:
        raise ValueError("anti-collapse Gram audit requires at least two regions")
    axis = torch.div(
        torch.arange(samples, dtype=torch.long) * count,
        samples,
        rounding_mode="floor",
    )
    if axis.unique().numel() != samples:
        raise RuntimeError("anti-collapse deterministic Gram axis repeats a region")
    return axis


def descriptor_statistics(value: torch.Tensor) -> dict[str, float | int]:
    descriptor = torch.as_tensor(value).detach().float().cpu().contiguous()
    if (
        descriptor.ndim != 2
        or descriptor.shape[0] < 2
        or descriptor.shape[1] != 1536
        or not bool(torch.isfinite(descriptor).all())
    ):
        raise ValueError("anti-collapse descriptor must be finite CPU [N,1536]")
    norms = torch.linalg.vector_norm(descriptor, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0.0, atol=2e-4):
        raise ValueError("anti-collapse descriptor must be unit L2")
    regions = int(descriptor.shape[0])
    centroid = descriptor.double().mean(dim=0)
    centroid_squared_norm = float(centroid.square().sum())

    first, second = deterministic_pair_axis(regions)
    pair_parts: list[torch.Tensor] = []
    for start in range(0, first.numel(), _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, first.numel())
        left = descriptor[first[start:stop]].double()
        right = descriptor[second[start:stop]].double()
        pair_parts.append((left * right).sum(dim=-1).cpu())
    pair_cosine = torch.cat(pair_parts)

    centered_sum = 0.0
    for start in range(0, regions, _ROW_CHUNK):
        centered = descriptor[start : start + _ROW_CHUNK].double() - centroid
        centered_sum += float(centered.square().sum())
    centered_radius = centered_sum / regions

    gram_axis = deterministic_gram_axis(regions)
    centered_sample = descriptor[gram_axis].double() - centroid
    gram = centered_sample @ centered_sample.T
    trace = float(torch.trace(gram))
    frobenius_squared = float(gram.square().sum())
    if trace <= 0.0 or frobenius_squared <= 0.0:
        raise ValueError("anti-collapse centered Gram has zero spread")
    effective_rank = trace * trace / frobenius_squared
    return {
        "regions": regions,
        "pair_samples": int(first.numel()),
        "gram_region_samples": int(gram_axis.numel()),
        "centroid_squared_norm": centroid_squared_norm,
        "pair_cosine_mean": float(pair_cosine.mean()),
        "pair_cosine_p90": float(torch.quantile(pair_cosine, 0.9)),
        "centered_mean_squared_radius": centered_radius,
        "centered_gram_effective_rank": effective_rank,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to clobber health audit: {output}")
    descriptor_record = {
        "path": str(Path(args.target_descriptor).expanduser().resolve()),
        "sha256": str(args.expected_target_descriptor_sha256),
    }
    baseline_record = {
        "path": str(Path(args.accepted_v2_baseline).expanduser().resolve()),
        "sha256": str(args.expected_accepted_v2_baseline_sha256),
    }
    descriptor_raw, descriptor_sha, descriptor_path = load_torch_mapping(
        descriptor_record["path"], expected_sha256=descriptor_record["sha256"],
        map_location="cpu", label="anti-collapse target descriptor",
    )
    descriptor = descriptor_formal.validate_target_descriptor_authority(
        descriptor_raw
    )
    baseline_raw, baseline_sha, baseline_path = load_torch_mapping(
        baseline_record["path"], expected_sha256=baseline_record["sha256"],
        map_location="cpu", label="anti-collapse AcceptedV2 baseline",
    )
    baseline = validate_target_accepted_v2_authority(baseline_raw)
    descriptor_record = {"path": str(descriptor_path), "sha256": descriptor_sha}
    baseline_record = {"path": str(baseline_path), "sha256": baseline_sha}
    candidate_values = descriptor["semantic_descriptor"]
    baseline_values = baseline["accepted_v2_e0"]
    active = descriptor["active_update_mask"]
    exact = descriptor["exact_state_anchor_mask"]
    fallback = descriptor["immutable_fallback_mask"]
    unit_finite = all(
        bool(torch.isfinite(values).all())
        and torch.allclose(
            torch.linalg.vector_norm(values, dim=-1),
            torch.ones(values.shape[0]), rtol=0.0, atol=2e-4,
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
            descriptor["input_authority"]["target_accepted_v2"]
            == baseline_record
        ),
        "exact_active_masks_equal": torch.equal(exact, active),
        "fallback_mask_complement_active": torch.equal(fallback, ~active),
        "fallback_descriptor_bitwise_equal": torch.equal(
            candidate_values[fallback], baseline_values[fallback]
        ),
        "candidate_and_baseline_unit_l2_finite": unit_finite,
    }
    if not all(alignment.values()):
        raise ValueError("factorized-native health descriptor/baseline alignment differs")
    candidate_stats = descriptor_statistics(candidate_values)
    baseline_stats = descriptor_statistics(baseline_values)
    checks = formal.expected_checks(candidate_stats, baseline_stats)
    eligible = all(checks.values())
    payload = {
        "schema": formal.HEALTH_AUDIT_SCHEMA,
        "schema_version": formal.HEALTH_AUDIT_SCHEMA_VERSION,
        "contract": formal.health_contract(),
        "contract_sha256": formal.HEALTH_CONTRACT_SHA256,
        "status": "pass" if eligible else "reject_more_collapsed_than_accepted",
        "scene_id": descriptor["scene_id"],
        "physical_space_id": descriptor["physical_space_id"],
        "producer": file_record(Path(__file__).resolve()),
        "input_authority": {
            "target_descriptor": descriptor_record,
            "accepted_v2_baseline": baseline_record,
        },
        "descriptor_channel_sha256": dict(descriptor["channel_sha256"]),
        "baseline_channel_sha256": dict(baseline["channel_sha256"]),
        "alignment_audit": alignment,
        "candidate_statistics": candidate_stats,
        "baseline_statistics": baseline_stats,
        "checks": checks,
        "query_authority_eligible": eligible,
        "access_audit": formal.health_access_audit(),
    }
    payload = formal.validate_health_audit(payload)
    write_frozen_json(output, payload)
    return {
        "status": payload["status"],
        "query_authority_eligible": eligible,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "output": file_record(output),
        "access_audit": formal.health_access_audit(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-descriptor", required=True)
    parser.add_argument("--expected-target-descriptor-sha256", required=True)
    parser.add_argument("--accepted-v2-baseline", required=True)
    parser.add_argument("--expected-accepted-v2-baseline-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    print(json.dumps(audit(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "audit",
    "build_parser",
    "descriptor_statistics",
    "deterministic_gram_axis",
    "deterministic_pair_axis",
]
