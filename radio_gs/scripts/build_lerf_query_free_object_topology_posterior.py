#!/usr/bin/env python3
"""Compile text identity and query-free source SAM into one object posterior."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from radio_gs.querying.identity_seeded_object_topology import (
    compile_view_exclusive_physical_tracks,
    identity_seeded_object_topology_posterior,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_payload,
    write_frozen_json,
    write_torch_noclobber,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    identity, identity_sha, identity_path = load_torch_payload(
        args.identity_posterior, expected_sha256=args.expected_identity_posterior_sha256,
        label="LERF typed identity posterior",
    )
    membership, membership_sha, membership_path = load_torch_payload(
        args.membership_cache, expected_sha256=args.expected_membership_cache_sha256,
        label="LERF query-free source SAM membership",
    )
    inference, inference_sha, inference_path = load_torch_payload(
        args.physical_track_inference,
        expected_sha256=args.expected_physical_track_inference_sha256,
        label="LERF label-free physical-track inference features",
    )
    calibrator, calibrator_sha, calibrator_path = load_torch_payload(
        args.physical_track_calibrator,
        expected_sha256=args.expected_physical_track_calibrator_sha256,
        label="LERF source-only physical-track calibrator",
    )
    if not isinstance(identity, Mapping) or not isinstance(membership, Mapping):
        raise ValueError("LERF topology inputs must be mappings")
    membership_metadata = membership.get("metadata", {})
    if (
        membership.get("schema") != "radio_gs.lerf_multiscale_sam3_exact_mpr_memberships.v1"
        or membership_metadata.get("query_independent_proposal_set") is not True
        or membership_metadata.get("query_independent_mask_hierarchy") is not True
    ):
        raise ValueError("LERF source SAM hierarchy is not query-independent")
    if any(bool(membership_metadata.get(key, False)) for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened", "text_queries_opened",
    )):
        raise ValueError("LERF source SAM hierarchy opened a forbidden channel")
    identity_channel = identity.get("identity_query_scores", identity.get("query_scores"))
    scores = torch.as_tensor(identity_channel).float()
    xyz = torch.as_tensor(identity.get("xyz")).float()
    valid = torch.as_tensor(identity.get("valid")).bool()
    identity_metadata = identity.get("metadata", {})
    query_names = list(map(str, identity_metadata.get("query_names", [])))
    if (
        scores.ndim != 2 or xyz.shape != (scores.shape[0], 3)
        or valid.shape != (scores.shape[0],) or len(query_names) != scores.shape[1]
        or int(membership.get("num_rows", -1)) != scores.shape[0]
    ):
        raise ValueError("LERF identity and topology row/query domains differ")
    device = torch.device(args.device)
    proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
    feature_names = list(map(str, inference.get("feature_names", [])))
    if (
        inference.get("schema") != "radio_gs.lerf_physical_track_inference_features.v1"
        or calibrator.get("schema") != "radio_gs.lerf_source_physical_track_calibrator_nested.v1"
        or feature_names != list(map(str, calibrator.get("feature_names", [])))
        or inference.get("metadata", {}).get("label_free") is not True
        or calibrator.get("metadata", {}).get("figurines_opened") is not False
    ):
        raise ValueError("LERF physical-track source/heldout contract differs")
    edge_features = torch.as_tensor(inference["edge_features"]).float()
    standardized = (
        edge_features - torch.as_tensor(calibrator["feature_mean"]).float()
    ) / torch.as_tensor(calibrator["feature_scale"]).float()
    edge_probability = torch.sigmoid((
        (standardized * torch.as_tensor(calibrator["weight"]).float()).sum(1)
        + float(calibrator["bias"])
    ) / float(calibrator["temperature"]))
    proposal_tracks = compile_view_exclusive_physical_tracks(
        torch.as_tensor(inference["edge_left"]).long(),
        torch.as_tensor(inference["edge_right"]).long(),
        edge_probability, proposal_views,
        minimum_probability=0.5, minimum_views=args.minimum_object_views,
    )
    posterior, stats = identity_seeded_object_topology_posterior(
        scores.to(device),
        torch.as_tensor(membership["row_indices"]).long().to(device),
        torch.as_tensor(membership["proposal_indices"]).long().to(device),
        torch.as_tensor(membership["weights"]).float().to(device),
        proposal_views.to(device),
        torch.full((proposal_views.numel(),), -1, dtype=torch.long, device=device),
        proposal_track_indices=proposal_tracks.to(device),
        proposal_scores=torch.as_tensor(membership["proposal_scores"]).float().to(device),
        seed_support_ratio=args.seed_support_ratio,
        identity_core_ratio=args.identity_core_ratio,
        minimum_object_views=args.minimum_object_views,
        minimum_row_views=args.minimum_row_views,
        extent_membership_floor=args.extent_membership_floor,
        sibling_exclusion_strength=args.sibling_exclusion_strength,
        unknown_policy=args.unknown_policy,
        membership_calibration="proposal_max",
        use_proposal_quality=False,
    )
    membership_observed = torch.as_tensor(membership["view_observed"]).bool().any(0)
    output_valid = valid | membership_observed
    output = Path(args.output).resolve()
    payload = {
        "schema": "radio_gs.lerf_query_free_object_topology_posterior.v1",
        "schema_version": 1, "scene": str(identity.get("scene")),
        "query_scores": posterior.cpu().contiguous(),
        "identity_query_scores": scores.cpu().contiguous(),
        "valid": output_valid.cpu(), "xyz": xyz.cpu(),
        "metadata": {
            "query_names": query_names, "query_family": "text_object_extent",
            "typed_posterior": "object_aware_universal_field_v2_text_object_posterior_query_free_source_sam_v1",
            "score_threshold": 0.5, "separate_identity_localization": True,
            "localization_authority": identity_metadata.get("localization_authority"),
            "segmentation_authority": "identity_seeded_query_free_source_sam_exact_mpr_topology",
            "persistent_second_semantic_field": False,
            "query_independent_mask_hierarchy": True,
            "benchmark_images_opened": False, "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False, "query_text_opened_at_readout": True,
            "topology": stats,
            "identity_posterior": {"path": str(identity_path), "sha256": identity_sha},
            "membership_cache": {"path": str(membership_path), "sha256": membership_sha},
            "physical_track_inference": {"path": str(inference_path), "sha256": inference_sha},
            "physical_track_calibrator": {"path": str(calibrator_path), "sha256": calibrator_sha},
            "physical_track_compiler": "calibrated_p_ge_0.5_view_exclusive_maximum_confidence_forest",
            "identity_channel": (
                "identity_query_scores" if identity.get("identity_query_scores") is not None
                else "query_scores"
            ),
        },
    }
    write_torch_noclobber(output, payload)
    report = {
        "status": "complete", "scene": payload["scene"],
        "queries": len(query_names), "valid_rows": int(output_valid.sum()),
        "physical_tracks": int(proposal_tracks.max().item() + 1),
        "tracked_proposals": int((proposal_tracks >= 0).sum()),
        "topology": stats, "output": file_record(output),
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-posterior", required=True)
    parser.add_argument("--expected-identity-posterior-sha256", required=True)
    parser.add_argument("--membership-cache", required=True)
    parser.add_argument("--expected-membership-cache-sha256", required=True)
    parser.add_argument("--physical-track-inference", required=True)
    parser.add_argument("--expected-physical-track-inference-sha256", required=True)
    parser.add_argument("--physical-track-calibrator", required=True)
    parser.add_argument("--expected-physical-track-calibrator-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-support-ratio", type=float, default=0.8)
    parser.add_argument("--identity-core-ratio", type=float, default=0.8)
    parser.add_argument("--minimum-object-views", type=int, default=2)
    parser.add_argument("--minimum-row-views", type=int, default=1)
    parser.add_argument("--extent-membership-floor", type=float, default=0.5)
    parser.add_argument("--sibling-exclusion-strength", type=float, default=0.0)
    parser.add_argument(
        "--unknown-policy", choices=("preserve_text_prior", "negative_outside_topology"),
        default="preserve_text_prior",
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
