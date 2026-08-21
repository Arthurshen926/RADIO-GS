#!/usr/bin/env python3
"""Compile source-only soft object tracks with exact visibility denominators."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from radio_gs.querying.object_aware_visibility_track_posterior import (
    object_aware_visibility_track_posterior,
)
from radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores import _select_embedding_rows
from radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores import _score_embeddings
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_object_aware_visibility_track_posterior.v1"


def _calibrated_association_logit(features: torch.Tensor, calibrator: dict) -> torch.Tensor:
    """Apply either legacy standardization or nested proper calibration."""

    if "feature_mean" in calibrator:
        mean = torch.as_tensor(calibrator["feature_mean"]).float()
        scale = torch.as_tensor(calibrator["feature_scale"]).float()
    else:
        mean = torch.as_tensor(calibrator["mean"]).float()
        scale = torch.as_tensor(calibrator["std"]).float()
    raw = ((features - mean) / scale) @ torch.as_tensor(calibrator["weight"]).float() + torch.as_tensor(calibrator["bias"]).float()
    if calibrator.get("schema") == "radio_gs.lerf_source_physical_track_calibrator_nested.v1":
        temperature = float(calibrator["temperature"])
        strength = float(calibrator["jeffreys_strength"])
        probability = (torch.sigmoid(raw / temperature) + 0.5 * strength) / (1.0 + strength)
        return torch.logit(probability.clamp(1e-7, 1 - 1e-7))
    return raw


def _nested_source_gate_pass(calibrator_path: Path, calibrator: dict) -> bool:
    if calibrator.get("schema") != "radio_gs.lerf_source_physical_track_calibrator_nested.v1":
        return False
    report_path = calibrator_path.with_suffix(calibrator_path.suffix + ".json")
    report = json.loads(report_path.read_text())
    return (
        report.get("status") == "source_heldout_gate_pass"
        and report.get("formal_stage_a_complete") is True
        and report.get("figurines_opened") is False
        and report.get("output_sha256") == sha256_file(calibrator_path)
    )


def build(args: argparse.Namespace) -> dict:
    names = ("v1_posterior", "authority", "checkpoint", "membership", "text", "canonical")
    paths = {name: Path(getattr(args, name)).expanduser().resolve() for name in names}
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"visibility track output exists: {output}")
    values = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in paths.items()}
    v1, authority, checkpoint, membership = (values[name] for name in names[:4])
    query_names = [str(value) for value in v1["metadata"]["query_names"]]
    text = torch.nn.functional.normalize(_select_embedding_rows(values["text"], query_names), dim=-1)
    canonical = torch.nn.functional.normalize(torch.as_tensor(values["canonical"]["embeddings"]).float(), dim=-1)
    object_score = _score_embeddings(
        torch.as_tensor(checkpoint["decoded_object_language"]).float(), text, canonical,
        device=torch.device(args.device), chunk_size=8192,
    )
    association_path = Path(args.association_authority).expanduser().resolve()
    calibrator_path = Path(args.association_calibrator).expanduser().resolve()
    association_authority = torch.load(association_path, map_location="cpu", weights_only=False)
    calibrator = torch.load(calibrator_path, map_location="cpu", weights_only=False)
    nested_gate_pass = _nested_source_gate_pass(calibrator_path, calibrator)
    if calibrator.get("schema") == "radio_gs.lerf_source_physical_track_calibrator_nested.v1" and not nested_gate_pass:
        raise ValueError("nested physical-track source proper gate did not pass")
    if association_authority.get("feature_names") != calibrator.get("feature_names"):
        raise ValueError("association authority/calibrator feature axes differ")
    association_features = torch.as_tensor(association_authority["edge_features"]).float()
    association_logit = _calibrated_association_logit(association_features, calibrator)
    result = object_aware_visibility_track_posterior(
        torch.as_tensor(v1["query_scores"]),
        torch.as_tensor(authority["proposal_probability"]),
        torch.as_tensor(authority["proposal_valid"]), object_score,
        torch.as_tensor(membership["row_indices"]),
        torch.as_tensor(membership["proposal_indices"]),
        torch.as_tensor(membership["weights"]),
        torch.as_tensor(membership["proposal_view_indices"]),
        torch.as_tensor(membership["proposal_area_fraction"]),
        torch.as_tensor(association_authority["edge_left"]),
        torch.as_tensor(association_authority["edge_right"]),
        association_logit / 8.0,
        torch.full((association_features.shape[0],), -1, dtype=torch.int8),
        torch.as_tensor(membership["view_denominator"]),
        torch.as_tensor(membership["view_observed"]),
    )
    payload = dict(v1)
    payload["schema"] = SCHEMA
    payload["query_scores"] = result.probability
    payload["metadata"] = dict(v1["metadata"])
    payload["metadata"].update({
        "typed_posterior": "object_aware_universal_field_v2_text_object_posterior_visibility_track_v1",
        "association": "per_view_exchangeable_proposal_plus_explicit_null_soft_marginal",
        "association_relation": "scene_disjoint_three_scene_independent_DINO_physical_track_nested_proper_calibration;figurines_labels_unopened",
        "proposal_count_prior": "minus_log_K",
        "scale_prior": "negative_absolute_log2_area_ratio",
        "extent_estimator": "association_weighted_positive_exact_MPR_mass_over_association_weighted_exact_visibility_denominator",
        "absence_without_visibility": "unknown",
        "graph_transitive_closure": False,
        "fixed_track_size": False,
        "target_selected_threshold": False,
        "fallback": "bitwise_v1_without_identity_seed_or_visibility_support",
        "fallback_queries": result.fallback.tolist(),
        "seed_proposal": result.seed_proposal.tolist(),
        "persistent_second_semantic_field": False,
        "formal_stage_a_complete": nested_gate_pass,
        "formal_stage_a_boundary": "three_scene_source_RGB_DINO_three_view_cycle_authority_with_explicit_visible_null_and_occlusion_unknown;figurines_labels_unopened",
        "object_aware_checkpoint": {"path": str(paths["checkpoint"]), "sha256": sha256_file(paths["checkpoint"])},
        "association_authority": {"path": str(association_path), "sha256": sha256_file(association_path)},
        "association_calibrator": {"path": str(calibrator_path), "sha256": sha256_file(calibrator_path)},
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, output)
    nonnull = 1.0 - result.null_probability
    report = {
        "schema": SCHEMA, "status": "complete", "scene": str(args.scene),
        "formal_stage_a_complete": payload["metadata"]["formal_stage_a_complete"],
        "fallback_queries": int(result.fallback.sum()),
        "association_nonnull_mean": float(nonnull.mean()),
        "visibility_denominator_positive_rows": (result.visibility_denominator > 0).sum(0).tolist(),
        "output": str(output), "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    for name in ("v1-posterior", "authority", "checkpoint", "membership", "text", "canonical", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--association-authority", required=True)
    parser.add_argument("--association-calibrator", required=True)
    parser.add_argument("--device", default="cpu")
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
