#!/usr/bin/env python3
"""Build identity-seeded object-track LERF posterior from frozen v2 affinity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from radio_gs.querying.object_aware_text_track_posterior import (
    object_aware_text_track_posterior,
)
from radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores import (
    _select_embedding_rows,
)
from radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores import (
    _score_embeddings,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_object_aware_field_v2_text_track_posterior.v1"


def build(args: argparse.Namespace) -> dict:
    names = ("v1_posterior", "authority", "checkpoint", "membership", "text", "canonical")
    paths = {name: Path(getattr(args, name)).expanduser().resolve() for name in names}
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"track posterior exists: {output}")
    values = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in paths.items()}
    v1, authority, checkpoint, membership = (values[name] for name in names[:4])
    query_names = [str(value) for value in v1["metadata"]["query_names"]]
    if query_names != [str(value) for value in authority["query_names"]]:
        raise ValueError("v1/authority query axes differ")
    text = torch.nn.functional.normalize(_select_embedding_rows(values["text"], query_names), dim=-1)
    canonical = torch.nn.functional.normalize(torch.as_tensor(values["canonical"]["embeddings"]).float(), dim=-1)
    object_score = _score_embeddings(
        torch.as_tensor(checkpoint["decoded_object_language"]).float(),
        text, canonical, device=torch.device(args.device), chunk_size=8192,
    )
    result = object_aware_text_track_posterior(
        torch.as_tensor(v1["query_scores"]),
        torch.as_tensor(authority["proposal_probability"]),
        torch.as_tensor(authority["proposal_valid"]),
        object_score,
        torch.as_tensor(membership["row_indices"]),
        torch.as_tensor(membership["proposal_indices"]),
        torch.as_tensor(membership["weights"]),
        torch.as_tensor(membership["proposal_view_indices"]),
        torch.as_tensor(membership["proposal_area_fraction"]),
        torch.as_tensor(authority["edge_left"]),
        torch.as_tensor(authority["edge_right"]),
        torch.as_tensor(checkpoint["edge_affinity"]),
        torch.as_tensor(authority["edge_relation"]),
        same_threshold=float(checkpoint["source_same_threshold"]),
    )
    payload = dict(v1)
    payload["schema"] = SCHEMA
    payload["query_scores"] = result.probability
    payload["metadata"] = dict(v1["metadata"])
    payload["metadata"].update({
        "typed_posterior": "object_aware_universal_field_v2_text_object_posterior_track_v1",
        "identity_admission": "v1_proposal_probability_argmax_among_source_valid",
        "extent_expansion": "query_independent_source_gated_seed_direct_one_hop_one_proposal_per_view_within_one_area_octave",
        "member_query_valid_required": False,
        "signed_relation_veto": "known_different_edges_cannot_join_track",
        "extent_consensus": "at_least_two_independent_source_views_positive;missing_is_unknown",
        "track_language_null": "sigmoid_8_times_track_mean_relevancy_minus_0.5",
        "fallback": "bitwise_v1_on_no_seed_no_edge_or_single_view",
        "fallback_queries": result.fallback.tolist(),
        "seed_proposal": result.seed_proposal.tolist(),
        "track_probability": result.track_probability.tolist(),
        "selected_proposal_counts": result.selected_membership.sum(0).tolist(),
        "persistent_second_semantic_field": False,
        "object_aware_checkpoint": {"path": str(paths["checkpoint"]), "sha256": sha256_file(paths["checkpoint"])},
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, output)
    report = {
        "schema": SCHEMA, "status": "complete", "scene": str(args.scene),
        "fallback_queries": int(result.fallback.sum()),
        "selected_proposal_counts": result.selected_membership.sum(0).tolist(),
        "output": str(output), "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    for name in ("v1-posterior", "authority", "checkpoint", "membership", "text", "canonical", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda:1")
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
