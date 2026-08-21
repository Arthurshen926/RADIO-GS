#!/usr/bin/env python3
"""Compose one LERF posterior from a source-gated object-aware field pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from radio_gs.querying.latent_proposal_posterior import (
    DIFFERENT_RELATION,
    SAME_RELATION,
    UNKNOWN_RELATION,
    latent_proposal_null_posterior,
)
from radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores import (
    _select_embedding_rows,
)
from radio_gs.scripts.build_lerf_reliable_proposal_component_posterior import compose
from radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores import (
    _score_embeddings,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_object_aware_field_v2_text_posterior.v1"


def object_aware_authority(
    authority: dict[str, Any],
    checkpoint: dict[str, Any],
    membership: dict[str, Any],
    text_payload: dict[str, Any],
    canonical_payload: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Fuse mask language and fill only previously unknown relation edges."""

    result = dict(authority)
    query_names = [str(value) for value in authority["query_names"]]
    decoded = torch.as_tensor(checkpoint["decoded_object_language"]).float()
    text = torch.nn.functional.normalize(
        _select_embedding_rows(text_payload, query_names), dim=-1
    )
    canonical = torch.nn.functional.normalize(
        torch.as_tensor(canonical_payload["embeddings"]).float(), dim=-1
    )
    object_score = _score_embeddings(
        decoded, text, canonical, device=device, chunk_size=8192
    )
    old_score = torch.as_tensor(authority["descriptor_score"]).float()
    descriptor = 0.5 * old_score + 0.5 * object_score

    relation = torch.as_tensor(authority["edge_relation"]).to(torch.int8).clone()
    affinity = torch.as_tensor(checkpoint["edge_affinity"]).float()
    if relation.shape != affinity.shape:
        raise ValueError("checkpoint/relation edge axes differ")
    threshold = float(checkpoint["source_same_threshold"])
    unknown = relation == UNKNOWN_RELATION
    relation[unknown & (affinity >= threshold)] = SAME_RELATION
    relation[unknown & (affinity <= 0.0)] = DIFFERENT_RELATION
    # The interval (0, source same threshold) is epistemic unknown.
    left = torch.as_tensor(authority["edge_left"]).long()
    right = torch.as_tensor(authority["edge_right"]).long()
    proposals, queries = descriptor.shape
    same_strength = torch.sigmoid(8.0 * (affinity - threshold))
    same_max = torch.zeros(proposals)
    same = relation == SAME_RELATION
    if bool(same.any()):
        same_max.scatter_reduce_(0, left[same], same_strength[same], reduce="amax", include_self=True)
        same_max.scatter_reduce_(0, right[same], same_strength[same], reduce="amax", include_self=True)
    field = torch.as_tensor(authority["field_tail"]).float()
    core = torch.as_tensor(authority["core_fraction"]).float()
    peak = torch.as_tensor(authority["peak_membership"]).float()
    quality = torch.as_tensor(membership["proposal_scores"]).float().clamp(0, 1)
    identity = (
        0.55 * descriptor + 0.20 * field + 0.15 * core.sqrt() + 0.10 * peak
    ) * (0.75 + 0.25 * quality[:, None])
    # Preserve the source-gated identity support; v2 changes association and
    # object-language ranking, not proposal admission.
    valid = torch.as_tensor(authority["proposal_valid"]).bool()
    logits = 8.0 * (identity - 0.55) + torch.log(same_max.clamp_min(1e-8))[:, None]
    counts = valid.sum(0)
    logits -= torch.log(counts.clamp_min(1).float())[None]
    rows = torch.as_tensor(membership["row_indices"]).long()
    props = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    maximum = torch.zeros(proposals)
    maximum.scatter_reduce_(0, props, weights, reduce="amax", include_self=True)
    conditional = weights / maximum[props].clamp_min(1e-8)
    posterior = latent_proposal_null_posterior(
        torch.zeros((int(membership["num_rows"]), queries)),
        rows,
        props,
        conditional.clamp(0, 1),
        logits,
        torch.zeros(queries),
        proposal_valid=valid,
    )
    result.update(
        {
            "edge_relation": relation,
            "edge_same_strength": same_strength,
            "same_strength_max": same_max,
            "descriptor_score": descriptor,
            "object_language_score": object_score,
            "identity_score": identity,
            "proposal_logits": logits,
            "proposal_probability": posterior.proposal_probability,
            "null_probability": posterior.null_probability,
        }
    )
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "anchor": args.anchor,
            "authority": args.authority,
            "checkpoint": args.checkpoint,
            "membership": args.membership,
            "text": args.text_embedding_cache,
            "canonical": args.canonical_embedding_cache,
        }.items()
    }
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"posterior output exists: {output}")
    loaded = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    checkpoint = loaded["checkpoint"]
    if checkpoint.get("metadata", {}).get("benchmark_masks_opened") is not False:
        raise ValueError("object-aware checkpoint information contract differs")
    enriched = object_aware_authority(
        loaded["authority"], checkpoint, loaded["membership"], loaded["text"],
        loaded["canonical"], device=torch.device(args.device)
    )
    payload = compose(loaded["anchor"], enriched, loaded["membership"])
    payload["schema"] = SCHEMA
    payload["metadata"].update(
        {
            "typed_posterior": "object_aware_universal_field_v2_text_object_posterior_v1",
            "object_affinity_dim": int(checkpoint["metadata"]["object_dim"]),
            "object_language_fusion": "equal_old_crop_and_distilled_object_language",
            "unknown_relation_policy": "same_if_source_threshold;different_if_cosine_le_0;unknown_otherwise",
            "persistent_second_semantic_field": False,
            "object_aware_checkpoint": {
                "path": str(paths["checkpoint"]), "sha256": sha256_file(paths["checkpoint"])
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    counts = {name: int((enriched["edge_relation"] == value).sum()) for name, value in (("same", 1), ("different", 0), ("unknown", -1))}
    report = {
        "schema": SCHEMA, "status": "complete", "scene": str(args.scene),
        "relations": counts,
        "selected_proposal_counts": payload["metadata"]["selected_proposal_counts"],
        "output": str(output), "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--text-embedding-cache", required=True)
    parser.add_argument("--canonical-embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
