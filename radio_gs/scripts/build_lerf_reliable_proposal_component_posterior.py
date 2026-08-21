#!/usr/bin/env python3
"""Build a bounded LERF posterior from one reliable proposal component."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.querying.latent_proposal_posterior import (
    DIFFERENT_RELATION,
    SAME_RELATION,
    latent_proposal_null_posterior,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_reliable_proposal_component_posterior.v1"


def select_reliable_components(authority: Mapping[str, Any]) -> torch.Tensor:
    """Select the same-instance component whose seed beats known different instances."""

    valid = torch.as_tensor(authority.get("proposal_valid")).bool().cpu()
    descriptor = torch.as_tensor(authority.get("descriptor_score")).float().cpu()
    field = torch.as_tensor(authority.get("field_tail")).float().cpu()
    probability = torch.as_tensor(authority.get("proposal_probability")).float().cpu()
    left = torch.as_tensor(authority.get("edge_left")).long().cpu()
    right = torch.as_tensor(authority.get("edge_right")).long().cpu()
    relation = torch.as_tensor(authority.get("edge_relation")).to(torch.int8).cpu()
    if valid.shape != descriptor.shape or field.shape != valid.shape or probability.shape != valid.shape:
        raise ValueError("proposal/query reliability axes differ")
    proposals, queries = valid.shape
    selected = torch.zeros_like(valid)
    same_neighbors: list[list[int]] = [[] for _ in range(proposals)]
    different_neighbors: list[list[int]] = [[] for _ in range(proposals)]
    for a, b, label in zip(left.tolist(), right.tolist(), relation.tolist()):
        target = same_neighbors if label == SAME_RELATION else different_neighbors if label == DIFFERENT_RELATION else None
        if target is not None:
            target[a].append(b); target[b].append(a)
    for query in range(queries):
        candidates = torch.where(valid[:, query])[0].tolist()
        reliable: list[int] = []
        for proposal in candidates:
            competitors = [value for value in different_neighbors[proposal] if bool(valid[value, query])]
            if competitors and (
                float(descriptor[proposal, query]) <= max(float(descriptor[value, query]) for value in competitors)
                or float(field[proposal, query]) <= max(float(field[value, query]) for value in competitors)
            ):
                continue
            reliable.append(proposal)
        if not reliable:
            continue
        seed = max(reliable, key=lambda value: (float(probability[value, query]), -value))
        stack = [seed]; component: set[int] = set()
        while stack:
            proposal = stack.pop()
            if proposal in component or not bool(valid[proposal, query]):
                continue
            component.add(proposal)
            stack.extend(same_neighbors[proposal])
        if len({int(torch.as_tensor(authority["proposal_view_indices"])[value]) for value in component}) < 2:
            continue
        selected[list(component), query] = True
    return selected


def compose(anchor: Mapping[str, Any], authority: Mapping[str, Any], membership: Mapping[str, Any]) -> dict[str, Any]:
    scores = torch.as_tensor(anchor.get("query_scores")).float().cpu()
    valid_rows = torch.as_tensor(anchor.get("valid")).bool().cpu()
    xyz = torch.as_tensor(anchor.get("xyz")).float().cpu()
    anchor_meta = dict(anchor.get("metadata", {}))
    query_names = [str(value) for value in authority.get("query_names", [])]
    if scores.ndim != 2 or valid_rows.shape != (scores.shape[0],) or query_names != list(anchor_meta.get("query_names", [])):
        raise ValueError("anchor/authority identity differs")
    rows = torch.as_tensor(membership.get("row_indices")).long().cpu()
    props = torch.as_tensor(membership.get("proposal_indices")).long().cpu()
    weights = torch.as_tensor(membership.get("weights")).float().cpu()
    proposal_count = int(membership.get("num_proposals", -1))
    proposal_max = torch.zeros(proposal_count); proposal_max.scatter_reduce_(0, props, weights, reduce="amax", include_self=True)
    conditional = weights / proposal_max[props].clamp_min(1e-8)
    selected = select_reliable_components(authority)
    logits = torch.as_tensor(authority.get("proposal_logits")).float().cpu()
    primitive = scores.clone(); primitive[~valid_rows] = 0; primitive = primitive.clamp(0, 1)
    marginal = latent_proposal_null_posterior(
        primitive, rows, props, conditional.clamp(0,1), logits, torch.zeros(scores.shape[1]), proposal_valid=selected
    ).probability
    marginal[~valid_rows] = scores[~valid_rows]
    identity = anchor.get("identity_query_scores")
    if identity is not None: identity = torch.as_tensor(identity).float().cpu()
    return {
        "schema": SCHEMA, "schema_version": 1, "scene": str(anchor.get("scene", "")),
        "query_scores": marginal, "identity_query_scores": identity, "valid": valid_rows, "xyz": xyz,
        "metadata": {
            "query_names": query_names, "query_family": "text_object_extent",
            "typed_posterior": "official_sam3_siglip2_identity_extent_factorization_reliable_component_v1",
            "separate_identity_localization": True, "localization_authority": "field_siglip2_relevancy_identity",
            "segmentation_authority": "sam_instance_extent_posterior",
            "proposal_selection": "same_component_seed_strictly_outranks_known_different_in_crop_and_field_identity",
            "selected_proposal_counts": selected.sum(dim=0).tolist(),
            "gaussian_union": False, "persistent_second_semantic_field": False,
            "benchmark_masks_opened": False, "evaluation_rgb_opened": False, "development_evidence": True,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    anchor_path=Path(args.anchor).resolve(); authority_path=Path(args.authority).resolve(); membership_path=Path(args.membership).resolve(); output=Path(args.output).resolve(); report_path=output.with_suffix(output.suffix+".json")
    if output.exists() or report_path.exists(): raise FileExistsError(f"output exists: {output}")
    anchor=torch.load(anchor_path,map_location="cpu",weights_only=False); authority=torch.load(authority_path,map_location="cpu",weights_only=False); membership=torch.load(membership_path,map_location="cpu",weights_only=False)
    payload=compose(anchor,authority,membership)
    payload["metadata"].update({"anchor":{"path":str(anchor_path),"sha256":sha256_file(anchor_path)},"authority":{"path":str(authority_path),"sha256":sha256_file(authority_path)},"membership":{"path":str(membership_path),"sha256":sha256_file(membership_path)}})
    output.parent.mkdir(parents=True,exist_ok=True); temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp"); torch.save(payload,temporary); os.replace(temporary,output)
    report={"schema":SCHEMA,"status":"complete","scene":payload["scene"],"selected_proposal_counts":payload["metadata"]["selected_proposal_counts"],"output":str(output),"output_sha256":sha256_file(output)}; report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--anchor",required=True); parser.add_argument("--authority",required=True); parser.add_argument("--membership",required=True); parser.add_argument("--output",required=True); print(json.dumps(build(parser.parse_args()),indent=2,sort_keys=True))


if __name__=="__main__": main()
