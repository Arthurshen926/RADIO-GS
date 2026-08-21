#!/usr/bin/env python3
"""Admit a LERF proposal posterior only under count-corrected null evidence.

For a query with K valid proposals and null probability pi_0, the average
proposal posterior is (1-pi_0)/K.  Requiring this conservative per-proposal
mass to exceed pi_0 prevents a large proposal cohort from defeating null by
multiplicity alone.  Rejected queries fall back to the deterministic source
extent bit-for-bit; no Gaussian-level union is performed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_count_corrected_proposal_gate.v1"


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("score metadata must be an object")
    return dict(metadata)


def compose(anchor_payload: Mapping[str, Any], marginal_payload: Mapping[str, Any]) -> dict[str, Any]:
    anchor = torch.as_tensor(anchor_payload.get("query_scores")).float().cpu()
    marginal = torch.as_tensor(marginal_payload.get("query_scores")).float().cpu()
    xyz = torch.as_tensor(anchor_payload.get("xyz")).float().cpu()
    marginal_xyz = torch.as_tensor(marginal_payload.get("xyz")).float().cpu()
    valid = torch.as_tensor(anchor_payload.get("valid")).bool().cpu()
    marginal_valid = torch.as_tensor(marginal_payload.get("valid")).bool().cpu()
    anchor_meta, marginal_meta = _metadata(anchor_payload), _metadata(marginal_payload)
    if (
        anchor.ndim != 2
        or marginal.shape != anchor.shape
        or xyz.shape != (anchor.shape[0], 3)
        or not torch.equal(xyz, marginal_xyz)
        or valid.shape != (anchor.shape[0],)
        or not torch.equal(valid, marginal_valid)
        or anchor_meta.get("query_names") != marginal_meta.get("query_names")
    ):
        raise ValueError("anchor and marginal score identities differ")
    topology = marginal_meta.get("topology", {})
    if (
        not isinstance(topology, Mapping)
        or "null_probability" not in topology
        or "valid_proposal_counts" not in topology
    ):
        raise ValueError("latent topology metadata is absent")
    null = torch.as_tensor(topology.get("null_probability")).float().cpu()
    counts = torch.as_tensor(topology.get("valid_proposal_counts")).long().cpu()
    if (
        null.shape != (anchor.shape[1],)
        or counts.shape != null.shape
        or not bool(torch.isfinite(null).all())
        or bool(((null < 0) | (null > 1)).any())
        or bool((counts < 0).any())
    ):
        raise ValueError("latent null/count evidence differs")
    denominator = counts.clamp_min(1).float()
    conservative_proposal_mass = (1.0 - null) / denominator
    admitted = (counts > 0) & (conservative_proposal_mass > null)
    scores = torch.where(admitted[None], marginal, anchor)
    scores[~valid] = anchor[~valid]
    identity = anchor_payload.get("identity_query_scores")
    if identity is not None:
        identity = torch.as_tensor(identity).float().cpu()
        if identity.shape != anchor.shape:
            raise ValueError("anchor identity scores differ")
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(anchor_payload.get("scene", "")),
        "query_scores": scores,
        "identity_query_scores": identity,
        "valid": valid,
        "xyz": xyz,
        "metadata": {
            "query_names": list(anchor_meta.get("query_names", [])),
            "query_family": "text_object_extent",
            "typed_posterior": (
                "official_sam3_siglip2_identity_extent_factorization_"
                "count_corrected_proposal_null_gate_v1"
            ),
            "separate_identity_localization": True,
            "localization_authority": "field_siglip2_relevancy_identity",
            "segmentation_authority": "sam_instance_extent_posterior",
            "proposal_admission": "average_proposal_mass_strictly_exceeds_null",
            "proposal_admission_formula": "K>0 and (1-p_null)/K > p_null",
            "admitted_queries": admitted.tolist(),
            "admitted_query_count": int(admitted.sum()),
            "valid_proposal_counts": counts.tolist(),
            "null_probability": null.tolist(),
            "parameter_free": True,
            "gaussian_union": False,
            "persistent_second_semantic_field": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "development_evidence": True,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    anchor_path = Path(args.anchor).expanduser().resolve()
    marginal_path = Path(args.marginal).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"output exists: {output}")
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    marginal = torch.load(marginal_path, map_location="cpu", weights_only=False)
    payload = compose(anchor, marginal)
    payload["metadata"].update(
        {
            "anchor_cache": str(anchor_path),
            "anchor_cache_sha256": sha256_file(anchor_path),
            "marginal_cache": str(marginal_path),
            "marginal_cache_sha256": sha256_file(marginal_path),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": payload["scene"],
        "rows": int(payload["query_scores"].shape[0]),
        "queries": int(payload["query_scores"].shape[1]),
        "admitted_queries": int(payload["metadata"]["admitted_query_count"]),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--marginal", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
