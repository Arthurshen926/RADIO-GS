#!/usr/bin/env python3
"""Compose a LERF extent anchor with a latent proposal/null completion.

This development sentinel is intentionally parameter free.  The deterministic
source-view SAM extent remains an absorbing lower bound; the proposal/null
marginal may only add support.  Consequently a weak or null latent proposal
cannot erase an already established object extent or its field identity peak.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_anchor_preserving_latent_completion.v1"


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("metadata", {})
    if not isinstance(value, Mapping):
        raise ValueError("score-cache metadata must be an object")
    return dict(value)


def compose(anchor_payload: Mapping[str, Any], marginal_payload: Mapping[str, Any]) -> dict[str, Any]:
    anchor = torch.as_tensor(anchor_payload.get("query_scores")).float().cpu()
    marginal = torch.as_tensor(marginal_payload.get("query_scores")).float().cpu()
    anchor_xyz = torch.as_tensor(anchor_payload.get("xyz")).float().cpu()
    marginal_xyz = torch.as_tensor(marginal_payload.get("xyz")).float().cpu()
    anchor_valid = torch.as_tensor(anchor_payload.get("valid")).bool().cpu()
    marginal_valid = torch.as_tensor(marginal_payload.get("valid")).bool().cpu()
    anchor_meta = _metadata(anchor_payload)
    marginal_meta = _metadata(marginal_payload)
    if (
        anchor.ndim != 2
        or anchor.shape != marginal.shape
        or anchor_xyz.shape != (anchor.shape[0], 3)
        or not torch.equal(anchor_xyz, marginal_xyz)
        or anchor_valid.shape != (anchor.shape[0],)
        or not torch.equal(anchor_valid, marginal_valid)
        or anchor_meta.get("query_names") != marginal_meta.get("query_names")
    ):
        raise ValueError("anchor and marginal score identities differ")
    valid_values = anchor_valid[:, None].expand_as(anchor)
    if not bool(torch.isfinite(anchor[valid_values]).all()) or not bool(
        torch.isfinite(marginal[valid_values]).all()
    ):
        raise ValueError("valid score values must be finite")
    result = torch.maximum(anchor, marginal)
    result[~anchor_valid] = anchor[~anchor_valid]
    identity = anchor_payload.get("identity_query_scores")
    if identity is not None:
        identity = torch.as_tensor(identity).float().cpu()
        if identity.shape != anchor.shape:
            raise ValueError("anchor identity score shape differs")
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": str(anchor_payload.get("scene", "")),
        "query_scores": result,
        "identity_query_scores": identity,
        "valid": anchor_valid,
        "xyz": anchor_xyz,
        "metadata": {
            "query_names": list(anchor_meta.get("query_names", [])),
            "query_family": "text_object_extent",
            "typed_posterior": (
                "official_sam3_siglip2_identity_extent_factorization_"
                "anchor_preserving_latent_completion_v1"
            ),
            "composition": "pointwise_max_absorbing_extent_anchor",
            "separate_identity_localization": True,
            "localization_authority": "field_siglip2_relevancy_identity",
            "segmentation_authority": "sam_instance_extent_posterior",
            "parameter_free": True,
            "changes_proposal_identity": False,
            "can_delete_anchor_support": False,
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
        "changed_fraction": float(
            (payload["query_scores"] > torch.as_tensor(anchor["query_scores"]).float()).float().mean()
        ),
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
