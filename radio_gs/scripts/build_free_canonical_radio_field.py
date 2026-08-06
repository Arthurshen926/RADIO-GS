#!/usr/bin/env python3
"""Build a full-dimensional, query-free canonical render-ceiling field.

The field uses an identity affine decoder and one free RADIO vector per
Gaussian.  It is initialized from a geometry-matched MPR cache so that the
existing exact-render fitting script can isolate representation capacity from
geometry/compositing limits without opening benchmark queries or masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.field import (
    AffineBasisDecoder,
    CanonicalGaussianField,
    load_canonical_field_checkpoint,
    validate_basis_conditioning,
)
from radio_gs.field.field_signature import FeatureSpaceSignature


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_free_field_payload(
    mpr_payload: dict,
    *,
    source_path: str,
    feature_signature: dict | None = None,
) -> dict:
    features = torch.as_tensor(mpr_payload["features"]).float()
    valid = torch.as_tensor(mpr_payload["valid"]).bool()
    reliability = torch.as_tensor(mpr_payload["reliability"]).float()
    if features.ndim != 2:
        raise ValueError("MPR features must be [num_gaussians,feature_dim]")
    if valid.shape != (features.shape[0],):
        raise ValueError("MPR validity must align with feature rows")
    if reliability.ndim != 2 or reliability.shape[0] != features.shape[0]:
        raise ValueError("MPR reliability must align with feature rows")
    metadata = dict(mpr_payload.get("metadata", {}))
    if bool(metadata.get("benchmark_masks_opened", False)):
        raise ValueError("MPR provenance reports benchmark-mask access")
    if bool(metadata.get("text_queries_opened", False)):
        raise ValueError("MPR provenance reports text-query access")

    feature_dim = int(features.shape[1])
    signature_values = dict(feature_signature or mpr_payload.get("feature_signature", {}))
    if not signature_values:
        raise ValueError(
            "MPR cache has no feature signature; provide a geometry-matched "
            "reference field signature"
        )
    signature_values["raw_feature_dim"] = feature_dim
    signature_values["token_type"] = "primitive"
    signature_values["field_checkpoint_sha256"] = ""
    signature = FeatureSpaceSignature.from_mapping(signature_values)
    decoder = AffineBasisDecoder(
        feature_dim=feature_dim,
        coefficient_dim=feature_dim,
        mean=torch.zeros(feature_dim),
        scale=torch.ones(feature_dim),
        basis=torch.eye(feature_dim),
        trainable_basis=False,
        trainable_statistics=False,
    )
    field = CanonicalGaussianField(
        num_gaussians=features.shape[0],
        decoder=decoder,
        signature=signature,
        local_dim=feature_dim,
        reliability=reliability,
        fusion_reliability=False,
        use_fusion=False,
    )
    with torch.no_grad():
        field.local_codes.copy_(features)
    fingerprint = dict(mpr_payload.get("geometry_fingerprint", {}))
    basis_conditioning = validate_basis_conditioning(
        field.decoder.basis
    ).to_dict()
    return {
        "schema_version": 1,
        "architecture": {
            "num_gaussians": int(features.shape[0]),
            "feature_dim": feature_dim,
            "coefficient_dim": feature_dim,
            "local_dim": feature_dim,
            "coarse_dim": 0,
            "spatial_hash": None,
            "position_storage": "none",
            "fusion_reliability": False,
            "hidden_dim": 1,
            "use_fusion": False,
            "trainable_basis": False,
            "trainable_statistics": False,
        },
        "feature_signature": signature.to_dict(),
        "state_dict": field.state_dict(),
        "reliability": reliability.half(),
        "geometry_fingerprint": fingerprint,
        "mpr_cache": str(Path(source_path).resolve()),
        "mpr_cache_metadata": metadata,
        "basis_fit_report": {
            "coefficient_dim": feature_dim,
            "feature_dim": feature_dim,
            "sample_count": int(valid.sum()),
            "explained_variance_ratio": 1.0,
            "reconstruction_cosine": 1.0,
        },
        "basis_conditioning": basis_conditioning,
        "render_ceiling": {
            "kind": "full_dimensional_free_primitive",
            "identity_decoder": True,
            "mpr_initialization": True,
            "query_free": True,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
        },
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument(
        "--reference-field-checkpoint",
        required=True,
        help="Geometry-matched field used only to inherit the RADIO signature.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mpr_path = Path(args.mpr_cache)
    payload = torch.load(mpr_path, map_location="cpu")
    _field, reference_payload = load_canonical_field_checkpoint(
        args.reference_field_checkpoint, map_location="cpu"
    )
    if dict(reference_payload.get("geometry_fingerprint", {})) != dict(
        payload.get("geometry_fingerprint", {})
    ):
        raise ValueError("reference field and MPR geometry fingerprints differ")
    output_payload = build_free_field_payload(
        payload,
        source_path=str(mpr_path),
        feature_signature=dict(reference_payload["feature_signature"]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output)
    report = {
        "output": str(output),
        "mpr_cache": str(mpr_path.resolve()),
        "num_gaussians": output_payload["architecture"]["num_gaussians"],
        "feature_dim": output_payload["architecture"]["feature_dim"],
        "valid_gaussians": output_payload["basis_fit_report"]["sample_count"],
        "geometry_fingerprint": output_payload["geometry_fingerprint"],
        "identity_decoder": True,
        "query_free": True,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
