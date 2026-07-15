"""Fail-closed checkpoint I/O for a canonical Gaussian RADIO field."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .basis_decoder import AffineBasisDecoder
from .canonical_gaussian_field import CanonicalGaussianField
from .field_signature import FeatureSpaceSignature


def load_canonical_field_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CanonicalGaussianField, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("not a canonical RADIO field schema-v1 checkpoint")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("canonical field checkpoint lacks architecture metadata")
    signature = FeatureSpaceSignature.from_mapping(payload["feature_signature"])
    decoder = AffineBasisDecoder(
        feature_dim=int(architecture["feature_dim"]),
        coefficient_dim=int(architecture["coefficient_dim"]),
        trainable_basis=bool(architecture.get("trainable_basis", True)),
        trainable_statistics=bool(architecture.get("trainable_statistics", False)),
    )
    reliability = payload.get("reliability")
    field = CanonicalGaussianField(
        num_gaussians=int(architecture["num_gaussians"]),
        decoder=decoder,
        signature=signature,
        local_dim=int(architecture["local_dim"]),
        coarse_dim=int(architecture.get("coarse_dim", 0)),
        spatial_hash=architecture.get("spatial_hash"),
        reliability=reliability,
        fusion_reliability=bool(architecture.get("fusion_reliability", True)),
        hidden_dim=int(architecture["hidden_dim"]),
        use_fusion=bool(architecture.get("use_fusion", True)),
    )
    field.load_state_dict(payload["state_dict"], strict=True)
    return field, payload
