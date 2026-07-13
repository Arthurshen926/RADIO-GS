"""Fail-closed loading of canonical primitive capability banks and graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from radio_gs.field.field_signature import FeatureSpaceSignature
from radio_gs.querying.support_solver import PrimitiveSupportGraph


@dataclass(frozen=True)
class CanonicalCapabilityBank:
    """Row-aligned official views derived from exactly one canonical field."""

    xyz: torch.Tensor
    valid: torch.Tensor
    appearance: torch.Tensor
    boundary: torch.Tensor
    signatures: Mapping[str, FeatureSpaceSignature]
    metadata: Mapping[str, Any]

    @property
    def global_rows(self) -> torch.Tensor:
        return torch.where(self.valid)[0]

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[0])

    def valid_feature_banks(self) -> dict[str, torch.Tensor]:
        rows = self.global_rows
        return {
            "appearance": self.appearance[rows],
            "boundary": self.boundary[rows],
        }


def _load_payload(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported canonical capability cache: {path}")
    return payload


def load_canonical_capability_bank(
    path: str | Path,
    *,
    expected_field_checkpoint_sha256: str = "",
    require_signatures: bool = True,
) -> CanonicalCapabilityBank:
    payload = _load_payload(path)
    required = {"xyz", "valid", "appearance_dino_v3", "boundary_sam3", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"capability cache lacks keys: {sorted(required - set(payload))}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("capability cache metadata must be a mapping")
    if metadata.get("source") != "canonical_radio_field_official_frozen_capability_views":
        raise ValueError("capability cache was not derived from a canonical RADIO field")
    if metadata.get("custom_adaptor_head") is not False:
        raise ValueError("canonical capability cache must use official frozen adaptors")
    if metadata.get("query_independent") is not True:
        raise ValueError("canonical capability cache must be query independent")
    actual_field_hash = str(metadata.get("field_checkpoint_sha256", ""))
    if expected_field_checkpoint_sha256 and actual_field_hash != expected_field_checkpoint_sha256:
        raise ValueError("capability cache canonical-field hash mismatch")

    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    valid = torch.as_tensor(payload["valid"]).bool().cpu()
    appearance = torch.as_tensor(payload["appearance_dino_v3"]).cpu()
    boundary = torch.as_tensor(payload["boundary_sam3"]).cpu()
    count = int(xyz.shape[0]) if xyz.ndim == 2 else -1
    if xyz.ndim != 2 or xyz.shape[1] != 3 or valid.shape != (count,):
        raise ValueError("capability xyz/valid rows are malformed")
    if appearance.ndim != 2 or boundary.ndim != 2:
        raise ValueError("capability features must be matrices")
    if appearance.shape[0] != count or boundary.shape[0] != count:
        raise ValueError("capability feature rows do not align with geometry")
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError("capability geometry contains NaN or infinity")
    if not bool(torch.isfinite(appearance).all()) or not bool(
        torch.isfinite(boundary).all()
    ):
        raise ValueError("capability features contain NaN or infinity")

    raw_signatures = metadata.get("capability_signatures")
    if require_signatures and not isinstance(raw_signatures, Mapping):
        raise ValueError("capability cache lacks fail-closed feature signatures")
    signatures = {
        name: FeatureSpaceSignature.from_mapping(value)
        for name, value in dict(raw_signatures or {}).items()
    }
    for name, matrix in (("appearance", appearance), ("boundary", boundary)):
        signature = signatures.get(name)
        if require_signatures and signature is None:
            raise ValueError(f"capability cache lacks {name} signature")
        if signature is not None and signature.adaptor_output_dim != matrix.shape[1]:
            raise ValueError(f"{name} signature output dimension does not match cache")
    return CanonicalCapabilityBank(
        xyz=xyz,
        valid=valid,
        appearance=appearance,
        boundary=boundary,
        signatures=signatures,
        metadata=metadata,
    )


def load_canonical_support_graph(
    path: str | Path,
    bank: CanonicalCapabilityBank,
) -> PrimitiveSupportGraph:
    payload = _load_payload(path)
    required = {
        "global_rows",
        "num_global_rows",
        "edge_index",
        "edge_weight",
        "raw_affinity",
        "local_sigma",
        "metadata",
    }
    if not required.issubset(payload):
        raise ValueError(f"support graph lacks keys: {sorted(required - set(payload))}")
    if int(payload["num_global_rows"]) != bank.num_gaussians:
        raise ValueError("support graph and capability bank global row counts differ")
    global_rows = torch.as_tensor(payload["global_rows"]).long().cpu()
    if not torch.equal(global_rows, bank.global_rows):
        raise ValueError("support graph nodes do not match valid capability rows")
    metadata = payload["metadata"]
    capability_metadata = metadata.get("capability_metadata", {})
    if capability_metadata.get("field_checkpoint_sha256") != bank.metadata.get(
        "field_checkpoint_sha256"
    ):
        raise ValueError("support graph and capability bank canonical-field hashes differ")
    if capability_metadata.get("radio_checkpoint_sha256") != bank.metadata.get(
        "radio_checkpoint_sha256"
    ):
        raise ValueError("support graph and capability bank RADIO hashes differ")
    graph_signatures = capability_metadata.get("capability_signatures")
    if not isinstance(graph_signatures, Mapping):
        raise ValueError("support graph lacks source capability signatures")
    for name, signature in bank.signatures.items():
        if graph_signatures.get(name) != signature.to_dict():
            raise ValueError(
                f"support graph and capability bank {name} signatures differ"
            )
    return PrimitiveSupportGraph(
        edge_index=payload["edge_index"],
        edge_weight=torch.as_tensor(payload["edge_weight"]).float(),
        raw_affinity=torch.as_tensor(payload["raw_affinity"]).float(),
        local_sigma=payload["local_sigma"],
        num_nodes=int(global_rows.numel()),
    )
