"""Fail-closed representation and authority contracts for SUGM-v3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

import torch


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StructuredMemoryContract:
    name: str = "structured_universal_gaussian_memory_v3"
    latent_dim: int = 512
    reliability_dim: int = 5
    shared_dim: int = 320
    semantic_dim: int = 128
    instance_dim: int = 48
    boundary_dim: int = 16
    capability_preservation_gate: str = "source_capability_pareto"
    raw_radio_fidelity_role: str = "diagnostic_soft_regularizer"
    partition_owned_writes: bool = True
    gaussian_indexed_high_dimensional_sidecars: int = 0
    target_rgb_allowed_in_strict_lerf: bool = False
    shared_2d_3d_posterior: bool = True

    def canonical_dict(self) -> dict[str, object]:
        return dict(self.__dict__)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


SUGM_V3_CONTRACT = StructuredMemoryContract()


def validate_scene_state(
    latent: torch.Tensor,
    reliability: torch.Tensor,
    *,
    source_authority_sha256: str,
) -> None:
    """Reject state that is not exactly one D512 plus five scalars per row."""

    z = torch.as_tensor(latent)
    r = torch.as_tensor(reliability)
    if z.ndim != 2 or z.shape[1] != SUGM_V3_CONTRACT.latent_dim:
        raise ValueError("SUGM-v3 requires exactly one D512 latent per Gaussian")
    if r.shape != (z.shape[0], SUGM_V3_CONTRACT.reliability_dim):
        raise ValueError("SUGM-v3 requires exactly five reliability scalars per Gaussian")
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(r).all()):
        raise ValueError("SUGM-v3 scene state must be finite")
    if _SHA256.fullmatch(str(source_authority_sha256)) is None:
        raise ValueError("SUGM-v3 checkpoint requires a source authority SHA-256")


def validate_frozen_method_receipt(receipt: Mapping[str, object]) -> None:
    required = {
        "schema",
        "contract_sha256",
        "source_authority_sha256",
        "checkpoint_sha256",
        "configuration_sha256",
        "selection_authority",
        "target_rgb_opened",
    }
    if set(receipt) != required:
        raise ValueError("frozen method receipt fields differ")
    for name in ("contract_sha256", "source_authority_sha256", "checkpoint_sha256", "configuration_sha256"):
        if _SHA256.fullmatch(str(receipt[name])) is None:
            raise ValueError(f"{name} must be a SHA-256 digest")
    if receipt["contract_sha256"] != SUGM_V3_CONTRACT.sha256:
        raise ValueError("frozen receipt uses another method contract")
    if receipt["selection_authority"] != "source_train_dev_audit_and_sentinel_only":
        raise ValueError("benchmark metrics may not select the v3 method")


__all__ = [
    "SUGM_V3_CONTRACT",
    "StructuredMemoryContract",
    "validate_frozen_method_receipt",
    "validate_scene_state",
]
