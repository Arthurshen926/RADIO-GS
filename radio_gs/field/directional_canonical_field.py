"""Compact two-mode carrier layered over a canonical fallback field."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from radio_gs.utils.immutable_artifacts import load_torch_mapping

from .basis_decoder import AffineBasisDecoder
from .canonical_gaussian_field import CanonicalGaussianField
from .checkpoint import load_canonical_field_checkpoint


class DirectionalCanonicalField(nn.Module):
    """Return two RADIO directions when supported and one safe fallback otherwise."""

    def __init__(
        self,
        base_field: CanonicalGaussianField,
        prototype_decoder: AffineBasisDecoder,
        global_rows: torch.Tensor,
        prototype_coefficients: torch.Tensor,
        mixture_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_field = base_field
        self.prototype_decoder = prototype_decoder
        rows = torch.as_tensor(global_rows).long()
        coefficients = torch.as_tensor(prototype_coefficients).float()
        mixture = torch.as_tensor(mixture_weight).float()
        if rows.ndim != 1 or coefficients.shape[:2] != (rows.numel(), 2):
            raise ValueError("directional coefficients must be row-aligned [M,2,D]")
        if coefficients.shape[2] != prototype_decoder.coefficient_dim:
            raise ValueError("directional coefficient dimension differs from decoder")
        if mixture.shape != (rows.numel(), 2):
            raise ValueError("directional mixture must be [M,2]")
        if rows.numel() and (
            int(rows.min()) < 0
            or int(rows.max()) >= base_field.num_gaussians
            or not bool((rows[1:] > rows[:-1]).all())
        ):
            raise ValueError("directional global rows must be strictly ascending")
        inverse = torch.full((base_field.num_gaussians,), -1, dtype=torch.long)
        inverse[rows] = torch.arange(rows.numel())
        self.register_buffer("global_rows", rows)
        self.register_buffer("global_to_directional", inverse)
        self.register_buffer("prototype_coefficients", coefficients)
        self.register_buffer("mixture_weight", mixture)

    def _indices(self, indices: torch.Tensor | None) -> torch.Tensor:
        if indices is None:
            return torch.arange(
                self.base_field.num_gaussians,
                device=self.global_to_directional.device,
            )
        rows = torch.as_tensor(
            indices, device=self.global_to_directional.device, dtype=torch.long
        )
        if rows.ndim != 1:
            raise ValueError("directional field indices must be one-dimensional")
        return rows

    def radio_prototypes(
        self, indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self._indices(indices)
        center = self.base_field.radio_features(rows)
        modes = center[:, None, :].expand(-1, 2, -1).clone()
        weights = torch.zeros(rows.numel(), 2, device=center.device, dtype=center.dtype)
        weights[:, 0] = 1.0
        local = self.global_to_directional[rows]
        supported = local >= 0
        if bool(supported.any()):
            selected = local[supported]
            modes[supported] = self.prototype_decoder(
                self.prototype_coefficients[selected]
            ).to(modes)
            weights[supported] = self.mixture_weight[selected].to(weights)
        return modes, weights

    def prototype_query_logits(
        self,
        query_features: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Existential set similarity: a query may match either valid view mode."""

        modes, _weights = self.radio_prototypes(indices)
        queries = torch.as_tensor(
            query_features, device=modes.device, dtype=modes.dtype
        )
        if queries.ndim != 2 or queries.shape[1] != modes.shape[2]:
            raise ValueError("query features must be [queries,RADIO channels]")
        similarity = torch.einsum(
            "nkd,qd->nkq",
            F.normalize(modes, dim=-1, eps=1e-8),
            F.normalize(queries, dim=-1, eps=1e-8),
        )
        return similarity.amax(dim=1)


def load_directional_canonical_field(
    *,
    base_field_checkpoint: str | Path,
    expected_base_field_sha256: str,
    compact_prototype_cache: str | Path,
    expected_compact_prototype_sha256: str,
    joint_basis_checkpoint: str | Path,
    expected_joint_basis_sha256: str,
    map_location: str | torch.device = "cpu",
) -> tuple[DirectionalCanonicalField, Mapping[str, Any]]:
    base, base_payload = load_canonical_field_checkpoint(
        base_field_checkpoint,
        map_location="cpu",
        expected_sha256=expected_base_field_sha256,
    )
    compact, compact_sha, compact_path = load_torch_mapping(
        compact_prototype_cache,
        expected_sha256=expected_compact_prototype_sha256,
        map_location="cpu",
        label="compact directional prototype field",
    )
    joint, joint_sha, joint_path = load_torch_mapping(
        joint_basis_checkpoint,
        expected_sha256=expected_joint_basis_sha256,
        map_location="cpu",
        label="Field-D joint basis",
    )
    if compact.get("contract") != "compact_directional_prototype_field_v1":
        raise ValueError("compact directional field contract differs")
    if joint.get("contract") not in {
        "field_d_joint_center_directional_basis_v1",
        "field_d_gauge_preserving_joint_basis_v1",
    }:
        raise ValueError("joint directional basis contract differs")
    geometry = base_payload.get("geometry_fingerprint")
    if compact.get("geometry_fingerprint") != geometry or joint.get(
        "geometry_fingerprint"
    ) != geometry:
        raise ValueError("directional carrier geometry authorities differ")
    compact_metadata = dict(compact["metadata"])
    basis_authority = dict(compact_metadata.get("basis_authority", {}))
    if basis_authority != {"path": str(joint_path), "sha256": joint_sha}:
        raise ValueError("compact prototype cache belongs to another joint basis")
    for metadata in (compact_metadata, dict(joint["metadata"])):
        if any(
            metadata.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        ):
            raise ValueError("directional carrier is task contaminated")
    architecture = dict(joint["architecture"])
    state = dict(joint["decoder_state_dict"])
    decoder = AffineBasisDecoder(
        feature_dim=int(architecture["feature_dim"]),
        coefficient_dim=int(architecture["coefficient_dim"]),
        mean=state["mean"],
        scale=torch.as_tensor(state["log_scale"]).exp(),
        basis=state["basis"],
        trainable_basis=False,
    )
    field = DirectionalCanonicalField(
        base,
        decoder,
        compact["global_rows"],
        compact["coefficients"],
        compact["mixture_weight"],
    )
    target = torch.device(map_location)
    if target.type != "cpu":
        field = field.to(target)
    provenance = {
        "base_field_checkpoint": {
            "path": str(Path(base_field_checkpoint).expanduser().resolve()),
            "sha256": expected_base_field_sha256,
        },
        "compact_prototype_cache": {
            "path": str(compact_path),
            "sha256": compact_sha,
        },
        "joint_basis_checkpoint": {"path": str(joint_path), "sha256": joint_sha},
        "query_pooling": "maximum_cosine_over_two_training_view_modes",
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    return field, provenance
