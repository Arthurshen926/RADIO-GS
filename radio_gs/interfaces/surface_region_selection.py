"""Version-aware surface-region loading and normalized selection tensors.

The V2 contract returns a ``(rows, core_mask, distance)`` tuple, whereas V3/V4
return a structured expansion with an explicit support-fill tier.  This
module is the narrow compatibility boundary between those immutable contract
APIs and cache/readout callers.  It does not change either expansion policy.

``token_mask`` is the sole tensor-padding authority.  Core, context, and
support-fill are mutually exclusive selected-token roles whose union is
exactly ``token_mask``.  In particular, a V3 support-fill token is a real
selected token (``token_mask=True``), never tensor padding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

import torch

from .surface_region_contract import (
    SurfaceRegionContractV2,
    SurfaceRegionContractV3,
    SurfaceRegionContractV4,
    SurfaceRegionExpansionV3,
    SurfaceRegionExpansionV4,
)


SurfaceRegionContract = Union[
    SurfaceRegionContractV2, SurfaceRegionContractV3, SurfaceRegionContractV4,
]
SurfaceRegionV2Result = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def surface_region_contract_from_specification(
    specification: Mapping[str, object],
) -> SurfaceRegionContract:
    """Construct exactly the contract version declared by a manifest mapping.

    Legacy V2 manifests may omit fields whose dataclass defaults preserve the
    frozen V2 digest.  Unknown versions and malformed/extra fields fail closed.
    """

    if not isinstance(specification, Mapping):
        raise ValueError("surface-region contract specification must be a mapping")
    values = dict(specification)
    version = values.get("version")
    radii = values.get("radii_m")
    if not isinstance(version, str):
        raise ValueError("surface-region contract specification lacks version")
    if not isinstance(radii, (list, tuple)):
        raise ValueError("surface-region contract specification has invalid radii_m")
    values["radii_m"] = tuple(radii)
    contract_type: (
        type[SurfaceRegionContractV2]
        | type[SurfaceRegionContractV3]
        | type[SurfaceRegionContractV4]
    )
    if version == "surface-region-contract-v2":
        contract_type = SurfaceRegionContractV2
        # V2 deliberately omits this field when it equals maximum_tokens.
        # Reconstruct that manifest-level default instead of the dataclass's
        # historical 256-token constructor default (teacher contracts commonly
        # use a larger equal maximum/candidate limit).
        if "token_candidate_limit" not in values and "maximum_tokens" in values:
            values["token_candidate_limit"] = values["maximum_tokens"]
    elif version == "surface-region-contract-v3":
        contract_type = SurfaceRegionContractV3
    elif version == "surface-region-contract-v4":
        contract_type = SurfaceRegionContractV4
    else:
        raise ValueError(f"unsupported surface-region contract version: {version!r}")
    try:
        return contract_type(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("surface-region contract specification is malformed") from error


def surface_region_contract_from_metadata(
    metadata: Mapping[str, object],
    *,
    contract_key: str = "region_contract",
    version_key: str = "region_contract_version",
    digest_key: str = "region_contract_sha256",
) -> SurfaceRegionContract:
    """Load a contract and verify its separately stored version/digest binding."""

    if not isinstance(metadata, Mapping):
        raise ValueError("surface-region metadata must be a mapping")
    raw = metadata.get(contract_key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"surface-region metadata lacks {contract_key}")
    contract = surface_region_contract_from_specification(raw)
    if metadata.get(version_key) != contract.version:
        raise ValueError("surface-region metadata version binding differs")
    if metadata.get(digest_key) != contract.digest:
        raise ValueError("surface-region metadata digest binding differs")
    return contract


@dataclass(frozen=True)
class RegionSelection:
    """Normalized V2/V3/V4 selection, optionally padded to a fixed width.

    Active token roles have these exact meanings:

    - ``core_mask``: semantic members at or inside the requested physical radius.
    - ``context_mask``: semantic members in the declared context shell.
    - ``support_fill_mask``: V3/V4 recovery support outside semantic membership.
    - ``token_mask``: every real selected token; false entries are tensor padding.

    Tensor-padding rows are canonicalized to row index zero, false role masks,
    and infinite distances.  Callers must independently zero token features,
    geometry, and reliability wherever ``token_mask`` is false.
    """

    rows: torch.Tensor
    core_mask: torch.Tensor
    context_mask: torch.Tensor
    support_fill_mask: torch.Tensor
    token_mask: torch.Tensor
    semantic_geodesic_distance: torch.Tensor
    recovery_distance: torch.Tensor
    anchor_index: int
    anchor_row: int
    contract_version: str

    def __post_init__(self) -> None:
        rows = torch.as_tensor(self.rows).detach().long().cpu().reshape(-1).clone()
        core = torch.as_tensor(self.core_mask).detach().bool().cpu().reshape(-1).clone()
        context = (
            torch.as_tensor(self.context_mask).detach().bool().cpu().reshape(-1).clone()
        )
        support_fill = (
            torch.as_tensor(self.support_fill_mask)
            .detach().bool().cpu().reshape(-1).clone()
        )
        token = torch.as_tensor(self.token_mask).detach().bool().cpu().reshape(-1).clone()
        semantic_distance = (
            torch.as_tensor(self.semantic_geodesic_distance)
            .detach().float().cpu().reshape(-1).clone()
        )
        recovery_distance = (
            torch.as_tensor(self.recovery_distance)
            .detach().float().cpu().reshape(-1).clone()
        )
        width = rows.numel()
        aligned = (core, context, support_fill, token, semantic_distance, recovery_distance)
        if width == 0 or any(value.numel() != width for value in aligned):
            raise ValueError("region-selection tensors must be non-empty and aligned")
        selected_count = int(token.sum())
        expected_token = torch.arange(width) < selected_count
        if selected_count == 0 or not torch.equal(token, expected_token):
            raise ValueError("region-selection token_mask must be a non-empty left prefix")
        roles = core.to(torch.int8) + context.to(torch.int8) + support_fill.to(torch.int8)
        if not torch.equal(roles, token.to(torch.int8)):
            raise ValueError(
                "region-selection core/context/support-fill roles must partition token_mask"
            )
        active_rows = rows[token]
        if int(torch.unique(active_rows).numel()) != selected_count:
            raise ValueError("region-selection active rows must be unique")
        if bool((rows[~token] != 0).any()):
            raise ValueError("region-selection tensor-padding rows must be zero")
        semantic = core | context
        if not bool(torch.isfinite(semantic_distance[semantic]).all()):
            raise ValueError("semantic region members need finite geodesic distances")
        if bool((semantic_distance[semantic] < 0).any()):
            raise ValueError("semantic geodesic distances cannot be negative")
        if not bool(torch.isinf(semantic_distance[support_fill | ~token]).all()):
            raise ValueError("support-fill and tensor padding need infinite semantic distance")
        if not bool(torch.isfinite(recovery_distance[support_fill]).all()) or bool(
            (recovery_distance[support_fill] < 0).any()
        ):
            raise ValueError("support-fill tokens need finite non-negative recovery distance")
        if not bool(torch.isinf(recovery_distance[semantic | ~token]).all()):
            raise ValueError("semantic members and tensor padding need infinite recovery distance")
        anchor_index = int(self.anchor_index)
        anchor_row = int(self.anchor_row)
        if (
            anchor_index < 0
            or anchor_index >= selected_count
            or not bool(core[anchor_index])
            or int(rows[anchor_index]) != anchor_row
            or float(semantic_distance[anchor_index]) != 0.0
        ):
            raise ValueError("region-selection anchor must be a zero-distance core token")
        if self.contract_version not in {
            "surface-region-contract-v2",
            "surface-region-contract-v3",
            "surface-region-contract-v4",
        }:
            raise ValueError("region-selection contract version is unsupported")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "context_mask", context)
        object.__setattr__(self, "support_fill_mask", support_fill)
        object.__setattr__(self, "token_mask", token)
        object.__setattr__(self, "semantic_geodesic_distance", semantic_distance)
        object.__setattr__(self, "recovery_distance", recovery_distance)
        object.__setattr__(self, "anchor_index", anchor_index)
        object.__setattr__(self, "anchor_row", anchor_row)

    @property
    def selected_count(self) -> int:
        return int(self.token_mask.sum())

    @property
    def width(self) -> int:
        return int(self.rows.numel())

    def pad_to(self, width: int) -> "RegionSelection":
        """Return a left-aligned, canonically padded representation."""

        target = int(width)
        if target < self.selected_count:
            raise ValueError("region-selection padding width truncates selected tokens")
        if target == self.width:
            return self
        if self.width != self.selected_count:
            raise ValueError("an already padded region selection cannot be repadded")
        padding = target - self.selected_count
        return RegionSelection(
            rows=torch.cat((self.rows, torch.zeros(padding, dtype=torch.long))),
            core_mask=torch.cat((self.core_mask, torch.zeros(padding, dtype=torch.bool))),
            context_mask=torch.cat(
                (self.context_mask, torch.zeros(padding, dtype=torch.bool))
            ),
            support_fill_mask=torch.cat(
                (self.support_fill_mask, torch.zeros(padding, dtype=torch.bool))
            ),
            token_mask=torch.cat(
                (self.token_mask, torch.zeros(padding, dtype=torch.bool))
            ),
            semantic_geodesic_distance=torch.cat(
                (self.semantic_geodesic_distance, torch.full((padding,), torch.inf))
            ),
            recovery_distance=torch.cat(
                (self.recovery_distance, torch.full((padding,), torch.inf))
            ),
            anchor_index=self.anchor_index,
            anchor_row=self.anchor_row,
            contract_version=self.contract_version,
        )

    @classmethod
    def from_v2(
        cls,
        result: SurfaceRegionV2Result,
        *,
        anchor_row: int,
    ) -> "RegionSelection":
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("V2 region result must be a three-tensor tuple")
        rows, core, distance = result
        rows = torch.as_tensor(rows).detach().long().cpu().reshape(-1)
        core = torch.as_tensor(core).detach().bool().cpu().reshape(-1)
        distance = torch.as_tensor(distance).detach().float().cpu().reshape(-1)
        positions = torch.where(rows == int(anchor_row))[0]
        if positions.numel() != 1:
            raise ValueError("V2 region result must contain its anchor exactly once")
        count = rows.numel()
        return cls(
            rows=rows,
            core_mask=core,
            context_mask=~core,
            support_fill_mask=torch.zeros(count, dtype=torch.bool),
            token_mask=torch.ones(count, dtype=torch.bool),
            semantic_geodesic_distance=distance,
            recovery_distance=torch.full((count,), torch.inf),
            anchor_index=int(positions[0]),
            anchor_row=int(anchor_row),
            contract_version="surface-region-contract-v2",
        )

    @classmethod
    def from_v3(cls, result: SurfaceRegionExpansionV3) -> "RegionSelection":
        if not isinstance(result, SurfaceRegionExpansionV3):
            raise ValueError("V3 region result must be SurfaceRegionExpansionV3")
        count = result.rows.numel()
        anchor_row = int(result.rows[result.anchor_index])
        return cls(
            rows=result.rows,
            core_mask=result.core_mask,
            context_mask=result.context_mask,
            support_fill_mask=result.support_fill_mask,
            token_mask=torch.ones(count, dtype=torch.bool),
            semantic_geodesic_distance=result.semantic_geodesic_distance,
            recovery_distance=result.recovery_distance,
            anchor_index=result.anchor_index,
            anchor_row=anchor_row,
            contract_version="surface-region-contract-v3",
        )

    @classmethod
    def from_v4(cls, result: SurfaceRegionExpansionV4) -> "RegionSelection":
        if not isinstance(result, SurfaceRegionExpansionV4):
            raise ValueError("V4 region result must be SurfaceRegionExpansionV4")
        count = result.rows.numel()
        anchor_row = int(result.rows[result.anchor_index])
        return cls(
            rows=result.rows,
            core_mask=result.core_mask,
            context_mask=result.context_mask,
            support_fill_mask=result.support_fill_mask,
            token_mask=torch.ones(count, dtype=torch.bool),
            semantic_geodesic_distance=result.semantic_geodesic_distance,
            recovery_distance=result.recovery_distance,
            anchor_index=result.anchor_index,
            anchor_row=anchor_row,
            contract_version="surface-region-contract-v4",
        )


def as_region_selection(
    result: SurfaceRegionV2Result | SurfaceRegionExpansionV3 | SurfaceRegionExpansionV4,
    *,
    anchor_row: int | None = None,
) -> RegionSelection:
    """Normalize one immutable V2, V3, or V4 expansion result."""

    if isinstance(result, SurfaceRegionExpansionV4):
        if anchor_row is not None and int(
            result.rows[result.anchor_index]
        ) != int(anchor_row):
            raise ValueError("declared anchor_row differs from V4 expansion")
        return RegionSelection.from_v4(result)
    if isinstance(result, SurfaceRegionExpansionV3):
        if anchor_row is not None and int(
            result.rows[result.anchor_index]
        ) != int(anchor_row):
            raise ValueError("declared anchor_row differs from V3 expansion")
        return RegionSelection.from_v3(result)
    if anchor_row is None:
        raise ValueError("V2 region adaptation requires anchor_row")
    return RegionSelection.from_v2(result, anchor_row=int(anchor_row))


__all__ = [
    "RegionSelection",
    "SurfaceRegionContract",
    "SurfaceRegionV2Result",
    "as_region_selection",
    "surface_region_contract_from_metadata",
    "surface_region_contract_from_specification",
]
