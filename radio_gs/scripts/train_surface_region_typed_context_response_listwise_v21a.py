#!/usr/bin/env python3
"""Complete-scene objective adapter for the source-only V2.1A rescue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from radio_gs.losses.source_global_response_listwise_loss import (
    FrozenSourceResponseAuthority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    FrozenCanonicalNegativeBank,
    FrozenCompositionalGenericBank,
    recommended_v21_config,
)
from radio_gs.losses.source_global_response_listwise_loss_v21a import (
    source_global_response_listwise_loss_v21a,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import train_surface_region_typed_context_residual as v1_trainer


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("diagnostic quantile requires a nonempty vector")
    ordered = values.sort().values
    index = int(math.floor(float(q) * max(0, values.numel() - 1)))
    return ordered[index]


def complete_scene_objective_v21a(
    model: SurfaceRegionAcceptedV2TypedContextResidualV1,
    scene: Mapping[str, Any],
    normalization: Mapping[str, Any],
    fit_text_embeddings: torch.Tensor,
    authority: FrozenSourceResponseAuthority,
    canonical_negative_bank: FrozenCanonicalNegativeBank,
    device: torch.device,
    *,
    compositional_banks: Sequence[FrozenCompositionalGenericBank] = (),
    relation_authority: FrozenTypedTextRelationAuthority | None = None,
    training: bool = False,
    routing_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
    """Forward one full scene and expose routing/update diagnostics."""

    row_count = len(scene["region_row_ids"])
    if routing_masks is None:
        declared, effective_ood, active = v1_trainer._routing(scene, normalization)
    else:
        declared, effective_ood, active = (
            torch.as_tensor(value).detach().bool().cpu() for value in routing_masks
        )
    if any(value.shape != (row_count,) for value in (declared, effective_ood, active)):
        raise ValueError("V2.1A routing masks differ")
    if not torch.equal(active, declared & ~effective_ood) or int(active.sum()) < 2:
        raise ValueError("V2.1A routing masks are inconsistent or too sparse")
    eligible = torch.as_tensor(scene["eligible"]).detach().bool().cpu()
    eligible_rows = int(eligible.sum())
    if eligible.shape != (row_count,) or eligible_rows <= 0:
        raise ValueError("V2.1A scene requires eligible canonical rows")

    base = scene["accepted_v2_e0"].to(device)
    result = model.forward_with_diagnostics(
        base,
        scene["pooled_context_radio_direction"].to(device),
        scene["raw_full_scalar_summary"].to(device),
        scene["typed_context_statistics"].to(device),
        active_mask=declared.to(device),
        ood_mask=effective_ood.to(device),
    )
    fallback = ~active.to(device)
    fallback_equal = torch.equal(result.semantic_descriptor[fallback], base[fallback])
    if not fallback_equal:
        raise RuntimeError("inactive/OOD rows are not bitwise AcceptedV2 e0")
    active_rows = torch.where(active)[0]
    teachers, teacher_mask = v1_trainer.gather_sparse_teacher_batch(scene, active_rows)
    base_loss, base_metrics = v1_trainer.typed_context_objective(
        result.semantic_descriptor[active.to(device)],
        teachers.to(device),
        teacher_mask.to(device),
        scene["typed_context_statistics"][active_rows].to(device),
        boundary_threshold=float(normalization["source_boundary_score_median"]),
    )
    total, response_metrics = source_global_response_listwise_loss_v21a(
        base_loss,
        result.semantic_descriptor,
        scene["official_multiview_siglip2_teacher_pair_descriptors"].to(device),
        scene["official_multiview_siglip2_teacher_pair_region_indices"],
        fit_text_embeddings,
        authority.payload["canonical_region_indices"],
        authority,
        canonical_negative_bank,
        accepted_v2_file_sha256=authority.accepted_v2_file_sha256,
        teacher_file_sha256=authority.teacher_file_sha256,
        teacher_pair_descriptors_sha256=authority.teacher_pair_descriptors_sha256,
        fit_text_bank_file_sha256=authority.fit_text_bank_file_sha256,
        compositional_banks=compositional_banks,
        relation_authority=relation_authority,
        trainable_region_mask=active,
        training=training,
        config=recommended_v21_config(),
    )
    selected_base = result.base_descriptor[active.to(device)]
    selected_candidate = result.semantic_descriptor[active.to(device)]
    cosine = (selected_base * selected_candidate).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    cap = float(model.max_angle_radians)
    angle_cap_fraction = (angle >= cap - 1e-6).float().mean()
    active_eligible_rows = int((active & eligible).sum())
    return total, {
        "base_objective": base_loss,
        **{f"base_{name}": value for name, value in base_metrics.items()},
        **{f"response_{name}": value for name, value in response_metrics.items()},
        "complete_canonical_rows": row_count,
        "eligible_rows": eligible_rows,
        "declared_rows": int(declared.sum()),
        "active_rows": int(active.sum()),
        "active_eligible_rows": active_eligible_rows,
        "active_over_eligible_coverage": result.semantic_descriptor.new_tensor(
            active_eligible_rows / eligible_rows
        ),
        "immutable_rows": int((~active).sum()),
        "angle_radians_p50": _quantile(angle.detach(), 0.50),
        "angle_radians_p95": _quantile(angle.detach(), 0.95),
        "angle_cap_fraction": angle_cap_fraction.detach(),
        "training_pair_denominator_filter_active": bool(training),
        "fallback_bitwise_accepted_v2_e0": fallback_equal,
    }


__all__ = ["complete_scene_objective_v21a"]
