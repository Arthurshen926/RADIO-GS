#!/usr/bin/env python3
"""Source-only objective adapter for response/listwise V2.1.

This module deliberately contains no CLI, optimizer, benchmark reader,
evaluator, or renderer.  It exposes the complete-scene objective needed by a
future separately authorized pilot while keeping the frozen V2 trainer and
loss byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from radio_gs.losses.source_global_response_listwise_loss import (
    FrozenSourceResponseAuthority,
)
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    FrozenCanonicalNegativeBank,
    FrozenCompositionalGenericBank,
    recommended_v21_config,
    source_global_response_listwise_loss_v21,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
)
from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_residual as v1_trainer,
)


FROZEN_V2_LOSS_SHA256 = (
    "552e7bf0e4d83e9346af731e6ce9eaf891968b14f32b49f728b5188c5e012ae7"
)
FROZEN_V2_TRAINER_SHA256 = (
    "56dddf57d03a50f933e722c653941ca9d02c39ec7b6ebd6ba78340b7a894b370"
)


def source_access() -> dict[str, bool]:
    return {
        **v1_trainer.typed_context_training_source_access(),
        "generic_target_blind_text_bank_opened": True,
        "canonical_generic_negative_bank_opened": True,
        "benchmark_text_queries_opened": False,
    }


def integration_contract() -> dict[str, Any]:
    """Describe the non-executing V2.1 trainer integration point."""

    return {
        "schema_version": 1,
        "artifact_type": "surface_region_typed_context_response_listwise_v21_adapter",
        "complete_canonical_scene_forward": True,
        "frozen_v2_loss_sha256": FROZEN_V2_LOSS_SHA256,
        "frozen_v2_trainer_sha256": FROZEN_V2_TRAINER_SHA256,
        "model_class": "SurfaceRegionAcceptedV2TypedContextResidualV1",
        "new_learnable_parameters": False,
        "optimizer_constructed_by_adapter": False,
        "benchmark_execution_supported": False,
        "typed_relation_authority_supported": True,
        "typed_relation_runtime_query_strings_consumed": False,
        "source_access": source_access(),
    }


def complete_scene_objective_v21(
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
    exclude_both_immutable_pairs: bool = False,
    routing_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | bool]]:
    """Forward one complete source scene and apply the V1 + V2.1 objective."""

    row_count = len(scene["region_row_ids"])
    if routing_masks is None:
        declared, effective_ood, active = v1_trainer._routing(scene, normalization)
    else:
        declared, effective_ood, active = (
            torch.as_tensor(value).detach().bool().cpu() for value in routing_masks
        )
        if any(
            value.shape != (row_count,) for value in (declared, effective_ood, active)
        ):
            raise ValueError("V2.1 externally supplied routing masks differ")
        if not torch.equal(active, declared & ~effective_ood):
            raise ValueError("V2.1 externally supplied routing masks are inconsistent")
    if active.shape != (row_count,) or int(active.sum()) < 2:
        raise ValueError("complete source scene requires at least two active rows")
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
    total, response_metrics = source_global_response_listwise_loss_v21(
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
        teacher_pair_descriptors_sha256=(authority.teacher_pair_descriptors_sha256),
        fit_text_bank_file_sha256=authority.fit_text_bank_file_sha256,
        compositional_banks=compositional_banks,
        relation_authority=relation_authority,
        trainable_region_mask=active,
        exclude_both_immutable_pairs=exclude_both_immutable_pairs,
        config=recommended_v21_config(),
    )
    if not bool(torch.isfinite(total.detach())):
        raise RuntimeError("complete-scene V2.1 objective is nonfinite")
    return total, {
        "base_objective": base_loss,
        **{f"base_{name}": value for name, value in base_metrics.items()},
        **{f"response_{name}": value for name, value in response_metrics.items()},
        "complete_canonical_rows": row_count,
        "active_rows": int(active.sum()),
        "immutable_rows": int((~active).sum()),
        "training_pair_denominator_filter_active": bool(exclude_both_immutable_pairs),
        "fallback_bitwise_accepted_v2_e0": fallback_equal,
    }


__all__ = [
    "complete_scene_objective_v21",
    "integration_contract",
    "source_access",
]
