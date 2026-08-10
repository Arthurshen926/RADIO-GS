from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_v21b_reliability_conditioned_residual import (
    SURFACE_REGION_V21B_INTERFACE_CONTRACT_SHA256,
    SURFACE_REGION_V21B_PREREGISTRATION_SHA256,
    build_model_from_source_normalization,
    forward_complete_scene,
    interface_contract,
    source_access,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _scene(scene_id: str) -> dict[str, object]:
    generator = torch.Generator().manual_seed(17)
    base = F.normalize(torch.randn(4, 1536, generator=generator), dim=-1)
    context = torch.zeros(4, 1280)
    context[[0, 2, 3]] = F.normalize(
        torch.randn(3, 1280, generator=generator), dim=-1
    )
    full_scalar = torch.zeros(4, 18)
    statistics = torch.zeros(4, 12)
    statistics[[0, 2, 3], 3] = 0.8
    statistics[[0, 2, 3], 7] = 0.9
    return {
        "scene_id": scene_id,
        "accepted_v2_e0": base,
        "pooled_context_radio_direction": context,
        "raw_full_scalar_summary": full_scalar,
        "typed_context_statistics": statistics,
    }


def test_preregistration_precedes_implementation_and_contract_is_stable() -> None:
    path = Path(
        "paper/artifacts/"
        "surface_region_v21b_reliability_conditioned_rank256_"
        "preregistration_20260807.json"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SURFACE_REGION_V21B_PREREGISTRATION_SHA256
    assert canonical_json_sha256(interface_contract()) == (
        SURFACE_REGION_V21B_INTERFACE_CONTRACT_SHA256
    )
    access = source_access()
    assert access["query_independent"] is True
    assert access["scene_identifiers_consumed_by_model"] is False
    assert access["per_scene_hyperparameters"] is False
    assert not access["benchmark_queries_opened"]
    assert not access["target_heldout_opened"]


def test_complete_scene_hook_uses_source_normalization_and_ignores_scene_id() -> None:
    normalization = {
        "median": torch.zeros(30),
        "robust_scale": torch.ones(30),
    }
    model = build_model_from_source_normalization(normalization)
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-0.2, 0.2, 1536))
    declared = torch.tensor([True, False, True, True])
    ood = torch.tensor([False, False, True, False])
    first = forward_complete_scene(
        model,
        _scene("scene0001_00"),
        declared_active_mask=declared,
        effective_ood_mask=ood,
        device=torch.device("cpu"),
    )
    second = forward_complete_scene(
        model,
        _scene("completely_different_scene_identifier"),
        declared_active_mask=declared,
        effective_ood_mask=ood,
        device=torch.device("cpu"),
    )
    assert torch.equal(first.semantic_descriptor, second.semantic_descriptor)
    fallback = ~(declared & ~ood)
    assert torch.equal(
        first.semantic_descriptor[fallback],
        first.base_descriptor[fallback],
    )
    contract = interface_contract()
    assert contract["future_trainer_integration"]["complete_scene_forward"] == (
        "forward_complete_scene"
    )
    assert contract["future_trainer_integration"]["losses"].startswith(
        "reuse_frozen_v21"
    )


def test_complete_scene_hook_rejects_non_boolean_routing() -> None:
    model = build_model_from_source_normalization(
        {"median": torch.zeros(30), "robust_scale": torch.ones(30)}
    )
    with pytest.raises(ValueError, match="routing masks"):
        forward_complete_scene(
            model,
            _scene("scene0001_00"),
            declared_active_mask=torch.ones(4),
            effective_ood_mask=torch.zeros(4, dtype=torch.bool),
            device=torch.device("cpu"),
        )
