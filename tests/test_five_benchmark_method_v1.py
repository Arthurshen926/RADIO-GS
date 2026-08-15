from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from radio_gs.five_benchmark_method_v1 import (
    MethodV1ValidationError,
    validate_complete_field_payload,
    validate_method_authority,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "paper/artifacts/five_benchmark_method_v1_authority_20260815.json"
INVENTORY = (
    ROOT / "paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json"
)


def _complete_payload() -> dict:
    return {
        "schema_version": 2,
        "checkpoint_contract": "canonical-factorized-radio-checkpoint-v1",
        "architecture": {
            "feature_dim": 1280,
            "coefficient_dim": 512,
            "local_dim": 512,
            "fusion_reliability": False,
        },
        "reliability": torch.empty(4, 0),
        "render_optimization": {
            "train_basis": False,
            "train_fusion": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "text_queries_opened": False,
            "official_render_capability": {
                "enabled": True,
                "adaptor_weights": {"siglip2-g": 0.05},
                "projection_order": "complete_rendered_2d_grid_vs_resample(official_runtime_adaptor_output)",
                "custom_adaptor_head": False,
            },
            "semantic_capability": {
                "enabled": True,
                "weight": 0.05,
                "uses_benchmark_masks": False,
                "uses_text_queries": False,
            },
            "generic_text_response": {
                "enabled": True,
                "weight": 0.05,
                "components": ["profile", "listwise", "sibling", "synonym"],
                "benchmark_text_queries_opened": False,
                "uses_benchmark_masks": False,
                "uses_target_metrics_for_selection": False,
            },
        },
    }


def test_checked_in_authority_is_method_v1() -> None:
    validate_method_authority(json.loads(AUTHORITY.read_text(encoding="utf-8")))


def test_inventory_is_exhaustive_and_cannot_claim_a_joint_row() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    counts = inventory["field_instance_accounting"]
    assert counts["total_required"] == 29
    assert (
        counts["method_v1_complete"]
        + counts["base_d512_l512_only"]
        + counts["base_d512_l512_materializing"]
        + counts["legacy_or_missing"]
        == counts["total_required"]
    )
    assert inventory["joint_row_eligible"] is False
    assert len(inventory["lerf_shared_four"]) == 4
    assert len(inventory["scannet_ovs_paper8"]["required_scenes"]) == 8
    assert len(inventory["nvos_full8"]["required_tasks"]) == 8
    assert len(inventory["spin_nerf_available9"]["required_scenes"]) == 9


def test_complete_field_passes() -> None:
    assert validate_complete_field_payload(_complete_payload())[
        "construction_stages_complete"
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["architecture"].update(local_dim=128), "local_dim"),
        (
            lambda value: value["render_optimization"].pop("generic_text_response"),
            "generic response stage",
        ),
        (
            lambda value: value["render_optimization"].update(train_basis=True),
            "basis was not frozen",
        ),
    ],
)
def test_incomplete_or_heterogeneous_field_fails(mutation, message: str) -> None:
    payload = _complete_payload()
    mutation(payload)
    with pytest.raises(MethodV1ValidationError, match=message):
        validate_complete_field_payload(payload)
