from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.build_nvos_source_heldout_view_reliability import (
    select_mapping_views,
    supported_binary_iou,
)


def test_supported_binary_iou_excludes_untransported_pixels_and_neutral_values():
    prediction = torch.tensor([[0.9, 0.5, 0.9], [0.1, 0.9, 0.1]])
    target = torch.tensor([[0.8, 0.9, 0.1], [0.1, 0.9, 0.9]])
    supported = torch.tensor([[True, True, True], [True, True, False]])
    assert supported_binary_iou(prediction, target, supported) == pytest.approx(0.5)


def test_mapping_selection_requires_geometry_and_uses_appearance_only_as_tie_breaker():
    roles = ["prompt", "registered_mapping", "registered_mapping", "evaluation"]
    geometry = torch.tensor([1.0, 0.49, 0.7, 1.0])
    appearance = torch.tensor([1.0, 0.999, 0.2, 1.0])
    selected = select_mapping_views(
        roles,
        geometry,
        appearance,
        minimum_heldout_iou=0.5,
        top_k=2,
    )
    assert selected == (0, 2, 3)


def test_mapping_selection_is_fixed_cap_and_deterministic():
    roles = ["prompt", "registered_mapping", "registered_mapping", "evaluation"]
    selected = select_mapping_views(
        roles,
        torch.tensor([1.0, 0.8, 0.8, 1.0]),
        torch.tensor([1.0, 0.3, 0.9, 1.0]),
        minimum_heldout_iou=0.5,
        top_k=1,
    )
    assert selected == (0, 2, 3)


def test_prompt_observation_can_abstain_while_evaluation_remains_mandatory():
    roles = ["prompt", "registered_mapping", "evaluation"]
    selected = select_mapping_views(
        roles,
        torch.tensor([0.49, 0.8, 1.0]),
        torch.tensor([1.0, 0.9, 1.0]),
        minimum_heldout_iou=0.5,
        top_k=1,
    )
    assert selected == (1, 2)
