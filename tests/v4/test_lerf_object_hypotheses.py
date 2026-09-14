from __future__ import annotations

import pytest
import torch

from radio_gs.v4.contracts.build_lerf_object_hypotheses import (
    SCHEMA,
    _select_hypothesis_prototypes,
    validate_hypothesis_memory,
)


def _payload():
    return {
        "schema": SCHEMA,
        "fragment_count": 2,
        "surface_element_count": 3,
        "fragment_assignment": torch.tensor([[0.7, 0.2], [0.0, 0.8]]),
        "fragment_null_probability": torch.tensor([0.1, 0.2]),
        "observed_membership": torch.tensor([[0.8, 0.0], [0.1, 0.7], [0.0, 0.2]]),
        "hypothesis_root_fragments": [[0], [1]],
        "information_policy": {
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_rgb_opened": False,
        },
    }


def test_hypothesis_memory_validates_soft_top2_simplex():
    result = validate_hypothesis_memory(_payload())
    assert result["fragment_assignment"].shape == (2, 2)


def test_hypothesis_memory_rejects_missing_probability_mass():
    payload = _payload()
    payload["fragment_null_probability"][0] = 0.5
    with pytest.raises(ValueError, match="simplex"):
        validate_hypothesis_memory(payload)


def test_hypothesis_memory_accepts_monotone_completed_membership():
    payload = _payload()
    payload["completed_membership"] = payload["observed_membership"].clone()
    payload["completed_membership"][2, 0] = 0.4
    result = validate_hypothesis_memory(payload)
    assert result["completed_membership"][2, 0] == pytest.approx(0.4)


def test_hypothesis_memory_rejects_completion_that_erases_observation():
    payload = _payload()
    payload["completed_membership"] = torch.zeros_like(payload["observed_membership"])
    with pytest.raises(ValueError, match="must not erase"):
        validate_hypothesis_memory(payload)


def test_hypothesis_prototypes_keep_multiple_source_regions():
    appearance = torch.arange(4 * 2 * 3).reshape(4, 2, 3).float()
    selected = _select_hypothesis_prototypes(
        appearance,
        torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.0, 0.7], [0.0, 0.6]]),
        torch.ones(4),
        maximum_prototypes=4,
    )
    assert selected["object_prototype_descriptors"].shape == (2, 4, 3)
    assert selected["object_prototype_fragment_ids"][0].tolist() == [0, 0, 1, 1]
    assert selected["object_prototype_kind"][0].tolist() == [0, 1, 0, 1]


def test_prototype_capacity_prioritizes_distinct_source_views():
    selected = _select_hypothesis_prototypes(
        torch.ones(3, 2, 4), torch.tensor([[.9], [.8], [.7]]),
        torch.ones(3), maximum_prototypes=4,
        fragment_view_index=torch.tensor([5, 5, 6]),
    )
    assert selected["object_prototype_fragment_ids"].tolist() == [[0, 0, 2, 2]]
    assert selected["object_prototype_view_ids"].tolist() == [[5, 5, 6, 6]]


@pytest.mark.parametrize("corruption", ["nan", "zero", "out_of_bounds", "conflicting_view"])
def test_prototype_receipts_reject_corruption(corruption):
    payload = _payload()
    payload.update(_select_hypothesis_prototypes(
        torch.ones(2, 2, 4), payload["fragment_assignment"], torch.ones(2),
        maximum_prototypes=4, fragment_view_index=torch.tensor([0, 1]),
    ))
    if corruption == "nan":
        payload["object_prototype_descriptors"][0, 0, 0] = float("nan")
    elif corruption == "zero":
        payload["object_prototype_descriptors"][0, 0] = 0
    elif corruption == "out_of_bounds":
        payload["object_prototype_fragment_ids"][0, 0] = 2
    else:
        payload["object_prototype_view_ids"][0, 0] = 2
    with pytest.raises(ValueError):
        validate_hypothesis_memory(payload)
