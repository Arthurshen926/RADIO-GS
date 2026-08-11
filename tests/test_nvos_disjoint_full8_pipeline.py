from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.scripts.run_nvos_disjoint_full8_prediction import (
    normalized_arguments,
    verify_partition_artifact,
)
from radio_gs.scripts.score_nvos_disjoint_full8 import score_probability


def _artifacts() -> tuple[dict[str, object], dict[str, object]]:
    valid = torch.tensor([True, False, True, True, True])
    valid_rows = torch.tensor([0, 2, 3, 4])
    observed = torch.tensor([True, False, True, False])
    memory = torch.tensor([False, True, False, False])
    abstained = torch.tensor([False, False, False, True])
    anchors = torch.tensor([True, False, False, False])
    base = {
        "protocol_hash": "protocol",
        "capability_cache": "/tmp/field.pt",
        "capability_cache_sha256": "field-sha",
        "valid": valid,
        "valid_rows": valid_rows,
    }
    candidate = {
        **base,
        "primitive_unary_probability": torch.tensor([1.0, 0.5, 0.3, 0.8, 0.4]),
        "disjoint_domain_partition": {
            "global_rows": valid_rows,
            "observed_rows": observed,
            "memory_rows": memory,
            "abstained_rows": abstained,
            "hard_anchor_rows": anchors,
            "tensor_sha256": {
                "observed_rows": tensor_sha256(observed),
                "memory_rows": tensor_sha256(memory),
                "abstained_rows": tensor_sha256(abstained),
                "hard_anchor_rows": tensor_sha256(anchors),
            },
        },
        "compiler_contract": {
            "registered_query_likelihood": {
                "observed_rows": 2,
                "abstained_rows": 2,
            },
            "object_multiview_region_memory": {
                "diagnostics": {
                    "base_observed_rows": 2,
                    "base_abstained_rows": 2,
                    "completed_rows": 1,
                    "observed_values_bitwise_equal": True,
                    "observed_confidence_bitwise_equal": True,
                    "observed_unary_bitwise_equal_to_likelihood": True,
                }
            },
            "registered_disjoint_domain_composition": {
                "observed_rows": 2,
                "memory_rows": 1,
                "abstained_rows": 1,
                "partition_exhaustive": True,
                "partition_pairwise_disjoint": True,
                "assignment_commutative": True,
                "same_row_double_counted": False,
                "probability_average_or_product_of_experts_used": False,
                "observed_unary_bitwise_equal_to_learned": True,
                "hard_anchor_unary_bitwise_equal_to_learned": True,
                "observed_mask_sha256": tensor_sha256(observed),
            },
        },
    }
    return candidate, base


def test_full8_partition_verifier_accepts_exhaustive_disjoint_assignment() -> None:
    candidate, base = _artifacts()
    summary = verify_partition_artifact(candidate, base)
    assert summary["observed_rows"] == 2
    assert summary["memory_rows"] == 1
    assert summary["abstained_rows"] == 1
    assert summary["observed_and_hard_anchor_unary_bitwise_preserved"] is True


def test_full8_partition_verifier_rejects_same_row_double_count() -> None:
    candidate, base = _artifacts()
    corrupted = copy.deepcopy(candidate)
    corrupted["disjoint_domain_partition"]["memory_rows"] = torch.tensor(
        [True, True, False, False]
    )
    corrupted["disjoint_domain_partition"]["tensor_sha256"]["memory_rows"] = tensor_sha256(
        corrupted["disjoint_domain_partition"]["memory_rows"]
    )
    with pytest.raises(ValueError, match="causal contract"):
        verify_partition_artifact(corrupted, base)


def test_full8_argument_normalization_removes_only_paths_and_new_factors() -> None:
    base = {"scene_id": "flower", "device": "cuda:0", "output_dir": "/old"}
    candidate = {
        **base,
        "output_dir": "/new",
        "primitive_unary_output": "/new/u.pt",
        "registered_query_likelihood_calibration": "mode",
        "object_multiview_region_memory": "memory",
    }
    assert normalized_arguments(candidate) == normalized_arguments(base)


def test_full8_scorer_uses_frozen_linear_resize_and_half_threshold() -> None:
    score = np.array([[0.49, 0.51], [0.2, 0.9]], dtype=np.float32)
    target = np.array([[False, True], [False, True]])
    metrics = score_probability(score, target)
    assert metrics["foreground_iou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0
