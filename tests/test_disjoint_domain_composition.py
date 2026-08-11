import pytest
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.disjoint_domain_composition import (
    MODE,
    compose_disjoint_domain_unary,
)
from radio_gs.scripts.score_nvos_disjoint_domain_composition import (
    _verify_partition,
)


def _compose():
    return compose_disjoint_domain_unary(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        original_observation_confidence=torch.tensor([0.8, 0.0, 0.0, 0.5]),
        memory_confidence=torch.tensor([0.9, 0.4, 0.0, 0.3]),
        hard_anchor_mask=torch.tensor([True, False, False, False]),
    )


def test_partition_is_exhaustive_disjoint_and_never_double_counts():
    result = _compose()
    assert torch.equal(result.observed_rows, torch.tensor([True, False, False, True]))
    assert torch.equal(result.memory_rows, torch.tensor([False, True, False, False]))
    assert torch.equal(result.abstained_rows, torch.tensor([False, False, True, False]))
    assert result.diagnostics["mode"] == MODE
    assert result.diagnostics["partition_exhaustive"] is True
    assert result.diagnostics["partition_pairwise_disjoint"] is True
    assert result.diagnostics["same_row_double_counted"] is False
    assert result.diagnostics["raw_memory_observed_rows_ignored"] == 2


def test_partition_assignments_are_commutative_and_bitwise():
    result = _compose()
    assert torch.equal(result.unary, torch.tensor([1.0, -2.0, 3.0, 4.0]))
    assert result.diagnostics["assignment_commutative"] is True
    assert result.diagnostics["observed_unary_bitwise_equal_to_learned"] is True
    assert result.diagnostics["memory_unary_bitwise_equal_to_memory_branch"] is True
    assert result.diagnostics["abstained_unary_bitwise_equal_to_learned_field_prior"] is True
    assert result.diagnostics["hard_anchor_unary_bitwise_equal_to_learned"] is True


def test_hard_anchor_outside_original_observed_domain_is_rejected():
    with pytest.raises(ValueError, match="hard anchors"):
        compose_disjoint_domain_unary(
            torch.tensor([1.0, 2.0]),
            torch.tensor([-1.0, -2.0]),
            original_observation_confidence=torch.tensor([0.8, 0.0]),
            memory_confidence=torch.tensor([0.0, 0.4]),
            hard_anchor_mask=torch.tensor([False, True]),
        )


def test_no_memory_write_on_an_abstention_is_rejected():
    with pytest.raises(ValueError, match="does not complete"):
        compose_disjoint_domain_unary(
            torch.tensor([1.0, 2.0]),
            torch.tensor([-1.0, -2.0]),
            original_observation_confidence=torch.tensor([0.8, 0.0]),
            memory_confidence=torch.tensor([0.5, 0.0]),
            hard_anchor_mask=torch.tensor([True, False]),
        )


def _sealed_partition_fixture():
    observed = torch.tensor([True, False, False, True])
    memory = torch.tensor([False, True, False, False])
    abstained = torch.tensor([False, False, True, False])
    anchors = torch.tensor([True, False, False, False])
    masks = {
        "observed_rows": observed,
        "memory_rows": memory,
        "abstained_rows": abstained,
        "hard_anchor_rows": anchors,
    }
    composed = {
        "valid_rows": torch.arange(4),
        "primitive_unary_probability": torch.tensor([0.8, 0.2, 0.4, 0.7]),
        "disjoint_domain_partition": {
            "mode": MODE,
            "global_rows": torch.arange(4),
            **masks,
            "tensor_sha256": {
                name: tensor_sha256(value.contiguous())
                for name, value in masks.items()
            },
        },
        "compiler_contract": {
            "registered_disjoint_domain_composition": {
                "partition_exhaustive": True,
                "partition_pairwise_disjoint": True,
                "assignment_commutative": True,
                "same_row_double_counted": False,
                "probability_average_or_product_of_experts_used": False,
                "observed_mask_sha256": tensor_sha256(observed.contiguous()),
                "observed_rows": 2,
                "memory_rows": 1,
                "abstained_rows": 1,
            }
        },
    }
    likelihood = {
        "valid_rows": torch.arange(4),
        "primitive_unary_probability": torch.tensor([0.8, 0.6, 0.4, 0.7]),
        "compiler_contract": {
            "registered_query_likelihood": {
                "observed_rows": 2,
                "abstained_rows": 2,
            }
        },
    }
    region_memory = {
        "valid_rows": torch.arange(4),
        "primitive_unary_probability": torch.tensor([0.1, 0.2, 0.9, 0.3]),
        "compiler_contract": {
            "object_multiview_region_memory": {
                "diagnostics": {
                    "base_observed_rows": 2,
                    "base_abstained_rows": 2,
                    "completed_rows": 1,
                }
            }
        },
    }
    return composed, likelihood, region_memory


def test_sealed_partition_replay_matches_both_branch_authorities():
    report = _verify_partition(*_sealed_partition_fixture())
    assert report["observed_rows"] == 2
    assert report["memory_rows"] == 1
    assert report["historical_branch_decision_flips_at_0_5"] == 0


def test_sealed_partition_rejects_overlapping_domains():
    composed, likelihood, memory = _sealed_partition_fixture()
    composed["disjoint_domain_partition"]["memory_rows"][0] = True
    composed["disjoint_domain_partition"]["tensor_sha256"]["memory_rows"] = (
        tensor_sha256(
            composed["disjoint_domain_partition"]["memory_rows"].contiguous()
        )
    )
    with pytest.raises(ValueError, match="causal contract"):
        _verify_partition(composed, likelihood, memory)
