from __future__ import annotations

import math

import pytest
import torch

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    CONSERVATIVE_ANGLE_RADIANS,
    MAXIMUM_ANGLE_RADIANS,
    RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256,
    VIEW_AGREEMENT_SCALAR,
    VIEW_AGREEMENT_SHA256_FIELD,
    reliability_conditioned_geodesic_fusion,
    reliability_geodesic_budget_contract,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _base(rows: int = 3) -> torch.Tensor:
    result = torch.zeros(rows, 3, 3)
    result[..., 0] = 1.0
    return result


def _teacher(rows: int = 3) -> torch.Tensor:
    result = torch.zeros(rows, 3)
    result[..., 1] = 1.0
    return result


def test_contract_is_query_free_scene_general_and_declares_minimal_v2_scalar() -> None:
    contract = reliability_geodesic_budget_contract()
    assert contract["required_for_expansion"]["name"] == VIEW_AGREEMENT_SCALAR
    change = contract["streamer_v2_minimal_record_change"]
    assert change["tensor_field"] == VIEW_AGREEMENT_SCALAR
    assert change["hash_field"] == VIEW_AGREEMENT_SHA256_FIELD
    assert change["calculation_point"] == "before_teacher_sum_l2_normalization"
    assert change["per_view_descriptors_retained"] is False
    assert contract["conservative_angle_radians"] == 0.15
    assert "not_yet_source_preregistered" in contract["maximum_angle_status"]
    assert contract["benchmark_candidate_authorized"] is False
    assert contract["scene_or_query_id_consumed"] is False
    assert contract["per_scene_parameters"] is False
    assert contract["benchmark_labels_consumed"] is False
    assert contract["query_independent"] is True
    assert RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256 == canonical_json_sha256(
        contract
    )


def test_v1_missing_agreement_is_exact_conservative_o1_budget() -> None:
    result = reliability_conditioned_geodesic_fusion(
        _base(),
        _teacher(),
        teacher_valid=torch.ones(3, dtype=torch.bool),
        retained_view_count=torch.tensor([1, 2, 4], dtype=torch.uint8),
    )
    assert result.agreement_available is False
    assert not bool(result.expanded_budget.any())
    torch.testing.assert_close(
        result.angular_budget_radians,
        torch.full((3,), CONSERVATIVE_ANGLE_RADIANS),
    )
    torch.testing.assert_close(
        result.angular_step_radians,
        torch.full((3, 3), CONSERVATIVE_ANGLE_RADIANS),
    )


def test_reliable_four_view_row_expands_continuously_to_fixed_maximum() -> None:
    result = reliability_conditioned_geodesic_fusion(
        _base(1),
        _teacher(1),
        teacher_valid=torch.tensor([True]),
        retained_view_count=torch.tensor([4]),
        teacher_view_directional_resultant=torch.tensor([1.0]),
    )
    torch.testing.assert_close(result.reliability_score, torch.ones(1))
    torch.testing.assert_close(
        result.angular_budget_radians,
        torch.full((1,), MAXIMUM_ANGLE_RADIANS),
    )
    angle = torch.acos(
        (result.descriptor.float() * _base(1)).sum(dim=-1).clamp(-1.0, 1.0)
    )
    torch.testing.assert_close(
        angle,
        torch.full((1, 3), MAXIMUM_ANGLE_RADIANS),
        atol=2e-6,
        rtol=0.0,
    )


def test_single_view_never_expands_even_with_unit_agreement() -> None:
    result = reliability_conditioned_geodesic_fusion(
        _base(1),
        _teacher(1),
        teacher_valid=torch.tensor([True]),
        retained_view_count=torch.tensor([1]),
        teacher_view_directional_resultant=torch.tensor([1.0]),
    )
    assert float(result.count_sufficiency[0]) == 0.0
    assert float(result.reliability_score[0]) == 0.0
    assert float(result.angular_budget_radians[0]) == pytest.approx(0.15)


def test_optional_query_free_reliabilities_only_attenuate_budget() -> None:
    common = {
        "teacher_valid": torch.tensor([True]),
        "retained_view_count": torch.tensor([4]),
        "teacher_view_directional_resultant": torch.tensor([0.81]),
    }
    view_only = reliability_conditioned_geodesic_fusion(
        _base(1), _teacher(1), **common
    )
    attenuated = reliability_conditioned_geodesic_fusion(
        _base(1),
        _teacher(1),
        responsibility_reliability=torch.tensor([0.5]),
        canonical_field_reliability=torch.tensor([0.8]),
        **common,
    )
    assert float(view_only.reliability_score[0]) == pytest.approx(0.9)
    assert float(attenuated.reliability_score[0]) == pytest.approx(0.36)
    assert bool(
        attenuated.angular_budget_radians[0]
        < view_only.angular_budget_radians[0]
    )


def test_each_scale_moves_from_its_own_o0_direction() -> None:
    base = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    teacher = torch.tensor([[1.0, 1.0, 1.0]]) / math.sqrt(3.0)
    result = reliability_conditioned_geodesic_fusion(
        base,
        teacher,
        teacher_valid=torch.tensor([True]),
        retained_view_count=torch.tensor([4]),
        teacher_view_directional_resultant=torch.tensor([0.25]),
    )
    requested = torch.acos(
        (base * teacher[:, None]).sum(dim=-1).clamp(-1.0, 1.0)
    )
    torch.testing.assert_close(result.requested_angle_radians, requested)
    assert not torch.equal(result.descriptor[:, 0], result.descriptor[:, 1])
    assert not torch.equal(result.descriptor[:, 1], result.descriptor[:, 2])


def test_invalid_teacher_and_antipodal_routes_are_bitwise_o0() -> None:
    base = _base(2)
    teacher = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    result = reliability_conditioned_geodesic_fusion(
        base,
        teacher,
        teacher_valid=torch.tensor([False, True]),
        retained_view_count=torch.tensor([0, 4]),
        teacher_view_directional_resultant=torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(result.descriptor, base)
    assert bool(result.fallback_to_o0.all())
    assert not bool(result.teacher_applied.any())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retained_view_count": torch.tensor([5])}, "integral in"),
        (
            {"teacher_view_directional_resultant": torch.tensor([1.1])},
            VIEW_AGREEMENT_SCALAR,
        ),
        (
            {"responsibility_reliability": torch.tensor([-0.1])},
            "responsibility_reliability",
        ),
    ],
)
def test_malformed_reliability_fails_closed(
    kwargs: dict[str, torch.Tensor], message: str
) -> None:
    arguments = {
        "teacher_valid": torch.tensor([True]),
        "retained_view_count": torch.tensor([4]),
        "teacher_view_directional_resultant": torch.tensor([1.0]),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        reliability_conditioned_geodesic_fusion(
            _base(1), _teacher(1), **arguments
        )


def test_teacher_validity_must_match_count_and_invalid_rows_are_zero() -> None:
    with pytest.raises(ValueError, match="must equal"):
        reliability_conditioned_geodesic_fusion(
            _base(1),
            _teacher(1),
            teacher_valid=torch.tensor([False]),
            retained_view_count=torch.tensor([1]),
        )
    with pytest.raises(ValueError, match="exact zero"):
        reliability_conditioned_geodesic_fusion(
            _base(1),
            _teacher(1),
            teacher_valid=torch.tensor([False]),
            retained_view_count=torch.tensor([0]),
        )
