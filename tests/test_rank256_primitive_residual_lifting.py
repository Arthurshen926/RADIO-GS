from __future__ import annotations

import math

import pytest
import torch

from radio_gs.field.rank256_primitive_residual_lifting import (
    CONTRACT_SHA256,
    Rank256PrimitiveLiftingConfig,
    lift_rank256_region_residual_to_o0_multiscale,
    lifting_contract,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _unit(angle: float, *, axis: int = 1, dimension: int = 4) -> torch.Tensor:
    value = torch.zeros(dimension, dtype=torch.float32)
    value[0] = math.cos(angle)
    value[axis] = math.sin(angle)
    return value


def _base_inputs() -> dict[str, torch.Tensor]:
    o0 = torch.zeros((4, 2, 4), dtype=torch.float32)
    o0[:, :, 0] = 1.0
    return {
        "o0_primitive_descriptor": o0,
        "primitive_valid_mask": torch.tensor([True, True, True, False]),
        "region_base_descriptor": torch.stack((_unit(0.0), _unit(0.0))),
        "region_semantic_descriptor": torch.stack((_unit(0.4), _unit(0.2, axis=2))),
        "region_rows": torch.tensor([[0, 1, -1], [1, 2, -1]]),
        "token_mask": torch.tensor(
            [[True, True, False], [True, True, False]]
        ),
        "canonical_region_indices": torch.tensor([20, 10], dtype=torch.int64),
        "region_reliability": torch.tensor([1.0, 0.5]),
        "region_active_mask": torch.tensor([True, True]),
        "region_ood_mask": torch.tensor([False, False]),
    }


def _run(
    values: dict[str, torch.Tensor] | None = None,
    *,
    max_angle: float = 1.0,
    minimum_reliability: float = 0.0,
):
    return lift_rank256_region_residual_to_o0_multiscale(
        **(_base_inputs() if values is None else values),
        config=Rank256PrimitiveLiftingConfig(
            max_angle_radians=max_angle,
            minimum_region_reliability=minimum_reliability,
        ),
    )


def _bytes(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().view(torch.uint8)


def test_contract_is_static_query_free_and_hash_bound() -> None:
    contract = lifting_contract()
    assert contract["query_conditioned_parameters"] is False
    assert contract["scene_conditioned_parameters"] is False
    assert contract["target_metrics_used"] is False
    assert "fp64" in contract["overlap_aggregation"]
    assert CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_single_region_log_map_lifts_exact_geodesic_to_every_scale() -> None:
    values = _base_inputs()
    values["region_rows"] = torch.tensor([[0], [2]])
    values["token_mask"] = torch.ones((2, 1), dtype=torch.bool)
    values["region_active_mask"] = torch.tensor([True, False])
    result = _run(values)
    expected = _unit(0.4)
    torch.testing.assert_close(
        result.primitive_descriptor[0], expected.expand(2, -1), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        result.angular_step_radians[0], torch.full((2,), 0.4, dtype=torch.float64),
        atol=2e-7, rtol=0,
    )
    assert result.coverage_count.tolist() == [1, 0, 0, 0]
    assert result.updated_mask[0].tolist() == [True, True]


def test_reliability_attenuates_instead_of_cancelling_for_one_region() -> None:
    values = _base_inputs()
    values["region_rows"] = torch.tensor([[0], [2]])
    values["token_mask"] = torch.ones((2, 1), dtype=torch.bool)
    values["region_active_mask"] = torch.tensor([True, False])
    values["region_reliability"] = torch.tensor([0.25, 1.0])
    result = _run(values)
    torch.testing.assert_close(
        result.angular_step_radians[0], torch.full((2,), 0.1, dtype=torch.float64),
        atol=2e-7, rtol=0,
    )
    assert result.aggregate_reliability[0] == pytest.approx(0.25)


def test_overlap_is_fp64_count_normalized_and_multidirectional() -> None:
    result = _run()
    # Primitive one sees 1.0 * 0.4 e1 and 0.5 * 0.2 e2, divided by two.
    expected_angle = math.sqrt(0.2**2 + 0.05**2)
    assert result.coverage_count.tolist() == [1, 2, 1, 0]
    assert result.aggregate_reliability.dtype == torch.float64
    assert result.aggregate_reliability[1] == pytest.approx(0.75)
    assert result.aggregate_residual_norm[1] == pytest.approx(expected_angle, abs=3e-7)
    assert result.angular_step_radians[1, 0] == pytest.approx(expected_angle, abs=3e-7)
    output = result.primitive_descriptor[1, 0]
    assert output[1] > 0 and output[2] > 0
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.primitive_descriptor.double(), dim=-1)[
            :3
        ],
        torch.ones((3, 2), dtype=torch.float64),
        atol=2e-7,
        rtol=0,
    )


def test_tangent_projection_is_independent_for_each_preserved_scale() -> None:
    values = _base_inputs()
    o0 = values["o0_primitive_descriptor"].clone()
    o0[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    values["o0_primitive_descriptor"] = o0
    values["region_rows"] = torch.tensor([[0], [2]])
    values["token_mask"] = torch.ones((2, 1), dtype=torch.bool)
    values["region_active_mask"] = torch.tensor([True, False])
    result = _run(values)
    # The region log residual is parallel to scale one's e1 base, so that
    # scale is an exact fallback while scale zero takes the spherical step.
    assert result.updated_mask[0].tolist() == [True, False]
    assert torch.equal(
        _bytes(result.primitive_descriptor[0, 1]), _bytes(o0[0, 1])
    )
    assert result.primitive_descriptor.shape == o0.shape


def test_hard_cap_bounds_the_primitive_geodesic_step() -> None:
    values = _base_inputs()
    values["region_rows"] = torch.tensor([[0], [2]])
    values["token_mask"] = torch.ones((2, 1), dtype=torch.bool)
    values["region_active_mask"] = torch.tensor([True, False])
    result = _run(values, max_angle=0.15)
    torch.testing.assert_close(
        result.angular_step_radians[0], torch.full((2,), 0.15, dtype=torch.float64),
        atol=1e-12, rtol=0,
    )
    cosine = float(
        (values["o0_primitive_descriptor"][0, 0]
         * result.primitive_descriptor[0, 0]).sum()
    )
    assert math.acos(min(1.0, max(-1.0, cosine))) == pytest.approx(0.15, abs=2e-6)


def test_canonical_region_order_makes_input_permutation_bitwise_stable() -> None:
    values = _base_inputs()
    first = _run(values)
    permutation = torch.tensor([1, 0])
    permuted = dict(values)
    for key in (
        "region_base_descriptor",
        "region_semantic_descriptor",
        "region_rows",
        "token_mask",
        "canonical_region_indices",
        "region_reliability",
        "region_active_mask",
        "region_ood_mask",
    ):
        permuted[key] = values[key][permutation]
    second = _run(permuted)
    assert torch.equal(
        _bytes(first.primitive_descriptor), _bytes(second.primitive_descriptor)
    )
    assert torch.equal(first.coverage_count, second.coverage_count)
    assert torch.equal(first.aggregate_reliability, second.aggregate_reliability)
    assert torch.equal(first.angular_step_radians, second.angular_step_radians)


def test_inactive_ood_low_reliability_uncovered_and_invalid_are_bitwise_o0() -> None:
    values = _base_inputs()
    o0 = values["o0_primitive_descriptor"].clone()
    o0[2, :, 1] = -0.0
    o0[3, :, 0] = -0.0
    values["o0_primitive_descriptor"] = o0
    values["region_active_mask"] = torch.tensor([False, True])
    values["region_ood_mask"] = torch.tensor([False, True])
    result = _run(values, minimum_reliability=0.75)
    assert result.region_contribution_mask.tolist() == [False, False]
    assert not bool(result.updated_mask.any())
    assert torch.equal(_bytes(result.primitive_descriptor), _bytes(o0))

    # Even covered invalid rows are forbidden to change.
    values["region_rows"] = torch.tensor([[3, -1, -1], [2, -1, -1]])
    values["token_mask"] = torch.tensor(
        [[True, False, False], [True, False, False]]
    )
    values["region_active_mask"] = torch.tensor([True, False])
    values["region_ood_mask"] = torch.tensor([False, False])
    result = _run(values)
    assert result.coverage_count[3] == 1
    assert not bool(result.updated_mask[3].any())
    assert torch.equal(
        _bytes(result.primitive_descriptor[3]), _bytes(o0[3])
    )


def test_fp16_fallback_bytes_and_updated_unit_gauge_are_preserved() -> None:
    values = _base_inputs()
    values["o0_primitive_descriptor"] = values[
        "o0_primitive_descriptor"
    ].half()
    values["region_rows"] = torch.tensor([[0], [2]])
    values["token_mask"] = torch.ones((2, 1), dtype=torch.bool)
    values["region_active_mask"] = torch.tensor([True, False])
    result = _run(values)
    assert result.primitive_descriptor.dtype == torch.float16
    assert torch.equal(
        _bytes(result.primitive_descriptor[1:]),
        _bytes(values["o0_primitive_descriptor"][1:]),
    )
    norms = torch.linalg.vector_norm(
        result.primitive_descriptor[0].float(), dim=-1
    )
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=8e-4, rtol=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate primitive"),
        ("canonical", "canonical_region_indices"),
        ("unit", "unit L2 gauge"),
        ("range", "out-of-range primitive"),
    ],
)
def test_malformed_alignment_fails_closed(mutation: str, message: str) -> None:
    values = _base_inputs()
    if mutation == "duplicate":
        values["region_rows"][0, 1] = values["region_rows"][0, 0]
    elif mutation == "canonical":
        values["canonical_region_indices"] = torch.tensor([1, 1])
    elif mutation == "unit":
        values["o0_primitive_descriptor"][0, 0] *= 0.5
    elif mutation == "range":
        values["region_rows"][0, 0] = 99
    with pytest.raises(ValueError, match=message):
        _run(values)

