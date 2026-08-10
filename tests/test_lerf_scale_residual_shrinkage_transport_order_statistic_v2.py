from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces import (
    lerf_scale_equivariant_geodesic_transport as common_transport,
)
from radio_gs.interfaces import (
    lerf_scale_residual_shrinkage_transport as sealed,
)
from radio_gs.interfaces import (
    lerf_scale_residual_shrinkage_transport_order_statistic_v2 as optimized,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_order_statistic_quantile_is_bitwise_torch_equivalent(
    dtype: torch.dtype,
) -> None:
    for length in (1, 2, 3, 7, 20, 21, 22, 101, 1000):
        generator = torch.Generator().manual_seed(9100 + length)
        values = torch.randn(25, length, generator=generator, dtype=dtype)
        if length >= 3:
            values[:, :3] = values[:, :1]
        for quantile in (0.0, 0.05, 0.5, 1.0):
            expected = torch.quantile(values, quantile, dim=1, interpolation="linear")
            actual = optimized.exact_linear_quantile_by_order_statistic(
                values, quantile
            )
            assert torch.equal(actual, expected)
            assert float((actual - expected).abs().max()) <= 1e-7


def _random_transport_fixture(
    *, seed: int, rows: int = 53, dimension: int = 1536
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(rows, 3, dimension, generator=generator), dim=-1)
    teacher = F.normalize(torch.randn(rows, dimension, generator=generator), dim=-1)
    heldout = F.normalize(torch.randn(rows, dimension, generator=generator), dim=-1)
    valid = torch.ones(rows, dtype=torch.bool)
    count = torch.randint(1, 5, (rows,), generator=generator)
    agreement = torch.rand(rows, generator=generator)
    return base, teacher, heldout, valid, count, agreement


def test_analytic_scalar_grid_matches_sealed_descriptor_readout() -> None:
    base, teacher, heldout, valid, count, agreement = _random_transport_fixture(
        seed=9301
    )
    maximum_error = 0.0
    for ceiling in common_transport.CEILING_GRID_RADIANS:
        common = common_transport.scale_equivariant_geodesic_transport(
            base,
            teacher,
            teacher_valid=valid,
            retained_view_count=count,
            teacher_view_directional_resultant=agreement,
            maximum_angle_radians=ceiling,
        )
        rho, dispersion = sealed._source_reliability_and_dispersion(
            base, count, agreement, valid
        )
        analytic = optimized.score_gamma_grid_from_common_transport(
            base,
            common.descriptor,
            common.teacher_applied,
            heldout,
            rho,
            dispersion,
        )
        references = []
        for policy in sealed.GAMMA_POLICY_EXPONENTS:
            output = sealed.scale_residual_shrinkage_transport(
                base,
                teacher,
                teacher_valid=valid,
                retained_view_count=count,
                teacher_view_directional_resultant=agreement,
                maximum_angle_radians=ceiling,
                gamma_policy=policy,
            )
            references.append((output.descriptor * heldout[:, None, :]).sum(dim=-1))
        reference = torch.stack(references)
        maximum_error = max(maximum_error, float((analytic - reference).abs().max()))
    assert maximum_error <= 1e-6


def test_analytic_scalar_grid_preserves_sealed_fallback_rows() -> None:
    base = torch.zeros(3, 3, 7)
    base[..., 0] = 1.0
    teacher = torch.zeros(3, 7)
    teacher[1, 0] = 1.0
    teacher[2, 0] = -1.0
    heldout = F.normalize(
        torch.randn(3, 7, generator=torch.Generator().manual_seed(8)), dim=-1
    )
    valid = torch.tensor([False, True, True])
    count = torch.tensor([0, 4, 4])
    agreement = torch.tensor([0.0, 1.0, 1.0])
    common = common_transport.scale_equivariant_geodesic_transport(
        base,
        teacher,
        teacher_valid=valid,
        retained_view_count=count,
        teacher_view_directional_resultant=agreement,
        maximum_angle_radians=0.75,
    )
    rho, dispersion = sealed._source_reliability_and_dispersion(
        base, count, agreement, valid
    )
    actual = optimized.score_gamma_grid_from_common_transport(
        base,
        common.descriptor,
        common.teacher_applied,
        heldout,
        rho,
        dispersion,
    )
    expected = (base * heldout[:, None, :]).sum(dim=-1)[None].expand(5, -1, -1)
    assert torch.equal(actual, expected)


def _loo_fixture(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rows, dimension = 13, 31
    views = F.normalize(torch.randn(rows, 4, dimension, generator=generator), dim=-1)
    frame_ids = torch.arange(4, dtype=torch.int32)[None].expand(rows, -1).clone()
    frame_ids[0, 1:] = -1
    frame_ids[1, 2:] = -1
    frame_ids[2:5, 3:] = -1
    views[frame_ids < 0] = 0
    base = F.normalize(torch.randn(rows, 3, dimension, generator=generator), dim=-1)
    return views, frame_ids, base


def test_optimized_audit_matches_sealed_statistics_and_gate() -> None:
    pairs = []
    for seed in (9401, 9402):
        views, frame_ids, base = _loo_fixture(seed)
        reference = sealed.source_only_leave_one_view_out_residual_shrinkage_audit(
            views, frame_ids, base, row_chunk=5
        )
        actual = optimized.source_only_leave_one_view_out_residual_shrinkage_audit(
            views, frame_ids, base, row_chunk=5
        )
        optimized.validate_source_only_residual_shrinkage_audit(actual)
        assert (
            actual["heldout_scale_observations"]
            == reference["heldout_scale_observations"]
        )
        for reference_row, actual_row in zip(
            reference["candidate_grid"], actual["candidate_grid"]
        ):
            assert (
                reference_row["maximum_angle_radians"]
                == actual_row["maximum_angle_radians"]
            )
            assert reference_row["gamma_policy"] == actual_row["gamma_policy"]
            assert abs(reference_row["mean_cosine"] - actual_row["mean_cosine"]) <= 1e-6
            assert abs(reference_row["p05_cosine"] - actual_row["p05_cosine"]) <= 1e-6
            assert (
                reference_row["mean_nonregression_vs_baseline"]
                is actual_row["mean_nonregression_vs_baseline"]
            )
            assert (
                reference_row["p05_nonregression_vs_baseline"]
                is actual_row["p05_nonregression_vs_baseline"]
            )
        pairs.append((reference, actual))
    sealed_gate = sealed.select_source_only_residual_shrinkage_candidate(
        {f"source_{index}": pair[0] for index, pair in enumerate(pairs)}
    )
    optimized_gate = optimized.select_source_only_residual_shrinkage_candidate(
        {f"source_{index}": pair[1] for index, pair in enumerate(pairs)}
    )
    assert (
        optimized_gate["selected_candidate_index"]
        == sealed_gate["selected_candidate_index"]
    )
    assert optimized_gate["source_gate_passed"] is sealed_gate["source_gate_passed"]
    for reference_row, actual_row in zip(
        sealed_gate["candidate_grid"], optimized_gate["candidate_grid"]
    ):
        assert reference_row["eligible"] is actual_row["eligible"]
        assert (
            abs(reference_row["pooled_mean_cosine"] - actual_row["pooled_mean_cosine"])
            <= 1e-6
        )


def test_explicit_per_cell_equivalence_report_covers_all_candidates() -> None:
    views, frame_ids, base = _loo_fixture(9403)
    report = optimized.compare_analytic_and_sealed_on_source_chunk(
        views, frame_ids, base, row_chunk=5
    )
    assert report["equivalence_gate_passed"] is True
    assert report["candidate_gate_identical"] is True
    assert report["selected_candidate_index_identical"] is True
    assert report["maximum_cell_abs_error"] <= 1e-6
    assert report["maximum_mean_abs_error"] <= 1e-6
    assert report["maximum_p05_abs_error"] <= 1e-6
    assert len(report["candidates"]) == len(sealed.candidate_grid()) == 25
    assert [row["candidate_index"] for row in report["candidates"]] == list(range(25))
    assert all(row["mean_gate_identical"] for row in report["candidates"])
    assert all(row["p05_gate_identical"] for row in report["candidates"])


def test_v2_contract_is_parent_bound_and_target_closed() -> None:
    contract = optimized.source_loo_execution_contract()
    assert contract["sealed_math_contract_sha256"] == (
        sealed.RESIDUAL_SHRINKAGE_CONTRACT_SHA256
    )
    assert contract["exact_linear_p05"]["approximate"] is False
    assert contract["target_candidate_authorized"] is False
    assert optimized.SOURCE_LOO_EXECUTION_CONTRACT_SHA256 == canonical_json_sha256(
        contract
    )
