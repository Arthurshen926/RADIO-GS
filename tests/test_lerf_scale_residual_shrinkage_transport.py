from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces import lerf_scale_equivariant_geodesic_transport as v1
from radio_gs.interfaces import lerf_scale_residual_shrinkage_transport as v2
from radio_gs.scripts import (
    materialize_lerf_transport_v2_source_loo_streaming_hook as hook,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _frame(rows: int = 2) -> torch.Tensor:
    base = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0, 0.0],
            [0.8, -0.6, 0.0, 0.0, 0.0],
        ]
    )
    return base[None].expand(rows, -1, -1).clone()


def _teacher(rows: int = 2) -> torch.Tensor:
    result = torch.zeros(rows, 5)
    result[:, 1] = 1.0
    return result


def test_joint_grid_and_contract_are_finite_global_and_hash_bound() -> None:
    grid = v2.candidate_grid()
    contract = v2.residual_shrinkage_contract()
    assert len(grid) == 25
    assert grid[0] == {
        "maximum_angle_radians": 0.15,
        "gamma_policy": "rigid",
        "dispersion_exponent": None,
    }
    assert len({(x["maximum_angle_radians"], x["gamma_policy"]) for x in grid}) == 25
    assert contract["scene_or_query_id_consumed"] is False
    assert contract["per_scene_parameters"] is False
    assert contract["formal_target_candidate_authorized"] is False
    assert contract["source_only_gate"]["minimum_source_scenes"] == 2
    assert v2.RESIDUAL_SHRINKAGE_CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_gamma_family_is_bounded_and_has_frozen_monotonicity() -> None:
    rho = torch.tensor([0.0, 0.25, 0.5, 1.0])
    dispersion = torch.full_like(rho, 0.36)
    for policy in ("k0", "k0p5", "k1", "k2"):
        gamma = v2.residual_shrinkage_gamma(rho, dispersion, gamma_policy=policy)
        assert bool(((0.0 <= gamma) & (gamma <= 1.0)).all())
        assert bool((gamma[1:] <= gamma[:-1]).all())
        varying_dispersion = torch.tensor([0.0, 0.25, 0.5, 1.0])
        gamma_by_dispersion = v2.residual_shrinkage_gamma(
            torch.full_like(varying_dispersion, 0.8),
            varying_dispersion,
            gamma_policy=policy,
        )
        assert bool((gamma_by_dispersion[1:] >= gamma_by_dispersion[:-1]).all())
    torch.testing.assert_close(
        v2.residual_shrinkage_gamma(rho, dispersion, gamma_policy="rigid"),
        torch.ones_like(rho),
    )
    torch.testing.assert_close(
        v2.residual_shrinkage_gamma(
            torch.tensor([0.8]), torch.tensor([1.0]), gamma_policy="k0"
        ),
        torch.tensor([0.2]),
    )


def test_rigid_endpoint_matches_normalized_common_transport() -> None:
    base = _frame()
    teacher = _teacher()
    common = v1.scale_equivariant_geodesic_transport(
        base,
        teacher,
        teacher_valid=torch.ones(2, dtype=torch.bool),
        retained_view_count=torch.full((2,), 4, dtype=torch.uint8),
        teacher_view_directional_resultant=torch.ones(2),
        maximum_angle_radians=0.75,
    )
    output = v2.scale_residual_shrinkage_transport(
        base,
        teacher,
        teacher_valid=torch.ones(2, dtype=torch.bool),
        retained_view_count=torch.full((2,), 4, dtype=torch.uint8),
        teacher_view_directional_resultant=torch.ones(2),
        maximum_angle_radians=0.75,
        gamma_policy="rigid",
    )
    torch.testing.assert_close(
        output.descriptor,
        F.normalize(common.descriptor, dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )
    assert bool(output.teacher_applied.all())
    assert bool(output.reconstruction_valid.all())


def test_k0_reliable_endpoint_collapses_only_centered_scale_residual() -> None:
    output = v2.scale_residual_shrinkage_transport(
        _frame(rows=1),
        _teacher(rows=1),
        teacher_valid=torch.ones(1, dtype=torch.bool),
        retained_view_count=torch.full((1,), 4, dtype=torch.uint8),
        teacher_view_directional_resultant=torch.ones(1),
        maximum_angle_radians=0.75,
        gamma_policy="k0",
    )
    torch.testing.assert_close(output.gamma, torch.zeros(1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        output.descriptor[:, 0], output.descriptor[:, 1], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        output.descriptor[:, 1], output.descriptor[:, 2], atol=2e-6, rtol=2e-6
    )


def test_single_view_cannot_shrink_residual_for_any_policy() -> None:
    outputs = []
    for policy in v2.GAMMA_POLICY_EXPONENTS:
        output = v2.scale_residual_shrinkage_transport(
            _frame(rows=1),
            _teacher(rows=1),
            teacher_valid=torch.ones(1, dtype=torch.bool),
            retained_view_count=torch.ones(1, dtype=torch.uint8),
            teacher_view_directional_resultant=torch.ones(1),
            maximum_angle_radians=0.75,
            gamma_policy=policy,
        )
        torch.testing.assert_close(output.gamma, torch.ones(1))
        outputs.append(output.descriptor)
    for descriptor in outputs[1:]:
        torch.testing.assert_close(descriptor, outputs[0], atol=2e-6, rtol=2e-6)


def test_invalid_undefined_same_and_antipodal_rows_fail_closed_bitwise() -> None:
    base = torch.zeros(4, 3, 5)
    base[:, :, 0] = 1.0
    base[1] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-0.5, 0.8660254, 0.0, 0.0, 0.0],
            [-0.5, -0.8660254, 0.0, 0.0, 0.0],
        ]
    )
    teacher = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    output = v2.scale_residual_shrinkage_transport(
        base,
        teacher,
        teacher_valid=torch.tensor([False, True, True, True]),
        retained_view_count=torch.tensor([0, 4, 4, 4], dtype=torch.uint8),
        teacher_view_directional_resultant=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        maximum_angle_radians=0.75,
        gamma_policy="k0p5",
    )
    assert torch.equal(output.descriptor, base)
    assert not bool(output.teacher_applied.any())
    assert bool(output.fallback_to_o0.all())


def _loo_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, dimension = 5, 7
    views = torch.zeros(rows, 4, dimension)
    views[..., 1] = 1.0
    frame_ids = torch.arange(4, dtype=torch.int32)[None].expand(rows, -1).clone()
    base = torch.zeros(rows, 3, dimension)
    base[:, 0, 0] = 1.0
    base[:, 1, :2] = torch.tensor([0.8, 0.6])
    base[:, 2, :2] = torch.tensor([0.8, -0.6])
    return views, frame_ids, base


def test_source_loo_reports_mean_exact_p05_and_nonregression_flags() -> None:
    views, frame_ids, base = _loo_fixture()
    audit = v2.source_only_leave_one_view_out_residual_shrinkage_audit(
        views, frame_ids, base, row_chunk=2
    )
    v2.validate_source_only_residual_shrinkage_audit(audit)
    assert len(audit["candidate_grid"]) == 25
    assert audit["heldout_predictions"] == 20
    assert audit["heldout_scale_observations"] == 60
    baseline = audit["candidate_grid"][0]
    assert baseline["mean_delta_vs_baseline"] == 0.0
    assert baseline["p05_delta_vs_baseline"] == 0.0
    assert baseline["mean_nonregression_vs_baseline"] is True
    assert baseline["p05_nonregression_vs_baseline"] is True
    assert max(row["mean_delta_vs_baseline"] for row in audit["candidate_grid"]) > 0


def test_source_loo_reuses_each_common_rotation_across_gamma_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views, frame_ids, base = _loo_fixture()
    original = v2.scale_equivariant_geodesic_transport
    calls = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(v2, "scale_equivariant_geodesic_transport", counted)
    v2.source_only_leave_one_view_out_residual_shrinkage_audit(
        views, frame_ids, base, row_chunk=2
    )
    assert calls == 3 * 4 * len(v1.CEILING_GRID_RADIANS)


def test_cross_scene_gate_requires_mean_and_p05_nonregression() -> None:
    views, frame_ids, base = _loo_fixture()
    audit_a = v2.source_only_leave_one_view_out_residual_shrinkage_audit(
        views, frame_ids, base, row_chunk=2
    )
    audit_b = v2.source_only_leave_one_view_out_residual_shrinkage_audit(
        views[:3], frame_ids[:3], base[:3], row_chunk=2
    )
    gate = v2.select_source_only_residual_shrinkage_candidate(
        {"source_a": audit_a, "source_b": audit_b}
    )
    v2.validate_source_only_residual_shrinkage_gate(gate)
    assert gate["source_gate_passed"] is True
    assert gate["selected_candidate_index"] not in (None, 0)
    selected = gate["candidate_grid"][gate["selected_candidate_index"]]
    assert selected["pooled_mean_delta_vs_baseline"] > 0
    assert selected["every_scene_mean_nonregression"] is True
    assert selected["every_scene_p05_nonregression"] is True
    assert gate["target_candidate_authorized"] is False
    with pytest.raises(ValueError, match="at least two scenes"):
        v2.select_source_only_residual_shrinkage_candidate({"source_a": audit_a})


def test_streaming_hook_emits_validated_scalar_only_capture() -> None:
    views, frame_ids, base = _loo_fixture()
    capture = hook.capture_source_only_transport_v2_loo(
        scene_id="source_scene_0001",
        top_descriptors=views,
        top_frame_ids=frame_ids,
        o0_descriptor_by_scale=base,
        row_chunk=3,
    )
    hook.validate_streaming_hook_capture(capture)
    assert hook.HOOK_CONTRACT_SHA256 == canonical_json_sha256(hook.hook_contract())

    def contains_tensor(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, Mapping):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(capture)
    assert capture["target_candidate_authorized"] is False

    malformed = copy.deepcopy(capture)
    malformed["access_audit"]["target_metric_executed"] = True
    with pytest.raises(ValueError, match="streaming capture differs"):
        hook.validate_streaming_hook_capture(malformed)


def test_bad_gamma_and_zero_retained_view_fail_closed_at_validation() -> None:
    with pytest.raises(ValueError, match="gamma inputs differ"):
        v2.residual_shrinkage_gamma(
            torch.tensor([1.1]), torch.tensor([0.5]), gamma_policy="k1"
        )
    views, frame_ids, base = _loo_fixture()
    views[0, 0] = 0
    with pytest.raises(ValueError, match="retained.*nonzero"):
        v2.source_only_leave_one_view_out_residual_shrinkage_audit(
            views, frame_ids, base
        )
