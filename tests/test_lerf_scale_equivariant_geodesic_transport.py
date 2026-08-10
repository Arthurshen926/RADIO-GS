from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces import lerf_scale_equivariant_geodesic_transport as transport
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _random_frame(rows: int = 9, dimension: int = 31) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3108)
    return F.normalize(
        torch.randn(rows, 3, dimension, generator=generator), dim=-1
    )


def test_common_rotation_preserves_each_norm_and_full_scale_gram() -> None:
    base = _random_frame()
    generator = torch.Generator().manual_seed(3109)
    teacher = F.normalize(torch.randn(9, 31, generator=generator), dim=-1)
    output = transport.scale_equivariant_geodesic_transport(
        base,
        teacher,
        teacher_valid=torch.ones(9, dtype=torch.bool),
        retained_view_count=torch.full((9,), 4, dtype=torch.uint8),
        teacher_view_directional_resultant=torch.ones(9),
        maximum_angle_radians=0.75,
    )
    before_norm = torch.linalg.vector_norm(base.float(), dim=-1)
    after_norm = torch.linalg.vector_norm(output.descriptor, dim=-1)
    before_gram = torch.einsum("bsd,btd->bst", base.float(), base.float())
    after_gram = torch.einsum(
        "bsd,btd->bst", output.descriptor, output.descriptor
    )
    torch.testing.assert_close(after_norm, before_norm, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(after_gram, before_gram, rtol=3e-6, atol=3e-6)
    assert bool(output.teacher_applied.all())


def test_rotated_frame_mean_moves_by_the_reliability_budget() -> None:
    base = torch.zeros(2, 3, 7)
    base[..., 0] = 1.0
    teacher = torch.zeros(2, 7)
    teacher[..., 1] = 1.0
    output = transport.scale_equivariant_geodesic_transport(
        base,
        teacher,
        teacher_valid=torch.ones(2, dtype=torch.bool),
        retained_view_count=torch.tensor([1, 4], dtype=torch.uint8),
        teacher_view_directional_resultant=torch.ones(2),
        maximum_angle_radians=0.75,
    )
    moved = F.normalize(output.descriptor.mean(dim=1), dim=-1)
    angle = torch.acos((moved * base[:, 0]).sum(dim=-1).clamp(-1.0, 1.0))
    torch.testing.assert_close(
        angle, torch.tensor([0.15, 0.75]), rtol=0.0, atol=2e-6
    )
    assert output.expanded_budget.tolist() == [False, True]


def test_invalid_undefined_same_and_antipodal_routes_fail_closed() -> None:
    base = torch.zeros(4, 3, 5)
    base[:, :, 0] = 1.0
    # Row one has an undefined scale mean while retaining unit scale vectors.
    base[1] = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0],
         [-0.5, 0.8660254, 0.0, 0.0, 0.0],
         [-0.5, -0.8660254, 0.0, 0.0, 0.0]]
    )
    teacher = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    output = transport.scale_equivariant_geodesic_transport(
        base,
        teacher,
        teacher_valid=torch.tensor([False, True, True, True]),
        retained_view_count=torch.tensor([0, 4, 4, 4], dtype=torch.uint8),
        teacher_view_directional_resultant=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        maximum_angle_radians=0.75,
    )
    torch.testing.assert_close(output.descriptor, base, rtol=0.0, atol=0.0)
    assert not bool(output.teacher_applied.any())
    assert output.frame_mean_valid.tolist() == [True, False, True, True]
    assert output.same_direction.tolist() == [False, False, True, False]
    assert output.antipodal.tolist() == [False, False, False, True]


def test_source_only_loo_statistic_prefers_more_rotation_for_aligned_views() -> None:
    rows, dimension = 6, 9
    base = torch.zeros(rows, 3, dimension)
    base[..., 0] = 1.0
    views = torch.zeros(rows, 4, dimension)
    views[..., 1] = 1.0
    frame_ids = torch.arange(4, dtype=torch.int32)[None].expand(rows, -1).clone()
    audit = transport.source_only_leave_one_view_out_transport_audit(
        views, frame_ids, base, row_chunk=2
    )
    transport.validate_source_only_transport_audit(audit)
    deltas = [
        row["mean_delta_cosine_vs_transport_0p15"]
        for row in audit["candidates"]
    ]
    assert deltas[0] == 0.0
    assert deltas == sorted(deltas)
    assert deltas[-1] > 0.0
    assert audit["heldout_scale_observations"] == rows * 4 * 3


def test_transport_contract_is_hash_bound_and_rejects_old_selector_authority() -> None:
    contract = transport.transport_contract()
    assert contract["same_orthogonal_map_for_all_scales"] is True
    assert contract["post_rotation_renormalization"] is False
    assert contract["old_independent_scale_selector_directly_authorizes_transport"] is False
    assert transport.TRANSPORT_CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_transport_rejects_non_grid_ceiling() -> None:
    base = _random_frame(rows=1, dimension=7)
    teacher = F.normalize(torch.randn(1, 7), dim=-1)
    with pytest.raises(ValueError, match="outside the frozen grid"):
        transport.scale_equivariant_geodesic_transport(
            base,
            teacher,
            teacher_valid=torch.ones(1, dtype=torch.bool),
            retained_view_count=torch.ones(1, dtype=torch.uint8),
            teacher_view_directional_resultant=torch.ones(1),
            maximum_angle_radians=0.5,
        )
