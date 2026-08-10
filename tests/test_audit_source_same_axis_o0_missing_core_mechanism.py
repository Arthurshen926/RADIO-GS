import torch

from radio_gs.interfaces.source_missing_core_conditional_utility import (
    MissingCoreConditionalUtility,
)
from radio_gs.scripts.audit_source_same_axis_o0_missing_core_mechanism import (
    FEATURE_NAMES,
    _average_precision,
    _auc,
    build_unit_feature_table,
)


def _utility() -> MissingCoreConditionalUtility:
    return MissingCoreConditionalUtility(
        valid_core_counts=torch.tensor([4, 4]),
        positive_fraction=torch.tensor([0.75, 0.75]),
        qualified_region_mask=torch.tensor([True, True]),
        missing_counts=torch.tensor([1, 1]),
        unit_region_indices=torch.tensor([0, 1]),
        unit_query_indices=torch.tensor([0, 1]),
        unit_primitive_rows=torch.tensor([3, 4]),
        unit_o0_scores=torch.tensor([0.5, 0.4]),
        unit_hard_labels=torch.tensor([True, False]),
        unit_soft_target_mass_fraction=torch.tensor([0.8, 0.2]),
        unit_signed_utility=torch.tensor([0.6, -0.6]),
    )


def test_build_unit_feature_table_has_frozen_axis_and_no_labels():
    rows = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    mask = torch.ones_like(rows, dtype=torch.bool)
    xyz = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    features = build_unit_feature_table(
        utility=_utility(),
        o0_scores=torch.tensor(
            [
                [0.9, 0.2],
                [0.8, 0.8],
                [0.7, 0.7],
                [0.5, 0.9],
                [0.1, 0.4],
            ]
        ),
        primitive_valid_mask=torch.ones(5, dtype=torch.bool),
        region_query_indices=torch.tensor([0, 1]),
        region_rows=rows,
        token_mask=mask,
        xyz=xyz,
        region_scale_indices=torch.tensor([0, 2]),
        selected_query_scale_indices=torch.tensor([1, 2]),
        appearance_concentration=torch.tensor([0.8, 0.6]),
        boundary_concentration=torch.tensor([0.7, 0.4]),
        raw_full_scalar_summary=torch.arange(36, dtype=torch.float32).reshape(2, 18),
        full_scalar_eligible=torch.tensor([True, True]),
    )
    assert features.shape == (2, len(FEATURE_NAMES))
    assert torch.isfinite(features).all()
    assert torch.allclose(features[:, 0], torch.tensor([0.5, 0.4]))
    assert features[:, 8].tolist() == [0.0, 2.0]
    assert features[:, 9].tolist() == [1.0, 2.0]
    assert torch.all((features[:, 4] >= 0.0) & (features[:, 4] <= 1.0))


def test_auc_handles_orientation_and_ties():
    labels = torch.tensor([False, False, True, True])
    assert _auc(torch.tensor([0.0, 1.0, 2.0, 3.0]), labels) == 1.0
    assert _auc(torch.tensor([3.0, 2.0, 1.0, 0.0]), labels) == 0.0
    assert _auc(torch.ones(4), labels) == 0.5
    assert _average_precision(torch.tensor([0.0, 1.0, 2.0, 3.0]), labels) == 1.0
