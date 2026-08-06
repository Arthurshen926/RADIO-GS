from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.audit_factorized_capability_materialization import (
    EXPERIMENT,
    _validate_registration_identity,
    audit_full_fp16_parity,
    compare_materialized_rows,
)


def test_preregistration_identity_is_explicit_and_exact() -> None:
    assert _validate_registration_identity(
        {"registration": EXPERIMENT}, EXPERIMENT
    ) == EXPERIMENT
    leaves = "canonical_factorized_radio_v1_leaves_capability_materialization"
    assert _validate_registration_identity({"registration": leaves}, leaves) == leaves
    with pytest.raises(ValueError, match="preregistration differs"):
        _validate_registration_identity({"registration": leaves}, EXPERIMENT)
    with pytest.raises(ValueError, match="preregistration differs"):
        _validate_registration_identity({"registration": leaves}, "")


class _Field(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "values",
            torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]], dtype=torch.float32
            ),
        )

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        return self.values[rows]


class _Adaptor(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (values[:, 0] + values[:, 1], values[:, 0] - values[:, 1]), dim=-1
        )


def test_materialized_comparison_reports_variation_and_cosine() -> None:
    exact = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    result = compare_materialized_rows(exact.half(), exact, batch_size=2)
    assert result["cosine_mean"] == pytest.approx(1.0)
    assert result["cosine_p05"] == pytest.approx(1.0)
    assert result["centered_row_variation_ratio_to_exact"] == pytest.approx(1.0)


def test_full_materialization_parity_is_exact_fp16() -> None:
    field = _Field()
    adaptor = _Adaptor()
    rows = torch.tensor([0, 2])
    stored = torch.nn.functional.normalize(
        adaptor(field.radio_features(rows)).float(), dim=-1
    ).half()
    passed = audit_full_fp16_parity(
        field=field,
        adaptor=adaptor,
        global_rows=rows,
        stored=stored,
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert passed["exact_fp16_parity"] is True
    assert passed["unequal_fp16_values"] == 0

    corrupted = stored.clone()
    corrupted[0, 0] += 0.125
    failed = audit_full_fp16_parity(
        field=field,
        adaptor=adaptor,
        global_rows=rows,
        stored=corrupted,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert failed["exact_fp16_parity"] is False
    assert failed["unequal_fp16_values"] == 1
