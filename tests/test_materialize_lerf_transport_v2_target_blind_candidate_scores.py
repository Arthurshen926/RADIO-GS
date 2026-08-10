from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_scale_residual_shrinkage_transport import (
    scale_residual_shrinkage_transport,
)
from radio_gs.scripts import (
    materialize_lerf_transport_v2_target_blind_candidate_scores as candidate,
)
from radio_gs.scripts import (
    materialize_lerf_transport_v2_target_blind_top4_candidate_scores as top4,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _fixture() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260809)
    accepted, dimension, global_count = 7, 19, 12
    base = F.normalize(
        torch.randn(accepted, 3, dimension, generator=generator), dim=-1
    ).half()
    teacher = F.normalize(
        torch.randn(accepted, dimension, generator=generator), dim=-1
    ).half()
    count = torch.tensor([4, 3, 2, 1, 0, 4, 3], dtype=torch.uint8)
    valid = count > 0
    teacher[~valid] = 0
    return {
        "base_features_by_scale": base,
        "global_rows": torch.tensor([0, 2, 3, 5, 7, 9, 11], dtype=torch.int64),
        "teacher_mean": teacher,
        "teacher_valid": valid,
        "retained_view_count": count,
        "directional_resultant": torch.tensor(
            [1.0, 0.8, 0.65, 1.0, 0.0, 0.9, 0.55]
        ),
        "positive_embeddings": F.normalize(
            torch.randn(5, dimension, generator=generator), dim=-1
        ),
        "negative_embeddings": F.normalize(
            torch.randn(4, dimension, generator=generator), dim=-1
        ),
        "o0_positive_scores": torch.randn(
            global_count, 3, 5, generator=generator
        ),
        "o0_negative_scores": torch.randn(
            global_count, 3, 4, generator=generator
        ),
    }


def test_chunked_descriptor_is_exact_and_scores_match_sealed_transport() -> None:
    fixture = _fixture()
    dense = candidate.materialize_scores_lowmem(
        **fixture, device=torch.device("cpu"), row_batch_size=7
    )
    chunked = candidate.materialize_scores_lowmem(
        **fixture, device=torch.device("cpu"), row_batch_size=2
    )
    torch.testing.assert_close(
        chunked["positive_scores"], dense["positive_scores"], rtol=0.0, atol=1e-7
    )
    torch.testing.assert_close(
        chunked["negative_scores"], dense["negative_scores"], rtol=0.0, atol=1e-7
    )
    assert chunked["descriptor_sha256"] == dense["descriptor_sha256"]

    sealed = scale_residual_shrinkage_transport(
        fixture["base_features_by_scale"].float(),
        fixture["teacher_mean"].float(),
        teacher_valid=fixture["teacher_valid"],
        retained_view_count=fixture["retained_view_count"],
        teacher_view_directional_resultant=fixture["directional_resultant"],
        maximum_angle_radians=candidate.SELECTED_MAXIMUM_ANGLE_RADIANS,
        gamma_policy=candidate.SELECTED_GAMMA_POLICY,
    )
    unit = F.normalize(sealed.descriptor, dim=-1)
    expected_positive = torch.einsum(
        "bsd,qd->bsq", unit, F.normalize(fixture["positive_embeddings"], dim=-1)
    )
    expected_negative = torch.einsum(
        "bsd,qd->bsq", unit, F.normalize(fixture["negative_embeddings"], dim=-1)
    )
    rows = fixture["global_rows"]
    torch.testing.assert_close(
        chunked["positive_scores"][rows], expected_positive, rtol=0.0, atol=1e-7
    )
    torch.testing.assert_close(
        chunked["negative_scores"][rows], expected_negative, rtol=0.0, atol=1e-7
    )
    assert chunked["rows_with_teacher_applied"] == int(
        sealed.teacher_applied.any(dim=-1).sum()
    )
    assert chunked["rows_with_o0_descriptor_fallback"] == int(
        (~sealed.teacher_applied.any(dim=-1)).sum()
    )
    assert chunked["maximum_unit_norm_absolute_error"] < 2e-6


def test_only_validity_domain_rows_are_recomputed() -> None:
    fixture = _fixture()
    actual = candidate.materialize_scores_lowmem(
        **fixture, device=torch.device("cpu"), row_batch_size=3
    )
    accepted = fixture["global_rows"]
    outside = torch.ones(fixture["o0_positive_scores"].shape[0], dtype=torch.bool)
    outside[accepted] = False
    assert torch.equal(
        actual["positive_scores"][outside], fixture["o0_positive_scores"][outside]
    )
    assert torch.equal(
        actual["negative_scores"][outside], fixture["o0_negative_scores"][outside]
    )

    # Row 4 is an accepted descriptor-fallback row.  It must still be rebound to
    # the frozen text bank; only rows outside the accepted validity domain retain
    # the exact O0 score template.
    fallback_global_row = int(accepted[4])
    assert not torch.equal(
        actual["positive_scores"][fallback_global_row],
        fixture["o0_positive_scores"][fallback_global_row],
    )


def test_axes_fail_closed_and_runtime_device_is_exactly_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    fixture["global_rows"] = fixture["global_rows"].flip(0)
    with pytest.raises(ValueError, match="axes differ"):
        candidate.materialize_scores_lowmem(
            **fixture, device=torch.device("cpu"), row_batch_size=2
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert candidate.validate_runtime_device(0) == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="visibility differs"):
        candidate.validate_runtime_device(1)


def test_contracts_are_target_closed_hash_bound_and_top4_is_explicit_adapter() -> None:
    contract = candidate.method_contract()
    assert contract["selected_candidate_index"] == 11
    assert contract["scene_or_query_specific_parameters"] is False
    assert contract["query_independent_transport"] is True
    assert contract["target_data_or_metric_access"] is False
    assert contract["metric_execution_authorized"] is False
    assert contract["row_batch_size"] == 256
    assert candidate.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)

    top4_contract = top4.method_contract()
    assert top4_contract["teacher_payload"] == (
        "validated_canonical_top4_deterministic_adapter"
    )
    assert top4_contract["teacher_adapter_contract_sha256"] == (
        top4._teacher.CONTRACT_SHA256
    )
    assert top4_contract["target_data_or_metric_access"] is False
    assert top4.METHOD_CONTRACT_SHA256 == canonical_json_sha256(top4_contract)


def test_lowmem_path_never_materializes_a_full_candidate_descriptor() -> None:
    source = textwrap.dedent(inspect.getsource(candidate.materialize_scores_lowmem))
    tree = ast.parse(source)
    assert "base[start:stop]" in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cat"
        for node in ast.walk(tree)
    )
