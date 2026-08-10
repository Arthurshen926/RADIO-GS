from __future__ import annotations

import torch

from radio_gs.scripts import build_lerf_o0_anchored_positive_utility_residual_cache_fix4c as builder


def _result(
    *, base: torch.Tensor, residual: torch.Tensor, query_gate: torch.Tensor
) -> builder.positive.O0AnchoredPositiveUtilityResidualResult:
    logits = torch.logit(
        base.clamp(builder.fix4b.O0_LOGIT_CLAMP, 1.0 - builder.fix4b.O0_LOGIT_CLAMP)
    )
    logits = logits + residual
    query_count = int(base.shape[1])
    return builder.positive.O0AnchoredPositiveUtilityResidualResult(
        fused_logits=logits,
        residual_logits=residual,
        query_gate=query_gate,
        selected_region_rows=tuple(() for _ in range(query_count)),
        selected_canonical_region_indices=tuple(() for _ in range(query_count)),
        selected_lower_scores=tuple(() for _ in range(query_count)),
        selected_gains=tuple(() for _ in range(query_count)),
        selected_marginal_primitives=tuple(() for _ in range(query_count)),
    )


def test_probability_one_endpoint_stays_bitwise_one_and_is_not_changed() -> None:
    base = torch.tensor([[1.0], [0.4]], dtype=torch.float32)
    residual = torch.tensor([[0.2], [0.2]], dtype=torch.float32)
    result = _result(
        base=base, residual=residual, query_gate=torch.tensor([True])
    )
    fused, changed, audit = builder.fuse_exact_o0_probabilities_monotone(
        base, result
    )
    assert fused[0, 0].view(torch.int32) == base[0, 0].view(torch.int32)
    assert not bool(changed[0, 0])
    assert fused[1, 0] > base[1, 0]
    assert bool(changed[1, 0])
    assert audit["residual_mask_primitive_total"] == 2
    assert audit["actual_changed_primitive_total"] == 1
    assert audit["quantized_no_change_primitive_total"] == 1
    assert audit["selected_updates_non_decreasing"] is True
    assert audit["actual_changes_strictly_increase_exact_O0"] is True


def test_failed_gate_and_zero_residual_entries_remain_bitwise_o0() -> None:
    base = torch.tensor(
        [[-0.0, 1.0], [0.3, 1.0 - torch.finfo(torch.float32).eps]],
        dtype=torch.float32,
    )
    residual = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    result = _result(
        base=base, residual=residual, query_gate=torch.tensor([True, False])
    )
    fused, changed, _ = builder.fuse_exact_o0_probabilities_monotone(base, result)
    assert torch.equal(fused[:, 1].view(torch.int32), base[:, 1].view(torch.int32))
    assert torch.equal(fused[~changed].view(torch.int32), base[~changed].view(torch.int32))
    assert fused[0, 0].view(torch.int32) == base[0, 0].view(torch.int32)


def test_cli_exposes_no_method_or_target_quality_controls() -> None:
    destinations = {action.dest for action in builder.build_parser()._actions}
    assert destinations == {
        "help",
        "execution_authority",
        "expected_execution_authority_sha256",
    }

