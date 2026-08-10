from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as v2,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2_lowmem as lowmem,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
)


def _fixture(rows: int = 9) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(8207)
    descriptors = F.normalize(
        torch.randn(rows, 4, 1536, generator=generator), dim=-1
    ).half()
    frame_ids = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4)
    frame_ids[1, 3] = -1
    frame_ids[2, 2:] = -1
    frame_ids[3, 1:] = -1
    frame_ids[-1] = -1
    descriptors[frame_ids < 0] = 0
    base = F.normalize(
        torch.randn(rows, 3, 1536, generator=generator), dim=-1
    ).half()
    return descriptors, frame_ids, base


def _dense_teacher_mean(
    descriptors: torch.Tensor, frame_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    valid = counts > 0
    mean = F.normalize(
        (descriptors.float() * mask[:, :, None]).sum(dim=1), dim=-1
    )
    mean[~valid] = 0
    return mean.half().contiguous(), valid.contiguous(), counts.to(torch.uint8)


def test_lowmem_teacher_mean_is_bitwise_dense_v2_equivalent() -> None:
    descriptors, frame_ids, _ = _fixture(13)
    expected = _dense_teacher_mean(descriptors, frame_ids)
    actual = lowmem._chunked_teacher_mean(
        descriptors, frame_ids, chunk_rows=3
    )
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)


def test_lowmem_agreement_and_loo_payload_are_exact_v2_equivalent() -> None:
    descriptors, frame_ids, base = _fixture()
    original_descriptors = descriptors.clone()
    original_base = base.clone()
    expected_agreement, expected_counts = (
        v2.directional_resultant_from_canonical_top_views(
            descriptors, frame_ids
        )
    )
    expected_loo = v2.source_only_leave_one_view_out_ceiling_audit(
        descriptors, frame_ids, base
    )
    actual = lowmem.finalize_teacher_statistics_lowmem(
        descriptors, frame_ids, base, chunk_rows=2
    )
    dense_mean, dense_valid, dense_counts = _dense_teacher_mean(
        descriptors, frame_ids
    )
    assert torch.equal(actual["teacher_mean"], dense_mean)
    assert torch.equal(actual["teacher_valid"], dense_valid)
    assert torch.equal(actual["retained_view_count"], dense_counts)
    assert torch.equal(actual["retained_view_count"], expected_counts)
    assert torch.equal(
        actual[v2.VIEW_AGREEMENT_SCALAR], expected_agreement
    )
    assert actual[v2.LOO_AUDIT_FIELD] == expected_loo
    assert torch.equal(descriptors, original_descriptors)
    assert torch.equal(base, original_base)


def test_teacher_mean_promotion_is_bounded_by_row_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors, frame_ids, _ = _fixture(11)
    leading_rows: list[int] = []
    original = lowmem.F.normalize

    def recording(value: torch.Tensor, *args: object, **kwargs: object):
        leading_rows.append(int(value.shape[0]))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(lowmem.F, "normalize", recording)
    lowmem._chunked_teacher_mean(descriptors, frame_ids, chunk_rows=3)
    assert leading_rows == [3, 3, 3, 2]


def test_contract_binds_every_upstream_and_keeps_v2_method_closed() -> None:
    contract = lowmem.method_contract()
    assert contract["streaming_entrypoint_implementation"] == (
        lowmem.ENTRYPOINT_IMPLEMENTATION
    )
    assert contract["streaming_core_implementation"] == (
        lowmem.CORE_IMPLEMENTATION
    )
    assert contract["teacher_agreement_v2_numerical_implementation"] == (
        lowmem.TEACHER_AGREEMENT_V2_IMPLEMENTATION
    )
    assert contract["lowmem_allocation_reference_implementation"] == (
        lowmem.LOWMEM_ALLOCATION_REFERENCE_IMPLEMENTATION
    )
    for record in (
        lowmem.ENTRYPOINT_IMPLEMENTATION,
        lowmem.CORE_IMPLEMENTATION,
        lowmem.TEACHER_AGREEMENT_V2_IMPLEMENTATION,
        lowmem.LOWMEM_ALLOCATION_REFERENCE_IMPLEMENTATION,
    ):
        assert record == file_record(Path(record["path"]))
    assert contract["teacher_mean_chunking_affects_method_numerics"] is False
    assert contract["agreement_and_loo_implementation_reused_without_change"] is True
    assert contract["O1"] == v2.method_contract()["O1"]
    assert contract["O2"] == v2.method_contract()["O2"]
    assert contract["reliability_budget_maximum_angle_authorized"] is False
    assert contract["execution_device_authority"] == {
        "implemented_physical_gpu": 0,
        "required_cuda_visible_devices": "0",
        "program_device": "cuda:0",
        "other_physical_gpu_authorized": False,
    }
    assert lowmem.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_runtime_device_and_contract_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="requires CUDA_VISIBLE_DEVICES=0"):
        lowmem.materialize(object())
    monkeypatch.setattr(lowmem, "METHOD_CONTRACT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="contract differs"):
        lowmem.materialize(object())


def test_projection_and_thermal_execution_contract_is_unpaced_external_guard() -> None:
    contract = lowmem.method_contract()
    assert contract["projection_pacing_seconds_per_batch"] == 0.0
    assert contract["thermal_safety_owner"] == "external_300s_hard88_guard"
    tree = ast.parse(textwrap.dedent(inspect.getsource(lowmem.materialize)))
    projection_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_CORE_PROJECT_VIEW"
    ]
    assert len(projection_calls) == 2
    for call in projection_calls:
        pace = next(
            keyword.value for keyword in call.keywords if keyword.arg == "pace"
        )
        assert isinstance(pace, ast.Constant)
        assert pace.value is False


def test_unretained_nonzero_teacher_fails_closed() -> None:
    descriptors, frame_ids, _ = _fixture()
    descriptors[-1, 0, 0] = 1.0
    with pytest.raises(ValueError, match="exact zero"):
        lowmem._chunked_teacher_mean(descriptors, frame_ids, chunk_rows=2)
