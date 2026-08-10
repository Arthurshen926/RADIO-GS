from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import (
    materialize_lerf_o1_o2_streaming_unpaced_gpu1_lowmem as lowmem,
)


def _dense_core_expression(
    descriptors: torch.Tensor, frame_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    valid = counts > 0
    mean = F.normalize(
        (descriptors.float() * mask[:, :, None]).sum(dim=1), dim=-1
    )
    mean[~valid] = 0
    return mean.half().contiguous(), valid, counts.to(torch.uint8)


def test_chunked_teacher_mean_is_bitwise_core_equivalent() -> None:
    generator = torch.Generator().manual_seed(1307)
    descriptors = torch.randn(
        19, 4, 37, generator=generator, dtype=torch.float16
    )
    frame_ids = torch.arange(4, dtype=torch.int32).repeat(19, 1)
    frame_ids[3, 2:] = -1
    expected = _dense_core_expression(descriptors, frame_ids)
    actual = lowmem._chunked_teacher_mean(
        descriptors, frame_ids, chunk_rows=5
    )
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)


def test_chunked_teacher_mean_invalid_rows_are_exact_zero() -> None:
    descriptors = torch.full((3, 4, 7), 9, dtype=torch.float16)
    frame_ids = torch.tensor(
        [[-1, -1, -1, -1], [2, -1, -1, -1], [-1, -1, -1, -1]],
        dtype=torch.int32,
    )
    mean, valid, counts = lowmem._chunked_teacher_mean(
        descriptors, frame_ids, chunk_rows=2
    )
    assert torch.equal(valid, torch.tensor([False, True, False]))
    assert torch.equal(counts, torch.tensor([0, 1, 0], dtype=torch.uint8))
    assert torch.count_nonzero(mean[[0, 2]]) == 0


def test_chunked_teacher_mean_never_normalizes_more_than_chunk_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = torch.ones(17, 4, 11, dtype=torch.float16)
    frame_ids = torch.zeros(17, 4, dtype=torch.int32)
    leading_dimensions: list[int] = []
    original = lowmem.F.normalize

    def recording_normalize(value: torch.Tensor, *args: object, **kwargs: object):
        leading_dimensions.append(int(value.shape[0]))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(lowmem.F, "normalize", recording_normalize)
    lowmem._chunked_teacher_mean(descriptors, frame_ids, chunk_rows=4)
    assert leading_dimensions == [4, 4, 4, 4, 1]


def test_lowmem_contract_binds_frozen_gpu1_entrypoint() -> None:
    contract = lowmem.method_contract()
    assert contract["gpu1_streaming_entrypoint"] == lowmem.GPU1_IMPLEMENTATION
    assert contract["gpu1_streaming_entrypoint"]["sha256"] == (
        "18cf7fa8871422cd807937433835e8ce36c89bed9b369a7f193534fd6e2e1a07"
    )
    assert contract["teacher_mean_chunking_affects_method_numerics"] is False


def test_lowmem_runtime_env_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="requires CUDA_VISIBLE_DEVICES=1"):
        lowmem.materialize(object())


def test_all_lowmem_projection_calls_are_explicitly_unpaced() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(lowmem.materialize)))
    projection_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_project_view"
    ]
    assert len(projection_calls) == 2
    for call in projection_calls:
        pace = next(keyword.value for keyword in call.keywords if keyword.arg == "pace")
        assert isinstance(pace, ast.Constant)
        assert pace.value is False
