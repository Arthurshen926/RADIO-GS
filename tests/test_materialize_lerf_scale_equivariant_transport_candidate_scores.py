from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import (
    materialize_lerf_scale_equivariant_transport_candidate_scores as candidate,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def _fixture() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(8042)
    accepted, dimension, global_count = 8, 17, 13
    base = F.normalize(
        torch.randn(accepted, 3, dimension, generator=generator), dim=-1
    ).half()
    teacher = F.normalize(
        torch.randn(accepted, dimension, generator=generator), dim=-1
    ).half()
    count = torch.tensor([4, 3, 2, 1, 0, 4, 3, 2], dtype=torch.uint8)
    valid = count > 0
    teacher[~valid] = 0
    agreement = torch.tensor([1.0, 0.8, 0.6, 1.0, 0.0, 0.9, 0.7, 0.5])
    return {
        "base_features_by_scale": base,
        "global_rows": torch.tensor([0, 1, 3, 4, 6, 8, 10, 12]),
        "teacher_mean": teacher,
        "teacher_valid": valid,
        "retained_view_count": count,
        "directional_resultant": agreement,
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


def test_lowmem_chunking_is_dense_equivalent_and_invariants_are_small() -> None:
    fixture = _fixture()
    dense = candidate.materialize_scores_lowmem(
        **fixture,
        global_ceiling_radians=0.75,
        device=torch.device("cpu"),
        row_batch_size=8,
    )
    chunked = candidate.materialize_scores_lowmem(
        **fixture,
        global_ceiling_radians=0.75,
        device=torch.device("cpu"),
        row_batch_size=3,
    )
    assert torch.equal(chunked["positive_scores"], dense["positive_scores"])
    assert torch.equal(chunked["negative_scores"], dense["negative_scores"])
    assert chunked["descriptor_sha256"] == dense["descriptor_sha256"]
    assert chunked["rows_with_score_replacement"] == 7
    assert chunked["scales_with_score_replacement"] == 21
    assert chunked["maximum_batch_rows_observed"] == 3
    assert chunked["maximum_per_scale_norm_absolute_error"] < 2e-6
    assert chunked["maximum_scale_gram_absolute_error"] < 3e-6


def test_invalid_same_and_antipodal_rows_keep_bitwise_o0_scores() -> None:
    base = torch.zeros(3, 3, 7)
    base[..., 0] = 1.0
    teacher = torch.zeros(3, 7)
    teacher[1, 0] = 1.0
    teacher[2, 0] = -1.0
    generator = torch.Generator().manual_seed(4)
    positive = torch.randn(3, 3, 2, generator=generator)
    negative = torch.randn(3, 3, 2, generator=generator)
    actual = candidate.materialize_scores_lowmem(
        base_features_by_scale=base,
        global_rows=torch.arange(3),
        teacher_mean=teacher,
        teacher_valid=torch.tensor([False, True, True]),
        retained_view_count=torch.tensor([0, 4, 4], dtype=torch.uint8),
        directional_resultant=torch.tensor([0.0, 1.0, 1.0]),
        positive_embeddings=F.normalize(
            torch.randn(2, 7, generator=generator), dim=-1
        ),
        negative_embeddings=F.normalize(
            torch.randn(2, 7, generator=generator), dim=-1
        ),
        o0_positive_scores=positive,
        o0_negative_scores=negative,
        global_ceiling_radians=0.75,
        device=torch.device("cpu"),
        row_batch_size=2,
    )
    assert torch.equal(actual["positive_scores"], positive)
    assert torch.equal(actual["negative_scores"], negative)
    assert actual["rows_with_score_replacement"] == 0


def test_runtime_device_is_exactly_bound() -> None:
    execution = {
        "physical_gpu": 1,
        "cuda_visible_devices": "1",
        "program_device": "cuda:0",
    }
    assert candidate.validate_runtime_device(
        execution,
        environ={"CUDA_VISIBLE_DEVICES": "1"},
        cuda_available=True,
    ) == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="device authority differs"):
        candidate.validate_runtime_device(
            execution,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            cuda_available=True,
        )


def test_lowmem_ast_never_allocates_full_candidate_descriptor() -> None:
    source = textwrap.dedent(inspect.getsource(candidate.materialize_scores_lowmem))
    tree = ast.parse(source)
    assert "base[start:stop]" in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cat"
        for node in ast.walk(tree)
    )
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"empty", "zeros", "ones", "full"}:
            continue
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        if "n_rows" in names and "SCALE_COUNT" in names:
            forbidden.append(node)
    assert forbidden == []


def test_development_contract_is_explicit_and_hash_bound() -> None:
    contract = candidate.method_contract()
    assert contract["old_selector_authorizes_transport"] is False
    assert contract["metric_execution_authorized"] is False
    assert contract["full_n_by_3_by_d_candidate_descriptor_allocated"] is False
    assert candidate.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)
