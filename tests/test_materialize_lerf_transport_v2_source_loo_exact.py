from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from radio_gs.scripts import materialize_lerf_transport_v2_source_loo_exact as exact
from radio_gs.scripts import (
    materialize_lerf_transport_v2_source_loo_streaming_hook as hook,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def test_memory_preflight_is_bounded_below_existing_ramen_top4() -> None:
    audit = exact.host_memory_preflight(221_608)
    assert audit["maximum_heldout_scale_observations"] == 2_659_296
    assert audit["existing_lowmem_top4_fp16_bytes"] == 2_723_119_104
    assert audit["transport_v2_scalar_matrix_bytes"] == 265_929_600
    assert audit["transport_v2_additional_host_bytes_upper_bound"] == 848_120_448
    assert audit["additional_below_existing_lowmem_top4_allocation"] is True
    with pytest.raises(ValueError, match="accepted-row"):
        exact.host_memory_preflight(0)


def test_method_contract_is_query_free_hash_bound_and_gpu1_only() -> None:
    contract = exact.method_contract()
    assert contract["source_view_count"] == 120
    assert contract["top4_descriptors_durable"] is False
    assert contract["query_embeddings_or_text_consumed"] is False
    assert contract["o0_query_score_cache_consumed"] is False
    assert contract["metric_execution_authorized"] is False
    assert contract["source_loo"]["candidate_count"] == 25
    assert contract["transport_v2_hook_contract_sha256"] == hook.HOOK_CONTRACT_SHA256
    assert contract["execution"]["physical_gpu"] == 1
    assert exact.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_materializer_ast_calls_hook_and_has_no_tensor_or_metric_writer() -> None:
    source = textwrap.dedent(inspect.getsource(exact.materialize))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert any(
        node.func.attr == "capture_source_only_transport_v2_loo" for node in calls
    )
    assert any(node.func.attr == "_project_view" for node in calls)
    assert not any(node.func.attr == "write_torch_noclobber" for node in calls)
    assert "positive_embeddings" not in source
    assert "negative_embeddings" not in source
    assert "query_scores" not in source


def test_runtime_device_fails_closed_on_wrong_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=1"):
        exact.validate_runtime_device()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert exact.validate_runtime_device() == torch.device("cuda:0")
