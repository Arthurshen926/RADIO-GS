from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F

from radio_gs.scripts import (
    materialize_lerf_transport_v2_analytic_source_loo_exact as exact,
)
from radio_gs.scripts import (
    materialize_lerf_transport_v2_analytic_source_loo_streaming_hook as hook,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


def test_analytic_memory_preflight_removes_full_sort_workspace() -> None:
    audit = exact.host_memory_preflight(221_608)
    assert audit["maximum_heldout_scale_observations"] == 2_659_296
    assert audit["existing_lowmem_top4_fp16_bytes"] == 2_723_119_104
    assert audit["transport_v2_scalar_matrix_bytes"] == 265_929_600
    assert audit["order_statistic_workspace_upper_bound_bytes"] == 265_929_600
    assert audit["transport_v2_additional_host_bytes_upper_bound"] == 582_190_848
    assert audit["additional_below_existing_lowmem_top4_allocation"] is True
    with pytest.raises(ValueError, match="accepted-row"):
        exact.host_memory_preflight(0)


def test_analytic_method_contract_is_query_free_hash_bound_and_gpu1_only() -> None:
    contract = exact.method_contract()
    assert contract["source_view_count"] == 120
    assert contract["top4_descriptors_durable"] is False
    assert contract["query_embeddings_or_text_consumed"] is False
    assert contract["o0_query_score_cache_consumed"] is False
    assert contract["metric_execution_authorized"] is False
    assert contract["source_loo"]["candidate_descriptor_materialization"] is False
    assert contract["transport_v2_hook_contract_sha256"] == hook.HOOK_CONTRACT_SHA256
    assert set(contract["run_modes"]) == {
        "equivalence_real_chunk",
        "source_loo",
    }
    assert contract["execution"]["physical_gpu"] == 1
    assert exact.METHOD_CONTRACT_SHA256 == canonical_json_sha256(contract)


def test_analytic_materializer_calls_both_versioned_paths_and_has_no_metric_writer() -> (
    None
):
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
    assert any(
        node.func.attr == "compare_analytic_and_sealed_on_source_chunk"
        for node in calls
    )
    assert any(node.func.attr == "_project_view" for node in calls)
    assert not any(node.func.attr == "write_torch_noclobber" for node in calls)
    assert "positive_embeddings" not in source
    assert "negative_embeddings" not in source
    assert "query_scores" not in source


def test_analytic_streaming_hook_is_scalar_only_and_validated() -> None:
    generator = torch.Generator().manual_seed(20260807)
    views = F.normalize(torch.randn(7, 4, 31, generator=generator), dim=-1)
    frame_ids = torch.arange(4, dtype=torch.int32)[None].expand(7, -1).clone()
    frame_ids[0, 2:] = -1
    views[frame_ids < 0] = 0
    base = F.normalize(torch.randn(7, 3, 31, generator=generator), dim=-1)
    capture = hook.capture_source_only_transport_v2_loo(
        scene_id="synthetic",
        top_descriptors=views,
        top_frame_ids=frame_ids,
        o0_descriptor_by_scale=base,
        row_chunk=3,
    )
    hook.validate_streaming_hook_capture(capture)
    assert capture["access_audit"]["candidate_descriptors_materialized"] is False
    assert capture["access_audit"]["source_view_descriptors_written"] is False
    assert capture["access_audit"]["target_metric_executed"] is False


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
