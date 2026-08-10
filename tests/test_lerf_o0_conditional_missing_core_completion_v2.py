from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from radio_gs.interfaces import lerf_o0_conditional_missing_core_completion_v2 as formal
from radio_gs.interfaces.lerf_o0_conditional_missing_core_completion import (
    ConditionalMissingCoreCompletionResult,
)
from radio_gs.utils.immutable_artifacts import file_record


def _result() -> ConditionalMissingCoreCompletionResult:
    return ConditionalMissingCoreCompletionResult(
        valid_core_counts=torch.tensor([2]),
        positive_fraction=torch.tensor([[1.0]]),
        qualified_anchor_mask=torch.tensor([[True]]),
        unit_region_indices=torch.tensor([0]),
        unit_query_indices=torch.tensor([0]),
        unit_primitive_rows=torch.tensor([1]),
        selector_probability=torch.tensor([0.8]),
        selected_unit_mask=torch.tensor([True]),
        unconditional_cell_mask=torch.tensor([[False], [True]]),
        selected_cell_mask=torch.tensor([[False], [True]]),
        completion_probability=torch.tensor([[0.0], [1.0]]),
        final_scores=torch.tensor([[0.8], [1.0]]),
        changed_mask=torch.tensor([[False], [True]]),
    )


def _payload(tmp_path: Path) -> dict:
    token = tmp_path / "record"
    token.write_text("bound", encoding="utf-8")
    record = file_record(token)
    inputs = {
        name: record
        for name in (*formal.TARGET_INPUT_NAMES, *formal.SOURCE_INPUT_NAMES)
    }
    return formal.build_external_query_score_cache(
        result=_result(),
        o0_valid=torch.tensor([True, True]),
        o0_xyz=torch.zeros(2, 3),
        query_names=["query"],
        scene_id="figurines",
        input_authority=inputs,
        threshold_inclusive=0.731,
        threshold_source=inputs["multisource_selector_model"],
    )


def test_fix6c_cache_has_independent_v2_contract_and_dynamic_threshold(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    assert payload["schema"] == formal.EXTERNAL_CACHE_SCHEMA
    assert payload["schema"].endswith(".v2")
    assert payload["contract"]["schema"] == formal.SCHEMA
    assert payload["metadata"]["frozen_threshold_inclusive"] == 0.731
    assert "scene0003_PASS" in payload["metadata"]["score_semantics"]


def test_fix6c_cache_rejects_v1_contract_and_incomplete_metadata(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    changed = copy.deepcopy(payload)
    changed["schema"] = (
        "radio_gs.lerf_o0_conditional_missing_core_completion_external_scores.v1"
    )
    with pytest.raises(ValueError, match="contract"):
        formal.validate_external_query_score_cache(changed)
    changed = copy.deepcopy(payload)
    del changed["metadata"]["threshold_source"]
    with pytest.raises(ValueError, match="contract|metadata"):
        formal.validate_external_query_score_cache(changed)


def test_fix6c_cache_rejects_counter_or_source_record_drift(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    changed = copy.deepcopy(payload)
    changed["metadata"]["strictly_changed_cells"] = 0
    with pytest.raises(ValueError, match="metadata"):
        formal.validate_external_query_score_cache(changed)
    changed = copy.deepcopy(payload)
    changed["metadata"]["input_authority"].pop("scene0003_pass_report")
    with pytest.raises(ValueError, match="metadata"):
        formal.validate_external_query_score_cache(changed)

