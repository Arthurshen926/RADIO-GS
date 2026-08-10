from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces import factorized_native_lerf_exact as formal
from radio_gs.models.factorized_native_gauge_state_readout import (
    FACTORIZED_NATIVE_READOUT_ARMS,
)
from radio_gs.scripts import run_factorized_native_lerf_exact_pipeline as pipeline


def _record(character: str) -> dict[str, str]:
    return {"path": f"/tmp/{character}", "sha256": character * 64}


def _query_payload() -> dict:
    descriptor = F.normalize(torch.randn(3, 1536), dim=-1)
    del descriptor  # The relevance payload deliberately contains no descriptor copy.
    relevance = torch.tensor(
        [[0.1, 0.9], [0.5, 0.6], [0.7, 0.2]], dtype=torch.float32
    )
    payload = {
        "schema": formal.QUERY_RELEVANCE_SCHEMA,
        "schema_version": 1,
        "contract": formal.query_contract(),
        "contract_sha256": formal.QUERY_CONTRACT_SHA256,
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:physical",
        "producer": _record("a"),
        "query_execution_authority": _record("b"),
        "input_authority": {
            "target_descriptor": _record("c"),
            "descriptor_health_audit": _record("5"),
            "exact_query_manifest": _record("d"),
            "positive_text_cache": _record("e"),
            "all_query_text_cache": _record("f"),
            "canonical_negative_bank": _record("1"),
        },
        "region_row_ids": ["r0", "r1", "r2"],
        "canonical_region_indices": torch.tensor([0, 2, 5]),
        "region_fingerprints": ["2" * 64, "3" * 64, "4" * 64],
        "query_ids": ["bag", "waldo"],
        "region_absolute_relevance": relevance,
        "access_audit": formal.query_access_audit(),
    }
    payload["channel_sha256"] = formal.query_channel_sha256(payload)
    return payload


def test_query_contract_freezes_existing_exact_scorer_without_remap() -> None:
    contract = formal.query_contract()
    assert contract["scorer"] == "existing_calibrated_v21_absolute_relevance"
    assert contract["formula"] == "binary_softmax_positive_vs_max_canonical_negative"
    assert contract["logit_scale"] == 10.0
    assert contract["postprocess"] == "none"
    assert contract["query_smoothing"] is False
    assert contract["scene_minmax_remap"] is False
    assert formal.FROZEN_CANONICAL_NEGATIVE_BANK["sha256"] == (
        "18d2aac56b50a9670ffe04b397d23a4652dd44fe8f18ed7a309a82b6c1102b67"
    )


def test_query_relevance_is_independent_schema_and_strict() -> None:
    payload = _query_payload()
    checked = formal.validate_query_relevance(payload)
    assert checked["schema"] == formal.QUERY_RELEVANCE_SCHEMA
    assert checked["schema"] != "radio_gs.surface_region_rank256_champion_query_relevance.v1"
    tampered = dict(payload)
    tampered["region_absolute_relevance"] = torch.full((3, 2), 1.1)
    tampered["channel_sha256"] = formal.query_channel_sha256(tampered)
    with pytest.raises(ValueError, match="tensor differs"):
        formal.validate_query_relevance(tampered)


def test_query_authority_is_source_first(monkeypatch, tmp_path: Path) -> None:
    authority = {
        "schema": formal.QUERY_EXECUTION_SCHEMA,
        "schema_version": 1,
        "status": pipeline.QUERY_STATUS,
        "source_arm_results": {
            arm: _record(str(index + 2))
            for index, arm in enumerate(FACTORIZED_NATIVE_READOUT_ARMS)
        },
        "winner_arm": FACTORIZED_NATIVE_READOUT_ARMS[0],
        "scene_id": "figurines",
        "physical_space_id": "physical",
        "implementation": _record("5"),
        "implementation_dependencies": {
            "query_formal": _record("6"),
            **{name: _record("7") for name in formal.FROZEN_QUERY_DEPENDENCIES},
        },
        "preregistration": _record("8"),
        "target_descriptor": _record("9"),
        "descriptor_health_audit": _record("1"),
        "exact_query_manifest": _record("a"),
        "positive_text_cache": _record("b"),
        "all_query_text_cache": _record("c"),
        "canonical_negative_bank": _record("d"),
        "query_relevance_output": str(tmp_path / "relevance.pt"),
        "query_execution_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": formal.query_access_audit(),
    }
    opened: list[str] = []
    monkeypatch.setattr(
        pipeline, "load_json_object",
        lambda *args, **kwargs: (authority, "e" * 64, tmp_path / "authority.json"),
    )

    def source_stop(*args, **kwargs):
        opened.append("source")
        raise RuntimeError("source gate stop")

    def forbidden_open(*args, **kwargs):
        opened.append("query")
        raise AssertionError("query opened before source gate")

    monkeypatch.setattr(pipeline.target_formal, "validate_source_arm_winner", source_stop)
    monkeypatch.setattr(pipeline, "validate_file_record", forbidden_open)
    with pytest.raises(RuntimeError, match="source gate stop"):
        pipeline.validate_query_authority(
            tmp_path / "authority.json", expected_sha256="e" * 64
        )
    assert opened == ["source"]


def test_query_health_loader_requires_formal_pass(monkeypatch, tmp_path: Path) -> None:
    observed: list[bool] = []
    monkeypatch.setattr(
        pipeline, "load_json_object",
        lambda *args, **kwargs: ({}, "a" * 64, tmp_path / "health.json"),
    )

    def reject(_value, *, require_pass=False):
        observed.append(require_pass)
        raise ValueError("anti-collapse gate")

    monkeypatch.setattr(pipeline.health_formal, "validate_health_audit", reject)
    with pytest.raises(ValueError, match="anti-collapse gate"):
        pipeline._load_descriptor_health_gate(
            _record("a"),
            descriptor_record=_record("b"),
            descriptor={},
        )
    assert observed == [True]


def test_external_cache_matches_frozen_evaluator_input_contract() -> None:
    cache = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA,
        "schema_version": 1,
        "contract": formal.external_contract(),
        "contract_sha256": formal.EXTERNAL_CONTRACT_SHA256,
        "query_scores": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32
        ),
        "valid": torch.tensor([True, True, False]),
        "xyz": torch.zeros(3, 3),
        "metadata": {
            "query_names": ["bag", "waldo"],
            "score_semantics": "binary_factorized_native_absolute_relevance_greedy_novelty_union",
            "producer": _record("a"),
            "execution_authority": _record("b"),
        },
        "selection": {
            "region_indices": [[0], [1]],
            "region_scores": [[0.8], [0.9]],
            "marginal_core_rows": [[1], [1]],
            "invalid_memberships_removed": 0,
        },
    }
    checked = formal.validate_external_cache(cache)
    assert checked["query_scores"].shape == (3, 2)
    assert checked["metadata"]["query_names"] == ["bag", "waldo"]
    assert formal.METRIC_PROTOCOL["protocol_preset"] == "vala_paper_3d"
    assert formal.METRIC_PROTOCOL["score_threshold"] == 0.6


def test_frozen_tuple_selection_is_canonicalized_without_reordering() -> None:
    readout = SimpleNamespace(
        selected_region_indices=((), (7, 2), (5,)),
        selected_region_scores=((), (0.7, 0.6), (0.9,)),
        selected_marginal_core_rows=((), (21, 13), (8,)),
    )
    selection = pipeline.canonical_external_selection(
        readout, expected_query_count=3
    )
    assert selection == {
        "region_indices": [[], [7, 2], [5]],
        "region_scores": [[], [0.7, 0.6], [0.9]],
        "marginal_core_rows": [[], [21, 13], [8]],
    }
    # This is serialization-only: query 1 retains selected-region order 7,2.
    assert selection["region_indices"][1] == [7, 2]


def test_external_validator_rejects_unserialized_frozen_tuple() -> None:
    cache = {
        "schema": formal.EXTERNAL_CACHE_SCHEMA,
        "schema_version": 1,
        "contract": formal.external_contract(),
        "contract_sha256": formal.EXTERNAL_CONTRACT_SHA256,
        "query_scores": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        "valid": torch.tensor([True, False]),
        "xyz": torch.zeros(2, 3),
        "metadata": {
            "query_names": ["bag"],
            "score_semantics": "binary_factorized_native_absolute_relevance_greedy_novelty_union",
            "producer": _record("a"),
            "execution_authority": _record("b"),
        },
        "selection": {
            "region_indices": ((7,),),
            "region_scores": ((0.7,),),
            "marginal_core_rows": ((21,),),
            "invalid_memberships_removed": 0,
        },
    }
    with pytest.raises(ValueError, match="query order differs"):
        formal.validate_external_cache(cache)


def test_run_metric_invokes_only_frozen_protocol(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    authority = {
        "output_dir": str(output),
        "scene_id": "figurines",
        "label_root": "/labels",
        "external_cache": _record("a"),
        "frozen_inputs": {
            "config": _record("b"),
            "renderer_geometry_checkpoint": _record("c"),
            "summary_head": _record("d"),
            "all_query_text_cache": formal.FROZEN_ALL_QUERY_CACHE,
            "canonical_negative_text_cache": formal.FROZEN_CANONICAL_NEGATIVE_BANK,
        },
        "verified_record": _record("e"),
    }
    monkeypatch.setattr(pipeline, "validate_metric_authority", lambda *a, **k: authority)
    observed: list[str] = []

    def fake_run(command, check):
        assert check is True
        observed.extend(command)
        result = output / "figurines" / "lerf_direct_3d_selection_results.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = pipeline.run_metric(
        argparse.Namespace(
            execution_authority="/authority.json",
            expected_execution_authority_sha256="f" * 64,
            gpu=1,
        )
    )
    assert result["status"] == "factorized_native_single_frozen_lerf_metric_complete"
    assert observed[1] == formal.FROZEN_EVALUATOR["path"]
    assert observed[observed.index("--protocol_preset") + 1] == "vala_paper_3d"
    assert observed[observed.index("--external_query_score_cache") + 1] == "/tmp/a"
    assert observed[observed.index("--gpu") + 1] == "1"


def test_figurines_build_query_cli_has_all_three_source_arms() -> None:
    args = ["build-query-authority"]
    for index, arm in enumerate(FACTORIZED_NATIVE_READOUT_ARMS):
        option = arm.replace("_", "-")
        args.extend(
            [f"--{option}-result", f"/tmp/result{index}.json",
             f"--{option}-result-sha256", str(index + 1) * 64]
        )
    args.extend(
        [
            "--target-descriptor", "/tmp/descriptor.pt",
            "--descriptor-health-audit", "/tmp/descriptor.health.json",
            "--exact-query-manifest", "/tmp/figurines.json",
            "--positive-text-cache", "/tmp/positive.pt",
            "--query-relevance-output", "/tmp/relevance.pt",
            "--output-authority", "/tmp/query.json",
        ]
    )
    parsed = pipeline.build_parser().parse_args(args)
    assert parsed.command == "build-query-authority"
    assert parsed.direction_plus_log_amplitude_plus_full_state_result == "/tmp/result2.json"
