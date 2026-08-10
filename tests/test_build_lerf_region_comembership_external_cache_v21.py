from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest
import torch

from radio_gs.scripts import (
    build_lerf_region_comembership_external_cache_v21 as builder,
)
from radio_gs.utils.immutable_artifacts import file_record, sha256_file


def test_v21_relevance_gates_greedy_novelty_without_graph_hard_edges() -> None:
    feature = {
        "scene_id": "figurines",
        "region_fingerprints": ["a", "b", "c", "d"],
        "canonical_region_indices": torch.arange(4),
        "pair_indices": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "region_rows": torch.tensor([[0, 1], [0, 1], [2, 3], [4, 5]]),
        "token_mask": torch.ones(4, 2, dtype=torch.bool),
    }
    inference = {
        "selected_rule": {
            "method": "maximum_product",
            "maximum_regions": 2,
            "threshold": 0.75,
        },
        "pair_probabilities": torch.tensor([0.9, 0.9, 0.9]),
    }
    relevance = {
        "scene_id": "figurines",
        "region_fingerprints": feature["region_fingerprints"],
        "canonical_region_indices": feature["canonical_region_indices"],
        "region_absolute_relevance": torch.tensor(
            [[0.90, 0.90], [0.89, 0.80], [0.49, 0.70], [0.95, 0.10]]
        ),
    }
    result = builder.greedy_novelty_readout_from_v21(
        feature=feature,
        inference=inference,
        relevance=relevance,
        num_primitives=8,
    )
    # Region 2 is below the exact 0.5 boundary for q0.  Region 1 has zero
    # novelty after region 0 and therefore cannot consume another set slot.
    assert result.selected_region_indices == ((3, 0), (0, 2))
    assert result.selected_marginal_core_rows == ((2, 2), (2, 2))


def test_valid_mask_clips_union_output_without_rescoring() -> None:
    membership = torch.tensor([[1.0], [1.0], [0.0], [1.0]])
    valid = torch.tensor([True, False, True, False])
    clipped, removed = builder.mask_union_to_valid(membership, valid)
    assert removed == 2
    assert clipped[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_cli_exposes_no_threshold_smoothing_or_scale_override() -> None:
    parser = builder.build_parser()
    assert {action.dest for action in parser._actions} == {
        "help",
        "execution_authority",
        "expected_execution_authority_sha256",
    }
    assert (
        builder.external_cache_access_audit()["legacy_o0_query_scores_opened"] is False
    )
    assert set(
        inspect.signature(builder.greedy_novelty_readout_from_v21).parameters
    ) == {
        "feature",
        "inference",
        "relevance",
        "num_primitives",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--execution-authority",
                "authority.json",
                "--expected-execution-authority-sha256",
                "a" * 64,
                "--knn",
                "10",
            ]
        )


def _execution(tmp_path: Path) -> Path:
    artifacts: dict[str, dict[str, str]] = {}
    names = (
        "v21_checkpoint",
        "v21_normalization",
        "canonical_negative_bank",
        "target_descriptor",
        "positive_text_cache",
        "query_relevance_execution_authority",
        "query_relevance_authority",
        "comembership_feature_authority",
        "comembership_inference_authority",
        "renderer_geometry_checkpoint",
        "preregistration",
    )
    for name in names:
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode("utf-8"))
        artifacts[name] = file_record(path)
    authority = {
        "schema": builder.EXECUTION_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": builder.EXECUTION_AUTHORITY_STATUS,
        "scene_id": "figurines",
        "physical_space_id": "lerf:figurines:geometry-checkpoint-sha256:" + "9" * 64,
        "source_pilot_result": {"path": "/source/result.json", "sha256": "8" * 64},
        **artifacts,
        "implementation": file_record(Path(builder.__file__).resolve()),
        "implementation_dependencies": {
            name: file_record(path)
            for name, path in builder.IMPLEMENTATION_DEPENDENCIES.items()
        },
        "output_cache": str((tmp_path / "external.pt").resolve()),
        "output_report": str((tmp_path / "external.json").resolve()),
        "query_readout_authorized": True,
        "target_metric_authorized": False,
        "access_audit": builder.external_cache_access_audit(),
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    return path


def test_external_gate_rejects_source_before_any_target_query_or_renderer_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execution = _execution(tmp_path)
    opened = {"files": 0}

    def reject(*args, **kwargs):
        raise ValueError("source promotion rejected")

    def opened_file(*args, **kwargs):
        opened["files"] += 1
        raise AssertionError("opened target/query/renderer file before source PASS")

    monkeypatch.setattr(builder, "validate_source_pilot_chain", reject)
    monkeypatch.setattr(builder, "validate_file_record", opened_file)
    with pytest.raises(ValueError, match="source promotion rejected"):
        builder.validate_external_execution_authority(
            execution, expected_sha256=sha256_file(execution)
        )
    assert opened["files"] == 0


def test_external_gate_binds_promoted_checkpoint_before_target_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execution = _execution(tmp_path)
    authority = json.loads(execution.read_text(encoding="utf-8"))
    source_execution = tmp_path / "source_execution.json"
    source_execution.write_text(
        json.dumps({"canonical_negative_bank": authority["canonical_negative_bank"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder,
        "validate_source_pilot_chain",
        lambda *args, **kwargs: {
            "source_promotion_authorized": True,
            "execution_authority": file_record(source_execution),
            "checkpoint": {"path": "/source/other.pt", "sha256": "7" * 64},
            "normalization_authority": authority["v21_normalization"],
        },
    )
    monkeypatch.setattr(
        builder,
        "validate_file_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("target/query file opened before checkpoint binding")
        ),
    )
    with pytest.raises(ValueError, match="model/calibration binding"):
        builder.validate_external_execution_authority(
            execution, expected_sha256=sha256_file(execution)
        )
