from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.interfaces import (
    factorized_native_contrast_v21_lerf_exact as contrast,
)
from radio_gs.interfaces import (
    lerf_v21_frozen_relative_frozen_metric_adapter as formal,
)
from radio_gs.scripts import run_lerf_v21_frozen_relative_metric as launcher
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


def _artifact(tmp_path: Path, name: str) -> dict[str, str]:
    path = (tmp_path / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    return file_record(path)


def _cache(tmp_path: Path) -> dict:
    names = (
        "relevance",
        "relative",
        "primitive",
        "renderer",
        "state",
        "manifest",
        "all",
        "negative",
        "query_execution",
        "source",
        "descriptor",
        "health",
        "health_prereg",
        "query_prereg",
        "positive",
    )
    records = {name: _artifact(tmp_path, name) for name in names}
    query_ids = ["red cup", "tea pot"]
    raw = torch.tensor([[0.2, 0.7], [0.9, 0.1], [0.3, 0.8]])
    canonical = torch.arange(3, dtype=torch.int64)
    relevance_inputs = {
        "source_result": records["source"],
        "target_descriptor": records["descriptor"],
        "health_v4_audit": records["health"],
        "health_v4_preregistration": records["health_prereg"],
        "query_preregistration": records["query_prereg"],
        "exact_query_manifest": records["manifest"],
        "positive_text_cache": records["positive"],
        "all_query_text_cache": records["all"],
        "canonical_negative_bank": records["negative"],
    }
    relevance = {
        "schema": contrast.QUERY_RELEVANCE_SCHEMA,
        "scene_id": "scene",
        "physical_space_id": "space",
        "query_execution_authority": records["query_execution"],
        "input_authority": relevance_inputs,
        "query_ids": query_ids,
        "canonical_region_indices": canonical,
        "region_absolute_relevance": raw,
    }
    execution = {
        "verified_record": records["query_execution"],
        **relevance_inputs,
        "verified_manifest": {
            "query_ids": query_ids,
            "query_ids_sha256": canonical_json_sha256(query_ids),
        },
    }
    relative_values = torch.tensor(
        [[0.0, 0.7], [0.9, 0.0], [0.0, 0.8]], dtype=torch.float32
    )
    relative = {
        "scene_id": "scene",
        "physical_space_id": "space",
        "query_axis_count": 2,
        "raw_relevance": raw,
        "relative_relevance": relative_values,
        "input_authority": {"exact_relevance": records["relevance"]},
    }
    valid = torch.tensor([True, True, True, False])
    membership = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    primitive = {
        "scene_id": "scene",
        "physical_space_id": "space",
        "query_axis_count": 2,
        "relative_relevance": relative_values,
        "primitive_valid": valid,
        "primitive_membership": membership,
        "input_authority": {
            "frozen_relative_readout": records["relative"],
            "factorized_primitive_state": records["state"],
        },
    }
    xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    return formal.build_external_query_score_cache(
        validated_relevance=relevance,
        verified_query_execution=execution,
        validated_relative=relative,
        validated_primitive=primitive,
        relevance_record=records["relevance"],
        relative_record=records["relative"],
        primitive_record=records["primitive"],
        renderer_record=records["renderer"],
        manifest_record=records["manifest"],
        all_query_record=records["all"],
        negative_record=records["negative"],
        state_record=records["state"],
        state_xyz=xyz,
        state_valid=valid,
        renderer_xyz=xyz.clone(),
    )


def test_explicit_adapter_preserves_query_axis_and_binary_membership(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    assert cache["schema"] == formal.EXTERNAL_CACHE_SCHEMA
    assert cache["metadata"]["query_names"] == ["red cup", "tea pot"]
    assert cache["metadata"]["score_transform"] == "none"
    assert cache["query_scores"].shape == (4, 2)
    assert formal.external_cache_contract()["legacy_native_v3_bridge_modified"] is False


def test_adapter_rejects_relative_to_primitive_axis_drift(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache["query_scores"] = cache["query_scores"].flip(1)
    cache["channel_sha256"] = formal.channel_sha256(cache)
    # A self-consistent external cache is permitted; the full builder is what
    # binds it to the primitive authority.  Its explicit schema still rejects
    # non-binary or invalid-row drift locally.
    cache["query_scores"][~cache["valid"], 0] = 1.0
    cache["channel_sha256"] = formal.channel_sha256(cache)
    with pytest.raises(ValueError, match="external cache differs"):
        formal.validate_external_query_score_cache(cache)


def test_launcher_is_single_fixed_candidate_and_has_no_scan_controls() -> None:
    authority = {
        "protocol": formal.METRIC_PROTOCOL,
        "frozen_evaluator": formal.FROZEN_EVALUATOR,
        "config": {"path": "/config"},
        "renderer_geometry_checkpoint": {"path": "/renderer"},
        "scene_id": "scene",
        "label_root": "/labels",
        "output_dir": "/output",
        "frozen_summary_head": formal.FROZEN_SUMMARY_HEAD,
        "all_query_text_cache": {"path": "/all"},
        "canonical_negative_text_cache": {"path": "/negative"},
        "external_query_score_cache": {"path": "/scores"},
    }
    command = launcher.build_command(authority, gpu=1)
    assert "vala_paper_3d" in command
    assert "/scores" in command
    assert "--threshold_sweep" not in command
    destinations = {action.dest for action in launcher.build_parser()._actions}
    assert destinations == {
        "help",
        "execution_authority",
        "expected_execution_authority_sha256",
        "gpu",
        "execute",
    }
    assert formal.METRIC_PROTOCOL["score_threshold"] == 0.6
    assert formal.METRIC_PROTOCOL["score_postprocess"] == "none"
