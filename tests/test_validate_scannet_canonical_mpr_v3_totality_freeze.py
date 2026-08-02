from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from radio_gs.scripts.validate_scannet_canonical_mpr_v3_totality_freeze import (
    DEFAULT_FREEZE,
    EXPECTED_FREEZE_SHA256,
    TotalityFreezeError,
    V1_EXPECTED_FREEZE_SHA256,
    V1_FREEZE,
    V2_FREEZE,
    _load_json,
    _sha256,
    _validate_semantics,
    validate_freeze,
)


def _payload() -> dict:
    return json.loads(DEFAULT_FREEZE.read_text(encoding="utf-8"))


def test_real_totality_freeze_and_all_bound_sources_validate() -> None:
    result = validate_freeze()
    assert _sha256(DEFAULT_FREEZE) == EXPECTED_FREEZE_SHA256
    assert result["status"] == "validated"
    assert result["schema_version"] == 2
    assert result["artifact_version"] == "v2"
    assert result["immutable_source_count"] == 8
    assert result["paper8_scene_count"] == 8


def test_immutable_v1_remains_recognized_but_its_drifted_producer_fails_closed() -> None:
    payload = _load_json(V1_FREEZE)
    _validate_semantics(payload)
    assert _sha256(V1_FREEZE) == V1_EXPECTED_FREEZE_SHA256
    with pytest.raises(TotalityFreezeError, match="producer_source byte size drifted"):
        validate_freeze(V1_FREEZE)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (
            "graph_observed",
            "scale_aggregation",
            "mean_over_scales",
            "three-scale cosine-max",
        ),
        (
            "no_graph_evidence",
            "neighbor_transfer",
            True,
            "primitive fallback",
        ),
    ],
)
def test_totality_semantic_drift_fails_closed(
    section: str, key: str, value: object, message: str
) -> None:
    payload = deepcopy(_payload())
    payload["semantic_totality"][section][key] = value
    with pytest.raises(TotalityFreezeError, match=message):
        _validate_semantics(payload)


def test_partial_valid_domain_fails_closed() -> None:
    payload = deepcopy(_payload())
    payload["semantic_totality"]["valid_is_total"] = False
    with pytest.raises(TotalityFreezeError, match="valid domain is not total"):
        _validate_semantics(payload)


def test_v2_cpu_activated_geometry_authority_drift_fails_closed() -> None:
    payload = deepcopy(_payload())
    payload["semantic_totality"]["geometry_row_authority"][
        "activation_device"
    ] = "cuda:0"
    with pytest.raises(TotalityFreezeError, match="CPU bitwise activated geometry"):
        _validate_semantics(payload)


def test_v2_supersession_is_strictly_pre_formal_evaluation() -> None:
    payload = deepcopy(_payload())
    payload["supersession"]["predecessor_formal_result_materialized"] = True
    with pytest.raises(TotalityFreezeError, match="pre-formal-evaluation supersession"):
        _validate_semantics(payload)


def test_freeze_path_is_repo_artifact() -> None:
    assert DEFAULT_FREEZE == V2_FREEZE == (
        Path(__file__).resolve().parents[1]
        / "paper/artifacts/scannet_canonical_mpr_v3_gaussian_semantic_totality_freeze_20260802_v2.json"
    )
