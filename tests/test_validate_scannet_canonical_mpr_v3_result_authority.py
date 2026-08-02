from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from radio_gs.scripts.validate_scannet_canonical_mpr_v3_result_authority import (
    DEFAULT_AUTHORITY,
    EXPECTED_AUTHORITY_SHA256,
    ResultAuthorityError,
    _load_authority,
    _sha256_bytes,
    _validate_payload_semantics,
    validate_result_authority,
)


def _payload() -> dict:
    return json.loads(DEFAULT_AUTHORITY.read_text(encoding="utf-8"))


def test_real_paper8_result_authority_replays_exactly() -> None:
    result = validate_result_authority()
    assert result["status"] == "validated"
    assert result["authority_sha256"] == EXPECTED_AUTHORITY_SHA256
    assert result["scene_count"] == 8
    assert result["total_gaussian_rows"] == 1_134_207
    assert result["region_observed_rows"] == 916_371
    assert result["no_evidence_fallback_rows"] == 217_836
    assert result["macro"]["19"] == {
        "miou": 0.3786329623189025,
        "macc": 0.5521891702607359,
    }


def test_result_authority_is_the_repo_frozen_artifact() -> None:
    assert DEFAULT_AUTHORITY == (
        Path(__file__).resolve().parents[1]
        / "paper/artifacts/scannet_canonical_mpr_v3_gaussian_semantic_result_authority_20260802.json"
    )
    assert _sha256_bytes(DEFAULT_AUTHORITY.read_bytes()) == EXPECTED_AUTHORITY_SHA256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["metrics"]["19"].__setitem__("extra", 0.99),
            "metric schema",
        ),
        (
            lambda value: value["benchmark_binding"]["scenes"].pop(),
            "paper8 benchmark binding",
        ),
        (
            lambda value: value["derivation_binding"].__setitem__(
                "exact_metric_replay_required", False
            ),
            "exact result derivation",
        ),
        (
            lambda value: value["scenes"]["scene0000_00"].__setitem__(
                "no_evidence_fallback_count", 0
            ),
            "totality counts",
        ),
    ],
)
def test_result_authority_semantic_drift_fails_closed(mutation, message: str) -> None:
    payload = deepcopy(_payload())
    mutation(payload)
    with pytest.raises(ResultAuthorityError, match=message):
        _validate_payload_semantics(payload)


def test_unknown_mutated_authority_sha_fails_before_evidence_reads(tmp_path: Path) -> None:
    payload = _payload()
    payload["metrics"]["19"]["miou"] = 0.99
    mutated = tmp_path / "mutated-authority.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResultAuthorityError, match="authority SHA256 differs"):
        _load_authority(mutated)
