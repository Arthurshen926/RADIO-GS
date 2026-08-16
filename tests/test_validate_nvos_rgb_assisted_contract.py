from __future__ import annotations

import json
from pathlib import Path

import pytest

from radio_gs.scripts.validate_nvos_rgb_assisted_contract import (
    DEFAULT_CONTRACT,
    EXPECTED_COHORT,
    validate_contract,
)


def test_frozen_rgb_assisted_contract_is_hash_bound_and_not_blind() -> None:
    report = validate_contract()

    assert tuple(report["scene_order"]) == EXPECTED_COHORT
    assert report["target_or_metric_bytes_opened"] is False
    assert report["rgb_assisted_main_method"] is True
    assert report["strict_unseen_retained_as_ablation"] is True
    assert report["result_classification"] == "development_evidence"


def test_contract_fails_closed_if_rgb_leaks_into_field_construction(
    tmp_path: Path,
) -> None:
    payload = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    payload["field_construction"]["target_rgb_opened"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="field-construction"):
        validate_contract(changed)


def test_contract_fails_closed_if_blind_claim_is_enabled(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    payload["claim_eligibility"]["blind_confirmation"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="claim"):
        validate_contract(changed)
