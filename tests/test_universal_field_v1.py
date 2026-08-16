from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from radio_gs.universal_field_v1 import (
    EXACT_REGISTRATION_WEIGHT_MODE,
    PRIMITIVE_READOUT_ID,
    RELIABILITY_NAMES,
    UNIVERSAL_FIELD_ID,
    UniversalFieldValidationError,
    migrate_universal_field_payload,
    validate_universal_field_authority,
    validate_universal_field_payload,
)
from tests.test_five_benchmark_method_v1 import _complete_payload


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "paper/artifacts/universal_field_v1_authority_20260816.json"


def _source_payload() -> dict:
    payload = _complete_payload()
    payload["architecture"].update(
        num_gaussians=4,
        coarse_dim=0,
        hidden_dim=2,
        use_fusion=True,
        fusion_residual_blocks=0,
    )
    payload["mpr_cache"] = "/tmp/factorized.pt"
    payload["mpr_cache_sha256"] = "a" * 64
    payload["mpr_cache_metadata"] = {
        "builder_contract": {
            "registration_weight_mode": EXACT_REGISTRATION_WEIGHT_MODE,
        }
    }
    payload["factorized_radio_metadata"] = {
        "factorized_radio_contract": {
            "reliability_scalar_names": list(RELIABILITY_NAMES),
        }
    }
    payload["state_dict"] = {"reliability": torch.empty(4, 0)}
    return payload


def _factorized_cache() -> dict:
    reliability = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.8, 0.7],
            [0.8, 0.2, 0.3, 0.7, 0.6],
            [0.7, 0.3, 0.4, 0.6, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return {
        "reliability": reliability,
        "valid": torch.tensor([True, True, True, False]),
        "reliability_scalar_names": list(RELIABILITY_NAMES),
        "reliability_scalar_names_sha256": "unused-by-migration-fixture",
    }


def test_checked_in_authority_separates_field_and_readout() -> None:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    validate_universal_field_authority(authority)
    assert authority["universal_field_id"] == UNIVERSAL_FIELD_ID
    assert authority["baseline_readout_id"] == PRIMITIVE_READOUT_ID
    assert (
        authority["construction"]["registration_weight_mode"]
        == EXACT_REGISTRATION_WEIGHT_MODE
    )


def test_reliability_migration_preserves_all_learned_state() -> None:
    source = _source_payload()
    source_before = copy.deepcopy(source)
    cache = _factorized_cache()
    migrated = migrate_universal_field_payload(
        source,
        cache,
        source_field_sha256="b" * 64,
        factorized_cache_sha256="a" * 64,
    )
    assert torch.equal(source["reliability"], source_before["reliability"])
    assert torch.equal(
        source["state_dict"]["reliability"],
        source_before["state_dict"]["reliability"],
    )
    assert torch.equal(migrated["reliability"], cache["reliability"])
    assert torch.equal(migrated["state_dict"]["reliability"], cache["reliability"])
    assert migrated["architecture"]["fusion_reliability"] is False
    assert migrated["universal_field_migration"]["decode_state_changed"] is False
    validate_universal_field_payload(migrated)


def test_universal_field_rejects_missing_or_mislabeled_reliability() -> None:
    migrated = migrate_universal_field_payload(
        _source_payload(),
        _factorized_cache(),
        source_field_sha256="b" * 64,
        factorized_cache_sha256="a" * 64,
    )
    migrated["universal_field_migration"]["reliability_scalar_names"][0] = "coverage"
    with pytest.raises(UniversalFieldValidationError, match="reliability scalar"):
        validate_universal_field_payload(migrated)
