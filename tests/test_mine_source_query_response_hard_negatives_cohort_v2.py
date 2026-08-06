from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from radio_gs.scripts import (
    mine_source_query_response_hard_negatives_cohort_v2 as miner,
)
from radio_gs.scripts import train_surface_region_full_scalar_residual as cohort_api
from radio_gs.utils.immutable_artifacts import sha256_file


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY / (
    "paper/artifacts/"
    "clean_scene_source_query_response_hard_negative_cohort_v2_"
    "preregistration_20260806.json"
)
PREREGISTRATION_SHA256 = (
    "de4aa57ab283b195d10ddb6965a4de03be3a8bd0fbb5647215a6a15cd43dab16"
)
COHORT = REPOSITORY / (
    "paper/artifacts/"
    "full_scalar_scannet_clean_24train_8validation_cohort_authority_20260805.json"
)
COHORT_SHA256 = "7f450c09d2db9f55fa8e1efc85905b29b2a7fc63a66169b6ffa123b6dd1c8463"
V1_MINER = REPOSITORY / "radio_gs/scripts/mine_source_query_response_hard_negatives.py"
V1_MINER_SHA256 = "06ba0089063a02eba29b5dc25fa930df686debc6aa0555427e725245c758840b"
V1_SCENE0001_AUTHORITY = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260806/"
    "source_query_response_hard_negatives_v1/scene0001_00/"
    "hard_negative_index_authority_bound_v1.pt"
)
V1_SCENE0001_AUTHORITY_SHA256 = (
    "243df78a8aaa103d61fd8ed1361af4031e90484cf028ce4da0ac0d612cdec36d"
)


def _cohort() -> dict[str, object]:
    value, _ = cohort_api.load_cohort_authority(
        COHORT, expected_sha256=COHORT_SHA256
    )
    return value


@pytest.mark.parametrize(
    ("scene_id", "expected_split"),
    [
        ("scene0001_00", "source_train"),
        ("scene0036_00", "source_train"),
        ("scene0004_00", "source_validation"),
        ("scene0037_00", "source_validation"),
    ],
)
def test_declared_scene_resolves_to_frozen_cohort_split(
    scene_id: str, expected_split: str
) -> None:
    assert miner.resolve_declared_scene(_cohort(), scene_id) == expected_split
    execution = miner.execution_audit(scene_id, expected_split)
    assert execution["scene_id"] == scene_id
    assert execution["cohort_split"] == expected_split
    assert execution["cross_scene_similarity_matrix"] is False
    assert "scene0002_skipped" not in execution
    assert "scene0002_reason" not in execution


def test_cohort_external_or_ambiguous_scene_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a unique member"):
        miner.resolve_declared_scene(_cohort(), "scene9999_00")
    synthetic = {
        "source_train_scene_ids": ["scene0004_00"],
        "source_validation_scene_ids": ["scene0004_00"],
    }
    with pytest.raises(ValueError, match="not a unique member"):
        miner.resolve_declared_scene(synthetic, "scene0004_00")


def test_preflight_rejects_external_scene_before_scene_inputs_are_needed() -> None:
    args = argparse.Namespace(
        scene_id="scene9999_00",
        cohort_authority=str(COHORT),
        expected_cohort_authority_sha256=COHORT_SHA256,
        expected_preregistration_sha256=PREREGISTRATION_SHA256,
        accepted_v2="/does/not/exist/accepted.pt",
        teacher="/does/not/exist/teacher.pt",
    )
    with pytest.raises(ValueError, match="not a unique member"):
        miner.preflight_declared_scene(args)


def test_preregistration_and_v1_baseline_are_byte_immutable() -> None:
    assert sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256
    value = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    validated = miner.validate_preregistration(value)
    assert validated["authorization"]["cohort_scene_source_only_mining_authorized"]
    assert validated["authorization"]["benchmark_execution_authorized"] is False
    assert sha256_file(V1_MINER) == V1_MINER_SHA256
    assert sha256_file(V1_SCENE0001_AUTHORITY) == V1_SCENE0001_AUTHORITY_SHA256


def test_v2_source_has_no_hardcoded_scene0002_execution_status() -> None:
    source = Path(miner.__file__).read_text(encoding="utf-8")
    assert "scene0002_skipped" not in source
    assert "scene0002_reason" not in source
