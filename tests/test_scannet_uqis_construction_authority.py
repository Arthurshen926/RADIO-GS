import json
from pathlib import Path

import pytest

from radio_gs.benchmarks.scannet_uqis.construction_authority import (
    _validate_cohort_ledger,
    _verify_binding,
)
from radio_gs.benchmarks.scannet_uqis.protocol import (
    COHORT_DERIVATION_LEDGER,
    PREREGISTERED_TEST_SCENES,
    sha256_file,
)


def _ledger() -> dict:
    return {
        "schema_version": "scannet_uqis_cohort_derivation_v1",
        "benchmark_version": "scannet-uqis-9-v0.1",
        "selection_information": "official_scannet_geometry_plus_nr3d_annotations_only",
        "formal_method_predictions_opened": False,
        "final_scene_order": list(PREREGISTERED_TEST_SCENES),
        "decisions": list(COHORT_DERIVATION_LEDGER),
    }


def test_cohort_authority_accepts_only_frozen_order_and_decisions(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(_ledger()), encoding="utf-8")
    assert _validate_cohort_ledger(path)["final_scene_order"] == list(
        PREREGISTERED_TEST_SCENES
    )

    changed = _ledger()
    changed["final_scene_order"] = list(reversed(PREREGISTERED_TEST_SCENES))
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="scene order changed"):
        _validate_cohort_ledger(path)


def test_source_binding_fails_after_content_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"official")
    binding = {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    assert _verify_binding(binding, label="fixture")["sha256"] == binding["sha256"]
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source hash changed"):
        _verify_binding({**binding, "bytes": source.stat().st_size}, label="fixture")


def test_source_binding_rejects_extra_self_attested_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"official")
    binding = {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "trusted": True,
    }
    with pytest.raises(ValueError, match="schema changed"):
        _verify_binding(binding, label="fixture")
