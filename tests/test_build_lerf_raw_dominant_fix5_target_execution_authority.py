from __future__ import annotations

from pathlib import Path

import pytest

from radio_gs.scripts import build_lerf_o0_anchored_raw_dominant_positive_utility_cache_fix5 as fix5
from radio_gs.scripts import build_lerf_raw_dominant_fix5_target_execution_authority as builder


def _record(name: str) -> dict[str, str]:
    return {"path": f"/immutable/{name}", "sha256": name[0] * 64}


def test_compose_authority_matches_exact_fix5_consumer_contract(tmp_path: Path) -> None:
    authority = builder._compose_authority(
        parent_execution=_record("a_parent"),
        parent_cache=_record("b_cache"),
        parent_report=_record("c_report"),
        source_execution=_record("d_source"),
        source_result=_record("e_result"),
        output_cache=(tmp_path / "scores.pt").resolve(),
        output_report=(tmp_path / "report.json").resolve(),
    )
    assert authority["schema"] == fix5.EXECUTION_SCHEMA
    assert authority["status"] == fix5.EXECUTION_STATUS
    assert authority["implementation"]["sha256"] == builder.file_record(
        fix5.IMPLEMENTATION
    )["sha256"]
    assert set(authority["dependencies"]) == set(fix5.DEPENDENCIES)
    assert authority["fixed_intervention"]["primitive_majority_threshold"] is None
    assert authority["fixed_intervention"]["anchor_order"].startswith("filter_O0_anchor")
    assert authority["target_score_cache_authorized"] is True
    assert authority["target_quality_execution_authorized"] is False
    assert authority["access_audit"]["target_quality_readout_executed"] is False


def test_canonical_new_rejects_relative_existing_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new canonical"):
        builder._canonical_new("relative.pt", label="new canonical")
    existing = tmp_path / "existing.pt"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="new canonical"):
        builder._canonical_new(str(existing.resolve()), label="new canonical")
    symlink = tmp_path / "link.pt"
    symlink.symlink_to(existing)
    with pytest.raises(ValueError, match="new canonical"):
        builder._canonical_new(str(symlink.absolute()), label="new canonical")


def test_expected_record_rejects_wrong_sha(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        builder._expected_record(
            str(source.resolve()), "0" * 64, label="source authority"
        )
