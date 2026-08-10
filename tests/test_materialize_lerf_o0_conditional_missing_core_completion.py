from __future__ import annotations

from pathlib import Path

import pytest

from radio_gs.interfaces import (
    lerf_o0_conditional_missing_core_completion as formal,
)
from radio_gs.scripts import (
    audit_source_same_axis_o0_missing_core_mechanism as source_feature_api,
)
from radio_gs.scripts import (
    materialize_lerf_o0_conditional_missing_core_completion as builder,
)
from radio_gs.utils.immutable_artifacts import file_record


def test_authority_rejects_modified_no_gt_gate(tmp_path: Path) -> None:
    token = tmp_path / "record"
    token.write_text("bound", encoding="utf-8")
    record = file_record(token)
    inputs = {
        name: record
        for name in (
            "exact_o0_cache",
            "target_accepted_v2",
            "target_capability_descriptor",
            "factorized_primitive_state",
            "source_selector_authority",
            "source_selector_model",
            "source_selector_report",
            "heldout_selector_authority",
            "heldout_selector_report",
            "heldout_selector_unit_table",
        )
    }
    gates = builder.fixed_no_gt_gates()
    gates["maximum_per_query_membership_expansion"] = 99.0
    authority = {
        "schema": builder.AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_strict_source_heldout_PASS_for_no_GT_cache_only",
        "scene_id": "scene",
        "implementation": file_record(Path(builder.__file__).resolve()),
        "interface": file_record(Path(formal.__file__).resolve()),
        "source_feature_implementation": file_record(
            Path(source_feature_api.__file__).resolve()
        ),
        "contract": formal.completion_contract(),
        "contract_sha256": formal.CONTRACT_SHA256,
        "frozen_threshold_inclusive": builder.FROZEN_THRESHOLD,
        "fixed_no_GT_gates": gates,
        "input_authority": inputs,
        "outputs": {"cache": "/tmp/cache.pt", "report": "/tmp/report.json"},
        "target_score_cache_authorized": True,
        "target_metric_execution_authorized": False,
        "access_audit": builder._build_access(),
    }
    with pytest.raises(ValueError, match="header differs"):
        builder.validate_authority(authority)
