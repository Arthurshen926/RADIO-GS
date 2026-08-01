from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from radio_gs.scripts.validate_evaluation_protocol_freeze import (
    FreezeError,
    load_and_validate,
    validate_freeze,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "paper" / "artifacts" / "evaluation_protocol_freeze_20260801.yaml"


def _payload() -> dict:
    return yaml.safe_load(FREEZE.read_text(encoding="utf-8"))


def test_checked_in_protocol_freeze_and_hashes_pass() -> None:
    payload = load_and_validate(FREEZE, root=ROOT)
    assert len(payload["canonical_tasks"]) == 7


def test_scannet_paper8_cannot_silently_become_code9() -> None:
    payload = deepcopy(_payload())
    payload["canonical_tasks"]["concept_scannet_ovs_vala_paper8"]["cohort"][
        "scenes"
    ].append("scene0645_00")
    with pytest.raises(FreezeError, match="paper8 scenes"):
        validate_freeze(payload, root=ROOT, verify_hashes=False)


def test_vala_lerf_split_must_be_extensionless_and_fail_closed() -> None:
    payload = deepcopy(_payload())
    payload["canonical_tasks"]["concept_lerf3d_vala"]["cohort"][
        "extensionless_test_stems_required"
    ] = False
    with pytest.raises(FreezeError, match="extensionless"):
        validate_freeze(payload, root=ROOT, verify_hashes=False)


def test_pfpr_cannot_gain_a_paper_comparison_or_formal_label() -> None:
    payload = deepcopy(_payload())
    protocol = payload["canonical_tasks"][
        "correspondence_pfpr_ludvig_adapter"
    ]["frozen_protocol"]
    protocol["paper_comparison"] = "diagnostic_only"
    protocol["formal_full_benchmark_deferred"] = False
    with pytest.raises(FreezeError, match="PFPR paper comparison"):
        validate_freeze(payload, root=ROOT, verify_hashes=False)


def test_authoritative_artifact_must_never_be_cleanup_eligible() -> None:
    payload = deepcopy(_payload())
    payload["canonical_tasks"]["spatial_nvos_ludvig"][
        "authoritative_artifacts"
    ][0]["retention"] = "safe_remove"
    with pytest.raises(FreezeError, match="retention"):
        validate_freeze(payload, root=ROOT, verify_hashes=False)
