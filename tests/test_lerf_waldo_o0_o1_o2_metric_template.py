from __future__ import annotations

import json
from pathlib import Path


TEMPLATE = Path(
    "paper/artifacts/lerf_waldo_kitchen_o0_o1_o2_exact_metric_authority_template_20260807.json"
)


def _load() -> dict[str, object]:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def test_template_is_fail_closed_and_strictly_ordered() -> None:
    payload = _load()
    assert payload["status"] == (
        "template_only_pending_source_materialization_and_control_replay"
    )
    assert payload["metric_execution_authorized"] is False
    assert payload["strict_execution_order"] == ["O0_exact_control", "O1", "O2"]
    assert payload["access_audit"]["v3_outputs_opened_during_template_build"] is False


def test_o0_control_is_exact_completed_fp32_pair() -> None:
    control = _load()["stages"]["O0_exact_control"]
    assert control["positive_cache"]["sha256"] == (
        "c0a4d000c5b1bd05da9e3e30b18132bc8ecfd5476a35474f94dfa335f09b45ef"
    )
    assert control["negative_cache"]["sha256"] == (
        "3eb9a71ebd196687beebe5c3b6464aefb6ed88b38cab5a699a06413199dabfcb"
    )


def test_candidate_hashes_remain_unresolved_until_materialization() -> None:
    stages = _load()["stages"]
    for oracle in ("O1", "O2"):
        assert stages[oracle]["positive_cache"]["sha256"].startswith("${")
        assert stages[oracle]["negative_cache"]["sha256"].startswith("${")


def test_command_matches_frozen_teatime_protocol() -> None:
    payload = _load()
    protocol = payload["frozen_protocol"]
    assert protocol["protocol_preset"] == "vala_repo_3d"
    assert protocol["ours_query_contrast"] == "none"
    assert protocol["ours_scale_fusion"] == "peak_select"
    argv = payload["command_template"]["argv"]
    assert argv[argv.index("--scene") + 1] == "waldo_kitchen"
    assert argv[argv.index("--protocol_preset") + 1] == "vala_repo_3d"
    assert argv[argv.index("--ours_query_contrast") + 1] == "none"
    assert argv[argv.index("--ours_scale_fusion") + 1] == "peak_select"
    assert argv[argv.index("--gpu") + 1] == "0"
