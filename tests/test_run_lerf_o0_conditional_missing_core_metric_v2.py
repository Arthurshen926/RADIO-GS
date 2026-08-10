from __future__ import annotations

import pytest

from radio_gs.scripts import run_lerf_o0_conditional_missing_core_metric_v2 as script


def _authority() -> dict:
    value = {
        "scene_id": "figurines",
        "label_root": "/data/labels",
        "output_dir": "/data/output",
        "external_query_score_cache": {"path": "/data/cache.pt", "sha256": "0" * 64},
    }
    value.update(script.FROZEN_INPUTS)
    return value


def test_fix6c_metric_protocol_and_command_have_no_sweep() -> None:
    assert script.SCHEMA.endswith(".v2")
    assert "FIX6c" in script.STATUS
    assert script.PROTOCOL["threshold_scan"] is False
    command = script.build_command(_authority(), gpu=1)
    assert command[command.index("--scene") + 1] == "figurines"
    assert command[command.index("--gpu") + 1] == "1"
    assert "--score_threshold" not in command
    assert "--threshold_scan" not in command


def test_fix6c_metric_parser_has_no_frozen_input_or_tuning_overrides() -> None:
    parser = script.build_parser()
    actions = parser._subparsers._group_actions[0].choices
    destinations = {
        action.dest for command in actions.values() for action in command._actions
    }
    assert not set(script.FROZEN_RECORD_NAMES) & destinations
    assert "scene_id" not in destinations
    assert "score_threshold" not in destinations
    assert "threshold_scan" not in destinations


def test_fix6c_metric_rejects_invalid_gpu() -> None:
    with pytest.raises(ValueError, match="gpu"):
        script.build_command(_authority(), gpu=-1)
    with pytest.raises(ValueError, match="gpu"):
        script.build_command(_authority(), gpu=True)

