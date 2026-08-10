from __future__ import annotations

from pathlib import Path

import pytest

from radio_gs.scripts import run_lerf_o0_conditional_missing_core_metric as script


def _authority() -> dict:
    value = {
        "scene_id": "figurines",
        "label_root": "/data/labels",
        "output_dir": "/data/output",
        "external_query_score_cache": {
            "path": "/data/cache.pt",
            "sha256": "0" * 64,
        },
    }
    value.update(script.FROZEN_INPUTS)
    return value


def test_protocol_is_exact_frozen_vala_without_scan() -> None:
    assert script.PROTOCOL == {
        "protocol_preset": "vala_paper_3d",
        "score_threshold": 0.6,
        "score_postprocess": "none",
        "selection_mode": "score_threshold",
        "projection_mode": "selected_only_alpha",
        "official_frames_only": True,
        "mask_refinement": "none",
        "alpha_binarization": "png_uint8_gt10",
        "silhouette_threshold": 10.0 / 255.0,
        "threshold_scan": False,
    }


def test_command_has_no_metric_hyperparameter_override() -> None:
    command = script.build_command(_authority(), gpu=1)
    assert command[command.index("--protocol_preset") + 1] == "vala_paper_3d"
    assert command[command.index("--external_query_score_cache") + 1] == "/data/cache.pt"
    assert command[command.index("--gpu") + 1] == "1"
    assert "--score_threshold" not in command
    assert "--threshold_scan" not in command


def test_command_rejects_invalid_gpu() -> None:
    with pytest.raises(ValueError, match="gpu"):
        script.build_command(_authority(), gpu=-1)
    with pytest.raises(ValueError, match="gpu"):
        script.build_command(_authority(), gpu=True)


def test_parser_exposes_no_protocol_tuning_flags() -> None:
    parser = script.build_parser()
    actions = parser._subparsers._group_actions[0].choices
    destinations = {
        action.dest
        for command in actions.values()
        for action in command._actions
    }
    assert "score_threshold" not in destinations
    assert "threshold_scan" not in destinations
    assert "projection_mode" not in destinations


def test_frozen_inputs_match_accepted_figurines_protocol() -> None:
    assert script.FROZEN_INPUTS["renderer_geometry_checkpoint"]["sha256"] == (
        "6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2"
    )
    assert script.FROZEN_INPUTS["config"]["sha256"] == (
        "a17ada0f1d34cf043f04ddc2f6503c262845d1fed8b4550df8f5d79f2dbd8f11"
    )


def test_passed_conditional_revalidates_actual_source_heldout_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {"path": "/data/authority.json", "sha256": "1" * 64}
    authority = {"input_authority": {"source_selector_model": record}}

    monkeypatch.setattr(
        script,
        "load_json_object",
        lambda *args, **kwargs: ({}, record["sha256"], Path(record["path"])),
    )
    monkeypatch.setattr(
        script.materializer, "validate_authority", lambda value: authority
    )

    def reject_unverified_source(inputs: object) -> float:
        assert inputs is authority["input_authority"]
        raise ValueError("source-frozen conditional selector gate differs")

    monkeypatch.setattr(
        script.materializer, "_validate_source_gate", reject_unverified_source
    )
    with pytest.raises(ValueError, match="source-frozen"):
        script._load_passed_conditional(
            authority_record=record,
            cache_record={"path": "/data/cache.pt", "sha256": "2" * 64},
            report_record={"path": "/data/report.json", "sha256": "3" * 64},
        )
