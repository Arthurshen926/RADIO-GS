from __future__ import annotations

from pathlib import Path

import pytest

from radio_gs.scripts import run_rank256_o0_full_lift_metric as script


def _authority() -> dict:
    return {
        "scene_id": "figurines",
        "frozen_evaluator": {"path": "/repo/eval.py", "sha256": "a" * 64},
        "config": {"path": "/repo/config.yaml", "sha256": "b" * 64},
        "renderer_geometry_checkpoint": {
            "path": "/data/renderer.pth",
            "sha256": "c" * 64,
        },
        "label_root": "/data/labels",
        "output_dir": "/data/output",
        "frozen_summary_head": {"path": "/repo/head.pth", "sha256": "d" * 64},
        "all_query_text_cache": {"path": "/repo/all.pt", "sha256": "e" * 64},
        "canonical_negative_text_cache": {
            "path": "/repo/negative.pt",
            "sha256": "f" * 64,
        },
        "external_query_score_cache": {
            "path": "/data/cache.pt",
            "sha256": "0" * 64,
        },
    }


def test_protocol_is_exact_single_frozen_vala_readout() -> None:
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


def _premetric_authority() -> dict:
    return {
        "scene_id": "figurines",
        "verified_parent": {
            "input_authority": {
                "renderer_geometry_checkpoint": {
                    "path": "/data/renderer.pth",
                    "sha256": "c" * 64,
                }
            }
        },
        "verified_query": {
            "all_query_record": {
                "path": "/repo/all.pt",
                "sha256": "e" * 64,
            },
            "negative_record": {
                "path": "/repo/negative.pt",
                "sha256": "f" * 64,
            },
        },
    }


def _frozen_inputs() -> dict:
    return {
        "frozen_evaluator": dict(script.FROZEN_EVALUATOR),
        "frozen_summary_head": dict(script.FROZEN_SUMMARY_HEAD),
        "config": dict(script.FROZEN_FIGURINES_CONFIG),
        "renderer_geometry_checkpoint": {
            "path": "/data/renderer.pth",
            "sha256": "c" * 64,
        },
        "all_query_text_cache": {
            "path": "/repo/all.pt",
            "sha256": "e" * 64,
        },
        "canonical_negative_text_cache": {
            "path": "/repo/negative.pt",
            "sha256": "f" * 64,
        },
    }


def test_frozen_bindings_require_exact_protocol_and_query_lineage() -> None:
    script._validate_frozen_bindings(
        frozen_inputs=_frozen_inputs(),
        premetric_authority=_premetric_authority(),
    )
    for name in (
        "frozen_evaluator",
        "frozen_summary_head",
        "config",
        "renderer_geometry_checkpoint",
        "all_query_text_cache",
        "canonical_negative_text_cache",
    ):
        changed = _frozen_inputs()
        changed[name] = {**changed[name], "sha256": "9" * 64}
        with pytest.raises(ValueError, match="frozen evaluator binding"):
            script._validate_frozen_bindings(
                frozen_inputs=changed,
                premetric_authority=_premetric_authority(),
            )


def test_frozen_bindings_reject_nonfigurines_scene() -> None:
    premetric = _premetric_authority()
    premetric["scene_id"] = "ramen"
    with pytest.raises(ValueError, match="frozen evaluator binding"):
        script._validate_frozen_bindings(
            frozen_inputs=_frozen_inputs(),
            premetric_authority=premetric,
        )


def test_build_command_uses_only_frozen_external_cache_path() -> None:
    command = script.build_command(_authority(), gpu=1)
    assert command[1] == "/repo/eval.py"
    assert command[command.index("--external_query_score_cache") + 1] == "/data/cache.pt"
    assert command[command.index("--protocol_preset") + 1] == "vala_paper_3d"
    assert command[command.index("--gpu") + 1] == "1"
    assert "--score_threshold" not in command
    assert "--threshold_scan" not in command


def test_build_command_rejects_invalid_gpu() -> None:
    with pytest.raises(ValueError, match="gpu"):
        script.build_command(_authority(), gpu=-1)
    with pytest.raises(ValueError, match="gpu"):
        script.build_command(_authority(), gpu=True)


def test_parser_has_no_metric_hyperparameter_flags() -> None:
    parser = script.build_parser()
    build = parser._subparsers._group_actions[0].choices["build-authority"]
    run = parser._subparsers._group_actions[0].choices["run"]
    destinations = {action.dest for action in [*build._actions, *run._actions]}
    assert "score_threshold" not in destinations
    assert "threshold_scan" not in destinations
    assert "projection_mode" not in destinations


def test_unopened_path_is_syntax_only(tmp_path: Path) -> None:
    absent = str((tmp_path / "not-created").resolve())
    assert script._unopened(absent, label="absent") == absent
    with pytest.raises(ValueError, match="canonical"):
        script._unopened(str(tmp_path / ".." / tmp_path.name), label="bad")
