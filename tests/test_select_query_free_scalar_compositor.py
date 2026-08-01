from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from radio_gs.scripts.select_query_free_scalar_compositor import (
    AUDIT_NAME,
    BASELINE_VARIANT,
    CANDIDATE_VARIANTS,
    FIXED_SCALAR_OPERATOR_CONTRACT,
    FIXED_SELECTION_CONTRACT,
    SCREEN_NAME,
    select_scalar_compositor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variant_report(
    *,
    mean_delta: float = 0.0,
    p05_delta: float = 0.0,
    affinity_gain: float = 0.0,
    boundary_gain: float = 0.0,
    support: float = 1.0,
) -> dict:
    report = {}
    for index, space in enumerate(("raw_radio", "official_dino_v3", "official_sam3")):
        relation_gain = affinity_gain if index else 0.0
        margin_gain = boundary_gain if index else 0.0
        report[space] = {
            "pixels": 64,
            "mean_cosine": 0.8 + mean_delta,
            "p05_cosine": 0.6 + p05_delta,
            "local_relation": {
                "pairs": 96,
                "affinity_pearson": 0.2 + relation_gain,
                "teacher_boundary_margin": 0.5,
                "boundary_margin_retention": 0.3 + margin_gain,
            },
        }
    report["support_fraction_on_visible"] = support
    return report


def _write_scene(
    tmp_path: Path,
    scene: str,
    *,
    candidate_overrides: dict[str, dict] | None = None,
    audit_flags: dict | None = None,
    manifest_flags: dict | None = None,
) -> Path:
    root = tmp_path / scene
    root.mkdir()
    false_flags = {
        "uses_benchmark_scenes": False,
        "queries_opened": False,
        "masks_opened": False,
        "labels_opened": False,
    }
    manifest = root / "run_manifest.json"
    manifest_payload = {
        "schema_version": 1,
        "screen": SCREEN_NAME,
        "scene_id": scene,
        "split_role": "development",
        **false_flags,
        "selection_contract": FIXED_SELECTION_CONTRACT,
        "scalar_operator_contract": FIXED_SCALAR_OPERATOR_CONTRACT,
    }
    manifest_payload.update(manifest_flags or {})
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    aggregate = {
        BASELINE_VARIANT: _variant_report(),
        "gamma_1.25": _variant_report(
            mean_delta=-0.001,
            p05_delta=-0.002,
            affinity_gain=0.002,
            boundary_gain=0.007,
        ),
        "gamma_1.5": _variant_report(
            mean_delta=-0.002,
            p05_delta=-0.004,
            affinity_gain=0.003,
            boundary_gain=0.009,
        ),
        "gamma_2": _variant_report(
            mean_delta=-0.006,
            p05_delta=-0.012,
            affinity_gain=0.006,
            boundary_gain=0.014,
        ),
        "top4": _variant_report(
            mean_delta=-0.01,
            p05_delta=-0.02,
            affinity_gain=0.01,
            boundary_gain=0.02,
        ),
    }
    for candidate, overrides in (candidate_overrides or {}).items():
        aggregate[candidate] = _variant_report(**overrides)
    assert tuple(aggregate) == CANDIDATE_VARIANTS
    protocol = {
        "frame_role": "development",
        "held_out_from_mpr": True,
        "mpr_training_overlap": [],
        "official_adaptors_frozen": True,
        "same_geometry_visible_pixels_for_all_variants": True,
        **false_flags,
    }
    protocol.update(audit_flags or {})
    audit = root / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit": AUDIT_NAME,
                "scene_id": scene,
                "run_manifest": str(manifest),
                "run_manifest_sha256": _sha256(manifest),
                "protocol": protocol,
                "aggregate": aggregate,
            }
        ),
        encoding="utf-8",
    )
    return audit


def test_selects_only_candidate_passing_every_scene_head_and_metric(
    tmp_path: Path,
) -> None:
    paths = [
        _write_scene(tmp_path, "development_scene_a"),
        _write_scene(tmp_path, "development_scene_b"),
    ]

    result = select_scalar_compositor(paths)

    assert result["selected_variant"] == "gamma_1.5"
    assert result["promotion_allowed"]
    assert result["candidates"]["gamma_1.5"]["promotion_eligible"]
    assert not result["selection_uses_benchmark_scenes"]


def test_cross_scene_relation_reversal_retains_alpha_mean(tmp_path: Path) -> None:
    first = _write_scene(
        tmp_path,
        "development_scene_a",
        candidate_overrides={
            "gamma_1.25": {
                "affinity_gain": 0.002,
                "boundary_gain": 0.007,
            },
            "gamma_1.5": {
                "affinity_gain": -0.001,
                "boundary_gain": 0.009,
            },
        },
    )
    second = _write_scene(
        tmp_path,
        "development_scene_b",
        candidate_overrides={
            "gamma_1.25": {
                "affinity_gain": -0.001,
                "boundary_gain": 0.007,
            },
            "gamma_1.5": {
                "affinity_gain": -0.001,
                "boundary_gain": 0.009,
            },
        },
    )

    result = select_scalar_compositor([first, second])

    assert result["selected_variant"] == BASELINE_VARIANT
    assert not result["promotion_allowed"]
    assert "retained" in result["selection_status"]
    assert not result["candidates"]["gamma_1.25"][
        "all_scene_per_head_relation_guard_passed"
    ]


def test_one_head_regression_cannot_be_hidden_by_other_head_gain(
    tmp_path: Path,
) -> None:
    paths = [
        _write_scene(tmp_path, "development_scene_a"),
        _write_scene(tmp_path, "development_scene_b"),
    ]
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    candidate = payload["aggregate"]["gamma_1.5"]
    candidate["official_dino_v3"]["local_relation"]["affinity_pearson"] = 0.8
    candidate["official_sam3"]["local_relation"]["affinity_pearson"] = 0.199
    paths[1].write_text(json.dumps(payload), encoding="utf-8")

    result = select_scalar_compositor(paths)

    assert not result["candidates"]["gamma_1.5"]["promotion_eligible"]
    assert not result["candidates"]["gamma_1.5"]["per_scene"]["development_scene_b"][
        "relation_guard_passed"
    ]


@pytest.mark.parametrize("container", ["audit", "manifest"])
def test_missing_explicit_query_free_flag_fails_closed(
    tmp_path: Path, container: str
) -> None:
    paths = [
        _write_scene(tmp_path, "development_scene_a"),
        _write_scene(tmp_path, "development_scene_b"),
    ]
    audit_path = paths[1]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if container == "audit":
        audit["protocol"].pop("labels_opened")
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
    else:
        manifest_path = Path(audit["run_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("labels_opened")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        audit["run_manifest_sha256"] = _sha256(manifest_path)
        audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly declare labels_opened=false"):
        select_scalar_compositor(paths)


def test_declared_benchmark_scene_fails_closed(tmp_path: Path) -> None:
    paths = [
        _write_scene(tmp_path, "development_scene_a"),
        _write_scene(
            tmp_path,
            "development_scene_b",
            manifest_flags={"uses_benchmark_scenes": True},
        ),
    ]

    with pytest.raises(ValueError, match="uses_benchmark_scenes must be false"):
        select_scalar_compositor(paths)


def test_repeated_development_scene_fails_closed(tmp_path: Path) -> None:
    first = _write_scene(tmp_path, "development_scene_a")

    with pytest.raises(ValueError, match="repeated scene"):
        select_scalar_compositor([first, first])


def test_unfrozen_candidate_is_rejected(tmp_path: Path) -> None:
    paths = [
        _write_scene(tmp_path, "development_scene_a"),
        _write_scene(tmp_path, "development_scene_b"),
    ]
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["aggregate"]["top1"] = payload["aggregate"].pop("top4")
    paths[1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidates must be exactly"):
        select_scalar_compositor(paths)


def test_cli_writes_atomic_query_free_decision(tmp_path: Path) -> None:
    paths = [
        _write_scene(tmp_path, "development_scene_a"),
        _write_scene(tmp_path, "development_scene_b"),
    ]
    output = tmp_path / "decision.json"
    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
            str(REPO_ROOT / "radio_gs/scripts/select_query_free_scalar_compositor.py"),
            "--scene-audits",
            " ".join(str(path) for path in paths),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["selected_variant"] == "gamma_1.5"
    assert result["queries_opened"] is False
    assert result["masks_opened"] is False
    assert result["labels_opened"] is False
