from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from radio_gs.evaluation.capability_fidelity import select_query_free_compositor
from radio_gs.scripts.build_free_canonical_radio_field import (
    build_free_field_payload,
)
from radio_gs.scripts.confirm_canonical_v5_cross_scene import confirm


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_BUNDLE_SHA256 = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _fidelity(relation_score: float, *, support: float = 1.0) -> dict:
    report = {
        "support_fraction_on_visible": support,
        "supported_visible_pixels": 10 if support == 1.0 else 0,
        "total_visible_pixels": 10,
    }
    relation = {
        "pairs": 8,
        "affinity_mae": 0.01,
        "affinity_pearson": relation_score,
        "teacher_boundary_margin": 0.5,
        "predicted_boundary_margin": 0.5 * relation_score,
        "boundary_margin_retention": relation_score,
    }
    for space in ("raw_radio", "official_dino_v3", "official_sam3"):
        report[space] = {
            "pixels": 10,
            "mean_cosine": 0.99,
            "p05_cosine": 0.98,
            "local_relation": dict(relation),
        }
    return report


def _official_source(scene: str, name: str) -> dict:
    return {
        "feature_space": name,
        "adaptor_name": "dino_v3_7b" if name == "dino_v3" else "sam3",
        "native_grid": [30, 40],
        "frame_manifest": f"/{scene}/frame_manifest.json",
        "frame_manifest_sha256": "1" * 64,
        "output_bundle_sha256": FEATURE_BUNDLE_SHA256,
        "radio_version": "c-radio_v4-h",
        "radio_checkpoint": "/radio.pth",
        "radio_checkpoint_sha256": "a" * 64,
        "radio_checkpoint_provenance": "explicit_file_sha256",
        "radio_checkpoint_load_contract": (
            "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
        ),
        "radio_source_tree_sha256": "2" * 64,
        "runtime_fingerprint_sha256": "3" * 64,
        "scene": scene,
        "image_dir": f"/{scene}/color",
        "frame_indices": [0, 1],
        "frame_indices_sha256": "4" * 64,
        "execution": "official_c_radio_runtime_adaptor_output",
        "feature_extraction_execution": {
            "resume_partial": True,
            "atomic_tensor_commit": "same_directory_temp_then_os_replace_v1",
            "atomic_manifest_commit": "same_directory_temp_then_os_replace_v1",
            "committed_frame_validation": (
                "same_fd_sha256_weights_only_dtype_shape_finite_v2"
            ),
            "invalid_or_missing_frame_policy": "recompute_entire_frame_v1",
            "radio_thermal_pacing_seconds_per_image": 8.0,
            "pacing_order": "frame_commit_then_cuda_synchronize_then_sleep_v1",
        },
    }


def _write_field(path: Path) -> None:
    payload = build_free_field_payload(
        {
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "valid": torch.tensor([True, True]),
            "reliability": torch.ones(2, 3),
            "geometry_fingerprint": {
                "num_gaussians": 2,
                "xyz_sha256": "geometry-rows",
            },
            "metadata": {
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        },
        source_path="mpr.pt",
        feature_signature={
            "radio_version": "c-radio_v4-h",
            "radio_checkpoint_sha256": "a" * 64,
            "raw_feature_dim": 2,
            "adaptor_name": "backbone",
            "token_type": "primitive",
            "normalization": "radio_direction_unit",
        },
    )
    payload["feature_output_bundle_sha256"] = FEATURE_BUNDLE_SHA256
    torch.save(payload, path)


def _screen(
    tmp_path: Path,
    scene: str,
    selected: str | None,
    *,
    suffix: str = "",
    max_mean_dense_drop: float = 0.005,
    forge_selection: bool = False,
) -> tuple[Path, dict]:
    root = tmp_path / f"{scene}{suffix}"
    root.mkdir()
    manifest = root / "run_manifest.json"
    selection_contract = {
        "baseline": "v5_r_reliability",
        "max_mean_dense_drop": max_mean_dense_drop,
        "max_p05_dense_drop": 0.01,
        "max_unsupported_fraction": 0.005,
        "min_relation_gain": 0.005,
        "objective": (
            "maximize_mean_official_dino_sam_affinity_pearson_and_"
            "boundary_margin_retention_under_dense_and_support_guards"
        ),
    }
    sources = {
        name: _official_source(scene, name)
        for name in ("dino_v3", "sam3")
    }
    manifest.write_text(
        json.dumps(
            {
                "screen": "canonical-v5-query-free-capacity",
                "benchmark_queries_opened": False,
                "benchmark_masks_opened": False,
                "config_sha256": "5" * 64,
                "resolved_config_sha256": "6" * 64,
                "geometry_checkpoint_sha256": "7" * 64,
                "fidelity_frame_ids": [1, 2],
                "radio_checkpoint_sha256": "a" * 64,
                "epochs": 50,
                "seed": 20260731,
                "runner_sha256": "8" * 64,
                "implementation_sources": {"trainer": "9" * 64},
                "implementation_source_tree": {"tree_sha256": "d" * 64},
                "fixed_training_contract": {
                    "epochs": 50,
                    "seed": 20260731,
                    "candidates": {
                        "v5_r_reliability": [0, 192, 0],
                        "v5_s_spatial": [64, 512, 0],
                    },
                },
                "fixed_audit_contract": {
                    "alpha_threshold": 0.02,
                    "support_eps": 1e-6,
                    "boundary_quantile": 0.2,
                    "residual_mode": "none",
                },
                "fixed_selection_contract": selection_contract,
                "feature_extraction_safety_contract": {
                    "final_output_bundle_sha256": FEATURE_BUNDLE_SHA256,
                    "resume_partial": True,
                    "radio_checkpoint_load": (
                        "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
                    ),
                },
                "thermal_safety_contract": {"hard_abort_temperature_c": 75},
                "continuous_stage_safety_contract": {
                    "policy": "uncharacterized_continuous_hard_abort_only"
                },
                "official_capability_sources": sources,
            },
            sort_keys=True,
        )
    )
    if selected == "v5_s_spatial":
        baseline = _fidelity(0.50)
        spatial = _fidelity(0.52)
    elif selected == "v5_r_reliability":
        baseline = _fidelity(0.50)
        spatial = _fidelity(0.501)
    else:
        baseline = _fidelity(0.50, support=0.0)
        spatial = _fidelity(0.52, support=0.0)
    variants = {
        "v5_r_reliability": baseline,
        "v5_s_spatial": spatial,
    }
    selection = select_query_free_compositor(
        variants,
        baseline="v5_r_reliability",
        max_mean_dense_drop=max_mean_dense_drop,
        max_p05_dense_drop=0.01,
        max_unsupported_fraction=0.005,
        min_relation_gain=0.005,
    )
    assert selection["selected_variant"] == selected
    if forge_selection:
        selection = dict(selection)
        selection["selected_variant"] = "not_a_candidate"
        selection["promotion_allowed"] = True

    candidate_rows = []
    authority_candidates = {}
    for name, held_out in variants.items():
        checkpoint = root / f"{name}.pth"
        _write_field(checkpoint)
        audit = root / f"{name}.audit.json"
        audit.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "audit": "canonical_capability_fidelity_v1",
                    "protocol": {
                        "held_out_from_mpr": True,
                        "frame_ids": [1, 2],
                        "benchmark_masks_opened": False,
                        "text_queries_opened": False,
                        "capability_map_source": "official_extracted",
                        "alpha_threshold": 0.02,
                        "support_eps": 1e-6,
                        "boundary_quantile": 0.2,
                        "residual_mode": "none",
                    },
                    "artifacts": {
                        "field_checkpoint": str(checkpoint.resolve()),
                        "field_checkpoint_sha256": _sha256(checkpoint),
                        "config_sha256": "5" * 64,
                        "resolved_config_sha256": "6" * 64,
                        "geometry_checkpoint_sha256": "7" * 64,
                        "radio_checkpoint_sha256": "a" * 64,
                        "feature_output_bundle_sha256": FEATURE_BUNDLE_SHA256,
                        "view_residual_checkpoint": "",
                        "boundary_residual_checkpoint": "",
                        "official_capability_sources": sources,
                    },
                    "aggregate": held_out,
                    "per_frame": [{"frame_id": 1}, {"frame_id": 2}],
                },
                sort_keys=True,
            )
        )
        candidate_rows.append(
            {
                "name": name,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "audit": str(audit.resolve()),
                "audit_sha256": _sha256(audit),
                "held_out_fidelity": held_out,
            }
        )
        authority_candidates[name] = {
            "checkpoint": _record(checkpoint),
            "audit": _record(audit),
        }
    report = root / "capacity_screen.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_manifest": str(manifest.resolve()),
                "run_manifest_sha256": _sha256(manifest),
                "benchmark_queries_opened": False,
                "benchmark_masks_opened": False,
                "fixed_selection_contract": selection_contract,
                "candidates": candidate_rows,
                "query_free_selection": selection,
            },
            sort_keys=True,
        )
    )
    report_record = _record(report)
    manifest_record = _record(manifest)
    completion = root / "capacity_screen.complete.json"
    completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "canonical-v5-capacity-screen-completion-v1",
                "screen": "canonical-v5-query-free-capacity",
                "scene": scene,
                "report": report_record,
                "run_manifest": manifest_record,
                "candidates": authority_candidates,
                "feature_output_bundle_sha256": FEATURE_BUNDLE_SHA256,
                "implementation_source_tree_sha256": "d" * 64,
                "benchmark_queries_opened": False,
                "benchmark_masks_opened": False,
            },
            sort_keys=True,
        )
    )
    authority = {
        "scene": scene,
        "report": report_record,
        "run_manifest": manifest_record,
        "completion_bundle": _record(completion),
        "candidates": authority_candidates,
    }
    return report, authority


def _registry(tmp_path: Path, authorities: list[dict]) -> tuple[Path, str]:
    path = tmp_path / "trusted_registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "canonical-v5-external-trusted-registry-v1",
                "screens": authorities,
            },
            sort_keys=True,
        )
    )
    return path, _sha256(path)


def _confirm(tmp_path: Path, screens: list[tuple[Path, dict]]) -> dict:
    registry, digest = _registry(tmp_path, [authority for _, authority in screens])
    return confirm(
        [report for report, _ in screens],
        trusted_registry=registry,
        expected_registry_sha256=digest,
    )


def test_cross_scene_confirmation_requires_same_query_free_candidate(
    tmp_path: Path,
) -> None:
    screens = [
        _screen(tmp_path, "scene0011_01", "v5_s_spatial"),
        _screen(tmp_path, "scene0046_00", "v5_s_spatial"),
    ]

    result = _confirm(tmp_path, screens)

    assert result["confirmation_status"] == (
        "cross_scene_query_free_candidate_confirmed"
    )
    assert result["confirmed_variant"] == "v5_s_spatial"
    assert result["benchmark_queries_opened"] is False


def test_cross_scene_confirmation_fails_closed_on_disagreement(
    tmp_path: Path,
) -> None:
    result = _confirm(
        tmp_path,
        [
            _screen(tmp_path, "scene0011_01", "v5_s_spatial"),
            _screen(tmp_path, "scene0046_00", "v5_r_reliability"),
        ],
    )

    assert result["confirmed_variant"] is None
    assert "inconsistent" in result["confirmation_status"]


def test_cross_scene_confirmation_rejects_repeated_scene(
    tmp_path: Path,
) -> None:
    screens = [
        _screen(tmp_path, "scene0011_01", "v5_s_spatial"),
        _screen(
            tmp_path,
            "scene0011_01",
            "v5_s_spatial",
            suffix="_duplicate",
        ),
    ]
    registry, digest = _registry(
        tmp_path, [authority for _, authority in screens]
    )

    with pytest.raises(ValueError, match="repeated"):
        confirm(
            [report for report, _ in screens],
            trusted_registry=registry,
            expected_registry_sha256=digest,
        )


def test_cross_scene_confirmation_rejects_selection_contract_drift(
    tmp_path: Path,
) -> None:
    screens = [
        _screen(tmp_path, "scene0011_01", "v5_s_spatial"),
        _screen(
            tmp_path,
            "scene0046_00",
            "v5_s_spatial",
            max_mean_dense_drop=0.002,
        ),
    ]

    with pytest.raises(ValueError, match="different frozen contracts"):
        _confirm(tmp_path, screens)


def test_cross_scene_confirmation_rejects_forged_selection(
    tmp_path: Path,
) -> None:
    screens = [
        _screen(
            tmp_path,
            "scene0011_01",
            "v5_s_spatial",
            forge_selection=True,
        ),
        _screen(
            tmp_path,
            "scene0046_00",
            "v5_s_spatial",
            forge_selection=True,
        ),
    ]

    with pytest.raises(ValueError, match="CPU replay"):
        _confirm(tmp_path, screens)


def test_cross_scene_confirmation_rejects_candidate_audit_tamper(
    tmp_path: Path,
) -> None:
    screens = [
        _screen(tmp_path, "scene0011_01", "v5_s_spatial"),
        _screen(tmp_path, "scene0046_00", "v5_s_spatial"),
    ]
    registry, digest = _registry(
        tmp_path, [authority for _, authority in screens]
    )
    audit = Path(screens[1][1]["candidates"]["v5_s_spatial"]["audit"]["path"])
    audit.write_text(json.dumps({"aggregate": _fidelity(0.99)}))

    with pytest.raises(ValueError, match="SHA-256"):
        confirm(
            [report for report, _ in screens],
            trusted_registry=registry,
            expected_registry_sha256=digest,
        )


def test_cross_scene_confirmation_rejects_registry_tamper(
    tmp_path: Path,
) -> None:
    screens = [
        _screen(tmp_path, "scene0011_01", "v5_s_spatial"),
        _screen(tmp_path, "scene0046_00", "v5_s_spatial"),
    ]
    registry, digest = _registry(
        tmp_path, [authority for _, authority in screens]
    )
    registry.write_text(registry.read_text() + "\n")

    with pytest.raises(ValueError, match="SHA-256"):
        confirm(
            [report for report, _ in screens],
            trusted_registry=registry,
            expected_registry_sha256=digest,
        )


def test_cross_scene_confirmation_function_requires_two_reports(
    tmp_path: Path,
) -> None:
    report, authority = _screen(
        tmp_path, "scene0011_01", "v5_s_spatial"
    )
    registry, digest = _registry(tmp_path, [authority])

    with pytest.raises(ValueError, match="at least two"):
        confirm(
            [report],
            trusted_registry=registry,
            expected_registry_sha256=digest,
        )


def test_v5_runner_freezes_aggressive_query_free_gate_and_safe_gpu1() -> None:
    runner = REPO_ROOT / "radio_gs/scripts/run_canonical_v5_capacity_screen.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    source = runner.read_text(encoding="utf-8")
    for chunk in source.split("<<'PY'\n")[1:]:
        code = chunk.split("\nPY\n", 1)[0]
        compile(code, str(runner), "exec")

    assert 'GPU="${GPU:-1}"' in source
    assert 'if [[ "$GPU" != "1" ]]' in source
    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-75}"' in source
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-52}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-1}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-0}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-0}"' in source
    assert 'GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-77}"' in source
    assert 'GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-75}"' in source
    assert 'GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"' in source
    assert 'GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-0}"' in source
    assert 'GPU_PEER_MAX_MEMORY_MIB="${GPU_PEER_MAX_MEMORY_MIB:-0}"' in source
    assert 'GPU_PEER_MAX_UTIL_PCT="${GPU_PEER_MAX_UTIL_PCT:-100}"' in source
    assert (
        'RADIO_THERMAL_PACING_SECONDS_PER_IMAGE="${'
        'RADIO_THERMAL_PACING_SECONDS_PER_IMAGE:-8.0}"'
    ) in source
    assert 'FEATURE_STAGING="${VERIFIED_FEATURE_DIR}.incomplete"' in source
    assert "BASHPID" not in source
    assert 'flock -n "$FEATURE_STAGING_LOCK_FD"' in source
    assert 'flock -n "$RUN_LOCK_FD"' in source
    assert 'RUN_LOCK="$OUTPUT_ROOT/.canonical_v5.lock"' in source
    assert 'output/.physical_gpu1.lock' in source
    assert "CANONICAL_V5_PHYSICAL_GPU_LOCK_HELD" in source
    assert "--query-compute-apps=gpu_uuid,pid" in source
    assert "V5_CONTINUOUS_CANARY_RECORD_SHA256" in source
    assert "uncharacterized_continuous_hard_abort_only" in source
    assert "--resume-partial" in source
    assert "--radio-thermal-pacing-seconds-per-image" in source
    assert '"resume_partial": True' in source
    assert '"staging": "deterministic_sibling_incomplete_v1"' in source
    assert (
        '"invalid_or_missing_frame_policy": "recompute_entire_frame_v1"'
        in source
    )
    assert '"final_output_bundle": "radio-feature-output-bundle-v1"' in source
    assert "implementation_source_tree" in source
    assert "validate_torch_stage" in source
    assert "validate_audit_stage" in source
    assert "stale_stage_error" in source
    assert "load_mpr_cache" in source
    assert "load_canonical_field_checkpoint" in source
    assert '"max_mean_dense_drop": 0.005' in source


def test_v5_runner_global_gpu1_lock_fails_before_any_gpu_probe(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_repo"
    scripts = fake_root / "radio_gs" / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_canonical_v5_capacity_screen.sh"
    shutil.copy2(
        REPO_ROOT / "radio_gs/scripts/run_canonical_v5_capacity_screen.sh",
        runner,
    )
    (scripts / "run_with_gpu_thermal_guard.sh").write_text("#!/bin/sh\n")
    output = fake_root / "output"
    output.mkdir()
    lock_path = output / ".physical_gpu1.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(runner)],
            cwd=fake_root,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            capture_output=True,
            timeout=10,
        )

    assert result.returncode != 0
    assert "another RADIO-GS task owns physical GPU1" in result.stderr


def test_v5_runner_output_root_lock_fails_before_any_gpu_probe(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake_repo"
    scripts = fake_root / "radio_gs" / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_canonical_v5_capacity_screen.sh"
    shutil.copy2(
        REPO_ROOT / "radio_gs/scripts/run_canonical_v5_capacity_screen.sh",
        runner,
    )
    (scripts / "run_with_gpu_thermal_guard.sh").write_text("#!/bin/sh\n")
    global_lock = fake_root / "output" / ".physical_gpu1.lock"
    global_lock.parent.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    required = []
    for name in ("config.yaml", "geometry.pth", "radio.pth"):
        path = tmp_path / name
        path.write_text("placeholder\n")
        required.append(path)
    lock_path = output / ".canonical_v5.lock"
    with global_lock.open("w") as global_handle, lock_path.open("w") as handle:
        fcntl.flock(global_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "CANONICAL_V5_PHYSICAL_GPU_LOCK_HELD": "1",
            "CANONICAL_V5_PHYSICAL_GPU_LOCK_FD": str(global_handle.fileno()),
            "CONFIG": str(required[0]),
            "GEOMETRY_CHECKPOINT": str(required[1]),
            "RADIO_CHECKPOINT": str(required[2]),
            "EXCLUDE_FRAME_IDS": "1",
            "OUTPUT_ROOT": str(output),
        }
        result = subprocess.run(
            ["bash", str(runner)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            pass_fds=(global_handle.fileno(),),
        )

    assert result.returncode == 2
    assert "another canonical-v5 runner owns OUTPUT_ROOT" in result.stderr


def test_v5_stage_validator_reopens_terminal_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    runner = REPO_ROOT / "radio_gs/scripts/run_canonical_v5_capacity_screen.sh"
    source = runner.read_text(encoding="utf-8")
    fragment = source.split("validate_torch_stage() {", 1)[1]
    code = fragment.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    validator = tmp_path / "validate_stage.py"
    validator.write_text(code, encoding="utf-8")
    mpr_root = tmp_path / "mpr"
    mpr_root.mkdir()
    terminal = mpr_root / "raw_radio_mean_resultant.pt"
    responsibility = mpr_root / "shared_responsibility.pt"
    responsibility_contract = {
        "selected_frame_indices": [0],
        "feature_height": 1,
        "feature_width": 2,
    }
    torch.save(
        {
            "schema_version": 1,
            "metadata": responsibility_contract,
            "assignments": [
                {
                    "gaussian_ids": torch.tensor([0, 1], dtype=torch.int32),
                    "pixel_ids": torch.tensor([0, 1], dtype=torch.int32),
                    "weights": torch.ones(2),
                }
            ],
        },
        responsibility,
    )
    config = tmp_path / "config.yaml"
    geometry = tmp_path / "geometry.pth"
    config.write_text("config\n")
    geometry.write_bytes(b"geometry")
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyz_sha256 = hashlib.sha256(
        xyz.numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "feature_space": "radio",
        "num_declared_views": 1,
        "selected_frame_indices": [0],
        "xyz_sha256": xyz_sha256,
        "raster_reliability_mode": "mean_resultant",
        "normalize_each_view": True,
        "aggregation_mode": "raster_gaussian_top1",
        "observation_lifting_contract": {"name": "canonical-mpr-v1"},
        "config": str(config),
        "checkpoint": str(geometry),
        "excluded_frame_ids": [1],
        "registration_responsibility_contract": responsibility_contract,
        "registration_responsibility_cache_sha256": _sha256(responsibility),
        "shared_registration_responsibility": True,
        "feature_output_bundle_sha256": FEATURE_BUNDLE_SHA256,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
    }
    torch.save(
        {
            "xyz": xyz,
            "features": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            "valid": torch.ones(2, dtype=torch.bool),
            "view_counts": torch.ones(2, dtype=torch.long),
            "reliability": torch.ones(2, 3),
            "geometry_fingerprint": {
                "num_gaussians": 2,
                "xyz_sha256": xyz_sha256,
            },
            "metadata": metadata,
        },
        terminal,
    )
    Path(str(terminal) + ".json").write_text(
        json.dumps({"output": str(terminal), "metadata": metadata})
    )
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config": str(config),
                "geometry_checkpoint": str(geometry),
                "exclude_frame_ids": [1],
                "feature_extraction_safety_contract": {
                    "final_output_bundle_sha256": FEATURE_BUNDLE_SHA256
                },
            }
        )
    )
    command = [
        "bash",
        str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
        str(validator),
        "mpr_raw_mean_resultant",
        str(terminal),
        str(manifest),
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    valid = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert valid.returncode == 0, valid.stderr

    terminal.write_bytes(b"corrupt")
    invalid = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert invalid.returncode != 0
    assert "cannot be safely reopened" in invalid.stderr
