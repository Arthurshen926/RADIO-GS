from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from radio_gs.scripts import nvos_registered_region_v3_authority as authority
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _candidate_method_manifest_contract,
)
from radio_gs.scripts.nvos_registered_region_v3_authority import (
    REQUIRED_NON_PACKAGE_SOURCES,
    build_runtime_closure,
    verify_manifest_closure,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh"
CANDIDATE = (
    REPO_ROOT
    / "paper/artifacts/nvos_registered_region_v3_candidate_20260731.yaml"
)
AUTHORITY = (
    REPO_ROOT
    / "radio_gs/scripts/nvos_registered_region_v3_authority.py"
)
THERMAL_GUARD = REPO_ROOT / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"


def _minimal_snapshot(root: Path) -> Path:
    required = {
        "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh",
        "radio_gs/scripts/nvos_registered_region_v3_authority.py",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "radio_gs/scripts/run_repo_python.sh",
        "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py",
        "radio_gs/scripts/aggregate_registered_prompt_closeout.py",
        *REQUIRED_NON_PACKAGE_SOURCES,
    }
    for relative in sorted(required):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen source: {relative}\n", encoding="utf-8")
    extra = root / "radio_gs/querying/transitive.py"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    return extra


def _make_readonly_snapshot(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _runner_method_contract() -> dict[str, object]:
    shell = RUNNER.read_text(encoding="utf-8")
    embedded = shell.split("<<'PY'\n", maxsplit=1)[1].split(
        "\nPY\n", maxsplit=1
    )[0]
    tree = ast.parse(embedded)
    payload = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "payload"
            for target in node.targets
        )
    )
    method = next(
        value
        for key, value in zip(payload.keys, payload.values)
        if isinstance(key, ast.Constant) and key.value == "method_contract"
    )
    return ast.literal_eval(method)


def _evaluator_args(method: dict[str, object]) -> argparse.Namespace:
    prompt = method["prompt_registration"]
    score = method["score_render"]
    graph = method["graph"]
    solver = method["solver"]
    assert isinstance(prompt, dict)
    assert isinstance(score, dict)
    assert isinstance(graph, dict)
    assert isinstance(solver, dict)
    return argparse.Namespace(
        support_mode=method["support_mode"],
        region_space=method["region_space"],
        prompt_registration_mode=prompt["mode"],
        prompt_registration_scale=prompt["scale"],
        alpha_threshold=prompt["alpha_threshold"],
        depth_tolerance=prompt["depth_tolerance"],
        relative_depth_tolerance=prompt["relative_depth_tolerance"],
        registered_seed_construction=method["seed_construction"],
        registered_observation_fusion=method["observation_fusion"],
        registered_seed_unary_weight=method["registered_seed_unary_weight"],
        registered_observation_confidence=method["observation_confidence"],
        registered_observation_mass_scale=method["observation_mass_scale"],
        registered_observation_coverage_power=method[
            "observation_coverage_power"
        ],
        support_threshold=method["prompt_support_threshold"],
        prototype_count=method["prototype_count"],
        prototype_strategy=method["prototype_strategy"],
        appearance_weight=method["appearance_weight"],
        boundary_weight=method["boundary_weight"],
        prototype_temperature=method["prototype_temperature"],
        feature_calibration=method["feature_calibration"],
        background_centroids=method["background_centroids"],
        score_calibration=method["score_calibration"],
        negative_spatial_mode=method["negative_spatial_mode"],
        registered_selection_mode=method["diagnostic_selection_mode"],
        registered_readout_stage=method["final_readout"],
        graph_policy=graph["policy"],
        component_graph_policy=graph["component_policy"],
        graph_legacy_residual=graph["legacy_residual"],
        channel_confidence_mode=graph["channel_confidence_mode"],
        score_render_resolution=score["resolution"],
        score_render_scale=score["scale"],
        valid_support_normalization=score["valid_support_normalization"],
        valid_support_coverage_power=score["valid_support_coverage_power"],
        feature_contribution_gamma=score["feature_contribution_gamma"],
        score_chunk_size=score["score_chunk_size"],
        solver_support_threshold=score["pixel_threshold"],
        solver_type=solver["type"],
        solver_iterations=solver["iterations"],
        solver_residual=solver["residual"],
        solver_unary_temperature=solver["unary_temperature"],
        laplacian_weight=solver["laplacian_weight"],
        cg_iterations=solver["cg_iterations"],
        cg_tolerance=solver["cg_tolerance"],
        hard_seed_threshold=solver["hard_seed_threshold"],
        hard_seed_conflict_policy=solver["hard_seed_conflict_policy"],
        hard_seed_conflict_margin=solver["hard_seed_conflict_margin"],
        component_edge_threshold=solver["component_edge_threshold"],
        seeded_component_min_weight=solver["seeded_component_min_weight"],
        canonical_reliability_cache=method["canonical_reliability_cache"],
        diagnostic_graph_affinity_override=method[
            "diagnostic_graph_affinity_override"
        ],
        require_asset_hashes=method["asset_hash_verification_required"],
    )


def test_v3_runner_method_contract_exactly_matches_evaluator() -> None:
    method = _runner_method_contract()

    assert method == _candidate_method_manifest_contract(_evaluator_args(method))
    assert method["observation_fusion"] == "hard_seed_anchored_probability"
    assert method["strong_unary"] == {
        "policy": "unit_confidence_on_shared_hard_seed_rows",
        "anchor_threshold_source": "solver.hard_seed_threshold",
        "anchor_threshold": 0.20,
        "formula": (
            "a=1[c>0 and abs(s)>=tau]; c_eff=a+(1-a)c; "
            "p=(1-c_eff)p_field+c_eff*q"
        ),
        "new_numeric_constant": False,
    }


def test_v3_runner_is_shell_valid_and_uses_conservative_thermal_guard() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    shell = RUNNER.read_text(encoding="utf-8")
    python_wrapper = (
        REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"
    ).read_text(encoding="utf-8")
    thermal_guard = THERMAL_GUARD.read_text(encoding="utf-8")

    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"' in shell
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"' in shell
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"' in shell
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"' in shell
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"' in shell
    assert 'GPU_PEER_INDEX="${GPU_PEER_INDEX:-}"' in shell
    assert 'GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-0}"' in shell
    assert 'GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-0}"' in shell
    assert 'GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"' in shell
    assert 'GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-0}"' in shell
    assert 'GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-200}' not in shell
    assert 'GPU_PEER_MAX_MEMORY_MIB="${GPU_PEER_MAX_MEMORY_MIB:-0}"' in shell
    assert 'GPU_PEER_MAX_UTIL_PCT="${GPU_PEER_MAX_UTIL_PCT:-100}"' in shell
    assert "physical GPU1 already has compute owner(s)" in shell
    assert 'MAIN_OUTPUT_ROOT="/root/RADIO-GS/output"' in shell
    assert 'GLOBAL_GPU1_LOCK="$MAIN_OUTPUT_ROOT/.physical_gpu1.lock"' in shell
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_OUTPUT_ROOT/' in shell
    assert 'SOURCE_ROOT="${SOURCE_ROOT:-$MAIN_OUTPUT_ROOT/' in shell
    assert 'PARENT_ASSET_MANIFEST="${PARENT_ASSET_MANIFEST:-$MAIN_OUTPUT_ROOT/' in shell
    assert "surface_gpu1_lock_supervisor.py" in shell
    assert 'MAIN_OUTPUT_REAL="$(readlink -f -- "$MAIN_OUTPUT_ROOT")"' in shell
    assert 'OUTPUT_ROOT="$(realpath -ms -- "$OUTPUT_ROOT")"' in shell
    assert 'OUTPUT_ROOT_REAL="$(readlink -m -- "$OUTPUT_ROOT")"' in shell
    assert '"$MAIN_OUTPUT_REAL"/*' in shell
    assert '"scene_attempt_telemetry_root"' in shell
    assert 'telemetry="$attempt_root/telemetry.csv"' in shell
    assert "verify-inherited" in shell
    assert "RADIO_GS_GPU1_SINGLETON_FD" in shell
    assert GPU1_SINGLETON_PROTOCOL_FRAGMENT in shell
    assert 'exec {gpu1_lock}<>"$GLOBAL_GPU1_LOCK"' not in shell
    assert 'EXPECTED_GPU1_UUID="GPU-0eac2c76-4004-49eb-bc0c-a9a30aec041a"' in shell
    assert 'EXPECTED_GPU1_PROC_BUS_ID="0000:82:00.0"' in shell
    assert 'EXPECTED_GPU1_NVIDIA_BUS_ID="00000000:82:00.0"' in shell
    assert "--query-compute-apps=gpu_uuid,pid" in shell
    assert '"source_snapshot_root"' in shell
    assert '"source_snapshot_import_root"' in shell
    assert '"source_snapshot_tree_sha256"' in shell
    assert '"runtime_closure": runtime_closure' in shell
    assert 'verify_runtime_closure' in shell
    assert "verify-readonly-snapshot" in shell
    assert "verify-output-tree" in shell
    assert "candidate thermal plan differs from the queue safety contract" in shell
    assert "--evaluator-log" in shell
    assert 'assert_gpu1_identity_unowned "pre_${scene}"' in shell
    assert 'assert_gpu1_identity_unowned "post_${scene}"' in shell
    assert 'exec {run_lock}>"$LOCK_ROOT/run.lock"' in shell
    assert 'FROZEN_CUDA_DEVICE_ORDER="PCI_BUS_ID"' in shell
    assert 'FROZEN_PYTHONDONTWRITEBYTECODE="1"' in shell
    assert "refusing non-frozen PYTHONDONTWRITEBYTECODE" in shell
    assert 'PYTHONDONTWRITEBYTECODE="$FROZEN_PYTHONDONTWRITEBYTECODE"' in shell
    assert "export PYTHONDONTWRITEBYTECODE=1" in python_wrapper
    assert 'FROZEN_NUMBA_CACHE_DIR="/root/.cache/radio_gs/numba"' in shell
    assert "refusing non-frozen NUMBA_CACHE_DIR" in shell
    assert 'NUMBA_CACHE_DIR="$FROZEN_NUMBA_CACHE_DIR"' in shell
    assert 'export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/root/.cache/radio_gs/numba}"' in python_wrapper
    assert (
        'FROZEN_GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"'
        in shell
    )
    assert "refusing non-frozen GPU_OWNER_PID_NAMESPACE_MODE" in shell
    assert (
        'GPU_OWNER_PID_NAMESPACE_MODE="$FROZEN_GPU_OWNER_PID_NAMESPACE_MODE"'
        in shell
    )
    assert 'GPU_OWNER_PID_NAMESPACE_MODE="${GPU_OWNER_PID_NAMESPACE_MODE:-strict}"' in thermal_guard
    assert "runtime_owner_audit_host_pid_singleton" in thermal_guard
    assert "bound_host_owner_pid" in thermal_guard
    assert 'CUDA_VISIBLE_DEVICES="$GPU_UUID"' in shell
    assert 'CUDA_VISIBLE_DEVICES="$GPU"' not in shell
    assert 'NVIDIA_VISIBLE_DEVICES="$GPU_UUID"' in shell
    assert "refusing numeric or foreign CUDA_VISIBLE_DEVICES" in shell
    assert "prepare-scene" in shell
    assert "postcheck-scene" in shell
    assert "finalize-scene" in shell
    assert "validate-scene" in shell
    assert "--gpu-attestation-output" in shell
    assert "GPU_OWNER_AUDIT_LOG" in shell
    assert "existing result failed exclusive-GPU receipt validation" in shell


GPU1_SINGLETON_PROTOCOL_FRAGMENT = (
    "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
)


def test_v3_snapshot_closure_is_git_independent_and_detects_any_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot_without_git"
    extra = _minimal_snapshot(snapshot)
    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(snapshot))
    before = build_runtime_closure(snapshot)
    assert not (snapshot / ".git").exists()
    assert before["source_snapshot_root"] == str(snapshot)
    loaded = before["runtime"]["loaded_modules"]
    assert {
        "gsplat",
        "yaml",
        "plyfile",
        "timm",
        "numba",
        "tqdm",
        "huggingface_hub",
        "safetensors",
    } <= set(loaded)
    for record in loaded.values():
        assert record["version"]
        assert len(record["origin"]["sha256"]) == 64
        for path, extension in record["extension_binaries"].items():
            assert Path(path).is_absolute()
            assert extension["path"] == path
            assert len(extension["sha256"]) == 64
    import cv2

    cv2_record = loaded["cv2"]
    assert cv2_record["version"] == cv2.__version__
    assert cv2_record["module_reported_version"] == cv2.__version__
    cv2_extensions = [
        extension
        for path, extension in cv2_record["extension_binaries"].items()
        if path.endswith("/cv2/cv2.abi3.so")
    ]
    assert len(cv2_extensions) == 1
    assert cv2_extensions[0]["bytes"] > 80_000_000
    assert any(
        instance["version"] == cv2.__version__
        or instance["version"].startswith(f"{cv2.__version__}.")
        for instance in cv2_record["distribution_instances"]
    )
    gsplat = before["runtime"]["gsplat_runtime"]
    assert gsplat["distribution_version"] == loaded["gsplat"]["version"]
    assert gsplat["cuda_extension"]["path"].endswith(
        "/py39_cu118/gsplat_cuda/gsplat_cuda.so"
    )
    assert len(gsplat["source_tree"]["files"]) > 100
    assert (
        "radio_gs/querying/transitive.py"
        in before["repository_sources"]["files"]
    )

    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    readonly = build_runtime_closure(snapshot)
    assert readonly["repository_sources"]["digest"] == before[
        "repository_sources"
    ]["digest"]

    extra.chmod(0o644)
    extra.write_text("VALUE = 2\n", encoding="utf-8")
    after = build_runtime_closure(snapshot)
    assert after["repository_sources"]["digest"] != before[
        "repository_sources"
    ]["digest"]


def test_v3_manifest_closure_rejects_source_drift_and_import_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    extra = _minimal_snapshot(snapshot)
    _make_readonly_snapshot(snapshot)
    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(snapshot))
    closure = build_runtime_closure(snapshot)
    fake_output_identity = {
        "logical_main_root": "/root/RADIO-GS/output",
        "logical_output_root": "/root/RADIO-GS/output/test",
    }
    monkeypatch.setattr(
        authority,
        "verify_output_tree",
        lambda record: {
            "status": "output_tree_real_directory_only",
            "output_identity": dict(record),
        },
    )
    runner_record = closure["repository_sources"]["files"][
        "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh"
    ]
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_snapshot_root": str(snapshot),
                "source_snapshot_import_root": str(snapshot),
                "source_snapshot_tree_sha256": closure[
                    "repository_sources"
                ]["digest"],
                "runner_sha256": runner_record["sha256"],
                "runtime_closure": closure,
                "source_snapshot_permissions": {
                    "status": "readonly_non_live_source_snapshot_verified",
                    "source_permissions": closure["source_snapshot_permissions"],
                },
                "output_identity": fake_output_identity,
            }
        ),
        encoding="utf-8",
    )
    verified = verify_manifest_closure(manifest, repo_root=snapshot)
    assert verified["status"] == "runtime_closure_verified"

    extra.chmod(0o644)
    extra.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime closure changed"):
        verify_manifest_closure(manifest, repo_root=snapshot)

    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(tmp_path / "escape"))
    with pytest.raises(ValueError, match="import root escaped"):
        build_runtime_closure(snapshot)


def test_v3_closure_rejects_symlinked_source_directory_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    _minimal_snapshot(snapshot)
    target = snapshot / "outside"
    target.mkdir()
    (target / "evil.py").write_text("VALUE = 1\n", encoding="utf-8")
    querying = snapshot / "radio_gs/querying"
    shutil.rmtree(querying)
    querying.symlink_to("../outside", target_is_directory=True)
    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(snapshot))
    with pytest.raises(ValueError, match="source tree contains a symlink entry"):
        build_runtime_closure(snapshot)


def test_v3_closure_rejects_required_source_symlink_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    _minimal_snapshot(snapshot)
    target = snapshot / "outside_paper"
    target.mkdir()
    (target / "nvos_registered_region_v3_candidate_20260731.yaml").write_text(
        "status: frozen\n",
        encoding="utf-8",
    )
    shutil.rmtree(snapshot / "paper")
    (snapshot / "paper").symlink_to("outside_paper", target_is_directory=True)
    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(snapshot))
    with pytest.raises(ValueError, match="nested symlink"):
        build_runtime_closure(snapshot)


def test_v3_launch_requires_non_live_readonly_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _minimal_snapshot(snapshot)
    with pytest.raises(ValueError, match="read-only source snapshot"):
        authority.verify_readonly_source_snapshot(snapshot)
    _make_readonly_snapshot(snapshot)
    verified = authority.verify_readonly_source_snapshot(snapshot)
    assert verified["status"] == "readonly_non_live_source_snapshot_verified"
    assert verified["source_permissions"]["writable_entries"] == []


def test_v3_output_tree_and_artifact_binding_reject_directory_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    attempts = output / "scene_attempts"
    attempts.mkdir(parents=True)
    info = output.stat()
    identity = {
        "logical_main_root": str(tmp_path),
        "resolved_main_root": str(tmp_path),
        "logical_output_root": str(output),
        "resolved_output_root": str(output),
        "main_target_device": int(tmp_path.stat().st_dev),
        "main_target_inode": int(tmp_path.stat().st_ino),
        "output_device": int(info.st_dev),
        "output_inode": int(info.st_ino),
    }
    monkeypatch.setattr(
        authority,
        "_validated_output_identity",
        lambda _record: dict(identity),
    )
    artifact = attempts / "telemetry.csv"
    binding = authority._artifact_binding(
        artifact,
        output_identity_record=identity,
        label="test artifact",
    )
    assert binding["logical_path"] == str(artifact)
    assert binding["resolved_path"] == str(artifact)

    outside = tmp_path / "outside"
    outside.mkdir()
    attempts.rmdir()
    attempts.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="output tree contains a symlink entry"):
        authority.verify_output_tree(identity)


def test_v3_runner_rejects_forged_lock_environment_before_gpu_or_output() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "RADIO_GS_GPU1_LOCK_FD": "999999",
            "RADIO_GS_GPU1_LOCK_PATH": "/root/RADIO-GS/output/.physical_gpu1.lock",
            "RADIO_GS_GPU1_SINGLETON_FD": "999998",
            "RADIO_GS_GPU1_SINGLETON_PROTOCOL": GPU1_SINGLETON_PROTOCOL_FRAGMENT,
            "GPU": "0",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert (
        "mutable live worktree" in completed.stderr
        or "Bad file descriptor" in completed.stderr
        or "fstat" in completed.stderr
    )
    assert "got GPU=0" not in completed.stderr


def test_v3_thermal_guard_owner_check_is_launch_adjacent_and_runtime_exclusive() -> None:
    subprocess.run(["bash", "-n", str(THERMAL_GUARD)], check=True)
    shell = THERMAL_GUARD.read_text(encoding="utf-8")
    preflight = shell.index('prelaunch_owners="$(target_compute_owners)"')
    launch = shell.index('setsid "$@" &', preflight)
    between = shell[preflight:launch]
    assert "sleep" not in between
    assert "setsid" not in between
    assert launch - preflight < 700
    assert 'audit_target_compute_owners()' in shell
    assert 'resolve_owner_process_group()' in shell
    assert 'if [[ "$owner_pgid" == "$child_pid" ]]' in shell
    assert 'append_owner_audit "$OWNER_AUDIT_EVENT"' in shell
    assert '&& ! -e "/proc/$owner_pid"' in shell
    assert "foreign_compute_owner_" in shell
    assert 'append_owner_audit "postexit_owner_clear"' in shell
    assert shell.rindex("verify_target_gpu_released") < shell.rindex(
        'append_owner_audit "postexit_owner_clear"'
    )


def test_v3_scene_receipt_binds_attestation_owner_timeline_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_uuid = "GPU-0eac2c76-4004-49eb-bc0c-a9a30aec041a"
    gpu_bus = "00000000:82:00.0"
    scene = "fern"
    root_info = tmp_path.stat()
    test_output_identity = {
        "logical_main_root": str(tmp_path),
        "resolved_main_root": str(tmp_path),
        "logical_output_root": str(tmp_path),
        "resolved_output_root": str(tmp_path),
        "main_target_device": int(root_info.st_dev),
        "main_target_inode": int(root_info.st_ino),
        "output_device": int(root_info.st_dev),
        "output_inode": int(root_info.st_ino),
    }
    monkeypatch.setattr(
        authority,
        "_validated_output_identity",
        lambda _record: dict(test_output_identity),
    )
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "frozen": True,
                "output_identity": test_output_identity,
                "thermal_safety_contract": {"test": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    guard = tmp_path / "guard.sh"
    guard.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = tmp_path / "fern_evaluation.json"
    result.write_text('{"foreground_iou":0.5}\n', encoding="utf-8")
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        ",".join(authority.TELEMETRY_COLUMNS)
        + "\n"
        + "2026-08-01T00:00:00+08:00,1,0000:82:00.0,50,100,300,80,1000,P2,sample\n"
        + "2026-08-01T00:00:01+08:00,1,0000:82:00.0,50,20,300,0,0,P8,cuda_release_verified_no_compute_owner\n",
        encoding="utf-8",
    )
    owner_audit = tmp_path / "owner.csv"
    owner_audit.write_text(
        ",".join(authority.OWNER_AUDIT_COLUMNS)
        + "\n"
        + f"t0,{gpu_uuid},4242,,,,prelaunch_owner_clear\n"
        + f"t1,{gpu_uuid},4242,777,777,,runtime_owner_audit\n"
        + f"t2,{gpu_uuid},4242,,,,postexit_owner_clear\n",
        encoding="utf-8",
    )
    attestation = tmp_path / "attestation.json"
    write_frozen_json(
        attestation,
        {
            "schema_version": 2,
            "artifact_type": "nvos-v3-cuda-child-attestation-v2",
            "status": "torch_cuda0_live_owner_matches_physical_gpu1_uuid_and_pci",
            "scene": scene,
            "observed_epoch": 1,
            "hostname": "test-host",
            "environment": {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "GPU_OWNER_PID_NAMESPACE_MODE": (
                    authority.GPU_OWNER_PID_NAMESPACE_MODE
                ),
                "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
            },
            "expected_gpu": {
                "physical_index": 1,
                "uuid": gpu_uuid,
                "pci_bus_id": gpu_bus,
            },
            "torch_cuda": {
                "visible_device_count": 1,
                "current_device": 0,
                "device": "cuda:0",
                "device_name": "RTX 3090",
                "compute_capability": [8, 6],
                "total_memory": 24 * 1024**3,
                "torch_version": "test",
                "torch_cuda_build": "11.8",
            },
            "process_namespace_pids": [777],
            "nvidia_inventory_row": ["1", gpu_uuid, gpu_bus, "RTX 3090"],
            "nvidia_preallocation_owner_rows": [],
            "nvidia_compute_owner_rows": [[gpu_uuid, "777", "python"]],
            "owner_pid_binding": "process_namespace_pid",
            "attestation_mechanism": authority.CUDA_ATTESTATION_MECHANISM,
        },
    )
    host_pid_attestation = tmp_path / "host_pid_attestation.json"
    host_pid_payload = json.loads(attestation.read_text(encoding="utf-8"))
    host_pid_payload["nvidia_compute_owner_rows"] = [
        [gpu_uuid, "888", "python"]
    ]
    host_pid_payload["owner_pid_binding"] = (
        "exclusive_invisible_host_pid_singleton_after_clear"
    )
    write_frozen_json(host_pid_attestation, host_pid_payload)
    host_pid_validated = authority._validate_cuda_attestation(
        host_pid_attestation,
        scene=scene,
        gpu_uuid=gpu_uuid,
        gpu_bus_id=gpu_bus,
    )
    assert host_pid_validated["payload"]["nvidia_compute_owner_rows"][0][1] == "888"

    forged_pid_binding = tmp_path / "forged_pid_binding.json"
    host_pid_payload["owner_pid_binding"] = "process_namespace_pid"
    write_frozen_json(forged_pid_binding, host_pid_payload)
    with pytest.raises(ValueError, match="PID namespace binding"):
        authority._validate_cuda_attestation(
            forged_pid_binding,
            scene=scene,
            gpu_uuid=gpu_uuid,
            gpu_bus_id=gpu_bus,
        )

    command = tmp_path / "command.json"
    argv = [
        "python",
        "eval.py",
        "--scene-id",
        scene,
        "--device",
        "cuda:0",
        "--candidate-id",
        "registered-region-v3",
        "--gpu-attestation-output",
        str(attestation),
        "--expected-gpu-uuid",
        gpu_uuid,
        "--expected-gpu-bus-id",
        gpu_bus,
        "--output-dir",
        str(result.parent),
    ]
    authority.prepare_scene_command(
        output=command,
        run_manifest=manifest,
        scene=scene,
        result=result,
        telemetry=telemetry,
        owner_audit=owner_audit,
        attestation=attestation,
        postcheck=tmp_path / "postcheck.json",
        receipt=tmp_path / "receipt.json",
        evaluator_log=tmp_path / "evaluator.log",
        guard=guard,
        gpu_uuid=gpu_uuid,
        gpu_bus_id=gpu_bus,
        command=argv,
    )
    postcheck = tmp_path / "postcheck.json"
    closure_sha256 = "0" * 64
    write_frozen_json(
        postcheck,
        {
            "schema_version": 1,
            "artifact_type": "nvos-v3-scene-postcheck-v1",
            "status": "closure_lock_uuid_pci_and_post_owner_verified",
            "scene": scene,
            "observed_epoch": 1,
            "run_manifest": file_record(manifest),
            "result": file_record(result),
            "runtime_closure_sha256": closure_sha256,
            "gpu_identity": {
                "physical_index": 1,
                "uuid": gpu_uuid,
                "pci_bus_id": gpu_bus,
            },
            "nvidia_inventory_row": ["1", gpu_uuid, gpu_bus],
            "proc_driver_identity": ["1", gpu_uuid, "0000:82:00.0"],
            "pcie_config_prefix_hex": "00" * 16,
            "compute_owners": [],
            "global_lock": {
                "path": "/root/RADIO-GS/output/.physical_gpu1.lock",
                "fd": 10,
                "device": 1,
                "inode": 1,
                "links": 1,
            },
            "kernel_singleton": {
                "protocol": GPU1_SINGLETON_PROTOCOL_FRAGMENT,
                "fd": 11,
                "socket_type": 1,
            },
        },
    )
    monkeypatch.setattr(
        authority,
        "verify_manifest_closure",
        lambda *_args, **_kwargs: {
            "runtime_closure_sha256": closure_sha256
        },
    )
    receipt = tmp_path / "receipt.json"
    authority.finalize_scene_receipt(
        output=receipt,
        command_record=command,
        postcheck=postcheck,
    )
    validated = authority.validate_scene_receipt(
        receipt,
        run_manifest=manifest,
        scene=scene,
        result=result,
    )
    assert validated["payload"]["owner_audit"]["child_pgid"] == 4242

    forged_receipt = tmp_path / "forged_receipt.json"
    forged_payload = json.loads(receipt.read_text(encoding="utf-8"))
    forged_payload["unbound_extra"] = True
    write_frozen_json(forged_receipt, forged_payload)
    with pytest.raises(ValueError, match="receipt schema/status"):
        authority.validate_scene_receipt(
            forged_receipt,
            run_manifest=manifest,
            scene=scene,
            result=result,
        )

    result.write_text('{"foreground_iou":0.6}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="scene result"):
        authority.validate_scene_receipt(
            receipt,
            run_manifest=manifest,
            scene=scene,
            result=result,
        )


def test_v3_owner_audit_rejects_foreign_process(tmp_path: Path) -> None:
    gpu_uuid = "GPU-0eac2c76-4004-49eb-bc0c-a9a30aec041a"
    audit = tmp_path / "foreign.csv"
    audit.write_text(
        ",".join(authority.OWNER_AUDIT_COLUMNS)
        + "\n"
        + f"t0,{gpu_uuid},42,,,,prelaunch_owner_clear\n"
        + f"t1,{gpu_uuid},42,7,,7,runtime_owner_audit\n"
        + f"t2,{gpu_uuid},42,,,,postexit_owner_clear\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exclusive child-PGID"):
        authority._validate_owner_audit(audit, gpu_uuid=gpu_uuid)


def test_v3_candidate_freezes_constants_before_gpu_diagnostic() -> None:
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["status"] == "theory_frozen_awaiting_one_shot_diagnostic"
    assert candidate["implementation"]["gpu_execution_started"] is False
    assert candidate["constant_provenance"] == {
        "mass_scale": 1.0,
        "mass_scale_basis": (
            "one_native_alpha_weighted_pixel_is_one_poisson_observation_unit"
        ),
        "coverage_power": 1.0,
        "coverage_power_basis": (
            "exact_labeled_fraction_without_shape_distortion"
        ),
        "hard_seed_threshold": 0.20,
        "hard_seed_threshold_basis": (
            "inherited_shared_solver_hard_evidence_contract"
        ),
        "hard_seed_threshold_new_for_v3": False,
        "unary_temperature": 0.10,
        "unary_temperature_basis": (
            "inherited_registered_region_solver_contract"
        ),
        "pixel_threshold": 0.50,
        "pixel_threshold_basis": "fixed_bernoulli_decision_boundary",
        "target_metric_or_mask_calibration": False,
    }
    assert candidate["thermal_plan"] == {
        "physical_gpu": 1,
        "maximum_temperature_c": 78,
        "maximum_start_temperature_c": 65,
        "soft_pause_temperature_c": 75,
        "soft_resume_temperature_c": 70,
        "peer_gpu": None,
        "peer_pause_temperature_c": 0,
        "peer_resume_temperature_c": 0,
        "peer_quiet_seconds_before_launch": 0,
        "peer_maximum_power_w": 0.0,
        "peer_maximum_memory_mib": 0,
        "peer_maximum_utilization_percent": 100,
        "poll_seconds": 3,
        "maximum_power_limit_w": 300.5,
    }
    assert candidate["eligibility"]["main_result_eligible"] is False
