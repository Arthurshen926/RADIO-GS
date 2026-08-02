from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
    canonical_json_sha256,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _load_registered_forward_protocol_authority,
    _validate_candidate_run_manifest,
)
from radio_gs.scripts.stage_nvos_forward_beta_coverage_v1_snapshot import (
    AUTHORITY_RECEIPT_RELATIVE,
    CANDIDATE_ID,
    CANDIDATE_RELATIVE,
    ELIGIBILITY,
    HOST_MEMORY_POLICY,
    MAXIMUM_CONCURRENT_SCENE_EVALUATORS,
    ORDERED_TASKS,
    SERIAL_SCENE_GPU_PLAN,
    STAGING_MANIFEST_RELATIVE,
    StagingError,
    _namespace_from_method_contract,
    build_run_manifest_payload,
    load_and_validate_candidate,
    publish_authority_receipt,
    stage_snapshot,
    validate_candidate_payload,
    validate_snapshot,
    write_run_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_RELATIVE = Path(
    "radio_gs/scripts/run_nvos_forward_beta_coverage_v1_queue.sh"
)


def _bytes_record(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_candidate_is_exactly_evaluator_derived_and_nonexact() -> None:
    payload, method, digest, _ = load_and_validate_candidate(
        REPO_ROOT / CANDIDATE_RELATIVE
    )
    assert method == payload["method_contract"]
    assert digest == canonical_json_sha256(method)
    assert payload["candidate_id"] == CANDIDATE_ID
    assert payload["status"] == ELIGIBILITY
    assert payload["cohort"]["ordered_tasks"] == list(ORDERED_TASKS)
    assert payload["cohort"]["execution"] == (
        "fixed_full_eight_without_metric_continuation"
    )
    assert payload["eligibility"]["strict_unseen_eligible"] is False
    assert payload["eligibility"]["frozen_diagnostic_eligible"] is False
    assert payload["eligibility"]["main_result_eligible"] is False
    assert payload["protocol_reuse"]["score_semantics_exact"] is False
    assert payload["protocol_reuse"]["prediction_representation_exact"] is False

    changed = deepcopy(payload)
    changed["method_contract"]["solver"]["iterations"] = 13
    with pytest.raises(StagingError, match="evaluator-derived"):
        validate_candidate_payload(changed)


def _make_parent_assets(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    source_root = tmp_path / "parent-assets"
    source_artifacts: dict[str, dict[str, object]] = {}
    queue_scene_inputs: dict[str, dict[str, object]] = {}
    for scene in ORDERED_TASKS:
        scene_root = source_root / scene
        scene_sources: dict[str, object] = {}
        for name in (
            "canonical_d256_l128_capability_first.pth",
            "official_dino_sam3_views.pt",
            "shared_support_graph_k16.pt",
        ):
            artifact = _write(scene_root / name, f"{scene}:{name}\n")
            metadata = _write(
                scene_root / f"{name}.metadata.json", "{\"schema_version\": 1}\n"
            )
            record = _bytes_record(artifact)
            scene_sources[name] = {
                "path": str(artifact.resolve()),
                **record,
                "metadata_path": str(metadata.resolve()),
                "metadata_sha256": _bytes_record(metadata)["sha256"],
            }
        source_artifacts[scene] = scene_sources
        renderer_input = _write(
            tmp_path / "queue-inputs" / f"{scene}.json", f"{{\"scene\": \"{scene}\"}}\n"
        )
        queue_scene_inputs[scene] = {
            str(renderer_input.resolve()): _bytes_record(renderer_input)
        }

    protocol_hash = "test-protocol-hash"
    queue = _write(
        tmp_path / "queue_plan.json",
        json.dumps(
            {
                "benchmark": "nvos",
                "protocol_hash": protocol_hash,
                "scenes": [{"scene_id": scene} for scene in ORDERED_TASKS],
            }
        )
        + "\n",
    )
    benchmark = _write(
        tmp_path / "benchmark.json",
        json.dumps({"benchmark": "nvos", "protocol_hash": protocol_hash}) + "\n",
    )
    checkpoint = _write(tmp_path / "radio.pth.tar", "checkpoint\n")
    parent = {
        "candidate": "registered-region-v2",
        "scenes": list(ORDERED_TASKS),
        "source_root": str(source_root.resolve()),
        "queue_plan": str(queue.resolve()),
        "queue_plan_sha256": _bytes_record(queue)["sha256"],
        "benchmark_manifest": str(benchmark.resolve()),
        "benchmark_manifest_sha256": _bytes_record(benchmark)["sha256"],
        "radio_checkpoint": str(checkpoint.resolve()),
        "radio_checkpoint_sha256": _bytes_record(checkpoint)["sha256"],
        "source_artifacts": source_artifacts,
        "queue_scene_inputs": queue_scene_inputs,
    }
    parent_path = _write(
        tmp_path / "parent_manifest.json", json.dumps(parent) + "\n"
    )
    return source_root, queue, benchmark, checkpoint, {
        "payload": parent,
        "path": parent_path,
    }


def test_snapshot_and_run_manifest_pass_real_evaluator_consumers(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "main-repo-receipts" / "authority.json"
    snapshot = tmp_path / "source-snapshot"
    staging = stage_snapshot(
        repo_root=REPO_ROOT,
        snapshot_root=snapshot,
        authority_receipt=receipt,
    )
    assert validate_snapshot(snapshot) == staging
    assert snapshot.stat().st_mode & 0o222 == 0
    assert (snapshot / CANDIDATE_RELATIVE).stat().st_mode & 0o222 == 0
    assert (snapshot / AUTHORITY_RECEIPT_RELATIVE).read_bytes() == receipt.read_bytes()
    assert staging["target_data_read"] is False
    assert staging["gpu_state_read"] is False

    _, method, method_sha256, _ = load_and_validate_candidate(
        REPO_ROOT / CANDIDATE_RELATIVE
    )
    before = (receipt.stat().st_ino, receipt.read_bytes())
    reused, reused_bytes = publish_authority_receipt(
        repo_root=REPO_ROOT,
        output=receipt,
        method_contract=method,
        method_sha256=method_sha256,
    )
    assert (receipt.stat().st_ino, receipt.read_bytes()) == before
    assert reused_bytes == before[1]
    assert reused["candidate"]["method_contract_sha256"] == method_sha256

    source_root, queue, benchmark, checkpoint, parent = _make_parent_assets(
        tmp_path
    )
    required_implementation = (
        "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "radio_gs/querying/evidence_scorer.py",
        "radio_gs/rendering/contribution_compositor.py",
        "radio_gs/scripts/bind_nvos_forward_beta_protocol_authority.py",
        "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
        "radio_gs/scripts/validate_evaluation_protocol_freeze.py",
    )
    runtime_files = {
        relative: _bytes_record(snapshot / relative)
        for relative in required_implementation
    }
    runtime_sources = {
        "selection": ["test evaluator closure"],
        "files": runtime_files,
    }
    runtime_sources["digest"] = canonical_json_sha256(runtime_sources)
    runtime_closure = {
        "repository_import_root": str(snapshot.resolve()),
        "repository_sources": runtime_sources,
        "digest": canonical_json_sha256(runtime_sources),
    }
    mounted_output = tmp_path / "mounted-output"
    mounted_output.mkdir()
    canonical_output = tmp_path / "output"
    canonical_output.symlink_to(mounted_output, target_is_directory=True)
    output_root = canonical_output / "candidate"
    manifest_kwargs = dict(
        snapshot_root=snapshot,
        source_root=source_root,
        queue_plan=queue,
        benchmark_manifest=benchmark,
        radio_checkpoint=checkpoint,
        parent_asset_manifest=parent["path"],
        output_root=output_root,
        runner=snapshot / RUNNER_RELATIVE,
        thermal_guard=snapshot
        / "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        gpu_authority=snapshot
        / "radio_gs/scripts/nvos_registered_region_v3_authority.py",
        runtime_closure=runtime_closure,
        source_snapshot_authority={
            "status": "readonly_non_live_source_snapshot_verified"
        },
        thermal_safety_contract={"physical_gpu": 1},
        output_identity={"test": True},
    )
    payload = build_run_manifest_payload(**manifest_kwargs)
    real_output_root = output_root.resolve()
    payload_from_real_root = build_run_manifest_payload(
        **{**manifest_kwargs, "output_root": real_output_root}
    )
    assert payload_from_real_root == payload

    real_output_root.mkdir()
    run_manifest = write_run_manifest(
        real_output_root / "run_manifest.json", payload
    )
    before_resume = (run_manifest.stat().st_ino, run_manifest.read_bytes())
    assert write_run_manifest(run_manifest, payload) == run_manifest
    assert (run_manifest.stat().st_ino, run_manifest.read_bytes()) == before_resume
    with pytest.raises(StagingError, match="refuses symlink"):
        write_run_manifest(output_root / "run_manifest.json", payload)
    assert payload["method_contract_sha256"] == method_sha256
    assert payload["candidate_method_contract_sha256"] == method_sha256
    assert (
        payload["registered_forward_protocol_authority_sha256"]
        == canonical_json_sha256(
            payload["registered_forward_protocol_authority"]
        )
    )
    assert "protocol_authority" not in payload
    assert payload["execution"] == "fixed_full_eight_without_metric_continuation"
    assert payload["target_metric_controls_continuation"] is False
    assert payload["scene_gpu_assignment"] == {
        "policy": "fixed_before_execution_no_target_metric_input",
        "gpu0": ["fern", "flower", "fortress", "horns_center"],
        "gpu1": ["horns_left", "leaves", "orchids", "trex"],
    }
    assert payload["physical_gpu_binding"] == (
        "runtime_inventory_index_uuid_pci_under_independent_flock"
    )
    assert payload["maximum_concurrent_scene_evaluators"] == 1
    assert payload["host_memory_policy"] == HOST_MEMORY_POLICY
    assert payload["serial_scene_gpu_plan"] == list(SERIAL_SCENE_GPU_PLAN)

    fern_sources = parent["payload"]["source_artifacts"]["fern"]
    args = _namespace_from_method_contract(method)
    args.run_manifest = str(run_manifest)
    args.candidate_id = CANDIDATE_ID
    args.radio_checkpoint = str(checkpoint)
    args.canonical_capability_cache = fern_sources[
        "official_dino_sam3_views.pt"
    ]["path"]
    args.canonical_support_graph = fern_sources[
        "shared_support_graph_k16.pt"
    ]["path"]
    args.canonical_field_sha256 = fern_sources[
        "canonical_d256_l128_capability_first.pth"
    ]["sha256"]
    consumed, manifest_sha256 = _validate_candidate_run_manifest(
        args,
        scene_id="fern",
        benchmark_manifest_path=benchmark.resolve(),
    )
    assert consumed == payload
    assert manifest_sha256 == _bytes_record(run_manifest)["sha256"]
    authority = _load_registered_forward_protocol_authority(
        args,
        consumed,
        canonical_json_sha256(consumed["method_contract"]),
    )
    assert authority == payload["registered_forward_protocol_authority"]
    assert authority["strict_unseen_protocol_exact_match"] is False

    extra = snapshot / "undeclared-extra.txt"
    extra.write_text("not in the staged tree digest\n", encoding="utf-8")
    extra.chmod(0o444)
    with pytest.raises(StagingError, match="disk file set"):
        validate_snapshot(snapshot)
    extra.unlink()
    assert validate_snapshot(snapshot) == staging


def test_beta_runner_is_fixed_full_eight_and_old_v3_is_unchanged() -> None:
    runner = REPO_ROOT / RUNNER_RELATIVE
    subprocess.run(["bash", "-n", str(runner)], check=True)
    text = runner.read_text(encoding="utf-8")
    assert "GPU0_SCENES=(fern flower fortress horns_center)" in text
    assert "GPU1_SCENES=(horns_left leaves orchids trex)" in text
    assert '"0:fern" "1:horns_left"' in text
    assert '"0:flower" "1:leaves"' in text
    assert '"0:fortress" "1:orchids"' in text
    assert '"0:horns_center" "1:trex"' in text
    assert 'for scene_gpu in "${SERIAL_SCENE_GPU_PLAN[@]}"; do' in text
    assert 'run_gpu_scene "$physical_index" "$scene"' in text
    assert 'exec {gpu0_lock}>"$MAIN_OUTPUT_ROOT/.physical_gpu0.lock"' in text
    assert 'exec {gpu1_lock}>"$MAIN_OUTPUT_ROOT/.physical_gpu1.lock"' in text
    assert text.index("bind_reserved_gpu_identity 0") < text.index(
        'for scene_gpu in "${SERIAL_SCENE_GPU_PLAN[@]}"; do'
    )
    assert text.index("bind_reserved_gpu_identity 1") < text.index(
        'for scene_gpu in "${SERIAL_SCENE_GPU_PLAN[@]}"; do'
    )
    assert "run_gpu_worker" not in text
    assert "gpu0_worker_pid" not in text
    assert "gpu1_worker_pid" not in text
    assert 'run_gpu_scene "$physical_index" "$scene" &' not in text
    assert '"maximum_concurrent_scene_evaluators": 1' in text
    assert (
        '"host_memory_policy": "fixed_mapping_single_scene_resident_v1"'
        in text
    )
    assert "--candidate-id nvos-forward-beta-coverage-v1" in text
    assert "--registered-forward-unary beta_coverage_v1" in text
    assert "--registered-observation-fusion probability_mixture" in text
    assert "--registered-readout-stage propagated" in text
    assert "--expected-gpu-physical-index \"$physical_index\"" in text
    assert "nvos_forward_beta_scene_authority.py" in text
    assert "aggregate_nvos_forward_beta_full8_nonexact.py" in text
    assert "--receipt-root \"$SCENE_RECEIPT_ROOT\"" in text
    assert "GPU_MAX_TEMP_C=\"${GPU_MAX_TEMP_C:-86}\"" in text
    assert "GPU_POLL_SECONDS=\"${GPU_POLL_SECONDS:-20}\"" in text
    assert (
        "GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="
        '"${GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES:-3}"'
        in text
    )
    assert (
        '"maximum_consecutive_telemetry_failures": int('
        in text
    )
    assert (
        'GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="'
        '$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES"'
        in text
    )
    containment_end = text.index(
        'echo "OUTPUT_ROOT resolves outside the canonical output target"'
    )
    resolved_assignment = text.index('OUTPUT_ROOT="$OUTPUT_ROOT_REAL"')
    manifest_assignment = text.index(
        'RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"'
    )
    assert containment_end < resolved_assignment < manifest_assignment
    for forbidden in (
        "screen_nvos_registered_region_v3_continuation.py",
        "nvos_registered_region_v3_authority.py",
        "aggregate_registered_prompt_closeout.py",
        "THREE_SCENE_SCREEN",
        "three_scene_screen",
        "foreground_iou",
        "reject_stop_after_three",
        "wait \"$gpu0_worker_pid\"",
        "wait \"$gpu1_worker_pid\"",
    ):
        assert forbidden not in text

    old_runner = (
        REPO_ROOT / "radio_gs/scripts/run_nvos_registered_region_v3_queue.sh"
    ).read_text(encoding="utf-8")
    assert "--registered-forward-unary beta_coverage_v1" not in old_runner
    assert "--candidate-id nvos-forward-beta-coverage-v1" not in old_runner
