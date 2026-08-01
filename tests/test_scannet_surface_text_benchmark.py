from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
import torch

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts import build_surface_region_semantic_cache as builder
from radio_gs.scripts import eval_scannet_canonical_text_query as evaluator
from radio_gs.scripts import finalize_gpu_guard_receipt as guard_receipt
from radio_gs.scripts import finalize_scannet_surface_text_benchmark as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_IDENTITY = {
    "physical_index": 1,
    "uuid": "GPU-00000000-test-fixed-gpu1",
    "pci_bus_id": "00000000:82:00.0",
}


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path.resolve()


def _torch(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return path.resolve()


def _record(path: Path) -> dict[str, str]:
    return benchmark._file_record(path)


def _xyz_sha256(xyz: torch.Tensor) -> str:
    value = xyz.float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _write_label_ply(path: Path, labels: list[int]) -> Path:
    rows = "\n".join(
        f"{float(index)} 0.0 0.0 {label}" for index, label in enumerate(labels)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(labels)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int label\nend_header\n"
        f"{rows}\n",
        encoding="utf-8",
    )
    return path.resolve()


def _readout(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    contract = SurfaceRegionContractV2()
    model = SurfaceRegionSummaryReadoutV2(feature_dim=1280, hidden_dim=8)
    provenance = {
        "training_scope": "global_cross_scene_fixture",
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "scene_disjoint": True,
        "region_contract": contract.to_dict(),
        "region_contract_sha256": contract.digest,
    }
    return _torch(
        path,
        {
            "schema_version": 3,
            "architecture": model.architecture(contract.digest),
            "state_dict": model.state_dict(),
            "provenance": provenance,
        },
    )


def _authority_receipt(tmp_path: Path) -> tuple[Path, dict]:
    surface = _json(tmp_path / "surface.json", {"accepted": True})
    surface_completion = _json(tmp_path / "surface.complete.json", {"ok": True})
    audit = _json(tmp_path / "audit.json", {"decision": "promote_confirmed"})
    audit_completion = _json(tmp_path / "audit.complete.json", {"ok": True})
    plan = _json(tmp_path / "plan.json", {"selected": "candidate"})
    readouts = []
    for seed in benchmark.REQUIRED_SEEDS:
        checkpoint = _readout(tmp_path / f"readout_seed{seed}.pt", seed)
        sidecar = _json(tmp_path / f"readout_seed{seed}.json", {"seed": seed})
        readouts.append(
            {
                "seed": seed,
                "checkpoint": _record(checkpoint),
                "sidecar": _record(sidecar),
            }
        )
    authority = {
        "selected_candidate": "response_candidate",
        "method_id": "surface-response-fixture",
        "surface_manifest": _record(surface),
        "surface_completion": _record(surface_completion),
        "text_audit_manifest": _record(audit),
        "text_audit_completion": _record(audit_completion),
        "text_plan": _record(plan),
        "readouts": readouts,
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": benchmark.AUTHORITY_RECEIPT_TYPE,
        "status": "accepted_before_benchmark_open",
        "required_seeds": list(benchmark.REQUIRED_SEEDS),
        "benchmark_data_opened": False,
        "forbidden_benchmark_modules_loaded": [],
        "promotion_authority": authority,
        "authority_inputs": {
            "surface_manifest": authority["surface_manifest"],
            "surface_completion": authority["surface_completion"],
            "text_audit_manifest": authority["text_audit_manifest"],
            "text_audit_completion": authority["text_audit_completion"],
        },
        "implementation_sources": [
            {"relative_path": relative, **_record(REPO_ROOT / relative)}
            for relative in benchmark.AUTHORITY_IMPLEMENTATION_SOURCES
        ],
    }
    receipt_path = _json(tmp_path / "authority.receipt.json", receipt)
    return receipt_path, authority


def _registry(tmp_path: Path) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    mpr = _torch(tmp_path / "mpr.pt", {"xyz": xyz})
    radio = _torch(tmp_path / "radio.pt", {"state_dict": {}})
    base = _torch(tmp_path / "text_base.pt", {"base": True})
    text_payloads: dict[str, dict[str, torch.Tensor]] = {}
    split_records = {}
    authority_embeddings: dict[str, torch.Tensor] = {}
    for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"]:
        vector = torch.zeros(1536, dtype=torch.float32)
        vector[0] = 1.0
        authority_embeddings[NYU40_ID_TO_NAME[class_id]] = vector
    for split in benchmark.REQUIRED_SPLITS:
        queries = [
            NYU40_ID_TO_NAME[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ]
        embeddings = torch.stack([authority_embeddings[query] for query in queries])
        payload = {
            "queries": queries,
            "prompt_templates": benchmark.PROTOCOL["prompt_templates"],
            "embeddings": embeddings,
            "text_encoder": "siglip2",
            "model_name": "google/siglip2-giant-opt-patch16-384",
        }
        text_payloads[split] = payload
        split_records[split] = _record(_torch(tmp_path / f"text_split{split}.pt", payload))
    scenes = {}
    scene_payloads: dict[str, dict[str, torch.Tensor]] = {}
    for scene in benchmark.REQUIRED_SCENES:
        scene_root = tmp_path / scene
        field = _torch(
            scene_root / "field.pt",
            {
                "architecture": {"num_gaussians": 2},
                "geometry_fingerprint": {
                    "num_gaussians": 2,
                    "xyz_sha256": _xyz_sha256(xyz),
                },
                "mpr_cache": str(mpr),
                "benchmark_masks_opened": False,
                "text_queries_opened": False,
            },
        )
        field_record = _record(field)
        graph = _torch(
            scene_root / "graph.pt",
            {
                "schema_version": 1,
                "num_global_rows": 2,
                "global_rows": torch.tensor([0, 1], dtype=torch.int64),
                "xyz": xyz.clone(),
                "metadata": {
                    "source": "canonical_official_dino_sam3_multichannel_support_graph",
                    "capability_metadata": {
                        "field_checkpoint": str(field),
                        "field_checkpoint_sha256": field_record["sha256"],
                        "mpr_cache": str(mpr),
                        "radio_checkpoint_sha256": _record(radio)["sha256"],
                    },
                },
            },
        )
        label = _write_label_ply(scene_root / "labels.ply", [1, 2, 5])
        scenes[scene] = {
            "field": field_record,
            "mpr": _record(mpr),
            "graph": _record(graph),
            "label": _record(label),
        }
        scene_payloads[scene] = {"xyz": xyz}
    return (
        {
            "prepared_root": str(benchmark.PREPARED_ROOT),
            "radio_checkpoint": _record(radio),
            "text_cache_base": _record(base),
            "text_split_caches": split_records,
            "text_encoder_provenance": {
                "encoder": "siglip2",
                "model_name": "google/siglip2-giant-opt-patch16-384",
                "encoder_authority_split": "19",
                "subset_embedding_max_abs_tolerance": 2e-7,
            },
            "scenes": scenes,
        },
        text_payloads,
    )


def _thermal_contract() -> dict:
    return {
        "guarded": True,
        "guard": _record(REPO_ROOT / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"),
        "physical_gpu": 1,
        "maximum_temperature_c": 75,
        "maximum_start_temperature_c": 52,
        "maximum_power_limit_w": 300.5,
        "poll_seconds": 1,
        "soft_pause_temperature_c": 0,
        "soft_resume_temperature_c": 0,
        "peer_gpu": 0,
        "peer_pause_temperature_c": 77,
        "peer_resume_temperature_c": 75,
        "peer_quiet_seconds_before_launch": 0,
        "peer_maximum_power_w": 0.0,
        "peer_maximum_memory_mib": 0,
        "peer_maximum_utilization_percent": 100,
        "peer_activity_gate": "temperature_only_activity_limits_disabled",
        "semantic_builder_resume": "durable_per_batch_resume_with_strict_contract",
        "hard_guard_is_safety_authority": True,
        "soft_pause_safety_role": "supplementary_only_not_a_safety_precondition",
    }


def _manifest(tmp_path: Path) -> tuple[Path, dict, dict[str, dict[str, torch.Tensor]]]:
    receipt, _ = _authority_receipt(tmp_path)
    registry, text_payloads = _registry(tmp_path)
    output_root = (tmp_path / "benchmark").resolve()
    with mock.patch.object(benchmark, "_materialize_input_registry", return_value=registry):
        payload = benchmark.build_run_manifest(
            authority_receipt=receipt,
            authority_receipt_sha256=benchmark._sha256(receipt),
            output_root=output_root,
            semantic_radio_batch_size=4,
            semantic_batch_size=2,
            semantic_pacing_seconds=4.0,
            thermal_contract=_thermal_contract(),
            gpu_identity=GPU_IDENTITY,
        )
    path = output_root / "run_manifest.json"
    benchmark._write_frozen_json(path, payload)
    return path, payload, text_payloads


def _semantic_payload(manifest: dict, seed: int, scene: str) -> dict:
    scene_inputs = manifest["input_registry"]["scenes"][scene]
    readout = next(
        row for row in manifest["promotion_authority"]["readouts"] if row["seed"] == seed
    )
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    descriptor = torch.zeros(2, 1536, dtype=torch.float16)
    descriptor[:, 0] = 1.0
    scales = descriptor[:, None, :].repeat(1, 3, 1)
    contract = SurfaceRegionContractV2()
    return {
        "xyz": xyz,
        "features": descriptor,
        "summary_features": descriptor.clone(),
        "global_rows": torch.tensor([0, 1], dtype=torch.int64),
        "features_by_scale": scales,
        "valid": torch.tensor([True, True]),
        "metadata": {
            "schema_version": 5,
            "feature_space": "official_siglip2_summary_descriptor_multiscale",
            "source": "canonical_radio_surface_region_readout",
            "construction": "canonical_radio_surface_region_readout_then_official_summary_head",
            "canonical_radio_source": "field_decode_only",
            "mpr_radio_features_opened": False,
            "readout_checkpoint": readout["checkpoint"]["path"],
            "readout_checkpoint_sha256": readout["checkpoint"]["sha256"],
            "bridge_checkpoint_sha256": readout["checkpoint"]["sha256"],
            "bridge_training_scope": "global_cross_scene",
            "bridge_training_scope_detail": "global_cross_scene_fixture",
            "field_checkpoint": scene_inputs["field"]["path"],
            "field_checkpoint_sha256": scene_inputs["field"]["sha256"],
            "mpr_cache": scene_inputs["mpr"]["path"],
            "mpr_cache_sha256": scene_inputs["mpr"]["sha256"],
            "field_geometry_xyz_sha256": _xyz_sha256(xyz),
            "support_graph": scene_inputs["graph"]["path"],
            "support_graph_sha256": scene_inputs["graph"]["sha256"],
            "radio_checkpoint_sha256": manifest["input_registry"]["radio_checkpoint"]["sha256"],
            "official_radio_checkpoint_sha256": manifest["input_registry"]["radio_checkpoint"]["sha256"],
            "readout_batch_size": 2,
            "region_radii_m": list(contract.radii_m),
            "region_topology": contract.expansion,
            "region_contract": contract.to_dict(),
            "region_contract_version": contract.version,
            "region_contract_sha256": contract.digest,
            "query_set_invariant": True,
            "official_summary_head": True,
            "custom_text_projection": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "cache_role": "disposable_derivative_not_scene_memory",
            "row_storage": "sparse_valid_rows_with_global_row_index",
            "scale_storage": "all_scales_preserved; mean_descriptor_legacy_only",
        },
    }


def _semantic_command(manifest: dict, seed: int, scene: str) -> list[str]:
    stage = benchmark._stage_record(manifest, seed, scene)
    inputs = manifest["input_registry"]["scenes"][scene]
    readout = benchmark._readout_record(manifest, seed)
    radio = manifest["input_registry"]["radio_checkpoint"]
    return [
        "bash",
        str(REPO_ROOT / "radio_gs/scripts/run_repo_python.sh"),
        str(REPO_ROOT / "radio_gs/scripts/build_surface_region_semantic_cache.py"),
        "--field-checkpoint", inputs["field"]["path"],
        "--field-checkpoint-sha256", inputs["field"]["sha256"],
        "--support-graph", inputs["graph"]["path"],
        "--support-graph-sha256", inputs["graph"]["sha256"],
        "--readout-checkpoint", readout["checkpoint"]["path"],
        "--readout-checkpoint-sha256", readout["checkpoint"]["sha256"],
        "--mpr-cache-sha256", inputs["mpr"]["sha256"],
        "--output", stage["semantic_cache"],
        "--query-output", str(Path(stage["semantic_cache"]).with_name("semantic_query.pt")),
        "--resume-dir", stage["semantic_resume_dir"],
        "--radio-batch-size", "4",
        "--semantic-batch-size", "2",
        "--thermal-pacing-seconds-per-batch", "4.0",
        "--radio-checkpoint", radio["path"],
        "--radio-checkpoint-sha256", radio["sha256"],
        "--device", "cuda:0",
    ]


def _finalize_semantic(
    tmp_path: Path,
    manifest_path: Path,
    manifest: dict,
    *,
    seed: int = 0,
    scene: str = benchmark.REQUIRED_SCENES[0],
    telemetry_temperature: float = 51.0,
    telemetry_power_limit: float = 300.0,
) -> dict:
    stage = benchmark._stage_record(manifest, seed, scene)
    semantic = Path(stage["semantic_cache"])
    _torch(semantic, _semantic_payload(manifest, seed, scene))
    command_path = tmp_path / "guard.command.json"
    guard_receipt.prepare_command(
        output=command_path,
        run_manifest=manifest_path,
        seed=seed,
        scene=scene,
        gpu_identity=GPU_IDENTITY,
        command=_semantic_command(manifest, seed, scene),
    )
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "timestamp,gpu,bus_id,temp_c,power_w,power_limit_w,util_pct,memory_mib,pstate,event\n"
        "2026-08-01T00:00:00+00:00,1,0000:82:00.0,"
        f"{telemetry_temperature},210.0,{telemetry_power_limit},95,9000,P2,sample\n",
        encoding="utf-8",
    )
    receipt_path = Path(stage["semantic_guard_receipt"])
    guard_receipt.finalize_receipt(
        output=receipt_path,
        command_record=command_path,
        telemetry=telemetry,
        guard=REPO_ROOT / "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        stage_output=semantic,
        exit_status=0,
    )
    return benchmark.finalize_semantic_stage(
        run_manifest=manifest_path,
        seed=seed,
        scene=scene,
        semantic_cache=semantic,
        guard_receipt=receipt_path,
        terminal=Path(stage["semantic_terminal"]),
    )


def _report(
    manifest: dict,
    text_payloads: dict[str, dict],
    *,
    seed: int = 0,
    scene: str = benchmark.REQUIRED_SCENES[0],
) -> dict:
    stage = benchmark._stage_record(manifest, seed, scene)
    registry = manifest["input_registry"]
    text = {
        split: torch.as_tensor(text_payloads[split]["embeddings"])
        for split in benchmark.REQUIRED_SPLITS
    }
    result = evaluator.evaluate(
        scene=scene,
        label_ply=registry["scenes"][scene]["label"]["path"],
        semantic_cache=stage["semantic_cache"],
        split_text_embeddings=text,
        split_names=list(benchmark.REQUIRED_SPLITS),
        projection_k=8,
        distance_epsilon=1e-4,
        chunk_size=2048,
        device=torch.device("cpu"),
        scale_aggregation="max",
        scale_specificity_margin=0.0,
        tree_workers=1,
        torch_threads=1,
    )
    result["protocol"]["prompt_templates"] = benchmark.PROTOCOL["prompt_templates"]
    result["protocol"]["class_aliases"] = "none"
    result["protocol"]["text_embedding_cache_base"] = registry["text_cache_base"]["path"]
    return result


def test_authority_gate_import_process_never_loads_scannet_modules() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from radio_gs.scripts import "
                "finalize_scannet_surface_text_authority_gate as g;"
                "assert g._forbidden_loaded_modules()==[]"
            ),
        ],
        cwd=REPO_ROOT,
        env={**dict(__import__("os").environ), "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0


def test_run_manifest_freezes_receipt_grid_tree_runtime_and_resume(tmp_path: Path) -> None:
    path, manifest, _ = _manifest(tmp_path)
    reopened = benchmark._validate_run_manifest(path, verify_registry=False)
    assert reopened == manifest
    assert len(manifest["stage_grid"]) == 9
    assert {
        (row["seed"], row["scene"]) for row in manifest["stage_grid"]
    } == {
        (seed, scene)
        for seed in benchmark.REQUIRED_SEEDS
        for scene in benchmark.REQUIRED_SCENES
    }
    tree = manifest["radio_gs_python_tree"]
    actual = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "radio_gs").rglob("*.py")
        if path.is_file()
    )
    assert [row["relative_path"] for row in tree["ordered_entries"]] == actual
    assert tree["file_count"] == len(actual)
    assert manifest["runtime_fingerprint"]["fingerprint_sha256"]
    assert manifest["semantic_execution"]["builder_internal_resume"] is True
    assert manifest["semantic_execution"]["builder_internal_pacing"] is True


def test_invalid_authority_receipt_fails_before_input_registry_is_opened(
    tmp_path: Path,
) -> None:
    receipt, _ = _authority_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["status"] = "rejected"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with mock.patch.object(benchmark, "_materialize_input_registry") as materialize:
        with pytest.raises(ValueError, match="authority receipt"):
            benchmark.build_run_manifest(
                authority_receipt=receipt,
                authority_receipt_sha256=benchmark._sha256(receipt),
                output_root=tmp_path / "must_not_open",
                semantic_radio_batch_size=4,
                semantic_batch_size=2,
                semantic_pacing_seconds=4.0,
                thermal_contract=_thermal_contract(),
                gpu_identity=GPU_IDENTITY,
            )
    materialize.assert_not_called()


def test_semantic_finalizer_closes_field_mpr_graph_readout_scales_and_guard(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, _ = _manifest(tmp_path)
    terminal = _finalize_semantic(tmp_path, manifest_path, manifest)
    assert terminal["semantic"]["mpr"] == manifest["input_registry"]["scenes"][
        benchmark.REQUIRED_SCENES[0]
    ]["mpr"]
    assert terminal["execution_safety"]["builder_internal_resume"] is True
    assert terminal["guard_receipt"]["sha256"]


@pytest.mark.parametrize(
    "temperature,power_limit",
    [(75.0, 300.0), (51.0, 301.0)],
)
def test_semantic_finalizer_rejects_guard_telemetry_outside_safety_envelope(
    tmp_path: Path,
    temperature: float,
    power_limit: float,
) -> None:
    manifest_path, manifest, _ = _manifest(tmp_path)
    with pytest.raises(ValueError, match="frozen safety envelope"):
        _finalize_semantic(
            tmp_path,
            manifest_path,
            manifest,
            telemetry_temperature=temperature,
            telemetry_power_limit=power_limit,
        )


def test_guard_receipt_rejects_failed_or_malformed_telemetry(tmp_path: Path) -> None:
    telemetry = tmp_path / "failed.csv"
    header = (
        "timestamp,gpu,bus_id,temp_c,power_w,power_limit_w,util_pct,"
        "memory_mib,pstate,event\n"
    )
    telemetry.write_text(
        header
        + "2026-08-01T00:00:00+00:00,1,0000:82:00.0,51,210,300,95,"
        "9000,P2,thermal_abort\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed guard event"):
        guard_receipt.summarize_telemetry(
            telemetry,
            gpu_identity=GPU_IDENTITY,
        )
    telemetry.write_text(
        header
        + "2026-08-01T00:00:00+00:00,1,0000:82:00.0,51,210,300,101,"
        "9000,P2,sample\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="utilization/memory is out of bounds"):
        guard_receipt.summarize_telemetry(
            telemetry,
            gpu_identity=GPU_IDENTITY,
        )


@pytest.mark.parametrize(
    "forgery,match",
    [
        ("aggregate", "derived from retained scales"),
        ("scale_norm", "unit-normalized"),
        ("graph_rows", "tensor geometry differs"),
        ("mpr_sha", "provenance differs"),
    ],
)
def test_semantic_closure_rejects_forged_components(
    tmp_path: Path,
    forgery: str,
    match: str,
) -> None:
    manifest_path, manifest, _ = _manifest(tmp_path)
    seed, scene = 0, benchmark.REQUIRED_SCENES[0]
    stage = benchmark._stage_record(manifest, seed, scene)
    payload = _semantic_payload(manifest, seed, scene)
    if forgery == "aggregate":
        payload["features"][0, 0] = 0.5
        payload["summary_features"] = payload["features"].clone()
    elif forgery == "scale_norm":
        payload["features_by_scale"][0, 0, 0] = 0.5
        payload["features"] = torch.nn.functional.normalize(
            payload["features_by_scale"].float().mean(1), dim=-1
        ).half()
        payload["summary_features"] = payload["features"].clone()
    elif forgery == "graph_rows":
        payload["global_rows"] = torch.tensor([1, 0])
    else:
        payload["metadata"]["mpr_cache_sha256"] = "0" * 64
    _torch(Path(stage["semantic_cache"]), payload)
    with pytest.raises(ValueError, match=match):
        benchmark._validate_semantic_cache(
            manifest,
            seed=seed,
            scene=scene,
            semantic_cache=Path(stage["semantic_cache"]),
        )


def test_evaluation_finalizer_replays_predictions_exactly(tmp_path: Path) -> None:
    manifest_path, manifest, text_payloads = _manifest(tmp_path)
    seed, scene = 0, benchmark.REQUIRED_SCENES[0]
    _finalize_semantic(tmp_path, manifest_path, manifest, seed=seed, scene=scene)
    stage = benchmark._stage_record(manifest, seed, scene)
    report = _report(manifest, text_payloads, seed=seed, scene=scene)
    report_path = _json(Path(stage["evaluation_report"]), report)
    terminal = benchmark.finalize_evaluation_stage(
        run_manifest=manifest_path,
        seed=seed,
        scene=scene,
        report=report_path,
        terminal=Path(stage["evaluation_terminal"]),
    )
    assert set(terminal["validated_metrics"]) == set(benchmark.REQUIRED_SPLITS)


@pytest.mark.parametrize("field", ["gt_count", "intersection", "pred_count"])
def test_evaluation_replay_rejects_forged_histogram_intersection_and_prediction(
    tmp_path: Path,
    field: str,
) -> None:
    manifest_path, manifest, text_payloads = _manifest(tmp_path)
    seed, scene = 0, benchmark.REQUIRED_SCENES[0]
    _finalize_semantic(tmp_path, manifest_path, manifest, seed=seed, scene=scene)
    report = _report(manifest, text_payloads, seed=seed, scene=scene)
    row = report["splits"]["19"]["per_class"]["1"]
    row[field] += 1
    if field != "intersection":
        row["union"] += 1
    stage = benchmark._stage_record(manifest, seed, scene)
    forged = _json(Path(stage["evaluation_report"]), report)
    with pytest.raises(ValueError, match="full deterministic prediction replay"):
        benchmark.finalize_evaluation_stage(
            run_manifest=manifest_path,
            seed=seed,
            scene=scene,
            report=forged,
            terminal=Path(stage["evaluation_terminal"]),
        )


def test_resume_batches_are_durable_reusable_and_fail_closed(tmp_path: Path) -> None:
    resume = tmp_path / "resume"
    contract = {"schema_version": 1, "artifact_type": "fixture", "input": "fixed"}
    digest = builder._load_or_create_resume_contract(resume, contract)
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    builder._commit_resume_tensor(
        resume,
        phase="radio",
        start=0,
        stop=1,
        contract_sha256=digest,
        value=tensor,
    )
    restored = builder._load_resume_tensor(
        resume,
        phase="radio",
        start=0,
        stop=1,
        contract_sha256=digest,
        expected_shape=(1, 2),
        expected_dtype=torch.float16,
    )
    assert torch.equal(restored, tensor)
    _, terminal = builder._resume_paths(resume, phase="radio", start=0, stop=1)
    terminal.write_text("{}", encoding="utf-8")
    shard, _ = builder._resume_paths(resume, phase="radio", start=0, stop=1)
    with pytest.raises(ValueError, match="quarantine path:.*not deleted automatically"):
        builder._load_resume_tensor(
            resume,
            phase="radio",
            start=0,
            stop=1,
            contract_sha256=digest,
            expected_shape=(1, 2),
            expected_dtype=torch.float16,
        )
    assert shard.exists()


def test_runner_locks_gpu1_receipts_and_quarantines_without_ack_bypass() -> None:
    runner = REPO_ROOT / "radio_gs/scripts/run_scannet_surface_text_benchmark.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    source = runner.read_text(encoding="utf-8")
    assert 'PHYSICAL_GPU_LOCK="$REPO_ROOT/output/.physical_gpu1.lock"' in source
    assert source.index('open_lock_file "$RUNNER_LOCK_PATH"') < source.index(
        'CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$AUTHORITY_GATE"'
    )
    assert "GPU_UUID" in source and "GPU_BUS_ID" in source
    assert 'CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert "reject_existing_gpu1_compute_owner" in source
    assert "finalize_gpu_guard_receipt.py" in source
    assert "--resume-dir" in source
    assert "--thermal-pacing-seconds-per-batch" in source
    assert "quarantine path:" in source and "nothing was deleted" in source
    assert "ACKNOWLEDGE_UNRESUMABLE" not in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-0}"' in source
    assert "mv " not in source and "rm " not in source
    assert "SCENES=(scene0062_00 scene0140_00 scene0200_00)" in source
    assert "SEEDS=(0 1 2)" in source


def test_formal_runtime_chain_has_no_unrestricted_torch_load() -> None:
    for relative in (
        "radio_gs/scripts/build_surface_region_semantic_cache.py",
        "radio_gs/scripts/eval_scannet_canonical_text_query.py",
        "radio_gs/scripts/finalize_scannet_surface_text_benchmark.py",
        "radio_gs/scripts/finalize_gpu_guard_receipt.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "torch.load(" not in source
