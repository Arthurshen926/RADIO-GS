from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import (
    aggregate_paired_seed_gate,
    row_identity_sha256,
    tensor_sha256,
)
from radio_gs.scripts import finalize_surface_region_query_free_promotion as surface_finalizer
from radio_gs.scripts import finalize_surface_text_response_promotion as promotion
from radio_gs.scripts import surface_text_response_distill_authority as distill_authority
from radio_gs.scripts.eval_text_response_fidelity_gate import load_descriptor_pair


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_surface_fixture_module() -> ModuleType:
    path = REPO_ROOT / "tests/test_finalize_surface_region_query_free_promotion.py"
    spec = importlib.util.spec_from_file_location("surface_finalizer_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SURFACE_FIXTURE = _load_surface_fixture_module()


def _sha256(path: Path) -> str:
    return promotion._sha256(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("artifact_type", "expected"),
    [
        (
            "surface_c1024_attention_pooling_postcache_continuation",
            "attention_postcache_screen",
        ),
        (surface_finalizer.ARTIFACT_TYPE, "query_free_promotion_bundle"),
    ],
)
def test_control_descriptor_authority_type_tracks_surface_plan(
    tmp_path: Path, artifact_type: str, expected: str
) -> None:
    manifest = tmp_path / "surface.json"
    _write_json(manifest, {"artifact_type": artifact_type})
    plan = {"surface_promotion": {"manifest": str(manifest)}}
    assert promotion._control_descriptor_authority_type(plan) == expected


def test_control_descriptor_authority_type_rejects_unknown_surface_plan(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "surface.json"
    _write_json(manifest, {"artifact_type": "forged_surface_authority"})
    with pytest.raises(ValueError, match="artifact type differs"):
        promotion._control_descriptor_authority_type(
            {"surface_promotion": {"manifest": str(manifest)}}
        )


def _freeze_test_snapshot(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    records = {relative: _sha256(root / relative) for relative in files}
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    return records


def test_authority_sources_are_verified_in_immutable_producer_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "producer-snapshot"
    records = _freeze_test_snapshot(root, {"producer/module.py": b"producer-v1\n"})

    resolved = promotion._validate_snapshot_source_hashes(
        records,
        required={"producer/module.py"},
        source_snapshot_root=str(root),
        label="test authority",
    )

    assert resolved == root.resolve()
    assert Path(promotion.__file__).resolve().parents[2] != resolved


def test_authority_source_snapshot_rejects_writable_or_hash_drift(
    tmp_path: Path,
) -> None:
    writable = tmp_path / "writable-snapshot"
    records = _freeze_test_snapshot(writable, {"producer.py": b"producer-v1\n"})
    writable.chmod(0o700)
    with pytest.raises(ValueError, match="writable entry"):
        promotion._validate_snapshot_source_hashes(
            records,
            required={"producer.py"},
            source_snapshot_root=str(writable),
            label="test authority",
        )

    frozen = tmp_path / "hash-drift-snapshot"
    records = _freeze_test_snapshot(frozen, {"producer.py": b"producer-v1\n"})
    with pytest.raises(ValueError, match="producer snapshot"):
        promotion._validate_snapshot_source_hashes(
            {"producer.py": "0" * 64},
            required={"producer.py"},
            source_snapshot_root=str(frozen),
            label="test authority",
        )


def test_authority_thermal_contract_requires_exact_owner_pid_namespace_mode(
    tmp_path: Path,
) -> None:
    guard = tmp_path / "guard.sh"
    guard.write_text("#!/bin/sh\n", encoding="utf-8")
    thermal = {
        "physical_gpu": 1,
        "maximum_temperature_c": 78,
        "maximum_start_temperature_c": 65,
        "maximum_power_limit_w": 300.5,
        "poll_seconds": 3,
        "soft_pause_temperature_c": 75,
        "soft_resume_temperature_c": 70,
        "peer_gpu": None,
        "peer_pause_temperature_c": 0,
        "peer_resume_temperature_c": 0,
        "peer_quiet_seconds_before_launch": 0,
        "peer_max_power_w": 0.0,
        "peer_max_memory_mib": 0,
        "peer_max_utilization_pct": 100,
        "peer_activity_action": "terminate",
        "owner_pid_namespace_mode": "exclusive-singleton-after-clear-v1",
        "guard": {"path": str(guard.resolve()), "sha256": _sha256(guard)},
    }
    assert promotion._validate_distill_thermal_contract(thermal, schema=2) == thermal

    for invalid in (None, "strict", "exclusive-singleton-after-clear-v2"):
        forged = dict(thermal)
        if invalid is None:
            forged.pop("owner_pid_namespace_mode")
        else:
            forged["owner_pid_namespace_mode"] = invalid
        with pytest.raises(ValueError, match="thermal-safety contract"):
            promotion._validate_distill_thermal_contract(forged, schema=2)


def test_schema3_per_seed_calibration_rejects_surface_control_swap(
    tmp_path: Path,
) -> None:
    fit_artifact = tmp_path / "fit.pt"
    fit_manifest = tmp_path / "fit.json"
    fit_artifact.write_bytes(b"fit")
    _write_json(fit_manifest, {"fit": True})
    fit_bank = {
        "artifact_path": str(fit_artifact.resolve()),
        "artifact_sha256": _sha256(fit_artifact),
        "manifest_path": str(fit_manifest.resolve()),
        "manifest_sha256": _sha256(fit_manifest),
        "split": "fit",
        "query_count": 2,
        **{
            field: str(index) * 64
            for index, field in enumerate(
                sorted(
                    promotion.FIT_TEXT_BANK_FIELDS
                    - {
                        "artifact_path", "artifact_sha256", "manifest_path",
                        "manifest_sha256", "split", "query_count",
                    }
                ),
                start=1,
            )
        },
    }
    control = {
        "path": str((tmp_path / "surface-seed1.pt").resolve()),
        "sha256": "a" * 64,
        "seed": 1,
        "architecture": {"digest": "b" * 64},
        "train_caches": [],
        "validation_caches": [],
        "source_best_epoch": 2,
        "source_best_selection_score": 0.8,
    }
    lambdas = {"independent_response": 16.0, "scene_response": 0.02}
    diagnostic_path = tmp_path / "diagnostic.json"
    _write_json(diagnostic_path, {"design": True})
    design_diagnostic = {
        "path": str(diagnostic_path.resolve()),
        "sha256": _sha256(diagnostic_path),
        "role": "seed0_design_prior_only_per_seed_values_remeasured",
        "measured_seed": 0,
        "calibration_reuses_measured_values": False,
        "diagnostic_surface_control": {
            "path": str((tmp_path / "surface-seed0.pt").resolve()),
            "sha256": "c" * 64,
        },
    }
    calibration = tmp_path / "calibration-seed1.json"
    payload = {
        "schema_version": 2,
        "artifact_type": "surface_text_response_gradient_calibration",
        "benchmark_vocabulary_opened": False,
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "seed": 1,
        "surface_control": control,
        "design_diagnostic": design_diagnostic,
        "objective_contract": {
            "gradient_bound_scope": (
                "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
            ),
            "training_batching": (
                "shuffle_complete_scene_groups_no_partial_scenes_v1"
            ),
            "max_complete_scene_batch_rows": 64,
        },
        "gradient_contract": {
            "measurement_point": "exact_seed_frozen_surface_control_state_dict",
            "branch_target_ratio": 0.25,
            "response_lambdas": lambdas,
            "combined_response_to_surface_upper_bound_ratio": 0.5,
            "trainable_parameter_count": 1,
            "trainable_parameters": [{"name": "weight", "shape": [2, 2]}],
            "loss_values": {
                key: 0.1
                for key in (
                    "surface", "token", "descriptor", "relation",
                    "independent_response", "scene_response", "scene_profile",
                    "scene_ranking",
                )
            },
        },
        "fit_text_bank": fit_bank,
    }
    _write_json(calibration, payload)
    promotion._validate_calibration_manifest(
        calibration,
        seed=1,
        response_lambdas=lambdas,
        surface_control=control,
        design_diagnostic=design_diagnostic,
        fit_text_bank=fit_bank,
        label="seed1",
    )
    swapped = {**control, "seed": 2, "path": str(tmp_path / "surface-seed2.pt")}
    with pytest.raises(ValueError, match="calibration contract differs"):
        promotion._validate_calibration_manifest(
            calibration,
            seed=1,
            response_lambdas=lambdas,
            surface_control=swapped,
            design_diagnostic=design_diagnostic,
            fit_text_bank=fit_bank,
            label="seed1",
        )


def test_schema3_shared_run_accepts_per_seed_calibrations_and_rejects_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def touch(name: str, payload: bytes = b"x") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    radio = touch("radio.pt")
    fit_artifact = touch("fit.pt")
    fit_manifest = touch("fit.json")
    diagnostic = touch("diagnostic.json")
    guard = touch("guard.sh")
    initial = tmp_path / "initial.json"
    gpu_identity = {
        "physical_index": 1,
        "uuid": "GPU-123456789",
        "pci_bus_id": "00000000:82:00.0",
    }
    _write_json(
        initial,
        {
            "status": "physical_gpu1_idle_and_pcie_responsive",
            "gpu_identity": gpu_identity,
            "compute_owners": [],
        },
    )
    train_caches = [promotion._file_record(touch(f"train{i}.pt")) for i in range(4)]
    validation_caches = [
        promotion._file_record(touch(f"validation{i}.pt")) for i in range(2)
    ]
    fit_bank = {
        "artifact_path": str(fit_artifact.resolve()),
        "artifact_sha256": _sha256(fit_artifact),
        "manifest_path": str(fit_manifest.resolve()),
        "manifest_sha256": _sha256(fit_manifest),
        "split": "fit",
        "query_count": 2,
        **{
            field: format(index, "x") * 64
            for index, field in enumerate(
                sorted(
                    promotion.FIT_TEXT_BANK_FIELDS
                    - {
                        "artifact_path", "artifact_sha256", "manifest_path",
                        "manifest_sha256", "split", "query_count",
                    }
                ),
                start=1,
            )
        },
    }
    controls = {
        seed: {
            "path": str(touch(f"surface{seed}.pt").resolve()),
            "sha256": _sha256(tmp_path / f"surface{seed}.pt"),
            "seed": seed,
            "architecture": {"digest": "d" * 64},
            "train_caches": train_caches,
            "validation_caches": validation_caches,
            "source_best_epoch": 1,
            "source_best_selection_score": 0.8,
        }
        for seed in promotion.REQUIRED_SEEDS
    }
    design = {
        "path": str(diagnostic.resolve()),
        "sha256": _sha256(diagnostic),
        "role": "seed0_design_prior_only_per_seed_values_remeasured",
        "measured_seed": 0,
        "calibration_reuses_measured_values": False,
        "diagnostic_surface_control": {
            "path": controls[0]["path"], "sha256": controls[0]["sha256"]
        },
    }
    calibrations = []
    responses = {}
    output_rows = []
    for seed in promotion.REQUIRED_SEEDS:
        calibration = touch(f"calibration{seed}.json")
        audit = touch(f"calibration{seed}.audit.json")
        lambdas = {
            "independent_response": 10.0 + seed,
            "scene_response": 0.01 + seed * 0.001,
        }
        calibration_row = {
            "seed": seed,
            "manifest": promotion._file_record(calibration),
            "audit": promotion._file_record(audit),
            "surface_control": controls[seed],
            "response_lambdas": lambdas,
        }
        calibrations.append(calibration_row)
        checkpoint = touch(f"response{seed}.pt")
        report = touch(f"response{seed}.pt.json")
        terminal = touch(f"terminal{seed}.json")
        output_row = {
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "report": str(report.resolve()),
            **{
                field: str(touch(f"{field}{seed}.dat").resolve())
                for field in (
                    "training_log", "audit_report", "guard_command",
                    "guard_telemetry", "guard_receipt", "kernel_journal",
                    "gpu_preflight", "gpu_postflight",
                )
            },
            "terminal": str(terminal.resolve()),
        }
        output_rows.append(output_row)
        responses[seed] = {
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": f"{seed + 1}" * 64,
            "sidecar": str(report.resolve()),
            "sidecar_sha256": f"{seed + 4}" * 64,
            "train_caches": train_caches,
            "validation_caches": validation_caches,
            "surface_control": controls[seed],
            "fit_text_bank": fit_bank,
            "calibration_manifest": str(calibration.resolve()),
            "calibration_manifest_sha256": _sha256(calibration),
            "response_lambdas": lambdas,
            "design_diagnostic": design,
            "distill_schema_version": 3,
            "payload": {
                "training_config": {
                    **{
                        field: value
                        for field, value in {
                            "hidden_dim": 256, "epochs": 60, "patience": 10,
                            "batch_size": 16, "learning_rate": 2e-4,
                            "weight_decay": 1e-4, "token_weight": 0.25,
                            "relation_weight": 0.1,
                            "reliability_attention_mode": "log_prior",
                            "context_pooling_mode": "joint_attention_v1",
                            "canonical_noise_degrees": 0.0,
                            "canonical_noise_calibration": "",
                        }.items()
                    },
                    "seed": seed,
                },
                "best_epoch": 1,
                "best_selection_score": 0.8,
            },
        }
    training_contract = {
        **{
            field: responses[0]["payload"]["training_config"][field]
            for field in promotion.COMMON_TRAINING_FIELDS
            if field != "seed"
        },
        "seeds": [0, 1, 2],
        "response_lambda_source": "per_seed_exact_surface_warmstart_gradient_budget",
        "response_branch_gradient_target_ratio": 0.25,
        "total_response_gradient_ratio_upper_bound": 0.5,
        "response_gradient_bound_scope": (
            "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
        ),
        "response_losses": [
            "independent_normalized_cosine_response_smooth_l1",
            "scene_wise_text_response_profile_ranking",
        ],
        "scene_profile_weight": 1.0,
        "scene_ranking_weight": 1.0,
        "scene_ranking_temperature": 0.1,
        "scene_tie_tolerance": 1e-6,
        "training_batching": "shuffle_complete_scene_groups_no_partial_scenes_v1",
        "max_complete_scene_batch_rows": 64,
        "epoch_selection": promotion.DISTILL_EPOCH_SELECTION,
        "surface_control_initialization": "exact_seed_checkpoint_state_dict",
        "surface_control_noninferiority_tolerance": 0.002,
    }
    surface_promotion = {"frozen": "surface"}
    run_manifest = tmp_path / "run_manifest.json"
    manifest = {
        "schema_version": 3,
        "artifact_type": "surface_region_text_response_distill_run",
        "authority_status": "query_free_three_seed_gpu1_run_frozen",
        "candidate": "context_c1024_geometric",
        "surface_promotion": surface_promotion,
        "train_caches": train_caches,
        "validation_caches": validation_caches,
        "fit_text_bank": {
            "artifact": promotion._file_record(fit_artifact),
            "manifest": promotion._file_record(fit_manifest),
        },
        "radio_checkpoint": promotion._file_record(radio),
        "calibrations": calibrations,
        "gradient_design_diagnostic": promotion._file_record(diagnostic),
        "initial_gpu_preflight": promotion._file_record(initial),
        "gpu_identity": gpu_identity,
        "outputs": output_rows,
        "training_contract": training_contract,
        "training_command_contract": {},
        "thermal_safety_contract": {
            "guard": promotion._file_record(guard),
            "peer_activity_action": "terminate",
            "owner_pid_namespace_mode": "exclusive-singleton-after-clear-v1",
        },
        "implementation_sources": {},
        "runtime_closure": {"digest": "e" * 64},
        "authority_contract": {
            "source_snapshot_root": str(tmp_path),
            "seed_resume": "skip_only_exact_guarded_terminal_v1",
            "closure_verification": "before_and_after_every_seed_v1",
            "global_gpu_lock": "/root/RADIO-GS/output/.physical_gpu1.lock",
            "main_output_root": "/root/RADIO-GS/output",
            "global_gpu_kernel_singleton_protocol": (
                "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
            ),
            "global_gpu_kernel_singleton_inherited_fd_verified": True,
        },
    }
    _write_json(run_manifest, manifest)
    for response in responses.values():
        response["distill_run_manifest"] = {
            "path": str(run_manifest.resolve()),
            "sha256": _sha256(run_manifest),
            "candidate": "context_c1024_geometric",
        }
    completion_seeds = []
    receipts = {}
    for seed, output_row in enumerate(output_rows):
        guard_receipt = promotion._file_record(Path(output_row["guard_receipt"]))
        receipts[output_row["guard_receipt"]] = {
            "payload": {"seed": seed, "gpu_identity": gpu_identity},
            "receipt": guard_receipt,
        }
        terminal_record = promotion._file_record(Path(output_row["terminal"]))
        completion_seeds.append(
            {
                "seed": seed,
                "checkpoint": {
                    "path": responses[seed]["checkpoint"],
                    "sha256": responses[seed]["checkpoint_sha256"],
                },
                "report": {
                    "path": responses[seed]["sidecar"],
                    "sha256": responses[seed]["sidecar_sha256"],
                },
                "guard_receipt": guard_receipt,
                "terminal": terminal_record,
                "best_epoch": 1,
                "best_selection_score": 0.8,
                "surface_control": controls[seed],
                "calibration": calibrations[seed],
            }
        )
        _write_json(
            Path(output_row["terminal"]),
            {
                "status": "complete_guarded_audited_no_xid_pcie_fault",
                "seed": seed,
                "candidate": "context_c1024_geometric",
                "runtime_closure_digest": "e" * 64,
                "evidence": {
                    "selection_contract": promotion.DISTILL_EPOCH_SELECTION,
                    "checkpoint": completion_seeds[-1]["checkpoint"],
                    "report": completion_seeds[-1]["report"],
                    "guard_receipt": guard_receipt,
                    "best_epoch": 1,
                    "best_selection_score": 0.8,
                    "surface_control": controls[seed],
                    "calibration": calibrations[seed],
                },
            },
        )
        completion_seeds[-1]["terminal"] = promotion._file_record(
            Path(output_row["terminal"])
        )
    completion = {
        "schema_version": 3,
        "artifact_type": "surface_region_text_response_distill_completion",
        "status": "complete_three_seed_guarded_authority",
        "candidate": "context_c1024_geometric",
        "run_manifest": promotion._file_record(run_manifest),
        "calibrations": calibrations,
        "gradient_design_diagnostic": promotion._file_record(diagnostic),
        "runtime_closure_digest": "e" * 64,
        "selection_contract": promotion.DISTILL_EPOCH_SELECTION,
        "seeds": completion_seeds,
    }
    _write_json(tmp_path / "text_response_distill.complete", completion)
    surface = {
        "selected_candidate": "context_c1024_geometric",
        "distill_surface_promotion": surface_promotion,
        "selected_caches": {
            "train": train_caches, "validation": validation_caches
        },
        "selected_by_seed": {
            seed: {
                "checkpoint": controls[seed]["path"],
                "checkpoint_sha256": controls[seed]["sha256"],
            }
            for seed in promotion.REQUIRED_SEEDS
        },
    }
    monkeypatch.setattr(
        promotion, "_validate_distill_thermal_contract", lambda value, schema: value
    )
    monkeypatch.setattr(
        promotion, "_validate_snapshot_source_hashes", lambda *args, **kwargs: tmp_path
    )
    monkeypatch.setattr(
        promotion, "_validate_distill_runtime_closure", lambda *args, **kwargs: "e" * 64
    )
    monkeypatch.setattr(promotion, "_validate_authority_seed_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(promotion, "validate_receipt", lambda path: receipts[str(path)])
    monkeypatch.setattr(
        distill_authority,
        "validate_training_command_contract",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        distill_authority,
        "validate_seed_terminal",
        lambda manifest_path, seed: {
            "terminal": promotion._file_record(Path(output_rows[seed]["terminal"]))
        },
    )
    result = promotion._validate_shared_distill_run(
        surface=surface, responses=responses, radio_path=radio
    )
    assert result["authority_schema_version"] == 3
    assert [row["seed"] for row in result["calibrations"]] == [0, 1, 2]

    manifest["calibrations"][1]["manifest"], manifest["calibrations"][2]["manifest"] = (
        manifest["calibrations"][2]["manifest"],
        manifest["calibrations"][1]["manifest"],
    )
    _write_json(run_manifest, manifest)
    for response in responses.values():
        response["distill_run_manifest"]["sha256"] = _sha256(run_manifest)
    with pytest.raises(ValueError, match="seed-1 calibration binding differs"):
        promotion._validate_shared_distill_run(
            surface=surface, responses=responses, radio_path=radio
        )


def _response_checkpoints(
    root: Path,
    surface_bundle: dict,
    *,
    promotion_manifest: Path,
    promotion_completion: Path,
    radio_path: Path,
    metric_delta: float = 0.001,
) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    fit_artifact = root / "fit.pt"
    fit_manifest = root / "fit.manifest.json"
    fit_artifact.write_bytes(b"fixed-fit-embedding-artifact")
    _write_json(fit_manifest, {"fixed_mock_fit_manifest": True})
    fit_bank = {
        "artifact_path": str(fit_artifact.resolve()),
        "artifact_sha256": _sha256(fit_artifact),
        "manifest_path": str(fit_manifest.resolve()),
        "manifest_sha256": _sha256(fit_manifest),
        "split": "fit",
        "query_count": 806,
        "split_synset_tab_query_lf_sha256": "1" * 64,
        "ordered_records_sha256": "2" * 64,
        "vocabulary_sha256": "3" * 64,
        "vocabulary_manifest_sha256": "4" * 64,
        "embedding_semantic_sha256": "5" * 64,
        "embedding_tensor_sha256": "6" * 64,
        "text_encoder_snapshot_files_sha256": "7" * 64,
    }
    calibration = root / "response_lambda_calibration.json"
    _write_json(
        calibration,
        {
            "schema_version": 1,
            "artifact_type": "surface_text_response_gradient_calibration",
            "benchmark_vocabulary_opened": False,
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "shared_training_seeds": [0, 1, 2],
            "gradient_contract": {
                "response_loss": "independent_normalized_cosine_response_smooth_l1",
                "response_lambda": 0.5,
            },
            "fit_text_bank": fit_bank,
        },
    )
    selected = str(surface_bundle["selected_candidate"])
    checkpoints = [
        root / f"response_seed{seed}.pt" for seed in promotion.REQUIRED_SEEDS
    ]
    reports = [path.with_suffix(".pt.json") for path in checkpoints]
    selected_caches = {}
    for role in ("train", "validation"):
        rows = sorted(
            (
                value
                for value in surface_bundle["bindings"]["caches"]
                if value["candidate"] == selected and value["role"] == role
            ),
            key=lambda value: int(value["shard"]),
        )
        selected_caches[role] = [
            {"path": str(Path(value["path"]).resolve()), "sha256": value["sha256"]}
            for value in rows
        ]
    first_control = torch.load(
        surface_bundle["selected_readouts"][0]["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    first_config = first_control["training_config"]
    training_contract = {
        field: first_config.get(field)
        for field in promotion.COMMON_TRAINING_FIELDS
        if field != "seed"
    }
    training_contract.update(
        {
            "seeds": [0, 1, 2],
            "response_lambda_source": "one_shot_initial_gradient_ratio",
            "response_loss": "independent_normalized_cosine_response_smooth_l1",
        }
    )
    repo = Path(promotion.__file__).resolve().parents[2]
    guard = repo / "radio_gs/scripts/run_with_gpu_thermal_guard.sh"
    run_manifest = root / "run_manifest.json"
    simple = surface_bundle["bindings"]
    run_payload = {
        "schema_version": 1,
        "artifact_type": "surface_region_text_response_distill_run",
        "candidate": selected,
        "surface_promotion": {
            "run_manifest": simple["run_manifest"],
            "cache_pairing": simple["cache_pairing"],
            "query_free_screen": simple["query_free_screen"],
            "screen_completion": simple["screen_completion"],
            "promotion_manifest": {
                "path": str(promotion_manifest.resolve()),
                "sha256": _sha256(promotion_manifest),
            },
            "promotion_completion": {
                "path": str(promotion_completion.resolve()),
                "sha256": _sha256(promotion_completion),
            },
        },
        "train_caches": selected_caches["train"],
        "validation_caches": selected_caches["validation"],
        "fit_text_bank": {
            key: fit_bank[key]
            for key in (
                "artifact_path",
                "artifact_sha256",
                "manifest_path",
                "manifest_sha256",
            )
        },
        "radio_checkpoint": {
            "path": str(radio_path.resolve()),
            "sha256": _sha256(radio_path),
        },
        "calibration_manifest": str(calibration.resolve()),
        "outputs": [
            {
                "seed": seed,
                "checkpoint": str(checkpoints[seed].resolve()),
                "report": str(reports[seed].resolve()),
            }
            for seed in promotion.REQUIRED_SEEDS
        ],
        "training_contract": training_contract,
        "thermal_safety_contract": {
            "physical_gpu": 1,
            "maximum_temperature_c": 68,
            "maximum_start_temperature_c": 52,
            "maximum_power_limit_w": 300.5,
            "poll_seconds": 2,
            "soft_pause_temperature_c": 64,
            "soft_resume_temperature_c": 62,
            "peer_gpu": 0,
            "peer_pause_temperature_c": 70,
            "peer_resume_temperature_c": 60,
            "peer_quiet_seconds_before_launch": 120,
            "peer_max_power_w": 80.0,
            "peer_max_memory_mib": 512,
            "peer_max_utilization_pct": 5,
            "guard": str(guard.resolve()),
            "guard_sha256": _sha256(guard),
        },
        "implementation_sources": {
            relative: _sha256(repo / relative)
            for relative in promotion.DISTILL_IMPLEMENTATION_SOURCES
        },
    }
    _write_json(run_manifest, run_payload)
    output = []
    for selected in surface_bundle["selected_readouts"]:
        seed = int(selected["seed"])
        control_path = Path(selected["checkpoint"])
        control = torch.load(control_path, map_location="cpu", weights_only=False)
        control_sidecar = json.loads(
            Path(selected["sidecar"]).read_text(encoding="utf-8")
        )
        checkpoint_path = checkpoints[seed]
        config = dict(control["training_config"])
        config.update(
            {
                "output": str(checkpoint_path),
                "train_caches": [Path(value["path"]) for value in selected_caches["train"]],
                "validation_caches": [
                    Path(value["path"]) for value in selected_caches["validation"]
                ],
                "fit_text_bank": fit_artifact,
                "fit_text_bank_manifest": fit_manifest,
                "calibration_manifest": calibration,
                "run_manifest": run_manifest,
                "radio_checkpoint": radio_path,
                "device": "cuda",
            }
        )
        validation = {
            key: float(value) + metric_delta
            for key, value in control_sidecar["validation"].items()
        }
        score = 0.5 * (
            validation["mean_descriptor_cosine"]
            + validation["all_view_descriptor_cosine"]
        )
        response_contract = {
            "fit_split_only": True,
            "benchmark_vocabulary_opened": False,
            "fit_text_bank": fit_bank,
            "calibration_manifest": str(calibration.resolve()),
            "calibration_manifest_sha256": _sha256(calibration),
            "response_lambda": 0.5,
            "shared_training_seeds": [0, 1, 2],
            "loss": "independent_normalized_cosine_response_smooth_l1",
        }
        provenance = {
            **control["provenance"],
            "train": {
                **control["provenance"]["train"],
                "cache_bindings": selected_caches["train"],
            },
            "validation": {
                **control["provenance"]["validation"],
                "cache_bindings": selected_caches["validation"],
            },
            "text_response_distillation": response_contract,
            "distill_run_manifest": {
                "path": str(run_manifest.resolve()),
                "sha256": _sha256(run_manifest),
                "candidate": str(surface_bundle["selected_candidate"]),
            },
        }
        history = [
            {
                "epoch": 1,
                "loss": 0.5,
                "token_loss": 0.2,
                "descriptor_loss": 0.2,
                "relation_loss": 0.1,
                "response_loss": 0.1,
                "response_lambda": 0.5,
                "selection_score": score,
                **validation,
            }
        ]
        payload = {
            **control,
            "provenance": provenance,
            "history": history,
            "best_epoch": 1,
            "best_selection_score": score,
            "training_config": config,
        }
        torch.save(payload, checkpoint_path)
        sidecar = {
            "output": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "architecture": payload["architecture"],
            "best_epoch": 1,
            "best_selection_score": score,
            "untrained_baseline": payload["untrained_baseline"],
            "selection_score_delta": score
            - float(payload["untrained_baseline_score"]),
            "validation": validation,
            "response_lambda": 0.5,
            "calibration_manifest": str(calibration.resolve()),
            "calibration_manifest_sha256": _sha256(calibration),
            "fit_text_bank_sha256": fit_bank["artifact_sha256"],
            "fit_query_count": 806,
            "distill_run_manifest": str(run_manifest.resolve()),
            "distill_run_manifest_sha256": _sha256(run_manifest),
            "validation_caches": selected_caches["validation"],
            "train_scenes": len(provenance["train"]["scenes"]),
            "validation_scenes": len(provenance["validation"]["scenes"]),
            "scene_overlap": [],
        }
        _write_json(checkpoint_path.with_suffix(".pt.json"), sidecar)
        output.append(checkpoint_path)
    completion = {
        "schema_version": 1,
        "artifact_type": "surface_region_text_response_distill_completion",
        "status": "complete",
        "candidate": str(surface_bundle["selected_candidate"]),
        "run_manifest": str(run_manifest.resolve()),
        "run_manifest_sha256": _sha256(run_manifest),
        "calibration_manifest": str(calibration.resolve()),
        "calibration_manifest_sha256": _sha256(calibration),
        "seeds": [
            {
                "seed": seed,
                "checkpoint": str(checkpoints[seed].resolve()),
                "checkpoint_sha256": _sha256(checkpoints[seed]),
                "report": str(reports[seed].resolve()),
                "report_sha256": _sha256(reports[seed]),
            }
            for seed in promotion.REQUIRED_SEEDS
        ],
    }
    _write_json(root / "text_response_distill.complete", completion)
    return output


def _write_descriptor(
    path: Path,
    *,
    plan: dict,
    role: str,
    method_id: str,
    seed: int,
    checkpoint: Path,
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_report = checkpoint.with_suffix(checkpoint.suffix + ".json")
    plan_rows = {int(row["seed"]): row for row in plan[role]}
    plan_row = plan_rows[seed]
    scenes: list[str] = []
    regions: list[str] = []
    validation_caches = []
    for value in plan["validation_caches"]:
        cache = torch.load(value["path"], map_location="cpu", weights_only=False)
        metadata = cache["metadata"]
        records = metadata["region_records"]
        scenes.extend(str(record["scene"]) for record in records)
        regions.extend(str(record["region_id"]) for record in records)
        validation_caches.append(
            {
                "path": value["path"],
                "sha256": value["sha256"],
                "rows": len(records),
                "split_file_sha256": metadata["split_file_sha256"],
                "region_contract_sha256": metadata["region_contract_sha256"],
                "radio_checkpoint_sha256": metadata["radio_checkpoint_sha256"],
                "teacher_target_protocol_sha256": metadata[
                    "teacher_target_protocol_sha256"
                ],
            }
        )
    assert len(scenes) == student.shape[0] == teacher.shape[0]
    if role == "control":
        authority = {
            "type": "query_free_promotion_bundle",
            "path": plan["surface_promotion"]["manifest"],
            "sha256": plan["surface_promotion"]["manifest_sha256"],
            "completion": plan["surface_promotion"]["completion"],
            "completion_sha256": plan["surface_promotion"]["completion_sha256"],
            "candidate": plan["selected_candidate"],
        }
    else:
        authority = {
            "type": "embedded_distill_run_manifest",
            **plan_row["distill_run_manifest"],
        }
    payload = {
        "schema_version": 1,
        "artifact_type": "surface_text_response_descriptor_pair",
        "method_id": method_id,
        "seed": seed,
        "split_role": "validation",
        "student_descriptors": student.float().contiguous(),
        "teacher_descriptors": teacher.float().contiguous(),
        "scene_ids": scenes,
        "region_ids": regions,
        "student_descriptors_sha256": tensor_sha256(student.float().contiguous()),
        "teacher_descriptors_sha256": tensor_sha256(teacher.float().contiguous()),
        "descriptor_rows_sha256": row_identity_sha256(scenes, regions),
        "descriptor_space": {
            "name": "official_siglip2_g_summary",
            "dimension": int(student.shape[1]),
            "normalization": "l2",
            "official_summary_head": "c-radio_v4 _heads.siglip2-g",
        },
        "provenance": {
            "uses_benchmark_scenes": False,
            "uses_benchmark_test_vocabulary": False,
            "annotations_opened": False,
            "labels_opened": False,
            "instances_opened": False,
            "masks_opened": False,
            "text_opened": False,
            "device": "cpu",
            "readout_checkpoint": str(checkpoint.resolve()),
            "readout_checkpoint_sha256": _sha256(checkpoint),
            "readout_report": str(checkpoint_report.resolve()),
            "readout_report_sha256": _sha256(checkpoint_report),
            "readout_binding_authority": authority,
            "radio_checkpoint": plan["radio_checkpoint"]["path"],
            "radio_checkpoint_sha256": plan["radio_checkpoint"]["sha256"],
            "region_contract_sha256": checkpoint_payload["provenance"][
                "region_contract_sha256"
            ],
            "validation_split_sha256": checkpoint_payload["provenance"][
                "validation"
            ]["split_hashes"][0],
            "validation_scenes": checkpoint_payload["provenance"]["validation"][
                "scenes"
            ],
            "teacher_region": checkpoint_payload["provenance"]["validation"][
                "teacher_region"
            ],
            "validation_caches": validation_caches,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _bank(root: Path, split: str, embeddings: torch.Tensor) -> dict:
    artifact = root / f"{split}_bank.pt"
    manifest = root / f"{split}_bank.manifest.json"
    artifact.write_bytes(f"{split}-bank".encode("utf-8"))
    manifest.write_text("{}\n", encoding="utf-8")
    return {
        "path": artifact.resolve(),
        "file_sha256": _sha256(artifact),
        "manifest_path": manifest.resolve(),
        "manifest_sha256": _sha256(manifest),
        "embeddings": F.normalize(embeddings.float(), dim=-1),
        "query_ids": [f"query-{index}" for index in range(len(embeddings))],
        "selected_records": [
            {
                "synset": f"n{index:08d}",
                "query": f"query {index}",
                "split": split,
            }
            for index in range(len(embeddings))
        ],
        "selected_records_sha256": ("1" if split == "dev" else "2") * 64,
        "ordered_records_sha256": ("3" if split == "dev" else "4") * 64,
        "query_split": split,
        "vocabulary_sha256": "5" * 64,
        "embedding_tensor_sha256": ("6" if split == "dev" else "7") * 64,
        "embedding_semantic_sha256": ("8" if split == "dev" else "9") * 64,
        "text_encoder": {"fixed_mock_snapshot": True},
    }


def _write_reports_and_gate(
    root: Path,
    *,
    split: str,
    control_descriptors: list[Path],
    candidate_descriptors: list[Path],
    bank: dict,
) -> tuple[list[Path], list[Path], Path]:
    control_reports, candidate_reports = [], []
    control_payloads, candidate_payloads = [], []
    for role, descriptors, paths, payloads in (
        ("control", control_descriptors, control_reports, control_payloads),
        ("candidate", candidate_descriptors, candidate_reports, candidate_payloads),
    ):
        for seed, descriptor_path in enumerate(descriptors):
            descriptor = load_descriptor_pair(descriptor_path)
            report = promotion._expected_report(
                descriptor,
                bank,
                query_split=split,
            )
            report_path = root / split / f"{role}_seed{seed}.json"
            _write_json(report_path, report)
            paths.append(report_path)
            payloads.append(report)
    gate = aggregate_paired_seed_gate(
        control_payloads,
        candidate_payloads,
        required_seeds=(0, 1, 2),
        minimum_improved_seeds=promotion.MINIMUM_IMPROVED_SEEDS,
        bootstrap_samples=promotion.BOOTSTRAP_SAMPLES,
        bootstrap_seed=promotion.BOOTSTRAP_SEED,
        quality_noninferiority_tolerance=promotion.QUALITY_NONINFERIORITY_TOLERANCE,
        phase=split,
        _test_expected_scene_ids=[
            row["scene_id"]
            for row in control_payloads[0]["metrics"]["scene_metrics"]
        ],
        _test_report_recomputer=lambda report, _phase: report,
    )
    gate_path = root / split / "gate.json"
    _write_json(gate_path, gate)
    return control_reports, candidate_reports, gate_path


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric_delta: float = 0.001,
) -> dict:
    SURFACE_FIXTURE.install_surface_finalizer_test_doubles(monkeypatch)
    surface_root = SURFACE_FIXTURE._fixture(tmp_path / "surface")
    surface_result = surface_finalizer.finalize(surface_root)
    surface_bundle = json.loads(
        Path(surface_result["promotion_manifest"]).read_text(encoding="utf-8")
    )
    response_checkpoints = _response_checkpoints(
        tmp_path / "response",
        surface_bundle,
        promotion_manifest=Path(surface_result["promotion_manifest"]),
        promotion_completion=Path(surface_result["completion"]),
        radio_path=Path(
            json.loads(
                Path(surface_bundle["bindings"]["run_manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )["radio_checkpoint"]
        ),
        metric_delta=metric_delta,
    )
    plan_path = tmp_path / "promotion_plan.json"
    run_manifest = json.loads(
        Path(surface_bundle["bindings"]["run_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    promotion.preflight(
        Path(surface_result["promotion_manifest"]),
        Path(surface_result["completion"]),
        response_checkpoints,
        Path(run_manifest["radio_checkpoint"]),
        plan_path,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    validation_scenes = sorted(
        {
            str(record["scene"])
            for value in plan["validation_caches"]
            for record in torch.load(
                value["path"], map_location="cpu", weights_only=False
            )["metadata"]["region_records"]
        }
    )

    def aggregate_fixture_gate(*args: object, **kwargs: object) -> dict:
        kwargs.setdefault("_test_expected_scene_ids", validation_scenes)
        kwargs.setdefault("_test_report_recomputer", lambda report, _phase: report)
        return aggregate_paired_seed_gate(*args, **kwargs)

    monkeypatch.setattr(
        promotion, "aggregate_paired_seed_gate", aggregate_fixture_gate
    )

    validation_rows = sum(
        len(
            torch.load(
                value["path"], map_location="cpu", weights_only=False
            )["metadata"]["region_records"]
        )
        for value in plan["validation_caches"]
    )
    torch.manual_seed(31)
    teacher = F.normalize(torch.randn(validation_rows, 4), dim=-1)
    control_student = F.normalize(
        teacher + 0.45 * torch.randn(validation_rows, 4), dim=-1
    )
    candidate_student = teacher.clone()
    control_descriptors, candidate_descriptors = [], []
    for seed in (0, 1, 2):
        control_path = tmp_path / "descriptors" / f"control_seed{seed}.pt"
        candidate_path = tmp_path / "descriptors" / f"candidate_seed{seed}.pt"
        _write_descriptor(
            control_path,
            plan=plan,
            role="control",
            method_id=plan["control_method_id"],
            seed=seed,
            checkpoint=Path(plan["control"][seed]["checkpoint"]),
            student=control_student,
            teacher=teacher,
        )
        _write_descriptor(
            candidate_path,
            plan=plan,
            role="candidate",
            method_id=plan["candidate_method_id"],
            seed=seed,
            checkpoint=Path(plan["candidate"][seed]["checkpoint"]),
            student=candidate_student,
            teacher=teacher,
        )
        control_descriptors.append(control_path)
        candidate_descriptors.append(candidate_path)

    dev_bank = _bank(tmp_path, "dev", torch.randn(4, 4))
    audit_bank = _bank(tmp_path, "audit", torch.randn(4, 4))
    banks = {"dev": dev_bank, "audit": audit_bank}
    monkeypatch.setattr(
        promotion,
        "load_text_embedding_bank",
        lambda path, manifest, split: banks[split],
    )
    dev_control_reports, dev_candidate_reports, dev_gate = _write_reports_and_gate(
        tmp_path,
        split="dev",
        control_descriptors=control_descriptors,
        candidate_descriptors=candidate_descriptors,
        bank=dev_bank,
    )
    audit_control_reports, audit_candidate_reports, audit_gate = _write_reports_and_gate(
        tmp_path,
        split="audit",
        control_descriptors=control_descriptors,
        candidate_descriptors=candidate_descriptors,
        bank=audit_bank,
    )
    return {
        "plan": plan_path,
        "plan_payload": plan,
        "control_descriptors": control_descriptors,
        "candidate_descriptors": candidate_descriptors,
        "dev_bank": dev_bank,
        "audit_bank": audit_bank,
        "dev_control_reports": dev_control_reports,
        "dev_candidate_reports": dev_candidate_reports,
        "dev_gate": dev_gate,
        "audit_control_reports": audit_control_reports,
        "audit_candidate_reports": audit_candidate_reports,
        "audit_gate": audit_gate,
    }


def _finalize_dev(tmp_path: Path, fixture: dict) -> dict:
    output = tmp_path / "dev_decision.json"
    completion = tmp_path / "dev_decision.complete.json"
    result = promotion.finalize_stage(
        stage="dev",
        plan_path=fixture["plan"],
        control_descriptors=fixture["control_descriptors"],
        candidate_descriptors=fixture["candidate_descriptors"],
        control_reports=fixture["dev_control_reports"],
        candidate_reports=fixture["dev_candidate_reports"],
        gate_path=fixture["dev_gate"],
        text_bank_path=fixture["dev_bank"]["path"],
        text_bank_manifest_path=fixture["dev_bank"]["manifest_path"],
        output=output,
        completion=completion,
    )
    return {"result": result, "output": output, "completion": completion}


def test_dev_then_audit_cpu_mock_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    dev = _finalize_dev(tmp_path, fixture)
    dev_payload = json.loads(dev["output"].read_text(encoding="utf-8"))

    assert dev["result"]["decision"] == "promote_audit_required"
    assert dev["result"]["main_result_eligible"] is False
    assert dev_payload["surface_retention"]["passes"] is True
    assert all(
        row["passes"] for row in dev_payload["surface_retention"]["per_seed"]
    )
    assert dev_payload["benchmark_vocabulary_opened"] is False

    audit_output = tmp_path / "audit_confirmation.json"
    audit_completion = tmp_path / "audit_confirmation.complete.json"
    audit = promotion.finalize_stage(
        stage="audit",
        plan_path=fixture["plan"],
        control_descriptors=fixture["control_descriptors"],
        candidate_descriptors=fixture["candidate_descriptors"],
        control_reports=fixture["audit_control_reports"],
        candidate_reports=fixture["audit_candidate_reports"],
        gate_path=fixture["audit_gate"],
        text_bank_path=fixture["audit_bank"]["path"],
        text_bank_manifest_path=fixture["audit_bank"]["manifest_path"],
        output=audit_output,
        completion=audit_completion,
        dev_manifest=dev["output"],
        dev_completion=dev["completion"],
    )
    audit_payload = json.loads(audit_output.read_text(encoding="utf-8"))

    assert audit["decision"] == "promote_confirmed"
    assert audit["main_result_eligible"] is True
    assert audit_payload["stage_role"] == "single_confirmation_only_no_retuning"
    assert audit_payload["benchmark_vocabulary_opened"] is False
    assert audit_payload["dev_dependency"]["manifest_sha256"] == _sha256(
        dev["output"]
    )
    assert json.loads(audit_completion.read_text(encoding="utf-8"))[
        "stage_manifest_sha256"
    ] == _sha256(audit_output)


def test_surface_drop_rejects_dev_and_blocks_audit_before_bank_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, metric_delta=-0.0021)
    dev = _finalize_dev(tmp_path, fixture)
    payload = json.loads(dev["output"].read_text(encoding="utf-8"))

    assert dev["result"]["decision"] == "reject_no_audit"
    assert dev["result"]["main_result_eligible"] is False
    assert payload["text_gate"]["decision"] == "promote"
    assert payload["surface_retention"]["passes"] is False

    audit_bank_loads = 0
    original_loader = promotion.load_text_embedding_bank
    audit_bank_path = Path(fixture["audit_bank"]["path"]).resolve()

    def forbid_audit_bank(path: Path, *args: object, **kwargs: object) -> dict:
        nonlocal audit_bank_loads
        if Path(path).resolve() == audit_bank_path:
            audit_bank_loads += 1
            raise AssertionError("audit bank must stay closed")
        return original_loader(path, *args, **kwargs)

    monkeypatch.setattr(promotion, "load_text_embedding_bank", forbid_audit_bank)
    with pytest.raises(ValueError, match="audit is forbidden"):
        promotion.finalize_stage(
            stage="audit",
            plan_path=fixture["plan"],
            control_descriptors=fixture["control_descriptors"],
            candidate_descriptors=fixture["candidate_descriptors"],
            control_reports=fixture["audit_control_reports"],
            candidate_reports=fixture["audit_candidate_reports"],
            gate_path=fixture["audit_gate"],
            text_bank_path=fixture["audit_bank"]["path"],
            text_bank_manifest_path=fixture["audit_bank"]["manifest_path"],
            output=tmp_path / "forbidden-audit.json",
            completion=tmp_path / "forbidden-audit.complete.json",
            dev_manifest=dev["output"],
            dev_completion=dev["completion"],
        )
    assert audit_bank_loads == 0


def test_coordinated_dev_manifest_completion_tamper_cannot_open_audit_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, metric_delta=-0.0021)
    dev = _finalize_dev(tmp_path, fixture)
    manifest = json.loads(dev["output"].read_text(encoding="utf-8"))
    assert manifest["decision"] == "reject_no_audit"
    assert manifest["surface_retention"]["passes"] is False

    # Forge the two self-authenticating JSON files together. The frozen bound
    # descriptors/reports still prove that the real dev decision was reject.
    manifest["decision"] = "promote_audit_required"
    manifest["surface_retention"]["passes"] = True
    _write_json(dev["output"], manifest)
    completion = json.loads(dev["completion"].read_text(encoding="utf-8"))
    completion["decision"] = "promote_audit_required"
    completion["stage_manifest_sha256"] = _sha256(dev["output"])
    _write_json(dev["completion"], completion)

    audit_bank_loads = 0
    original_loader = promotion.load_text_embedding_bank
    audit_bank_path = Path(fixture["audit_bank"]["path"]).resolve()

    def track_audit_bank(path: Path, *args: object, **kwargs: object) -> dict:
        nonlocal audit_bank_loads
        if Path(path).resolve() == audit_bank_path:
            audit_bank_loads += 1
        return original_loader(path, *args, **kwargs)

    monkeypatch.setattr(promotion, "load_text_embedding_bank", track_audit_bank)
    with pytest.raises(ValueError, match="strict recomputation"):
        promotion.finalize_stage(
            stage="audit",
            plan_path=fixture["plan"],
            control_descriptors=fixture["control_descriptors"],
            candidate_descriptors=fixture["candidate_descriptors"],
            control_reports=fixture["audit_control_reports"],
            candidate_reports=fixture["audit_candidate_reports"],
            gate_path=fixture["audit_gate"],
            text_bank_path=fixture["audit_bank"]["path"],
            text_bank_manifest_path=fixture["audit_bank"]["manifest_path"],
            output=tmp_path / "forged-audit.json",
            completion=tmp_path / "forged-audit.complete.json",
            dev_manifest=dev["output"],
            dev_completion=dev["completion"],
        )
    assert audit_bank_loads == 0


def test_tampered_report_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    report_path = fixture["dev_candidate_reports"][0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metrics"]["aggregate"]["smooth_l1"] += 0.01
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="aggregate smooth_l1"):
        _finalize_dev(tmp_path, fixture)


def test_plan_rejects_response_checkpoint_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    checkpoint = Path(fixture["plan_payload"]["candidate"][0]["checkpoint"])
    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="checkpoint/sidecar selection drift"):
        promotion.validate_plan(fixture["plan"])


def test_plan_rejects_distill_completion_seed_sha_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    completion = Path(
        fixture["plan_payload"]["distill_run"]["completion"]["path"]
    )
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["seeds"][0]["checkpoint_sha256"] = "0" * 64
    _write_json(completion, payload)

    with pytest.raises(ValueError, match="distill completion"):
        promotion.validate_plan(fixture["plan"])


def test_response_rejects_coordinated_surface_authority_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SURFACE_FIXTURE.install_surface_finalizer_test_doubles(monkeypatch)
    surface_root = SURFACE_FIXTURE._fixture(tmp_path / "surface-authority")
    result = surface_finalizer.finalize(surface_root)
    manifest_path = Path(result["promotion_manifest"])
    completion_path = Path(result["completion"])
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    selected = original_manifest["selected_candidate"]

    forged_metrics = json.loads(json.dumps(original_manifest))
    for row in forged_metrics["selected_readouts"]:
        if row["seed"] == 0:
            row["validation"]["summary_token_cosine"] = -0.75
    for row in forged_metrics["bindings"]["all_compared_readouts"]:
        if row["candidate"] == selected and row["seed"] == 0:
            row["validation"]["summary_token_cosine"] = -0.75
    _write_json(manifest_path, forged_metrics)
    completion = dict(original_completion)
    completion["promotion_manifest_sha256"] = _sha256(manifest_path)
    _write_json(completion_path, completion)
    with pytest.raises(ValueError, match="authoritative validation exceeds"):
        promotion._validate_surface_bundle(manifest_path, completion_path)

    forged_aggregate = json.loads(json.dumps(original_manifest))
    forged_aggregate["query_free_selection"]["selected_candidate_metrics"][
        "mean_selection_score"
    ] = -0.75
    _write_json(manifest_path, forged_aggregate)
    completion["promotion_manifest_sha256"] = _sha256(manifest_path)
    _write_json(completion_path, completion)
    with pytest.raises(ValueError, match="selected candidate differs"):
        promotion._validate_surface_bundle(manifest_path, completion_path)


def test_runner_is_cpu_only_and_dev_precedes_audit() -> None:
    runner = REPO_ROOT / "radio_gs/scripts/run_surface_text_response_promotion.sh"
    source = runner.read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert source.index("finalize-dev") < source.index("AUDIT_TEXT_BANK=")
    assert source.index('if [[ "$DEV_DECISION" == "reject_no_audit" ]]') < source.index(
        "AUDIT_TEXT_BANK="
    )
    assert "--required-seeds 0,1,2" in source
    assert "--bootstrap-samples 2000" in source
    assert source.count("--phase dev") == 1
    assert source.count("--phase audit") == 1
    assert source.count('"$TEXT_GATE" evaluate-many') == 2
    assert "DEV_EVALUATE_ARGS" in source
    assert "AUDIT_EVALUATE_ARGS" in source
    assert '--readout-binding-manifest "$PROMOTION_MANIFEST"' in source
    assert source.count("--readout-binding-manifest") == 1
    assert "benchmark" not in source.lower()
