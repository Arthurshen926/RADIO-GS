from __future__ import annotations

import argparse
import errno
import os
import json
from pathlib import Path
import sys

import pytest
import torch

from radio_gs.scripts import finalize_surface_text_response_promotion as promotion
from radio_gs.scripts import surface_gpu1_lock_supervisor as gpu1_lock
from radio_gs.scripts import surface_text_response_distill_authority as authority
from radio_gs.scripts import train_surface_region_text_response_distill as trainer


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "radio_gs/scripts/run_surface_region_text_response_distill.sh"


def _history() -> list[dict[str, object]]:
    def row(
        epoch: int,
        *,
        surface: float,
        fit_quality: float,
        response_smooth_l1: float,
        response_mae: float,
    ) -> dict[str, object]:
        scene_metrics = {
            scene: {
                **{
                    field: fit_quality
                    for field in authority.FIT_RESPONSE_SCENE_QUALITY_METRICS
                },
                "smooth_l1": response_smooth_l1 + offset,
                "mae": response_mae + offset,
            }
            for scene, offset in (("scene-a", 0.0), ("scene-b", 0.001))
        }
        result: dict[str, object] = {
            "epoch": epoch,
            "surface_selection_score": surface,
            "summary_token_cosine": surface,
            "mean_descriptor_cosine": surface,
            "all_view_descriptor_cosine": surface,
            "text_support_top1_agreement": 0.5 + 0.1 * epoch,
            "text_response_smooth_l1": response_smooth_l1,
            "text_response_mae": response_mae,
            "descriptor_relation_smooth_l1": 0.001 + 0.001 * epoch,
            "scene_response_objective": dict(
                authority.SCENE_RESPONSE_OBJECTIVE
            ),
            "text_response_scene_metrics": scene_metrics,
            "text_response_scene_worst_smooth_l1": max(
                metrics["smooth_l1"] for metrics in scene_metrics.values()
            ),
            "text_response_scene_worst_mae": max(
                metrics["mae"] for metrics in scene_metrics.values()
            ),
            **{
                field: fit_quality
                for field in authority.FIT_RESPONSE_QUALITY_METRICS
            },
            **{
                f"text_response_scene_worst_{field}": fit_quality
                for field in authority.FIT_RESPONSE_SCENE_QUALITY_METRICS
            },
        }
        if epoch:
            result.update(
                {
                    "independent_response_loss": 0.03,
                    "scene_response_loss": 0.02,
                    "scene_profile_loss": 0.01,
                    "scene_ranking_loss": 0.018,
                }
            )
        return result

    rows = [
        row(
            0,
            surface=0.96,
            fit_quality=0.90,
            response_smooth_l1=0.02,
            response_mae=0.04,
        ),
        row(
            1,
            surface=0.959,
            fit_quality=0.899,
            response_smooth_l1=0.01,
            response_mae=0.02,
        ),
        row(
            2,
            surface=0.97,
            fit_quality=0.91,
            response_smooth_l1=0.02,
            response_mae=0.03,
        ),
    ]
    return trainer.finalize_response_primary_epoch_selection(rows)[0]


def _v4_checkpoint() -> tuple[dict[str, object], dict[str, object]]:
    rows = json.loads(json.dumps(_history()))[:2]
    rows[0].update(
        {
            "initialization": "frozen_surface_control_checkpoint",
            "state_machine_role": "frozen_control_initial_anchor",
        }
    )
    rows[1]["state_machine_role"] = "fixed_micro_ray_trial"
    template = json.loads(json.dumps(_history()[2]))
    for epoch in range(2, 12):
        row = json.loads(json.dumps(template))
        row["epoch"] = epoch
        row["state_machine_role"] = "fixed_micro_ray_trial"
        if epoch % 2 == 0:
            row.update(
                {
                    "surface_selection_score": 0.95,
                    "summary_token_cosine": 0.95,
                    "mean_descriptor_cosine": 0.95,
                    "all_view_descriptor_cosine": 0.95,
                }
            )
        rows.append(row)
    history, best_epoch, best_score = trainer.finalize_response_primary_epoch_selection(
        rows
    )
    assert (best_epoch, best_score) == (1, 0.959)

    state_dict = {"weight": torch.tensor([1.0], dtype=torch.float32)}
    best_hash = authority._state_dict_sha256(state_dict)
    anchor_epoch = 0
    anchor_hash = "0" * 64
    accepted_count = 0
    rejected_count = 0
    previous_hash: str | None = None
    stale = 0
    for index, row in enumerate(history):
        if index == 0:
            row.update(
                {
                    "accepted": True,
                    "rejected": False,
                    "anchor_epoch_after_proposal": 0,
                    "anchor_state_dict_sha256_after_proposal": anchor_hash,
                    "best_updated": True,
                    "best_epoch_after_proposal": 0,
                    "best_state_dict_sha256_after_proposal": anchor_hash,
                    "patience_stale_after_proposal": 0,
                    "patience_stop_after_proposal": False,
                }
            )
        else:
            trial_hash = best_hash if index == 1 else f"{1000 + index:064x}"
            proposal_losses = {
                field: 0.01 + offset * 0.001
                for offset, field in enumerate(authority.PROPOSAL_LOSS_FIELDS)
            }
            row.update(
                {
                    "proposal": {
                        "index": index,
                        "source_anchor_epoch": anchor_epoch,
                        "anchor_state_dict_sha256": anchor_hash,
                        "raw_state_dict_sha256": f"{2000 + index:064x}",
                        "trial_state_dict_sha256": trial_hash,
                        "alpha_numerator": 1,
                        "alpha_denominator": 2048,
                        "optimizer_state_reset": True,
                        "validation_evaluations": 1,
                        "backtracking": "none_fixed_alpha_single_trial",
                        "persistent_generator": "advanced_never_rolled_back",
                    },
                    "loss_measurement_state": (
                        authority.PROPOSAL_LOSS_MEASUREMENT_STATE
                    ),
                    "proposal_losses": proposal_losses,
                    **{
                        flat_field: proposal_losses[field]
                        for field, flat_field in (
                            authority.LEGACY_FLAT_PROPOSAL_LOSS_FIELDS.items()
                        )
                    },
                }
            )
            accepted = row["response_selection_feasible"] is True
            if accepted:
                anchor_epoch = index
                anchor_hash = trial_hash
                accepted_count += 1
            else:
                rejected_count += 1
            best_updated = index == 1
            stale = 0 if best_updated else stale + 1
            row.update(
                {
                    "accepted": accepted,
                    "rejected": not accepted,
                    "anchor_epoch_after_proposal": anchor_epoch,
                    "anchor_state_dict_sha256_after_proposal": anchor_hash,
                    "best_updated": best_updated,
                    "best_epoch_after_proposal": 1,
                    "best_state_dict_sha256_after_proposal": best_hash,
                    "patience_stale_after_proposal": stale,
                    "patience_stop_after_proposal": stale >= 10,
                }
            )
        history[index] = trainer.attach_history_hash_chain(row, previous_hash)
        previous_hash = history[index]["history_hash_chain"]["sha256"]

    checkpoint: dict[str, object] = {
        "state_dict": state_dict,
        "history": history,
        "best_epoch": 1,
        "best_selection_score": best_score,
        "best_state_dict_sha256": best_hash,
        "proposal_state_machine": trainer._proposal_state_machine_contract(),
        "accepted_anchor": {
            "epoch": anchor_epoch,
            "state_dict_sha256": anchor_hash,
            "accepted_proposal_count": accepted_count,
            "rejected_proposal_count": rejected_count,
        },
        "history_hash_chain_sha256": previous_hash,
        "training_config": {
            "epochs": 60,
            "patience": 10,
            "proposal_state_machine": trainer._proposal_state_machine_contract(),
        },
        "provenance": {
            "proposal_state_machine": trainer._proposal_state_machine_contract()
        },
    }
    report = {
        field: checkpoint[field]
        for field in (
            "best_epoch",
            "best_state_dict_sha256",
            "proposal_state_machine",
            "accepted_anchor",
            "history_hash_chain_sha256",
        )
    }
    return checkpoint, report


def test_runner_freezes_gpu1_authority_defaults_and_per_seed_receipts() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'LOCK_ROOT="/root/RADIO-GS/output"' in source
    assert 'GLOBAL_GPU1_LOCK="$LOCK_ROOT/.physical_gpu1.lock"' in source
    assert "TEXT_RESPONSE_DISTILL_LOCK_HELD" not in source
    assert "TEXT_RESPONSE_DISTILL_GLOBAL_LOCK_FD" in source
    assert "TEXT_RESPONSE_DISTILL_RUN_LOCK_FD" in source
    assert "RADIO_GS_GPU1_SINGLETON_FD" in source
    assert "RADIO_GS_GPU1_SINGLETON_PROTOCOL" in source
    assert (
        authority.LOCK_ROOT_BINDING_ENV
        == "TEXT_RESPONSE_DISTILL_LOCK_ROOT_BINDING_SHA256"
    )
    assert "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1" in source
    assert "--singleton-fd" in source
    assert "verify-lock-fds" in source
    assert "git -C" not in source
    assert 'GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-80}"' in source
    assert 'GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-70}"' in source
    assert 'GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-5}"' in source
    assert 'GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-76}"' in source
    assert 'GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-72}"' in source
    assert 'GPU_PEER_INDEX=""' in source
    assert "GPU_PEER_PAUSE_TEMP_C=0" in source
    assert "GPU_PEER_RESUME_TEMP_C=0" in source
    assert "GPU_PEER_MAX_POWER_W=0" in source
    assert "GPU_PEER_MAX_MEMORY_MIB=0" in source
    assert "GPU_PEER_MAX_UTIL_PCT=100" in source
    assert (
        'GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"'
        in source
    )
    assert (
        '--gpu-owner-pid-namespace-mode "$GPU_OWNER_PID_NAMESPACE_MODE"'
        in source
    )
    assert (
        'GPU_OWNER_PID_NAMESPACE_MODE="$GPU_OWNER_PID_NAMESPACE_MODE"'
        in source
    )
    assert 'GPU_TELEMETRY_LOG="$telemetry"' in source
    assert 'f"seed{seed}.csv"' not in source
    assert "seed${seed}.csv" in source
    assert "prepare-command" in source and "finalize" in source
    assert "bind_surface_control" in source
    assert "--surface-control-checkpoint" in source
    assert "--surface-control-checkpoint-sha256" in source
    assert "journalctl -k" in source
    assert 'authority verify-manifest "${MANIFEST_ARGUMENTS[@]}"' in source
    assert "finalize-seed" in source and "verify-seed" in source
    assert "accepted_anchor_" in (
        REPO_ROOT / "radio_gs/scripts/train_surface_region_text_response_distill.py"
    ).read_text(encoding="utf-8")
    assert authority.EPOCH_SELECTION == trainer.RESPONSE_EPOCH_SELECTION
    assert authority.TRAINING_CONTRACT["response_losses"] == [
        authority.INDEPENDENT_RESPONSE_LOSS,
        trainer.SCENE_RESPONSE_LOSS,
    ]
    assert authority.TRAINING_CONTRACT["scene_response_objective"] == (
        trainer._scene_response_objective_contract()
    )
    assert authority.PROPOSAL_STATE_MACHINE == trainer._proposal_state_machine_contract()
    assert authority.TRAINING_CONTRACT["proposal_state_machine"] == (
        trainer._proposal_state_machine_contract()
    )
    assert "scene_profile_weight" not in authority.TRAINING_CONTRACT
    assert "scene_ranking_weight" not in authority.TRAINING_CONTRACT
    assert "scene_ranking_temperature" not in authority.TRAINING_CONTRACT


def _manifest_cli_arguments(
    tmp_path: Path,
    *,
    command: str,
    owner_pid_namespace_mode: str,
) -> list[str]:
    return [
        command,
        "--repo-root",
        str(tmp_path),
        "--lock-root",
        str(tmp_path / "lock-root"),
        "--candidate",
        "context_c1024_geometric",
        "--surface-root",
        str(tmp_path / "surface"),
        "--output-root",
        str(tmp_path / "output"),
        "--train-caches",
        str(tmp_path / "train*.pt"),
        "--validation-caches",
        str(tmp_path / "validation*.pt"),
        "--fit-text-bank",
        str(tmp_path / "fit.pt"),
        "--fit-text-bank-manifest",
        str(tmp_path / "fit.json"),
        "--radio-checkpoint",
        str(tmp_path / "radio.pt"),
        "--calibration-manifest",
        f"0={tmp_path / 'calibration0.json'}",
        "--calibration-audit",
        f"0={tmp_path / 'calibration0.audit.json'}",
        "--calibration-manifest",
        f"1={tmp_path / 'calibration1.json'}",
        "--calibration-audit",
        f"1={tmp_path / 'calibration1.audit.json'}",
        "--calibration-manifest",
        f"2={tmp_path / 'calibration2.json'}",
        "--calibration-audit",
        f"2={tmp_path / 'calibration2.audit.json'}",
        "--gradient-diagnostic",
        str(tmp_path / "gradient-diagnostic.json"),
        "--gradient-diagnostic-sha256",
        "a" * 64,
        "--initial-gpu-preflight",
        str(tmp_path / "gpu.initial.json"),
        "--thermal-guard",
        str(tmp_path / "guard.sh"),
        "--run-manifest",
        str(tmp_path / "run_manifest.json"),
        "--gpu-max-temp-c",
        "80",
        "--gpu-start-max-temp-c",
        "70",
        "--gpu-max-power-limit-w",
        "300.5",
        "--gpu-poll-seconds",
        "5",
        "--gpu-soft-pause-temp-c",
        "76",
        "--gpu-soft-resume-temp-c",
        "72",
        "--gpu-peer-pause-temp-c",
        "0",
        "--gpu-peer-resume-temp-c",
        "0",
        "--gpu-peer-quiet-seconds",
        "0",
        "--gpu-peer-max-power-w",
        "0",
        "--gpu-peer-max-memory-mib",
        "0",
        "--gpu-peer-max-util-pct",
        "100",
        "--gpu-peer-activity-action",
        "terminate",
        "--gpu-owner-pid-namespace-mode",
        owner_pid_namespace_mode,
    ]


def test_manifest_cli_and_thermal_contract_freeze_owner_pid_namespace_mode(
    tmp_path: Path,
) -> None:
    guard = tmp_path / "guard.sh"
    guard.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    parser = authority.build_parser()
    expected = "exclusive-singleton-after-clear-v1"
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    for command in ("create-manifest", "verify-manifest"):
        mode_action = next(
            action
            for action in subparsers.choices[command]._actions
            if action.dest == "gpu_owner_pid_namespace_mode"
        )
        assert mode_action.required is True
        args = parser.parse_args(
            _manifest_cli_arguments(
                tmp_path,
                command=command,
                owner_pid_namespace_mode=expected,
            )
        )
        assert isinstance(args, argparse.Namespace)
        assert args.gpu_owner_pid_namespace_mode == expected
        thermal = authority._thermal_contract(args, args.thermal_guard)
        assert thermal["owner_pid_namespace_mode"] == expected

    for rejected in ("strict", "exclusive-singleton-after-clear-v2"):
        rejected_args = parser.parse_args(
            _manifest_cli_arguments(
                tmp_path,
                command="create-manifest",
                owner_pid_namespace_mode=rejected,
            )
        )
        with pytest.raises(ValueError, match="owner PID namespace contract"):
            authority._thermal_contract(
                rejected_args,
                rejected_args.thermal_guard,
            )


def test_authority_output_index_never_reuses_telemetry_between_seeds(
    tmp_path: Path,
) -> None:
    rows = authority._seed_outputs(tmp_path, "context_c1024_geometric")
    telemetry = [row["guard_telemetry"] for row in rows]
    receipts = [row["guard_receipt"] for row in rows]
    terminals = [row["terminal"] for row in rows]
    assert len(set(telemetry)) == len(set(receipts)) == len(set(terminals)) == 3
    assert [Path(path).name for path in telemetry] == [
        "seed0.csv",
        "seed1.csv",
        "seed2.csv",
    ]


def test_gpu_observation_rejects_owner_bus_and_dead_pci() -> None:
    valid = dict(
        phase="pre_seed0",
        gpu_uuid="GPU-123456789",
        nvidia_bus_id="00000000:82:00.0",
        proc_bus_id="0000:82:00.0",
        pci_prefix="de100022000000000000000000000000",
        compute_owners=[],
        observed_epoch=1_754_000_000,
    )
    payload = authority.gpu_check_payload(**valid)
    assert payload["gpu_identity"]["physical_index"] == 1
    with pytest.raises(ValueError, match="compute owners"):
        authority.gpu_check_payload(**{**valid, "compute_owners": ["1234"]})
    with pytest.raises(ValueError, match="bus identity"):
        authority.gpu_check_payload(**{**valid, "proc_bus_id": "0000:83:00.0"})
    with pytest.raises(ValueError, match="not responding"):
        authority.gpu_check_payload(**{**valid, "pci_prefix": "f" * 32})


def _minimal_training_manifest(tmp_path: Path) -> tuple[dict[str, object], Path]:
    repo = tmp_path / "snapshot"
    repo.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(4):
        (inputs / f"train_{index}.pt").write_bytes(f"train-{index}".encode())
    for index in range(2):
        (inputs / f"validation_{index}.pt").write_bytes(f"validation-{index}".encode())
    fit = inputs / "fit.pt"
    fit_manifest = inputs / "fit.json"
    radio = inputs / "radio.pt"
    for path in (fit, fit_manifest, radio):
        path.write_bytes(path.name.encode())
    calibrations = {}
    calibration_rows = []
    for seed in range(3):
        calibration = inputs / f"calibration{seed}.json"
        audit = inputs / f"calibration{seed}.audit.json"
        calibration.write_bytes(calibration.name.encode())
        audit.write_bytes(audit.name.encode())
        calibrations[str(seed)] = str(calibration)
        calibration_rows.append(
            {
                "seed": seed,
                "manifest": authority.file_record(calibration),
                "audit": authority.file_record(audit),
            }
        )
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text("{}\n", encoding="utf-8")
    arguments = {
        "train_caches": str(inputs / "train_*.pt"),
        "validation_caches": str(inputs / "validation_*.pt"),
        "fit_text_bank": str(fit),
        "fit_text_bank_manifest": str(fit_manifest),
        "calibration_manifests": calibrations,
        "run_manifest": str(run_manifest),
        "radio_checkpoint": str(radio),
        "output_root": str(tmp_path / "output"),
    }
    outputs = authority._seed_outputs(tmp_path / "output", "context_c1024_geometric")
    surface_controls = []
    for seed in range(3):
        control = inputs / f"surface-control-seed{seed}.pt"
        control.write_bytes(f"surface-control-{seed}".encode())
        surface_controls.append(
            {
                "seed": seed,
                "checkpoint": authority.file_record(control),
                "best_epoch": 1,
                "best_selection_score": 0.9,
                "validation": {
                    "summary_token_cosine": 0.8,
                    "mean_descriptor_cosine": 0.9,
                    "all_view_descriptor_cosine": 0.9,
                },
            }
        )
    surface_promotion = {
        "binding_mode": authority.ATTENTION_BINDING_MODE,
        "selected_variant": authority.CONTEXT_POOLING_MODE,
        "selected_readouts": surface_controls,
    }
    manifest: dict[str, object] = {
        "candidate": "context_c1024_geometric",
        "gpu_identity": {
            "physical_index": 1,
            "uuid": "GPU-123456789",
            "pci_bus_id": "00000000:82:00.0",
        },
        "authority_contract": {
            "source_snapshot_root": str(repo),
            "output_root": str(tmp_path / "output"),
        },
        "train_caches": [
            authority.file_record(inputs / f"train_{index}.pt") for index in range(4)
        ],
        "validation_caches": [
            authority.file_record(inputs / f"validation_{index}.pt") for index in range(2)
        ],
        "fit_text_bank": {
            "artifact": authority.file_record(fit),
            "manifest": authority.file_record(fit_manifest),
        },
        "calibrations": calibration_rows,
        "radio_checkpoint": authority.file_record(radio),
        "surface_promotion": surface_promotion,
        "outputs": outputs,
    }
    manifest["training_command_contract"] = authority._build_training_command_contract(
        repo_root=repo,
        arguments=arguments,
        outputs=outputs,
        surface_promotion=surface_promotion,
    )
    return manifest, run_manifest


def test_receipt_command_requires_current_manifest_scene_seed_and_complete_argv(
    tmp_path: Path,
) -> None:
    manifest, run_manifest = _minimal_training_manifest(tmp_path)
    expected = authority.expected_training_argv(
        manifest,
        manifest_path=run_manifest,
        seed=1,
    )
    control = manifest["surface_promotion"]["selected_readouts"][1][
        "checkpoint"
    ]
    assert expected[expected.index("--surface-control-checkpoint") + 1] == control[
        "path"
    ]
    assert expected[
        expected.index("--surface-control-checkpoint-sha256") + 1
    ] == control["sha256"]
    command = {
        "run_manifest": authority.file_record(run_manifest),
        "seed": 1,
        "scene": manifest["candidate"],
        "gpu_identity": manifest["gpu_identity"],
        "argv": expected,
        "argv_sha256": authority.canonical_json_sha256(expected),
        "prepared_epoch": 1_754_000_001,
    }
    assert authority.validate_receipt_training_command(
        command,
        manifest=manifest,
        manifest_path=run_manifest,
        manifest_sha256=authority.sha256_file(run_manifest),
        seed=1,
    ) == 1_754_000_001

    another_manifest = tmp_path / "another_manifest.json"
    another_manifest.write_text("{}\n", encoding="utf-8")
    attacks = [
        {**command, "run_manifest": authority.file_record(another_manifest)},
        {**command, "scene": "another_candidate"},
        {**command, "seed": 2},
        {**command, "argv": ["bash", "arbitrary.py"]},
    ]
    for attack in attacks:
        with pytest.raises(ValueError, match="complete training argv"):
            authority.validate_receipt_training_command(
                attack,
                manifest=manifest,
                manifest_path=run_manifest,
                manifest_sha256=authority.sha256_file(run_manifest),
                seed=1,
            )

    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["training_command_contract"]["commands"][1]["argv"][-1] = "other.pt"
    with pytest.raises(ValueError, match="was not reproduced"):
        authority.validate_training_command_contract(
            tampered_manifest,
            manifest_path=run_manifest,
        )


def test_checkpoint_response_provenance_is_exact_and_pairwise_weight_bound(
    tmp_path: Path,
) -> None:
    manifest, _ = _minimal_training_manifest(tmp_path)
    diagnostic = {"path": str(tmp_path / "diagnostic.json"), "sha256": "a" * 64}
    manifest["gradient_design_diagnostic"] = diagnostic
    calibration = dict(manifest["calibrations"][1])
    calibration["response_lambdas"] = {
        "independent_response": 0.25,
        "scene_response": 0.5,
    }
    fit_files = manifest["fit_text_bank"]
    fit_binding = {
        "artifact_path": fit_files["artifact"]["path"],
        "artifact_sha256": fit_files["artifact"]["sha256"],
        "manifest_path": fit_files["manifest"]["path"],
        "manifest_sha256": fit_files["manifest"]["sha256"],
        "split": "fit",
        "query_count": 9808,
        "split_synset_tab_query_lf_sha256": "b" * 64,
        "ordered_records_sha256": "c" * 64,
        "vocabulary_sha256": "d" * 64,
        "vocabulary_manifest_sha256": "e" * 64,
        "embedding_semantic_sha256": "f" * 64,
        "embedding_tensor_sha256": "1" * 64,
        "text_encoder_snapshot_files_sha256": "2" * 64,
    }
    response_contract = {
        "fit_split_only": True,
        "benchmark_vocabulary_opened": False,
        "fit_text_bank": fit_binding,
        "calibration_manifest": calibration["manifest"]["path"],
        "calibration_manifest_sha256": calibration["manifest"]["sha256"],
        "calibration_seed": 1,
        "response_lambdas": calibration["response_lambdas"],
        "response_branch_gradient_target_ratio": 0.25,
        "total_response_gradient_ratio_upper_bound": 0.5,
        "losses": [
            authority.INDEPENDENT_RESPONSE_LOSS,
            authority.SCENE_RESPONSE_LOSS,
        ],
        "scene_response_objective": dict(authority.SCENE_RESPONSE_OBJECTIVE),
        "complete_scene_batching": True,
        "design_diagnostic": {
            **diagnostic,
            "role": "seed0_design_prior_only_per_seed_values_remeasured",
            "measured_seed": 0,
            "calibration_reuses_measured_values": False,
            "diagnostic_surface_control": manifest["surface_promotion"][
                "selected_readouts"
            ][0]["checkpoint"],
        },
    }
    assert authority.validate_response_distillation_provenance(
        response_contract,
        manifest=manifest,
        calibration=calibration,
        seed=1,
    ) == response_contract

    stale_weight = json.loads(json.dumps(response_contract))
    stale_weight["scene_response_objective"]["profile_weight"] = 0.001
    with pytest.raises(ValueError, match="response-loss provenance"):
        authority.validate_response_distillation_provenance(
            stale_weight,
            manifest=manifest,
            calibration=calibration,
            seed=1,
        )

    extra_field = {**response_contract, "legacy_compatibility": True}
    with pytest.raises(ValueError, match="response-loss provenance"):
        authority.validate_response_distillation_provenance(
            extra_field,
            manifest=manifest,
            calibration=calibration,
            seed=1,
        )


def test_schema2_gradient_diagnostic_recomputes_all_pairwise_balance_bounds() -> None:
    record = authority.FROZEN_GRADIENT_DESIGN_DIAGNOSTIC
    source = Path(authority.FROZEN_GRADIENT_DESIGN_DIAGNOSTIC_LEXICAL_PATH)
    assert authority.file_record(source) == record
    payload = json.loads(source.read_text(encoding="utf-8"))
    bindings = payload["bindings"]
    arguments = {
        "surface_control": bindings["surface_control"],
        "train_caches": bindings["train_caches"],
        "radio_checkpoint": bindings["radio_checkpoint"],
        "fit_text_bank": {
            "artifact": bindings["fit_text_bank"],
            "manifest": bindings["fit_text_bank_manifest"],
        },
    }
    assert authority.validate_gradient_design_diagnostic(
        payload, **arguments
    ) == payload
    assert payload["component_balance"][
        "weighted_profile_to_ranking_gradient_ratio"
    ] == pytest.approx(1.0449448819955465, rel=0.0, abs=0.0)

    attacks = []
    stale_schema = json.loads(json.dumps(payload))
    stale_schema["schema_version"] = 1
    attacks.append(stale_schema)
    stale_composite = json.loads(json.dumps(payload))
    stale_composite["losses"]["scene_response"] += 1e-4
    attacks.append(stale_composite)
    stale_lambda = json.loads(json.dumps(payload))
    stale_lambda["equal_surface_gradient_lambdas"]["scene_profile"] += 1e-4
    attacks.append(stale_lambda)
    stale_bound = json.loads(json.dumps(payload))
    stale_bound["weighted_component_gradient_l2_upper_bounds"][
        "triangle_sum"
    ] += 1e-4
    attacks.append(stale_bound)
    stale_balance = json.loads(json.dumps(payload))
    stale_balance["component_balance"][
        "weighted_profile_to_ranking_gradient_ratio"
    ] = 1.0
    attacks.append(stale_balance)
    stale_training = json.loads(json.dumps(payload))
    stale_training["bindings"]["training_implementation"]["sha256"] = "0" * 64
    attacks.append(stale_training)
    stale_loss = json.loads(json.dumps(payload))
    stale_loss["bindings"]["loss_implementation"]["sha256"] = "0" * 64
    attacks.append(stale_loss)
    for attack in attacks:
        with pytest.raises(ValueError):
            authority.validate_gradient_design_diagnostic(attack, **arguments)


def test_journal_and_telemetry_intervals_reject_cross_seed_replay(tmp_path: Path) -> None:
    start = 1_754_000_000
    end = start + 10
    journal = tmp_path / "seed0.kernel.log"
    journal.write_text(
        f"surface_text_response_seed=0\tstart_epoch={start}\tend_epoch={end}\n"
        "kernel interval clean\n",
        encoding="utf-8",
    )
    record = authority._kernel_journal_record(
        journal,
        seed=0,
        start_epoch=start,
        end_epoch=end,
    )
    assert record["fault_count"] == 0
    with pytest.raises(ValueError, match="interval header differs"):
        authority._kernel_journal_record(
            journal,
            seed=1,
            start_epoch=start,
            end_epoch=end,
        )

    telemetry = tmp_path / "seed0.csv"
    first_timestamp = "2026-08-01T12:00:02+08:00"
    last_timestamp = "2026-08-01T12:00:03+08:00"
    telemetry.write_text(
        ",".join(authority.TELEMETRY_COLUMNS)
        + "\n"
        + f"{first_timestamp},1,0000:82:00.0,50,20,300,10,100,P2,running\n"
        + f"{last_timestamp},1,0000:82:00.0,51,21,300,11,101,P2,running\n",
        encoding="utf-8",
    )
    interval = authority._telemetry_interval_record(
        telemetry,
        seed=0,
        receipt_summary={
            "sample_count": 2,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        },
    )
    first_epoch = int(interval["first_epoch"])
    authority.validate_seed_execution_timeline(
        seed=0,
        gpu_preflight_epoch=first_epoch - 3,
        command_prepared_epoch=first_epoch - 2,
        journal_start_epoch=first_epoch - 1,
        telemetry_first_epoch=interval["first_epoch"],
        telemetry_last_epoch=interval["last_epoch"],
        journal_end_epoch=first_epoch + 2,
        gpu_postflight_epoch=first_epoch + 3,
    )
    with pytest.raises(ValueError, match="timeline is not strictly bound"):
        authority.validate_seed_execution_timeline(
            seed=1,
            gpu_preflight_epoch=first_epoch + 20,
            command_prepared_epoch=first_epoch + 21,
            journal_start_epoch=first_epoch + 22,
            telemetry_first_epoch=interval["first_epoch"],
            telemetry_last_epoch=interval["last_epoch"],
            journal_end_epoch=first_epoch + 30,
            gpu_postflight_epoch=first_epoch + 31,
        )


def test_completion_rejects_replayed_clean_seed_evidence() -> None:
    rows = []
    for seed in range(3):
        rows.append(
            {
                "guard_command": {"sha256": str(seed) * 64},
                "telemetry_interval": {
                    "sha256": str(seed + 3) * 64,
                    "row_interval_sha256": str(seed + 6) * 64,
                },
                "kernel_journal": {"sha256": str(seed + 9) * 64},
                "execution_timeline": {
                    "journal_start_epoch": 100 + seed * 20,
                    "journal_end_epoch": 110 + seed * 20,
                },
            }
        )
    authority.validate_cross_seed_replay(rows)
    rows[2]["telemetry_interval"] = dict(rows[0]["telemetry_interval"])
    with pytest.raises(ValueError, match="cross-seed replay"):
        authority.validate_cross_seed_replay(rows)


def test_nofollow_nonblocking_lock_rejects_symlink_and_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "frozen-source-without-git"
    repo.mkdir()
    repo.chmod(0o555)
    lock_parent = tmp_path / "main"
    lock_parent.mkdir()
    lock_root = lock_parent / "output"
    lock_root.mkdir()
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lock_root)
    monkeypatch.setattr(
        authority,
        "GPU1_SINGLETON_ADDRESS",
        f"\0radio-gs-text-lock-test-{os.getpid()}".encode("ascii"),
    )
    output = lock_root / "runs/run"
    status = authority.run_locked(
        repo_root=repo,
        lock_root=lock_root,
        output_root=output,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )
    assert status == 0
    global_lock = lock_root / ".physical_gpu1.lock"
    assert global_lock.is_file() and not global_lock.is_symlink()

    descriptor = authority._acquire_nofollow_lock(global_lock)
    try:
        with pytest.raises(RuntimeError, match="already held"):
            authority.run_locked(
                repo_root=repo,
                lock_root=lock_root,
                output_root=output,
                command=[sys.executable, "-c", "raise SystemExit(0)"],
            )
    finally:
        os.close(descriptor)

    global_lock.unlink()
    target = tmp_path / "alias.lock"
    target.touch()
    global_lock.symlink_to(target)
    with pytest.raises(OSError):
        authority.run_locked(
            repo_root=repo,
            lock_root=lock_root,
            output_root=output,
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )


def test_inherited_fd_contract_cannot_be_forged_by_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "main/output"
    output = lock_root / "run"
    (output / "locks").mkdir(parents=True)
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lock_root)
    address = f"\0radio-gs-text-inherited-{os.getpid()}".encode("ascii")
    monkeypatch.setattr(authority, "GPU1_SINGLETON_ADDRESS", address)
    global_path = lock_root / ".physical_gpu1.lock"
    run_path = output / "locks/text_response_distill.run.lock"
    global_fd = authority._acquire_nofollow_lock(global_path)
    run_fd = authority._acquire_nofollow_lock(run_path)
    singleton_fd = gpu1_lock._open_kernel_singleton(address)
    try:
        root_binding = authority.inspect_canonical_lock_root(lock_root)
        monkeypatch.setenv(
            authority.LOCK_ROOT_BINDING_ENV,
            authority.canonical_json_sha256(root_binding),
        )
        monkeypatch.setenv(gpu1_lock.SINGLETON_FD_ENV, str(singleton_fd))
        monkeypatch.setenv(
            gpu1_lock.SINGLETON_PROTOCOL_ENV,
            gpu1_lock._singleton_protocol(address),
        )
        verified = authority.verify_inherited_locks(
            lock_root=lock_root,
            output_root=output,
            global_descriptor=global_fd,
            run_descriptor=run_fd,
            singleton_descriptor=singleton_fd,
        )
        assert verified["global_lock"] == str(global_path)
        assert verified["kernel_singleton"]["protocol"] == gpu1_lock._singleton_protocol(
            address
        )
        monkeypatch.setenv(authority.LOCK_ROOT_BINDING_ENV, "0" * 64)
        with pytest.raises(ValueError, match="root binding differs"):
            authority.verify_inherited_locks(
                lock_root=lock_root,
                output_root=output,
                global_descriptor=global_fd,
                run_descriptor=run_fd,
                singleton_descriptor=singleton_fd,
            )
        monkeypatch.setenv(
            authority.LOCK_ROOT_BINDING_ENV,
            authority.canonical_json_sha256(root_binding),
        )
        read_fd, write_fd = os.pipe()
        try:
            with pytest.raises(ValueError, match="does not own"):
                authority.verify_inherited_locks(
                    lock_root=lock_root,
                    output_root=output,
                    global_descriptor=read_fd,
                    run_descriptor=run_fd,
                    singleton_descriptor=singleton_fd,
                )
            monkeypatch.setenv(gpu1_lock.SINGLETON_FD_ENV, str(read_fd))
            with pytest.raises(OSError):
                authority.verify_inherited_locks(
                    lock_root=lock_root,
                    output_root=output,
                    global_descriptor=global_fd,
                    run_descriptor=run_fd,
                    singleton_descriptor=read_fd,
                )
        finally:
            os.close(read_fd)
            os.close(write_fd)
    finally:
        os.close(singleton_fd)
        os.close(run_fd)
        os.close(global_fd)


def test_text_and_surface_singleton_blocks_unlink_recreate_cross_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "main/output"
    lock_root.mkdir(parents=True)
    output = lock_root / "text-run"
    lock_path = lock_root / ".physical_gpu1.lock"
    address = f"\0radio-gs-cross-runner-{os.getpid()}".encode("ascii")
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lock_root)
    monkeypatch.setattr(authority, "GPU1_SINGLETON_ADDRESS", address)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot.chmod(0o555)

    surface_file_lock = gpu1_lock._open_canonical_lock(lock_path)
    surface_singleton = gpu1_lock._open_kernel_singleton(address)
    try:
        lock_path.unlink()
        with pytest.raises(OSError) as caught:
            authority.run_locked(
                repo_root=snapshot,
                lock_root=lock_root,
                output_root=output,
                command=[sys.executable, "-c", "raise SystemExit(0)"],
            )
        assert caught.value.errno == errno.EADDRINUSE
        assert lock_path.is_file()
    finally:
        os.close(surface_singleton)
        os.close(surface_file_lock)


def test_run_locked_rejects_any_noncanonical_lock_root(tmp_path: Path) -> None:
    source = tmp_path / "snapshot"
    source.mkdir()
    source.chmod(0o555)
    fake = tmp_path / "other/output"
    fake.mkdir(parents=True)
    with pytest.raises(ValueError, match="/root/RADIO-GS/output"):
        authority.run_locked(
            repo_root=source,
            lock_root=fake,
            output_root=fake / "run",
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )


def test_run_locked_accepts_only_the_controlled_root_symlink_from_readonly_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "frozen-snapshot"
    snapshot.mkdir()
    snapshot.chmod(0o555)
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    resolved_root = tmp_path / "mounted-output"
    resolved_root.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(resolved_root, target_is_directory=True)
    output = lexical_root / "optimization/run"
    address = f"\0radio-gs-text-root-symlink-{os.getpid()}".encode("ascii")
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    monkeypatch.setattr(authority, "GPU1_SINGLETON_ADDRESS", address)

    status = authority.run_locked(
        repo_root=snapshot,
        lock_root=lexical_root,
        output_root=output,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )
    assert status == 0
    binding = authority.inspect_canonical_lock_root(lexical_root)
    assert binding["entry_type"] == "controlled_symlink"
    assert binding["resolved_path"] == str(resolved_root.resolve())
    assert (resolved_root / "optimization/run/locks/text_response_distill.run.lock").is_file()
    assert (resolved_root / ".physical_gpu1.lock").is_file()
    assert not (snapshot / ".git").exists()
    output_row = authority._seed_outputs(
        resolved_root / "optimization/run",
        "context_c1024_geometric",
    )[0]
    arguments = {
        "train_caches": "train*.pt",
        "validation_caches": "validation*.pt",
        "fit_text_bank": "fit.pt",
        "fit_text_bank_manifest": "fit.json",
        "calibration_manifests": {
            "0": "calibration0.json",
            "1": "calibration1.json",
            "2": "calibration2.json",
        },
        "run_manifest": str(output / "run_manifest.json"),
        "radio_checkpoint": "radio.pt",
        "output_root": str(output),
    }
    control = tmp_path / "surface-control-seed0.pt"
    control.write_bytes(b"surface-control")
    argv = authority._training_argv(
        repo_root=snapshot,
        arguments=arguments,
        output_row=output_row,
        surface_control={
            "seed": 0,
            "checkpoint": authority.file_record(control),
        },
        seed=0,
    )
    assert argv[argv.index("--output") + 1] == str(
        output / "readouts/context_c1024_geometric_text_response_seed0.pt"
    )


def test_bound_root_rejects_symlink_repoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    first = tmp_path / "first-output"
    second = tmp_path / "second-output"
    first.mkdir()
    second.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(first, target_is_directory=True)
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    frozen = authority.inspect_canonical_lock_root(lexical_root)

    lexical_root.unlink()
    lexical_root.symlink_to(second, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink/target identity changed"):
        authority.validate_canonical_lock_root_binding(
            frozen,
            lock_root=lexical_root,
        )


def test_controlled_root_still_rejects_every_child_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    resolved_root = tmp_path / "mounted-output"
    resolved_root.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(resolved_root, target_is_directory=True)
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (resolved_root / "optimization").symlink_to(escaped, target_is_directory=True)
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    monkeypatch.setattr(
        authority,
        "GPU1_SINGLETON_ADDRESS",
        f"\0radio-gs-text-child-symlink-{os.getpid()}".encode("ascii"),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot.chmod(0o555)

    with pytest.raises(ValueError, match="symlink/non-directory output component"):
        authority.run_locked(
            repo_root=snapshot,
            lock_root=lexical_root,
            output_root=lexical_root / "optimization/run",
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )
    assert list(escaped.iterdir()) == []


def test_manifest_path_contract_rechecks_output_directory_identity_per_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_parent = tmp_path / "main-repository"
    lexical_parent.mkdir()
    resolved_root = tmp_path / "mounted-output"
    resolved_root.mkdir()
    lexical_root = lexical_parent / "output"
    lexical_root.symlink_to(resolved_root, target_is_directory=True)
    output = lexical_root / "optimization/text-run"
    monkeypatch.setattr(authority, "CANONICAL_LOCK_ROOT", lexical_root)
    root_binding = authority.inspect_canonical_lock_root(lexical_root)
    output_binding = authority._secure_output_directory(
        output,
        root_binding=root_binding,
        create=True,
    )
    directory_bindings = authority._output_directory_bindings(
        output,
        root_binding=root_binding,
        create=True,
    )
    manifest = {
        "authority_contract": {
            "main_output_root": str(lexical_root),
            "main_output_root_binding": root_binding,
            "output_root": str(output),
            "output_root_binding": output_binding,
            "output_directory_bindings": directory_bindings,
            "root_path_protocol": authority.LOCK_ROOT_BINDING_VERSION,
            "global_gpu_lock": str(lexical_root / ".physical_gpu1.lock"),
            "output_run_lock": str(output / "locks/text_response_distill.run.lock"),
        }
    }
    authority.validate_authority_path_contract(manifest)

    readouts = resolved_root / "optimization/text-run/readouts"
    readouts.rename(readouts.with_name("readouts-old"))
    readouts.mkdir()
    with pytest.raises(ValueError, match="output directory identity changed"):
        authority.validate_authority_path_contract(manifest)


def test_real_deployment_lock_root_preflight_is_read_only() -> None:
    deployed_root = Path("/root/RADIO-GS/output")
    if REPO_ROOT != Path("/root/RADIO-GS") or not deployed_root.is_symlink():
        pytest.skip("real mounted RADIO-GS output root is not present")
    before = os.lstat(deployed_root)
    binding = authority.inspect_canonical_lock_root(deployed_root)
    after = os.lstat(deployed_root)
    assert binding["lexical_path"] == str(deployed_root)
    assert binding["entry_type"] == "controlled_symlink"
    assert Path(binding["resolved_path"]) == deployed_root.resolve(strict=True)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_authority_independently_recomputes_v3_robust_response_selection() -> None:
    history = _history()
    def recompute(value: object) -> tuple[int, float]:
        return authority.recompute_response_epoch_selection(
            value, selection_contract=authority.LEGACY_EPOCH_SELECTION
        )

    assert recompute(history) == (1, 0.959)
    tampered = [dict(row) for row in history]
    tampered[0]["selection_score"] = 0.99
    with pytest.raises(ValueError, match="independently reproduced"):
        recompute(tampered)

    stale_fit_feasibility = json.loads(json.dumps(history))
    stale_fit_feasibility[1]["fit_response_per_scene_control_feasible"] = False
    with pytest.raises(ValueError, match="fit-scene feasibility"):
        recompute(stale_fit_feasibility)

    missing_scene_metrics = json.loads(json.dumps(history))
    del missing_scene_metrics[1]["text_response_scene_metrics"]
    with pytest.raises(ValueError, match="scene_metrics must be non-empty"):
        recompute(missing_scene_metrics)

    stale_objective = json.loads(json.dumps(history))
    stale_objective[2]["scene_response_objective"]["profile_weight"] = 0.001
    with pytest.raises(ValueError, match="scene-response objective"):
        recompute(stale_objective)


def test_manifest_accepts_only_the_exact_registered_legacy_contract() -> None:
    legacy = {"training_contract": authority.LEGACY_TRAINING_CONTRACT}
    registered_path = Path(authority.REGISTERED_LEGACY_MANIFEST["path"])
    registered_sha = authority.REGISTERED_LEGACY_MANIFEST["sha256"]
    assert authority._manifest_selection_contract(
        legacy,
        digest=registered_sha,
        source=registered_path,
    ) == authority.LEGACY_EPOCH_SELECTION

    with pytest.raises(ValueError, match="exact registered formal legacy"):
        authority._manifest_selection_contract(
            legacy,
            digest="0" * 64,
            source=registered_path,
        )
    with pytest.raises(ValueError, match="exact registered formal legacy"):
        authority._manifest_selection_contract(
            legacy,
            digest=registered_sha,
            source=registered_path.with_name("copied_manifest.json"),
        )

    current = {"training_contract": authority.TRAINING_CONTRACT}
    assert authority._manifest_selection_contract(
        current,
        digest="0" * 64,
        source=Path("/unregistered/current/run_manifest.json"),
    ) == authority.EPOCH_SELECTION


def test_authority_replays_v4_accepted_anchor_hash_chain_best_and_patience() -> None:
    checkpoint, report = _v4_checkpoint()
    assert authority.validate_v4_proposal_checkpoint(
        checkpoint, report=report
    ) == (1, 0.959)
    accepted_anchor = checkpoint["accepted_anchor"]
    assert accepted_anchor["accepted_proposal_count"] == 6
    assert accepted_anchor["rejected_proposal_count"] == 5
    assert checkpoint["history"][-1]["patience_stale_after_proposal"] == 10
    assert checkpoint["history"][-1]["patience_stop_after_proposal"] is True


def test_authority_v4_proposal_contract_is_fail_closed_under_tampering() -> None:
    def alpha(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["proposal"]["alpha_numerator"] = 2

    def reset(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["proposal"]["optimizer_state_reset"] = False

    def validation_count(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["proposal"]["validation_evaluations"] = 2

    def backtracking(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["proposal"]["backtracking"] = "halve_until_fit"

    def anchor_chain(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][2]["proposal"]["source_anchor_epoch"] = 0

    def acceptance(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][2]["accepted"] = True

    def robust_feasibility(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][2]["response_selection_feasible"] = True

    def best_fields(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["best_updated"] = False

    def patience(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][3]["patience_stale_after_proposal"] = 99

    def loss_state(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["loss_measurement_state"] = "trial_state"

    def loss_mirror(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][1]["loss"] += 1.0

    def history_hash(checkpoint: dict[str, object]) -> None:
        checkpoint["history"][4]["history_hash_chain"]["sha256"] = "f" * 64

    for mutate in (
        alpha,
        reset,
        validation_count,
        backtracking,
        anchor_chain,
        acceptance,
        robust_feasibility,
        best_fields,
        patience,
        loss_state,
        loss_mirror,
        history_hash,
    ):
        checkpoint, report = _v4_checkpoint()
        mutate(checkpoint)
        with pytest.raises(ValueError):
            authority.validate_v4_proposal_checkpoint(checkpoint, report=report)


def test_authority_v4_binds_checkpoint_and_report_terminal_fields() -> None:
    checkpoint, report = _v4_checkpoint()
    checkpoint["accepted_anchor"]["accepted_proposal_count"] += 1
    with pytest.raises(ValueError, match="accepted-anchor/best/hash-chain"):
        authority.validate_v4_proposal_checkpoint(checkpoint, report=report)

    checkpoint, report = _v4_checkpoint()
    checkpoint["training_config"]["proposal_state_machine"] = {}
    with pytest.raises(ValueError, match="proposal contract"):
        authority.validate_v4_proposal_checkpoint(checkpoint, report=report)

    checkpoint, report = _v4_checkpoint()
    report["best_state_dict_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="trainer report proposal"):
        authority.validate_v4_proposal_checkpoint(checkpoint, report=report)


def test_runtime_closure_covers_authority_transitive_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The production runner always exports this before authority closure
    # capture.  Reproduce that contract even when pytest is invoked directly
    # with the environment's non-symlink CPython executable.
    monkeypatch.setenv("RADIO_GS_REPO_ROOT", str(REPO_ROOT))
    closure = authority.build_runtime_closure(REPO_ROOT)
    files = closure["repository_sources"]["files"]
    assert {
        "radio_gs/scripts/run_surface_region_text_response_distill.sh",
        "radio_gs/scripts/surface_text_response_distill_authority.py",
        "radio_gs/scripts/train_surface_region_text_response_distill.py",
        "radio_gs/scripts/finalize_gpu_guard_receipt.py",
        "radio_gs/scripts/finalize_surface_text_response_promotion.py",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/models/siglip_projection.py",
        "radio_gs/utils/immutable_artifacts.py",
    } <= set(files)
    assert closure["digest"] == authority.canonical_json_sha256(
        {
            "schema_version": closure["schema_version"],
            "repository_sources": closure["repository_sources"],
            "runtime_fingerprint": closure["runtime_fingerprint"],
        }
    )
    assert promotion._validate_distill_runtime_closure(
        closure,
        source_snapshot_root=REPO_ROOT,
    ) == closure["digest"]
    tampered = {
        **closure,
        "repository_sources": {
            **closure["repository_sources"],
            "files": {
                **closure["repository_sources"]["files"],
                "radio_gs/interfaces/surface_region_summary.py": "0" * 64,
            },
        },
    }
    with pytest.raises(ValueError, match="source closure changed"):
        promotion._validate_distill_runtime_closure(
            tampered,
            source_snapshot_root=REPO_ROOT,
        )


def test_readonly_source_snapshot_rejects_mutability_and_aliases(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    package = snapshot / "radio_gs"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="writable entry"):
        authority.verify_readonly_source_snapshot(snapshot)

    source.chmod(0o444)
    package.chmod(0o555)
    snapshot.chmod(0o555)
    assert authority.verify_readonly_source_snapshot(snapshot) == snapshot.resolve()

    alias = tmp_path / "snapshot-link"
    alias.symlink_to(snapshot, target_is_directory=True)
    with pytest.raises(ValueError, match="must not traverse a symlink"):
        authority.verify_readonly_source_snapshot(alias)

    snapshot.chmod(0o755)
    package.chmod(0o755)
    hardlink = package / "hardlink.py"
    os.link(source, hardlink)
    package.chmod(0o555)
    snapshot.chmod(0o555)
    with pytest.raises(ValueError, match="multiply linked file"):
        authority.verify_readonly_source_snapshot(snapshot)


def test_training_inputs_do_not_use_generic_torch_load_fallback() -> None:
    trainer = (
        REPO_ROOT / "radio_gs/scripts/train_surface_region_text_response_distill.py"
    ).read_text(encoding="utf-8")
    base = (
        REPO_ROOT / "radio_gs/scripts/train_surface_region_summary_readout.py"
    ).read_text(encoding="utf-8")
    assert "torch.load(" not in trainer
    assert "torch.load(path, map_location=\"cpu\")" not in base
    assert "load_torch_mapping" in trainer and "load_torch_mapping" in base
    assert "load_surface_region_summary_readout_v2" in trainer


def test_promotion_authority_rejects_bound_kernel_xid(
    tmp_path: Path,
) -> None:
    identity = {
        "physical_index": 1,
        "uuid": "GPU-123456789",
        "pci_bus_id": "00000000:82:00.0",
    }
    evidence: dict[str, object] = {}
    for field in (
        "checkpoint",
        "report",
        "training_log",
        "audit_report",
        "guard_command",
        "guard_telemetry",
        "guard_receipt",
    ):
        path = tmp_path / field
        path.write_text(field, encoding="utf-8")
        evidence[field] = promotion._file_record(path)
    for phase, field in (("pre_seed0", "gpu_preflight"), ("post_seed0", "gpu_postflight")):
        path = tmp_path / f"{field}.json"
        observed_epoch = 1_754_000_000 if field == "gpu_preflight" else 1_754_000_010
        path.write_text(
            json.dumps(
                {
                    "status": "physical_gpu1_idle_and_pcie_responsive",
                    "phase": phase,
                    "gpu_identity": identity,
                    "compute_owners": [],
                    "observed_epoch": observed_epoch,
                }
            ),
            encoding="utf-8",
        )
        evidence[field] = {
            **promotion._file_record(path),
            "observed_epoch": observed_epoch,
        }
    journal = tmp_path / "kernel.log"
    journal.write_text(
        "surface_text_response_seed=0\tstart_epoch=1754000001\tend_epoch=1754000009\n"
        "NVRM: Xid 79, GPU has fallen off the bus\n",
        encoding="utf-8",
    )
    evidence["kernel_journal"] = {
        **promotion._file_record(journal),
        "seed": 0,
        "start_epoch": 1_754_000_001,
        "end_epoch": 1_754_000_009,
        "fault_count": 0,
    }
    with pytest.raises(ValueError, match="Xid/PCIe faults"):
        promotion._validate_authority_seed_evidence(
            evidence,
            seed=0,
            gpu_identity=identity,
        )
