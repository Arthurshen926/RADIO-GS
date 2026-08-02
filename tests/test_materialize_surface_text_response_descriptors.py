from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

import radio_gs.scripts.materialize_surface_text_response_descriptors as module
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _cache(path: Path, *, scene: str, radio_sha: str) -> None:
    record = {
        "region_id": f"region-{scene}",
        "scene": scene,
        "seed": 17,
        "physical_radius_m": 0.25,
        "teacher_views": [0, 1],
        "teacher_target_sha256": "a" * 64,
        "teacher_support_sha256": "b" * 64,
    }
    metadata = {
        "schema_version": 3,
        "split_role": "validation",
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "annotations_opened": False,
        "labels_opened": False,
        "instances_opened": False,
        "masks_opened": False,
        "text_opened": False,
        "physical_space_disjoint": True,
        "complete_scene_regions": True,
        "failed_scenes": [],
        "teacher_regions_saturated": 0,
        "region_records": [record],
        "scene_names": [scene],
        "scene_region_counts": {scene: 1},
        "region_contract_sha256": "c" * 64,
        "region_contract": {"name": "mock-surface-contract"},
        "radio_checkpoint_sha256": radio_sha,
        "split_file_sha256": "d" * 64,
        "teacher_region_semantics": (
            "fixed_core_geodesic_support_without_input_context_v1"
        ),
        "teacher_region_contract": {"name": "fixed-core"},
        "teacher_region_contract_sha256": "e" * 64,
        "teacher_target_source": "exact_cache_replay",
        "teacher_target_protocol_sha256": "f" * 64,
        "excluded_physical_spaces": ["heldout-space"],
        "exclusion_files": [{"path": "/frozen/split.txt", "sha256": "1" * 64}],
    }
    payload = {
        "radio_features": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        "geometry": torch.zeros(1, 2, 14),
        "token_mask": torch.tensor([[True, True]]),
        "reliability": torch.ones(1, 2),
        "official_crop_summaries": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
        ),
        "teacher_mask": torch.tensor([[True, True]]),
        "anchor_index": torch.tensor([0]),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    radio = tmp_path / "radio.pt"
    radio.write_bytes(b"fixed-radio")
    cache = tmp_path / "validation_shard0.pt"
    _cache(cache, scene="scene-validation", radio_sha=_sha256(radio))
    _, cache_meta = module._load_validation_caches([cache])

    checkpoint = tmp_path / "candidate_text_response_seed0.pt"
    report = checkpoint.with_suffix(".pt.json")
    run_manifest = tmp_path / "run_manifest.json"
    materializer_relative = (
        "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    )
    run_payload = {
        "schema_version": 1,
        "artifact_type": "surface_region_text_response_distill_run",
        "candidate": "context_c1024_geometric",
        "validation_caches": cache_meta["cache_bindings"],
        "radio_checkpoint": {"path": str(radio), "sha256": _sha256(radio)},
        "outputs": [
            {"seed": 0, "checkpoint": str(checkpoint), "report": str(report)},
            {
                "seed": 1,
                "checkpoint": str(tmp_path / "candidate_text_response_seed1.pt"),
                "report": str(tmp_path / "candidate_text_response_seed1.pt.json"),
            },
            {
                "seed": 2,
                "checkpoint": str(tmp_path / "candidate_text_response_seed2.pt"),
                "report": str(tmp_path / "candidate_text_response_seed2.pt.json"),
            },
        ],
        "implementation_sources": {
            materializer_relative: _sha256(Path(module.__file__).resolve())
        },
    }
    _write_json(run_manifest, run_payload)

    model = SurfaceRegionSummaryReadoutV2(feature_dim=4, hidden_dim=2)
    architecture = model.architecture("c" * 64)
    baseline = {
        "summary_token_cosine": 0.4,
        "mean_descriptor_cosine": 0.4,
        "all_view_descriptor_cosine": 0.4,
    }
    validation = {
        "summary_token_cosine": 0.6,
        "mean_descriptor_cosine": 0.6,
        "all_view_descriptor_cosine": 0.6,
    }
    fit_binding = {"artifact_sha256": "2" * 64, "query_count": 806}
    provenance = {
        "uses_benchmark_scenes": False,
        "uses_benchmark_test_vocabulary": False,
        "scene_disjoint": True,
        "custom_text_projection": False,
        "train": {"scenes": ["scene-train"]},
        "validation": cache_meta["checkpoint_validation"],
        "region_contract_sha256": "c" * 64,
        "region_contract": {"name": "mock-surface-contract"},
        "random_seed_contract": {
            "seed": 0,
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        },
        "text_response_distillation": {
            "response_lambda": 0.25,
            "calibration_manifest": str(tmp_path / "calibration.json"),
            "calibration_manifest_sha256": "3" * 64,
            "fit_text_bank": fit_binding,
        },
        "distill_run_manifest": {
            "path": str(run_manifest),
            "sha256": _sha256(run_manifest),
            "candidate": "context_c1024_geometric",
        },
    }
    checkpoint_payload = {
        "schema_version": 3,
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "provenance": provenance,
        "training_config": {"seed": 0},
        "best_epoch": 1,
        "best_selection_score": 0.6,
        "untrained_baseline": baseline,
        "untrained_baseline_score": 0.4,
    }
    torch.save(checkpoint_payload, checkpoint)
    report_payload = {
        "output": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "architecture": architecture,
        "best_epoch": 1,
        "best_selection_score": 0.6,
        "untrained_baseline": baseline,
        "selection_score_delta": 0.6 - 0.4,
        "validation": validation,
        "response_lambda": 0.25,
        "calibration_manifest": str(tmp_path / "calibration.json"),
        "calibration_manifest_sha256": "3" * 64,
        "fit_text_bank_sha256": "2" * 64,
        "fit_query_count": 806,
        "distill_run_manifest": str(run_manifest),
        "distill_run_manifest_sha256": _sha256(run_manifest),
        "validation_caches": cache_meta["cache_bindings"],
        "train_scenes": 1,
        "validation_scenes": 1,
        "scene_overlap": [],
    }
    _write_json(report, report_payload)

    class IdentityHead:
        @classmethod
        def from_radio_checkpoint(cls, _: str) -> torch.nn.Module:
            return torch.nn.Identity()

    monkeypatch.setattr(module, "SigLIP2SummaryHead", IdentityHead)
    return {
        "radio": radio,
        "cache": cache,
        "checkpoint": checkpoint,
        "report": report,
        "run_manifest": run_manifest,
    }


def _args(paths: dict[str, Path], output: Path, cache: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        validation_cache=[str(cache or paths["cache"])],
        readout_checkpoint=str(paths["checkpoint"]),
        readout_binding_manifest=None,
        radio_checkpoint=str(paths["radio"]),
        method_id="distilled-candidate",
        batch_size=2,
        device="cpu",
        output=str(output),
    )


def _rebind_distill_manifest(paths: dict[str, Path]) -> None:
    manifest_sha = _sha256(paths["run_manifest"])
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    checkpoint["provenance"]["distill_run_manifest"]["sha256"] = manifest_sha
    torch.save(checkpoint, paths["checkpoint"])

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = _sha256(paths["checkpoint"])
    report["distill_run_manifest_sha256"] = manifest_sha
    _write_json(paths["report"], report)


def _upgrade_to_authority_schema_v2(
    paths: dict[str, Path], tmp_path: Path
) -> Path:
    snapshot_root = tmp_path / "frozen-producer-snapshot"
    producer_source = snapshot_root / (
        "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    )
    producer_source.parent.mkdir(parents=True)
    producer_source.write_bytes(b"frozen producer materializer implementation\n")
    assert _sha256(producer_source) != _sha256(Path(module.__file__).resolve())

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest.update(
        {
            "surface_promotion": {},
            "train_caches": [],
            "fit_text_bank": {},
            "calibration_manifest": {},
            "training_contract": {},
            "thermal_safety_contract": {},
            "authority_status": "query_free_three_seed_gpu1_run_frozen",
            "calibration_audit": {},
            "initial_gpu_preflight": {},
            "gpu_identity": {},
            "runtime_closure": {},
            "authority_contract": {"source_snapshot_root": str(snapshot_root)},
            "training_command_contract": {},
        }
    )
    extra_output_fields = {
        "training_log",
        "audit_report",
        "guard_command",
        "guard_telemetry",
        "guard_receipt",
        "kernel_journal",
        "gpu_preflight",
        "gpu_postflight",
        "terminal",
    }
    for row in manifest["outputs"]:
        for field in extra_output_fields:
            row[field] = str(tmp_path / f"seed{row['seed']}.{field}")
    relative = "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    manifest["implementation_sources"] = {relative: _sha256(producer_source)}
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)
    return producer_source


def _upgrade_to_authority_schema_v3(
    paths: dict[str, Path], tmp_path: Path
) -> dict[str, object]:
    producer_source = _upgrade_to_authority_schema_v2(paths, tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    architecture = checkpoint["architecture"]
    provenance = checkpoint["provenance"]
    train_cache = tmp_path / "train_shard0.pt"
    train_cache.write_bytes(b"fixed-train-cache")
    train_caches = [{"path": str(train_cache), "sha256": _sha256(train_cache)}]
    validation_caches = list(provenance["validation"]["cache_bindings"])
    provenance["train"]["cache_bindings"] = train_caches
    control_train_provenance = dict(provenance["train"])
    control_train_provenance.pop("cache_bindings")
    control_validation_provenance = dict(provenance["validation"])
    control_validation_provenance.pop("cache_bindings")

    fit_artifact = tmp_path / "fit_text_bank.pt"
    fit_manifest = tmp_path / "fit_text_bank.manifest.json"
    fit_artifact.write_bytes(b"fixed-fit-text-bank")
    fit_manifest.write_bytes(b"fixed-fit-text-bank-manifest")
    fit_binding = {
        "artifact_path": str(fit_artifact),
        "artifact_sha256": _sha256(fit_artifact),
        "manifest_path": str(fit_manifest),
        "manifest_sha256": _sha256(fit_manifest),
        "split": "fit",
        "query_count": 806,
        "split_synset_tab_query_lf_sha256": "4" * 64,
        "ordered_records_sha256": "5" * 64,
        "vocabulary_sha256": "6" * 64,
        "vocabulary_manifest_sha256": "7" * 64,
        "embedding_semantic_sha256": "8" * 64,
        "embedding_tensor_sha256": "9" * 64,
        "text_encoder_snapshot_files_sha256": "a" * 64,
    }
    controls = []
    for seed in range(3):
        control_path = tmp_path / f"surface_control_seed{seed}.pt"
        control_payload = {
            "schema_version": 3,
            "architecture": architecture,
            "state_dict": checkpoint["state_dict"],
            "provenance": {
                "uses_benchmark_scenes": False,
                "uses_benchmark_test_vocabulary": False,
                "train": control_train_provenance,
                "validation": control_validation_provenance,
                "region_contract": provenance["region_contract"],
                "region_contract_sha256": provenance["region_contract_sha256"],
                "random_seed_contract": {
                    "seed": seed,
                    "model_initialization": True,
                    "data_order": True,
                    "canonical_noise": True,
                },
            },
            "training_config": {"seed": seed},
            "best_epoch": 1,
            "best_selection_score": 0.4,
        }
        torch.save(control_payload, control_path)
        controls.append(
            {
                "path": str(control_path),
                "sha256": _sha256(control_path),
                "seed": seed,
                "architecture": architecture,
                "train_caches": train_caches,
                "validation_caches": validation_caches,
                "source_best_epoch": 1,
                "source_best_selection_score": 0.4,
            }
        )

    diagnostic = tmp_path / "gradient_diagnostic.json"
    _write_json(diagnostic, {"formal_design_evidence": True})
    design = {
        "path": str(diagnostic),
        "sha256": _sha256(diagnostic),
        "role": "seed0_design_prior_only_per_seed_values_remeasured",
        "measured_seed": 0,
        "calibration_reuses_measured_values": False,
        "diagnostic_surface_control": {
            "path": controls[0]["path"],
            "sha256": controls[0]["sha256"],
        },
    }
    lambdas_by_seed = [
        {"independent_response": 0.5 + seed, "scene_response": 0.25 + seed}
        for seed in range(3)
    ]
    calibrations = []
    for seed in range(3):
        calibration_path = tmp_path / f"calibration_seed{seed}.json"
        audit_path = tmp_path / f"calibration_seed{seed}.audit.json"
        surface_norm = 2.0
        independent_norm = surface_norm * 0.25 / lambdas_by_seed[seed][
            "independent_response"
        ]
        scene_norm = surface_norm * 0.25 / lambdas_by_seed[seed][
            "scene_response"
        ]
        calibration_payload = {
                "schema_version": 2,
                "artifact_type": "surface_text_response_gradient_calibration",
                "algorithm_version": (
                    "per-seed-surface-warmstart-dual-response-pairwise-gradient-budget-v3"
                ),
                "benchmark_vocabulary_opened": False,
                "uses_benchmark_scenes": False,
                "uses_benchmark_test_vocabulary": False,
                "seed": seed,
                "surface_control": controls[seed],
                "fixed_calibration_scene_batch": {
                    "split_role": "train",
                    "scene_selection_algorithm": (
                        "lexicographically_first_complete_train_scenes_v1"
                    ),
                    "requested_scene_count": 4,
                    "scenes": [f"scene-{index}" for index in range(4)],
                    "scene_row_counts": {
                        f"scene-{index}": 2 for index in range(4)
                    },
                    "row_indices": list(range(8)),
                    "effective_row_count": 8,
                    "complete_scenes": True,
                    "augmentation": "none",
                },
                "objective_contract": {
                    "surface_objective": (
                        "token_weight*(1-cosine_summary_token)"
                        "+masked_mean_one_minus_all_view_cosine"
                        "+relation_weight*smooth_l1_descriptor_relation"
                    ),
                    "token_weight": 0.5,
                    "relation_weight": 0.1,
                    "independent_response_loss": (
                        "independent_normalized_cosine_response_smooth_l1"
                    ),
                    "scene_response_loss": (
                        "scene_wise_text_response_weighted_profile_"
                        "pairwise_gap_smooth_l1"
                    ),
                    "scene_response_objective": {
                        "name": (
                            "scene_wise_text_response_weighted_profile_"
                            "pairwise_gap_smooth_l1"
                        ),
                        "profile_loss": (
                            "scene_wise_centered_text_response_profile_"
                            "cosine_distance"
                        ),
                        "profile_weight": 0.2,
                        "ranking_loss": (
                            "scene_wise_text_response_pairwise_gap_smooth_l1"
                        ),
                        "ranking_weight": 1.0,
                        "tie_tolerance": 1e-6,
                        "pairwise_gap_normalization": (
                            "per_scene_query_teacher_response_span"
                        ),
                    },
                    "scene_tie_tolerance": 1e-6,
                    "branch_gradient_target_ratio": 0.25,
                    "combined_response_gradient_ratio_upper_bound": 0.5,
                    "upper_bound_derivation": (
                        "triangle_inequality_sum_of_two_branch_l2_budgets"
                    ),
                    "gradient_bound_scope": (
                        "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
                    ),
                    "training_batching": (
                        "shuffle_complete_scene_groups_no_partial_scenes_v1"
                    ),
                    "max_complete_scene_batch_rows": 64,
                },
                "gradient_contract": {
                    "parameter_set": (
                        "all_trainable_surface_region_summary_readout_v2_parameters"
                    ),
                    "measurement_point": (
                        "exact_seed_frozen_surface_control_state_dict"
                    ),
                    "norm": "joint_parameter_gradient_l2",
                    "epsilon": 1e-12,
                    "loss_values": {
                        "surface": 1.0,
                        "token": 1.0,
                        "descriptor": 1.0,
                        "relation": 1.0,
                        "independent_response": 1.0,
                        "scene_response": 1.0,
                        "scene_profile": 1.0,
                        "scene_ranking": 1.0,
                    },
                    "gradient_l2": {
                        "surface": surface_norm,
                        "independent_response": independent_norm,
                        "scene_response": scene_norm,
                    },
                    "branch_target_ratio": 0.25,
                    "trainable_parameter_count": 1,
                    "trainable_parameters": [{"name": "mock", "shape": [1]}],
                    "response_lambdas": lambdas_by_seed[seed],
                    "weighted_response_gradient_l2": {
                        "independent_response": 0.5,
                        "scene_response": 0.5,
                    },
                    "combined_response_gradient_l2_upper_bound": 1.0,
                    "combined_response_to_surface_upper_bound_ratio": 0.5,
                },
                "design_diagnostic": design,
                "architecture": architecture,
                "train_caches": train_caches,
                "validation_caches": validation_caches,
                "train_contract": {},
                "radio_checkpoint": {
                    "path": str(paths["radio"]),
                    "sha256": _sha256(paths["radio"]),
                },
                "fit_text_bank": fit_binding,
                "implementation": [],
        }
        _write_json(calibration_path, calibration_payload)
        _write_json(audit_path, {"seed": seed, "status": "verified"})
        calibrations.append(
            {
                "seed": seed,
                "manifest": {
                    "path": str(calibration_path),
                    "sha256": _sha256(calibration_path),
                },
                "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
                "surface_control": controls[seed],
                "response_lambdas": lambdas_by_seed[seed],
            }
        )

    provenance["random_seed_contract"] = {
        "seed": 0,
        "model_initialization": False,
        "model_initialization_source": "frozen_seed_surface_control",
        "data_order": True,
        "canonical_noise": True,
    }
    provenance["surface_control_warm_start"] = {
        **controls[0],
        "epoch": 0,
        "noninferiority_metrics": [
            "summary_token_cosine",
            "mean_descriptor_cosine",
            "all_view_descriptor_cosine",
        ],
        "noninferiority_tolerance": 0.002,
        "selection_policy": (
            "surface_control_0p002_fit_scene_robust_0p005_then_response_error_v3"
        ),
    }
    provenance["text_response_distillation"] = {
        "fit_split_only": True,
        "benchmark_vocabulary_opened": False,
        "fit_text_bank": fit_binding,
        "calibration_manifest": calibrations[0]["manifest"]["path"],
        "calibration_manifest_sha256": calibrations[0]["manifest"]["sha256"],
        "calibration_seed": 0,
        "response_lambdas": lambdas_by_seed[0],
        "response_branch_gradient_target_ratio": 0.25,
        "total_response_gradient_ratio_upper_bound": 0.5,
        "losses": [
            "independent_normalized_cosine_response_smooth_l1",
            (
                "scene_wise_text_response_weighted_profile_"
                "pairwise_gap_smooth_l1"
            ),
        ],
        "scene_response_objective": {
            "name": (
                "scene_wise_text_response_weighted_profile_"
                "pairwise_gap_smooth_l1"
            ),
            "profile_loss": (
                "scene_wise_centered_text_response_profile_cosine_distance"
            ),
            "profile_weight": 0.2,
            "ranking_loss": (
                "scene_wise_text_response_pairwise_gap_smooth_l1"
            ),
            "ranking_weight": 1.0,
            "tie_tolerance": 1e-6,
            "pairwise_gap_normalization": (
                "per_scene_query_teacher_response_span"
            ),
        },
        "complete_scene_batching": True,
        "design_diagnostic": design,
    }
    checkpoint["surface_control_checkpoint"] = controls[0]
    checkpoint["surface_control_validation"] = {
        "summary_token_cosine": 0.4,
        "mean_descriptor_cosine": 0.4,
        "all_view_descriptor_cosine": 0.4,
    }
    checkpoint["surface_control_score"] = 0.4
    checkpoint["complete_scene_batching"] = {
        "algorithm": "shuffle_complete_scene_groups_no_partial_scenes_v1",
        "row_limit": 64,
        "observed_peak_rows": 8,
        "observed_batch_count": 4,
    }
    checkpoint["training_config"] = {
        "train_caches": [record["path"] for record in train_caches],
        "validation_caches": [record["path"] for record in validation_caches],
        "fit_text_bank": fit_binding["artifact_path"],
        "fit_text_bank_manifest": fit_binding["manifest_path"],
        "calibration_manifest": calibrations[0]["manifest"]["path"],
        "run_manifest": str(paths["run_manifest"]),
        "surface_control_checkpoint": controls[0]["path"],
        "surface_control_checkpoint_sha256": controls[0]["sha256"],
        "output": str(paths["checkpoint"]),
        "hidden_dim": 2,
        "epochs": 10,
        "patience": 4,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "token_weight": 0.5,
        "relation_weight": 0.1,
        "reliability_attention_mode": "signed_logit_bias",
        "context_pooling_mode": "joint_attention_v1",
        "canonical_noise_degrees": 1.0,
        "canonical_noise_calibration": "fixed_query_free_v1",
        "seed": 0,
        "device": "cpu",
        "radio_checkpoint": str(paths["radio"]),
    }
    checkpoint.pop("untrained_baseline", None)
    checkpoint.pop("untrained_baseline_score", None)
    torch.save(checkpoint, paths["checkpoint"])

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["schema_version"] = 3
    manifest.pop("calibration_manifest")
    manifest.pop("calibration_audit")
    manifest["calibrations"] = calibrations
    manifest["gradient_design_diagnostic"] = {
        "path": str(diagnostic),
        "sha256": _sha256(diagnostic),
    }
    manifest["train_caches"] = train_caches
    manifest["fit_text_bank"] = {
        "artifact": {
            "path": str(fit_artifact),
            "sha256": _sha256(fit_artifact),
        },
        "manifest": {
            "path": str(fit_manifest),
            "sha256": _sha256(fit_manifest),
        },
    }
    manifest["training_contract"] = module._pairwise_training_contract(
        checkpoint["training_config"]
    )
    _write_json(paths["run_manifest"], manifest)

    report = {
        "output": str(paths["checkpoint"]),
        "checkpoint_sha256": _sha256(paths["checkpoint"]),
        "architecture": architecture,
        "best_epoch": 1,
        "best_selection_score": 0.6,
        "surface_control_checkpoint": controls[0],
        "surface_control_validation": checkpoint["surface_control_validation"],
        "surface_control_score": 0.4,
        "selection_score_delta": 0.6 - 0.4,
        "validation": {
            "summary_token_cosine": 0.6,
            "mean_descriptor_cosine": 0.6,
            "all_view_descriptor_cosine": 0.6,
        },
        "response_lambdas": lambdas_by_seed[0],
        "complete_scene_batching": checkpoint["complete_scene_batching"],
        "calibration_manifest": calibrations[0]["manifest"]["path"],
        "calibration_manifest_sha256": calibrations[0]["manifest"]["sha256"],
        "fit_text_bank_sha256": fit_binding["artifact_sha256"],
        "fit_query_count": fit_binding["query_count"],
        "distill_run_manifest": str(paths["run_manifest"]),
        "distill_run_manifest_sha256": _sha256(paths["run_manifest"]),
        "validation_caches": validation_caches,
        "train_scenes": 1,
        "validation_scenes": 1,
        "scene_overlap": [],
    }
    _write_json(paths["report"], report)
    _rebind_distill_manifest(paths)
    return {
        "producer_source": producer_source,
        "controls": controls,
        "calibrations": calibrations,
        "design": design,
    }


def _upgrade_to_accepted_anchor_v4(
    paths: dict[str, Path], tmp_path: Path
) -> dict[str, object]:
    upgraded = _upgrade_to_authority_schema_v3(paths, tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    control = torch.load(
        upgraded["controls"][0]["path"], map_location="cpu", weights_only=True
    )
    best_sha = module._state_dict_sha256(
        checkpoint["state_dict"], label="test best state"
    )
    anchor_state = {
        name: value.detach().clone() for name, value in checkpoint["state_dict"].items()
    }
    first_name = next(iter(anchor_state))
    anchor_state[first_name] = anchor_state[first_name] + 0.01
    anchor_sha = module._state_dict_sha256(anchor_state, label="test anchor state")

    def response_fields(*, smooth_l1: float, mae: float) -> dict[str, object]:
        return {
            "surface_selection_score": 0.4,
            "selection_score": 0.4,
            "response_selection_feasible": True,
            "text_response_smooth_l1": smooth_l1,
            "text_response_mae": mae,
            "text_response_ranking_spearman_p05": 0.5,
            "text_response_ranking_spearman_mean": 0.6,
            "text_response_profile_cosine_p05": 0.7,
            "text_response_profile_cosine_mean": 0.8,
            "text_response_top_decile_overlap_p05": 0.6,
            "text_response_top_decile_overlap_mean": 0.7,
            "text_support_top1_agreement": 0.75,
            "descriptor_relation_smooth_l1": 0.02,
        }

    control_row = {
        "epoch": 0,
        "initialization": "frozen_surface_control_checkpoint",
        "state_machine_role": "frozen_control_initial_anchor",
        **response_fields(smooth_l1=0.02, mae=0.03),
        "accepted": True,
        "rejected": False,
        "anchor_epoch_after_proposal": 0,
        "anchor_state_dict_sha256_after_proposal": best_sha,
        "best_updated": True,
        "best_epoch_after_proposal": 0,
        "best_state_dict_sha256_after_proposal": best_sha,
        "patience_stale_after_proposal": 0,
        "patience_stop_after_proposal": False,
    }
    control_row["history_hash_chain"] = {
        "algorithm": module._HISTORY_HASH_CHAIN_ALGORITHM,
        "previous_sha256": None,
        "sha256": module._history_chain_digest(control_row, None),
    }
    losses = {
        field: 0.1 + index * 0.01
        for index, field in enumerate(module._PROPOSAL_LOSS_FIELDS)
    }
    proposal_row = {
        "epoch": 1,
        "state_machine_role": "fixed_micro_ray_trial",
        "proposal": {
            "index": 1,
            "source_anchor_epoch": 0,
            "anchor_state_dict_sha256": best_sha,
            "raw_state_dict_sha256": "a" * 64,
            "trial_state_dict_sha256": anchor_sha,
            "alpha_numerator": 1,
            "alpha_denominator": 2048,
            "optimizer_state_reset": True,
            "validation_evaluations": 1,
            "backtracking": "none_fixed_alpha_single_trial",
            "persistent_generator": "advanced_never_rolled_back",
        },
        "loss_measurement_state": "raw_proposal_before_micro_projection",
        "proposal_losses": losses,
        **{
            module._PROPOSAL_LOSS_FLAT_MIRROR[field]: value
            for field, value in losses.items()
        },
        **response_fields(smooth_l1=0.03, mae=0.04),
        "selection_score": -1.0,
        "accepted": True,
        "rejected": False,
        "anchor_epoch_after_proposal": 1,
        "anchor_state_dict_sha256_after_proposal": anchor_sha,
        "best_updated": False,
        "best_epoch_after_proposal": 0,
        "best_state_dict_sha256_after_proposal": best_sha,
        "patience_stale_after_proposal": 1,
        "patience_stop_after_proposal": False,
    }
    previous_sha = control_row["history_hash_chain"]["sha256"]
    proposal_row["history_hash_chain"] = {
        "algorithm": module._HISTORY_HASH_CHAIN_ALGORITHM,
        "previous_sha256": previous_sha,
        "sha256": module._history_chain_digest(proposal_row, previous_sha),
    }
    history = [control_row, proposal_row]
    checkpoint.update(
        {
            "history": history,
            "best_epoch": 0,
            "best_selection_score": 0.4,
            "best_state_dict_sha256": best_sha,
            "proposal_state_machine": dict(module._PROPOSAL_STATE_MACHINE),
            "accepted_anchor": {
                "epoch": 1,
                "state_dict_sha256": anchor_sha,
                "accepted_proposal_count": 1,
                "rejected_proposal_count": 0,
            },
            "history_hash_chain_sha256": proposal_row["history_hash_chain"][
                "sha256"
            ],
        }
    )
    checkpoint["provenance"]["proposal_state_machine"] = dict(
        module._PROPOSAL_STATE_MACHINE
    )
    checkpoint["provenance"]["surface_control_warm_start"][
        "selection_policy"
    ] = module._ACCEPTED_ANCHOR_EPOCH_SELECTION
    checkpoint["training_config"]["proposal_state_machine"] = dict(
        module._PROPOSAL_STATE_MACHINE
    )
    torch.save(checkpoint, paths["checkpoint"])

    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["training_contract"] = module._pairwise_training_contract(
        checkpoint["training_config"], accepted_anchor_v4=True
    )
    _write_json(paths["run_manifest"], manifest)

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report.update(
        {
            "best_epoch": 0,
            "best_selection_score": 0.4,
            "best_state_dict_sha256": best_sha,
            "proposal_state_machine": dict(module._PROPOSAL_STATE_MACHINE),
            "accepted_anchor": checkpoint["accepted_anchor"],
            "history_hash_chain_sha256": checkpoint[
                "history_hash_chain_sha256"
            ],
            "selection_score_delta": 0.0,
            "validation": dict(checkpoint["surface_control_validation"]),
        }
    )
    _write_json(paths["report"], report)
    _rebind_distill_manifest(paths)
    return {**upgraded, "anchor_state": anchor_state, "best_state_sha": best_sha}


def test_materializer_binds_exact_checkpoint_validation_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = module.materialize(_args(paths, tmp_path / "descriptors.pt"))
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)

    provenance = payload["provenance"]
    assert provenance["readout_checkpoint_sha256"] == _sha256(paths["checkpoint"])
    assert provenance["readout_report_sha256"] == _sha256(paths["report"])
    assert provenance["readout_binding_authority"] == {
        "type": "embedded_distill_run_manifest",
        "path": str(paths["run_manifest"]),
        "sha256": _sha256(paths["run_manifest"]),
        "candidate": "context_c1024_geometric",
    }
    assert provenance["validation_caches"][0]["sha256"] == _sha256(paths["cache"])
    assert provenance["validation_scenes"] == ["scene-validation"]
    assert provenance["validation_split_sha256"] == "d" * 64


def test_materializer_accepts_authority_schema_v2_frozen_producer_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    producer_source = _upgrade_to_authority_schema_v2(paths, tmp_path)

    result = module.materialize(_args(paths, tmp_path / "descriptors.pt"))
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)

    assert _sha256(producer_source) != _sha256(Path(module.__file__).resolve())
    assert payload["provenance"]["readout_binding_authority"] == {
        "type": "embedded_distill_run_manifest",
        "path": str(paths["run_manifest"]),
        "sha256": _sha256(paths["run_manifest"]),
        "candidate": "context_c1024_geometric",
    }


def test_materializer_accepts_authority_schema_v3_per_seed_warm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    upgraded = _upgrade_to_authority_schema_v3(paths, tmp_path)

    result = module.materialize(_args(paths, tmp_path / "descriptors-v3.pt"))
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)

    assert _sha256(upgraded["producer_source"]) != _sha256(
        Path(module.__file__).resolve()
    )
    assert payload["provenance"]["readout_binding_authority"] == {
        "type": "embedded_distill_run_manifest",
        "path": str(paths["run_manifest"]),
        "sha256": _sha256(paths["run_manifest"]),
        "candidate": "context_c1024_geometric",
    }


def test_materializer_accepts_v4_best_state_with_newer_nonbest_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_accepted_anchor_v4(paths, tmp_path)

    result = module.materialize(_args(paths, tmp_path / "descriptors-v4.pt"))

    assert Path(result["output"]).is_file()
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    assert checkpoint["accepted_anchor"]["epoch"] == 1
    assert checkpoint["best_epoch"] == 0
    assert (
        module._state_dict_sha256(checkpoint["state_dict"], label="published")
        == checkpoint["best_state_dict_sha256"]
    )
    assert (
        checkpoint["accepted_anchor"]["state_dict_sha256"]
        != checkpoint["best_state_dict_sha256"]
    )


def test_materializer_accepts_v4_absolute_cache_globs_through_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_accepted_anchor_v4(paths, tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    cache_alias = tmp_path / "cache-alias"
    cache_alias.symlink_to(tmp_path, target_is_directory=True)
    train_path = Path(checkpoint["training_config"]["train_caches"][0])
    validation_path = Path(
        checkpoint["training_config"]["validation_caches"][0]
    )
    checkpoint["training_config"]["train_caches"] = str(
        cache_alias / f"{train_path.stem}*.pt"
    )
    checkpoint["training_config"]["validation_caches"] = str(
        cache_alias / f"{validation_path.stem}*.pt"
    )
    for field in (
        "fit_text_bank",
        "fit_text_bank_manifest",
        "calibration_manifest",
        "run_manifest",
        "surface_control_checkpoint",
        "output",
        "radio_checkpoint",
    ):
        configured_path = Path(checkpoint["training_config"][field])
        if configured_path.parent == tmp_path:
            aliased = cache_alias / configured_path.name
            assert module._configured_path_text(
                aliased, label=f"test {field}"
            ) == str(configured_path)
            checkpoint["training_config"][field] = str(aliased)
    torch.save(checkpoint, paths["checkpoint"])
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = _sha256(paths["checkpoint"])
    _write_json(paths["report"], report)
    _rebind_distill_manifest(paths)

    result = module.materialize(
        _args(paths, tmp_path / "descriptors-v4-globs.pt")
    )

    assert Path(result["output"]).is_file()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("last_anchor_published", "does not publish the best state"),
        ("last_anchor_declared_best", "does not publish the best state"),
        ("loss_source", "raw-proposal loss accounting differs"),
        ("history_chain", "history hash chain differs"),
    ],
)
def test_materializer_rejects_v4_raw_last_anchor_and_provenance_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    upgraded = _upgrade_to_accepted_anchor_v4(paths, tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    if mutation == "last_anchor_published":
        checkpoint["state_dict"] = upgraded["anchor_state"]
    elif mutation == "last_anchor_declared_best":
        checkpoint["best_state_dict_sha256"] = checkpoint["accepted_anchor"][
            "state_dict_sha256"
        ]
    elif mutation == "loss_source":
        checkpoint["history"][1]["loss_measurement_state"] = (
            "trial_after_micro_projection"
        )
        previous = checkpoint["history"][0]["history_hash_chain"]["sha256"]
        checkpoint["history"][1]["history_hash_chain"]["sha256"] = (
            module._history_chain_digest(checkpoint["history"][1], previous)
        )
        checkpoint["history_hash_chain_sha256"] = checkpoint["history"][1][
            "history_hash_chain"
        ]["sha256"]
    else:
        checkpoint["history"][1]["history_hash_chain"]["sha256"] = "f" * 64
        checkpoint["history_hash_chain_sha256"] = "f" * 64
    torch.save(checkpoint, paths["checkpoint"])
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = _sha256(paths["checkpoint"])
    if mutation == "last_anchor_declared_best":
        report["best_state_dict_sha256"] = checkpoint[
            "best_state_dict_sha256"
        ]
    elif mutation in {"loss_source", "history_chain"}:
        report["history_hash_chain_sha256"] = checkpoint[
            "history_hash_chain_sha256"
        ]
    _write_json(paths["report"], report)

    with pytest.raises(ValueError, match=match):
        module.materialize(_args(paths, tmp_path / "descriptors-v4.pt"))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("calibration_swap", "calibration immutable binding differs"),
        ("other_calibration_swap", "calibration immutable binding differs"),
        ("surface_control_swap", "calibration/control differs"),
        ("design_mapping", "gradient design diagnostic differs"),
        ("checkpoint_objective", "text-response provenance differs"),
        ("manifest_training_contract", "training contract differs"),
        ("calibration_objective", "pairwise calibration objective differs"),
        ("sidecar_batching", "report fields differ"),
    ],
)
def test_materializer_rejects_authority_schema_v3_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    upgraded = _upgrade_to_authority_schema_v3(paths, tmp_path)
    if mutation in {
        "calibration_swap",
        "other_calibration_swap",
        "surface_control_swap",
    }:
        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
        field = (
            "manifest"
            if mutation in {"calibration_swap", "other_calibration_swap"}
            else "surface_control"
        )
        left, right = (1, 2) if mutation == "other_calibration_swap" else (0, 1)
        manifest["calibrations"][left][field], manifest["calibrations"][right][field] = (
            manifest["calibrations"][right][field],
            manifest["calibrations"][left][field],
        )
        _write_json(paths["run_manifest"], manifest)
        _rebind_distill_manifest(paths)
    elif mutation in {"design_mapping", "checkpoint_objective"}:
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True
        )
        distillation = checkpoint["provenance"]["text_response_distillation"]
        if mutation == "design_mapping":
            distillation["design_diagnostic"] = {
                **upgraded["design"],
                "role": "tampered-role",
            }
        else:
            distillation["scene_response_objective"] = {
                **distillation["scene_response_objective"],
                "profile_weight": 0.19,
            }
        torch.save(checkpoint, paths["checkpoint"])
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = _sha256(paths["checkpoint"])
        _write_json(paths["report"], report)
    elif mutation == "manifest_training_contract":
        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
        manifest["training_contract"]["scene_response_objective"][
            "profile_weight"
        ] = 0.19
        _write_json(paths["run_manifest"], manifest)
        _rebind_distill_manifest(paths)
    elif mutation == "calibration_objective":
        calibration_path = Path(
            upgraded["calibrations"][0]["manifest"]["path"]
        )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration["objective_contract"]["scene_response_objective"][
            "profile_weight"
        ] = 0.19
        _write_json(calibration_path, calibration)
        calibration_sha = _sha256(calibration_path)
        manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
        manifest["calibrations"][0]["manifest"]["sha256"] = calibration_sha
        _write_json(paths["run_manifest"], manifest)
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True
        )
        checkpoint["provenance"]["text_response_distillation"][
            "calibration_manifest_sha256"
        ] = calibration_sha
        torch.save(checkpoint, paths["checkpoint"])
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        report["calibration_manifest_sha256"] = calibration_sha
        _write_json(paths["report"], report)
        _rebind_distill_manifest(paths)
    else:
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        report.pop("complete_scene_batching")
        _write_json(paths["report"], report)

    with pytest.raises(ValueError, match=match):
        module.materialize(_args(paths, tmp_path / "descriptors-v3.pt"))


def test_materializer_rejects_unregistered_schema_v3_legacy_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v3(paths, tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    distillation = checkpoint["provenance"]["text_response_distillation"]
    distillation["losses"] = [
        "independent_normalized_cosine_response_smooth_l1",
        "scene_wise_text_response_profile_ranking",
    ]
    distillation.pop("scene_response_objective")
    checkpoint["provenance"]["surface_control_warm_start"][
        "selection_policy"
    ] = (
        "surface_control_feasible_0p002_then_fit_support_"
        "response_relation_surface_v2"
    )
    torch.save(checkpoint, paths["checkpoint"])
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = _sha256(paths["checkpoint"])
    _write_json(paths["report"], report)

    with pytest.raises(ValueError, match="text-response provenance differs"):
        module.materialize(_args(paths, tmp_path / "descriptors-v3.pt"))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_materializer_rejects_authority_schema_v2_output_field_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v2(paths, tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest["outputs"][0].pop("guard_receipt")
    else:
        manifest["outputs"][0]["unexpected"] = str(tmp_path / "unexpected")
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="output index fields differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_authority_schema_v2_producer_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v2(paths, tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    relative = "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
    manifest["implementation_sources"][relative] = "0" * 64
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="producer materializer implementation"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_authority_schema_v2_symlinked_snapshot_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    producer_source = _upgrade_to_authority_schema_v2(paths, tmp_path)
    snapshot_link = tmp_path / "snapshot-link"
    snapshot_link.symlink_to(producer_source.parents[2], target_is_directory=True)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["authority_contract"]["source_snapshot_root"] = str(snapshot_link)
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="canonical non-symlink path"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_authority_schema_v2_top_level_field_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _upgrade_to_authority_schema_v2(paths, tmp_path)
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    _write_json(paths["run_manifest"], manifest)
    _rebind_distill_manifest(paths)

    with pytest.raises(ValueError, match="run-manifest fields differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_same_contract_different_scene_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    substitute = tmp_path / "substitute_validation.pt"
    _cache(substitute, scene="scene-substitute", radio_sha=_sha256(paths["radio"]))

    with pytest.raises(ValueError, match="provided validation caches differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt", substitute))


def test_materializer_rejects_in_place_validation_cache_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload = torch.load(paths["cache"], map_location="cpu", weights_only=True)
    payload["radio_features"][0, 0, 0] = 9.0
    torch.save(payload, paths["cache"])

    with pytest.raises(ValueError, match="provided validation caches differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_checkpoint_without_cache_sha_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    checkpoint["provenance"]["validation"].pop("cache_bindings")
    torch.save(checkpoint, paths["checkpoint"])

    with pytest.raises(ValueError, match="provided validation caches differ"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_rejects_checkpoint_report_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = "0" * 64
    _write_json(paths["report"], report)

    with pytest.raises(ValueError, match="checkpoint_sha256 binding differs"):
        module.materialize(_args(paths, tmp_path / "descriptors.pt"))


def test_materializer_refuses_to_overwrite_descriptor_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "descriptors.pt"
    module.materialize(_args(paths, output))
    with pytest.raises(FileExistsError, match="already exists"):
        module.materialize(_args(paths, output))


def _make_legacy_bundle(paths: dict[str, Path], tmp_path: Path) -> Path:
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True
    )
    checkpoint["provenance"]["validation"].pop("cache_bindings")
    checkpoint["provenance"].pop("text_response_distillation")
    checkpoint["provenance"].pop("distill_run_manifest")
    torch.save(checkpoint, paths["checkpoint"])
    baseline = checkpoint["untrained_baseline"]
    report = {
        "output": str(paths["checkpoint"]),
        "checkpoint_sha256": _sha256(paths["checkpoint"]),
        "architecture": checkpoint["architecture"],
        "best_epoch": checkpoint["best_epoch"],
        "best_selection_score": checkpoint["best_selection_score"],
        "untrained_baseline": baseline,
        "selection_score_delta": checkpoint["best_selection_score"]
        - 0.5
        * (
            baseline["mean_descriptor_cosine"]
            + baseline["all_view_descriptor_cosine"]
        ),
        "validation": {
            "summary_token_cosine": 0.6,
            "mean_descriptor_cosine": 0.6,
            "all_view_descriptor_cosine": 0.6,
        },
        "train_scenes": 1,
        "validation_scenes": 1,
        "scene_overlap": [],
    }
    _write_json(paths["report"], report)
    cache_sidecar = paths["cache"].with_suffix(".pt.json")
    _write_json(cache_sidecar, {"output": str(paths["cache"])})
    anchor = paths["run_manifest"]
    anchor_binding = {"path": str(anchor), "sha256": _sha256(anchor)}
    candidate = "context_c1024_geometric"
    readout = {
        "candidate": candidate,
        "seed": 0,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": _sha256(paths["checkpoint"]),
        "sidecar": str(paths["report"]),
        "sidecar_sha256": _sha256(paths["report"]),
    }
    selected_readouts = [
        readout,
        {
            **readout,
            "seed": 1,
            "checkpoint": str(tmp_path / "seed1.pt"),
            "checkpoint_sha256": "4" * 64,
            "sidecar": str(tmp_path / "seed1.pt.json"),
            "sidecar_sha256": "5" * 64,
        },
        {
            **readout,
            "seed": 2,
            "checkpoint": str(tmp_path / "seed2.pt"),
            "checkpoint_sha256": "6" * 64,
            "sidecar": str(tmp_path / "seed2.pt.json"),
            "sidecar_sha256": "7" * 64,
        },
    ]
    bundle = {
        "schema_version": 1,
        "artifact_type": "surface_region_query_free_three_seed_bundle",
        "status": "query_free_three_seed_bundle_frozen_benchmark_gate_closed",
        "selected_candidate": candidate,
        "seed_selection_policy": "all_required_seeds_no_single_seed_selection",
        "required_seeds": [0, 1, 2],
        "selected_readouts": selected_readouts,
        "benchmark_gate": {
            "status": "closed_not_evaluated",
            "main_result_eligible": False,
        },
        "bindings": {
            "finalizer": anchor_binding,
            "run_manifest": anchor_binding,
            "cache_pairing": anchor_binding,
            "query_free_screen": anchor_binding,
            "screen_completion": anchor_binding,
            "all_compared_readouts": selected_readouts,
            "caches": [
                {
                    "candidate": candidate,
                    "role": "validation",
                    "shard": 0,
                    "path": str(paths["cache"]),
                    "sha256": _sha256(paths["cache"]),
                    "sidecar": str(cache_sidecar),
                    "sidecar_sha256": _sha256(cache_sidecar),
                }
            ],
        },
    }
    bundle_path = tmp_path / "query_free_promotion_bundle.json"
    _write_json(bundle_path, bundle)
    completion = {
        "schema_version": 1,
        "artifact_type": "surface_region_query_free_promotion_completion",
        "status": "complete_benchmark_gate_closed",
        "promotion_manifest": str(bundle_path),
        "promotion_manifest_sha256": _sha256(bundle_path),
        "selected_candidate": candidate,
        "required_seeds": [0, 1, 2],
        "benchmark_gate_status": "closed_not_evaluated",
        "main_result_eligible": False,
    }
    _write_json(tmp_path / "query_free_promotion.complete.json", completion)
    return bundle_path


def test_materializer_allows_legacy_selected_seed_only_through_promotion_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle = _make_legacy_bundle(paths, tmp_path)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle)

    result = module.materialize(args)
    payload = torch.load(result["output"], map_location="cpu", weights_only=True)
    authority = payload["provenance"]["readout_binding_authority"]
    assert authority["type"] == "query_free_promotion_bundle"
    assert authority["sha256"] == _sha256(bundle)
    assert authority["candidate"] == "context_c1024_geometric"


def test_materializer_rejects_legacy_bundle_wrong_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle_path = _make_legacy_bundle(paths, tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["selected_candidate"] = "control_c256_geometric"
    _write_json(bundle_path, bundle)
    completion_path = tmp_path / "query_free_promotion.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["selected_candidate"] = "control_c256_geometric"
    completion["promotion_manifest_sha256"] = _sha256(bundle_path)
    _write_json(completion_path, completion)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle_path)

    with pytest.raises(ValueError, match="selected candidate/seed"):
        module.materialize(args)


def test_materializer_rejects_legacy_bundle_wrong_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle_path = _make_legacy_bundle(paths, tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["bindings"]["all_compared_readouts"][0]["seed"] = 1
    _write_json(bundle_path, bundle)
    completion_path = tmp_path / "query_free_promotion.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["promotion_manifest_sha256"] = _sha256(bundle_path)
    _write_json(completion_path, completion)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle_path)

    with pytest.raises(ValueError, match="selected candidate/seed"):
        module.materialize(args)


def test_materializer_rejects_legacy_completion_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bundle_path = _make_legacy_bundle(paths, tmp_path)
    completion_path = tmp_path / "query_free_promotion.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["promotion_manifest_sha256"] = "0" * 64
    _write_json(completion_path, completion)
    args = _args(paths, tmp_path / "legacy-descriptors.pt")
    args.readout_binding_manifest = str(bundle_path)

    with pytest.raises(ValueError, match="does not bind"):
        module.materialize(args)


def test_materializer_forbids_distilled_checkpoint_legacy_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    args = _args(paths, tmp_path / "descriptors.pt")
    args.readout_binding_manifest = str(paths["run_manifest"])

    with pytest.raises(ValueError, match="cannot use the legacy"):
        module.materialize(args)
