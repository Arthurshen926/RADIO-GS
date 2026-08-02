#!/usr/bin/env python3
"""Freeze dev-selected, audit-confirmed Surface text-response promotion.

This CPU-only companion starts from the immutable query-free SurfaceRegion
three-seed bundle.  It pairs that selected candidate's original checkpoints
against exactly seeds 0/1/2 of the response-distillation treatment, validates
the existing target-blind text-response gate, and adds a per-seed Surface
non-inferiority gate.  ``dev`` is the only selection split.  ``audit`` is
accepted only after a frozen dev promotion and can confirm or reject it once;
it can never change the method or thresholds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import (
    REPORT_ARTIFACT_TYPE,
    REPORT_SCHEMA_VERSION,
    _validate_report,
    aggregate_paired_seed_gate,
    evaluate_response_fidelity,
    selection_contract_for_bank_family,
    tensor_sha256,
)
from radio_gs.scripts import finalize_surface_region_query_free_promotion as surface_finalizer
from radio_gs.scripts.finalize_gpu_guard_receipt import validate_receipt
from radio_gs.scripts.surface_region_run_guard import discover_repo_python_closure
from radio_gs.scripts.eval_text_response_fidelity_gate import (
    load_descriptor_pair,
    load_text_embedding_bank,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    load_json_object,
    load_sha_bound_project_checkpoint_mapping,
    load_surface_region_summary_readout_v2,
    sha256_file,
    stable_descriptor_load,
)


SCHEMA_VERSION = 1
PLAN_ARTIFACT_TYPE = "surface_text_response_promotion_plan"
STAGE_ARTIFACT_TYPE = "surface_text_response_promotion_stage"
COMPLETION_ARTIFACT_TYPE = "surface_text_response_promotion_completion"
REQUIRED_SEEDS = (0, 1, 2)
SURFACE_NONINFERIORITY_TOLERANCE = 0.002
MINIMUM_IMPROVED_SEEDS = 2
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260731
QUALITY_NONINFERIORITY_TOLERANCE = 0.005
SURFACE_METRICS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
    "relation_fidelity",
)
COMMON_TRAINING_FIELDS = (
    "hidden_dim",
    "epochs",
    "patience",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "token_weight",
    "relation_weight",
    "reliability_attention_mode",
    "context_pooling_mode",
    "canonical_noise_degrees",
    "canonical_noise_calibration",
    "seed",
)
FIT_TEXT_BANK_FIELDS = {
    "artifact_path",
    "artifact_sha256",
    "manifest_path",
    "manifest_sha256",
    "split",
    "query_count",
    "split_synset_tab_query_lf_sha256",
    "ordered_records_sha256",
    "vocabulary_sha256",
    "vocabulary_manifest_sha256",
    "embedding_semantic_sha256",
    "embedding_tensor_sha256",
    "text_encoder_snapshot_files_sha256",
}
DISTILL_IMPLEMENTATION_SOURCES = {
    "radio_gs/scripts/run_surface_region_text_response_distill.sh",
    "radio_gs/scripts/train_surface_region_text_response_distill.py",
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py",
    "radio_gs/scripts/train_surface_region_summary_readout.py",
    "radio_gs/losses/direct_point_query_logit_distill_loss.py",
    "radio_gs/interfaces/surface_region_summary.py",
    "radio_gs/models/siglip_projection.py",
    "radio_gs/scripts/build_target_blind_siglip2_embedding_artifact.py",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
}
AUTHORITY_DISTILL_IMPLEMENTATION_SOURCES = {
    *DISTILL_IMPLEMENTATION_SOURCES,
    "radio_gs/scripts/surface_text_response_distill_authority.py",
    "radio_gs/scripts/finalize_gpu_guard_receipt.py",
    "radio_gs/scripts/finalize_surface_text_response_promotion.py",
    "radio_gs/scripts/run_repo_python.sh",
    "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
    "radio_gs/utils/immutable_artifacts.py",
    "radio_gs/scripts/surface_attention_pooling_screen.py",
}
LEGACY_DISTILL_EPOCH_SELECTION = (
    "surface_control_feasible_0p002_then_fit_support_response_relation_surface_v2"
)
DISTILL_EPOCH_SELECTION = (
    "surface_control_0p002_fit_scene_robust_0p005_then_response_error_v3"
)
ACCEPTED_ANCHOR_DISTILL_EPOCH_SELECTION = (
    "surface_control_0p002_fit_scene_robust_0p005_accepted_anchor_"
    "fixed_1over2048_then_response_error_v4"
)
HISTORY_HASH_CHAIN_ALGORITHM = (
    "sha256_canonical_json_previous_plus_record_without_selection_score_v1"
)
PROPOSAL_STATE_MACHINE = {
    "name": "accepted_anchor_fixed_micro_ray_fresh_adamw_v1",
    "proposal_source": "current_accepted_anchor",
    "proposal_optimizer": "fresh_adamw_complete_epoch",
    "alpha_numerator": 1,
    "alpha_denominator": 2048,
    "trial_interpolation": "anchor+alpha*(raw-anchor)",
    "validation_evaluations_per_proposal": 1,
    "acceptance": "response_selection_feasible_is_exactly_true",
    "feasible_nonbest_action": "accept_as_next_anchor",
    "infeasible_action": "restore_exact_preproposal_anchor",
    "optimizer_moments": "reset_before_every_proposal",
    "best_selection": "existing_v3_robust_lexicographic_rank_control_and_trials",
    "patience": "consecutive_proposals_without_global_best_update",
    "persistent_generator": "advanced_across_proposals_never_rolled_back",
    "backtracking": "none_fixed_alpha_single_trial",
    "proposal_loss_accounting": {
        "measurement_state": "raw_proposal_before_micro_projection",
        "fields": [
            "total",
            "token",
            "descriptor",
            "relation",
            "independent_response",
            "scene_response",
            "scene_profile",
            "scene_ranking",
        ],
        "legacy_flat_mirror": {
            "total": "loss",
            "token": "token_loss",
            "descriptor": "descriptor_loss",
            "relation": "relation_loss",
            "independent_response": "independent_response_loss",
            "scene_response": "scene_response_loss",
            "scene_profile": "scene_profile_loss",
            "scene_ranking": "scene_ranking_loss",
        },
    },
}
PROPOSAL_LOSS_FIELDS = tuple(
    PROPOSAL_STATE_MACHINE["proposal_loss_accounting"]["fields"]
)
PROPOSAL_LOSS_FLAT_MIRROR = dict(
    PROPOSAL_STATE_MACHINE["proposal_loss_accounting"]["legacy_flat_mirror"]
)
FIT_RESPONSE_NONINFERIORITY_TOLERANCE = 0.005
DISTILL_SURFACE_CONTROL_METRICS = (
    "summary_token_cosine",
    "mean_descriptor_cosine",
    "all_view_descriptor_cosine",
)
FIT_RESPONSE_QUALITY_METRICS = (
    "text_response_profile_cosine_mean",
    "text_response_profile_cosine_p05",
    "text_response_ranking_spearman_mean",
    "text_response_ranking_spearman_p05",
    "text_response_top_decile_overlap_mean",
    "text_response_top_decile_overlap_p05",
)
FIT_RESPONSE_SCENE_QUALITY_METRICS = (
    "profile_cosine_mean",
    "profile_cosine_p05",
    "ranking_spearman_mean",
    "ranking_spearman_p05",
    "top_decile_overlap_mean",
    "top_decile_overlap_p05",
)
FIT_RESPONSE_SCENE_ERROR_METRICS = ("smooth_l1", "mae")
PAIRWISE_CALIBRATION_ALGORITHM_VERSION = (
    "per-seed-surface-warmstart-dual-response-pairwise-gradient-budget-v3"
)
LEGACY_BRIER_CALIBRATION_ALGORITHM_VERSION = (
    "per-seed-surface-warmstart-dual-response-gradient-budget-v2"
)
SCENE_RESPONSE_LOSS = (
    "scene_wise_text_response_weighted_profile_pairwise_gap_smooth_l1"
)
SCENE_PROFILE_LOSS = "scene_wise_centered_text_response_profile_cosine_distance"
SCENE_PAIRWISE_GAP_LOSS = "scene_wise_text_response_pairwise_gap_smooth_l1"
SCENE_RESPONSE_OBJECTIVE = {
    "name": SCENE_RESPONSE_LOSS,
    "profile_loss": SCENE_PROFILE_LOSS,
    "profile_weight": 0.2,
    "ranking_loss": SCENE_PAIRWISE_GAP_LOSS,
    "ranking_weight": 1.0,
    "tie_tolerance": 1e-6,
    "pairwise_gap_normalization": "per_scene_query_teacher_response_span",
}
LEGACY_BRIER_RESPONSE_LOSS = "scene_wise_text_response_profile_ranking"


def _calibration_objective_contract(
    *,
    response_protocol: str,
    token_weight: float,
    relation_weight: float,
) -> dict[str, Any]:
    common = {
        "surface_objective": (
            "token_weight*(1-cosine_summary_token)"
            "+masked_mean_one_minus_all_view_cosine"
            "+relation_weight*smooth_l1_descriptor_relation"
        ),
        "token_weight": float(token_weight),
        "relation_weight": float(relation_weight),
        "independent_response_loss": (
            "independent_normalized_cosine_response_smooth_l1"
        ),
        "branch_gradient_target_ratio": 0.25,
        "combined_response_gradient_ratio_upper_bound": 0.5,
        "upper_bound_derivation": (
            "triangle_inequality_sum_of_two_branch_l2_budgets"
        ),
        "gradient_bound_scope": (
            "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
        ),
        "training_batching": "shuffle_complete_scene_groups_no_partial_scenes_v1",
        "max_complete_scene_batch_rows": 64,
    }
    if response_protocol in {
        "pairwise_selection_v3",
        "accepted_anchor_selection_v4",
    }:
        return {
            **common,
            "scene_response_loss": SCENE_RESPONSE_LOSS,
            "scene_response_objective": dict(SCENE_RESPONSE_OBJECTIVE),
            "scene_tie_tolerance": 1e-6,
        }
    _require(
        response_protocol == "legacy_brier_selection_v2",
        "unsupported calibration response protocol",
    )
    return {
        **common,
        "scene_response_loss": LEGACY_BRIER_RESPONSE_LOSS,
        "scene_profile_weight": 1.0,
        "scene_ranking_weight": 1.0,
        "scene_ranking_temperature": 0.1,
        "scene_tie_tolerance": 1e-6,
    }
DISTILL_RUNTIME_PYTHON_ENTRYPOINTS = (
    "radio_gs/scripts/surface_text_response_distill_authority.py",
    "radio_gs/scripts/train_surface_region_text_response_distill.py",
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py",
    "radio_gs/scripts/finalize_gpu_guard_receipt.py",
    "radio_gs/scripts/finalize_surface_text_response_promotion.py",
)
DISTILL_RUNTIME_SHELL_SOURCES = (
    "radio_gs/scripts/run_surface_region_text_response_distill.sh",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
    "radio_gs/scripts/run_repo_python.sh",
)
DISTILL_RUNTIME_PACKAGES = (
    "numpy",
    "Pillow",
    "scipy",
    "timm",
    "torch",
    "torchvision",
)
DISTILL_RUNTIME_MODULES = (
    "radio_gs",
    "radio_gs.interfaces.surface_region_summary",
    "radio_gs.losses.direct_point_query_logit_distill_loss",
    "radio_gs.models.siglip_projection",
    "radio_gs.scripts.train_surface_region_summary_readout",
    "radio_gs.scripts.surface_attention_pooling_screen",
    "radio_gs.scripts.train_surface_region_text_response_distill",
    "radio_gs.utils.immutable_artifacts",
)
DISTILL_RUNTIME_ENVIRONMENT_KEYS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "RADIO_GS_REPO_ROOT",
    "RADIO_GS_DRIVER_LIBRARY",
    "RADIO_GS_LD_LIBRARY_PATH",
    "RADIO_GS_SITE_PACKAGES",
)
FORMAL_RECORDED_SOURCE_CLOSURE_AUTHORITIES = {
    "497cc23d08db7500d78c5741de48c132c78e493a6fd205e34489668725f16615": {
        "manifest_path": (
            "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
            "surface_text_response_warmstart_dualprofile_joint_c1024_gpu1only_v3/"
            "run_manifest.json"
        ),
        "source_root": "/root/RADIO-GS",
        "runtime_closure_digest": (
            "56736482529dc7f7fc34c6b29cba2373c59993820989c4f41bca3f8a1f6767e3"
        ),
        "response_protocol": "legacy_brier_selection_v2",
    }
}
DISTILL_KERNEL_FAULT_PATTERN = re.compile(
    r"(?:\bNVRM\b.*\bXid\b|fallen off|PCIe.*(?:error|fatal)|GPU.*lost PCIe)",
    re.IGNORECASE,
)
IMPLEMENTATION_SOURCES = (
    "radio_gs/scripts/run_surface_text_response_promotion.sh",
    "radio_gs/scripts/finalize_surface_text_response_promotion.py",
    "radio_gs/scripts/finalize_surface_region_query_free_promotion.py",
    "radio_gs/scripts/surface_text_response_distill_authority.py",
    "radio_gs/scripts/surface_attention_pooling_screen.py",
    "radio_gs/scripts/train_surface_region_text_response_distill.py",
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py",
    "radio_gs/scripts/eval_text_response_fidelity_gate.py",
    "radio_gs/evaluation/text_response_fidelity.py",
    "radio_gs/interfaces/surface_region_summary.py",
)

_SHA256_CACHE: dict[tuple[str, tuple[int, ...]], str] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_identity(path: Path) -> tuple[Path, tuple[int, ...]]:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    source = raw.parent.resolve(strict=True) / raw.name
    info = os.stat(source, follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode), f"hashed artifact is not regular: {source}")
    return source, (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _sha256(path: Path) -> str:
    """Hash an unchanged file once per process and stable file identity.

    ``sha256_file`` already performs a fail-closed double read.  Promotion
    validation asks for the same 1.6-GB RADIO digest many times, so subsequent
    checks reuse that result only while device/inode/mode/link-count/size and
    nanosecond mtime/ctime are all unchanged.
    """

    source, fingerprint = _sha256_identity(path)
    key = (str(source), fingerprint)
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(source)
    source_after, fingerprint_after = _sha256_identity(source)
    _require(
        source_after == source and fingerprint_after == fingerprint,
        f"hashed artifact changed while being read: {source}",
    )
    _SHA256_CACHE[key] = digest
    return digest


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _json_object(path: Path) -> dict[str, Any]:
    value, _, _ = load_json_object(path, label="text-response JSON artifact")
    return value


def _torch_mapping(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    value, _, _ = load_sha_bound_project_checkpoint_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="text-response torch artifact",
    )
    return value


def _finite(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def _close(left: object, right: object, tolerance: float = 1e-7) -> bool:
    return math.isclose(
        _finite(left, "left comparison"),
        _finite(right, "right comparison"),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _response_protocol_for_manifest(
    manifest_path: Path,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
) -> str:
    """Classify accepted-anchor-v4, pairwise-v3, or registered Brier authority."""

    schema = manifest.get("schema_version")
    selection = manifest.get("training_contract", {}).get("epoch_selection")
    if schema == 1 and selection is None:
        return "legacy_single_response_v1"
    if selection == ACCEPTED_ANCHOR_DISTILL_EPOCH_SELECTION:
        _require(
            schema == 3
            and manifest.get("training_contract", {}).get(
                "proposal_state_machine"
            )
            == PROPOSAL_STATE_MACHINE,
            "accepted-anchor selection-v4 requires its exact schema-3 contract",
        )
        return "accepted_anchor_selection_v4"
    if selection == DISTILL_EPOCH_SELECTION:
        _require(
            schema == 3,
            "pairwise selection-v3 requires distill manifest schema 3",
        )
        return "pairwise_selection_v3"
    if selection == LEGACY_DISTILL_EPOCH_SELECTION:
        registered = FORMAL_RECORDED_SOURCE_CLOSURE_AUTHORITIES.get(
            str(manifest_sha256)
        )
        _require(
            registered is not None
            and registered.get("response_protocol")
            == "legacy_brier_selection_v2"
            and Path(manifest_path).resolve()
            == Path(str(registered["manifest_path"])).resolve()
            and schema == 3,
            "unregistered legacy Brier selection-v2 authority is forbidden",
        )
        return "legacy_brier_selection_v2"
    raise ValueError("distill epoch-selection protocol is unsupported")


def _recompute_legacy_response_primary_selection(
    history: object,
) -> tuple[int, float]:
    """Reproduce the registered historical Surface/top-1 selection."""

    _require(isinstance(history, list) and history, "response history is empty")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        _require(isinstance(raw, Mapping), f"response history row {index} is invalid")
        row = dict(raw)
        _require(
            row.get("epoch") == index,
            "response history must start at control epoch 0 and be contiguous",
        )
        for field in (
            "surface_selection_score",
            "selection_score",
            "text_support_top1_agreement",
            "text_response_smooth_l1",
            "descriptor_relation_smooth_l1",
            "summary_token_cosine",
            "mean_descriptor_cosine",
            "all_view_descriptor_cosine",
        ):
            _finite(row.get(field), f"response history {field}")
        rows.append(row)
    control = {field: float(rows[0][field]) for field in (
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    )}
    ranked: list[tuple[int, tuple[float, float, float, float]]] = []
    for row in rows:
        deltas = {field: float(row[field]) - control[field] for field in control}
        feasible = all(
            delta >= -SURFACE_NONINFERIORITY_TOLERANCE
            or math.isclose(
                delta,
                -SURFACE_NONINFERIORITY_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for delta in deltas.values()
        )
        _require(
            row.get("surface_control_deltas") == deltas
            and row.get("surface_control_feasible") is feasible
            and row.get("surface_control_tolerance")
            == SURFACE_NONINFERIORITY_TOLERANCE,
            "response history Surface feasibility differs from independent recomputation",
        )
        if feasible:
            ranked.append(
                (
                    int(row["epoch"]),
                    (
                        float(row["text_support_top1_agreement"]),
                        -float(row["text_response_smooth_l1"]),
                        -float(row["descriptor_relation_smooth_l1"]),
                        float(row["surface_selection_score"]),
                    ),
                )
            )
    _require(ranked, "response history has no Surface-feasible epoch")
    best_epoch, _ = max(ranked, key=lambda value: value[1])
    best_score = float(rows[best_epoch]["surface_selection_score"])
    for row in rows:
        expected = best_score if int(row["epoch"]) == best_epoch else -1.0
        _require(
            math.isclose(
                float(row["selection_score"]), expected, rel_tol=0.0, abs_tol=0.0
            ),
            "response history selection score differs from independent recomputation",
        )
    return best_epoch, best_score


def _recompute_response_primary_selection(
    history: object,
) -> tuple[int, float]:
    """Independently reproduce pairwise-v3 robust fit-response selection."""

    _require(isinstance(history, list) and history, "response history is empty")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        _require(isinstance(raw, Mapping), f"response history row {index} is invalid")
        row = dict(raw)
        _require(
            row.get("epoch") == index,
            "response history must start at control epoch 0 and be contiguous",
        )
        for field in (
            "surface_selection_score",
            "selection_score",
            "text_support_top1_agreement",
            "text_response_smooth_l1",
            "text_response_mae",
            "descriptor_relation_smooth_l1",
            *FIT_RESPONSE_QUALITY_METRICS,
            *DISTILL_SURFACE_CONTROL_METRICS,
        ):
            _finite(row.get(field), f"response history {field}")
        _require(
            row.get("scene_response_objective") == SCENE_RESPONSE_OBJECTIVE,
            "response history scene-response objective differs",
        )
        raw_scenes = row.get("text_response_scene_metrics")
        _require(
            isinstance(raw_scenes, Mapping) and bool(raw_scenes),
            "response history fit scene metrics are empty",
        )
        scenes: dict[str, dict[str, float]] = {}
        for scene, raw_metrics in raw_scenes.items():
            _require(
                isinstance(scene, str)
                and bool(scene)
                and scene not in scenes
                and isinstance(raw_metrics, Mapping)
                and set(raw_metrics)
                == {
                    *FIT_RESPONSE_SCENE_QUALITY_METRICS,
                    *FIT_RESPONSE_SCENE_ERROR_METRICS,
                },
                "response history fit scene metrics are malformed",
            )
            scenes[scene] = {
                field: _finite(
                    raw_metrics.get(field),
                    f"response history scene {scene} {field}",
                )
                for field in (
                    *FIT_RESPONSE_SCENE_QUALITY_METRICS,
                    *FIT_RESPONSE_SCENE_ERROR_METRICS,
                )
            }
        for field in FIT_RESPONSE_SCENE_ERROR_METRICS:
            expected = max(values[field] for values in scenes.values())
            _require(
                _close(
                    row.get(f"text_response_scene_worst_{field}"),
                    expected,
                    tolerance=1e-12,
                ),
                "response history worst-scene error differs",
            )
        for field in FIT_RESPONSE_SCENE_QUALITY_METRICS:
            expected = min(values[field] for values in scenes.values())
            _require(
                _close(
                    row.get(f"text_response_scene_worst_{field}"),
                    expected,
                    tolerance=1e-12,
                ),
                "response history worst-scene quality differs",
            )
        row["_validated_scene_metrics"] = scenes
        rows.append(row)

    control = rows[0]
    control_surface = {
        field: float(control[field]) for field in DISTILL_SURFACE_CONTROL_METRICS
    }
    control_quality = {
        field: float(control[field]) for field in FIT_RESPONSE_QUALITY_METRICS
    }
    control_scenes = control["_validated_scene_metrics"]
    ranked: list[tuple[int, tuple[float, ...]]] = []
    for row in rows:
        scenes = row["_validated_scene_metrics"]
        _require(
            set(scenes) == set(control_scenes),
            "response history fit scene IDs drifted from epoch 0",
        )
        surface_deltas = {
            field: float(row[field]) - control_surface[field]
            for field in DISTILL_SURFACE_CONTROL_METRICS
        }
        surface_feasible = all(
            delta >= -SURFACE_NONINFERIORITY_TOLERANCE
            or math.isclose(
                delta,
                -SURFACE_NONINFERIORITY_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for delta in surface_deltas.values()
        )
        aggregate_deltas = {
            field: float(row[field]) - control_quality[field]
            for field in FIT_RESPONSE_QUALITY_METRICS
        }
        aggregate_feasible = all(
            delta >= -FIT_RESPONSE_NONINFERIORITY_TOLERANCE
            or math.isclose(
                delta,
                -FIT_RESPONSE_NONINFERIORITY_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for delta in aggregate_deltas.values()
        )
        scene_deltas: dict[str, dict[str, float]] = {}
        per_scene_feasible = True
        for scene in sorted(control_scenes):
            scene_deltas[scene] = {
                field: scenes[scene][field] - control_scenes[scene][field]
                for field in (
                    *FIT_RESPONSE_SCENE_QUALITY_METRICS,
                    *FIT_RESPONSE_SCENE_ERROR_METRICS,
                )
            }
            quality_feasible = all(
                scene_deltas[scene][field]
                >= -FIT_RESPONSE_NONINFERIORITY_TOLERANCE
                or math.isclose(
                    scene_deltas[scene][field],
                    -FIT_RESPONSE_NONINFERIORITY_TOLERANCE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field in FIT_RESPONSE_SCENE_QUALITY_METRICS
            )
            error_feasible = all(
                scene_deltas[scene][field]
                <= FIT_RESPONSE_NONINFERIORITY_TOLERANCE
                or math.isclose(
                    scene_deltas[scene][field],
                    FIT_RESPONSE_NONINFERIORITY_TOLERANCE,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field in FIT_RESPONSE_SCENE_ERROR_METRICS
            )
            per_scene_feasible = (
                per_scene_feasible and quality_feasible and error_feasible
            )
        fit_feasible = aggregate_feasible and per_scene_feasible
        feasible = surface_feasible and fit_feasible
        error_improvement = {
            "smooth_l1": float(control["text_response_smooth_l1"])
            - float(row["text_response_smooth_l1"]),
            "mae": float(control["text_response_mae"])
            - float(row["text_response_mae"]),
        }
        _require(
            row.get("surface_control_deltas") == surface_deltas
            and row.get("surface_control_feasible") is surface_feasible
            and row.get("surface_control_tolerance")
            == SURFACE_NONINFERIORITY_TOLERANCE
            and row.get("fit_response_control_deltas")
            == {
                "aggregate_quality": aggregate_deltas,
                "per_scene": scene_deltas,
            }
            and row.get("fit_response_aggregate_control_feasible")
            is aggregate_feasible
            and row.get("fit_response_per_scene_control_feasible")
            is per_scene_feasible
            and row.get("fit_response_control_feasible") is fit_feasible
            and row.get("fit_response_control_tolerance")
            == FIT_RESPONSE_NONINFERIORITY_TOLERANCE
            and row.get("response_selection_feasible") is feasible
            and row.get("fit_response_error_improvement_control_minus_candidate")
            == error_improvement,
            "response history robust selection fields differ from recomputation",
        )
        if int(row["epoch"]) > 0:
            expected_scene_loss = 0.2 * _finite(
                row.get("scene_profile_loss"), "history scene profile loss"
            ) + _finite(row.get("scene_ranking_loss"), "history scene ranking loss")
            _require(
                _close(
                    row.get("scene_response_loss"),
                    expected_scene_loss,
                    tolerance=1e-7,
                ),
                "response history scene composite loss differs",
            )
        if feasible:
            ranked.append(
                (
                    int(row["epoch"]),
                    (
                        -float(row["text_response_smooth_l1"]),
                        -float(row["text_response_mae"]),
                        float(row["text_response_ranking_spearman_p05"]),
                        float(row["text_response_ranking_spearman_mean"]),
                        float(row["text_response_profile_cosine_p05"]),
                        float(row["text_response_profile_cosine_mean"]),
                        float(row["text_response_top_decile_overlap_p05"]),
                        float(row["text_response_top_decile_overlap_mean"]),
                        float(row["text_support_top1_agreement"]),
                        -float(row["descriptor_relation_smooth_l1"]),
                        float(row["surface_selection_score"]),
                    ),
                )
            )
    _require(ranked, "response history has no robust feasible epoch")
    best_epoch, _ = max(ranked, key=lambda value: value[1])
    best_score = float(rows[best_epoch]["surface_selection_score"])
    for row in rows:
        expected = best_score if int(row["epoch"]) == best_epoch else -1.0
        _require(
            math.isclose(
                float(row["selection_score"]),
                expected,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "response history selection score differs from independent recomputation",
        )
    return best_epoch, best_score


def _state_dict_sha256(state: object, *, label: str) -> str:
    _require(
        isinstance(state, Mapping) and bool(state),
        f"{label} must be a non-empty state_dict",
    )
    records = []
    for name in sorted(state):
        value = state[name]
        _require(
            isinstance(name, str) and bool(name) and torch.is_tensor(value),
            f"{label} fields differ",
        )
        tensor = value.detach().cpu().contiguous()
        _require(
            not (tensor.is_floating_point() or tensor.is_complex())
            or bool(torch.isfinite(tensor).all()),
            f"{label} contains a non-finite tensor",
        )
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_sha256": tensor_sha256(tensor),
            }
        )
    return canonical_json_sha256(records)


def _history_chain_digest(
    record: Mapping[str, Any], previous_sha256: str | None
) -> str:
    payload = dict(record)
    payload.pop("history_hash_chain", None)
    payload.pop("selection_score", None)
    return canonical_json_sha256(
        {"previous_sha256": previous_sha256, "record": payload}
    )


def _accepted_anchor_rank(
    record: Mapping[str, Any], *, label: str
) -> tuple[float, ...]:
    values = {
        field: _finite(record.get(field), f"{label} {field}")
        for field in (
            "text_response_smooth_l1",
            "text_response_mae",
            "text_response_ranking_spearman_p05",
            "text_response_ranking_spearman_mean",
            "text_response_profile_cosine_p05",
            "text_response_profile_cosine_mean",
            "text_response_top_decile_overlap_p05",
            "text_response_top_decile_overlap_mean",
            "text_support_top1_agreement",
            "descriptor_relation_smooth_l1",
            "surface_selection_score",
        )
    }
    return (
        -values["text_response_smooth_l1"],
        -values["text_response_mae"],
        values["text_response_ranking_spearman_p05"],
        values["text_response_ranking_spearman_mean"],
        values["text_response_profile_cosine_p05"],
        values["text_response_profile_cosine_mean"],
        values["text_response_top_decile_overlap_p05"],
        values["text_response_top_decile_overlap_mean"],
        values["text_support_top1_agreement"],
        -values["descriptor_relation_smooth_l1"],
        values["surface_selection_score"],
    )


def _validate_accepted_anchor_checkpoint_state(
    checkpoint: Mapping[str, Any],
    *,
    control_payload: Mapping[str, Any],
    patience: int,
) -> dict[str, Any]:
    """Replay v4 state transitions and prove the serialized state is the best."""

    provenance = checkpoint.get("provenance")
    config = checkpoint.get("training_config")
    _require(
        checkpoint.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE
        and isinstance(provenance, Mapping)
        and provenance.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE
        and isinstance(config, Mapping)
        and config.get("proposal_state_machine") == PROPOSAL_STATE_MACHINE,
        "accepted-anchor proposal state-machine contract differs",
    )
    best_state_sha = checkpoint.get("best_state_dict_sha256")
    _require(
        _is_sha256(best_state_sha)
        and _state_dict_sha256(
            checkpoint.get("state_dict"), label="published response checkpoint state"
        )
        == best_state_sha,
        "response checkpoint does not publish the best state",
    )
    accepted_anchor = checkpoint.get("accepted_anchor")
    _require(
        isinstance(accepted_anchor, Mapping)
        and set(accepted_anchor)
        == {
            "epoch",
            "state_dict_sha256",
            "accepted_proposal_count",
            "rejected_proposal_count",
        }
        and _is_sha256(accepted_anchor.get("state_dict_sha256"))
        and all(
            isinstance(accepted_anchor.get(field), int)
            and not isinstance(accepted_anchor.get(field), bool)
            and int(accepted_anchor[field]) >= 0
            for field in (
                "epoch",
                "accepted_proposal_count",
                "rejected_proposal_count",
            )
        ),
        "accepted-anchor terminal metadata differs",
    )
    history = checkpoint.get("history")
    _require(
        isinstance(history, list) and bool(history),
        "accepted-anchor checkpoint history is empty",
    )
    control_state_sha = _state_dict_sha256(
        control_payload.get("state_dict"), label="Surface control state"
    )
    previous_chain: str | None = None
    anchor_epoch = 0
    anchor_sha = control_state_sha
    best_epoch = 0
    best_sha = control_state_sha
    stale = 0
    accepted_count = 0
    rejected_count = 0
    feasible_ranks: list[tuple[int, tuple[float, ...]]] = []
    proposal_fields = {
        "index",
        "source_anchor_epoch",
        "anchor_state_dict_sha256",
        "raw_state_dict_sha256",
        "trial_state_dict_sha256",
        "alpha_numerator",
        "alpha_denominator",
        "optimizer_state_reset",
        "validation_evaluations",
        "backtracking",
        "persistent_generator",
    }
    for index, raw_record in enumerate(history):
        _require(
            isinstance(raw_record, Mapping) and raw_record.get("epoch") == index,
            "accepted-anchor history must be contiguous from epoch 0",
        )
        record = dict(raw_record)
        expected_chain_sha = _history_chain_digest(record, previous_chain)
        _require(
            record.get("history_hash_chain")
            == {
                "algorithm": HISTORY_HASH_CHAIN_ALGORITHM,
                "previous_sha256": previous_chain,
                "sha256": expected_chain_sha,
            },
            f"accepted-anchor history hash chain differs at row {index}",
        )
        previous_chain = expected_chain_sha
        if record.get("response_selection_feasible") is True:
            feasible_ranks.append(
                (index, _accepted_anchor_rank(record, label=f"history row {index}"))
            )
        if index == 0:
            _require(
                record.get("initialization") == "frozen_surface_control_checkpoint"
                and record.get("state_machine_role")
                == "frozen_control_initial_anchor"
                and "proposal" not in record
                and "proposal_losses" not in record
                and "loss_measurement_state" not in record
                and not any(
                    field in record for field in PROPOSAL_LOSS_FLAT_MIRROR.values()
                )
                and record.get("accepted") is True
                and record.get("rejected") is False,
                "accepted-anchor control history row differs",
            )
            best_updated = True
        else:
            proposal = record.get("proposal")
            _require(
                isinstance(proposal, Mapping)
                and set(proposal) == proposal_fields
                and proposal.get("index") == index
                and proposal.get("source_anchor_epoch") == anchor_epoch
                and proposal.get("anchor_state_dict_sha256") == anchor_sha
                and _is_sha256(proposal.get("raw_state_dict_sha256"))
                and _is_sha256(proposal.get("trial_state_dict_sha256"))
                and proposal.get("alpha_numerator") == 1
                and proposal.get("alpha_denominator") == 2048
                and proposal.get("optimizer_state_reset") is True
                and proposal.get("validation_evaluations") == 1
                and proposal.get("backtracking")
                == "none_fixed_alpha_single_trial"
                and proposal.get("persistent_generator")
                == "advanced_never_rolled_back"
                and record.get("state_machine_role") == "fixed_micro_ray_trial",
                f"accepted-anchor proposal row {index} differs",
            )
            proposal_losses = record.get("proposal_losses")
            _require(
                record.get("loss_measurement_state")
                == "raw_proposal_before_micro_projection"
                and isinstance(proposal_losses, Mapping)
                and tuple(proposal_losses) == PROPOSAL_LOSS_FIELDS
                and all(
                    isinstance(proposal_losses.get(field), (int, float))
                    and not isinstance(proposal_losses.get(field), bool)
                    and math.isfinite(float(proposal_losses[field]))
                    and float(proposal_losses[field]) >= 0.0
                    and record.get(PROPOSAL_LOSS_FLAT_MIRROR[field])
                    == proposal_losses[field]
                    for field in PROPOSAL_LOSS_FIELDS
                ),
                f"accepted-anchor raw-proposal loss accounting differs at row {index}",
            )
            accepted = record.get("response_selection_feasible") is True
            _require(
                record.get("accepted") is accepted
                and record.get("rejected") is (not accepted),
                "accepted-anchor acceptance decision differs",
            )
            if accepted:
                anchor_epoch = index
                anchor_sha = str(proposal["trial_state_dict_sha256"])
                accepted_count += 1
            else:
                rejected_count += 1
            _require(bool(feasible_ranks), "accepted-anchor feasible set is empty")
            selected_epoch = max(feasible_ranks, key=lambda value: value[1])[0]
            best_updated = selected_epoch == index
            if best_updated:
                best_epoch = index
                best_sha = str(proposal["trial_state_dict_sha256"])
                stale = 0
            else:
                stale += 1
        patience_stop = bool(patience and stale >= patience)
        _require(
            record.get("anchor_epoch_after_proposal") == anchor_epoch
            and record.get("anchor_state_dict_sha256_after_proposal") == anchor_sha
            and record.get("best_updated") is best_updated
            and record.get("best_epoch_after_proposal") == best_epoch
            and record.get("best_state_dict_sha256_after_proposal") == best_sha
            and record.get("patience_stale_after_proposal") == stale
            and record.get("patience_stop_after_proposal") is patience_stop
            and (not patience_stop or index == len(history) - 1),
            "accepted-anchor history transition differs",
        )
    expected_terminal = {
        "epoch": anchor_epoch,
        "state_dict_sha256": anchor_sha,
        "accepted_proposal_count": accepted_count,
        "rejected_proposal_count": rejected_count,
    }
    _require(
        bool(feasible_ranks)
        and accepted_anchor == expected_terminal
        and accepted_count + rejected_count == len(history) - 1
        and checkpoint.get("best_epoch") == best_epoch
        and checkpoint.get("best_state_dict_sha256") == best_sha
        and checkpoint.get("history_hash_chain_sha256") == previous_chain,
        "accepted-anchor terminal checkpoint provenance differs",
    )
    return {
        "proposal_state_machine": dict(PROPOSAL_STATE_MACHINE),
        "best_state_dict_sha256": str(best_state_sha),
        "accepted_anchor": dict(accepted_anchor),
        "history_hash_chain_sha256": str(previous_chain),
    }


def _validate_immutable_source_snapshot(raw_root: object, *, label: str) -> Path:
    _require(
        isinstance(raw_root, str) and raw_root and Path(raw_root).is_absolute(),
        f"{label} source snapshot root must be an absolute path",
    )
    lexical_root = Path(raw_root)
    try:
        root = lexical_root.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"{label} source snapshot root is missing") from exc
    _require(
        lexical_root == root,
        f"{label} source snapshot root must not traverse a symlink",
    )

    def validate_node(path: Path, *, expect_directory: bool = False) -> None:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise ValueError(f"{label} source snapshot entry is unavailable: {path}") from exc
        _require(
            not stat.S_ISLNK(info.st_mode),
            f"{label} source snapshot contains a symlink: {path}",
        )
        if expect_directory:
            _require(
                stat.S_ISDIR(info.st_mode),
                f"{label} source snapshot root is not a directory",
            )
        else:
            _require(
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
                f"{label} source snapshot contains a non-file entry: {path}",
            )
            if stat.S_ISREG(info.st_mode):
                _require(
                    info.st_nlink == 1,
                    f"{label} source snapshot contains a multiply linked file: {path}",
                )
        _require(
            stat.S_IMODE(info.st_mode) & 0o222 == 0,
            f"{label} source snapshot contains a writable entry: {path}",
        )

    validate_node(root, expect_directory=True)
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        for name in sorted([*directory_names, *file_names]):
            validate_node(Path(current) / name)
    return root


def _validate_snapshot_source_hashes(
    records: object,
    *,
    required: set[str],
    source_snapshot_root: object,
    label: str,
) -> Path:
    _require(
        isinstance(records, Mapping) and set(records) == required,
        f"{label} implementation source set differs",
    )
    root = _validate_immutable_source_snapshot(source_snapshot_root, label=label)
    for relative in sorted(required):
        source = root / relative
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"{label} implementation escaped or is missing: {relative}"
            ) from exc
        info = os.lstat(source)
        _require(
            resolved == source
            and stat.S_ISREG(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) & 0o222 == 0
            and records.get(relative) == _sha256(source),
            f"{label} implementation changed in producer snapshot: {relative}",
        )
    return root


def _validate_distill_runtime_closure(
    value: object,
    *,
    source_snapshot_root: Path,
) -> str:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"schema_version", "repository_sources", "runtime_fingerprint", "digest"}
        and value.get("schema_version") == 1,
        "distill runtime closure schema differs",
    )
    repo = Path(source_snapshot_root).resolve()
    repository = value.get("repository_sources")
    _require(
        isinstance(repository, Mapping)
        and set(repository)
        == {"python_entrypoints", "shell_sources", "files", "digest"}
        and repository.get("python_entrypoints")
        == list(DISTILL_RUNTIME_PYTHON_ENTRYPOINTS)
        and repository.get("shell_sources")
        == list(DISTILL_RUNTIME_SHELL_SOURCES),
        "distill repository closure contract differs",
    )
    expected_python = discover_repo_python_closure(
        repo,
        DISTILL_RUNTIME_PYTHON_ENTRYPOINTS,
    )
    expected_paths = sorted(set(expected_python) | set(DISTILL_RUNTIME_SHELL_SOURCES))
    expected_files = {relative: _sha256(repo / relative) for relative in expected_paths}
    expected_repository = {
        "python_entrypoints": list(DISTILL_RUNTIME_PYTHON_ENTRYPOINTS),
        "shell_sources": list(DISTILL_RUNTIME_SHELL_SOURCES),
        "files": expected_files,
    }
    expected_repository["digest"] = canonical_json_sha256(expected_repository)
    _require(
        dict(repository) == expected_repository,
        "distill repository source closure changed",
    )
    runtime = value.get("runtime_fingerprint")
    runtime_fields = {
        "repository_import_root",
        "imported_modules",
        "python_executable",
        "python_version",
        "python_prefix",
        "platform",
        "machine",
        "packages",
        "torch_git_version",
        "torch_cuda_version",
        "torch_cudnn_version",
        "environment",
    }
    _require(
        isinstance(runtime, Mapping)
        and set(runtime) == runtime_fields
        and runtime.get("repository_import_root") == str(repo),
        "distill runtime fingerprint differs from producer snapshot",
    )
    imports = runtime.get("imported_modules")
    _require(
        isinstance(imports, Mapping) and set(imports) == set(DISTILL_RUNTIME_MODULES),
        "distill runtime import set differs",
    )
    for module_name in DISTILL_RUNTIME_MODULES:
        expected_relative = (
            "radio_gs/__init__.py"
            if module_name == "radio_gs"
            else f"{module_name.replace('.', '/')}.py"
        )
        expected_path = repo / expected_relative
        record = imports.get(module_name)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "relative_path", "sha256"}
            and record.get("path") == str(expected_path)
            and record.get("relative_path") == expected_relative
            and record.get("sha256") == _sha256(expected_path),
            f"distill runtime import differs from producer snapshot: {module_name}",
        )
    _validate_record_file(runtime.get("python_executable"), "distill Python executable")
    packages = runtime.get("packages")
    environment = runtime.get("environment")
    _require(
        isinstance(packages, Mapping)
        and set(packages) == set(DISTILL_RUNTIME_PACKAGES)
        and all(isinstance(version, str) for version in packages.values())
        and isinstance(environment, Mapping)
        and set(environment) == set(DISTILL_RUNTIME_ENVIRONMENT_KEYS)
        and all(value is None or isinstance(value, str) for value in environment.values())
        and environment.get("RADIO_GS_REPO_ROOT") == str(repo),
        "distill producer runtime environment differs",
    )
    expected_closure = {
        "schema_version": 1,
        "repository_sources": expected_repository,
        "runtime_fingerprint": dict(runtime),
    }
    expected_closure["digest"] = canonical_json_sha256(expected_closure)
    _require(dict(value) == expected_closure, "distill runtime closure digest differs")
    return str(expected_closure["digest"])


def _validate_registered_recorded_runtime_closure(
    value: object,
    *,
    source_root: Path,
    implementation_sources: object,
    expected_digest: str,
) -> str:
    """Validate an exact registered producer closure without live source reads.

    The registered schema-3 authority recorded the execution repository rather
    than an archived read-only copy. Its manifest is nevertheless immutable and
    checkpoint-bound. For that one exact manifest, validate every recorded
    digest and cross-binding internally instead of comparing against a worktree
    that has legitimately advanced during downstream evaluation development.
    """

    _require(
        isinstance(value, Mapping)
        and set(value)
        == {"schema_version", "repository_sources", "runtime_fingerprint", "digest"}
        and value.get("schema_version") == 1,
        "registered distill runtime closure schema differs",
    )
    source_root = Path(source_root)
    _require(
        source_root.is_absolute()
        and source_root.resolve(strict=True) == source_root
        and not source_root.is_symlink(),
        "registered distill source root identity differs",
    )
    repository = value.get("repository_sources")
    _require(
        isinstance(repository, Mapping)
        and set(repository)
        == {"python_entrypoints", "shell_sources", "files", "digest"}
        and repository.get("python_entrypoints")
        == list(DISTILL_RUNTIME_PYTHON_ENTRYPOINTS)
        and repository.get("shell_sources")
        == list(DISTILL_RUNTIME_SHELL_SOURCES),
        "registered distill repository closure contract differs",
    )
    files = repository.get("files")
    _require(
        isinstance(files, Mapping)
        and bool(files)
        and all(
            isinstance(relative, str)
            and bool(relative)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and _is_sha256(digest)
            for relative, digest in files.items()
        ),
        "registered distill repository file index differs",
    )
    expected_repository = {
        "python_entrypoints": list(DISTILL_RUNTIME_PYTHON_ENTRYPOINTS),
        "shell_sources": list(DISTILL_RUNTIME_SHELL_SOURCES),
        "files": dict(files),
    }
    expected_repository["digest"] = canonical_json_sha256(expected_repository)
    _require(
        dict(repository) == expected_repository,
        "registered distill repository digest differs",
    )
    _require(
        isinstance(implementation_sources, Mapping)
        and set(implementation_sources) == AUTHORITY_DISTILL_IMPLEMENTATION_SOURCES
        and all(
            implementation_sources.get(relative) == files.get(relative)
            for relative in AUTHORITY_DISTILL_IMPLEMENTATION_SOURCES
        ),
        "registered distill implementation sources differ from its closure",
    )

    runtime = value.get("runtime_fingerprint")
    runtime_fields = {
        "repository_import_root",
        "imported_modules",
        "python_executable",
        "python_version",
        "python_prefix",
        "platform",
        "machine",
        "packages",
        "torch_git_version",
        "torch_cuda_version",
        "torch_cudnn_version",
        "environment",
    }
    _require(
        isinstance(runtime, Mapping)
        and set(runtime) == runtime_fields
        and runtime.get("repository_import_root") == str(source_root),
        "registered distill runtime fingerprint differs",
    )
    imports = runtime.get("imported_modules")
    _require(
        isinstance(imports, Mapping) and set(imports) == set(DISTILL_RUNTIME_MODULES),
        "registered distill runtime import set differs",
    )
    for module_name in DISTILL_RUNTIME_MODULES:
        expected_relative = (
            "radio_gs/__init__.py"
            if module_name == "radio_gs"
            else f"{module_name.replace('.', '/')}.py"
        )
        record = imports.get(module_name)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "relative_path", "sha256"}
            and record.get("path") == str(source_root / expected_relative)
            and record.get("relative_path") == expected_relative
            and record.get("sha256") == files.get(expected_relative),
            f"registered distill runtime import differs: {module_name}",
        )
    _validate_record_file(runtime.get("python_executable"), "distill Python executable")
    packages = runtime.get("packages")
    environment = runtime.get("environment")
    _require(
        isinstance(packages, Mapping)
        and set(packages) == set(DISTILL_RUNTIME_PACKAGES)
        and all(isinstance(version, str) for version in packages.values())
        and isinstance(environment, Mapping)
        and set(environment) == set(DISTILL_RUNTIME_ENVIRONMENT_KEYS)
        and all(item is None or isinstance(item, str) for item in environment.values())
        and environment.get("RADIO_GS_REPO_ROOT") == str(source_root),
        "registered distill runtime environment differs",
    )
    expected_closure = {
        "schema_version": 1,
        "repository_sources": expected_repository,
        "runtime_fingerprint": dict(runtime),
    }
    expected_closure["digest"] = canonical_json_sha256(expected_closure)
    _require(
        dict(value) == expected_closure
        and expected_closure["digest"] == expected_digest,
        "registered distill runtime closure digest differs",
    )
    return expected_digest


def _validate_authority_seed_evidence(
    evidence: object,
    *,
    seed: int,
    gpu_identity: object,
) -> None:
    _require(isinstance(evidence, Mapping), f"authority seed {seed} lacks evidence")
    for field in (
        "checkpoint",
        "report",
        "training_log",
        "audit_report",
        "guard_command",
        "guard_telemetry",
        "guard_receipt",
    ):
        _validate_record_file(evidence.get(field), f"authority seed {seed} {field}")
    journal = evidence.get("kernel_journal")
    _require(
        isinstance(journal, Mapping)
        and set(journal)
        == {"path", "sha256", "seed", "start_epoch", "end_epoch", "fault_count"}
        and journal.get("seed") == seed
        and isinstance(journal.get("start_epoch"), int)
        and isinstance(journal.get("end_epoch"), int)
        and int(journal["start_epoch"]) > 0
        and int(journal["end_epoch"]) >= int(journal["start_epoch"])
        and journal.get("fault_count") == 0,
        f"authority seed {seed} kernel journal binding differs",
    )
    journal_path = _bound_file(
        journal.get("path"), journal.get("sha256"), f"authority seed {seed} journal"
    )

    def load_journal(handle) -> str:
        try:
            return handle.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"authority seed {seed} journal is not UTF-8") from exc

    journal_text, _, _ = stable_descriptor_load(
        journal_path,
        load_journal,
        expected_sha256=str(journal["sha256"]),
        label=f"authority seed {seed} kernel journal",
    )
    expected_header = (
        f"surface_text_response_seed={seed}\t"
        f"start_epoch={journal['start_epoch']}\tend_epoch={journal['end_epoch']}"
    )
    _require(
        journal_text.splitlines()
        and journal_text.splitlines()[0] == expected_header
        and not any(
            DISTILL_KERNEL_FAULT_PATTERN.search(line)
            for line in journal_text.splitlines()[1:]
        ),
        f"authority seed {seed} kernel journal contains Xid/PCIe faults",
    )
    gpu_epochs: dict[str, int] = {}
    for phase_name, field in (
        (f"pre_seed{seed}", "gpu_preflight"),
        (f"post_seed{seed}", "gpu_postflight"),
    ):
        check_record = evidence.get(field)
        _require(
            isinstance(check_record, Mapping)
            and set(check_record) == {"path", "sha256", "observed_epoch"}
            and isinstance(check_record.get("observed_epoch"), int)
            and int(check_record["observed_epoch"]) > 0,
            f"authority seed {seed} {field} binding differs",
        )
        check_path = _bound_file(
            check_record["path"],
            check_record["sha256"],
            f"authority seed {seed} {field}",
        )
        check = _json_object(check_path)
        _require(
            check.get("status") == "physical_gpu1_idle_and_pcie_responsive"
            and check.get("phase") == phase_name
            and check.get("gpu_identity") == gpu_identity
            and check.get("compute_owners") == []
            and check.get("observed_epoch") == check_record["observed_epoch"],
            f"authority seed {seed} {field} differs",
        )
        gpu_epochs[field] = int(check_record["observed_epoch"])
    telemetry = evidence.get("telemetry_interval")
    telemetry_fields = {
        "path",
        "sha256",
        "seed",
        "first_row",
        "last_row",
        "row_count",
        "row_interval_sha256",
        "first_timestamp",
        "last_timestamp",
        "first_epoch",
        "last_epoch",
    }
    _require(
        isinstance(telemetry, Mapping)
        and set(telemetry) == telemetry_fields
        and telemetry.get("seed") == seed
        and telemetry.get("first_row") == 0
        and isinstance(telemetry.get("row_count"), int)
        and int(telemetry["row_count"]) > 0
        and telemetry.get("last_row") == int(telemetry["row_count"]) - 1
        and _is_sha256(telemetry.get("row_interval_sha256"))
        and evidence.get("guard_telemetry")
        == {"path": telemetry.get("path"), "sha256": telemetry.get("sha256")},
        f"authority seed {seed} telemetry interval binding differs",
    )
    _bound_file(
        telemetry["path"], telemetry["sha256"], f"authority seed {seed} telemetry interval"
    )
    timeline = evidence.get("execution_timeline")
    timeline_fields = {
        "gpu_preflight_observed_epoch",
        "command_prepared_epoch",
        "journal_start_epoch",
        "telemetry_first_epoch",
        "telemetry_last_epoch",
        "journal_end_epoch",
        "gpu_postflight_observed_epoch",
    }
    _require(
        isinstance(timeline, Mapping)
        and set(timeline) == timeline_fields
        and timeline.get("gpu_preflight_observed_epoch") == gpu_epochs["gpu_preflight"]
        and timeline.get("journal_start_epoch") == journal["start_epoch"]
        and timeline.get("telemetry_first_epoch") == telemetry["first_epoch"]
        and timeline.get("telemetry_last_epoch") == telemetry["last_epoch"]
        and timeline.get("journal_end_epoch") == journal["end_epoch"]
        and timeline.get("gpu_postflight_observed_epoch") == gpu_epochs["gpu_postflight"]
        and float(timeline["gpu_preflight_observed_epoch"])
        <= float(timeline["command_prepared_epoch"])
        <= float(timeline["journal_start_epoch"])
        <= float(timeline["telemetry_first_epoch"])
        <= float(timeline["telemetry_last_epoch"])
        <= float(timeline["journal_end_epoch"])
        <= float(timeline["gpu_postflight_observed_epoch"]),
        f"authority seed {seed} execution timeline differs",
    )


def _bound_file(raw_path: object, raw_sha: object, label: str) -> Path:
    path = Path(str(raw_path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bound {label} is missing: {path}")
    if not _is_sha256(raw_sha) or _sha256(path) != str(raw_sha):
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def _file_record(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _cache_binding_records(metadata: object, label: str) -> list[dict[str, str]]:
    _require(isinstance(metadata, Mapping), f"{label} cache provenance is missing")
    paths = metadata.get("cache_paths")
    bindings = metadata.get("cache_bindings")
    _require(
        isinstance(paths, list)
        and paths
        and isinstance(bindings, list)
        and len(bindings) == len(paths),
        f"{label} must bind every cache path",
    )
    expected = [_file_record(Path(path)) for path in paths]
    _require(bindings == expected, f"{label} cache SHA256 bindings differ")
    return expected


def _semantic_cache_provenance(metadata: object, label: str) -> dict[str, Any]:
    _require(isinstance(metadata, Mapping), f"{label} cache provenance is missing")
    value = dict(metadata)
    value.pop("cache_bindings", None)
    return value


def _validate_fit_text_bank(binding: object, label: str) -> dict[str, Any]:
    _require(
        isinstance(binding, Mapping) and set(binding) == FIT_TEXT_BANK_FIELDS,
        f"{label} fit text-bank binding fields differ",
    )
    artifact = _bound_file(
        binding["artifact_path"], binding["artifact_sha256"], f"{label} fit text bank"
    )
    manifest = _bound_file(
        binding["manifest_path"],
        binding["manifest_sha256"],
        f"{label} fit text-bank manifest",
    )
    _require(
        binding.get("split") == "fit"
        and isinstance(binding.get("query_count"), int)
        and not isinstance(binding.get("query_count"), bool)
        and int(binding["query_count"]) > 0,
        f"{label} text bank is not a non-empty frozen fit split",
    )
    for field in FIT_TEXT_BANK_FIELDS - {
        "artifact_path",
        "artifact_sha256",
        "manifest_path",
        "manifest_sha256",
        "split",
        "query_count",
    }:
        _require(_is_sha256(binding.get(field)), f"{label} {field} is not SHA256")
    return {
        **dict(binding),
        "artifact_path": str(artifact),
        "manifest_path": str(manifest),
    }


def _validate_calibration_manifest(
    path: Path,
    *,
    seed: int,
    response_lambdas: Mapping[str, Any],
    surface_control: Mapping[str, Any],
    design_diagnostic: Mapping[str, Any],
    fit_text_bank: Mapping[str, Any],
    response_protocol: str,
    token_weight: float,
    relation_weight: float,
    label: str,
) -> None:
    payload = _json_object(path)
    expected_algorithm = (
        PAIRWISE_CALIBRATION_ALGORITHM_VERSION
        if response_protocol
        in {"pairwise_selection_v3", "accepted_anchor_selection_v4"}
        else LEGACY_BRIER_CALIBRATION_ALGORITHM_VERSION
    )
    _require(
        payload.get("schema_version") == 2
        and payload.get("artifact_type")
        == "surface_text_response_gradient_calibration"
        and payload.get("algorithm_version") == expected_algorithm
        and payload.get("benchmark_vocabulary_opened") is False
        and payload.get("uses_benchmark_scenes") is False
        and payload.get("uses_benchmark_test_vocabulary") is False
        and payload.get("seed") == seed
        and payload.get("surface_control") == dict(surface_control),
        f"{label} calibration contract differs",
    )
    _require(
        payload.get("design_diagnostic") == dict(design_diagnostic),
        f"{label} calibration design diagnostic differs",
    )
    gradient = payload.get("gradient_contract")
    _require(
        isinstance(gradient, Mapping)
        and gradient.get("measurement_point")
        == "exact_seed_frozen_surface_control_state_dict"
        and gradient.get("branch_target_ratio") == 0.25
        and gradient.get("response_lambdas") == dict(response_lambdas)
        and _close(
            gradient.get("combined_response_to_surface_upper_bound_ratio"),
            0.5,
            tolerance=1e-12,
        ),
        f"{label} calibration response budgets differ",
    )
    objective = payload.get("objective_contract")
    _require(
        objective
        == _calibration_objective_contract(
            response_protocol=response_protocol,
            token_weight=token_weight,
            relation_weight=relation_weight,
        ),
        f"{label} calibration objective contract differs",
    )
    inventory = gradient.get("trainable_parameters")
    losses = gradient.get("loss_values")
    _require(
        isinstance(inventory, list)
        and len(inventory) == gradient.get("trainable_parameter_count")
        and len({row.get("name") for row in inventory if isinstance(row, Mapping)})
        == len(inventory)
        and all(
            isinstance(row, Mapping)
            and set(row) == {"name", "shape"}
            and isinstance(row["name"], str)
            and isinstance(row["shape"], list)
            and all(isinstance(size, int) and size >= 0 for size in row["shape"])
            for row in inventory
        )
        and isinstance(losses, Mapping)
        and set(losses)
        == {
            "surface", "token", "descriptor", "relation",
            "independent_response", "scene_response", "scene_profile",
            "scene_ranking",
        }
        and all(math.isfinite(float(value)) and float(value) >= 0 for value in losses.values()),
        f"{label} calibration topology/loss measurements differ",
    )
    _require(
        payload.get("fit_text_bank") == dict(fit_text_bank),
        f"{label} calibration fit text-bank binding differs",
    )


def _validate_legacy_calibration_manifest(
    path: Path,
    *,
    response_lambda: float,
    fit_text_bank: Mapping[str, Any],
    label: str,
) -> None:
    payload = _json_object(path)
    gradient = payload.get("gradient_contract")
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "surface_text_response_gradient_calibration"
        and payload.get("benchmark_vocabulary_opened") is False
        and payload.get("shared_training_seeds") == list(REQUIRED_SEEDS)
        and isinstance(gradient, Mapping)
        and gradient.get("response_loss")
        == "independent_normalized_cosine_response_smooth_l1"
        and _close(gradient.get("response_lambda"), response_lambda, tolerance=1e-12)
        and payload.get("fit_text_bank") == dict(fit_text_bank),
        f"{label} legacy calibration contract differs",
    )


def _validate_current_source_hashes(
    records: object,
    *,
    required: set[str],
    label: str,
) -> None:
    _require(
        isinstance(records, Mapping) and set(records) == required,
        f"{label} implementation source set differs",
    )
    repo = Path(__file__).resolve().parents[2]
    for relative in sorted(required):
        source = repo / relative
        _require(
            source.is_file() and records.get(relative) == _sha256(source),
            f"{label} implementation changed: {relative}",
        )


def _validate_distill_thermal_contract(
    value: object,
    *,
    schema: int,
) -> Mapping[str, Any]:
    legacy_keys = {
        "physical_gpu",
        "maximum_temperature_c",
        "maximum_start_temperature_c",
        "maximum_power_limit_w",
        "poll_seconds",
        "soft_pause_temperature_c",
        "soft_resume_temperature_c",
        "peer_gpu",
        "peer_pause_temperature_c",
        "peer_resume_temperature_c",
        "peer_quiet_seconds_before_launch",
        "peer_max_power_w",
        "peer_max_memory_mib",
        "peer_max_utilization_pct",
        "guard",
        "guard_sha256",
    }
    authority_keys = legacy_keys - {"guard_sha256"} | {
        "owner_pid_namespace_mode",
        "peer_activity_action",
    }
    _require(
        schema in {1, 2, 3}
        and isinstance(value, Mapping)
        and set(value) == (legacy_keys if schema == 1 else authority_keys)
        and value.get("physical_gpu") == 1
        and (
            schema == 1
            or value.get("owner_pid_namespace_mode")
            == "exclusive-singleton-after-clear-v1"
        ),
        "distill run thermal-safety contract differs",
    )
    return value


def _implementation_binding() -> list[dict[str, str]]:
    repo = Path(__file__).resolve().parents[2]
    return [_file_record(repo / relative) for relative in IMPLEMENTATION_SOURCES]


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_frozen_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        _require(
            _json_object(path) == dict(value),
            f"existing frozen output differs from strict recomputation: {path}",
        )
    else:
        _write_atomic_json(path, value)


def _validate_record_file(record: object, label: str) -> Path:
    _require(isinstance(record, Mapping), f"{label} binding must be an object")
    _require(set(record) == {"path", "sha256"}, f"{label} binding fields differ")
    return _bound_file(record["path"], record["sha256"], label)


def _validate_authoritative_surface_metrics(
    record: Mapping[str, Any],
    label: str,
) -> dict[str, float]:
    """Bind CPU-authoritative metrics to the immutable trainer sidecar witness."""

    tolerance = float(surface_finalizer.VALIDATION_RECOMPUTE_TOLERANCE)
    _require(
        record.get("metric_source")
        == "cpu_recomputed_from_bound_validation_caches",
        f"{label} lacks the CPU-recomputed metric authority",
    )
    validation = record.get("validation")
    reported_validation = record.get("reported_validation")
    baseline = record.get("recomputed_untrained_baseline")
    metric_keys = set(surface_finalizer.METRIC_KEYS)
    _require(
        isinstance(validation, Mapping)
        and set(validation) == metric_keys
        and isinstance(reported_validation, Mapping)
        and set(reported_validation) == metric_keys
        and isinstance(baseline, Mapping)
        and set(baseline) == metric_keys,
        f"{label} CPU/reported metric fields differ",
    )
    authoritative = {
        key: _finite(validation[key], f"{label} authoritative {key}")
        for key in surface_finalizer.METRIC_KEYS
    }
    reported = {
        key: _finite(reported_validation[key], f"{label} reported {key}")
        for key in surface_finalizer.METRIC_KEYS
    }
    baseline_metrics = {
        key: _finite(baseline[key], f"{label} baseline {key}")
        for key in surface_finalizer.METRIC_KEYS
    }
    _require(
        all(
            abs(authoritative[key] - reported[key]) <= tolerance
            for key in surface_finalizer.METRIC_KEYS
        ),
        f"{label} authoritative validation exceeds the CPU/GPU tolerance",
    )

    best_score = _finite(record.get("best_selection_score"), f"{label} best score")
    reported_best = _finite(
        record.get("reported_best_selection_score"), f"{label} reported best score"
    )
    score_delta = _finite(
        record.get("selection_score_delta"), f"{label} selection delta"
    )
    reported_delta = _finite(
        record.get("reported_selection_score_delta"),
        f"{label} reported selection delta",
    )
    _require(
        abs(best_score - reported_best) <= tolerance
        and abs(score_delta - reported_delta) <= tolerance,
        f"{label} authoritative score exceeds the CPU/GPU tolerance",
    )
    expected_best = 0.5 * (
        authoritative["mean_descriptor_cosine"]
        + authoritative["all_view_descriptor_cosine"]
    )
    expected_baseline = 0.5 * (
        baseline_metrics["mean_descriptor_cosine"]
        + baseline_metrics["all_view_descriptor_cosine"]
    )
    _require(
        _close(best_score, expected_best)
        and _close(score_delta, expected_best - expected_baseline),
        f"{label} authoritative score is not reproduced from CPU metrics",
    )

    sidecar_path = _bound_file(
        record.get("sidecar"), record.get("sidecar_sha256"), f"{label} sidecar"
    )
    sidecar = _json_object(sidecar_path)
    _require(
        sidecar.get("validation") == dict(reported_validation)
        and _close(sidecar.get("best_selection_score"), reported_best)
        and _close(sidecar.get("selection_score_delta"), reported_delta),
        f"{label} reported metrics differ from the bound trainer sidecar",
    )
    return authoritative


def _validate_attention_surface_bundle(
    manifest_path: Path,
    completion_path: Path,
) -> dict[str, Any]:
    from radio_gs.scripts import surface_text_response_distill_authority as distill_authority

    screen = _json_object(manifest_path)
    _require(
        screen.get("artifact_type")
        == "surface_c1024_attention_pooling_postcache_continuation"
        and screen.get("selected_variant") == "joint_attention_v1"
        and screen.get("selection_status") == "joint_attention_retained"
        and screen.get("promotion_gate_passed") is False
        and screen.get("benchmark_queries_opened") is False
        and screen.get("benchmark_masks_opened") is False,
        "invalid frozen Surface attention-postcache screen",
    )
    try:
        completed = datetime.fromisoformat(
            completion_path.read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError) as error:
        raise ValueError("invalid Surface attention-postcache completion") from error
    _require(
        completed.tzinfo is not None and completed.utcoffset() is not None,
        "Surface attention-postcache completion lacks timezone",
    )
    pairing_record = screen.get("cache_pairing_report")
    _require(
        isinstance(pairing_record, Mapping)
        and set(pairing_record) == {"path", "sha256"},
        "Surface attention-postcache screen lacks cache pairing",
    )
    pairing_path = _bound_file(
        pairing_record["path"], pairing_record["sha256"], "Surface cache pairing"
    )
    pairing = _json_object(pairing_path)
    rows = pairing.get("rows")
    _require(isinstance(rows, list) and len(rows) == 6, "Surface pairing rows differ")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "Surface pairing row is invalid")
        key = (str(row.get("role")), int(row.get("shard", -1)))
        _require(key not in indexed, "Surface pairing row is duplicated")
        indexed[key] = row
    expected_keys = {
        *(("train", shard) for shard in range(4)),
        *(("validation", shard) for shard in range(2)),
    }
    _require(set(indexed) == expected_keys, "Surface pairing role/shard grid differs")
    selected_caches = {
        role: [
            dict(indexed[(role, shard)]["c1024"])
            for shard in range(4 if role == "train" else 2)
        ]
        for role in ("train", "validation")
    }
    binding = distill_authority._surface_binding(
        surface_root=manifest_path.parent,
        candidate="context_c1024_geometric",
        train=selected_caches["train"],
        validation=selected_caches["validation"],
    )
    _require(
        binding.get("binding_mode") == distill_authority.ATTENTION_BINDING_MODE,
        "Surface attention-postcache authority binding differs",
    )
    run_record = screen.get("run_manifest")
    _require(
        isinstance(run_record, Mapping) and set(run_record) == {"path", "sha256"},
        "Surface attention-postcache screen lacks run manifest",
    )
    run_path = _bound_file(
        run_record["path"], run_record["sha256"], "Surface run manifest"
    )
    run_manifest = _json_object(run_path)
    radio_record = run_manifest.get("radio_checkpoint")
    if isinstance(radio_record, Mapping):
        radio_path_value = radio_record.get("path")
        radio_sha_value = radio_record.get("sha256")
    else:
        radio_path_value = radio_record
        radio_sha_value = run_manifest.get("radio_checkpoint_sha256")
    _require(
        isinstance(radio_path_value, str) and _is_sha256(radio_sha_value),
        "Surface run manifest lacks RADIO checkpoint",
    )
    radio_path = _bound_file(
        radio_path_value, radio_sha_value, "Surface RADIO checkpoint"
    )
    cache_state = {
        ("validation", shard): {
            "path": selected_caches["validation"][shard]["path"],
            "sha256": selected_caches["validation"][shard]["sha256"],
        }
        for shard in range(2)
    }
    validation_data = surface_finalizer._load_validation_evaluation_data(cache_state)
    summary_head = surface_finalizer._load_summary_head(radio_path)
    variants = screen.get("variants")
    joint = variants.get("joint_attention_v1") if isinstance(variants, Mapping) else None
    seed_rows = joint.get("seeds") if isinstance(joint, Mapping) else None
    _require(
        isinstance(seed_rows, list)
        and len(seed_rows) == 3
        and {row.get("seed") for row in seed_rows if isinstance(row, Mapping)}
        == set(REQUIRED_SEEDS),
        "Surface attention-postcache joint seeds differ",
    )
    selected_by_seed: dict[int, dict[str, Any]] = {}
    for row in seed_rows:
        seed = int(row["seed"])
        checkpoint_record = row.get("checkpoint")
        _require(
            isinstance(checkpoint_record, Mapping)
            and set(checkpoint_record) == {"path", "sha256"},
            f"Surface attention seed-{seed} checkpoint binding differs",
        )
        checkpoint_path = _bound_file(
            checkpoint_record["path"],
            checkpoint_record["sha256"],
            f"Surface attention seed-{seed} checkpoint",
        )
        sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        sidecar = _json_object(sidecar_path)
        model, checkpoint, checkpoint_sha, _ = load_surface_region_summary_readout_v2(
            checkpoint_path,
            expected_sha256=str(checkpoint_record["sha256"]),
            map_location="cpu",
        )
        model.cpu().eval().requires_grad_(False)
        architecture = checkpoint.get("architecture")
        config = checkpoint.get("training_config")
        provenance = checkpoint.get("provenance")
        _require(
            isinstance(architecture, Mapping)
            and architecture.get("context_pooling_mode", "joint_attention_v1")
            == "joint_attention_v1"
            and isinstance(config, Mapping)
            and config.get("seed") == seed
            and config.get("context_pooling_mode") == "joint_attention_v1"
            and isinstance(provenance, Mapping)
            and provenance.get("uses_benchmark_scenes") is False
            and provenance.get("uses_benchmark_test_vocabulary") is False
            and provenance.get("train", {}).get("cache_paths")
            == [record["path"] for record in selected_caches["train"]]
            and provenance.get("validation", {}).get("cache_paths")
            == [record["path"] for record in selected_caches["validation"]],
            f"Surface attention seed-{seed} checkpoint contract differs",
        )
        recomputed = surface_finalizer._evaluate_readout_cpu(
            model,
            summary_head,
            validation_data,
            batch_size=int(config["batch_size"]),
        )
        reported = sidecar.get("validation")
        _require(
            isinstance(reported, Mapping)
            and all(
                _close(
                    recomputed[key],
                    reported.get(key),
                    tolerance=surface_finalizer.VALIDATION_RECOMPUTE_TOLERANCE,
                )
                for key in surface_finalizer.METRIC_KEYS
            )
            and sidecar.get("checkpoint_sha256") == checkpoint_sha
            and sidecar.get("best_epoch") == row.get("best_epoch"),
            f"Surface attention seed-{seed} CPU metrics differ",
        )
        baseline_model = surface_finalizer._untrained_baseline_model(
            feature_dim=int(architecture["feature_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            reliability_attention_mode=str(
                architecture.get("reliability_attention_mode", "log_prior")
            ),
            seed=seed,
        )
        baseline = surface_finalizer._evaluate_readout_cpu(
            baseline_model,
            summary_head,
            validation_data,
            batch_size=int(config["batch_size"]),
        )
        best_score = 0.5 * (
            recomputed["mean_descriptor_cosine"]
            + recomputed["all_view_descriptor_cosine"]
        )
        baseline_score = 0.5 * (
            baseline["mean_descriptor_cosine"]
            + baseline["all_view_descriptor_cosine"]
        )
        _require(
            _close(
                best_score,
                sidecar.get("best_selection_score"),
                tolerance=surface_finalizer.VALIDATION_RECOMPUTE_TOLERANCE,
            ),
            f"Surface attention seed-{seed} score differs",
        )
        selected_by_seed[seed] = {
            "candidate": "context_c1024_geometric",
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "sidecar": str(sidecar_path),
            "sidecar_sha256": _sha256(sidecar_path),
            "architecture_sha256": architecture["digest"],
            "best_epoch": int(row["best_epoch"]),
            "best_selection_score": float(best_score),
            "selection_score_delta": float(best_score - baseline_score),
            "validation": {
                key: float(recomputed[key]) for key in surface_finalizer.METRIC_KEYS
            },
            "metric_source": "cpu_recomputed_from_bound_validation_caches",
            "reported_best_selection_score": float(sidecar["best_selection_score"]),
            "reported_selection_score_delta": float(sidecar["selection_score_delta"]),
            "reported_validation": {
                key: float(reported[key]) for key in surface_finalizer.METRIC_KEYS
            },
            "recomputed_untrained_baseline": {
                key: float(baseline[key]) for key in surface_finalizer.METRIC_KEYS
            },
        }
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "completion_path": completion_path,
        "completion_sha256": _sha256(completion_path),
        "manifest": screen,
        "run_manifest": run_manifest,
        "surface_binding_paths": {
            "run_manifest": run_path,
            "cache_pairing": pairing_path,
            "attention_pooling_screen": manifest_path,
            "screen_completion": completion_path,
            "runtime_closure_final": Path(binding["runtime_closure_final"]["path"]),
        },
        "selected_candidate": "context_c1024_geometric",
        "selected_by_seed": selected_by_seed,
        "selected_caches": selected_caches,
        "distill_surface_promotion": binding,
    }


def _validate_surface_bundle(
    promotion_manifest: Path,
    promotion_completion: Path,
) -> dict[str, Any]:
    """Validate the frozen finalizer schema without mutating Surface outputs."""

    manifest_path = Path(promotion_manifest).resolve()
    completion_path = Path(promotion_completion).resolve()
    manifest = _json_object(manifest_path)
    if manifest.get("artifact_type") == (
        "surface_c1024_attention_pooling_postcache_continuation"
    ):
        return _validate_attention_surface_bundle(manifest_path, completion_path)
    completion = _json_object(completion_path)
    expected_manifest_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "selected_candidate",
        "seed_selection_policy",
        "required_seeds",
        "selected_readouts",
        "query_free_selection",
        "benchmark_gate",
        "bindings",
    }
    _require(
        set(manifest) == expected_manifest_keys
        and manifest.get("schema_version") == surface_finalizer.SCHEMA_VERSION
        and manifest.get("artifact_type") == surface_finalizer.ARTIFACT_TYPE
        and manifest.get("status")
        == "query_free_three_seed_bundle_frozen_benchmark_gate_closed",
        "invalid frozen Surface query-free promotion bundle",
    )
    _require(
        manifest.get("seed_selection_policy")
        == "all_required_seeds_no_single_seed_selection"
        and manifest.get("required_seeds") == list(REQUIRED_SEEDS),
        "Surface bundle does not freeze exactly seeds 0/1/2",
    )
    expected_benchmark_gate = {
        "status": "closed_not_evaluated",
        "text_response_gate": "required_before_benchmark_evaluation",
        "benchmark_queries_opened": False,
        "benchmark_masks_opened": False,
        "main_result_eligible": False,
    }
    _require(
        manifest.get("benchmark_gate") == expected_benchmark_gate,
        "Surface bundle benchmark gate must remain closed",
    )

    expected_completion_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "promotion_manifest",
        "promotion_manifest_sha256",
        "selected_candidate",
        "required_seeds",
        "benchmark_gate_status",
        "main_result_eligible",
        "finalizer_sha256",
    }
    _require(
        set(completion) == expected_completion_keys
        and completion.get("schema_version") == surface_finalizer.SCHEMA_VERSION
        and completion.get("artifact_type")
        == surface_finalizer.COMPLETION_ARTIFACT_TYPE
        and completion.get("status") == "complete_benchmark_gate_closed"
        and Path(str(completion.get("promotion_manifest", ""))).resolve()
        == manifest_path
        and completion.get("promotion_manifest_sha256") == _sha256(manifest_path)
        and completion.get("selected_candidate") == manifest["selected_candidate"]
        and completion.get("required_seeds") == list(REQUIRED_SEEDS)
        and completion.get("benchmark_gate_status") == "closed_not_evaluated"
        and completion.get("main_result_eligible") is False,
        "invalid Surface query-free promotion completion",
    )

    bindings = manifest.get("bindings")
    expected_binding_keys = {
        "finalizer",
        "run_manifest",
        "cache_pairing",
        "query_free_screen",
        "screen_completion",
        "caches",
        "all_compared_readouts",
    }
    _require(
        isinstance(bindings, Mapping) and set(bindings) == expected_binding_keys,
        "Surface bundle binding fields differ from the finalizer schema",
    )
    simple_paths = {
        name: _validate_record_file(bindings[name], f"Surface {name}")
        for name in (
            "finalizer",
            "run_manifest",
            "cache_pairing",
            "query_free_screen",
            "screen_completion",
        )
    }
    expected_finalizer = Path(surface_finalizer.__file__).resolve()
    _require(
        simple_paths["finalizer"] == expected_finalizer
        and completion.get("finalizer_sha256") == _sha256(expected_finalizer),
        "Surface bundle does not bind the current strict finalizer",
    )
    run_manifest = _json_object(simple_paths["run_manifest"])
    expected_cache_keys = {
        (candidate, role, shard)
        for candidate in surface_finalizer.EXPECTED_CANDIDATES
        for role, count in (
            ("train", int(run_manifest.get("cache_contract", {}).get("train_shards", -1))),
            (
                "validation",
                int(run_manifest.get("cache_contract", {}).get("validation_shards", -1)),
            ),
        )
        for shard in range(max(0, count))
    }
    expected_readout_keys = {
        (candidate, seed)
        for candidate in surface_finalizer.EXPECTED_CANDIDATES
        for seed in REQUIRED_SEEDS
    }
    observed_cache_keys = set()
    observed_readout_keys = set()
    authoritative_readouts: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in surface_finalizer.EXPECTED_CANDIDATES
    }
    for collection_name in ("caches", "all_compared_readouts"):
        collection = bindings[collection_name]
        _require(isinstance(collection, list) and collection, f"Surface {collection_name} is empty")
        for index, record in enumerate(collection):
            _require(isinstance(record, Mapping), f"Surface {collection_name}[{index}] is invalid")
            _bound_file(record.get("path") or record.get("checkpoint"), record.get("sha256") or record.get("checkpoint_sha256"), f"Surface {collection_name}[{index}]")
            if collection_name == "caches":
                key = (record.get("candidate"), record.get("role"), record.get("shard"))
                _require(key not in observed_cache_keys, "Surface cache bindings contain duplicates")
                observed_cache_keys.add(key)
                _bound_file(
                    record.get("sidecar"),
                    record.get("sidecar_sha256"),
                    f"Surface cache sidecar[{index}]",
                )
            else:
                key = (record.get("candidate"), record.get("seed"))
                _require(key not in observed_readout_keys, "Surface readout bindings contain duplicates")
                observed_readout_keys.add(key)
                _bound_file(
                    record.get("sidecar"),
                    record.get("sidecar_sha256"),
                    f"Surface readout sidecar[{index}]",
                )
                candidate = str(record.get("candidate", ""))
                _require(
                    candidate in authoritative_readouts,
                    "Surface readout names an unknown candidate",
                )
                _validate_authoritative_surface_metrics(
                    record, f"Surface {candidate} seed {record.get('seed')}"
                )
                authoritative_readouts[candidate].append(dict(record))
    _require(
        observed_cache_keys == expected_cache_keys,
        "Surface bundle does not bind the complete candidate/cache grid",
    )
    _require(
        observed_readout_keys == expected_readout_keys,
        "Surface bundle does not bind the complete candidate/seed grid",
    )

    selected = str(manifest.get("selected_candidate", ""))
    _require(selected in surface_finalizer.EXPECTED_CANDIDATES, "unknown selected Surface candidate")
    selected_readouts = manifest.get("selected_readouts")
    _require(isinstance(selected_readouts, list), "Surface bundle lacks selected readouts")
    selected_by_seed: dict[int, dict[str, Any]] = {}
    for raw in selected_readouts:
        _require(isinstance(raw, Mapping), "selected Surface readout must be an object")
        seed = raw.get("seed")
        _require(
            raw.get("candidate") == selected
            and isinstance(seed, int)
            and seed in REQUIRED_SEEDS
            and seed not in selected_by_seed,
            "selected Surface readouts are not exact candidate/seed pairs",
        )
        checkpoint = _bound_file(
            raw.get("checkpoint"), raw.get("checkpoint_sha256"), f"control seed {seed} checkpoint"
        )
        sidecar = _bound_file(
            raw.get("sidecar"), raw.get("sidecar_sha256"), f"control seed {seed} sidecar"
        )
        selected_by_seed[int(seed)] = {**dict(raw), "checkpoint": str(checkpoint), "sidecar": str(sidecar)}
    _require(set(selected_by_seed) == set(REQUIRED_SEEDS), "selected Surface readouts do not cover seeds 0/1/2")
    compared_selected = sorted(
        [
            dict(value)
            for value in bindings["all_compared_readouts"]
            if value.get("candidate") == selected
        ],
        key=lambda value: int(value["seed"]),
    )
    _require(
        compared_selected == [dict(selected_by_seed[seed]) for seed in REQUIRED_SEEDS],
        "selected Surface readouts differ from all-compared bindings",
    )

    screen = _json_object(simple_paths["query_free_screen"])
    selection = manifest.get("query_free_selection")
    expected_selection_keys = {
        "control",
        "contract",
        "metric_source",
        "cpu_gpu_metric_tolerance",
        "selected_candidate_metrics",
        "reported_metric_source",
        "reported_selected_candidate_metrics",
    }
    _require(
        isinstance(selection, Mapping) and set(selection) == expected_selection_keys,
        "Surface bundle lacks the exact query-free selection contract",
    )
    for candidate in authoritative_readouts:
        authoritative_readouts[candidate].sort(key=lambda row: int(row["seed"]))
    recomputed_rows, recomputed_selected = surface_finalizer._recompute_candidate_rows(
        authoritative_readouts,
        run_manifest.get("selection_contract", {}),
    )
    _require(
        screen.get("selected_candidate") == selected
        and recomputed_selected == selected
        and selection.get("control") == surface_finalizer.CONTROL
        and selection.get("contract") == run_manifest.get("selection_contract")
        and selection.get("metric_source")
        == "cpu_recomputed_from_bound_validation_caches"
        and selection.get("cpu_gpu_metric_tolerance")
        == surface_finalizer.VALIDATION_RECOMPUTE_TOLERANCE
        and selection.get("selected_candidate_metrics") == recomputed_rows[selected]
        and selection.get("reported_metric_source") == "bound_query_free_screen"
        and selection.get("reported_selected_candidate_metrics")
        == screen.get("candidates", {}).get(selected),
        "Surface selected candidate differs from its frozen query-free screen",
    )
    _require(
        screen.get("benchmark_queries_opened") is False
        and screen.get("benchmark_masks_opened") is False,
        "Surface query-free screen opened benchmark data",
    )
    selected_caches: dict[str, list[dict[str, str]]] = {}
    for role in ("train", "validation"):
        rows = sorted(
            (
                value
                for value in bindings["caches"]
                if value.get("candidate") == selected and value.get("role") == role
            ),
            key=lambda value: int(value["shard"]),
        )
        _require(rows, f"Surface bundle lacks selected {role} caches")
        selected_caches[role] = [
            {
                "path": str(Path(str(value["path"])).resolve()),
                "sha256": str(value["sha256"]),
            }
            for value in rows
        ]
    distill_surface_promotion = {
        "run_manifest": _file_record(simple_paths["run_manifest"]),
        "cache_pairing": _file_record(simple_paths["cache_pairing"]),
        "query_free_screen": _file_record(simple_paths["query_free_screen"]),
        "screen_completion": _file_record(simple_paths["screen_completion"]),
        "promotion_manifest": _file_record(manifest_path),
        "promotion_completion": _file_record(completion_path),
    }
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "completion_path": completion_path,
        "completion_sha256": _sha256(completion_path),
        "manifest": manifest,
        "run_manifest": run_manifest,
        "surface_binding_paths": simple_paths,
        "selected_candidate": selected,
        "selected_by_seed": selected_by_seed,
        "selected_caches": selected_caches,
        "distill_surface_promotion": distill_surface_promotion,
    }


def _validate_control_checkpoint(entry: Mapping[str, Any], seed: int) -> dict[str, Any]:
    checkpoint_path = Path(str(entry["checkpoint"])).resolve()
    sidecar_path = Path(str(entry["sidecar"])).resolve()
    expected_checkpoint_sha = entry.get("checkpoint_sha256")
    _require(
        _is_sha256(expected_checkpoint_sha),
        f"control seed {seed} checkpoint SHA binding is invalid",
    )
    model, checkpoint, checkpoint_sha, _ = load_surface_region_summary_readout_v2(
        checkpoint_path,
        expected_sha256=str(expected_checkpoint_sha),
        map_location="cpu",
    )
    model.cpu().eval().requires_grad_(False)
    sidecar = _json_object(sidecar_path)
    authoritative_validation = _validate_authoritative_surface_metrics(
        entry, f"control seed {seed}"
    )
    _require(
        checkpoint.get("training_config", {}).get("seed") == seed
        and checkpoint.get("provenance", {}).get("random_seed_contract", {}).get("seed") == seed,
        f"control seed {seed} checkpoint seed drift",
    )
    _require(
        checkpoint_sha == expected_checkpoint_sha
        and sidecar.get("checkpoint_sha256") == checkpoint_sha
        and sidecar.get("architecture") == checkpoint.get("architecture"),
        f"control seed {seed} sidecar drift after Surface finalization",
    )
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "sidecar": sidecar_path,
        "sidecar_sha256": _sha256(sidecar_path),
        "payload": checkpoint,
        "sidecar_payload": sidecar,
        "validation": authoritative_validation,
    }


def _validate_response_checkpoint(
    path: Path,
    *,
    seed: int,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    sidecar = _json_object(sidecar_path)
    expected_checkpoint_sha = sidecar.get("checkpoint_sha256")
    _require(
        _is_sha256(expected_checkpoint_sha),
        f"response seed {seed} sidecar checkpoint SHA is invalid",
    )
    model, checkpoint, checkpoint_sha, _ = load_surface_region_summary_readout_v2(
        checkpoint_path,
        expected_sha256=str(expected_checkpoint_sha),
        map_location="cpu",
    )
    model.cpu().eval().requires_grad_(False)
    _require(
        checkpoint.get("architecture") == control["payload"].get("architecture"),
        f"response seed {seed} changes the frozen inference architecture",
    )
    config = checkpoint.get("training_config")
    control_config = control["payload"].get("training_config")
    _require(isinstance(config, Mapping) and isinstance(control_config, Mapping), "checkpoint lacks training configuration")
    for field in COMMON_TRAINING_FIELDS:
        _require(
            config.get(field) == control_config.get(field),
            f"response seed {seed} changes paired training field {field}",
        )
    _require(config.get("seed") == seed, f"response seed {seed} training seed differs")
    provenance = checkpoint.get("provenance")
    control_provenance = control["payload"].get("provenance")
    _require(isinstance(provenance, Mapping) and isinstance(control_provenance, Mapping), "checkpoint lacks provenance")
    _require(
        provenance.get("uses_benchmark_scenes") is False
        and provenance.get("uses_benchmark_test_vocabulary") is False
        and provenance.get("scene_disjoint") is True
        and provenance.get("custom_text_projection") is False,
        f"response seed {seed} violates query-free provenance",
    )
    for field in ("train", "validation"):
        _require(
            _semantic_cache_provenance(
                provenance.get(field), f"response seed {seed} {field}"
            )
            == _semantic_cache_provenance(
                control_provenance.get(field), f"control seed {seed} {field}"
            ),
            f"response seed {seed} differs from control {field} provenance",
        )
        _cache_binding_records(
            provenance.get(field), f"response seed {seed} {field}"
        )
    for field in ("region_contract", "region_contract_sha256"):
        _require(
            provenance.get(field) == control_provenance.get(field),
            f"response seed {seed} differs from control {field} provenance",
        )
    train_cache_records = _cache_binding_records(
        provenance.get("train"), f"response seed {seed} train"
    )
    validation_cache_records = _cache_binding_records(
        provenance.get("validation"), f"response seed {seed} validation"
    )
    expected_surface_control = {
        "path": str(control["checkpoint"]),
        "sha256": control["checkpoint_sha256"],
        "seed": seed,
        "architecture": checkpoint["architecture"],
        "train_caches": train_cache_records,
        "validation_caches": validation_cache_records,
        "source_best_epoch": int(control["payload"]["best_epoch"]),
        "source_best_selection_score": float(
            control["payload"]["best_selection_score"]
        ),
    }
    run_binding = provenance.get("distill_run_manifest")
    _require(
        isinstance(run_binding, Mapping)
        and set(run_binding) == {"path", "sha256", "candidate"}
        and isinstance(run_binding.get("candidate"), str)
        and bool(run_binding.get("candidate")),
        f"response seed {seed} lacks an exact distill run-manifest binding",
    )
    run_manifest_path = _bound_file(
        run_binding["path"],
        run_binding["sha256"],
        f"response seed {seed} distill run manifest",
    )
    distill_manifest = _json_object(run_manifest_path)
    distill_schema = int(distill_manifest.get("schema_version", -1))
    _require(
        distill_schema in {1, 2, 3},
        f"response seed {seed} uses an unsupported distill manifest schema",
    )
    response_protocol = _response_protocol_for_manifest(
        run_manifest_path,
        _sha256(run_manifest_path),
        distill_manifest,
    )
    selection_contract = (
        ACCEPTED_ANCHOR_DISTILL_EPOCH_SELECTION
        if response_protocol == "accepted_anchor_selection_v4"
        else DISTILL_EPOCH_SELECTION
        if response_protocol == "pairwise_selection_v3"
        else LEGACY_DISTILL_EPOCH_SELECTION
        if response_protocol == "legacy_brier_selection_v2"
        else None
    )
    warm_start = provenance.get("surface_control_warm_start") is not None
    if warm_start:
        _require(
            provenance.get("random_seed_contract")
            == {
                "seed": seed,
                "model_initialization": False,
                "model_initialization_source": "frozen_seed_surface_control",
                "data_order": True,
                "canonical_noise": True,
            }
            and checkpoint.get("surface_control_checkpoint")
            == expected_surface_control
            and Path(str(config.get("surface_control_checkpoint", ""))).resolve()
            == control["checkpoint"]
            and config.get("surface_control_checkpoint_sha256")
            == control["checkpoint_sha256"]
            and provenance.get("surface_control_warm_start")
            == {
                **expected_surface_control,
                "epoch": 0,
                "noninferiority_metrics": [
                    "summary_token_cosine",
                    "mean_descriptor_cosine",
                    "all_view_descriptor_cosine",
                ],
                "noninferiority_tolerance": SURFACE_NONINFERIORITY_TOLERANCE,
                "selection_policy": selection_contract,
            },
            f"response seed {seed} Surface warm-start contract differs",
        )
    else:
        _require(
            provenance.get("random_seed_contract")
            == {
                "seed": seed,
                "model_initialization": True,
                "data_order": True,
                "canonical_noise": True,
            },
            f"response seed {seed} random-seed contract differs",
        )
    response_contract = provenance.get("text_response_distillation")
    _require(isinstance(response_contract, Mapping), "response provenance differs")
    v3_response = "response_lambdas" in response_contract
    expected_response_fields = (
        (
            {
            "fit_split_only", "benchmark_vocabulary_opened", "fit_text_bank",
            "calibration_manifest", "calibration_manifest_sha256",
            "calibration_seed", "response_lambdas",
            "response_branch_gradient_target_ratio",
            "total_response_gradient_ratio_upper_bound", "losses",
            "complete_scene_batching", "design_diagnostic",
            }
            | (
                {"scene_response_objective"}
                if response_protocol
                in {"pairwise_selection_v3", "accepted_anchor_selection_v4"}
                else set()
            )
        )
        if v3_response
        else {
            "fit_split_only", "benchmark_vocabulary_opened", "fit_text_bank",
            "calibration_manifest", "calibration_manifest_sha256",
            "response_lambda", "shared_training_seeds", "loss",
        }
    )
    _require(
        set(response_contract) == expected_response_fields
        and response_contract.get("fit_split_only") is True
        and response_contract.get("benchmark_vocabulary_opened") is False,
        f"response seed {seed} text-response contract differs",
    )
    calibration_path = _bound_file(
        response_contract.get("calibration_manifest"),
        response_contract.get("calibration_manifest_sha256"),
        f"response seed {seed} calibration manifest",
    )
    fit_text_bank = _validate_fit_text_bank(
        response_contract.get("fit_text_bank"), f"response seed {seed}"
    )
    if v3_response:
        response_lambdas = response_contract.get("response_lambdas")
        expected_scene_loss = (
            SCENE_RESPONSE_LOSS
            if response_protocol
            in {"pairwise_selection_v3", "accepted_anchor_selection_v4"}
            else LEGACY_BRIER_RESPONSE_LOSS
        )
        _require(
            response_contract.get("calibration_seed") == seed
            and isinstance(response_lambdas, Mapping)
            and set(response_lambdas) == {
                "independent_response", "scene_response"
            }
            and all(_finite(value, "response lambda") > 0 for value in response_lambdas.values())
            and response_contract.get("response_branch_gradient_target_ratio") == 0.25
            and response_contract.get("total_response_gradient_ratio_upper_bound") == 0.5
            and response_contract.get("losses")
            == [
                "independent_normalized_cosine_response_smooth_l1",
                expected_scene_loss,
            ]
            and (
                response_contract.get("scene_response_objective")
                == SCENE_RESPONSE_OBJECTIVE
                if response_protocol
                in {"pairwise_selection_v3", "accepted_anchor_selection_v4"}
                else "scene_response_objective" not in response_contract
            )
            and response_contract.get("complete_scene_batching") is True,
            f"response seed {seed} dual response budget differs",
        )
        response_lambdas = dict(response_lambdas)
        response_lambda = None
        design_diagnostic = response_contract.get("design_diagnostic")
        _require(
            isinstance(design_diagnostic, Mapping)
            and set(design_diagnostic)
            == {
                "path", "sha256", "role", "measured_seed",
                "calibration_reuses_measured_values",
                "diagnostic_surface_control",
            }
            and design_diagnostic.get("role")
            == "seed0_design_prior_only_per_seed_values_remeasured"
            and design_diagnostic.get("measured_seed") == 0
            and design_diagnostic.get("calibration_reuses_measured_values") is False,
            f"response seed {seed} design diagnostic differs",
        )
        _bound_file(
            design_diagnostic["path"],
            design_diagnostic["sha256"],
            f"response seed {seed} design diagnostic",
        )
        _validate_calibration_manifest(
            calibration_path,
            seed=seed,
            response_lambdas=response_lambdas,
            surface_control=expected_surface_control,
            design_diagnostic=design_diagnostic,
            fit_text_bank=fit_text_bank,
            response_protocol=response_protocol,
            token_weight=_finite(config.get("token_weight"), "token weight"),
            relation_weight=_finite(
                config.get("relation_weight"), "relation weight"
            ),
            label=f"response seed {seed}",
        )
    else:
        response_lambda = _finite(
            response_contract.get("response_lambda"), "response lambda"
        )
        response_lambdas = None
        _require(response_lambda > 0.0, "response lambda must be positive")
        _validate_legacy_calibration_manifest(
            calibration_path,
            response_lambda=response_lambda,
            fit_text_bank=fit_text_bank,
            label=f"response seed {seed}",
        )
    _require(
        (distill_schema in {2, 3} and warm_start)
        or (distill_schema == 1 and not warm_start),
        f"response seed {seed} initialization does not match its authority schema",
    )

    history = checkpoint.get("history")
    _require(isinstance(history, list) and history, f"response seed {seed} lacks history")
    _require(v3_response is (distill_schema == 3), "response schema/calibration mismatch")
    if distill_schema in {2, 3}:
        for index, row in enumerate(history):
            for field in (
                "selection_score",
                "surface_selection_score",
                "text_support_top1_agreement",
                "text_response_smooth_l1",
                "descriptor_relation_smooth_l1",
                *surface_finalizer.METRIC_KEYS,
            ):
                _finite(row.get(field), f"response seed {seed} history {field}")
            if index == 0:
                _require(
                    row.get("initialization")
                    == "frozen_surface_control_checkpoint",
                    f"response seed {seed} history lacks control epoch 0",
                )
            else:
                loss_fields = (
                    (
                        "loss", "token_loss", "descriptor_loss", "relation_loss",
                        "independent_response_loss", "scene_response_loss",
                        "scene_profile_loss", "scene_ranking_loss",
                    )
                    if distill_schema == 3
                    else (
                        "loss", "token_loss", "descriptor_loss", "relation_loss",
                        "response_loss",
                    )
                )
                for field in loss_fields:
                    _finite(row.get(field), f"response seed {seed} history {field}")
                if distill_schema == 3:
                    _require(
                        isinstance(row.get("complete_scene_batch_count"), int)
                        and row["complete_scene_batch_count"] > 0
                        and isinstance(row.get("max_complete_scene_batch_rows"), int)
                        and 1 < row["max_complete_scene_batch_rows"] <= 64,
                        "history complete-scene batching drift",
                    )
            if distill_schema == 3:
                _require(
                    row.get("response_lambdas") == response_lambdas
                    and (
                        row.get("scene_response_objective")
                        == SCENE_RESPONSE_OBJECTIVE
                        if response_protocol
                        in {
                            "pairwise_selection_v3",
                            "accepted_anchor_selection_v4",
                        }
                        else "scene_response_objective" not in row
                    ),
                    "history response lambdas drift",
                )
            else:
                _require(
                    _close(row["response_lambda"], response_lambda),
                    "history response lambda drift",
                )
    else:
        for row in history:
            for field in (
                "loss",
                "token_loss",
                "descriptor_loss",
                "relation_loss",
                "response_loss",
                "response_lambda",
                "selection_score",
                *surface_finalizer.METRIC_KEYS,
            ):
                _finite(row.get(field), f"response seed {seed} history {field}")
            _require(
                _close(row["response_lambda"], response_lambda),
                "history response lambda drift",
            )
    if distill_schema in {2, 3}:
        _require(
            distill_manifest.get("training_contract", {}).get("epoch_selection")
            == selection_contract,
            f"response seed {seed} authority selection contract differs",
        )
        if response_protocol in {
            "pairwise_selection_v3",
            "accepted_anchor_selection_v4",
        }:
            best_epoch, best_score = _recompute_response_primary_selection(history)
        else:
            best_epoch, best_score = _recompute_legacy_response_primary_selection(
                history
            )
    else:
        # Schema-1 artifacts remain readable, but never acquire newer
        # authority retroactively.
        best_score = max(float(row["selection_score"]) for row in history)
        best_epoch = next(
            int(row["epoch"])
            for row in history
            if float(row["selection_score"]) == best_score
        )
    validation = sidecar.get("validation")
    _require(isinstance(validation, Mapping), f"response seed {seed} lacks validation metrics")
    for metric in surface_finalizer.METRIC_KEYS:
        _finite(validation.get(metric), f"response seed {seed} validation {metric}")
    validation_score = 0.5 * (
        float(validation["mean_descriptor_cosine"])
        + float(validation["all_view_descriptor_cosine"])
    )
    surface_control_validation = (
        checkpoint.get("surface_control_validation")
        if distill_schema in {2, 3}
        else checkpoint.get("untrained_baseline")
    )
    _require(
        isinstance(surface_control_validation, Mapping)
        and set(surface_control_validation) == set(surface_finalizer.METRIC_KEYS)
        and (
            distill_schema == 1
            or all(
                _close(surface_control_validation[field], history[0][field])
                for field in surface_finalizer.METRIC_KEYS
            )
        ),
        f"response seed {seed} Surface control validation differs",
    )
    surface_control_score = 0.5 * (
        float(surface_control_validation["mean_descriptor_cosine"])
        + float(surface_control_validation["all_view_descriptor_cosine"])
    )
    state_machine = None
    if response_protocol == "accepted_anchor_selection_v4":
        patience = config.get("patience")
        _require(
            isinstance(patience, int)
            and not isinstance(patience, bool)
            and patience >= 0,
            f"response seed {seed} accepted-anchor patience differs",
        )
        state_machine = _validate_accepted_anchor_checkpoint_state(
            checkpoint,
            control_payload=control["payload"],
            patience=patience,
        )
        _require(
            state_machine["best_state_dict_sha256"]
            == checkpoint.get("best_state_dict_sha256"),
            f"response seed {seed} accepted-anchor best state differs",
        )
    elif any(
        field in checkpoint
        for field in (
            "best_state_dict_sha256",
            "proposal_state_machine",
            "accepted_anchor",
            "history_hash_chain_sha256",
        )
    ) or "proposal_state_machine" in provenance or "proposal_state_machine" in config:
        raise ValueError(
            f"response seed {seed} non-v4 checkpoint contains accepted-anchor fields"
        )
    _require(
        sidecar.get("checkpoint_sha256") == checkpoint_sha
        and Path(str(sidecar.get("output", ""))).resolve() == checkpoint_path
        and sidecar.get("architecture") == checkpoint.get("architecture")
        and sidecar.get("best_epoch") == best_epoch
        and _close(sidecar.get("best_selection_score"), best_score)
        and _close(validation_score, best_score, tolerance=1e-6)
        and checkpoint.get("best_epoch") == best_epoch
        and _close(checkpoint.get("best_selection_score"), best_score)
        and (
            sidecar.get("response_lambdas") == response_lambdas
            if distill_schema == 3
            else _close(sidecar.get("response_lambda"), response_lambda)
        )
        and Path(str(sidecar.get("calibration_manifest", ""))).resolve()
        == calibration_path
        and sidecar.get("calibration_manifest_sha256") == _sha256(calibration_path),
        f"response seed {seed} checkpoint/sidecar selection drift",
    )
    expected_sidecar_keys = {
        "output",
        "checkpoint_sha256",
        "architecture",
        "best_epoch",
        "best_selection_score",
        "selection_score_delta",
        "validation",
        "calibration_manifest",
        "calibration_manifest_sha256",
        "fit_text_bank_sha256",
        "fit_query_count",
        "distill_run_manifest",
        "distill_run_manifest_sha256",
        "validation_caches",
        "train_scenes",
        "validation_scenes",
        "scene_overlap",
    }
    if distill_schema == 3:
        expected_sidecar_keys.update(
            {"response_lambdas", "complete_scene_batching"}
        )
        if response_protocol == "accepted_anchor_selection_v4":
            expected_sidecar_keys.update(
                {
                    "best_state_dict_sha256",
                    "proposal_state_machine",
                    "accepted_anchor",
                    "history_hash_chain_sha256",
                }
            )
    else:
        expected_sidecar_keys.add("response_lambda")
    if distill_schema in {2, 3}:
        expected_sidecar_keys.update(
            {
                "surface_control_checkpoint",
                "surface_control_validation",
                "surface_control_score",
            }
        )
        control_sidecar_valid = (
            sidecar.get("surface_control_checkpoint") == expected_surface_control
            and sidecar.get("surface_control_validation")
            == dict(surface_control_validation)
            and _close(
                sidecar.get("surface_control_score"), surface_control_score
            )
            and _close(
                checkpoint.get("surface_control_score"), surface_control_score
            )
            and (
                distill_schema != 3
                or sidecar.get("complete_scene_batching")
                == checkpoint.get("complete_scene_batching")
            )
        )
    else:
        expected_sidecar_keys.add("untrained_baseline")
        control_sidecar_valid = (
            sidecar.get("untrained_baseline")
            == checkpoint.get("untrained_baseline")
            and _close(
                checkpoint.get("untrained_baseline_score"),
                surface_control_score,
            )
        )
    _require(
        set(sidecar) == expected_sidecar_keys
        and control_sidecar_valid
        and _close(
            sidecar.get("selection_score_delta"),
            best_score - surface_control_score,
        )
        and sidecar.get("fit_text_bank_sha256")
        == fit_text_bank["artifact_sha256"]
        and sidecar.get("fit_query_count") == fit_text_bank["query_count"]
        and Path(str(sidecar.get("distill_run_manifest", ""))).resolve()
        == run_manifest_path
        and sidecar.get("distill_run_manifest_sha256") == _sha256(run_manifest_path)
        and sidecar.get("validation_caches")
        == _cache_binding_records(
            provenance.get("validation"), f"response seed {seed} validation"
        )
        and sidecar.get("train_scenes")
        == len(provenance.get("train", {}).get("scenes", []))
        and sidecar.get("validation_scenes")
        == len(provenance.get("validation", {}).get("scenes", []))
        and sidecar.get("scene_overlap") == [],
        f"response seed {seed} sidecar immutable bindings differ",
    )
    if state_machine is not None:
        _require(
            all(sidecar.get(field) == value for field, value in state_machine.items()),
            f"response seed {seed} accepted-anchor sidecar provenance differs",
        )
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "architecture_sha256": checkpoint["architecture"]["digest"],
        "validation": {
            key: float(validation[key]) for key in surface_finalizer.METRIC_KEYS
        },
        "calibration_manifest": str(calibration_path),
        "calibration_manifest_sha256": _sha256(calibration_path),
        "response_lambda": response_lambda,
        "response_lambdas": response_lambdas,
        "design_diagnostic": (
            dict(response_contract["design_diagnostic"])
            if distill_schema == 3
            else None
        ),
        "fit_text_bank": fit_text_bank,
        "distill_run_manifest": {
            "path": str(run_manifest_path),
            "sha256": _sha256(run_manifest_path),
            "candidate": str(run_binding["candidate"]),
        },
        "distill_schema_version": distill_schema,
        "response_protocol": response_protocol,
        "proposal_state_machine": (
            dict(PROPOSAL_STATE_MACHINE) if state_machine is not None else None
        ),
        "surface_control": (
            expected_surface_control if distill_schema in {2, 3} else None
        ),
        "train_caches": train_cache_records,
        "validation_caches": validation_cache_records,
        "payload": checkpoint,
    }


def _validate_shared_distill_run(
    *,
    surface: Mapping[str, Any],
    responses: Mapping[int, Mapping[str, Any]],
    radio_path: Path,
) -> dict[str, Any]:
    bindings = [responses[seed]["distill_run_manifest"] for seed in REQUIRED_SEEDS]
    _require(
        all(binding == bindings[0] for binding in bindings[1:]),
        "response seeds do not share one distill run manifest",
    )
    binding = bindings[0]
    _require(
        binding["candidate"] == surface["selected_candidate"],
        "distill run candidate differs from the frozen Surface selection",
    )
    manifest_path = _bound_file(
        binding["path"], binding["sha256"], "shared distill run manifest"
    )
    manifest = _json_object(manifest_path)
    schema = int(manifest.get("schema_version", -1))
    response_protocol = _response_protocol_for_manifest(
        manifest_path,
        _sha256(manifest_path),
        manifest,
    )
    _require(
        schema in {1, 2, 3}
        and all(
            responses[seed].get("distill_schema_version") == schema
            for seed in REQUIRED_SEEDS
        )
        and all(
            responses[seed].get("response_protocol") == response_protocol
            for seed in REQUIRED_SEEDS
        ),
        "response checkpoints disagree on the distill authority schema",
    )
    selection_contract = (
        ACCEPTED_ANCHOR_DISTILL_EPOCH_SELECTION
        if response_protocol == "accepted_anchor_selection_v4"
        else DISTILL_EPOCH_SELECTION
        if response_protocol == "pairwise_selection_v3"
        else LEGACY_DISTILL_EPOCH_SELECTION
        if response_protocol == "legacy_brier_selection_v2"
        else None
    )
    legacy_keys = {
        "schema_version",
        "artifact_type",
        "candidate",
        "surface_promotion",
        "train_caches",
        "validation_caches",
        "fit_text_bank",
        "radio_checkpoint",
        "calibration_manifest",
        "outputs",
        "training_contract",
        "thermal_safety_contract",
        "implementation_sources",
    }
    authority_keys = legacy_keys | {
        "authority_status",
        "calibration_audit",
        "initial_gpu_preflight",
        "gpu_identity",
        "runtime_closure",
        "authority_contract",
        "training_command_contract",
    }
    authority_v3_keys = (authority_keys - {
        "calibration_manifest", "calibration_audit"
    }) | {"calibrations", "gradient_design_diagnostic"}
    _require(
        set(manifest)
        == (
            legacy_keys
            if schema == 1
            else authority_keys if schema == 2 else authority_v3_keys
        )
        and manifest.get("artifact_type")
        == "surface_region_text_response_distill_run"
        and manifest.get("candidate") == surface["selected_candidate"],
        "invalid shared distill run-manifest schema",
    )
    expected_surface_promotion = surface["distill_surface_promotion"]
    _require(
        manifest.get("surface_promotion") == expected_surface_promotion,
        "distill run binds different Surface selection artifacts",
    )
    _require(
        manifest.get("train_caches") == surface["selected_caches"]["train"]
        and manifest.get("validation_caches")
        == surface["selected_caches"]["validation"],
        "distill run caches differ from the frozen selected candidate",
    )
    for seed in REQUIRED_SEEDS:
        _require(
            responses[seed]["train_caches"] == manifest["train_caches"]
            and responses[seed]["validation_caches"]
            == manifest["validation_caches"],
            f"response seed {seed} cache bindings differ from the distill run",
        )
        if schema in {2, 3}:
            selected_control = surface["selected_by_seed"][seed]
            _require(
                responses[seed]["surface_control"]["path"]
                == str(Path(str(selected_control["checkpoint"])).resolve())
                and responses[seed]["surface_control"]["sha256"]
                == selected_control["checkpoint_sha256"],
                f"response seed {seed} warm start differs from the selected Surface control",
            )

    shared_fit_banks = [responses[seed]["fit_text_bank"] for seed in REQUIRED_SEEDS]
    _require(
        all(value == shared_fit_banks[0] for value in shared_fit_banks[1:]),
        "response seeds do not share one fit text bank",
    )
    fit_bank = shared_fit_banks[0]
    expected_fit_binding = (
        {
            "artifact_path": fit_bank["artifact_path"],
            "artifact_sha256": fit_bank["artifact_sha256"],
            "manifest_path": fit_bank["manifest_path"],
            "manifest_sha256": fit_bank["manifest_sha256"],
        }
        if schema == 1
        else {
            "artifact": {
                "path": fit_bank["artifact_path"],
                "sha256": fit_bank["artifact_sha256"],
            },
            "manifest": {
                "path": fit_bank["manifest_path"],
                "sha256": fit_bank["manifest_sha256"],
            },
        }
    )
    _require(
        manifest.get("fit_text_bank") == expected_fit_binding,
        "distill run fit text-bank binding differs",
    )
    _require(
        manifest.get("radio_checkpoint") == _file_record(radio_path),
        "distill run RADIO checkpoint binding differs",
    )
    if schema == 3:
        calibration_rows = manifest.get("calibrations")
        _require(
            isinstance(calibration_rows, list)
            and len(calibration_rows) == len(REQUIRED_SEEDS),
            "distill per-seed calibration index differs",
        )
        calibration_by_seed = {
            row.get("seed"): row
            for row in calibration_rows
            if isinstance(row, Mapping)
        }
        _require(
            set(calibration_by_seed) == set(REQUIRED_SEEDS),
            "distill per-seed calibration seeds differ",
        )
        for seed in REQUIRED_SEEDS:
            row = calibration_by_seed[seed]
            _require(
                set(row) == {
                    "seed", "manifest", "audit", "surface_control",
                    "response_lambdas",
                }
                and row["manifest"]
                == {
                    "path": responses[seed]["calibration_manifest"],
                    "sha256": responses[seed]["calibration_manifest_sha256"],
                }
                and row["surface_control"] == responses[seed]["surface_control"]
                and row["response_lambdas"] == responses[seed]["response_lambdas"],
                f"distill seed-{seed} calibration binding differs",
            )
            _validate_record_file(row["audit"], f"distill seed-{seed} calibration audit")
        diagnostics = [responses[seed]["design_diagnostic"] for seed in REQUIRED_SEEDS]
        _require(
            all(value == diagnostics[0] for value in diagnostics[1:])
            and manifest.get("gradient_design_diagnostic")
            == {key: diagnostics[0][key] for key in ("path", "sha256")},
            "distill gradient design diagnostic binding differs",
        )
        calibration_path = None
    else:
        calibration_paths = {
            row["calibration_manifest"] for row in responses.values()
        }
        calibration_path = Path(next(iter(calibration_paths))).resolve()
        expected_calibration = (
            str(calibration_path) if schema == 1 else _file_record(calibration_path)
        )
        _require(
            len(calibration_paths) == 1
            and manifest.get("calibration_manifest") == expected_calibration,
            "distill run calibration binding differs",
        )
    expected_output_pairs = [
        {
            "seed": seed,
            "checkpoint": responses[seed]["checkpoint"],
            "report": responses[seed]["sidecar"],
        }
        for seed in REQUIRED_SEEDS
    ]
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, list), "distill run output index differs")
    if schema == 1:
        _require(
            outputs == expected_output_pairs,
            "distill run output index differs from the paired three-seed bundle",
        )
    else:
        expected_output_fields = {
            "seed",
            "checkpoint",
            "report",
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
        _require(
            len(outputs) == len(REQUIRED_SEEDS)
            and all(isinstance(row, Mapping) and set(row) == expected_output_fields for row in outputs),
            "authority distill output index fields differ",
        )
        output_by_seed = {int(row["seed"]): row for row in outputs}
        _require(set(output_by_seed) == set(REQUIRED_SEEDS), "authority distill output seeds differ")
        for expected in expected_output_pairs:
            row = output_by_seed[int(expected["seed"])]
            _require(
                row["checkpoint"] == expected["checkpoint"]
                and row["report"] == expected["report"],
                "authority distill output checkpoint/report differs",
            )

    first_config = responses[REQUIRED_SEEDS[0]]["payload"]["training_config"]
    expected_training_contract = {
        field: first_config.get(field)
        for field in COMMON_TRAINING_FIELDS
        if field != "seed"
    }
    if schema == 3:
        response_training_contract = {
                "seeds": list(REQUIRED_SEEDS),
                "response_lambda_source": (
                    "per_seed_exact_surface_warmstart_gradient_budget"
                ),
                "response_branch_gradient_target_ratio": 0.25,
                "total_response_gradient_ratio_upper_bound": 0.5,
                "response_gradient_bound_scope": (
                    "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
                ),
                "response_losses": [
                    "independent_normalized_cosine_response_smooth_l1",
                    (
                        SCENE_RESPONSE_LOSS
                        if response_protocol
                        in {
                            "pairwise_selection_v3",
                            "accepted_anchor_selection_v4",
                        }
                        else LEGACY_BRIER_RESPONSE_LOSS
                    ),
                ],
                "scene_tie_tolerance": 1e-6,
                "training_batching": (
                    "shuffle_complete_scene_groups_no_partial_scenes_v1"
                ),
                "max_complete_scene_batch_rows": 64,
            }
        response_training_contract.update(
            {
                "scene_response_objective": dict(SCENE_RESPONSE_OBJECTIVE),
            }
            if response_protocol
            in {"pairwise_selection_v3", "accepted_anchor_selection_v4"}
            else {
                "scene_profile_weight": 1.0,
                "scene_ranking_weight": 1.0,
                "scene_ranking_temperature": 0.1,
            }
        )
        if response_protocol == "accepted_anchor_selection_v4":
            _require(
                all(
                    responses[seed].get("proposal_state_machine")
                    == PROPOSAL_STATE_MACHINE
                    and responses[seed]["payload"].get("training_config", {}).get(
                        "proposal_state_machine"
                    )
                    == PROPOSAL_STATE_MACHINE
                    for seed in REQUIRED_SEEDS
                ),
                "response checkpoints disagree on the accepted-anchor protocol",
            )
            response_training_contract["proposal_state_machine"] = dict(
                PROPOSAL_STATE_MACHINE
            )
        expected_training_contract.update(response_training_contract)
    else:
        expected_training_contract.update(
            {
                "seeds": list(REQUIRED_SEEDS),
                "response_lambda_source": "one_shot_initial_gradient_ratio",
                "response_loss": "independent_normalized_cosine_response_smooth_l1",
            }
        )
    if schema in {2, 3}:
        expected_training_contract.update(
            {
                "epoch_selection": selection_contract,
                "surface_control_initialization": (
                    "exact_seed_checkpoint_state_dict"
                ),
                "surface_control_noninferiority_tolerance": (
                    SURFACE_NONINFERIORITY_TOLERANCE
                ),
            }
        )
    _require(
        manifest.get("training_contract") == expected_training_contract,
        "distill run training contract differs from paired checkpoints",
    )
    thermal = _validate_distill_thermal_contract(
        manifest.get("thermal_safety_contract"),
        schema=schema,
    )
    if schema == 1:
        _bound_file(
            thermal.get("guard"), thermal.get("guard_sha256"), "distill thermal guard"
        )
        _validate_current_source_hashes(
            manifest.get("implementation_sources"),
            required=DISTILL_IMPLEMENTATION_SOURCES,
            label="legacy distill run",
        )
    else:
        from radio_gs.scripts import surface_text_response_distill_authority as distill_authority

        _validate_record_file(thermal.get("guard"), "distill thermal guard")
        _require(
            manifest.get("authority_status")
            == "query_free_three_seed_gpu1_run_frozen"
            and manifest.get("training_contract", {}).get("epoch_selection")
            == selection_contract
            and thermal.get("peer_activity_action") in {"pause", "terminate"}
            and thermal.get("owner_pid_namespace_mode")
            == "exclusive-singleton-after-clear-v1",
            "authority distill method/thermal contract differs",
        )
        authority_contract = manifest.get("authority_contract")
        _require(
            isinstance(authority_contract, Mapping)
            and authority_contract.get("seed_resume")
            == "skip_only_exact_guarded_terminal_v1"
            and authority_contract.get("closure_verification")
            == "before_and_after_every_seed_v1"
            and authority_contract.get("global_gpu_lock")
            == "/root/RADIO-GS/output/.physical_gpu1.lock"
            and authority_contract.get("main_output_root")
            == "/root/RADIO-GS/output"
            and authority_contract.get("global_gpu_kernel_singleton_protocol")
            == "linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
            and authority_contract.get(
                "global_gpu_kernel_singleton_inherited_fd_verified"
            )
            is True,
            "authority distill execution contract differs",
        )
        manifest_sha256 = _sha256(manifest_path)
        registered_closure = FORMAL_RECORDED_SOURCE_CLOSURE_AUTHORITIES.get(
            manifest_sha256
        )
        if registered_closure is not None:
            _require(
                schema == 3
                and manifest_path
                == Path(registered_closure["manifest_path"]).resolve()
                and binding.get("sha256") == manifest_sha256
                and authority_contract.get("source_snapshot_root")
                == registered_closure["source_root"],
                "registered distill authority identity differs",
            )
            source_snapshot_root = Path(registered_closure["source_root"])
            closure_digest = _validate_registered_recorded_runtime_closure(
                manifest.get("runtime_closure"),
                source_root=source_snapshot_root,
                implementation_sources=manifest.get("implementation_sources"),
                expected_digest=registered_closure["runtime_closure_digest"],
            )
        else:
            source_snapshot_root = _validate_snapshot_source_hashes(
                manifest.get("implementation_sources"),
                required=AUTHORITY_DISTILL_IMPLEMENTATION_SOURCES,
                source_snapshot_root=authority_contract.get("source_snapshot_root"),
                label="authority distill run",
            )
            closure_digest = _validate_distill_runtime_closure(
                manifest.get("runtime_closure"),
                source_snapshot_root=source_snapshot_root,
            )
        initial_path = _validate_record_file(
            manifest.get("initial_gpu_preflight"),
            "initial distill GPU preflight",
        )
        initial = _json_object(initial_path)
        _require(
            manifest.get("gpu_identity")
            == {
                "physical_index": 1,
                "uuid": initial.get("gpu_identity", {}).get("uuid"),
                "pci_bus_id": initial.get("gpu_identity", {}).get("pci_bus_id"),
            }
            and initial.get("status")
            == "physical_gpu1_idle_and_pcie_responsive"
            and initial.get("compute_owners") == [],
            "authority distill GPU1 identity/preflight differs",
        )
        if schema == 2:
            _validate_record_file(
                manifest.get("calibration_audit"),
                "distill calibration CPU audit",
            )
        distill_authority.validate_training_command_contract(
            manifest,
            manifest_path=manifest_path,
        )

    completion_path = manifest_path.parent / "text_response_distill.complete"
    completion = _json_object(completion_path)
    legacy_completion_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "candidate",
        "run_manifest",
        "run_manifest_sha256",
        "calibration_manifest",
        "calibration_manifest_sha256",
        "seeds",
    }
    expected_legacy_seeds = [
        {
            "seed": seed,
            "checkpoint": responses[seed]["checkpoint"],
            "checkpoint_sha256": responses[seed]["checkpoint_sha256"],
            "report": responses[seed]["sidecar"],
            "report_sha256": responses[seed]["sidecar_sha256"],
        }
        for seed in REQUIRED_SEEDS
    ]
    if schema == 1:
        _require(
            set(completion) == legacy_completion_keys
            and completion.get("schema_version") == 1
            and completion.get("artifact_type")
            == "surface_region_text_response_distill_completion"
            and completion.get("status") == "complete"
            and completion.get("candidate") == surface["selected_candidate"]
            and Path(str(completion.get("run_manifest", ""))).resolve()
            == manifest_path
            and completion.get("run_manifest_sha256") == _sha256(manifest_path)
            and Path(str(completion.get("calibration_manifest", ""))).resolve()
            == calibration_path
            and completion.get("calibration_manifest_sha256")
            == _sha256(calibration_path)
            and completion.get("seeds") == expected_legacy_seeds,
            "legacy distill completion does not freeze the exact three-seed outputs",
        )
    else:
        authority_completion_keys = {
            "schema_version",
            "artifact_type",
            "status",
            "candidate",
            "run_manifest",
            "calibration_manifest",
            "runtime_closure_digest",
            "selection_contract",
            "seeds",
        }
        if schema == 3:
            authority_completion_keys.remove("calibration_manifest")
            authority_completion_keys.update(
                {"calibrations", "gradient_design_diagnostic"}
            )
        output_by_seed = {int(row["seed"]): row for row in outputs}
        expected_authority_seeds = []
        for seed in REQUIRED_SEEDS:
            output_row = output_by_seed[seed]
            strict_terminal = distill_authority.validate_seed_terminal(
                manifest_path,
                seed,
            )
            receipt = validate_receipt(output_row["guard_receipt"])
            terminal_path = _validate_record_file(
                _file_record(Path(output_row["terminal"])),
                f"distill seed {seed} terminal",
            )
            terminal = _json_object(terminal_path)
            evidence = terminal.get("evidence")
            _validate_authority_seed_evidence(
                evidence,
                seed=seed,
                gpu_identity=manifest.get("gpu_identity"),
            )
            _require(
                receipt["payload"].get("seed") == seed
                and strict_terminal.get("terminal") == _file_record(terminal_path)
                and receipt["payload"].get("gpu_identity")
                == manifest.get("gpu_identity")
                and terminal.get("status")
                == "complete_guarded_audited_no_xid_pcie_fault"
                and terminal.get("seed") == seed
                and terminal.get("candidate") == surface["selected_candidate"]
                and terminal.get("runtime_closure_digest") == closure_digest
                and evidence.get("selection_contract")
                == selection_contract
                and evidence.get("checkpoint")
                == {
                    "path": responses[seed]["checkpoint"],
                    "sha256": responses[seed]["checkpoint_sha256"],
                }
                and evidence.get("report")
                == {
                    "path": responses[seed]["sidecar"],
                    "sha256": responses[seed]["sidecar_sha256"],
                }
                and evidence.get("guard_receipt")
                == receipt["receipt"]
                and evidence.get("best_epoch")
                == responses[seed]["payload"]["best_epoch"]
                and evidence.get("surface_control")
                == responses[seed]["surface_control"]
                and (
                    schema != 3
                    or evidence.get("calibration") == calibration_by_seed[seed]
                )
                and _close(
                    evidence.get("best_selection_score"),
                    responses[seed]["payload"]["best_selection_score"],
                ),
                f"authority distill seed {seed} terminal/receipt differs",
            )
            expected_authority_seeds.append(
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
                    "guard_receipt": receipt["receipt"],
                    "terminal": _file_record(terminal_path),
                    "best_epoch": responses[seed]["payload"]["best_epoch"],
                    "best_selection_score": responses[seed]["payload"][
                        "best_selection_score"
                    ],
                    "surface_control": responses[seed]["surface_control"],
                    **(
                        {"calibration": calibration_by_seed[seed]}
                        if schema == 3
                        else {}
                    ),
                }
            )
        _require(
            set(completion) == authority_completion_keys
            and completion.get("schema_version") == schema
            and completion.get("artifact_type")
            == "surface_region_text_response_distill_completion"
            and completion.get("status")
            == "complete_three_seed_guarded_authority"
            and completion.get("candidate") == surface["selected_candidate"]
            and completion.get("run_manifest") == _file_record(manifest_path)
            and (
                completion.get("calibrations") == manifest.get("calibrations")
                and completion.get("gradient_design_diagnostic")
                == manifest.get("gradient_design_diagnostic")
                if schema == 3
                else completion.get("calibration_manifest")
                == _file_record(calibration_path)
            )
            and completion.get("runtime_closure_digest") == closure_digest
            and completion.get("selection_contract") == selection_contract
            and completion.get("seeds") == expected_authority_seeds,
            "authority distill completion differs from independent recomputation",
        )
    return {
        "manifest": _file_record(manifest_path),
        "completion": _file_record(completion_path),
        "fit_text_bank": fit_bank,
        "calibration_manifest": (
            _file_record(calibration_path) if calibration_path is not None else None
        ),
        "calibrations": (
            [dict(row) for row in manifest["calibrations"]]
            if schema == 3
            else None
        ),
        "authority_schema_version": schema,
        "response_protocol": response_protocol,
    }


def build_plan(
    promotion_manifest: Path,
    promotion_completion: Path,
    response_checkpoints: Sequence[Path],
    radio_checkpoint: Path,
) -> dict[str, Any]:
    """Strictly pair the frozen Surface control and response seeds."""

    surface = _validate_surface_bundle(promotion_manifest, promotion_completion)
    radio_path = Path(radio_checkpoint).resolve()
    run_manifest = surface["run_manifest"]
    _require(
        Path(str(run_manifest.get("radio_checkpoint", ""))).resolve() == radio_path
        and run_manifest.get("radio_checkpoint_sha256") == _sha256(radio_path),
        "RADIO checkpoint differs from the frozen Surface screen",
    )
    _require(len(response_checkpoints) == len(REQUIRED_SEEDS), "exactly three response checkpoints are required")

    control_rows = []
    controls: dict[int, dict[str, Any]] = {}
    for seed in REQUIRED_SEEDS:
        entry = surface["selected_by_seed"][seed]
        control = _validate_control_checkpoint(entry, seed)
        controls[seed] = control
        control_rows.append(
            {
                "seed": seed,
                "checkpoint": str(control["checkpoint"]),
                "checkpoint_sha256": control["checkpoint_sha256"],
                "sidecar": str(control["sidecar"]),
                "sidecar_sha256": control["sidecar_sha256"],
                "architecture_sha256": control["payload"]["architecture"]["digest"],
                "validation": dict(control["validation"]),
            }
        )

    response_by_seed: dict[int, dict[str, Any]] = {}
    for raw_path in response_checkpoints:
        raw_path = Path(raw_path)
        sidecar = _json_object(raw_path.with_suffix(raw_path.suffix + ".json"))
        checkpoint_sha = sidecar.get("checkpoint_sha256")
        _require(
            _is_sha256(checkpoint_sha),
            "response checkpoint sidecar SHA is invalid",
        )
        try:
            payload = _torch_mapping(
                raw_path,
                expected_sha256=str(checkpoint_sha),
            )
        except ValueError as exc:
            if "SHA-256 differs" in str(exc):
                raise ValueError(
                    "response checkpoint/sidecar selection drift"
                ) from exc
            raise
        seed = payload.get("training_config", {}).get("seed")
        _require(
            isinstance(seed, int)
            and seed in REQUIRED_SEEDS
            and seed not in response_by_seed,
            "response checkpoints must exactly cover unique seeds 0/1/2",
        )
        response_by_seed[seed] = _validate_response_checkpoint(
            Path(raw_path), seed=seed, control=controls[seed]
        )
    _require(set(response_by_seed) == set(REQUIRED_SEEDS), "response checkpoints do not cover seeds 0/1/2")
    distill_run = _validate_shared_distill_run(
        surface=surface,
        responses=response_by_seed,
        radio_path=radio_path,
    )
    response_rows = []
    for seed in REQUIRED_SEEDS:
        row = dict(response_by_seed[seed])
        row.pop("payload")
        if row["distill_schema_version"] == 3:
            row.pop("response_lambda")
        else:
            row.pop("response_lambdas")
            row.pop("design_diagnostic")
        response_rows.append(row)

    if distill_run["authority_schema_version"] == 3:
        _require(
            len({row["calibration_manifest"] for row in response_rows})
            == len(REQUIRED_SEEDS)
            and all(
                set(row["response_lambdas"])
                == {"independent_response", "scene_response"}
                for row in response_rows
            ),
            "response seeds do not bind unique per-seed dual calibrations",
        )
    else:
        shared_calibrations = {
            (
                row["calibration_manifest"],
                row["calibration_manifest_sha256"],
                row["response_lambda"],
                json.dumps(row["fit_text_bank"], sort_keys=True),
            )
            for row in response_rows
        }
        _require(
            len(shared_calibrations) == 1,
            "legacy response seeds do not share one calibration/lambda/fit bank",
        )
    control_validation_paths = controls[0]["payload"]["provenance"]["validation"].get("cache_paths")
    _require(isinstance(control_validation_paths, list) and control_validation_paths, "control lacks validation cache paths")
    for seed in REQUIRED_SEEDS:
        _require(
            controls[seed]["payload"]["provenance"]["validation"].get("cache_paths")
            == control_validation_paths,
            "control seeds do not share validation caches",
        )
    validation_caches = [_file_record(Path(path)) for path in control_validation_paths]
    _require(
        validation_caches == surface["selected_caches"]["validation"],
        "control validation caches differ from the frozen selected candidate",
    )
    selected = surface["selected_candidate"]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PLAN_ARTIFACT_TYPE,
        "status": "frozen_dev_selection_plan_audit_not_opened",
        "selected_candidate": selected,
        "required_seeds": list(REQUIRED_SEEDS),
        "control_method_id": f"surface-{selected}-control-v1",
        "candidate_method_id": f"surface-{selected}-response-distill-v1",
        "surface_noninferiority_tolerance": SURFACE_NONINFERIORITY_TOLERANCE,
        "text_gate_protocol": {
            "dev_is_only_selection_split": True,
            "audit_is_confirmation_only": True,
            "minimum_improved_seeds": MINIMUM_IMPROVED_SEEDS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "quality_noninferiority_tolerance": QUALITY_NONINFERIORITY_TOLERANCE,
        },
        "benchmark_vocabulary_opened": False,
        "benchmark_gate_status": "closed",
        "main_result_eligible": False,
        "surface_promotion": {
            "manifest": str(surface["manifest_path"]),
            "manifest_sha256": surface["manifest_sha256"],
            "completion": str(surface["completion_path"]),
            "completion_sha256": surface["completion_sha256"],
        },
        "radio_checkpoint": _file_record(radio_path),
        "validation_caches": validation_caches,
        "distill_run": distill_run,
        "control": control_rows,
        "candidate": response_rows,
        "implementation": _implementation_binding(),
    }


def preflight(
    promotion_manifest: Path,
    promotion_completion: Path,
    response_checkpoints: Sequence[Path],
    radio_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    plan = build_plan(
        promotion_manifest,
        promotion_completion,
        response_checkpoints,
        radio_checkpoint,
    )
    _write_frozen_json(output, plan)
    return {
        "output": str(Path(output).resolve()),
        "sha256": _sha256(Path(output).resolve()),
        "selected_candidate": plan["selected_candidate"],
        "required_seeds": list(REQUIRED_SEEDS),
        "main_result_eligible": False,
        "device": "cpu",
    }


def validate_plan(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    plan = _json_object(path)
    _require(
        plan.get("schema_version") == SCHEMA_VERSION
        and plan.get("artifact_type") == PLAN_ARTIFACT_TYPE,
        "invalid text-response promotion plan",
    )
    recomputed = build_plan(
        Path(plan["surface_promotion"]["manifest"]),
        Path(plan["surface_promotion"]["completion"]),
        [Path(row["checkpoint"]) for row in plan["candidate"]],
        Path(plan["radio_checkpoint"]["path"]),
    )
    _require(plan == recomputed, "promotion plan differs from strict recomputation")
    return {"path": path, "sha256": _sha256(path), "payload": plan}


def _index_paths_by_seed(
    paths: Sequence[Path],
    *,
    loader,
    role: str,
) -> dict[int, dict[str, Any]]:
    _require(len(paths) == len(REQUIRED_SEEDS), f"{role} must contain exactly three artifacts")
    result = {}
    for path in paths:
        loaded = loader(Path(path))
        seed = loaded.get("seed")
        _require(
            isinstance(seed, int) and seed in REQUIRED_SEEDS and seed not in result,
            f"{role} must exactly cover unique seeds 0/1/2",
        )
        result[seed] = loaded
    _require(set(result) == set(REQUIRED_SEEDS), f"{role} does not cover seeds 0/1/2")
    return result


def _relation_smooth_l1(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> float:
    student = F.normalize(torch.as_tensor(student).float(), dim=-1, eps=1e-8)
    teacher = F.normalize(torch.as_tensor(teacher).float(), dim=-1, eps=1e-8)
    _require(student.shape == teacher.shape and student.ndim == 2, "relation descriptors must align [N,D]")
    count = int(student.shape[0])
    _require(count > 0, "relation descriptors are empty")
    total = 0.0
    for start in range(0, count, int(chunk_size)):
        predicted = student[start : start + int(chunk_size)] @ student.T
        target = teacher[start : start + int(chunk_size)] @ teacher.T
        total += float(F.smooth_l1_loss(predicted, target, reduction="sum"))
    return total / float(count * count)


def _expected_report(
    descriptor: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    query_split: str,
) -> dict[str, Any]:
    metrics = evaluate_response_fidelity(
        descriptor["student"],
        descriptor["teacher"],
        bank["embeddings"],
        scene_ids=descriptor["scene_ids"],
        region_ids=descriptor["region_ids"],
        query_ids=bank["query_ids"],
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "method_id": descriptor["method_id"],
        "seed": descriptor["seed"],
        "split_role": "query_free_validation",
        "query_split": query_split,
        "selection_contract": selection_contract_for_bank_family(
            bank["bank_family"]
        ),
        "descriptor_artifact": {
            "path": str(descriptor["path"]),
            "sha256": descriptor["file_sha256"],
        },
        "descriptor_rows_sha256": descriptor["rows_sha256"],
        "teacher_descriptors_sha256": descriptor["teacher_sha256"],
        "query_bank": {
            "path": str(bank["path"]),
            "sha256": bank["file_sha256"],
            "manifest_path": str(bank["manifest_path"]),
            "manifest_sha256": bank["manifest_sha256"],
            "vocabulary_sha256": bank["vocabulary_sha256"],
            "query_split": query_split,
            "selected_queries": len(bank["selected_records"]),
            "selected_records_sha256": bank["selected_records_sha256"],
            "ordered_records_sha256": bank["ordered_records_sha256"],
            "embedding_tensor_sha256": bank["embedding_tensor_sha256"],
            "embedding_semantic_sha256": bank["embedding_semantic_sha256"],
            "text_encoder": bank["text_encoder"],
        },
        "metrics": metrics,
    }


def _load_report(path: Path) -> dict[str, Any]:
    payload = _json_object(path)
    _validate_report(payload)
    return {
        "path": Path(path).resolve(),
        "sha256": _sha256(Path(path).resolve()),
        "seed": payload["seed"],
        "payload": payload,
    }


def _control_descriptor_authority_type(plan: Mapping[str, Any]) -> str:
    manifest = _json_object(Path(plan["surface_promotion"]["manifest"]))
    artifact_type = manifest.get("artifact_type")
    if artifact_type == "surface_c1024_attention_pooling_postcache_continuation":
        return "attention_postcache_screen"
    _require(
        artifact_type == surface_finalizer.ARTIFACT_TYPE,
        "plan Surface authority artifact type differs",
    )
    return "query_free_promotion_bundle"


def _validate_descriptor_role(
    descriptors: Sequence[Path],
    *,
    plan: Mapping[str, Any],
    plan_rows: Mapping[int, Mapping[str, Any]],
    method_id: str,
    role: str,
) -> dict[int, dict[str, Any]]:
    loaded = _index_paths_by_seed(
        descriptors,
        loader=load_descriptor_pair,
        role=f"{role} descriptors",
    )
    for seed, value in loaded.items():
        provenance = value["payload"].get("provenance", {})
        expected = plan_rows[seed]
        _require(value["method_id"] == method_id, f"{role} descriptor method_id drift")
        _require(
            Path(str(provenance.get("readout_checkpoint", ""))).resolve()
            == Path(expected["checkpoint"]).resolve()
            and provenance.get("readout_checkpoint_sha256")
            == expected["checkpoint_sha256"]
            and Path(str(provenance.get("readout_report", ""))).resolve()
            == Path(expected["sidecar"]).resolve()
            and provenance.get("readout_report_sha256")
            == expected["sidecar_sha256"],
            f"{role} seed {seed} descriptor checkpoint binding mismatch",
        )
        _require(
            provenance.get("uses_benchmark_scenes") is False
            and provenance.get("uses_benchmark_test_vocabulary") is False
            and provenance.get("annotations_opened") is False
            and provenance.get("labels_opened") is False
            and provenance.get("instances_opened") is False
            and provenance.get("masks_opened") is False
            and provenance.get("text_opened") is False
            and provenance.get("device") == "cpu",
            f"{role} seed {seed} descriptor opened forbidden data or device",
        )
        _require(
            Path(str(provenance.get("radio_checkpoint", ""))).resolve()
            == Path(plan["radio_checkpoint"]["path"]).resolve()
            and provenance.get("radio_checkpoint_sha256")
            == plan["radio_checkpoint"]["sha256"],
            f"{role} seed {seed} descriptor RADIO binding mismatch",
        )
        cache_records = provenance.get("validation_caches")
        _require(
            isinstance(cache_records, list)
            and [
                {"path": record.get("path"), "sha256": record.get("sha256")}
                for record in cache_records
                if isinstance(record, Mapping)
            ]
            == plan["validation_caches"],
            f"{role} seed {seed} descriptor validation-cache binding mismatch",
        )
        authority = provenance.get("readout_binding_authority")
        if role == "control":
            expected_authority = {
                "type": _control_descriptor_authority_type(plan),
                "path": plan["surface_promotion"]["manifest"],
                "sha256": plan["surface_promotion"]["manifest_sha256"],
                "completion": plan["surface_promotion"]["completion"],
                "completion_sha256": plan["surface_promotion"]["completion_sha256"],
                "candidate": plan["selected_candidate"],
            }
        else:
            expected_authority = {
                "type": "embedded_distill_run_manifest",
                **expected["distill_run_manifest"],
            }
        _require(
            authority == expected_authority,
            f"{role} seed {seed} descriptor authority binding mismatch",
        )
    return loaded


def _validate_reports(
    reports: Sequence[Path],
    *,
    descriptors: Mapping[int, Mapping[str, Any]],
    bank: Mapping[str, Any],
    query_split: str,
    role: str,
) -> dict[int, dict[str, Any]]:
    loaded = _index_paths_by_seed(
        reports,
        loader=_load_report,
        role=f"{role} reports",
    )
    for seed, value in loaded.items():
        expected = _expected_report(descriptors[seed], bank, query_split=query_split)
        _require(
            value["payload"] == expected,
            f"{role} seed {seed} report differs from strict descriptor-response recomputation",
        )
    return loaded


def _surface_retention(
    plan: Mapping[str, Any],
    controls: Mapping[int, Mapping[str, Any]],
    candidates: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    control_plan = {int(row["seed"]): row for row in plan["control"]}
    candidate_plan = {int(row["seed"]): row for row in plan["candidate"]}
    per_seed = []
    all_checks = []
    for seed in REQUIRED_SEEDS:
        control_relation_loss = _relation_smooth_l1(
            controls[seed]["student"], controls[seed]["teacher"]
        )
        candidate_relation_loss = _relation_smooth_l1(
            candidates[seed]["student"], candidates[seed]["teacher"]
        )
        control_metrics = {
            **control_plan[seed]["validation"],
            "relation_fidelity": 1.0 - control_relation_loss,
        }
        candidate_metrics = {
            **candidate_plan[seed]["validation"],
            "relation_fidelity": 1.0 - candidate_relation_loss,
        }
        deltas = {
            metric: float(candidate_metrics[metric] - control_metrics[metric])
            for metric in SURFACE_METRICS
        }
        checks = {
            metric: delta >= -SURFACE_NONINFERIORITY_TOLERANCE
            for metric, delta in deltas.items()
        }
        all_checks.extend(checks.values())
        per_seed.append(
            {
                "seed": seed,
                "control": control_metrics,
                "candidate": candidate_metrics,
                "candidate_minus_control": deltas,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    return {
        "definition": {
            "metric_direction": "higher_is_better",
            "minimum_candidate_minus_control": -SURFACE_NONINFERIORITY_TOLERANCE,
            "relation_fidelity": "1-global_pairwise_relation_smooth_l1",
            "relation_pairs": "all_validation_region_pairs_including_diagonal",
        },
        "per_seed": per_seed,
        "passes": all(all_checks),
    }


def _stage_completion(
    output: Path,
    completion: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output).resolve()
    completion = Path(completion).resolve()
    _require(output != completion, "stage output and completion must differ")
    if not output.exists():
        _require(not completion.exists(), "completion exists without stage output")
    _write_frozen_json(output, payload)
    completion_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETION_ARTIFACT_TYPE,
        "stage": payload["stage"],
        "status": "complete",
        "stage_manifest": str(output),
        "stage_manifest_sha256": _sha256(output),
        "decision": payload["decision"],
        "main_result_eligible": payload["main_result_eligible"],
        "benchmark_vocabulary_opened": False,
    }
    _write_frozen_json(completion, completion_payload)
    return {
        "output": str(output),
        "output_sha256": _sha256(output),
        "completion": str(completion),
        "completion_sha256": _sha256(completion),
        "stage": payload["stage"],
        "decision": payload["decision"],
        "main_result_eligible": payload["main_result_eligible"],
        "benchmark_vocabulary_opened": False,
        "device": "cpu",
    }


def _build_stage_payload(
    *,
    stage: str,
    decision: str,
    main_result_eligible: bool,
    plan_record: Mapping[str, Any],
    plan: Mapping[str, Any],
    controls: Mapping[int, Mapping[str, Any]],
    candidates: Mapping[int, Mapping[str, Any]],
    control_report_rows: Mapping[int, Mapping[str, Any]],
    candidate_report_rows: Mapping[int, Mapping[str, Any]],
    gate: Mapping[str, Any],
    retention: Mapping[str, Any],
    text_bank_path: Path,
    text_bank_manifest_path: Path,
    gate_path: Path,
    dev_dependency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical stage payload from validated frozen evidence."""

    descriptor_bindings = {
        "control": [_file_record(controls[seed]["path"]) for seed in REQUIRED_SEEDS],
        "candidate": [
            _file_record(candidates[seed]["path"]) for seed in REQUIRED_SEEDS
        ],
    }
    report_bindings = {
        "control": [
            _file_record(control_report_rows[seed]["path"])
            for seed in REQUIRED_SEEDS
        ],
        "candidate": [
            _file_record(candidate_report_rows[seed]["path"])
            for seed in REQUIRED_SEEDS
        ],
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": STAGE_ARTIFACT_TYPE,
        "stage": stage,
        "stage_role": (
            "only_method_selection_split"
            if stage == "dev"
            else "single_confirmation_only_no_retuning"
        ),
        "decision": decision,
        "selected_candidate": plan["selected_candidate"],
        "required_seeds": list(REQUIRED_SEEDS),
        "control_method_id": plan["control_method_id"],
        "candidate_method_id": plan["candidate_method_id"],
        "text_gate": dict(gate),
        "surface_retention": dict(retention),
        "benchmark_vocabulary_opened": False,
        "benchmark_gate_status": "closed",
        "main_result_eligible": bool(main_result_eligible),
        "plan": {
            "path": str(plan_record["path"]),
            "sha256": str(plan_record["sha256"]),
        },
        "bindings": {
            "text_bank": _file_record(Path(text_bank_path)),
            "text_bank_manifest": _file_record(Path(text_bank_manifest_path)),
            "descriptors": descriptor_bindings,
            "reports": report_bindings,
            "gate": _file_record(Path(gate_path)),
            "implementation": _implementation_binding(),
        },
    }
    if dev_dependency is not None:
        payload["dev_dependency"] = {
            "manifest": str(dev_dependency["path"]),
            "manifest_sha256": str(dev_dependency["sha256"]),
            "completion": str(dev_dependency["completion"]),
            "completion_sha256": str(dev_dependency["completion_sha256"]),
            "decision": "promote_audit_required",
        }
    return payload


def _validate_dev_dependency(
    manifest_path: Path,
    completion_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    completion_path = Path(completion_path).resolve()
    manifest = _json_object(manifest_path)
    completion = _json_object(completion_path)
    expected_manifest_keys = {
        "schema_version",
        "artifact_type",
        "stage",
        "stage_role",
        "decision",
        "selected_candidate",
        "required_seeds",
        "control_method_id",
        "candidate_method_id",
        "text_gate",
        "surface_retention",
        "benchmark_vocabulary_opened",
        "benchmark_gate_status",
        "main_result_eligible",
        "plan",
        "bindings",
    }
    _require(
        set(manifest) == expected_manifest_keys
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == STAGE_ARTIFACT_TYPE
        and manifest.get("stage") == "dev"
        and manifest.get("stage_role") == "only_method_selection_split"
        and manifest.get("required_seeds") == list(REQUIRED_SEEDS)
        and manifest.get("benchmark_vocabulary_opened") is False
        and manifest.get("benchmark_gate_status") == "closed"
        and manifest.get("main_result_eligible") is False,
        "invalid frozen dev stage schema",
    )
    expected_completion_keys = {
        "schema_version",
        "artifact_type",
        "stage",
        "status",
        "stage_manifest",
        "stage_manifest_sha256",
        "decision",
        "main_result_eligible",
        "benchmark_vocabulary_opened",
    }
    _require(
        set(completion) == expected_completion_keys
        and completion.get("schema_version") == SCHEMA_VERSION
        and completion.get("artifact_type") == COMPLETION_ARTIFACT_TYPE
        and completion.get("stage") == "dev"
        and completion.get("status") == "complete"
        and Path(str(completion.get("stage_manifest", ""))).resolve()
        == manifest_path
        and completion.get("stage_manifest_sha256") == _sha256(manifest_path)
        and completion.get("main_result_eligible") is False
        and completion.get("benchmark_vocabulary_opened") is False,
        "invalid frozen dev completion",
    )
    _require(
        manifest.get("plan")
        == {"path": str(plan["path"]), "sha256": plan["sha256"]},
        "audit plan differs from frozen dev selection",
    )
    bindings = manifest.get("bindings")
    _require(
        isinstance(bindings, Mapping)
        and set(bindings)
        == {
            "text_bank",
            "text_bank_manifest",
            "descriptors",
            "reports",
            "gate",
            "implementation",
        },
        "frozen dev bindings are incomplete",
    )
    text_bank_path = _validate_record_file(bindings["text_bank"], "dev text bank")
    text_bank_manifest_path = _validate_record_file(
        bindings["text_bank_manifest"], "dev text bank manifest"
    )
    gate_path = _validate_record_file(bindings["gate"], "dev gate")
    _require(
        bindings["implementation"] == _implementation_binding(),
        "dev implementation binding changed before audit",
    )
    bound_paths: dict[str, list[Path]] = {}
    for group in ("descriptors", "reports"):
        value = bindings[group]
        _require(
            isinstance(value, Mapping) and set(value) == {"control", "candidate"},
            f"dev {group} bindings are incomplete",
        )
        for role in ("control", "candidate"):
            records = value[role]
            _require(
                isinstance(records, list) and len(records) == len(REQUIRED_SEEDS),
                f"dev {role} {group} do not cover three seeds",
            )
            bound_paths[f"{role}_{group}"] = [
                _validate_record_file(record, f"dev {role} {group} row {index}")
                for index, record in enumerate(records)
            ]

    # The dev manifest and completion are mutually hashed JSON, not an
    # independent authority. Recompute the complete selection from the frozen
    # dev evidence before any one-shot audit artifact may be opened.
    plan_payload = plan["payload"]
    control_plan = {int(row["seed"]): row for row in plan_payload["control"]}
    candidate_plan = {int(row["seed"]): row for row in plan_payload["candidate"]}
    controls = _validate_descriptor_role(
        bound_paths["control_descriptors"],
        plan=plan_payload,
        plan_rows=control_plan,
        method_id=plan_payload["control_method_id"],
        role="control",
    )
    candidates = _validate_descriptor_role(
        bound_paths["candidate_descriptors"],
        plan=plan_payload,
        plan_rows=candidate_plan,
        method_id=plan_payload["candidate_method_id"],
        role="candidate",
    )
    bank = load_text_embedding_bank(text_bank_path, text_bank_manifest_path, "dev")
    control_report_rows = _validate_reports(
        bound_paths["control_reports"],
        descriptors=controls,
        bank=bank,
        query_split="dev",
        role="control",
    )
    candidate_report_rows = _validate_reports(
        bound_paths["candidate_reports"],
        descriptors=candidates,
        bank=bank,
        query_split="dev",
        role="candidate",
    )
    recomputed_gate = aggregate_paired_seed_gate(
        [control_report_rows[seed]["payload"] for seed in REQUIRED_SEEDS],
        [candidate_report_rows[seed]["payload"] for seed in REQUIRED_SEEDS],
        required_seeds=REQUIRED_SEEDS,
        minimum_improved_seeds=MINIMUM_IMPROVED_SEEDS,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        quality_noninferiority_tolerance=QUALITY_NONINFERIORITY_TOLERANCE,
        phase="dev",
    )
    _require(
        _json_object(gate_path) == recomputed_gate,
        "frozen dev gate differs from strict paired recomputation",
    )
    retention = _surface_retention(plan_payload, controls, candidates)
    text_promotes = recomputed_gate.get("decision") == "promote"
    decision = (
        "promote_audit_required"
        if text_promotes and retention["passes"]
        else "reject_no_audit"
    )
    recomputed_manifest = _build_stage_payload(
        stage="dev",
        decision=decision,
        main_result_eligible=False,
        plan_record=plan,
        plan=plan_payload,
        controls=controls,
        candidates=candidates,
        control_report_rows=control_report_rows,
        candidate_report_rows=candidate_report_rows,
        gate=recomputed_gate,
        retention=retention,
        text_bank_path=text_bank_path,
        text_bank_manifest_path=text_bank_manifest_path,
        gate_path=gate_path,
    )
    _require(
        manifest == recomputed_manifest,
        "frozen dev stage differs from strict recomputation",
    )
    recomputed_completion = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETION_ARTIFACT_TYPE,
        "stage": "dev",
        "status": "complete",
        "stage_manifest": str(manifest_path),
        "stage_manifest_sha256": _sha256(manifest_path),
        "decision": decision,
        "main_result_eligible": False,
        "benchmark_vocabulary_opened": False,
    }
    _require(
        completion == recomputed_completion,
        "frozen dev completion differs from strict recomputation",
    )
    _require(
        decision == "promote_audit_required",
        "audit is forbidden unless both frozen dev gates passed",
    )
    return {
        "path": manifest_path,
        "sha256": _sha256(manifest_path),
        "completion": completion_path,
        "completion_sha256": _sha256(completion_path),
        "payload": manifest,
        "text_bank_identity": _validated_text_bank_identity(bank, "dev"),
    }


def _validated_text_bank_identity(
    bank: Mapping[str, Any],
    query_split: str,
) -> dict[str, str]:
    """Return the fail-closed vocabulary/embedding-family identity of a bank."""

    bank_family = bank.get("bank_family")
    algorithm_version = bank.get("algorithm_version")
    _require(
        isinstance(bank_family, str)
        and bool(bank_family)
        and isinstance(algorithm_version, str)
        and bool(algorithm_version),
        f"{query_split} text bank lacks a validated family identity",
    )
    return {
        "bank_family": bank_family,
        "algorithm_version": algorithm_version,
    }


def _require_matching_text_bank_family(
    dev_identity: Mapping[str, Any],
    audit_bank: Mapping[str, Any],
) -> None:
    audit_identity = _validated_text_bank_identity(audit_bank, "audit")
    _require(
        dict(dev_identity) == audit_identity,
        "audit text bank family/algorithm differs from the frozen dev text bank",
    )


def finalize_stage(
    *,
    stage: str,
    plan_path: Path,
    control_descriptors: Sequence[Path],
    candidate_descriptors: Sequence[Path],
    control_reports: Sequence[Path],
    candidate_reports: Sequence[Path],
    gate_path: Path,
    text_bank_path: Path,
    text_bank_manifest_path: Path,
    output: Path,
    completion: Path,
    dev_manifest: Path | None = None,
    dev_completion: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"dev", "audit"}:
        raise ValueError("stage must be dev or audit")
    query_split = stage
    plan_record = validate_plan(plan_path)
    plan = plan_record["payload"]
    dev_dependency = None
    if stage == "audit":
        if dev_manifest is None or dev_completion is None:
            raise ValueError("audit requires frozen dev manifest and completion")
        # This check deliberately precedes every audit-bank/report read.
        dev_dependency = _validate_dev_dependency(
            dev_manifest,
            dev_completion,
            plan_record,
        )
    control_plan = {int(row["seed"]): row for row in plan["control"]}
    candidate_plan = {int(row["seed"]): row for row in plan["candidate"]}
    controls = _validate_descriptor_role(
        control_descriptors,
        plan=plan,
        plan_rows=control_plan,
        method_id=plan["control_method_id"],
        role="control",
    )
    candidates = _validate_descriptor_role(
        candidate_descriptors,
        plan=plan,
        plan_rows=candidate_plan,
        method_id=plan["candidate_method_id"],
        role="candidate",
    )
    bank = load_text_embedding_bank(
        Path(text_bank_path),
        Path(text_bank_manifest_path),
        query_split,
    )
    if stage == "audit":
        assert dev_dependency is not None
        # The audit bank must be validated to learn its frozen family identity,
        # but a cross-family pair is rejected before any audit report or metric
        # artifact is opened.
        _require_matching_text_bank_family(
            dev_dependency["text_bank_identity"],
            bank,
        )
    control_report_rows = _validate_reports(
        control_reports,
        descriptors=controls,
        bank=bank,
        query_split=query_split,
        role="control",
    )
    candidate_report_rows = _validate_reports(
        candidate_reports,
        descriptors=candidates,
        bank=bank,
        query_split=query_split,
        role="candidate",
    )
    recomputed_gate = aggregate_paired_seed_gate(
        [control_report_rows[seed]["payload"] for seed in REQUIRED_SEEDS],
        [candidate_report_rows[seed]["payload"] for seed in REQUIRED_SEEDS],
        required_seeds=REQUIRED_SEEDS,
        minimum_improved_seeds=MINIMUM_IMPROVED_SEEDS,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        quality_noninferiority_tolerance=QUALITY_NONINFERIORITY_TOLERANCE,
        phase=stage,
    )
    gate_path = Path(gate_path).resolve()
    gate = _json_object(gate_path)
    _require(gate == recomputed_gate, f"{stage} gate differs from strict paired recomputation")
    retention = _surface_retention(plan, controls, candidates)

    if stage == "audit":
        assert dev_dependency is not None
        frozen_descriptors = dev_dependency["payload"]["bindings"]["descriptors"]
        current_descriptor_bindings = {
            "control": [
                _file_record(controls[seed]["path"]) for seed in REQUIRED_SEEDS
            ],
            "candidate": [
                _file_record(candidates[seed]["path"]) for seed in REQUIRED_SEEDS
            ],
        }
        _require(
            frozen_descriptors == current_descriptor_bindings,
            "audit descriptors differ from the frozen dev method",
        )

    text_promotes = gate.get("decision") == "promote"
    if stage == "dev":
        decision = (
            "promote_audit_required"
            if text_promotes and retention["passes"]
            else "reject_no_audit"
        )
        main_result_eligible = False
    else:
        decision = (
            "promote_confirmed"
            if text_promotes and retention["passes"]
            else "audit_reject_no_retuning"
        )
        main_result_eligible = decision == "promote_confirmed"

    payload = _build_stage_payload(
        stage=stage,
        decision=decision,
        main_result_eligible=main_result_eligible,
        plan_record=plan_record,
        plan=plan,
        controls=controls,
        candidates=candidates,
        control_report_rows=control_report_rows,
        candidate_report_rows=candidate_report_rows,
        gate=gate,
        retention=retention,
        text_bank_path=Path(text_bank_path),
        text_bank_manifest_path=Path(text_bank_manifest_path),
        gate_path=gate_path,
        dev_dependency=dev_dependency,
    )
    return _stage_completion(output, completion, payload)


def _paths(values: Iterable[str]) -> list[Path]:
    return [Path(value).resolve() for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--promotion-manifest", required=True, type=Path)
    preflight_parser.add_argument("--promotion-completion", required=True, type=Path)
    preflight_parser.add_argument("--response-checkpoint", action="append", required=True, type=Path)
    preflight_parser.add_argument("--radio-checkpoint", required=True, type=Path)
    preflight_parser.add_argument("--output", required=True, type=Path)

    for stage in ("dev", "audit"):
        stage_parser = subparsers.add_parser(f"finalize-{stage}")
        stage_parser.add_argument("--plan", required=True, type=Path)
        stage_parser.add_argument("--control-descriptor", action="append", required=True, type=Path)
        stage_parser.add_argument("--candidate-descriptor", action="append", required=True, type=Path)
        stage_parser.add_argument("--control-report", action="append", required=True, type=Path)
        stage_parser.add_argument("--candidate-report", action="append", required=True, type=Path)
        stage_parser.add_argument("--gate", required=True, type=Path)
        stage_parser.add_argument("--text-bank", required=True, type=Path)
        stage_parser.add_argument("--text-bank-manifest", required=True, type=Path)
        stage_parser.add_argument("--output", required=True, type=Path)
        stage_parser.add_argument("--completion", required=True, type=Path)
        if stage == "audit":
            stage_parser.add_argument("--dev-manifest", required=True, type=Path)
            stage_parser.add_argument("--dev-completion", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        result = preflight(
            args.promotion_manifest,
            args.promotion_completion,
            args.response_checkpoint,
            args.radio_checkpoint,
            args.output,
        )
    else:
        stage = args.command.removeprefix("finalize-")
        result = finalize_stage(
            stage=stage,
            plan_path=args.plan,
            control_descriptors=args.control_descriptor,
            candidate_descriptors=args.candidate_descriptor,
            control_reports=args.control_report,
            candidate_reports=args.candidate_report,
            gate_path=args.gate,
            text_bank_path=args.text_bank,
            text_bank_manifest_path=args.text_bank_manifest,
            output=args.output,
            completion=args.completion,
            dev_manifest=getattr(args, "dev_manifest", None),
            dev_completion=getattr(args, "dev_completion", None),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
