#!/usr/bin/env python3
"""Materialize paired Surface readout/teacher descriptors on CPU only."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import (
    canonical_json_sha256,
    row_identity_sha256,
    tensor_sha256,
)
from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "surface_text_response_descriptor_pair"
_DISTILL_RUN_ARTIFACT_TYPE = "surface_region_text_response_distill_run"
_MATERIALIZER_RELATIVE_PATH = (
    "radio_gs/scripts/materialize_surface_text_response_descriptors.py"
)
_FORMAL_V3_MANIFEST_CANONICAL_SHA256 = (
    "a3aaafb42799fcfa46c2fa088a748e52ab204e4a88d6fd72cc0a8389d3b940cd"
)
_FORMAL_V3_MANIFEST_SHA256 = (
    "497cc23d08db7500d78c5741de48c132c78e493a6fd205e34489668725f16615"
)
_FORMAL_V3_MANIFEST_PATH = (
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
    "surface_text_response_warmstart_dualprofile_joint_c1024_gpu1only_v3/"
    "run_manifest.json"
)
_FORMAL_V3_RUNTIME_CLOSURE_SHA256 = (
    "56736482529dc7f7fc34c6b29cba2373c59993820989c4f41bca3f8a1f6767e3"
)
_FORMAL_V3_PRODUCER_MATERIALIZER_SHA256 = (
    "df80eff185913e0b4fc81f2d9c482e9e8b873ce8f0991ba32ed05c04db93b651"
)
_AUTHORITY_DISTILL_MANIFEST_FIELDS_V2 = frozenset(
    {
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
        "authority_status",
        "calibration_audit",
        "initial_gpu_preflight",
        "gpu_identity",
        "runtime_closure",
        "authority_contract",
        "training_command_contract",
    }
)
_AUTHORITY_DISTILL_MANIFEST_FIELDS_V3 = frozenset(
    (
        _AUTHORITY_DISTILL_MANIFEST_FIELDS_V2
        - {"calibration_manifest", "calibration_audit"}
    )
    | {"calibrations", "gradient_design_diagnostic"}
)
_AUTHORITY_DISTILL_OUTPUT_FIELDS = frozenset(
    {
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
)
_QUERY_FREE_FLAGS = (
    "uses_benchmark_scenes",
    "uses_benchmark_test_vocabulary",
    "annotations_opened",
    "labels_opened",
    "instances_opened",
    "masks_opened",
    "text_opened",
)

_INDEPENDENT_RESPONSE_LOSS = (
    "independent_normalized_cosine_response_smooth_l1"
)
_PAIRWISE_SCENE_RESPONSE_LOSS = (
    "scene_wise_text_response_weighted_profile_pairwise_gap_smooth_l1"
)
_PAIRWISE_SCENE_RESPONSE_OBJECTIVE = {
    "name": _PAIRWISE_SCENE_RESPONSE_LOSS,
    "profile_loss": (
        "scene_wise_centered_text_response_profile_cosine_distance"
    ),
    "profile_weight": 0.2,
    "ranking_loss": "scene_wise_text_response_pairwise_gap_smooth_l1",
    "ranking_weight": 1.0,
    "tie_tolerance": 1e-6,
    "pairwise_gap_normalization": "per_scene_query_teacher_response_span",
}
_PAIRWISE_RESPONSE_LOSSES = [
    _INDEPENDENT_RESPONSE_LOSS,
    _PAIRWISE_SCENE_RESPONSE_LOSS,
]
_PAIRWISE_CALIBRATION_ALGORITHM = (
    "per-seed-surface-warmstart-dual-response-pairwise-gradient-budget-v3"
)
_PAIRWISE_EPOCH_SELECTION = (
    "surface_control_0p002_fit_scene_robust_0p005_then_response_error_v3"
)
_ACCEPTED_ANCHOR_EPOCH_SELECTION = (
    "surface_control_0p002_fit_scene_robust_0p005_accepted_anchor_"
    "fixed_1over2048_then_response_error_v4"
)
_HISTORY_HASH_CHAIN_ALGORITHM = (
    "sha256_canonical_json_previous_plus_record_without_selection_score_v1"
)
_PROPOSAL_STATE_MACHINE = {
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
_PROPOSAL_LOSS_FIELDS = tuple(
    _PROPOSAL_STATE_MACHINE["proposal_loss_accounting"]["fields"]
)
_PROPOSAL_LOSS_FLAT_MIRROR = dict(
    _PROPOSAL_STATE_MACHINE["proposal_loss_accounting"]["legacy_flat_mirror"]
)
_LEGACY_RESPONSE_LOSSES = [
    _INDEPENDENT_RESPONSE_LOSS,
    "scene_wise_text_response_profile_ranking",
]
_LEGACY_CALIBRATION_ALGORITHM = (
    "per-seed-surface-warmstart-dual-response-gradient-budget-v2"
)
_LEGACY_EPOCH_SELECTION = (
    "surface_control_feasible_0p002_then_fit_support_response_relation_surface_v2"
)

_V3_RESPONSE_LAMBDA_FIELDS = frozenset(
    {"independent_response", "scene_response"}
)
_V3_RESPONSE_PROVENANCE_FIELDS = frozenset(
    {
        "fit_split_only",
        "benchmark_vocabulary_opened",
        "fit_text_bank",
        "calibration_manifest",
        "calibration_manifest_sha256",
        "calibration_seed",
        "response_lambdas",
        "response_branch_gradient_target_ratio",
        "total_response_gradient_ratio_upper_bound",
        "losses",
        "scene_response_objective",
        "complete_scene_batching",
        "design_diagnostic",
    }
)
_LEGACY_V3_RESPONSE_PROVENANCE_FIELDS = (
    _V3_RESPONSE_PROVENANCE_FIELDS - {"scene_response_objective"}
)
_V3_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "train_caches",
        "validation_caches",
        "fit_text_bank",
        "fit_text_bank_manifest",
        "calibration_manifest",
        "run_manifest",
        "surface_control_checkpoint",
        "surface_control_checkpoint_sha256",
        "output",
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
        "device",
        "radio_checkpoint",
    }
)
_V4_TRAINING_CONFIG_FIELDS = _V3_TRAINING_CONFIG_FIELDS | {
    "proposal_state_machine"
}
_V4_CHECKPOINT_STATE_FIELDS = frozenset(
    {
        "best_state_dict_sha256",
        "proposal_state_machine",
        "accepted_anchor",
        "history_hash_chain_sha256",
    }
)
_V4_ACCEPTED_ANCHOR_FIELDS = frozenset(
    {
        "epoch",
        "state_dict_sha256",
        "accepted_proposal_count",
        "rejected_proposal_count",
    }
)
_V4_PROPOSAL_FIELDS = frozenset(
    {
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
)
_V3_DESIGN_DIAGNOSTIC_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "role",
        "measured_seed",
        "calibration_reuses_measured_values",
        "diagnostic_surface_control",
    }
)
_V3_SURFACE_CONTROL_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "seed",
        "architecture",
        "train_caches",
        "validation_caches",
        "source_best_epoch",
        "source_best_selection_score",
    }
)
_V3_FIT_TEXT_BANK_FIELDS = frozenset(
    {
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
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path) -> Mapping:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be a non-empty absolute path")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} is missing or cannot be resolved") from error
    if path != resolved:
        raise ValueError(f"{label} must be a canonical non-symlink path")
    return resolved


def _validate_file_record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA256 record")
    path = _canonical_absolute_path(value.get("path"), label=label)
    if not path.is_file() or value.get("sha256") != _sha256_file(path):
        raise ValueError(f"{label} SHA256 mismatch")
    return {"path": str(path), "sha256": str(value["sha256"])}


def _cache_bindings(value: object, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} cache provenance is missing")
    records = value.get("cache_bindings")
    if (
        not isinstance(records, list)
        or not records
        or any(
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            for record in records
        )
    ):
        raise ValueError(f"{label} cache SHA bindings differ")
    return [dict(record) for record in records]


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _is_registered_legacy_formal_v3_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> bool:
    runtime_closure = manifest.get("runtime_closure")
    return (
        manifest.get("schema_version") == 3
        and manifest_path == Path(_FORMAL_V3_MANIFEST_PATH).resolve()
        and manifest_sha256 == _FORMAL_V3_MANIFEST_SHA256
        and canonical_json_sha256(dict(manifest))
        == _FORMAL_V3_MANIFEST_CANONICAL_SHA256
        and isinstance(runtime_closure, Mapping)
        and runtime_closure.get("digest")
        == _FORMAL_V3_RUNTIME_CLOSURE_SHA256
    )


def _registered_legacy_formal_v3_binding(binding: object) -> bool:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "candidate",
    }:
        return False
    raw_path = Path(str(binding.get("path", "")))
    if not raw_path.is_absolute():
        return False
    try:
        manifest_path = raw_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False
    if raw_path != manifest_path or not manifest_path.is_file():
        return False
    manifest_sha = _sha256_file(manifest_path)
    if binding.get("sha256") != manifest_sha:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(manifest, Mapping) and _is_registered_legacy_formal_v3_manifest(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
    )


def _path_text(value: object, *, label: str) -> str:
    if isinstance(value, Path):
        value = str(value)
    return str(_canonical_absolute_path(value, label=label))


def _configured_path_text(value: object, *, label: str) -> str:
    """Resolve a trainer-persisted CLI path before immutable binding checks."""

    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be a non-empty absolute path")
    try:
        return str(Path(value).resolve(strict=True))
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} is missing or cannot be resolved") from error


def _path_text_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty path list")
    return [
        _path_text(item, label=f"{label} item {index}")
        for index, item in enumerate(value)
    ]


def _path_pattern_or_list(value: object, *, label: str) -> list[str]:
    """Resolve the cache syntax persisted by the trainer into canonical paths.

    Authority-schema checkpoints persist the original CLI cache argument, which
    may be one or more comma/space-separated absolute glob patterns.  Older
    checkpoints and test fixtures persist the already-expanded path list.  The
    expansion below intentionally mirrors the trainer's ``_paths`` ordering;
    only the resulting files are canonicalized because a glob itself can pass
    through a symlinked directory prefix.
    """

    if isinstance(value, list):
        return _path_text_list(value, label=label)
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{label} must be a non-empty absolute path pattern or path list"
        )
    patterns = value.replace(",", " ").split()
    if not patterns:
        raise ValueError(
            f"{label} must be a non-empty absolute path pattern or path list"
        )
    result: list[str] = []
    for pattern_index, pattern in enumerate(patterns):
        if not Path(pattern).is_absolute():
            raise ValueError(f"{label} pattern {pattern_index} must be absolute")
        matches = (
            sorted(glob.glob(pattern))
            if any(character in pattern for character in "*?[")
            else [pattern]
        )
        if not matches:
            raise ValueError(f"{label} pattern {pattern_index} matched no files")
        for match_index, match in enumerate(matches):
            try:
                resolved = Path(match).resolve(strict=True)
            except (FileNotFoundError, OSError) as error:
                raise ValueError(
                    f"{label} pattern {pattern_index} match {match_index} "
                    "is missing or cannot be resolved"
                ) from error
            if not resolved.is_file():
                raise ValueError(
                    f"{label} pattern {pattern_index} match {match_index} "
                    "must be a file"
                )
            result.append(str(resolved))
    return result


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _state_dict_sha256(state: object, *, label: str) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} must be a non-empty state_dict")
    records = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise ValueError(f"{label} fields differ")
        tensor = value.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{label} contains a non-finite tensor")
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


def _response_rank(record: Mapping[str, Any], *, label: str) -> tuple[float, ...]:
    fields = (
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
    values = []
    for field in fields:
        value = record.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} {field} must be finite numeric")
        values.append(float(value))
    return (
        -values[0],
        -values[1],
        *values[2:9],
        -values[9],
        values[10],
    )


def _validate_v4_checkpoint_state_machine(
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    control_payload: Mapping[str, Any],
    *,
    patience: int,
) -> dict[str, Any]:
    if (
        checkpoint.get("proposal_state_machine") != _PROPOSAL_STATE_MACHINE
        or provenance.get("proposal_state_machine") != _PROPOSAL_STATE_MACHINE
    ):
        raise ValueError("accepted-anchor proposal state-machine contract differs")
    for field in _V4_CHECKPOINT_STATE_FIELDS:
        if field not in checkpoint:
            raise ValueError("accepted-anchor checkpoint state fields differ")
    best_state_sha = checkpoint.get("best_state_dict_sha256")
    published_state_sha = _state_dict_sha256(
        checkpoint.get("state_dict"), label="published response checkpoint state"
    )
    if not _sha256_text(best_state_sha) or published_state_sha != best_state_sha:
        raise ValueError("response checkpoint does not publish the best state")

    accepted_anchor = checkpoint.get("accepted_anchor")
    if (
        not isinstance(accepted_anchor, Mapping)
        or set(accepted_anchor) != _V4_ACCEPTED_ANCHOR_FIELDS
        or not _sha256_text(accepted_anchor.get("state_dict_sha256"))
        or any(
            not isinstance(accepted_anchor.get(field), int)
            or isinstance(accepted_anchor.get(field), bool)
            or int(accepted_anchor[field]) < 0
            for field in (
                "epoch",
                "accepted_proposal_count",
                "rejected_proposal_count",
            )
        )
    ):
        raise ValueError("accepted-anchor terminal metadata differs")
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("accepted-anchor checkpoint history is empty")
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
    for index, raw_record in enumerate(history):
        if not isinstance(raw_record, Mapping) or raw_record.get("epoch") != index:
            raise ValueError("accepted-anchor history must be contiguous from epoch 0")
        record = dict(raw_record)
        chain = record.get("history_hash_chain")
        expected_chain_sha = _history_chain_digest(record, previous_chain)
        if chain != {
            "algorithm": _HISTORY_HASH_CHAIN_ALGORITHM,
            "previous_sha256": previous_chain,
            "sha256": expected_chain_sha,
        }:
            raise ValueError(f"accepted-anchor history hash chain differs at row {index}")
        previous_chain = expected_chain_sha
        if record.get("response_selection_feasible") is True:
            feasible_ranks.append((index, _response_rank(record, label=f"history row {index}")))
        if index == 0:
            if (
                record.get("initialization") != "frozen_surface_control_checkpoint"
                or record.get("state_machine_role") != "frozen_control_initial_anchor"
                or "proposal" in record
                or "proposal_losses" in record
                or "loss_measurement_state" in record
                or any(
                    field in record
                    for field in _PROPOSAL_LOSS_FLAT_MIRROR.values()
                )
                or record.get("accepted") is not True
                or record.get("rejected") is not False
            ):
                raise ValueError("accepted-anchor control history row differs")
            best_updated = True
        else:
            proposal = record.get("proposal")
            if (
                not isinstance(proposal, Mapping)
                or set(proposal) != _V4_PROPOSAL_FIELDS
                or proposal.get("index") != index
                or proposal.get("source_anchor_epoch") != anchor_epoch
                or proposal.get("anchor_state_dict_sha256") != anchor_sha
                or not _sha256_text(proposal.get("raw_state_dict_sha256"))
                or not _sha256_text(proposal.get("trial_state_dict_sha256"))
                or proposal.get("alpha_numerator") != 1
                or proposal.get("alpha_denominator") != 2048
                or proposal.get("optimizer_state_reset") is not True
                or proposal.get("validation_evaluations") != 1
                or proposal.get("backtracking") != "none_fixed_alpha_single_trial"
                or proposal.get("persistent_generator")
                != "advanced_never_rolled_back"
                or record.get("state_machine_role") != "fixed_micro_ray_trial"
            ):
                raise ValueError(f"accepted-anchor proposal row {index} differs")
            proposal_losses = record.get("proposal_losses")
            if (
                record.get("loss_measurement_state")
                != "raw_proposal_before_micro_projection"
                or not isinstance(proposal_losses, Mapping)
                or tuple(proposal_losses) != _PROPOSAL_LOSS_FIELDS
                or any(
                    not isinstance(proposal_losses.get(field), (int, float))
                    or isinstance(proposal_losses.get(field), bool)
                    or not math.isfinite(float(proposal_losses[field]))
                    or float(proposal_losses[field]) < 0.0
                    or record.get(_PROPOSAL_LOSS_FLAT_MIRROR[field])
                    != proposal_losses[field]
                    for field in _PROPOSAL_LOSS_FIELDS
                )
            ):
                raise ValueError(
                    f"accepted-anchor raw-proposal loss accounting differs at row {index}"
                )
            accepted = record.get("response_selection_feasible") is True
            if record.get("accepted") is not accepted or record.get("rejected") is accepted:
                raise ValueError("accepted-anchor acceptance decision differs")
            if accepted:
                anchor_epoch = index
                anchor_sha = str(proposal["trial_state_dict_sha256"])
                accepted_count += 1
            else:
                rejected_count += 1
            selected_epoch = max(feasible_ranks, key=lambda value: value[1])[0]
            best_updated = selected_epoch == index
            if best_updated:
                best_epoch = index
                best_sha = str(proposal["trial_state_dict_sha256"])
                stale = 0
            else:
                stale += 1
        patience_stop = bool(patience and stale >= patience)
        if (
            record.get("anchor_epoch_after_proposal") != anchor_epoch
            or record.get("anchor_state_dict_sha256_after_proposal") != anchor_sha
            or record.get("best_updated") is not best_updated
            or record.get("best_epoch_after_proposal") != best_epoch
            or record.get("best_state_dict_sha256_after_proposal") != best_sha
            or record.get("patience_stale_after_proposal") != stale
            or record.get("patience_stop_after_proposal") is not patience_stop
            or (patience_stop and index != len(history) - 1)
        ):
            raise ValueError("accepted-anchor history transition differs")
    if (
        not feasible_ranks
        or checkpoint.get("best_epoch") != best_epoch
        or checkpoint.get("best_state_dict_sha256") != best_sha
        or checkpoint.get("history_hash_chain_sha256") != previous_chain
        or accepted_anchor.get("epoch") != anchor_epoch
        or accepted_anchor.get("state_dict_sha256") != anchor_sha
        or accepted_anchor.get("accepted_proposal_count") != accepted_count
        or accepted_anchor.get("rejected_proposal_count") != rejected_count
        or accepted_count + rejected_count != len(history) - 1
    ):
        raise ValueError("accepted-anchor terminal checkpoint provenance differs")
    return {
        "proposal_state_machine": dict(_PROPOSAL_STATE_MACHINE),
        "best_state_dict_sha256": str(best_state_sha),
        "accepted_anchor": dict(accepted_anchor),
        "history_hash_chain_sha256": str(previous_chain),
    }


def _pairwise_training_contract(
    config: Mapping[str, Any], *, accepted_anchor_v4: bool = False
) -> dict[str, Any]:
    expected_config_fields = (
        _V4_TRAINING_CONFIG_FIELDS
        if accepted_anchor_v4
        else _V3_TRAINING_CONFIG_FIELDS
    )
    if set(config) != expected_config_fields or (
        accepted_anchor_v4
        and config.get("proposal_state_machine") != _PROPOSAL_STATE_MACHINE
    ):
        raise ValueError("schema-v3 training config fields differ")
    integer_fields = ("hidden_dim", "epochs", "patience", "batch_size")
    float_fields = (
        "learning_rate",
        "weight_decay",
        "token_weight",
        "relation_weight",
        "canonical_noise_degrees",
    )
    if (
        any(
            not isinstance(config.get(field), int)
            or isinstance(config.get(field), bool)
            or int(config[field]) < 0
            for field in integer_fields
        )
        or int(config["hidden_dim"]) <= 0
        or int(config["batch_size"]) <= 0
        or any(
            not isinstance(config.get(field), (int, float))
            or isinstance(config.get(field), bool)
            or not math.isfinite(float(config[field]))
            for field in float_fields
        )
        or not isinstance(config.get("reliability_attention_mode"), str)
        or not isinstance(config.get("context_pooling_mode"), str)
        or not isinstance(config.get("canonical_noise_calibration"), str)
    ):
        raise ValueError("schema-v3 training config scalar contract differs")
    return {
        "hidden_dim": int(config["hidden_dim"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "token_weight": float(config["token_weight"]),
        "relation_weight": float(config["relation_weight"]),
        "reliability_attention_mode": str(config["reliability_attention_mode"]),
        "context_pooling_mode": str(config["context_pooling_mode"]),
        "canonical_noise_degrees": float(config["canonical_noise_degrees"]),
        "canonical_noise_calibration": str(config["canonical_noise_calibration"]),
        "seeds": [0, 1, 2],
        "response_lambda_source": (
            "per_seed_exact_surface_warmstart_gradient_budget"
        ),
        "response_branch_gradient_target_ratio": 0.25,
        "total_response_gradient_ratio_upper_bound": 0.5,
        "response_gradient_bound_scope": (
            "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
        ),
        "response_losses": list(_PAIRWISE_RESPONSE_LOSSES),
        "scene_response_objective": dict(_PAIRWISE_SCENE_RESPONSE_OBJECTIVE),
        "scene_tie_tolerance": 1e-6,
        "training_batching": (
            "shuffle_complete_scene_groups_no_partial_scenes_v1"
        ),
        "max_complete_scene_batch_rows": 64,
        **(
            {"proposal_state_machine": dict(_PROPOSAL_STATE_MACHINE)}
            if accepted_anchor_v4
            else {}
        ),
        "epoch_selection": (
            _ACCEPTED_ANCHOR_EPOCH_SELECTION
            if accepted_anchor_v4
            else _PAIRWISE_EPOCH_SELECTION
        ),
        "surface_control_initialization": "exact_seed_checkpoint_state_dict",
        "surface_control_noninferiority_tolerance": 0.002,
    }


def _pairwise_calibration_objective(
    training_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "surface_objective": (
            "token_weight*(1-cosine_summary_token)"
            "+masked_mean_one_minus_all_view_cosine"
            "+relation_weight*smooth_l1_descriptor_relation"
        ),
        "token_weight": training_contract["token_weight"],
        "relation_weight": training_contract["relation_weight"],
        "independent_response_loss": _INDEPENDENT_RESPONSE_LOSS,
        "scene_response_loss": _PAIRWISE_SCENE_RESPONSE_LOSS,
        "scene_response_objective": dict(_PAIRWISE_SCENE_RESPONSE_OBJECTIVE),
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
    }


def _validate_authority_materializer_source(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> None:
    authority = manifest.get("authority_contract")
    if not isinstance(authority, Mapping):
        raise ValueError("authority distill run lacks its authority contract")
    snapshot_root = _canonical_absolute_path(
        authority.get("source_snapshot_root"),
        label="authority source_snapshot_root",
    )
    if not snapshot_root.is_dir():
        raise ValueError("authority source_snapshot_root is not a directory")
    implementations = manifest.get("implementation_sources")
    producer_sha = (
        implementations.get(_MATERIALIZER_RELATIVE_PATH)
        if isinstance(implementations, Mapping)
        else None
    )
    runtime_closure = manifest.get("runtime_closure")
    repository_files = (
        runtime_closure.get("repository_sources", {}).get("files", {})
        if isinstance(runtime_closure, Mapping)
        else {}
    )
    if (
        _is_registered_legacy_formal_v3_manifest(
            manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        and snapshot_root == Path("/root/RADIO-GS")
        and producer_sha == _FORMAL_V3_PRODUCER_MATERIALIZER_SHA256
        and isinstance(repository_files, Mapping)
        and repository_files.get(_MATERIALIZER_RELATIVE_PATH) == producer_sha
    ):
        # This exact checkpoint-bound authority recorded its live execution
        # root, not an archived source tree. The frozen manifest/closure hashes
        # certify the producer while this newer consumer supplies schema-3
        # support; do not compare the producer hash to the advanced worktree.
        return
    producer_source = _canonical_absolute_path(
        str(snapshot_root / _MATERIALIZER_RELATIVE_PATH),
        label="authority producer materializer",
    )
    try:
        producer_source.relative_to(snapshot_root)
    except ValueError as error:
        raise ValueError("authority producer materializer escapes its snapshot") from error
    if not producer_source.is_file():
        raise ValueError("authority producer materializer is not a regular file")

    if (
        not isinstance(producer_sha, str)
        or len(producer_sha) != 64
        or any(character not in "0123456789abcdef" for character in producer_sha)
        or producer_sha != _sha256_file(producer_source)
    ):
        raise ValueError(
            "authority distill run binds another producer materializer implementation"
        )


def _assert_query_free(metadata: Mapping, source: str) -> None:
    for key in _QUERY_FREE_FLAGS:
        if metadata.get(key) is not False:
            raise ValueError(f"{source} must explicitly certify {key}=false")


def _legacy_region_id(record: Mapping) -> str:
    """Build a context-independent ID for pre-region_id schema-v3 caches."""

    identity = {
        "scene": record.get("scene"),
        "seed": record.get("seed"),
        "physical_radius_m": record.get("physical_radius_m"),
        "teacher_views": record.get("teacher_views"),
        "teacher_target_sha256": record.get("teacher_target_sha256"),
        "teacher_support_sha256": record.get("teacher_support_sha256"),
    }
    if not str(identity["scene"] or "") or identity["seed"] is None:
        raise ValueError("legacy region record lacks stable scene/seed identity")
    return "legacy-" + canonical_json_sha256(identity)


def _load_validation_caches(
    paths: list[Path],
    *,
    include_summary_tokens: bool = False,
) -> tuple[dict, dict]:
    if not paths:
        raise ValueError("at least one validation cache is required")
    tensor_keys = (
        "radio_features",
        "geometry",
        "token_mask",
        "reliability",
        "official_crop_summaries",
        "teacher_mask",
        "anchor_index",
    )
    if include_summary_tokens:
        # Weight-interpolation diagnostics need token fidelity as well as the
        # descriptor pair materialized by this module.  Keeping this opt-in
        # preserves the descriptor artifact schema while still loading each
        # immutable validation cache only once.
        tensor_keys = (*tensor_keys, "official_summary_tokens")
    parts = {key: [] for key in tensor_keys}
    scene_ids: list[str] = []
    region_ids: list[str] = []
    cache_records = []
    contract_hashes = set()
    radio_hashes = set()
    split_hashes = set()
    contract_specs = []
    teacher_specs = []
    excluded_spaces: list[str] | None = None
    exclusion_files: list[dict[str, str]] | None = None
    for raw_path in sorted(Path(value).resolve() for value in paths):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _torch_load(path)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path} lacks cache metadata")
        if metadata.get("schema_version") != 3 or metadata.get("split_role") != "validation":
            raise ValueError(f"{path} is not a schema-v3 validation cache")
        _assert_query_free(metadata, str(path))
        if (
            metadata.get("physical_space_disjoint") is not True
            or metadata.get("complete_scene_regions") is not True
            or metadata.get("failed_scenes")
            or metadata.get("teacher_regions_saturated") != 0
        ):
            raise ValueError(f"{path} violates the complete disjoint validation contract")
        records = metadata.get("region_records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path} lacks row-aligned region_records")
        row_count = len(records)
        for key in tensor_keys:
            tensor = torch.as_tensor(payload.get(key))
            if tensor.device.type != "cpu" or tensor.shape[0] != row_count:
                raise ValueError(f"{path} has a misaligned CPU tensor {key}")
            parts[key].append(tensor)
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"{path} contains a non-object region record")
            scene = str(record.get("scene", ""))
            region = str(record.get("region_id", "")) or _legacy_region_id(record)
            if not scene:
                raise ValueError(f"{path} contains an empty scene identity")
            scene_ids.append(scene)
            region_ids.append(region)
        local_scenes = sorted({str(record.get("scene", "")) for record in records})
        local_counts = {
            scene: sum(str(record.get("scene", "")) == scene for record in records)
            for scene in local_scenes
        }
        if (
            metadata.get("scene_names") != local_scenes
            or metadata.get("scene_region_counts") != local_counts
        ):
            raise ValueError(f"{path} has inconsistent scene metadata")
        contract_hash = str(metadata.get("region_contract_sha256", ""))
        radio_hash = str(metadata.get("radio_checkpoint_sha256", ""))
        split_hash = str(metadata.get("split_file_sha256", ""))
        if any(len(value) != 64 for value in (contract_hash, radio_hash, split_hash)):
            raise ValueError(f"{path} lacks contract/checkpoint hashes")
        contract_hashes.add(contract_hash)
        radio_hashes.add(radio_hash)
        split_hashes.add(split_hash)
        contract_specs.append(metadata.get("region_contract"))
        teacher_specs.append(
            {
                "semantics": metadata.get("teacher_region_semantics"),
                "contract": metadata.get("teacher_region_contract"),
                "contract_sha256": metadata.get("teacher_region_contract_sha256"),
                "target_source": metadata.get("teacher_target_source"),
                "target_protocol_sha256": metadata.get(
                    "teacher_target_protocol_sha256"
                ),
            }
        )
        current_spaces = list(metadata.get("excluded_physical_spaces", []))
        current_files = list(metadata.get("exclusion_files", []))
        if excluded_spaces is None:
            excluded_spaces = current_spaces
            exclusion_files = current_files
        elif excluded_spaces != current_spaces or exclusion_files != current_files:
            raise ValueError("validation cache exclusion contracts differ")
        cache_records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": row_count,
                "split_file_sha256": split_hash,
                "region_contract_sha256": contract_hash,
                "radio_checkpoint_sha256": radio_hash,
                "teacher_target_protocol_sha256": str(
                    metadata.get("teacher_target_protocol_sha256", "")
                ),
            }
        )
    if (
        len(contract_hashes) != 1
        or len(radio_hashes) != 1
        or len(split_hashes) != 1
        or any(value != contract_specs[0] for value in contract_specs[1:])
        or any(value != teacher_specs[0] for value in teacher_specs[1:])
    ):
        raise ValueError(
            "validation caches do not share one split/region/teacher/RADIO contract"
        )
    if len(set(zip(scene_ids, region_ids))) != len(scene_ids):
        raise ValueError("validation caches contain duplicate scene/region identities")
    merged = {key: torch.cat(value, dim=0) for key, value in parts.items()}
    bound_paths = [record["path"] for record in cache_records]
    cache_bindings = [
        {"path": record["path"], "sha256": record["sha256"]}
        for record in cache_records
    ]
    checkpoint_validation = {
        "scenes": sorted(set(scene_ids)),
        "split_hashes": sorted(split_hashes),
        "cache_paths": bound_paths,
        "region_contract_sha256": next(iter(contract_hashes)),
        "region_contract": contract_specs[0],
        "teacher_region": teacher_specs[0],
        "radio_checkpoint_sha256": next(iter(radio_hashes)),
        "excluded_physical_spaces": excluded_spaces or [],
        "exclusion_files": exclusion_files or [],
        "physical_space_disjoint": True,
        "cache_bindings": cache_bindings,
    }
    return merged, {
        "scene_ids": scene_ids,
        "region_ids": region_ids,
        "caches": cache_records,
        "cache_bindings": cache_bindings,
        "cache_paths": bound_paths,
        "split_hashes": sorted(split_hashes),
        "scenes": sorted(set(scene_ids)),
        "region_contract_sha256": next(iter(contract_hashes)),
        "region_contract": contract_specs[0],
        "radio_checkpoint_sha256": next(iter(radio_hashes)),
        "teacher_region": teacher_specs[0],
        "excluded_physical_spaces": excluded_spaces or [],
        "exclusion_files": exclusion_files or [],
        "checkpoint_validation": checkpoint_validation,
    }


def _teacher_descriptor(
    official_crop_summaries: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    descriptors = F.normalize(official_crop_summaries.float(), dim=-1, eps=1e-8)
    mask = teacher_mask.bool()
    if descriptors.ndim != 3 or mask.shape != descriptors.shape[:2]:
        raise ValueError("official teacher descriptors/mask are misaligned")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every region requires at least one teacher descriptor")
    weights = mask.float() / mask.sum(dim=1, keepdim=True)
    return F.normalize(
        (descriptors * weights[..., None]).sum(dim=1),
        dim=-1,
        eps=1e-8,
    )


def _validate_v3_fit_text_bank(binding: object) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != _V3_FIT_TEXT_BANK_FIELDS:
        raise ValueError("schema-v3 fit text-bank fields differ")
    artifact = _validate_file_record(
        {
            "path": binding.get("artifact_path"),
            "sha256": binding.get("artifact_sha256"),
        },
        label="schema-v3 fit text bank",
    )
    manifest = _validate_file_record(
        {
            "path": binding.get("manifest_path"),
            "sha256": binding.get("manifest_sha256"),
        },
        label="schema-v3 fit text-bank manifest",
    )
    digest_fields = _V3_FIT_TEXT_BANK_FIELDS - {
        "artifact_path",
        "artifact_sha256",
        "manifest_path",
        "manifest_sha256",
        "split",
        "query_count",
    }
    if (
        binding.get("split") != "fit"
        or not isinstance(binding.get("query_count"), int)
        or isinstance(binding.get("query_count"), bool)
        or int(binding["query_count"]) <= 0
        or any(
            not isinstance(binding.get(field), str)
            or len(str(binding[field])) != 64
            or any(character not in "0123456789abcdef" for character in str(binding[field]))
            for field in digest_fields
        )
    ):
        raise ValueError("schema-v3 fit text-bank immutable contract differs")
    return {
        **dict(binding),
        "artifact_path": artifact["path"],
        "manifest_path": manifest["path"],
    }


def _validate_v3_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    distillation: Mapping[str, Any],
    *,
    seed: int,
    cache_meta: Mapping[str, Any],
    legacy_protocol: bool,
) -> dict[str, Any]:
    raw_warm_start = provenance.get("surface_control_warm_start")
    accepted_anchor_v4 = (
        not legacy_protocol
        and isinstance(raw_warm_start, Mapping)
        and raw_warm_start.get("selection_policy")
        == _ACCEPTED_ANCHOR_EPOCH_SELECTION
    )
    expected_provenance_fields = (
        _LEGACY_V3_RESPONSE_PROVENANCE_FIELDS
        if legacy_protocol
        else _V3_RESPONSE_PROVENANCE_FIELDS
    )
    expected_losses = (
        _LEGACY_RESPONSE_LOSSES
        if legacy_protocol
        else _PAIRWISE_RESPONSE_LOSSES
    )
    if (
        set(distillation) != expected_provenance_fields
        or distillation.get("fit_split_only") is not True
        or distillation.get("benchmark_vocabulary_opened") is not False
        or distillation.get("calibration_seed") != seed
        or distillation.get("response_branch_gradient_target_ratio") != 0.25
        or distillation.get("total_response_gradient_ratio_upper_bound") != 0.5
        or distillation.get("losses") != expected_losses
        or (
            not legacy_protocol
            and distillation.get("scene_response_objective")
            != _PAIRWISE_SCENE_RESPONSE_OBJECTIVE
        )
        or distillation.get("complete_scene_batching") is not True
    ):
        raise ValueError("schema-v3 text-response provenance differs")
    lambdas = distillation.get("response_lambdas")
    if (
        not isinstance(lambdas, Mapping)
        or set(lambdas) != _V3_RESPONSE_LAMBDA_FIELDS
        or not all(_positive_finite(value) for value in lambdas.values())
    ):
        raise ValueError("schema-v3 response lambdas differ")
    design = distillation.get("design_diagnostic")
    if (
        not isinstance(design, Mapping)
        or set(design) != _V3_DESIGN_DIAGNOSTIC_FIELDS
        or design.get("role")
        != "seed0_design_prior_only_per_seed_values_remeasured"
        or design.get("measured_seed") != 0
        or design.get("calibration_reuses_measured_values") is not False
    ):
        raise ValueError("schema-v3 gradient design diagnostic differs")
    diagnostic_record = _validate_file_record(
        {"path": design.get("path"), "sha256": design.get("sha256")},
        label="schema-v3 gradient design diagnostic",
    )

    surface = checkpoint.get("surface_control_checkpoint")
    if not isinstance(surface, Mapping) or set(surface) != _V3_SURFACE_CONTROL_FIELDS:
        raise ValueError("schema-v3 Surface control binding fields differ")
    train_caches = _cache_bindings(provenance.get("train"), label="readout train")
    validation_caches = _cache_bindings(
        provenance.get("validation"), label="readout validation"
    )
    if (
        surface.get("seed") != seed
        or surface.get("architecture") != checkpoint.get("architecture")
        or surface.get("train_caches") != train_caches
        or surface.get("validation_caches") != validation_caches
        or validation_caches != cache_meta["cache_bindings"]
        or not isinstance(surface.get("source_best_epoch"), int)
        or isinstance(surface.get("source_best_epoch"), bool)
        or not isinstance(surface.get("source_best_selection_score"), (int, float))
        or isinstance(surface.get("source_best_selection_score"), bool)
        or not math.isfinite(float(surface["source_best_selection_score"]))
    ):
        raise ValueError("schema-v3 Surface control seed/cache/architecture differs")
    control_record = _validate_file_record(
        {"path": surface.get("path"), "sha256": surface.get("sha256")},
        label="schema-v3 Surface control checkpoint",
    )
    control_payload = _torch_load(Path(control_record["path"]))
    control_provenance = control_payload.get("provenance")
    control_config = control_payload.get("training_config")
    expected_control_train = dict(provenance["train"])
    expected_control_train.pop("cache_bindings", None)
    expected_control_validation = dict(provenance["validation"])
    expected_control_validation.pop("cache_bindings", None)
    if (
        control_payload.get("architecture") != checkpoint.get("architecture")
        or not isinstance(control_provenance, Mapping)
        or not isinstance(control_config, Mapping)
        or control_config.get("seed") != seed
        or control_provenance.get("random_seed_contract")
        != {
            "seed": seed,
            "model_initialization": True,
            "data_order": True,
            "canonical_noise": True,
        }
        or control_provenance.get("uses_benchmark_scenes") is not False
        or control_provenance.get("uses_benchmark_test_vocabulary") is not False
        or control_provenance.get("train") != expected_control_train
        or control_provenance.get("validation") != expected_control_validation
        or control_provenance.get("region_contract")
        != provenance.get("region_contract")
        or control_provenance.get("region_contract_sha256")
        != provenance.get("region_contract_sha256")
        or control_payload.get("best_epoch") != surface.get("source_best_epoch")
        or not math.isclose(
            float(control_payload.get("best_selection_score")),
            float(surface["source_best_selection_score"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("schema-v3 frozen Surface control payload differs")
    expected_seed_contract = {
        "seed": seed,
        "model_initialization": False,
        "model_initialization_source": "frozen_seed_surface_control",
        "data_order": True,
        "canonical_noise": True,
    }
    warm_start = raw_warm_start
    expected_warm_start = {
        **dict(surface),
        "epoch": 0,
        "noninferiority_metrics": [
            "summary_token_cosine",
            "mean_descriptor_cosine",
            "all_view_descriptor_cosine",
        ],
        "noninferiority_tolerance": 0.002,
        "selection_policy": (
            _LEGACY_EPOCH_SELECTION
            if legacy_protocol
            else (
                _ACCEPTED_ANCHOR_EPOCH_SELECTION
                if accepted_anchor_v4
                else _PAIRWISE_EPOCH_SELECTION
            )
        ),
    }
    config = checkpoint.get("training_config")
    config_surface_value = (
        config.get("surface_control_checkpoint")
        if isinstance(config, Mapping)
        else None
    )
    config_surface_path = (
        Path(_configured_path_text(
            config_surface_value,
            label="schema-v3 training Surface control checkpoint",
        ))
        if isinstance(config, Mapping)
        else None
    )
    if (
        provenance.get("random_seed_contract") != expected_seed_contract
        or warm_start != expected_warm_start
        or not isinstance(config, Mapping)
        or config_surface_path != Path(surface["path"])
        or config.get("surface_control_checkpoint_sha256") != surface["sha256"]
    ):
        raise ValueError("schema-v3 Surface warm-start provenance differs")
    state_machine = None
    if accepted_anchor_v4:
        patience_value = config.get("patience")
        if (
            not isinstance(patience_value, int)
            or isinstance(patience_value, bool)
            or patience_value < 0
        ):
            raise ValueError("accepted-anchor patience contract differs")
        state_machine = _validate_v4_checkpoint_state_machine(
            checkpoint,
            provenance,
            control_payload,
            patience=patience_value,
        )
    elif (
        any(field in checkpoint for field in _V4_CHECKPOINT_STATE_FIELDS)
        or "proposal_state_machine" in provenance
    ):
        raise ValueError("non-v4 checkpoint contains accepted-anchor state fields")
    fit_bank = _validate_v3_fit_text_bank(distillation.get("fit_text_bank"))
    training_contract = None
    config_bindings = None
    if not legacy_protocol:
        training_contract = _pairwise_training_contract(
            config, accepted_anchor_v4=accepted_anchor_v4
        )
        config_bindings = {
            "train_caches": _path_pattern_or_list(
                config.get("train_caches"),
                label="schema-v3 training train caches",
            ),
            "validation_caches": _path_pattern_or_list(
                config.get("validation_caches"),
                label="schema-v3 training validation caches",
            ),
            "fit_text_bank": _configured_path_text(
                config.get("fit_text_bank"),
                label="schema-v3 training fit text bank",
            ),
            "fit_text_bank_manifest": _configured_path_text(
                config.get("fit_text_bank_manifest"),
                label="schema-v3 training fit text-bank manifest",
            ),
            "calibration_manifest": _configured_path_text(
                config.get("calibration_manifest"),
                label="schema-v3 training calibration manifest",
            ),
            "run_manifest": _configured_path_text(
                config.get("run_manifest"),
                label="schema-v3 training run manifest",
            ),
            "surface_control_checkpoint": config_surface_path,
            "output": _configured_path_text(
                config.get("output"),
                label="schema-v3 training output",
            ),
            "radio_checkpoint": _configured_path_text(
                config.get("radio_checkpoint"),
                label="schema-v3 training RADIO checkpoint",
            ),
        }
        if (
            config.get("seed") != seed
            or not isinstance(config.get("device"), str)
            or not config["device"]
            or config_bindings["train_caches"]
            != [record["path"] for record in train_caches]
            or config_bindings["validation_caches"]
            != [record["path"] for record in validation_caches]
            or config_bindings["fit_text_bank"] != fit_bank["artifact_path"]
            or config_bindings["fit_text_bank_manifest"]
            != fit_bank["manifest_path"]
            or config_bindings["calibration_manifest"]
            != str(
                _canonical_absolute_path(
                    distillation.get("calibration_manifest"),
                    label="schema-v3 provenance calibration manifest",
                )
            )
        ):
            raise ValueError("schema-v3 training config/provenance binding differs")
    return {
        "protocol": (
            "legacy_registered_v2"
            if legacy_protocol
            else "accepted_anchor_v4" if accepted_anchor_v4 else "pairwise_v3"
        ),
        "surface_control": dict(surface),
        "response_lambdas": dict(lambdas),
        "calibration_manifest": distillation.get("calibration_manifest"),
        "calibration_manifest_sha256": distillation.get(
            "calibration_manifest_sha256"
        ),
        "design_diagnostic": dict(design),
        "diagnostic_record": diagnostic_record,
        "fit_text_bank": fit_bank,
        "train_caches": train_caches,
        "validation_caches": validation_caches,
        "training_contract": training_contract,
        "config_bindings": config_bindings,
        "state_machine": state_machine,
    }


def _validate_v3_calibration_payload(
    path: Path,
    *,
    seed: int,
    contract: Mapping[str, Any],
    radio_path: Path,
    radio_sha256: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    legacy_protocol = contract.get("protocol") == "legacy_registered_v2"
    if contract.get("protocol") not in {
        "legacy_registered_v2",
        "pairwise_v3",
        "accepted_anchor_v4",
    }:
        raise ValueError("schema-v3 calibration protocol is unbound")
    expected_fields = {
        "schema_version", "artifact_type", "algorithm_version",
        "benchmark_vocabulary_opened", "uses_benchmark_scenes",
        "uses_benchmark_test_vocabulary", "seed", "surface_control",
        "fixed_calibration_scene_batch", "objective_contract",
        "gradient_contract", "design_diagnostic", "architecture",
        "train_caches", "validation_caches", "train_contract",
        "radio_checkpoint", "fit_text_bank", "implementation",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_fields
        or payload.get("schema_version") != 2
        or payload.get("artifact_type") != "surface_text_response_gradient_calibration"
        or payload.get("algorithm_version")
        != (
            _LEGACY_CALIBRATION_ALGORITHM
            if legacy_protocol
            else _PAIRWISE_CALIBRATION_ALGORITHM
        )
        or any(
            payload.get(field) is not False
            for field in (
                "benchmark_vocabulary_opened",
                "uses_benchmark_scenes",
                "uses_benchmark_test_vocabulary",
            )
        )
        or payload.get("seed") != seed
        or payload.get("surface_control") != contract["surface_control"]
        or payload.get("architecture") != contract["surface_control"]["architecture"]
        or payload.get("train_caches") != contract["train_caches"]
        or payload.get("validation_caches") != contract["validation_caches"]
        or payload.get("radio_checkpoint")
        != {"path": str(radio_path), "sha256": radio_sha256}
        or payload.get("fit_text_bank") != contract["fit_text_bank"]
        or payload.get("design_diagnostic") != contract["design_diagnostic"]
    ):
        raise ValueError("schema-v3 per-seed calibration immutable binding differs")
    batch = payload.get("fixed_calibration_scene_batch")
    if (
        not isinstance(batch, Mapping)
        or batch.get("split_role") != "train"
        or batch.get("scene_selection_algorithm")
        != "lexicographically_first_complete_train_scenes_v1"
        or batch.get("requested_scene_count") != 4
        or not isinstance(batch.get("scenes"), list)
        or len(batch["scenes"]) != 4
        or len(set(batch["scenes"])) != 4
        or batch.get("complete_scenes") is not True
        or batch.get("augmentation") != "none"
        or not isinstance(batch.get("scene_row_counts"), Mapping)
        or set(batch["scene_row_counts"]) != set(batch["scenes"])
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 2
            for value in batch["scene_row_counts"].values()
        )
        or batch.get("effective_row_count")
        != sum(batch["scene_row_counts"].values())
        or not isinstance(batch.get("row_indices"), list)
        or len(batch["row_indices"]) != batch["effective_row_count"]
        or len(set(batch["row_indices"])) != len(batch["row_indices"])
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in batch["row_indices"]
        )
    ):
        raise ValueError("schema-v3 calibration complete-scene batch differs")
    objective = payload.get("objective_contract")
    if legacy_protocol:
        if (
            not isinstance(objective, Mapping)
            or objective.get("gradient_bound_scope")
            != "local_at_unaugmented_exact_warmstart_not_a_global_training_bound"
            or objective.get("training_batching")
            != "shuffle_complete_scene_groups_no_partial_scenes_v1"
            or objective.get("max_complete_scene_batch_rows") != 64
        ):
            raise ValueError("schema-v3 legacy calibration objective differs")
    elif objective != _pairwise_calibration_objective(
        contract["training_contract"]
    ):
        raise ValueError("schema-v3 pairwise calibration objective differs")
    gradient = payload.get("gradient_contract")
    gradient_fields = {
        "parameter_set", "measurement_point", "norm", "epsilon",
        "loss_values", "gradient_l2", "branch_target_ratio",
        "trainable_parameter_count", "trainable_parameters",
        "response_lambdas", "weighted_response_gradient_l2",
        "combined_response_gradient_l2_upper_bound",
        "combined_response_to_surface_upper_bound_ratio",
    }
    if (
        not isinstance(gradient, Mapping)
        or set(gradient) != gradient_fields
        or gradient.get("parameter_set")
        != "all_trainable_surface_region_summary_readout_v2_parameters"
        or gradient.get("measurement_point")
        != "exact_seed_frozen_surface_control_state_dict"
        or gradient.get("norm") != "joint_parameter_gradient_l2"
        or gradient.get("epsilon") != 1e-12
        or gradient.get("branch_target_ratio") != 0.25
        or gradient.get("response_lambdas") != contract["response_lambdas"]
    ):
        raise ValueError("schema-v3 calibration gradient contract differs")
    inventory = gradient.get("trainable_parameters")
    losses = gradient.get("loss_values")
    norms = gradient.get("gradient_l2")
    weighted = gradient.get("weighted_response_gradient_l2")
    branches = _V3_RESPONSE_LAMBDA_FIELDS
    if (
        not isinstance(inventory, list)
        or len(inventory) != gradient.get("trainable_parameter_count")
        or len({row.get("name") for row in inventory if isinstance(row, Mapping)})
        != len(inventory)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"name", "shape"}
            or not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(row["shape"], list)
            or any(not isinstance(size, int) or size < 0 for size in row["shape"])
            for row in inventory
        )
        or not isinstance(losses, Mapping)
        or set(losses)
        != {
            "surface", "token", "descriptor", "relation",
            "independent_response", "scene_response", "scene_profile",
            "scene_ranking",
        }
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in losses.values()
        )
        or not isinstance(norms, Mapping)
        or set(norms) != {"surface", *branches}
        or not isinstance(weighted, Mapping)
        or set(weighted) != branches
        or any(not _positive_finite(value) for value in [*norms.values(), *weighted.values()])
    ):
        raise ValueError("schema-v3 calibration gradient measurements differ")
    surface_norm = float(norms["surface"])
    for branch in branches:
        response_norm = float(norms[branch])
        expected_lambda = 0.25 * surface_norm / response_norm
        if (
            not math.isclose(
                float(contract["response_lambdas"][branch]),
                expected_lambda,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(weighted[branch]),
                0.25 * surface_norm,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("schema-v3 calibration response budget differs")
    combined = sum(float(weighted[branch]) for branch in branches)
    if (
        not math.isclose(
            float(gradient.get("combined_response_gradient_l2_upper_bound")),
            combined,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(gradient.get("combined_response_to_surface_upper_bound_ratio")),
            0.5,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("schema-v3 calibration combined response budget differs")


def _validate_distill_run_manifest(
    binding: object,
    *,
    checkpoint_path: Path,
    report_path: Path,
    seed: int,
    cache_meta: Mapping[str, Any],
    radio_path: Path,
    radio_sha256: str,
    checkpoint_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "candidate",
    }:
        raise ValueError("readout checkpoint lacks an exact distill run-manifest binding")
    bound_manifest_path = Path(str(binding.get("path", "")))
    manifest_path = bound_manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("bound distill run manifest is missing")
    manifest_sha = _sha256_file(manifest_path)
    if binding.get("sha256") != manifest_sha:
        raise ValueError("distill run-manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or type(schema_version) is not int
        or schema_version not in {1, 2, 3}
        or manifest.get("artifact_type") != _DISTILL_RUN_ARTIFACT_TYPE
    ):
        raise ValueError("invalid distill run-manifest schema")
    registered_legacy_v3 = (
        schema_version == 3
        and _is_registered_legacy_formal_v3_manifest(
            manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
        )
    )
    expected_authority_fields = (
        _AUTHORITY_DISTILL_MANIFEST_FIELDS_V2
        if schema_version == 2
        else _AUTHORITY_DISTILL_MANIFEST_FIELDS_V3
    )
    if schema_version in {2, 3} and set(manifest) != expected_authority_fields:
        raise ValueError("authority distill run-manifest fields differ")
    if schema_version in {2, 3} and (
        not bound_manifest_path.is_absolute() or bound_manifest_path != manifest_path
    ):
        raise ValueError(
            "authority distill run-manifest path must be canonical and non-symlinked"
        )
    if manifest.get("candidate") != binding.get("candidate"):
        raise ValueError("distill checkpoint/run candidate binding differs")
    if manifest.get("validation_caches") != cache_meta["cache_bindings"]:
        raise ValueError("distill run manifest binds different validation caches")
    if manifest.get("radio_checkpoint") != {
        "path": str(radio_path),
        "sha256": radio_sha256,
    }:
        raise ValueError("distill run manifest binds a different RADIO checkpoint")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("distill run manifest lacks its seed output index")
    expected_output = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
    }
    if schema_version == 1:
        rows = [
            value
            for value in outputs
            if isinstance(value, Mapping) and value.get("seed") == seed
        ]
        if rows != [expected_output]:
            raise ValueError("distill run manifest binds another checkpoint/report")
    else:
        if (
            len(outputs) != 3
            or any(
                not isinstance(row, Mapping)
                or set(row) != _AUTHORITY_DISTILL_OUTPUT_FIELDS
                or type(row.get("seed")) is not int
                for row in outputs
            )
        ):
            raise ValueError("authority distill output index fields differ")
        by_seed = {row["seed"]: row for row in outputs}
        if set(by_seed) != {0, 1, 2}:
            raise ValueError("authority distill output seeds differ")
        for row in outputs:
            for field in _AUTHORITY_DISTILL_OUTPUT_FIELDS - {"seed"}:
                value = row.get(field)
                if not isinstance(value, str) or not value or not Path(value).is_absolute():
                    raise ValueError(
                        f"authority distill output {field} must be an absolute path"
                    )
                if str(Path(value)) != value:
                    raise ValueError(
                        f"authority distill output {field} must be a canonical path"
                    )
        row = by_seed[seed]
        if any(row.get(key) != value for key, value in expected_output.items()):
            raise ValueError("distill run manifest binds another checkpoint/report")
    if schema_version == 3:
        if checkpoint_contract is None:
            raise ValueError("schema-v3 distill run lacks checkpoint calibration contract")
        if (
            checkpoint_contract.get("protocol") == "legacy_registered_v2"
        ) != registered_legacy_v3:
            raise ValueError("schema-v3 checkpoint/run protocol binding differs")
        if not registered_legacy_v3:
            config_bindings = checkpoint_contract.get("config_bindings")
            if (
                manifest.get("training_contract")
                != checkpoint_contract.get("training_contract")
                or not isinstance(config_bindings, Mapping)
                or config_bindings.get("run_manifest") != str(manifest_path)
                or config_bindings.get("output") != str(checkpoint_path)
                or config_bindings.get("radio_checkpoint") != str(radio_path)
                or manifest.get("train_caches")
                != checkpoint_contract.get("train_caches")
                or manifest.get("validation_caches")
                != checkpoint_contract.get("validation_caches")
            ):
                raise ValueError(
                    "schema-v3 checkpoint/config/manifest training contract differs"
                )
        calibration_rows = manifest.get("calibrations")
        if (
            not isinstance(calibration_rows, list)
            or len(calibration_rows) != 3
            or any(
                not isinstance(value, Mapping)
                or set(value)
                != {
                    "seed", "manifest", "audit", "surface_control",
                    "response_lambdas",
                }
                or type(value.get("seed")) is not int
                for value in calibration_rows
            )
        ):
            raise ValueError("schema-v3 per-seed calibration index differs")
        calibration_by_seed = {value["seed"]: value for value in calibration_rows}
        if set(calibration_by_seed) != {0, 1, 2}:
            raise ValueError("schema-v3 per-seed calibration seeds differ")
        for calibration_seed, calibration_row in calibration_by_seed.items():
            manifest_record = _validate_file_record(
                calibration_row["manifest"],
                label=f"schema-v3 seed-{calibration_seed} calibration manifest",
            )
            _validate_file_record(
                calibration_row["audit"],
                label=f"schema-v3 seed-{calibration_seed} calibration audit",
            )
            surface = calibration_row["surface_control"]
            lambdas = calibration_row["response_lambdas"]
            if (
                not isinstance(surface, Mapping)
                or set(surface) != _V3_SURFACE_CONTROL_FIELDS
                or surface.get("seed") != calibration_seed
                or not isinstance(lambdas, Mapping)
                or set(lambdas) != _V3_RESPONSE_LAMBDA_FIELDS
                or not all(_positive_finite(value) for value in lambdas.values())
            ):
                raise ValueError(
                    f"schema-v3 seed-{calibration_seed} calibration/control differs"
                )
            _validate_file_record(
                {"path": surface.get("path"), "sha256": surface.get("sha256")},
                label=f"schema-v3 seed-{calibration_seed} Surface control",
            )
            row_contract = {
                **dict(checkpoint_contract),
                "surface_control": dict(surface),
                "response_lambdas": dict(lambdas),
            }
            _validate_v3_calibration_payload(
                Path(manifest_record["path"]),
                seed=calibration_seed,
                contract=row_contract,
                radio_path=radio_path,
                radio_sha256=radio_sha256,
            )
            if calibration_seed == seed:
                expected_manifest = {
                    "path": str(
                        _canonical_absolute_path(
                            checkpoint_contract.get("calibration_manifest"),
                            label="checkpoint calibration manifest",
                        )
                    ),
                    "sha256": checkpoint_contract.get("calibration_manifest_sha256"),
                }
                if (
                    manifest_record != expected_manifest
                    or surface != checkpoint_contract["surface_control"]
                    or dict(lambdas) != checkpoint_contract["response_lambdas"]
                ):
                    raise ValueError(
                        f"schema-v3 seed-{seed} checkpoint/calibration binding differs"
                    )
        diagnostic_record = _validate_file_record(
            manifest.get("gradient_design_diagnostic"),
            label="schema-v3 gradient design diagnostic",
        )
        if (
            diagnostic_record != checkpoint_contract["diagnostic_record"]
            or checkpoint_contract["design_diagnostic"].get(
                "diagnostic_surface_control"
            )
            != {
                "path": calibration_by_seed[0]["surface_control"]["path"],
                "sha256": calibration_by_seed[0]["surface_control"]["sha256"],
            }
            or manifest.get("fit_text_bank")
            != {
                "artifact": {
                    "path": checkpoint_contract["fit_text_bank"]["artifact_path"],
                    "sha256": checkpoint_contract["fit_text_bank"]["artifact_sha256"],
                },
                "manifest": {
                    "path": checkpoint_contract["fit_text_bank"]["manifest_path"],
                    "sha256": checkpoint_contract["fit_text_bank"]["manifest_sha256"],
                },
            }
        ):
            raise ValueError("schema-v3 shared diagnostic/fit-bank binding differs")
    implementations = manifest.get("implementation_sources")
    if schema_version == 1:
        source = Path(__file__).resolve()
        if (
            not isinstance(implementations, Mapping)
            or implementations.get(_MATERIALIZER_RELATIVE_PATH)
            != _sha256_file(source)
        ):
            raise ValueError(
                "distill run manifest binds another materializer implementation"
            )
    else:
        _validate_authority_materializer_source(
            manifest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
        )
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "candidate": str(binding["candidate"]),
        "schema_version": schema_version,
    }


def _validate_checkpoint_report(
    report_path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: Mapping[str, Any],
    cache_meta: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> dict[str, str]:
    if not report_path.is_file():
        raise FileNotFoundError("readout checkpoint report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("readout checkpoint report must contain an object")
    provenance = checkpoint["provenance"]
    distillation = provenance.get("text_response_distillation")
    if not isinstance(distillation, Mapping):
        raise ValueError("readout checkpoint lacks text-response provenance")
    common_required = {
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
    v3_response = run_manifest.get("schema_version") == 3
    v4_response = v3_response and (
        checkpoint.get("proposal_state_machine") == _PROPOSAL_STATE_MACHINE
    )
    required = common_required | (
        {
            "surface_control_checkpoint",
            "surface_control_validation",
            "surface_control_score",
            "response_lambdas",
            "complete_scene_batching",
        }
        if v3_response
        else {"untrained_baseline", "response_lambda"}
    )
    if v4_response:
        required.update(_V4_CHECKPOINT_STATE_FIELDS)
    if set(report) != required:
        raise ValueError("readout checkpoint report fields differ from the fixed schema")
    baseline_field = (
        "surface_control_validation" if v3_response else "untrained_baseline"
    )
    baseline = checkpoint.get(baseline_field)
    if not isinstance(baseline, Mapping) or set(baseline) != {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    }:
        raise ValueError(f"readout checkpoint lacks its {baseline_field}")
    try:
        baseline_score = 0.5 * (
            float(baseline["mean_descriptor_cosine"])
            + float(baseline["all_view_descriptor_cosine"])
        )
    except (TypeError, ValueError) as error:
        raise ValueError("readout checkpoint baseline metrics are invalid") from error
    best_score = float(checkpoint.get("best_selection_score"))
    expected = {
        "output": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": checkpoint.get("architecture"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_selection_score": best_score,
        "selection_score_delta": best_score - baseline_score,
        "calibration_manifest": distillation.get("calibration_manifest"),
        "calibration_manifest_sha256": distillation.get(
            "calibration_manifest_sha256"
        ),
        "fit_text_bank_sha256": distillation.get("fit_text_bank", {}).get(
            "artifact_sha256"
        ),
        "fit_query_count": distillation.get("fit_text_bank", {}).get(
            "query_count"
        ),
        "distill_run_manifest": run_manifest["path"],
        "distill_run_manifest_sha256": run_manifest["sha256"],
        "validation_caches": cache_meta["cache_bindings"],
        "train_scenes": len(provenance["train"]["scenes"]),
        "validation_scenes": len(cache_meta["scenes"]),
        "scene_overlap": [],
    }
    if v3_response:
        batching = checkpoint.get("complete_scene_batching")
        if (
            not isinstance(batching, Mapping)
            or set(batching)
            != {
                "algorithm",
                "row_limit",
                "observed_peak_rows",
                "observed_batch_count",
            }
            or batching.get("algorithm")
            != "shuffle_complete_scene_groups_no_partial_scenes_v1"
            or batching.get("row_limit") != 64
            or not isinstance(batching.get("observed_peak_rows"), int)
            or isinstance(batching.get("observed_peak_rows"), bool)
            or not 1 < batching["observed_peak_rows"] <= 64
            or not isinstance(batching.get("observed_batch_count"), int)
            or isinstance(batching.get("observed_batch_count"), bool)
            or batching["observed_batch_count"] <= 0
        ):
            raise ValueError("schema-v3 complete-scene batching differs")
        expected.update(
            {
                "surface_control_checkpoint": checkpoint.get(
                    "surface_control_checkpoint"
                ),
                "surface_control_validation": baseline,
                "surface_control_score": baseline_score,
                "response_lambdas": distillation.get("response_lambdas"),
                "complete_scene_batching": batching,
            }
        )
        if v4_response:
            expected.update(
                {
                    "best_state_dict_sha256": checkpoint.get(
                        "best_state_dict_sha256"
                    ),
                    "proposal_state_machine": checkpoint.get(
                        "proposal_state_machine"
                    ),
                    "accepted_anchor": checkpoint.get("accepted_anchor"),
                    "history_hash_chain_sha256": checkpoint.get(
                        "history_hash_chain_sha256"
                    ),
                }
            )
        if not math.isclose(
            float(checkpoint.get("surface_control_score")),
            baseline_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("schema-v3 Surface control score differs")
    else:
        expected.update(
            {
                "untrained_baseline": baseline,
                "response_lambda": distillation.get("response_lambda"),
            }
        )
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"readout checkpoint report {key} binding differs")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or set(validation) != {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    }:
        raise ValueError("readout checkpoint report validation metrics differ")
    validation_score = 0.5 * (
        float(validation["mean_descriptor_cosine"])
        + float(validation["all_view_descriptor_cosine"])
    )
    if not math.isclose(validation_score, best_score, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("readout report validation does not reproduce the best score")
    return {"path": str(report_path), "sha256": _sha256_file(report_path)}


def _validate_legacy_report(
    report_path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: Mapping[str, Any],
    cache_meta: Mapping[str, Any],
) -> dict[str, str]:
    if not report_path.is_file():
        raise FileNotFoundError("legacy readout checkpoint report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "output",
        "checkpoint_sha256",
        "architecture",
        "best_epoch",
        "best_selection_score",
        "untrained_baseline",
        "selection_score_delta",
        "validation",
        "train_scenes",
        "validation_scenes",
        "scene_overlap",
    }
    if not isinstance(report, Mapping) or set(report) != required:
        raise ValueError("legacy readout report fields differ from the fixed schema")
    baseline = checkpoint.get("untrained_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("legacy readout checkpoint lacks its baseline")
    baseline_score = 0.5 * (
        float(baseline["mean_descriptor_cosine"])
        + float(baseline["all_view_descriptor_cosine"])
    )
    best_score = float(checkpoint.get("best_selection_score"))
    expected = {
        "output": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": checkpoint.get("architecture"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_selection_score": best_score,
        "untrained_baseline": baseline,
        "selection_score_delta": best_score - baseline_score,
        "train_scenes": len(checkpoint["provenance"]["train"]["scenes"]),
        "validation_scenes": len(cache_meta["scenes"]),
        "scene_overlap": [],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"legacy readout report {key} binding differs")
    validation = report.get("validation")
    if not isinstance(validation, Mapping) or set(validation) != {
        "summary_token_cosine",
        "mean_descriptor_cosine",
        "all_view_descriptor_cosine",
    }:
        raise ValueError("legacy readout report validation metrics differ")
    validation_score = 0.5 * (
        float(validation["mean_descriptor_cosine"])
        + float(validation["all_view_descriptor_cosine"])
    )
    if not math.isclose(validation_score, best_score, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("legacy readout validation does not reproduce its best score")
    return {"path": str(report_path), "sha256": _sha256_file(report_path)}


def _validate_legacy_promotion_bundle(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    report_path: Path,
    seed: int,
    cache_meta: Mapping[str, Any],
) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("legacy readout requires a query-free promotion bundle")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("schema_version") != 1
        or bundle.get("artifact_type")
        != "surface_region_query_free_three_seed_bundle"
        or bundle.get("status")
        != "query_free_three_seed_bundle_frozen_benchmark_gate_closed"
        or bundle.get("seed_selection_policy")
        != "all_required_seeds_no_single_seed_selection"
        or bundle.get("required_seeds") != [0, 1, 2]
    ):
        raise ValueError("invalid legacy query-free promotion bundle")
    candidate = str(bundle.get("selected_candidate", ""))
    if not candidate:
        raise ValueError("legacy promotion bundle lacks a selected candidate")
    gate = bundle.get("benchmark_gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "closed_not_evaluated"
        or gate.get("main_result_eligible") is not False
    ):
        raise ValueError("legacy promotion bundle has an open downstream gate")

    completion_path = path.parent / "query_free_promotion.complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError("legacy promotion completion is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        not isinstance(completion, Mapping)
        or completion.get("artifact_type")
        != "surface_region_query_free_promotion_completion"
        or completion.get("promotion_manifest") != str(path)
        or completion.get("promotion_manifest_sha256") != _sha256_file(path)
        or completion.get("selected_candidate") != candidate
        or completion.get("required_seeds") != [0, 1, 2]
        or completion.get("main_result_eligible") is not False
    ):
        raise ValueError("legacy promotion completion does not bind its bundle")

    bindings = bundle.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("legacy promotion bundle lacks artifact bindings")
    for label in (
        "finalizer",
        "run_manifest",
        "cache_pairing",
        "query_free_screen",
        "screen_completion",
    ):
        record = bindings.get(label)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"legacy promotion bundle lacks {label} binding")
        bound = Path(str(record["path"])).resolve()
        if not bound.is_file() or record["sha256"] != _sha256_file(bound):
            raise ValueError(f"legacy promotion bundle {label} SHA256 mismatch")

    readouts = bindings.get("all_compared_readouts")
    if not isinstance(readouts, list):
        raise ValueError("legacy promotion bundle lacks compared readouts")
    matches = [
        value
        for value in readouts
        if isinstance(value, Mapping)
        and value.get("candidate") == candidate
        and value.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("legacy promotion bundle lacks the selected candidate/seed")
    readout = matches[0]
    if (
        Path(str(readout.get("checkpoint", ""))).resolve() != checkpoint_path
        or readout.get("checkpoint_sha256") != checkpoint_sha256
        or Path(str(readout.get("sidecar", ""))).resolve() != report_path
        or readout.get("sidecar_sha256") != _sha256_file(report_path)
    ):
        raise ValueError("legacy promotion bundle readout binding differs")
    selected_rows = bundle.get("selected_readouts")
    compared_selected = [
        value
        for value in readouts
        if isinstance(value, Mapping) and value.get("candidate") == candidate
    ]
    if (
        not isinstance(selected_rows, list)
        or selected_rows != compared_selected
        or len(selected_rows) != 3
        or {value.get("seed") for value in selected_rows} != {0, 1, 2}
        or not any(
            value.get("seed") == seed
            and value.get("checkpoint_sha256") == checkpoint_sha256
            for value in selected_rows
        )
    ):
        raise ValueError("legacy readout is not in the selected three-seed bundle")

    cache_rows = bindings.get("caches")
    if not isinstance(cache_rows, list):
        raise ValueError("legacy promotion bundle lacks cache bindings")
    selected_validation = [
        value
        for value in cache_rows
        if isinstance(value, Mapping)
        and value.get("candidate") == candidate
        and value.get("role") == "validation"
    ]
    selected_validation.sort(key=lambda value: int(value.get("shard", -1)))
    expected_caches = [
        {"path": value.get("path"), "sha256": value.get("sha256")}
        for value in selected_validation
    ]
    if expected_caches != cache_meta["cache_bindings"]:
        raise ValueError("legacy promotion bundle binds different validation caches")
    for value in selected_validation:
        sidecar = Path(str(value.get("sidecar", ""))).resolve()
        if (
            not sidecar.is_file()
            or value.get("sidecar_sha256") != _sha256_file(sidecar)
        ):
            raise ValueError("legacy promotion cache sidecar SHA256 mismatch")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "completion": str(completion_path),
        "completion_sha256": _sha256_file(completion_path),
        "candidate": candidate,
    }


def _validate_attention_postcache_binding(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    report_path: Path,
    seed: int,
    cache_meta: Mapping[str, Any],
) -> dict[str, str]:
    from radio_gs.scripts import surface_text_response_distill_authority as authority

    path = Path(path).resolve()
    screen = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(screen, Mapping)
        or screen.get("artifact_type")
        != "surface_c1024_attention_pooling_postcache_continuation"
        or screen.get("selected_variant") != "joint_attention_v1"
        or screen.get("selection_status") != "joint_attention_retained"
        or screen.get("promotion_gate_passed") is not False
        or screen.get("benchmark_queries_opened") is not False
        or screen.get("benchmark_masks_opened") is not False
    ):
        raise ValueError("invalid attention-postcache readout binding screen")
    pairing_path = Path(
        str(screen.get("cache_pairing_report", {}).get("path", ""))
    ).resolve()
    if (
        not pairing_path.is_file()
        or screen.get("cache_pairing_report", {}).get("sha256")
        != _sha256_file(pairing_path)
    ):
        raise ValueError("attention-postcache cache-pairing binding differs")
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    rows = pairing.get("rows")
    if not isinstance(rows, list):
        raise ValueError("attention-postcache cache pairing lacks rows")
    train = [
        dict(row["c1024"])
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == "train"
    ]
    validation = [
        dict(row["c1024"])
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == "validation"
    ]
    train.sort(key=lambda record: record["path"])
    validation.sort(key=lambda record: record["path"])
    binding = authority._surface_binding(
        surface_root=path.parent,
        candidate="context_c1024_geometric",
        train=train,
        validation=validation,
    )
    if binding.get("binding_mode") != authority.ATTENTION_BINDING_MODE:
        raise ValueError("attention-postcache authority mode differs")
    variants = screen.get("variants")
    joint = variants.get("joint_attention_v1") if isinstance(variants, Mapping) else None
    seeds = joint.get("seeds") if isinstance(joint, Mapping) else None
    matches = [
        row
        for row in seeds or []
        if isinstance(row, Mapping) and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("attention-postcache screen lacks the selected joint seed")
    checkpoint_record = matches[0].get("checkpoint")
    if (
        not isinstance(checkpoint_record, Mapping)
        or Path(str(checkpoint_record.get("path", ""))).resolve() != checkpoint_path
        or checkpoint_record.get("sha256") != checkpoint_sha256
        or not report_path.is_file()
        or _sha256_file(report_path)
        != _sha256_file(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"))
    ):
        raise ValueError("attention-postcache selected readout binding differs")
    if validation != cache_meta["cache_bindings"]:
        raise ValueError("attention-postcache screen binds different validation caches")
    completion_path = path.parent / "screen.complete"
    if not completion_path.is_file():
        raise FileNotFoundError("attention-postcache screen completion is missing")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "completion": str(completion_path),
        "completion_sha256": _sha256_file(completion_path),
        "candidate": "context_c1024_geometric",
    }


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict:
    if str(args.device).lower() != "cpu":
        raise ValueError("this materializer is CPU-only; --device must be cpu")
    cache_paths = [Path(value) for value in args.validation_cache]
    data, cache_meta = _load_validation_caches(cache_paths)
    checkpoint_path = Path(args.readout_checkpoint).resolve()
    radio_path = Path(args.radio_checkpoint).resolve()
    if not checkpoint_path.is_file() or not radio_path.is_file():
        raise FileNotFoundError("readout/RADIO checkpoint is missing")
    checkpoint_sha = _sha256_file(checkpoint_path)
    radio_sha = _sha256_file(radio_path)
    if radio_sha != cache_meta["radio_checkpoint_sha256"]:
        raise ValueError("RADIO checkpoint does not match validation cache provenance")

    model, checkpoint = SurfaceRegionSummaryReadoutV2.from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    provenance = checkpoint.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("readout checkpoint lacks provenance")
    if provenance.get("uses_benchmark_scenes") is not False:
        raise ValueError("readout checkpoint used benchmark scenes")
    if provenance.get("uses_benchmark_test_vocabulary") is not False:
        raise ValueError("readout checkpoint used benchmark test vocabulary")
    if (
        provenance.get("scene_disjoint") is not True
        or provenance.get("custom_text_projection") is not False
    ):
        raise ValueError("readout checkpoint is not a frozen scene-disjoint readout")
    checkpoint_validation = provenance.get("validation")
    if not isinstance(checkpoint_validation, Mapping):
        raise ValueError("readout checkpoint lacks validation-cache provenance")
    distillation = provenance.get("text_response_distillation")
    is_distilled = isinstance(distillation, Mapping)
    is_v3_distilled = is_distilled and "response_lambdas" in distillation
    legacy_binding_raw = getattr(args, "readout_binding_manifest", None)
    expected_validation = dict(cache_meta["checkpoint_validation"])
    if not is_distilled:
        expected_validation.pop("cache_bindings")
    if checkpoint_validation != expected_validation:
        raise ValueError(
            "provided validation caches differ from checkpoint provenance "
            "(path/SHA/scene/split/teacher/exclusion/RADIO contract)"
        )
    checkpoint_train = provenance.get("train")
    if not isinstance(checkpoint_train, Mapping):
        raise ValueError("readout checkpoint lacks training-cache provenance")
    if set(checkpoint_train.get("scenes", [])) & set(cache_meta["scenes"]):
        raise ValueError("readout checkpoint train/validation scenes overlap")
    if provenance.get("region_contract_sha256") != cache_meta["region_contract_sha256"]:
        raise ValueError("readout/cache SurfaceRegion contracts differ")
    if provenance.get("region_contract") != cache_meta["region_contract"]:
        raise ValueError("readout/cache SurfaceRegion contract payloads differ")
    training_config = checkpoint.get("training_config", {})
    seed = training_config.get("seed")
    if seed not in {0, 1, 2}:
        raise ValueError("readout checkpoint lacks one of the frozen seeds 0/1/2")
    if not is_v3_distilled and provenance.get("random_seed_contract") != {
        "seed": seed,
        "model_initialization": True,
        "data_order": True,
        "canonical_noise": True,
    }:
        raise ValueError("readout checkpoint seed provenance differs")
    architecture = checkpoint.get("architecture", {})
    if (
        architecture.get("name") != "surface_region_summary_readout_v2"
        or architecture.get("contract_sha256")
        != cache_meta["region_contract_sha256"]
    ):
        raise ValueError("only SurfaceRegionSummaryReadoutV2 checkpoints are supported")
    report_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if is_distilled:
        if legacy_binding_raw:
            raise ValueError("distilled checkpoints cannot use the legacy bundle path")
        checkpoint_contract = None
        if is_v3_distilled:
            legacy_formal_v3 = _registered_legacy_formal_v3_binding(
                provenance.get("distill_run_manifest")
            )
            checkpoint_contract = _validate_v3_checkpoint_contract(
                checkpoint,
                provenance,
                distillation,
                seed=int(seed),
                cache_meta=cache_meta,
                legacy_protocol=legacy_formal_v3,
            )
        run_manifest = _validate_distill_run_manifest(
            provenance.get("distill_run_manifest"),
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            seed=int(seed),
            cache_meta=cache_meta,
            radio_path=radio_path,
            radio_sha256=radio_sha,
            checkpoint_contract=checkpoint_contract,
        )
        report_binding = _validate_checkpoint_report(
            report_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            checkpoint=checkpoint,
            cache_meta=cache_meta,
            run_manifest=run_manifest,
        )
        authority = {
            "type": "embedded_distill_run_manifest",
            "path": run_manifest["path"],
            "sha256": run_manifest["sha256"],
            "candidate": run_manifest["candidate"],
        }
    else:
        if not legacy_binding_raw:
            raise ValueError(
                "legacy readout requires --readout-binding-manifest with the "
                "query-free promotion bundle"
            )
        binding_path = Path(legacy_binding_raw)
        binding_payload = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            isinstance(binding_payload, Mapping)
            and binding_payload.get("artifact_type")
            == "surface_c1024_attention_pooling_postcache_continuation"
        ):
            legacy_bundle = _validate_attention_postcache_binding(
                binding_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                report_path=report_path,
                seed=int(seed),
                cache_meta=cache_meta,
            )
            authority_type = "attention_postcache_screen"
        else:
            legacy_bundle = _validate_legacy_promotion_bundle(
                binding_path,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                report_path=report_path,
                seed=int(seed),
                cache_meta=cache_meta,
            )
            authority_type = "query_free_promotion_bundle"
        report_binding = _validate_legacy_report(
            report_path,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            checkpoint=checkpoint,
            cache_meta=cache_meta,
        )
        authority = {
            "type": authority_type,
            **legacy_bundle,
        }

    head = SigLIP2SummaryHead.from_radio_checkpoint(str(radio_path)).cpu().eval()
    model = model.cpu().eval()
    model.requires_grad_(False)
    head.requires_grad_(False)
    students = []
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(data["radio_features"]), batch_size):
        stop = min(start + batch_size, len(data["radio_features"]))
        predicted_token = model(
            data["radio_features"][start:stop],
            data["geometry"][start:stop],
            anchor_index=data["anchor_index"][start:stop],
            token_mask=data["token_mask"][start:stop],
            reliability=data["reliability"][start:stop],
        )
        descriptor = head(predicted_token[:, None])[:, 0]
        students.append(F.normalize(descriptor.float(), dim=-1, eps=1e-8).cpu())
    student = torch.cat(students, dim=0).contiguous()
    teacher = _teacher_descriptor(
        data["official_crop_summaries"],
        data["teacher_mask"],
    ).contiguous()
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher descriptor shape mismatch: {student.shape} vs {teacher.shape}"
        )

    descriptor_rows_sha = row_identity_sha256(
        cache_meta["scene_ids"], cache_meta["region_ids"]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "method_id": str(args.method_id),
        "seed": int(seed),
        "split_role": "validation",
        "student_descriptors": student,
        "teacher_descriptors": teacher,
        "scene_ids": cache_meta["scene_ids"],
        "region_ids": cache_meta["region_ids"],
        "student_descriptors_sha256": tensor_sha256(student),
        "teacher_descriptors_sha256": tensor_sha256(teacher),
        "descriptor_rows_sha256": descriptor_rows_sha,
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
            "readout_checkpoint": str(checkpoint_path),
            "readout_checkpoint_sha256": checkpoint_sha,
            "readout_report": report_binding["path"],
            "readout_report_sha256": report_binding["sha256"],
            "readout_binding_authority": authority,
            "radio_checkpoint": str(radio_path),
            "radio_checkpoint_sha256": radio_sha,
            "region_contract_sha256": cache_meta["region_contract_sha256"],
            "validation_split_sha256": cache_meta["split_hashes"][0],
            "validation_scenes": cache_meta["scenes"],
            "teacher_region": cache_meta["teacher_region"],
            "validation_caches": cache_meta["caches"],
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"descriptor artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "sha256": _sha256_file(output),
        "method_id": str(args.method_id),
        "seed": int(seed),
        "regions": int(student.shape[0]),
        "descriptor_dimension": int(student.shape[1]),
        "scenes": len(set(cache_meta["scene_ids"])),
        "descriptor_rows_sha256": descriptor_rows_sha,
        "student_descriptors_sha256": payload["student_descriptors_sha256"],
        "teacher_descriptors_sha256": payload["teacher_descriptors_sha256"],
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-cache", action="append", required=True)
    parser.add_argument("--readout-checkpoint", required=True)
    parser.add_argument(
        "--readout-binding-manifest",
        help=(
            "Required only for a legacy query-free baseline checkpoint; must "
            "be its frozen query_free_promotion_bundle.json. Distilled "
            "checkpoints must use their embedded run-manifest binding."
        ),
    )
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not str(args.method_id).strip():
        raise ValueError("method_id cannot be empty")
    print(json.dumps(materialize(args), indent=2))


if __name__ == "__main__":
    main()
