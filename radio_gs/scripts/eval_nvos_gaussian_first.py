#!/usr/bin/env python3
"""Evaluate a protocol-locked Gaussian-first NVOS readout.

Prediction generation opens only the declared reference scribbles.  The target
ground-truth mask is opened only after the continuous score has been written,
so it cannot affect prototypes, thresholds, or the 3-D support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import (
    load_ground_truth_mask,
    resize_mask_nearest,
)
from radio_gs.interfaces.capability_cache import (
    CanonicalCapabilityBank,
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)
from radio_gs.interfaces.query_diffusion_cache import (
    load_query_diffusion_knn_cache,
    load_query_diffusion_relation_cache,
)
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.evidence_scorer import (
    EvidenceScoringConfig,
    RegisteredForwardBetaDiagnostics,
    registered_forward_beta_balanced_residual_observation,
    registered_forward_beta_observation,
    registered_observation_anchor_only_confidence,
    registered_observation_anchor_mask,
    registered_observation_effective_confidence,
    registered_raster_adjoint_observation,
    registered_seed_observation,
)
from radio_gs.querying.query_compilers import compile_registered_primitive_seeds
from radio_gs.querying.query_specific_propagation_cv import (
    ACTION_SOURCE_UNARY,
    ACTION_SURFACE_SAFE_PROPAGATED,
    SourceObservationOOFFold,
    audit_signed_cv_population,
    evaluate_source_observation_footprint_oof_artifacts,
    evaluate_source_observation_oof_artifacts,
    prepare_source_observation_footprint_oof_fold,
    prepare_source_observation_oof_fold,
)
from radio_gs.querying.source_footprint_fold_authority import (
    FIELD_BASE_ACTION,
    SourceFoldBaseDecision,
    SourceFootprintFoldAuthority,
    load_source_footprint_fold_authority,
)
from radio_gs.querying.source_observation_authority import (
    SourceObservationEvidenceAuthority,
    seal_or_load_source_observation_evidence_authority,
)
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_conditioned_diffusion import (
    QueryConditionedDiffusionConfig,
    cap_positive_reference_evidence,
    compute_query_conditioned_support,
    knn_feature_distances,
    normalize_node_features,
    rbf_similarity_from_distances,
    run_query_conditioned_diffusion,
    solve_continuous_query_support,
    weighted_logistic_query_compatibility,
)
from radio_gs.rendering.camera_clearance import (
    CAMERA_PLANE_CLEARANCE_CONTRACT,
    camera_plane_clearance_confidence,
)
from radio_gs.querying.reliability_fusion import (
    DUAL_PROTOTYPE_SEED_PROVENANCE,
    DUAL_SOLVER_SEED_PROVENANCE,
)
from radio_gs.querying.nvos_source_completion_calibration import (
    METHOD as SOURCE_COMPLETION_LOO_METHOD,
    load_source_completion_loo_gate,
    source_completion_loo_method_contract,
)
from radio_gs.querying.nvos_local_positive_completion import (
    local_majority_positive_evidence,
    method_contract as local_positive_completion_method_contract,
)
from radio_gs.querying.sam3_reference_completion import (
    probability_preserving_entropy_observation,
)
from radio_gs.querying.query_spec import (
    PrimitiveUnaryEvidence,
    SelectionMode,
)
from radio_gs.querying.support_solver import SupportSolverConfig
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.rendering.contribution_compositor import (
    rasterize_single_view_contributions,
)
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    raster_adjoint_registered_view_features,
    rasterize_registered_view_features,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_lerf_grounding import render_1280d
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views
from radio_gs.scripts.nvos_registered_region_v3_authority import (
    write_cuda_child_attestation,
)


_EXACT_RASTER_OBSERVATION_FUSIONS = frozenset(
    {
        "direct_raster_adjoint",
        "raster_adjoint_bernoulli_poe",
        "dual_registration_bernoulli_poe",
    }
)
_BERNOULLI_POE_FUSIONS = frozenset(
    {"raster_adjoint_bernoulli_poe", "dual_registration_bernoulli_poe"}
)
_FROZEN_LEGACY_PROTOTYPE_ALPHA_THRESHOLD = 0.02
_EXACT_WINNER_TAKE_ALL_PROTOTYPE_SEED_PROVENANCE = (
    "native_exact_adjoint_conditional_fraction_winner_take_all_prototype_v1"
)
_EXACT_JOINT_SIGNED_SOLVER_SEED_PROVENANCE = (
    "native_exact_adjoint_joint_signed_solver_anchor_v1"
)
_CANONICAL_FIELD_CAPABILITY_SOURCE = (
    "canonical_radio_field_official_frozen_capability_views"
)
_EXACT_MPR_CAPABILITY_SOURCE = (
    "exact_radio_mpr_official_frozen_capability_views"
)
_EXACT_CAPABILITY_MPR_SOURCE = (
    "exact_capability_mpr_official_frozen_capability_views"
)
_PROBABILITY_PRESERVING_SOURCE_UNARY = (
    "sam3_entropy_probability_preserving_mixture_v1"
)
_SOURCE_COMPLETION_LOO_CALIBRATION = SOURCE_COMPLETION_LOO_METHOD
_SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION = (
    "all_trial_loo_hierarchical_local_positive_v2"
)


def _apply_hierarchical_source_completion_trust(
    probability: torch.Tensor,
    reliability: torch.Tensor,
    *,
    accept_full_completion: bool,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Apply the fixed v1-global/v2-local source-only trust hierarchy."""

    if bool(accept_full_completion):
        return probability, reliability, {
            "branch": "v1_full_probability_preserving_completion",
            "v1_global_gate_accepted": True,
            "local_positive_contract": None,
        }
    positive = torch.from_numpy(np.ascontiguousarray(raw_positive)).bool()
    negative = torch.from_numpy(np.ascontiguousarray(raw_negative)).bool()
    local_probability, local_reliability = local_majority_positive_evidence(
        probability,
        positive_scribble=positive,
        negative_scribble=negative,
    )
    return local_probability, local_reliability, {
        "branch": "v2_local_majority_positive_completion",
        "v1_global_gate_accepted": False,
        "local_positive_contract": local_positive_completion_method_contract(),
    }


def _probability_preserving_registration_maps(
    prompt_maps: torch.Tensor,
    probability: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Pack raw signed prompts and calibrated completion into one adjoint."""

    raw = torch.as_tensor(prompt_maps)
    foreground = torch.as_tensor(
        probability, device=raw.device, dtype=torch.float32
    )
    confidence = torch.as_tensor(
        reliability, device=raw.device, dtype=torch.float32
    )
    if raw.ndim != 4 or raw.shape[:2] != (1, 2):
        raise ValueError("raw registered prompt maps must have shape [1,2,H,W]")
    if foreground.shape != raw.shape[2:] or confidence.shape != raw.shape[2:]:
        raise ValueError("source completion probability/reliability shape differs")
    if (
        not bool(torch.isfinite(foreground).all())
        or not bool(torch.isfinite(confidence).all())
        or bool(((foreground < 0) | (foreground > 1)).any())
        or bool(((confidence < 0) | (confidence > 1)).any())
    ):
        raise ValueError("source completion probability/reliability must be in [0,1]")
    completion = torch.stack(
        [confidence * foreground, confidence * (1.0 - foreground)], dim=0
    )[None]
    return torch.cat([raw.to(dtype=torch.float32), completion], dim=1)


def _source_completion_unary_contract(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    mode = str(getattr(args, "source_completion_unary", "none"))
    if mode == "none":
        return None
    if mode != _PROBABILITY_PRESERVING_SOURCE_UNARY:
        raise ValueError(f"unknown source completion unary mode {mode!r}")
    calibration = str(
        getattr(args, "source_completion_calibration", "none")
    )
    if calibration not in {
        "none",
        _SOURCE_COMPLETION_LOO_CALIBRATION,
        _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION,
    }:
        raise ValueError(f"unknown source completion calibration {calibration!r}")
    return {
        "mode": mode,
        "source_probability": (
            "q=mean_of_frozen_official_SAM3_binary_trials_with_signed_"
            "scribble_overwrite"
        ),
        "source_reliability": "c(q)=1-H2(q)/ln2_with_signed_scribble_c=1",
        "exact_adjoint_channels": ["c*q", "c*(1-q)"],
        "primitive_probability": "W^T(c*q)/W^T(c)",
        "primitive_confidence": "W^T(c)/W^T(1)",
        "fusion": "(1-c_primitive)*p_field+c_primitive*p_source",
        "uncertain_probability_policy": "preserve_q_separately_from_confidence",
        "graph": "disabled_zero_edge_unary_prior_only",
        "source_only_calibration": (
            source_completion_loo_method_contract()
            if calibration
            in {
                _SOURCE_COMPLETION_LOO_CALIBRATION,
                _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION,
            }
            else None
        ),
        "calibration_reject_fallback": (
            "frozen_compact_hard_seed_anchor_only_probability_base"
            if calibration == _SOURCE_COMPLETION_LOO_CALIBRATION
            else (
                "v2_local_majority_positive_completion"
                if calibration
                == _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION
                else None
            )
        ),
        "hierarchical_local_positive_contract": (
            local_positive_completion_method_contract()
            if calibration
            == _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION
            else None
        ),
        "target_rgb_or_mask_used": False,
        "scene_specific_numeric_constants": False,
    }


def _validate_source_completion_unary_args(args: argparse.Namespace) -> None:
    """Fail closed before model loading for source-completed unary fusion."""

    contract = _source_completion_unary_contract(args)
    asset_names = (
        "source_completion",
        "source_completion_sha256",
        "source_completion_receipt",
        "source_completion_receipt_sha256",
    )
    assets = {
        name: str(getattr(args, name, "")).strip() for name in asset_names
    }
    calibration = str(
        getattr(args, "source_completion_calibration", "none")
    )
    gate_assets = {
        "source_completion_calibration_gate": str(
            getattr(args, "source_completion_calibration_gate", "")
        ).strip(),
        "source_completion_calibration_gate_sha256": str(
            getattr(args, "source_completion_calibration_gate_sha256", "")
        ).strip(),
    }
    if contract is None:
        dangling = [f"--{name.replace('_', '-')}" for name, value in assets.items() if value]
        dangling.extend(
            f"--{name.replace('_', '-')}"
            for name, value in gate_assets.items()
            if value
        )
        if calibration != "none":
            dangling.append("--source-completion-calibration")
        if dangling:
            raise ValueError(
                "source completion assets require --source-completion-unary "
                f"{_PROBABILITY_PRESERVING_SOURCE_UNARY}: " + ", ".join(dangling)
            )
        return
    calibrated = calibration in {
        _SOURCE_COMPLETION_LOO_CALIBRATION,
        _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION,
    }
    if calibration == "none" and any(gate_assets.values()):
        raise ValueError(
            "source completion calibration gate assets require "
            "a source-only --source-completion-calibration"
        )
    requirements = {
        "--support-mode canonical_support": str(args.support_mode)
        == "canonical_support",
        "--disable-registered-graph": bool(args.disable_registered_graph),
        "--registered-readout-stage unary_prior": str(args.registered_readout_stage)
        == "unary_prior",
        "--query-conditioned-diffusion-kernel none": str(
            args.query_conditioned_diffusion_kernel
        )
        == "none",
        "--registered-forward-unary none": str(args.registered_forward_unary)
        == "none",
        "--registered-observation-fusion probability_mixture": str(
            args.registered_observation_fusion
        )
        == "probability_mixture",
        "--registered-seed-unary-weight 0": float(
            args.registered_seed_unary_weight
        )
        == 0.0,
        "--prompt-registration-mode raster_adjoint": str(
            args.prompt_registration_mode
        )
        == "raster_adjoint",
        "--prompt-registration-scale 1": float(args.prompt_registration_scale)
        == 1.0,
        "--alpha-threshold 0": float(args.alpha_threshold) == 0.0,
        "--feature-contribution-gamma 1": float(args.feature_contribution_gamma)
        == 1.0,
        "no --registered-reference-threshold-calibration": not bool(
            args.registered_reference_threshold_calibration
        ),
        **{
            f"--{name.replace('_', '-')}": bool(value)
            for name, value in assets.items()
        },
        **(
            {
                f"--{name.replace('_', '-')}": bool(value)
                for name, value in gate_assets.items()
            }
            if calibrated
            else {}
        ),
    }
    for name in ("source_completion_sha256", "source_completion_receipt_sha256"):
        requirements[f"--{name.replace('_', '-')} valid SHA256"] = (
            len(assets[name]) == 64
            and all(character in "0123456789abcdef" for character in assets[name].lower())
        )
    if calibrated:
        gate_sha256 = gate_assets["source_completion_calibration_gate_sha256"]
        requirements["--source-completion-calibration-gate-sha256 valid SHA256"] = (
            len(gate_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in gate_sha256.lower()
            )
        )
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    if failed:
        raise ValueError(
            f"{_PROBABILITY_PRESERVING_SOURCE_UNARY} requires "
            + ", ".join(failed)
        )


def _load_probability_preserving_source_completion(
    args: argparse.Namespace,
    *,
    scene_id: str,
    frame_id: str,
    positive_path: str | Path,
    negative_path: str | Path,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Load and bind a frozen source-only completion without opening targets."""

    completion_path = Path(args.source_completion).expanduser().resolve()
    receipt_path = Path(args.source_completion_receipt).expanduser().resolve()
    completion_sha256 = _file_sha256(completion_path)
    receipt_sha256 = _file_sha256(receipt_path)
    if completion_sha256 != str(args.source_completion_sha256):
        raise ValueError("source completion SHA256 differs")
    if receipt_sha256 != str(args.source_completion_receipt_sha256):
        raise ValueError("source completion receipt SHA256 differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version")
        != "nvos_sam3_reference_completion_receipt_v1"
        or receipt.get("scene_id") != str(scene_id)
        or receipt.get("frame_id") != str(frame_id)
        or Path(str(receipt.get("artifact_path"))).expanduser().resolve()
        != completion_path
        or receipt.get("artifact_sha256") != completion_sha256
        or receipt.get("target_rgb_opened") is not False
        or receipt.get("target_mask_opened") is not False
        or receipt.get("target_metric_opened") is not False
    ):
        raise ValueError("source completion receipt authority differs")
    payload = torch.load(completion_path, map_location="cpu", weights_only=True)
    authority = payload.get("authority", {}) if isinstance(payload, Mapping) else {}
    tensors = payload.get("tensors") if isinstance(payload, Mapping) else None
    expected_tensor_keys = {
        "trial_masks",
        "aggregate_probability",
        "completed_positive",
        "raw_positive",
        "raw_negative",
        "point_coordinates_xy",
        "quality",
    }
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type")
        != "radio_gs.nvos_sam3_reference_completion"
        or payload.get("schema_version") != 1
        or authority.get("scene_id") != str(scene_id)
        or authority.get("frame_id") != str(frame_id)
        or authority.get("target_rgb_opened") is not False
        or authority.get("target_mask_opened") is not False
        or not isinstance(tensors, Mapping)
        or set(tensors) != expected_tensor_keys
    ):
        raise ValueError("source completion artifact authority differs")
    digests = {name: tensor_sha256(value) for name, value in sorted(tensors.items())}
    tensor_bundle_sha256 = hashlib.sha256(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        payload.get("tensor_sha256") != digests
        or receipt.get("tensor_sha256") != digests
        or payload.get("tensor_bundle_sha256") != tensor_bundle_sha256
        or receipt.get("tensor_bundle_sha256") != tensor_bundle_sha256
    ):
        raise ValueError("source completion tensor hashes differ")
    positive_cpu = torch.from_numpy(np.ascontiguousarray(raw_positive)).bool()
    negative_cpu = torch.from_numpy(np.ascontiguousarray(raw_negative)).bool()
    aggregate = torch.as_tensor(tensors["aggregate_probability"])
    if (
        aggregate.device.type != "cpu"
        or aggregate.dtype != torch.float32
        or tuple(aggregate.shape) != tuple(positive_cpu.shape)
        or not bool(torch.isfinite(aggregate).all())
        or bool(((aggregate < 0) | (aggregate > 1)).any())
        or not torch.equal(torch.as_tensor(tensors["raw_positive"]), positive_cpu)
        or not torch.equal(torch.as_tensor(tensors["raw_negative"]), negative_cpu)
    ):
        raise ValueError("source completion probability or scribble tensors differ")
    if (
        authority.get("positive_scribble_sha256")
        != _file_sha256(Path(positive_path).expanduser().resolve())
        or authority.get("negative_scribble_sha256")
        != _file_sha256(Path(negative_path).expanduser().resolve())
    ):
        raise ValueError("source completion signed scribble authority differs")
    probability, reliability = probability_preserving_entropy_observation(
        aggregate.numpy(), raw_positive, raw_negative
    )
    return (
        torch.from_numpy(probability).contiguous(),
        torch.from_numpy(reliability).contiguous(),
        {
            "contract": _source_completion_unary_contract(args),
            "completion_path": str(completion_path),
            "completion_sha256": completion_sha256,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha256,
            "tensor_bundle_sha256": tensor_bundle_sha256,
            "target_rgb_opened": False,
            "target_mask_opened": False,
        },
    )


def _disabled_registered_graph(num_nodes: int) -> PrimitiveSupportGraph:
    """Return a zero-edge graph for a strictly unary-only compiler audit."""

    if int(num_nodes) <= 0:
        raise ValueError("disabled registered graph requires positive nodes")
    return PrimitiveSupportGraph(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,), dtype=torch.float32),
        raw_affinity=torch.empty((0,), dtype=torch.float32),
        local_sigma=torch.ones((int(num_nodes),), dtype=torch.float32),
        num_nodes=int(num_nodes),
        edge_channels={},
    )


def _write_primitive_unary_artifact(
    path: str | Path,
    *,
    scene_id: str,
    protocol_hash: str,
    capability_cache: str | Path,
    capability_source_contract: str,
    valid: torch.Tensor,
    primitive_unary_probability: torch.Tensor,
    compiler_contract: Mapping[str, object],
) -> Path:
    """Persist the pre-render primitive unary before target GT is opened."""

    valid_cpu = torch.as_tensor(valid).detach().bool().cpu().reshape(-1)
    unary_cpu = (
        torch.as_tensor(primitive_unary_probability)
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    if unary_cpu.shape != valid_cpu.shape:
        raise ValueError("primitive unary must align with the global row domain")
    if not bool(torch.isfinite(unary_cpu).all()) or bool(
        ((unary_cpu < 0) | (unary_cpu > 1)).any()
    ):
        raise ValueError("primitive unary probabilities must be finite and in [0,1]")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "nvos_frozen_k16_primitive_unary_probability_v1",
            "scene_id": str(scene_id),
            "protocol_hash": str(protocol_hash),
            "capability_cache": str(Path(capability_cache).expanduser().resolve()),
            "capability_cache_sha256": _file_sha256(
                Path(capability_cache).expanduser().resolve()
            ),
            "capability_source_contract": str(capability_source_contract),
            "valid": valid_cpu,
            "valid_rows": torch.where(valid_cpu)[0],
            "primitive_unary_probability": unary_cpu,
            "compiler_contract": dict(compiler_contract),
            "written_before_target_ground_truth_open": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
        },
        output,
    )
    return output


def _write_source_observation_oof_artifact(
    path: str | Path,
    *,
    scene_id: str,
    protocol_hash: str,
    heldout_fold: int,
    capability_cache: str | Path,
    support_graph: str | Path,
    authority: SourceObservationOOFFold,
    evidence_authority: SourceObservationEvidenceAuthority,
    valid: torch.Tensor,
    global_rows: torch.Tensor,
    unary_probability: torch.Tensor,
    propagated_probability: torch.Tensor,
    method_contract: Mapping[str, object],
) -> tuple[Path, str, Path]:
    """Seal one source-only OOF fold before any target asset is opened."""

    tensors = {
        "valid": torch.as_tensor(valid).detach().bool().cpu().reshape(-1),
        "global_rows": torch.as_tensor(global_rows).detach().long().cpu().reshape(-1),
        "fold_ids": authority.fold_ids.detach().long().cpu().reshape(-1),
        "observed": authority.observed.detach().bool().cpu().reshape(-1),
        "heldout": authority.heldout.detach().bool().cpu().reshape(-1),
        "signed_reference_evidence": authority.signed_reference_evidence.detach()
        .float()
        .cpu()
        .reshape(-1),
        "reference_weight": authority.reference_weight.detach()
        .float()
        .cpu()
        .reshape(-1),
        "unary_probability": torch.as_tensor(unary_probability)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "surface_safe_propagated_probability": torch.as_tensor(
            propagated_probability
        )
        .detach()
        .float()
        .cpu()
        .reshape(-1),
    }
    full_shape = tensors["valid"].shape
    full_names = set(tensors) - {"global_rows"}
    if any(tensors[name].shape != full_shape for name in full_names):
        raise ValueError("OOF artifact tensors must align with the global row domain")
    if not torch.equal(tensors["global_rows"], torch.where(tensors["valid"])[0]):
        raise ValueError("OOF artifact global rows differ from valid-row authority")
    for name in ("unary_probability", "surface_safe_propagated_probability"):
        value = tensors[name]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0) | (value > 1)).any()
        ):
            raise ValueError(f"OOF {name} must be finite and in [0,1]")
    heldout = tensors["heldout"]
    cleared = {
        "positive_weight_sum": float(authority.training_positive_weight[heldout].sum()),
        "negative_weight_sum": float(authority.training_negative_weight[heldout].sum()),
        "raw_positive_mass_sum": float(
            authority.training_raw_positive_mass[heldout].sum()
        ),
        "raw_negative_mass_sum": float(
            authority.training_raw_negative_mass[heldout].sum()
        ),
    }
    if any(value != 0.0 for value in cleared.values()):
        raise RuntimeError("OOF artifact detected held-out prompt-evidence leakage")

    capability_path = Path(capability_cache).expanduser().resolve()
    graph_path = Path(support_graph).expanduser().resolve()
    evidence_path = evidence_authority.path.expanduser().resolve()
    if (
        not capability_path.is_file()
        or not graph_path.is_file()
        or not evidence_path.is_file()
    ):
        raise FileNotFoundError(
            "OOF artifact requires immutable capability, graph, and source-evidence assets"
        )
    if _file_sha256(evidence_path) != evidence_authority.sha256:
        raise ValueError("OOF source-evidence authority changed before fold sealing")

    def tensor_sha256(value: torch.Tensor) -> str:
        array = value.contiguous().numpy()
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    authority_contract = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_oof_fold_v1",
        "scene_id": str(scene_id),
        "protocol_hash": str(protocol_hash),
        "heldout_fold": int(heldout_fold),
        "num_folds": 3,
        "fold_assignment": "splitmix64_global_primitive_row_v1",
        "capability_cache": str(capability_path),
        "capability_cache_sha256": _file_sha256(capability_path),
        "support_graph": str(graph_path),
        "support_graph_sha256": _file_sha256(graph_path),
        "source_evidence_authority": str(evidence_path),
        "source_evidence_authority_sha256": evidence_authority.sha256,
        "source_evidence_authority_content_sha256": (
            evidence_authority.content_sha256
        ),
        "source_evidence_replay_max_relative_error": dict(
            evidence_authority.replay_max_relative_error
        ),
        "method_contract": dict(method_contract),
        "method_contract_sha256": _json_sha256(dict(method_contract)),
        "tensor_sha256": tensor_hashes,
        "heldout_prompt_evidence_after_clear": cleared,
        "heldout_clear_boundary": (
            "before_direct_observation_prototypes_anchors_unary_and_graph_propagation"
        ),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    content_sha256 = _json_sha256(authority_contract)
    payload = {
        **authority_contract,
        "content_sha256": content_sha256,
        **tensors,
    }
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(existing, Mapping) or existing.get("content_sha256") != content_sha256:
            raise FileExistsError(f"refusing to overwrite different OOF artifact: {output}")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
    artifact_sha256 = _file_sha256(output)
    receipt = {
        **authority_contract,
        "content_sha256": content_sha256,
        "artifact_path": str(output),
        "artifact_sha256": artifact_sha256,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if receipt_path.exists() and receipt_path.read_bytes() != receipt_bytes:
        raise FileExistsError(f"refusing to overwrite different OOF receipt: {receipt_path}")
    if not receipt_path.exists():
        temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        temporary_receipt.write_bytes(receipt_bytes)
        temporary_receipt.replace(receipt_path)
    return output, artifact_sha256, receipt_path


def _write_source_observation_oof_gate_receipt(
    output_dir: str | Path,
) -> tuple[Path | None, dict[str, object]]:
    """Seal the gate once all three independently executed folds exist."""

    root = Path(output_dir).expanduser().resolve()
    fold_paths = {fold: root / f"fold_{fold}.pt" for fold in range(3)}
    missing = [fold for fold, path in fold_paths.items() if not path.is_file()]
    if missing:
        return None, {
            "status": "awaiting_source_observation_oof_folds",
            "missing_folds": missing,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        }
    payloads = {
        fold: torch.load(path, map_location="cpu", weights_only=False)
        for fold, path in fold_paths.items()
    }
    if any(not isinstance(payload, Mapping) for payload in payloads.values()):
        raise ValueError("source-observation OOF fold payload is not a mapping")
    result = evaluate_source_observation_oof_artifacts(payloads)
    fold_records = {
        str(fold): {
            "path": str(path),
            "sha256": _file_sha256(path),
            "receipt_path": str(path.with_suffix(path.suffix + ".receipt.json")),
            "receipt_sha256": _file_sha256(
                path.with_suffix(path.suffix + ".receipt.json")
            ),
        }
        for fold, path in fold_paths.items()
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_oof_gate_v1",
        "scene_id": result.scene_id,
        "protocol_hash": result.protocol_hash,
        "method_contract_sha256": result.method_contract_sha256,
        "capability_cache_sha256": result.capability_cache_sha256,
        "support_graph_sha256": result.support_graph_sha256,
        "source_evidence_authority_sha256": (
            result.source_evidence_authority_sha256
        ),
        "source_evidence_authority_content_sha256": (
            result.source_evidence_authority_content_sha256
        ),
        "fold_artifacts": fold_records,
        "fold_assignment": "splitmix64_global_primitive_row_v1",
        "num_folds": 3,
        "minimum_positive_rows_per_training_or_heldout_fold": 32,
        "minimum_negative_rows_per_training_or_heldout_fold": 32,
        "metric_round_decimals": 12,
        "probability_epsilon": 1e-7,
        "metrics": result.metrics,
        "fold_reports": result.fold_reports,
        "selected_action": result.selected_action,
        "observed_rows": int(result.observed.sum()),
        "selection_rule": (
            "minimize rounded balanced log-loss, maximize rounded weighted "
            "AUC, then choose unary"
        ),
        "full_fit_predictions_used_as_oof": False,
        "connected_selection": "off",
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    encoded = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output = root / "source_observation_oof_gate_receipt.json"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(
            f"refusing to overwrite different source-observation gate: {output}"
        )
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    return output, receipt


def _load_source_observation_oof_deployment_gate(
    args: argparse.Namespace,
    *,
    scene_id: str,
    protocol_hash: str,
) -> dict[str, object] | None:
    """Bind a source-only OOF decision to the deployed pre-metric readout.

    OOF generation deliberately exits before target rendering.  A separate
    deployment invocation therefore has to replay the immutable gate and
    prove that the selected action is exactly the readout stage being used.
    """

    raw_path = str(
        getattr(args, "source_observation_oof_gate_receipt", "")
    ).strip()
    raw_sha256 = str(
        getattr(args, "source_observation_oof_gate_receipt_sha256", "")
    ).strip()
    if bool(raw_path) != bool(raw_sha256):
        raise ValueError(
            "source-observation OOF deployment requires both gate receipt "
            "path and expected SHA-256"
        )
    if not raw_path:
        return None
    gate_path = Path(raw_path).expanduser().resolve()
    actual_sha256 = _file_sha256(gate_path)
    if actual_sha256 != raw_sha256:
        raise ValueError("source-observation OOF deployment gate SHA-256 differs")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("source-observation OOF deployment gate is unreadable") from error
    if not isinstance(gate, Mapping):
        raise ValueError("source-observation OOF deployment gate is not a mapping")
    supported_schemas = {
        "source_observation_surface_safe_oof_gate_v1",
        "source_observation_surface_safe_footprint_oof_gate_v1",
    }
    if gate.get("schema_version") != 1 or gate.get("artifact_type") not in supported_schemas:
        raise ValueError("source-observation OOF deployment gate schema differs")
    if str(gate.get("scene_id", "")) != str(scene_id):
        raise ValueError("source-observation OOF deployment gate scene differs")
    if str(gate.get("protocol_hash", "")) != str(protocol_hash):
        raise ValueError("source-observation OOF deployment gate protocol differs")
    for flag in ("target_rgb_opened", "target_mask_opened", "target_metric_computed"):
        if gate.get(flag) is not False:
            raise ValueError(
                "source-observation OOF deployment gate target-access flag differs"
            )
    if gate.get("full_fit_predictions_used_as_oof") is not False:
        raise ValueError("source-observation OOF gate used full-fit predictions as OOF")
    if str(gate.get("connected_selection", "")) != "off":
        raise ValueError("source-observation OOF deployment gate enables connected selection")

    capability_path = Path(args.canonical_capability_cache).expanduser().resolve()
    graph_path = Path(args.canonical_support_graph).expanduser().resolve()
    if str(gate.get("capability_cache_sha256", "")) != _file_sha256(capability_path):
        raise ValueError("source-observation OOF gate capability cache differs")
    if str(gate.get("support_graph_sha256", "")) != _file_sha256(graph_path):
        raise ValueError("source-observation OOF gate support graph differs")

    selected_action = str(gate.get("selected_action", ""))
    required_stage = {
        ACTION_SOURCE_UNARY: "unary_prior",
        ACTION_SURFACE_SAFE_PROPAGATED: "propagated",
    }.get(selected_action)
    if required_stage is None:
        raise ValueError(
            "source-observation OOF deployment gate selected an unsupported "
            f"full-fit action: {selected_action!r}"
        )
    if str(args.registered_readout_stage) != required_stage:
        raise ValueError(
            "source-observation OOF selected action requires "
            f"--registered-readout-stage {required_stage}"
        )
    if str(args.registered_selection_mode) != "all_components":
        raise ValueError(
            "source-observation OOF deployment requires all-components selection"
        )
    if str(args.query_conditioned_diffusion_kernel) != "none":
        raise ValueError(
            "source-observation OOF deployment forbids query-conditioned diffusion"
        )
    return {
        "path": str(gate_path),
        "sha256": actual_sha256,
        "artifact_type": str(gate["artifact_type"]),
        "method_contract_sha256": str(gate.get("method_contract_sha256", "")),
        "selected_action": selected_action,
        "required_readout_stage": required_stage,
        "selection_rule": str(gate.get("selection_rule", "")),
        "target_feedback": False,
    }


def _write_source_observation_footprint_oof_artifact(
    path: str | Path,
    *,
    scene_id: str,
    protocol_hash: str,
    heldout_fold: int,
    capability_cache: str | Path,
    support_graph: str | Path,
    authority: SourceObservationOOFFold,
    evidence_authority: SourceObservationEvidenceAuthority,
    footprint_path: str | Path,
    footprint_file_sha256: str,
    footprint_authority: SourceFootprintFoldAuthority,
    valid: torch.Tensor,
    global_rows: torch.Tensor,
    population_positive_weight: torch.Tensor,
    population_negative_weight: torch.Tensor,
    unary_probability: torch.Tensor,
    propagated_probability: torch.Tensor,
    method_contract: Mapping[str, object],
) -> tuple[Path, str, Path]:
    """Seal one whole-footprint OOF fold before any target asset is opened."""

    footprint_authority.validate(
        expected_authority_sha256=footprint_authority.authority_sha256
    )
    tensors = {
        "valid": torch.as_tensor(valid).detach().bool().cpu().reshape(-1),
        "global_rows": torch.as_tensor(global_rows).detach().long().cpu().reshape(-1),
        "fold_ids": authority.fold_ids.detach().long().cpu().reshape(-1),
        "observed": authority.observed.detach().bool().cpu().reshape(-1),
        "heldout": authority.heldout.detach().bool().cpu().reshape(-1),
        "signed_reference_evidence": authority.signed_reference_evidence.detach()
        .float()
        .cpu()
        .reshape(-1),
        "reference_weight": authority.reference_weight.detach()
        .float()
        .cpu()
        .reshape(-1),
        "population_positive_weight": torch.as_tensor(population_positive_weight)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "population_negative_weight": torch.as_tensor(population_negative_weight)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "unary_probability": torch.as_tensor(unary_probability)
        .detach()
        .float()
        .cpu()
        .reshape(-1),
        "surface_safe_propagated_probability": torch.as_tensor(
            propagated_probability
        )
        .detach()
        .float()
        .cpu()
        .reshape(-1),
    }
    full_shape = tensors["valid"].shape
    if any(
        tensors[name].shape != full_shape
        for name in set(tensors) - {"global_rows"}
    ):
        raise ValueError("source-footprint OOF tensors must align globally")
    if (
        not torch.equal(tensors["global_rows"], torch.where(tensors["valid"])[0])
        or not torch.equal(
            tensors["global_rows"], footprint_authority.primitive_rows
        )
    ):
        raise ValueError(
            "source-footprint rows, capability rows, and valid rows differ"
        )
    for name in (
        "population_positive_weight",
        "population_negative_weight",
    ):
        value = tensors[name]
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"source-footprint {name} is invalid")
    for name in ("unary_probability", "surface_safe_propagated_probability"):
        value = tensors[name]
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0) | (value > 1)).any()
        ):
            raise ValueError(f"source-footprint {name} must be in [0,1]")
    heldout = tensors["heldout"]
    cleared = {
        "positive_weight_sum": float(authority.training_positive_weight[heldout].sum()),
        "negative_weight_sum": float(authority.training_negative_weight[heldout].sum()),
        "raw_positive_mass_sum": float(
            authority.training_raw_positive_mass[heldout].sum()
        ),
        "raw_negative_mass_sum": float(
            authority.training_raw_negative_mass[heldout].sum()
        ),
    }
    if any(value != 0.0 for value in cleared.values()):
        raise RuntimeError("source-footprint held-out evidence leaked")

    capability_path = Path(capability_cache).expanduser().resolve()
    graph_path = Path(support_graph).expanduser().resolve()
    evidence_path = evidence_authority.path.expanduser().resolve()
    footprint_asset = Path(footprint_path).expanduser().resolve()
    if any(
        not asset.is_file()
        for asset in (capability_path, graph_path, evidence_path, footprint_asset)
    ):
        raise FileNotFoundError(
            "source-footprint OOF requires immutable capability, graph, evidence, and footprint assets"
        )
    if _file_sha256(evidence_path) != evidence_authority.sha256:
        raise ValueError("source-footprint source-evidence authority changed")
    if _file_sha256(footprint_asset) != str(footprint_file_sha256):
        raise ValueError("source-footprint authority file changed before sealing")

    def tensor_digest(value: torch.Tensor) -> str:
        return hashlib.sha256(
            value.contiguous().numpy().tobytes(order="C")
        ).hexdigest()

    tensor_hashes = {name: tensor_digest(value) for name, value in tensors.items()}
    authority_contract = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_footprint_oof_fold_v1",
        "scene_id": str(scene_id),
        "protocol_hash": str(protocol_hash),
        "heldout_fold": int(heldout_fold),
        "num_folds": 3,
        "fold_assignment": "splitmix64_source_footprint_group_v1",
        "capability_cache": str(capability_path),
        "capability_cache_sha256": _file_sha256(capability_path),
        "support_graph": str(graph_path),
        "support_graph_sha256": _file_sha256(graph_path),
        "source_evidence_authority": str(evidence_path),
        "source_evidence_authority_sha256": evidence_authority.sha256,
        "source_evidence_authority_content_sha256": evidence_authority.content_sha256,
        "source_evidence_replay_max_relative_error": dict(
            evidence_authority.replay_max_relative_error
        ),
        "source_footprint_fold_authority": str(footprint_asset),
        "source_footprint_fold_authority_file_sha256": str(
            footprint_file_sha256
        ),
        "source_footprint_fold_authority_sha256": (
            footprint_authority.authority_sha256
        ),
        "source_footprint_fold_authority_tensor_bundle_sha256": (
            footprint_authority.tensor_bundle_sha256
        ),
        "method_contract": dict(method_contract),
        "method_contract_sha256": _json_sha256(dict(method_contract)),
        "tensor_sha256": tensor_hashes,
        "heldout_prompt_evidence_after_clear": cleared,
        "heldout_clear_boundary": (
            "after_query_free_footprint_groups_before_prototypes_anchors_unary_and_graph"
        ),
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    content_sha256 = _json_sha256(authority_contract)
    payload = {**authority_contract, "content_sha256": content_sha256, **tensors}
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if not isinstance(existing, Mapping) or existing.get("content_sha256") != content_sha256:
            raise FileExistsError(
                f"refusing to overwrite different source-footprint OOF artifact: {output}"
            )
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output)
    artifact_sha256 = _file_sha256(output)
    receipt = {
        **authority_contract,
        "content_sha256": content_sha256,
        "artifact_path": str(output),
        "artifact_sha256": artifact_sha256,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if receipt_path.exists() and receipt_path.read_bytes() != receipt_bytes:
        raise FileExistsError(
            f"refusing to overwrite different source-footprint receipt: {receipt_path}"
        )
    if not receipt_path.exists():
        temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        temporary_receipt.write_bytes(receipt_bytes)
        temporary_receipt.replace(receipt_path)
    return output, artifact_sha256, receipt_path


def _write_source_observation_footprint_oof_gate_receipt(
    output_dir: str | Path,
) -> tuple[Path | None, dict[str, object]]:
    """Seal an eligible structured gate from exactly three footprint folds."""

    root = Path(output_dir).expanduser().resolve()
    fold_paths = {fold: root / f"fold_{fold}.pt" for fold in range(3)}
    missing = [fold for fold, path in fold_paths.items() if not path.is_file()]
    if missing:
        return None, {
            "status": "awaiting_source_observation_footprint_oof_folds",
            "missing_folds": missing,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
        }
    payloads = {
        fold: torch.load(path, map_location="cpu", weights_only=False)
        for fold, path in fold_paths.items()
    }
    if any(not isinstance(payload, Mapping) for payload in payloads.values()):
        raise ValueError("source-footprint OOF payload is not a mapping")
    reference = payloads[0]
    footprint_path = Path(
        str(reference.get("source_footprint_fold_authority", ""))
    ).expanduser().resolve()
    footprint_file_sha256 = str(
        reference.get("source_footprint_fold_authority_file_sha256", "")
    )
    footprint_authority_sha256 = str(
        reference.get("source_footprint_fold_authority_sha256", "")
    )
    footprint_authority = load_source_footprint_fold_authority(
        footprint_path,
        expected_file_sha256=footprint_file_sha256,
        expected_authority_sha256=footprint_authority_sha256,
    )
    result = evaluate_source_observation_footprint_oof_artifacts(
        payloads,
        footprint_authority=footprint_authority,
        footprint_authority_path=str(footprint_path),
        footprint_authority_file_sha256=footprint_file_sha256,
    )
    fold_records = {
        str(fold): {
            "path": str(path),
            "sha256": _file_sha256(path),
            "receipt_path": str(path.with_suffix(path.suffix + ".receipt.json")),
            "receipt_sha256": _file_sha256(
                path.with_suffix(path.suffix + ".receipt.json")
            ),
        }
        for fold, path in fold_paths.items()
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_footprint_oof_gate_v1",
        "gate_mode": "eligible_source_oof",
        "scene_id": result.scene_id,
        "protocol_hash": result.protocol_hash,
        "method_contract_sha256": result.method_contract_sha256,
        "capability_cache_sha256": result.capability_cache_sha256,
        "support_graph_sha256": result.support_graph_sha256,
        "source_evidence_authority_sha256": result.source_evidence_authority_sha256,
        "source_evidence_authority_content_sha256": (
            result.source_evidence_authority_content_sha256
        ),
        "source_footprint_fold_authority": str(footprint_path),
        "source_footprint_fold_authority_file_sha256": footprint_file_sha256,
        "source_footprint_fold_authority_sha256": (
            footprint_authority.authority_sha256
        ),
        "source_footprint_fold_authority_tensor_bundle_sha256": (
            footprint_authority.tensor_bundle_sha256
        ),
        "fold_artifacts": fold_records,
        "fold_assignment": "splitmix64_source_footprint_group_v1",
        "num_folds": 3,
        "minimum_positive_rows_per_training_or_heldout_fold": 32,
        "minimum_negative_rows_per_training_or_heldout_fold": 32,
        "metric_round_decimals": 12,
        "probability_epsilon": 1e-7,
        "metrics": result.metrics,
        "fold_reports": result.fold_reports,
        "selected_action": result.selected_action,
        "observed_rows": int(result.observed.sum()),
        "selection_rule": (
            "minimize rounded balanced log-loss, maximize rounded weighted AUC, then choose unary"
        ),
        "full_fit_predictions_used_as_oof": False,
        "connected_selection": "off",
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = root / "source_observation_oof_gate_receipt.json"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(
            f"refusing to overwrite different source-footprint gate: {output}"
        )
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    return output, receipt


def _write_source_observation_footprint_field_base_receipt(
    output_dir: str | Path,
    *,
    scene_id: str,
    protocol_hash: str,
    capability_cache: str | Path,
    support_graph: str | Path,
    evidence_authority: SourceObservationEvidenceAuthority,
    footprint_path: str | Path,
    footprint_file_sha256: str,
    footprint_authority: SourceFootprintFoldAuthority,
    population_decision: SourceFoldBaseDecision,
    method_contract: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    """Seal the preregistered field-base fallback for degenerate groups."""

    if population_decision.run_source_oof or population_decision.selected_action != FIELD_BASE_ACTION:
        raise ValueError("field-base receipt requires a degenerate population")
    if population_decision.authority_sha256 != footprint_authority.authority_sha256:
        raise ValueError("field-base population authority differs")
    capability_path = Path(capability_cache).expanduser().resolve()
    graph_path = Path(support_graph).expanduser().resolve()
    footprint_asset = Path(footprint_path).expanduser().resolve()
    evidence_path = evidence_authority.path.expanduser().resolve()
    if any(
        not path.is_file()
        for path in (capability_path, graph_path, footprint_asset, evidence_path)
    ):
        raise FileNotFoundError("field-base receipt requires immutable source assets")
    if _file_sha256(evidence_path) != evidence_authority.sha256:
        raise ValueError("field-base source-evidence authority changed")
    if _file_sha256(footprint_asset) != str(footprint_file_sha256):
        raise ValueError("field-base footprint authority file changed")
    receipt = {
        "schema_version": 1,
        "artifact_type": "source_observation_surface_safe_footprint_oof_gate_v1",
        "gate_mode": "degenerate_population_field_base",
        "scene_id": str(scene_id),
        "protocol_hash": str(protocol_hash),
        "method_contract_sha256": _json_sha256(dict(method_contract)),
        "capability_cache_sha256": _file_sha256(capability_path),
        "support_graph_sha256": _file_sha256(graph_path),
        "source_evidence_authority_sha256": evidence_authority.sha256,
        "source_evidence_authority_content_sha256": evidence_authority.content_sha256,
        "source_footprint_fold_authority": str(footprint_asset),
        "source_footprint_fold_authority_file_sha256": str(footprint_file_sha256),
        "source_footprint_fold_authority_sha256": footprint_authority.authority_sha256,
        "source_footprint_fold_authority_tensor_bundle_sha256": (
            footprint_authority.tensor_bundle_sha256
        ),
        "fold_artifacts": {},
        "fold_assignment": "splitmix64_source_footprint_group_v1",
        "num_folds": 3,
        "minimum_positive_rows_per_training_or_heldout_fold": (
            population_decision.minimum_class_rows
        ),
        "minimum_negative_rows_per_training_or_heldout_fold": (
            population_decision.minimum_class_rows
        ),
        "metrics": {},
        "fold_reports": list(population_decision.fold_reports),
        "selected_action": FIELD_BASE_ACTION,
        "degenerate_reason": population_decision.reason,
        "full_fit_predictions_used_as_oof": False,
        "connected_selection": "off",
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = Path(output_dir).expanduser().resolve() / "source_observation_oof_gate_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(
            f"refusing to overwrite different source-footprint base gate: {output}"
        )
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    return output, receipt


def _load_source_observation_footprint_authority(
    args: argparse.Namespace,
    capability_bank,
) -> tuple[Path, str, SourceFootprintFoldAuthority] | None:
    """Load the explicit structured-fold authority before prompt labels open."""

    mode = str(
        getattr(
            args,
            "source_observation_oof_fold_mode",
            "stable_primitive_rows_v1",
        )
    )
    path_value = str(
        getattr(args, "source_footprint_fold_authority", "")
    ).strip()
    file_sha256 = str(
        getattr(args, "source_footprint_fold_authority_file_sha256", "")
    ).strip()
    authority_sha256 = str(
        getattr(args, "source_footprint_fold_authority_sha256", "")
    ).strip()
    values = (path_value, file_sha256, authority_sha256)
    if mode == "stable_primitive_rows_v1":
        if any(values):
            raise ValueError(
                "legacy source-observation folds forbid source-footprint authority inputs"
            )
        return None
    if mode != "source_footprint_v1":
        raise ValueError("source-observation OOF fold mode is unregistered")
    if not str(getattr(args, "source_observation_oof_output_dir", "")).strip():
        raise ValueError("source_footprint_v1 requires source-observation OOF output")
    if not all(values):
        raise ValueError(
            "source_footprint_v1 requires authority path, file SHA-256, and authority SHA-256"
        )
    if capability_bank is None:
        raise ValueError("source_footprint_v1 requires a canonical capability bank")
    path = Path(path_value).expanduser().resolve()
    authority = load_source_footprint_fold_authority(
        path,
        expected_file_sha256=file_sha256,
        expected_authority_sha256=authority_sha256,
    )
    valid = capability_bank.valid.detach().bool().cpu().reshape(-1)
    rows = capability_bank.global_rows.detach().long().cpu().reshape(-1)
    if (
        not torch.equal(rows, torch.where(valid)[0])
        or not torch.equal(authority.primitive_rows, rows)
    ):
        raise ValueError(
            "footprint rows, capability global_rows, and sorted valid rows must match exactly"
        )
    return path, file_sha256, authority


def _validate_source_observation_oof_contract(
    args: argparse.Namespace,
    *,
    prompt_type: str,
) -> None:
    """Fail closed unless the invocation is exactly the preregistered gate."""

    if not str(getattr(args, "source_observation_oof_output_dir", "")).strip():
        return
    common = {
        "canonical support": str(args.support_mode) == "canonical_support",
        "real support graph": bool(str(args.canonical_support_graph).strip())
        and not bool(args.disable_registered_graph),
        "field capability contract": str(args.canonical_capability_source_contract)
        == "field",
        "exact raster adjoint": str(args.prompt_registration_mode)
        == "raster_adjoint",
        "native prompt scale": float(args.prompt_registration_scale) == 1.0,
        "zero registration alpha threshold": float(args.alpha_threshold) == 0.0,
        "typed graph policy": str(args.graph_policy) == "typed",
        "same component graph policy": str(args.component_graph_policy) == "same",
        "no primitive reliability modulation": not str(
            args.canonical_reliability_cache
        ).strip(),
        "no graph affinity override": not str(
            args.diagnostic_graph_affinity_override
        ).strip(),
        "no query-conditioned diffusion": str(args.query_conditioned_diffusion_kernel)
        == "none",
        "no forward unary": str(args.registered_forward_unary) == "none",
        "no negative spatial augmentation": str(args.negative_spatial_mode) == "none",
        "all-component selection": str(args.registered_selection_mode)
        == SelectionMode.ALL_COMPONENTS.value,
        "propagated readout": str(args.registered_readout_stage) == "propagated",
        "typed confidence random walker": str(args.solver_type)
        == "confidence_random_walker",
        "no prompt-cycle diagnostic": not bool(
            args.export_registered_prompt_cycle_diagnostic
        ),
        "no reference threshold calibration": not bool(
            args.registered_reference_threshold_calibration
        ),
        "asset hashes required": bool(args.require_asset_hashes),
    }
    if prompt_type == "reference_binary_mask":
        task = {
            "SPIn K4 compiler": int(args.prototype_count) == 4,
            "SPIn direct adjoint unary": str(args.registered_observation_fusion)
            == "direct_raster_adjoint",
            "SPIn joint signed seeds": str(args.registered_seed_construction)
            == "joint_signed",
            "SPIn shared prototype seeds": str(
                args.registered_prototype_seed_construction
            )
            == "shared",
        }
    elif prompt_type == "positive_negative_scribbles":
        task = {
            "NVOS K16 compiler": int(args.prototype_count) == 16,
            "NVOS anchor-only unary": str(args.registered_observation_fusion)
            == "hard_seed_anchor_only_probability",
            "NVOS Poisson coverage confidence": str(
                args.registered_observation_confidence
            )
            == "poisson_mass_coverage",
            "NVOS joint signed seeds": str(args.registered_seed_construction)
            == "joint_signed",
            "NVOS winner-take-all prototype seeds": str(
                args.registered_prototype_seed_construction
            )
            == "winner_take_all",
            "NVOS fixed mass scale": float(args.registered_observation_mass_scale)
            == 1.0,
            "NVOS fixed coverage power": float(
                args.registered_observation_coverage_power
            )
            == 1.0,
            "NVOS fixed hard-seed threshold": float(args.hard_seed_threshold)
            == 0.2,
            "NVOS exclusive relative conflicts": str(
                args.hard_seed_conflict_policy
            )
            == "exclusive_relative",
            "NVOS zero conflict margin": float(args.hard_seed_conflict_margin)
            == 0.0,
        }
    else:
        raise ValueError("source-observation OOF gate prompt type is unregistered")
    failed = [name for name, valid in {**common, **task}.items() if not valid]
    if failed:
        raise ValueError(
            "source-observation OOF contract differs: " + ", ".join(failed)
        )


def _source_observation_oof_method_contract(
    args: argparse.Namespace,
    *,
    prompt_type: str,
) -> dict[str, object]:
    """Return every primitive-domain constant shared by the three folds."""

    return {
        "schema_version": 1,
        "method": "target_blind_source_observation_evidence_gate_v1",
        "preregistration": {
            "path": str(
                Path(__file__).resolve().parents[2]
                / (
                    "paper/artifacts/"
                    "source_observation_evidence_gate_preregistration_20260805.json"
                )
            ),
            "sha256": _file_sha256(
                Path(__file__).resolve().parents[2]
                / (
                    "paper/artifacts/"
                    "source_observation_evidence_gate_preregistration_20260805.json"
                )
            ),
        },
        "implementation_sha256": {
            "query_specific_propagation_cv.py": _file_sha256(
                Path(__file__).resolve().parents[1]
                / "querying"
                / "query_specific_propagation_cv.py"
            ),
            "eval_nvos_gaussian_first.py": _file_sha256(Path(__file__).resolve()),
            "source_observation_authority.py": _file_sha256(
                Path(__file__).resolve().parents[1]
                / "querying"
                / "source_observation_authority.py"
            ),
        },
        "numeric_replay_correction": {
            "path": str(
                Path(__file__).resolve().parents[2]
                / (
                    "paper/artifacts/"
                    "source_observation_evidence_authority_correction_addendum_20260805.json"
                )
            ),
            "sha256": _file_sha256(
                Path(__file__).resolve().parents[2]
                / (
                    "paper/artifacts/"
                    "source_observation_evidence_authority_correction_addendum_20260805.json"
                )
            ),
        },
        "prompt_type": str(prompt_type),
        "candidate_actions": ["unary", "surface_safe_propagated"],
        "folds": 3,
        "heldout_clear_tensors": [
            "positive_weight",
            "negative_weight",
            "raw_positive_mass",
            "raw_negative_mass",
        ],
        "capability_source_contract": str(
            args.canonical_capability_source_contract
        ),
        "prompt_registration": {
            "mode": str(args.prompt_registration_mode),
            "scale": float(args.prompt_registration_scale),
            "alpha_threshold": float(args.alpha_threshold),
        },
        "compiler": {
            "prototype_count": int(args.prototype_count),
            "prototype_strategy": str(args.prototype_strategy),
            "prompt_support_threshold": float(args.support_threshold),
            "registered_observation_fusion": str(
                args.registered_observation_fusion
            ),
            "registered_observation_confidence": str(
                args.registered_observation_confidence
            ),
            "registered_observation_mass_scale": float(
                args.registered_observation_mass_scale
            ),
            "registered_observation_coverage_power": float(
                args.registered_observation_coverage_power
            ),
            "registered_seed_construction": str(
                args.registered_seed_construction
            ),
            "registered_prototype_seed_construction": str(
                args.registered_prototype_seed_construction
            ),
            "registered_seed_unary_weight": float(
                args.registered_seed_unary_weight
            ),
            "registered_selection_mode": str(args.registered_selection_mode),
        },
        "scoring": {
            "appearance_weight": float(args.appearance_weight),
            "boundary_weight": float(args.boundary_weight),
            "prototype_temperature": float(args.prototype_temperature),
            "feature_calibration": str(args.feature_calibration),
            "background_centroids": int(args.background_centroids),
            "calibration_sample_size": int(args.calibration_sample_size),
            "centroid_iterations": int(args.centroid_iterations),
            "score_calibration": str(args.score_calibration),
            "score_tanh_scale": float(args.score_tanh_scale),
            "score_chunk_size": int(args.score_chunk_size),
            "negative_spatial_mode": str(args.negative_spatial_mode),
        },
        "surface_safe_graph_readout": {
            "graph_policy": str(args.graph_policy),
            "component_graph_policy": str(args.component_graph_policy),
            "graph_legacy_residual": float(args.graph_legacy_residual),
            "channel_confidence_mode": str(args.channel_confidence_mode),
            "solver_type": str(args.solver_type),
            "iterations": int(args.solver_iterations),
            "residual": float(args.solver_residual),
            "unary_temperature": float(args.solver_unary_temperature),
            "support_threshold": float(args.solver_support_threshold),
            "laplacian_weight": float(args.laplacian_weight),
            "cg_iterations": int(args.cg_iterations),
            "cg_tolerance": float(args.cg_tolerance),
            "hard_seed_threshold": float(args.hard_seed_threshold),
            "hard_seed_conflict_policy": str(args.hard_seed_conflict_policy),
            "hard_seed_conflict_margin": float(args.hard_seed_conflict_margin),
            "component_edge_threshold": float(args.component_edge_threshold),
            "seeded_component_min_weight": float(
                args.seeded_component_min_weight
            ),
            "connected_selection": "off",
        },
        "selection": {
            "primary": "responsibility_balanced_log_loss",
            "secondary": "responsibility_weighted_auc",
            "metric_round_decimals": 12,
            "complexity_tiebreak": "unary",
        },
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }


def _source_observation_footprint_oof_method_contract(
    args: argparse.Namespace,
    *,
    prompt_type: str,
    footprint_path: Path,
    footprint_file_sha256: str,
    footprint_authority: SourceFootprintFoldAuthority,
) -> dict[str, object]:
    """Bind the structured source groups without changing the legacy contract."""

    contract = _source_observation_oof_method_contract(
        args,
        prompt_type=prompt_type,
    )
    contract["method"] = "target_blind_source_footprint_observation_evidence_gate_v1"
    contract["fold_assignment"] = "splitmix64_source_footprint_group_v1"
    structured_preregistration = (
        Path(__file__).resolve().parents[2]
        / "paper"
        / "artifacts"
        / "source_raster_dominant_footprint_blocks_v1_implementation_closure_20260805.json"
    )
    contract["structured_fold_preregistration"] = {
        "path": str(structured_preregistration),
        "sha256": _file_sha256(structured_preregistration),
    }
    contract["source_footprint_fold_authority"] = {
        "path": str(footprint_path),
        "file_sha256": str(footprint_file_sha256),
        "authority_sha256": footprint_authority.authority_sha256,
        "tensor_bundle_sha256": footprint_authority.tensor_bundle_sha256,
        "method_contract": "source_raster_dominant_footprint_blocks_v1",
        "loaded_before_source_prompt_labels": True,
    }
    implementation = dict(contract["implementation_sha256"])
    implementation["source_footprint_fold_authority.py"] = _file_sha256(
        Path(__file__).resolve().parents[1]
        / "querying"
        / "source_footprint_fold_authority.py"
    )
    contract["implementation_sha256"] = implementation
    return contract


def _write_pre_metric_prediction_receipt(
    path: str | Path,
    *,
    scene_id: str,
    protocol_hash: str,
    capability_cache: str | Path,
    support_graph: str | Path,
    score_paths: Mapping[str, str],
    score_sha256: Mapping[str, str],
    stage_score_paths: Mapping[str, Mapping[str, str]],
    stage_score_sha256: Mapping[str, Mapping[str, str]],
    method_contract: Mapping[str, object],
    graph_disabled: bool = False,
) -> tuple[Path, str]:
    """Seal every rendered target score before opening target masks.

    The evaluator has always rendered and persisted all target scores before
    entering its metric loop.  This explicit receipt makes that ordering
    externally auditable and fail-closed: an existing receipt may be reused
    only when its canonical JSON bytes are identical.
    """

    if not score_paths or set(score_paths) != set(score_sha256):
        raise ValueError("prediction receipt requires aligned non-empty score maps")
    if set(stage_score_paths) != set(stage_score_sha256):
        raise ValueError("prediction receipt stage names do not align")
    for stage_name, stage_paths in stage_score_paths.items():
        if set(stage_paths) != set(score_paths):
            raise ValueError(
                f"prediction receipt stage {stage_name!r} lacks target frames"
            )
        if set(stage_paths) != set(stage_score_sha256[stage_name]):
            raise ValueError(
                f"prediction receipt stage {stage_name!r} hashes do not align"
            )

    capability_path = Path(capability_cache).expanduser().resolve()
    if not capability_path.is_file():
        raise FileNotFoundError(
            f"prediction receipt capability cache is missing: {capability_path}"
        )
    graph_value = str(support_graph).strip()
    if bool(graph_disabled):
        if graph_value:
            raise ValueError("graph-disabled prediction receipt forbids a graph asset")
        graph_authority: dict[str, object] = {
            "policy": "disabled_zero_edge_unary_prior_only",
            "path": None,
            "sha256": None,
        }
    else:
        graph_path = Path(graph_value).expanduser().resolve()
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"prediction receipt support graph is missing: {graph_path}"
            )
        graph_authority = {
            "policy": "frozen_asset",
            "path": str(graph_path),
            "sha256": _file_sha256(graph_path),
        }

    def _records(
        paths: Mapping[str, str], hashes: Mapping[str, str]
    ) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for frame_id in sorted(paths):
            score_path = Path(paths[frame_id]).expanduser().resolve()
            expected = str(hashes[frame_id])
            if not score_path.is_file() or _file_sha256(score_path) != expected:
                raise ValueError(
                    f"prediction receipt score changed before sealing: {score_path}"
                )
            records[str(frame_id)] = {
                "path": str(score_path),
                "sha256": expected,
            }
        return records

    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_pre_metric_prediction_receipt_v1",
        "scene_id": str(scene_id),
        "protocol_hash": str(protocol_hash),
        "capability_cache": {
            "path": str(capability_path),
            "sha256": _file_sha256(capability_path),
        },
        "support_graph": graph_authority,
        "method_contract": dict(method_contract),
        "target_scores": _records(score_paths, score_sha256),
        "stage_target_scores": {
            str(stage_name): _records(
                stage_score_paths[stage_name], stage_score_sha256[stage_name]
            )
            for stage_name in sorted(stage_score_paths)
        },
        "sealed_before_target_ground_truth_open": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_opened": False,
    }
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise ValueError(
                "refusing to overwrite a different frozen prediction receipt: "
                f"{output}"
            )
    else:
        output.write_text(encoded, encoding="utf-8")
    return output, _file_sha256(output)


def _requires_legacy_prototype_observation(observation_fusion: str) -> bool:
    """Return whether a second, frozen legacy observation must be rendered."""

    return str(observation_fusion) == "dual_registration_bernoulli_poe"


def _rasterize_frozen_legacy_prototype_support(
    *,
    model: object,
    renderer: object,
    viewmat: torch.Tensor,
    prompt_maps: torch.Tensor,
    depth_tolerance: float,
    relative_depth_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact historical canonical-support prompt operator.

    The native exact-adjoint path requires alpha threshold zero, but the
    historical K4 prototype expert used the ordinary evaluator default 0.02.
    Keeping this helper independent prevents the exact contract from silently
    changing the legacy expert again.
    """

    prompt_aux = renderer.render_features(model, viewmat)
    return rasterize_registered_view_features(
        model=model,
        renderer=renderer,
        viewmat=viewmat,
        siglip_feat=prompt_maps,
        depth_map=prompt_aux["depth_map"][None],
        alpha_map=prompt_aux["alpha_map"][None],
        registration_depth_tolerance=float(depth_tolerance),
        registration_relative_depth_tolerance=float(
            relative_depth_tolerance
        ),
        registration_alpha_threshold=(
            _FROZEN_LEGACY_PROTOTYPE_ALPHA_THRESHOLD
        ),
        registration_weight_mode="alpha_depth",
        deterministic_cpu_accumulation=True,
    )


def _scaled_raster_shape(
    height: int,
    width: int,
    scale: float,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("raster dimensions must be positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("raster scale must be finite and positive")
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _valid_normalized_score_map(
    rendered_channels: torch.Tensor,
    *,
    eps: float = 1e-6,
    coverage_power: float = 0.0,
) -> torch.Tensor:
    """Return a valid-conditioned score with optional coverage abstention.

    ``coverage_power=0`` is ``E[p | valid]`` while ``coverage_power=1``
    exactly recovers the total-alpha score ``E[v*p]``. Intermediate values
    keep the conditional score but lower confidence where few visible
    contributions have a valid capability row.
    """

    channels = torch.as_tensor(rendered_channels)
    if channels.ndim != 3 or channels.shape[0] != 2:
        raise ValueError("valid-normalized render must contain [numerator,validity]")
    if not np.isfinite(coverage_power) or coverage_power < 0:
        raise ValueError("coverage_power must be finite and non-negative")
    numerator, valid_mass = channels
    supported = valid_mass > float(eps)
    conditional = torch.where(
        supported,
        numerator / valid_mass.clamp_min(float(eps)),
        torch.zeros_like(numerator),
    )
    if coverage_power == 0:
        return conditional
    return conditional * valid_mass.clamp(0.0, 1.0).pow(float(coverage_power))


def _registered_solver_masses(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    support_threshold: float,
    construction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive continuous solver/prototype masses from one joint observation.

    ``winner_take_all`` is the historical behavior.  ``joint_signed`` keeps
    the positive/negative competition in one scale: equal evidence becomes
    neutral, while raster tails are discounted by both purity and prompt
    coverage instead of being independently promoted to a hard seed.
    """

    foreground = torch.as_tensor(positive).float().reshape(-1)
    background = torch.as_tensor(negative).float().reshape(-1)
    if (
        foreground.shape != background.shape
        or foreground.numel() == 0
        or not bool(torch.isfinite(foreground).all())
        or not bool(torch.isfinite(background).all())
        or bool((foreground < 0).any())
        or bool((background < 0).any())
    ):
        raise ValueError("registered positive/negative masses must be finite and aligned")
    threshold = float(support_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("support_threshold must be finite and non-negative")
    mode = str(construction)
    if mode == "winner_take_all":
        positive_support = (foreground > threshold) & (
            foreground >= background
        )
        negative_support = (background > threshold) & (
            background > foreground
        )
        return (
            torch.where(positive_support, foreground, 0.0),
            torch.where(negative_support, background, 0.0),
        )
    if mode != "joint_signed":
        raise ValueError(
            "registered seed construction must be winner_take_all or joint_signed"
        )
    observed = foreground + background > threshold
    signed = foreground - background
    return (
        torch.where(observed, signed.clamp_min(0.0), 0.0),
        torch.where(observed, (-signed).clamp_min(0.0), 0.0),
    )


def _joint_signed_observation_seeds(
    signed_observation: torch.Tensor,
    confidence: torch.Tensor,
    *,
    support_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one bounded signed observation without per-sign renormalization."""

    signed = torch.as_tensor(signed_observation).float().reshape(-1)
    mass = torch.as_tensor(confidence).float().reshape(-1)
    threshold = float(support_threshold)
    if (
        signed.numel() == 0
        or signed.shape != mass.shape
        or not bool(torch.isfinite(signed).all())
        or not bool(torch.isfinite(mass).all())
        or bool((mass < 0).any())
        or bool((mass > 1).any())
        or bool((signed.abs() > mass + 1e-6).any())
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("joint signed observation/threshold is invalid")
    observed = mass > threshold
    return (
        torch.where(observed, signed.clamp_min(0.0), 0.0),
        torch.where(observed, (-signed).clamp_min(0.0), 0.0),
    )


def _require_bipolar_solver_support(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    label: str,
) -> tuple[int, int]:
    positive_count = int((torch.as_tensor(positive) > 0).sum())
    negative_count = int((torch.as_tensor(negative) > 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            f"{label} registered prompt support is empty: "
            f"pos={positive_count}, neg={negative_count}"
        )
    return positive_count, negative_count


def _render_registered_stage_maps(
    stage_values: Mapping[str, torch.Tensor],
    *,
    final_stage: str,
    final_rendered: np.ndarray,
    render: Callable[[torch.Tensor], np.ndarray],
) -> dict[str, np.ndarray]:
    """Render every diagnostic from its own primitive tensor.

    Reusing the already rendered final tensor is an optimization only for the
    stage that actually produced it.  This prevents a propagated final output
    from being mislabeled as the connected diagnostic.
    """

    resolved = str(final_stage)
    if resolved not in stage_values:
        raise ValueError(f"final registered readout stage {resolved!r} is unavailable")
    return {
        name: final_rendered if name == resolved else render(values)
        for name, values in stage_values.items()
    }


def _prompt_cycle_reconstruction_metrics(
    score: np.ndarray,
    prompt_mask: np.ndarray,
    visibility: np.ndarray,
) -> dict[str, float]:
    """Measure reference-view reconstruction without any target-frame state."""

    probability = np.asarray(score, dtype=np.float64)
    mask = np.asarray(prompt_mask, dtype=bool)
    visible = np.asarray(visibility, dtype=np.float64)
    if probability.shape != mask.shape or visible.shape != mask.shape:
        raise ValueError("prompt-cycle score, mask, and visibility must align")
    if (
        probability.size == 0
        or not np.isfinite(probability).all()
        or not np.isfinite(visible).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or np.any(visible < 0.0)
        or np.any(visible > 1.0)
        or not mask.any()
        or mask.all()
    ):
        raise ValueError("prompt-cycle inputs must be finite, bounded, and bipolar")
    lower = np.nextafter(np.float64(0.0), np.float64(1.0))
    upper = np.nextafter(np.float64(1.0), np.float64(0.0))
    clipped = np.clip(probability, lower, upper)
    per_pixel_bce = -(
        mask.astype(np.float64) * np.log(clipped)
        + (~mask).astype(np.float64) * np.log1p(-clipped)
    )
    foreground_bce = float(per_pixel_bce[mask].mean())
    background_bce = float(per_pixel_bce[~mask].mean())
    intersection = float(probability[mask].sum())
    union = float(
        (probability + mask.astype(np.float64) - probability * mask).sum()
    )
    visible_mass = float(visible.sum())
    visible_intersection = float((visible * probability * mask).sum())
    visible_union = float(
        (
            visible
            * (
                probability
                + mask.astype(np.float64)
                - probability * mask
            )
        ).sum()
    )
    prompt_foreground_mass = float(mask.sum())
    return {
        "bce": float(per_pixel_bce.mean()),
        "balanced_bce": 0.5 * (foreground_bce + background_bce),
        "foreground_bce": foreground_bce,
        "background_bce": background_bce,
        "soft_iou": intersection / union if union > 0.0 else 1.0,
        "visibility_weighted_bce": (
            float((visible * per_pixel_bce).sum()) / visible_mass
            if visible_mass > 0.0
            else float("inf")
        ),
        "visibility_weighted_soft_iou": (
            visible_intersection / visible_union
            if visible_union > 0.0
            else 1.0
        ),
        "foreground_soft_recall": float(probability[mask].mean()),
        "background_soft_specificity": float((1.0 - probability[~mask]).mean()),
        "predicted_to_prompt_foreground_mass_ratio": (
            float(probability.sum()) / prompt_foreground_mass
        ),
        "mean_visibility": float(visible.mean()),
        "foreground_mean_visibility": float(visible[mask].mean()),
        "background_mean_visibility": float(visible[~mask].mean()),
        "visible_pixel_fraction": float((visible > 0.0).mean()),
    }


def _prompt_cycle_fixed_ranking(
    expert_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Apply predeclared parameter-free rankings in prompt space only."""

    if set(expert_metrics) != {"prototype_expert", "exact_expert"}:
        raise ValueError("prompt-cycle ranking requires exactly two named experts")
    soft_choice = max(
        expert_metrics,
        key=lambda name: float(expert_metrics[name]["soft_iou"]),
    )
    bce_choice = min(
        expert_metrics,
        key=lambda name: float(expert_metrics[name]["balanced_bce"]),
    )
    soft_values = {
        name: float(metrics["soft_iou"])
        for name, metrics in expert_metrics.items()
    }
    soft_sum = sum(soft_values.values())
    soft_weights = (
        {name: value / soft_sum for name, value in soft_values.items()}
        if soft_sum > 0.0
        else {name: 0.5 for name in expert_metrics}
    )
    likelihoods = {
        name: float(np.exp(-float(metrics["balanced_bce"])))
        for name, metrics in expert_metrics.items()
    }
    likelihood_sum = sum(likelihoods.values())
    likelihood_weights = {
        name: value / likelihood_sum for name, value in likelihoods.items()
    }
    return {
        "soft_iou_choice": soft_choice,
        "balanced_bce_choice": bce_choice,
        "consensus_choice": soft_choice if soft_choice == bce_choice else None,
        "abstains_on_metric_disagreement": True,
        "soft_iou_normalized_weights": soft_weights,
        "mean_bernoulli_likelihood_weights": likelihood_weights,
        "uses_target_rgb_or_mask": False,
        "learned_or_scene_tuned_constants": False,
    }


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registered_forward_unary_contract(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Return the new-method contract without perturbing the legacy default."""

    mode = str(getattr(args, "registered_forward_unary", "none"))
    if mode == "none":
        return None
    if mode not in {"beta_coverage_v1", "beta_balanced_residual_v2"}:
        raise ValueError(f"unknown registered forward unary mode {mode!r}")
    if mode == "beta_balanced_residual_v2":
        return {
            "mode": mode,
            "status": "protocol_authority_bound_non_exact_diagnostic",
            "strict_unseen_eligible": False,
            "strict_unseen_scoring_binding": (
                "nvos_strict_unseen_v1_non_exact_beta_centered_posterior"
            ),
            "field_prior_stage": "post_reliability_pre_registered_fusion",
            "field_prior_formula": (
                "sigmoid(field_unary/solver_unary_temperature)"
            ),
            "field_prior_precision_source": (
                "canonical_query_independent_reliability_v1"
            ),
            "field_prior_concentration_formula": (
                "kappa_i=1+reliability_i*observation_coverage_i"
            ),
            "field_prior_concentration_bounds": {
                "minimum": 1.0,
                "maximum": 2.0,
            },
            "class_balance": {
                "scope": "global_expected_counts",
                "formula": (
                    "B=(sum(n_pos_raw)+sum(n_neg_raw))/2; "
                    "n_pos=B*n_pos_raw/sum(n_pos_raw); "
                    "n_neg=B*n_neg_raw/sum(n_neg_raw)"
                ),
                "class_prior_from_scribble_area": False,
                "one_sided_observable_policy": "fail_closed",
                "zero_observable_policy": "exact_field_fallback",
            },
            "residual_evidence_concentration_formula": "m_i/(1+m_i)",
            "residual_evidence_concentration_bounds": {
                "minimum": 0.0,
                "maximum_exclusive": 1.0,
            },
            "semantic_precision_is_primary_for_nonanchors": True,
            "anchor": {
                "strength_formula": (
                    "abs(n_pos_raw-n_neg_raw)/(1+n_pos_raw+n_neg_raw)"
                ),
                "threshold_source": "solver.hard_seed_threshold",
                "positive": "strength>=threshold and n_pos_raw>n_neg_raw",
                "negative": "strength>=threshold and n_neg_raw>n_pos_raw",
                "probability_override": "positive=1;negative=0",
                "solver_constraint": "promote_matching_seed_weight_to_one",
                "conflict_policy": "strict_count_dominance_ties_unanchored",
            },
            "registered_observation_confidence_role": "seed_construction_only",
            "compositor": "exact_front_to_back_sparse_triplets",
            "prompt_registration_mode": "raster_adjoint",
            "prompt_registration_scale": 1.0,
            "compositor_resolution": "native_prompt_registration_raster",
            "compositor_alpha_threshold": 0.0,
            "score_feature_contribution_gamma": 1.0,
            "score_gamma_role": "target_score_render_only_not_forward_e_step",
            "capability_invalid_policy": "exclude_before_forward_and_e_step",
            "labeled_policy": "positive_or_negative_only",
            "unlabeled_scribble_policy": "unobserved_not_negative",
            "all_pixel_policy": "all_prompt_registration_raster_pixels",
            "e_steps": 1,
            "posterior_formula": (
                "(kappa_i*p_field_i+n_res_i*mu_i)/(kappa_i+n_res_i)"
            ),
            "accumulation_dtype": "float64_cpu",
            "evidence_dtype": "float32",
            "nll_eps": 1e-12,
            "saturated_likelihood_policy": (
                "sign_symmetric_common_perturbation_limit"
            ),
            "nll_used_for_selection_or_calibration": False,
            "selection_applied_to_main_output": False,
            "required_final_readout": "propagated",
            "uses_target_calibration": False,
            "uses_scene_id_branching": False,
            "scoring_adapter": _registered_forward_scoring_contract(args),
        }
    return {
        "mode": mode,
        "status": "protocol_authority_bound_non_exact_diagnostic",
        "strict_unseen_eligible": False,
        "strict_unseen_scoring_binding": (
            "nvos_strict_unseen_v1_non_exact_beta_centered_posterior"
        ),
        "field_prior_stage": "post_reliability_pre_registered_fusion",
        "field_prior_formula": (
            "sigmoid(field_unary/solver_unary_temperature)"
        ),
        "registered_observation_confidence_role": "seed_construction_only",
        "compositor": "exact_front_to_back_sparse_triplets",
        "prompt_registration_mode": "raster_adjoint",
        "prompt_registration_scale": 1.0,
        "compositor_resolution": "native_prompt_registration_raster",
        "compositor_alpha_threshold": 0.0,
        "score_feature_contribution_gamma": 1.0,
        "score_gamma_role": "target_score_render_only_not_forward_e_step",
        "capability_invalid_policy": "exclude_before_forward_and_e_step",
        "labeled_policy": "positive_or_negative_only",
        "unlabeled_scribble_policy": "unobserved_not_negative",
        "all_pixel_policy": "all_prompt_registration_raster_pixels",
        "e_steps": 1,
        "prior_pseudocount": 1.0,
        "confidence_formula": "1-(1-rho)/(1+n)",
        "fusion_formula": "(1-c)*p_field+c*mu",
        "accumulation_dtype": "float64_cpu",
        "evidence_dtype": "float32",
        "nll_eps": 1e-12,
        "saturated_likelihood_policy": (
            "sign_symmetric_common_perturbation_limit"
        ),
        "nll_used_for_selection_or_calibration": False,
        "selection_applied_to_main_output": False,
        "required_final_readout": "propagated",
        "scoring_adapter": _registered_forward_scoring_contract(args),
    }


def _registered_forward_scoring_contract(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Return the non-exact strict-row scoring adapter selected by the method."""

    mode = str(getattr(args, "registered_forward_unary", "none"))
    if mode == "none":
        return None
    if mode not in {"beta_coverage_v1", "beta_balanced_residual_v2"}:
        raise ValueError(f"unknown registered forward unary mode {mode!r}")
    return {
        "score_semantics": "beta_centered_posterior",
        "prediction_representation": "continuous_beta_centered_posterior",
        "threshold": {"comparison": "greater_or_equal", "value": 0.0},
        "resize": "nearest",
    }


def _center_registered_forward_score_map(
    posterior: np.ndarray,
) -> np.ndarray:
    """Map a rendered foreground posterior to a zero-centered score."""

    values = np.asarray(posterior)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("rendered beta posterior must be a floating 2-D array")
    if not bool(np.isfinite(values).all()):
        raise ValueError("rendered beta posterior contains NaN or infinity")
    tolerance = 1e-6
    if bool((values < -tolerance).any()) or bool((values > 1.0 + tolerance).any()):
        raise ValueError("rendered beta posterior must lie in [0,1]")
    clipped = np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)
    return clipped * np.float32(2.0) - np.float32(1.0)


def _resize_nvos_score_for_evaluation(
    score: np.ndarray,
    target_shape: tuple[int, int],
    *,
    registered_forward_unary: str,
) -> np.ndarray:
    """Apply the selected method's explicit score-resize adapter."""

    height, width = map(int, target_shape)
    if height <= 0 or width <= 0:
        raise ValueError("target score shape must be positive")
    mode = str(registered_forward_unary)
    if mode == "none":
        interpolation = cv2.INTER_LINEAR
    elif mode in {"beta_coverage_v1", "beta_balanced_residual_v2"}:
        interpolation = cv2.INTER_NEAREST
    else:
        raise ValueError(f"unknown registered forward unary mode {mode!r}")
    return cv2.resize(
        np.asarray(score),
        (width, height),
        interpolation=interpolation,
    )


def _load_registered_forward_protocol_authority(
    args: argparse.Namespace,
    candidate_run_manifest: Mapping[str, object] | None,
    candidate_method_contract_sha256: str,
) -> dict[str, object] | None:
    """Validate the hash-bound inline receipt without opening authority sources."""

    scoring = _registered_forward_scoring_contract(args)
    if scoring is None:
        return None
    method_sha = str(candidate_method_contract_sha256)
    if len(method_sha) != 64 or any(
        character not in "0123456789abcdef" for character in method_sha
    ):
        raise ValueError(
            "registered forward Beta requires a validated candidate method contract SHA256"
        )
    if not isinstance(candidate_run_manifest, Mapping):
        raise ValueError(
            "registered forward Beta requires an authority-bound candidate run manifest"
        )
    raw_authority = candidate_run_manifest.get(
        "registered_forward_protocol_authority"
    )
    declared_sha256 = candidate_run_manifest.get(
        "registered_forward_protocol_authority_sha256"
    )
    if not isinstance(raw_authority, Mapping):
        raise ValueError(
            "beta candidate run manifest lacks inline protocol authority"
        )
    authority = json.loads(json.dumps(raw_authority))
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in declared_sha256
        )
        or _json_sha256(authority) != declared_sha256
    ):
        raise ValueError("inline protocol authority canonical SHA256 differs")

    # Lazy import keeps the historical path independent of the new authority
    # implementation. Validation is pure: the checked-in authority builder is
    # used by the manifest producer, while snapshot runtime opens no paper/
    # source authority file and accepts no caller authority path or exact flag.
    from radio_gs.scripts.bind_nvos_forward_beta_protocol_authority import (
        validate_authority_payload,
    )

    validate_authority_payload(authority)
    expected_top_level = {
        "schema_version",
        "artifact_type",
        "status",
        "candidate",
        "scoring_contract",
        "strict_unseen_protocol_exact_match",
        "strict_unseen_exact_match_blockers",
        "protocol_provenance",
        "protocol_provenance_sha256",
        "external_comparator_provenance",
    }
    if set(authority) != expected_top_level:
        raise ValueError("inline protocol authority fields differ")
    candidate = authority.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "method_family",
        "method_contract_sha256",
        "parent_method_exact_match",
    }:
        raise ValueError("inline protocol authority candidate fields differ")
    if authority.get("scoring_contract") != scoring:
        raise ValueError("inline protocol authority scoring contract differs")
    if authority.get("strict_unseen_protocol_exact_match") is not False:
        raise ValueError(
            "registered forward Beta protocol authority must remain strict-unseen non-exact"
        )
    if authority.get("strict_unseen_exact_match_blockers") != [
        "score_semantics_differs",
        "prediction_representation_differs",
    ]:
        raise ValueError("inline protocol authority exactness blockers differ")
    if (
        candidate.get("method_contract_sha256")
        != method_sha
    ):
        raise ValueError("protocol authority candidate method SHA256 differs")
    return authority


def _validate_registered_forward_unary_args(args: argparse.Namespace) -> None:
    """Fail closed before model loading when the diagnostic is misconfigured."""

    contract = _registered_forward_unary_contract(args)
    if contract is None:
        return
    requirements = {
        "--support-mode canonical_support": (
            str(args.support_mode) == "canonical_support"
        ),
        "--registered-observation-fusion probability_mixture": (
            str(args.registered_observation_fusion) == "probability_mixture"
        ),
        "--registered-seed-unary-weight 0": (
            float(args.registered_seed_unary_weight) == 0.0
        ),
        "--registered-readout-stage propagated": (
            str(args.registered_readout_stage) == "propagated"
        ),
        "--prompt-registration-mode raster_adjoint": (
            str(args.prompt_registration_mode) == "raster_adjoint"
        ),
        "--prompt-registration-scale 1": (
            float(args.prompt_registration_scale) == 1.0
        ),
        "--alpha-threshold 0": float(args.alpha_threshold) == 0.0,
        "--feature-contribution-gamma 1": (
            float(args.feature_contribution_gamma) == 1.0
        ),
    }
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    mode = str(getattr(args, "registered_forward_unary", "none"))
    if mode == "beta_balanced_residual_v2" and not str(
        getattr(args, "canonical_reliability_cache", "")
    ).strip():
        failed.append("--canonical-reliability-cache <query-independent-v1>")
    if mode == "beta_balanced_residual_v2" and str(
        getattr(args, "registered_seed_construction", "winner_take_all")
    ) != "joint_signed":
        failed.append("--registered-seed-construction joint_signed")
    if mode == "beta_balanced_residual_v2" and not (
        0.0 < float(getattr(args, "hard_seed_threshold", 0.0)) <= 1.0
    ):
        failed.append("--hard-seed-threshold in (0,1]")
    if failed:
        raise ValueError(
            f"{mode} requires " + ", ".join(failed)
        )


def _validate_direct_raster_adjoint_args(args: argparse.Namespace) -> None:
    """Validate the exact direct-observation contract before model loading."""

    observation_fusion = str(args.registered_observation_fusion)
    if observation_fusion not in _EXACT_RASTER_OBSERVATION_FUSIONS:
        return
    requirements = {
        "--support-mode canonical_support": (
            str(args.support_mode) == "canonical_support"
        ),
        "--prompt-registration-mode raster_adjoint": (
            str(args.prompt_registration_mode) == "raster_adjoint"
        ),
        "--prompt-registration-scale 1": (
            float(args.prompt_registration_scale) == 1.0
        ),
        "--alpha-threshold 0": float(args.alpha_threshold) == 0.0,
        "--registered-seed-unary-weight 0": (
            float(args.registered_seed_unary_weight) == 0.0
        ),
        "--registered-seed-construction joint_signed": (
            str(args.registered_seed_construction) == "joint_signed"
        ),
        "--registered-forward-unary none": (
            str(args.registered_forward_unary) == "none"
        ),
    }
    if observation_fusion == "dual_registration_bernoulli_poe":
        requirements.update(
            {
                "--depth-tolerance 0.08": (
                    float(args.depth_tolerance) == 0.08
                ),
                "--relative-depth-tolerance 0.02": (
                    float(args.relative_depth_tolerance) == 0.02
                ),
                "--support-threshold 0": (
                    float(args.support_threshold) == 0.0
                ),
                "--prototype-count 4": int(args.prototype_count) == 4,
                "--prototype-strategy spherical_mean_fps": (
                    str(args.prototype_strategy) == "spherical_mean_fps"
                ),
                "--appearance-weight 1": (
                    float(args.appearance_weight) == 1.0
                ),
                "--boundary-weight 0.35": (
                    float(args.boundary_weight) == 0.35
                ),
                "--prototype-temperature 0.07": (
                    float(args.prototype_temperature) == 0.07
                ),
                "--feature-calibration none": (
                    str(args.feature_calibration) == "none"
                ),
                "--score-calibration none": (
                    str(args.score_calibration) == "none"
                ),
            }
        )
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    if failed:
        raise ValueError(observation_fusion + " requires " + ", ".join(failed))


def _validate_hard_seed_anchor_only_probability_args(
    args: argparse.Namespace,
) -> None:
    """Fail closed on any change to the fixed target-blind anchor-only path."""

    if str(args.registered_observation_fusion) != (
        "hard_seed_anchor_only_probability"
    ):
        return
    requirements = {
        "--support-mode canonical_support": (
            str(args.support_mode) == "canonical_support"
        ),
        "--prompt-registration-mode raster_adjoint": (
            str(args.prompt_registration_mode) == "raster_adjoint"
        ),
        "--prompt-registration-scale 1": (
            float(args.prompt_registration_scale) == 1.0
        ),
        "--alpha-threshold 0": float(args.alpha_threshold) == 0.0,
        "--registered-seed-unary-weight 0": (
            float(args.registered_seed_unary_weight) == 0.0
        ),
        "--registered-seed-construction joint_signed": (
            str(args.registered_seed_construction) == "joint_signed"
        ),
        "--registered-observation-confidence poisson_mass_coverage": (
            str(args.registered_observation_confidence)
            == "poisson_mass_coverage"
        ),
        "--registered-observation-mass-scale 1": (
            float(args.registered_observation_mass_scale) == 1.0
        ),
        "--registered-observation-coverage-power 1": (
            float(args.registered_observation_coverage_power) == 1.0
        ),
        "--hard-seed-threshold 0.2": (
            float(args.hard_seed_threshold) == 0.20
        ),
        "--hard-seed-conflict-policy exclusive_relative": (
            str(args.hard_seed_conflict_policy) == "exclusive_relative"
        ),
        "--hard-seed-conflict-margin 0": (
            float(args.hard_seed_conflict_margin) == 0.0
        ),
        "--registered-forward-unary none": (
            str(args.registered_forward_unary) == "none"
        ),
    }
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    if failed:
        raise ValueError(
            "hard_seed_anchor_only_probability requires " + ", ".join(failed)
        )


def _validate_registered_prototype_seed_construction_args(
    args: argparse.Namespace,
) -> None:
    """Fail closed when prototype coverage is decoupled from solver seeds."""

    construction = str(
        getattr(args, "registered_prototype_seed_construction", "shared")
    )
    if construction == "shared":
        return
    source_completion = _source_completion_unary_contract(args)
    requirements = {
        "--registered-prototype-seed-construction winner_take_all": (
            construction == "winner_take_all"
        ),
        "--support-mode canonical_support": (
            str(args.support_mode) == "canonical_support"
        ),
        "--prompt-registration-mode raster_adjoint": (
            str(args.prompt_registration_mode) == "raster_adjoint"
        ),
        "--prompt-registration-scale 1": (
            float(args.prompt_registration_scale) == 1.0
        ),
        "--alpha-threshold 0": float(args.alpha_threshold) == 0.0,
        "--registered-seed-construction joint_signed": (
            str(args.registered_seed_construction) == "joint_signed"
        ),
        "--registered-observation-fusion hard_seed_anchor_only_probability": (
            str(args.registered_observation_fusion)
            == "hard_seed_anchor_only_probability"
            or (
                source_completion is not None
                and str(args.registered_observation_fusion)
                == "probability_mixture"
            )
        ),
        "--registered-forward-unary none": (
            str(args.registered_forward_unary) == "none"
        ),
    }
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    if failed:
        raise ValueError(
            "decoupled registered prototype seeds require "
            + ", ".join(failed)
        )


def _validate_registered_reference_threshold_calibration_args(
    args: argparse.Namespace,
) -> None:
    """Fail closed before model loading for source-view unary calibration."""

    if not bool(
        getattr(args, "registered_reference_threshold_calibration", False)
    ):
        return
    requirements = {
        "--support-mode canonical_support": (
            str(args.support_mode) == "canonical_support"
        ),
        "--registered-readout-stage unary_prior": (
            str(args.registered_readout_stage) == "unary_prior"
        ),
        "--query-conditioned-diffusion-kernel none": (
            str(args.query_conditioned_diffusion_kernel) == "none"
        ),
        "--registered-forward-unary none": (
            str(args.registered_forward_unary) == "none"
        ),
    }
    failed = [name for name, satisfied in requirements.items() if not satisfied]
    if failed:
        raise ValueError(
            "registered reference-threshold calibration requires "
            + ", ".join(failed)
        )


def _compact_registered_forward_beta_diagnostics(
    diagnostics: RegisteredForwardBetaDiagnostics,
    capability_valid: torch.Tensor,
) -> dict[str, object]:
    """Summarize the CPU vectors without persisting primitive/pixel arrays."""

    valid = torch.as_tensor(capability_valid).detach().bool().cpu().reshape(-1)
    primitive_vectors = {
        "positive_expected_count": diagnostics.positive_expected_count,
        "negative_expected_count": diagnostics.negative_expected_count,
        "labeled_expected_count": diagnostics.labeled_expected_count,
        "visible_contribution_mass": diagnostics.visible_contribution_mass,
        "labeled_contribution_mass": diagnostics.labeled_contribution_mass,
        "labeled_coverage": diagnostics.labeled_coverage,
        "beta_confidence": diagnostics.beta_confidence,
        "effective_confidence": diagnostics.effective_confidence,
    }
    vectors = {
        name: torch.as_tensor(values).detach().double().cpu().reshape(-1)
        for name, values in primitive_vectors.items()
    }
    if any(values.shape != valid.shape for values in vectors.values()):
        raise ValueError("registered forward diagnostics do not align with capability rows")
    if any(not bool(torch.isfinite(values).all()) for values in vectors.values()):
        raise ValueError("registered forward diagnostics contain NaN or infinity")

    observed = valid & (vectors["labeled_expected_count"] > 0)
    visible = valid & (vectors["visible_contribution_mass"] > 0)

    def distribution(values: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
        selected = values[mask]
        if selected.numel() == 0:
            return {"count": 0, "min": None, "q50": None, "q90": None, "max": None}
        quantiles = torch.quantile(
            selected,
            torch.tensor([0.5, 0.9], dtype=torch.float64),
        )
        return {
            "count": int(selected.numel()),
            "min": float(selected.min()),
            "q50": float(quantiles[0]),
            "q90": float(quantiles[1]),
            "max": float(selected.max()),
        }

    compact = {
        "protocol_status": str(diagnostics.protocol_status),
        "nll": {
            "before": float(diagnostics.nll_before),
            "after": float(diagnostics.nll_after),
            "delta": float(diagnostics.nll_after - diagnostics.nll_before),
            "used_for_selection_or_calibration": False,
        },
        "observable_labeled_alpha_mass": float(
            diagnostics.observable_labeled_alpha_mass
        ),
        "observable_labeled_pixel_count": int(
            diagnostics.observable_labeled_pixel_count
        ),
        "unobservable_labeled_pixel_count": int(
            diagnostics.unobservable_labeled_pixel_count
        ),
        "valid_hit_count": int(diagnostics.valid_hit_count),
        "row_counts": {
            "all": int(valid.numel()),
            "capability_valid": int(valid.sum()),
            "visible_valid": int(visible.sum()),
            "observed_valid": int(observed.sum()),
        },
        "sums": {
            name: float(values[valid].sum())
            for name, values in vectors.items()
        },
        "distributions": {
            "labeled_expected_count_observed": distribution(
                vectors["labeled_expected_count"], observed
            ),
            "labeled_coverage_visible": distribution(
                vectors["labeled_coverage"], visible
            ),
            "beta_confidence_observed": distribution(
                vectors["beta_confidence"], observed
            ),
            "effective_confidence_observed": distribution(
                vectors["effective_confidence"], observed
            ),
        },
        "vectors_persisted": False,
    }
    optional_vectors = {
        "raw_positive_expected_count": diagnostics.raw_positive_expected_count,
        "raw_negative_expected_count": diagnostics.raw_negative_expected_count,
        "field_prior_reliability": diagnostics.field_prior_reliability,
        "field_prior_coverage": diagnostics.field_prior_coverage,
        "field_prior_concentration": diagnostics.field_prior_concentration,
        "residual_evidence_concentration": (
            diagnostics.residual_evidence_concentration
        ),
    }
    if any(value is not None for value in optional_vectors.values()):
        if any(value is None for value in optional_vectors.values()):
            raise ValueError("registered forward v2 diagnostics are incomplete")
        v2_vectors = {
            name: torch.as_tensor(value).detach().double().cpu().reshape(-1)
            for name, value in optional_vectors.items()
        }
        if any(values.shape != valid.shape for values in v2_vectors.values()):
            raise ValueError("registered forward v2 diagnostics do not align")
        if any(
            not bool(torch.isfinite(values).all())
            for values in v2_vectors.values()
        ):
            raise ValueError("registered forward v2 diagnostics are non-finite")
        if (
            diagnostics.positive_anchor_mask is None
            or diagnostics.negative_anchor_mask is None
            or diagnostics.positive_class_balance_scale is None
            or diagnostics.negative_class_balance_scale is None
        ):
            raise ValueError("registered forward v2 anchor diagnostics are absent")
        positive_anchor = torch.as_tensor(
            diagnostics.positive_anchor_mask
        ).detach().bool().cpu().reshape(-1)
        negative_anchor = torch.as_tensor(
            diagnostics.negative_anchor_mask
        ).detach().bool().cpu().reshape(-1)
        if (
            positive_anchor.shape != valid.shape
            or negative_anchor.shape != valid.shape
            or bool((positive_anchor & negative_anchor).any())
        ):
            raise ValueError("registered forward v2 anchor diagnostics are invalid")
        compact["v2"] = {
            "class_balance": {
                "positive_scale": float(
                    diagnostics.positive_class_balance_scale
                ),
                "negative_scale": float(
                    diagnostics.negative_class_balance_scale
                ),
                "raw_positive_sum": float(
                    v2_vectors["raw_positive_expected_count"][valid].sum()
                ),
                "raw_negative_sum": float(
                    v2_vectors["raw_negative_expected_count"][valid].sum()
                ),
                "balanced_positive_sum": float(
                    vectors["positive_expected_count"][valid].sum()
                ),
                "balanced_negative_sum": float(
                    vectors["negative_expected_count"][valid].sum()
                ),
            },
            "anchors": {
                "positive": int((positive_anchor & valid).sum()),
                "negative": int((negative_anchor & valid).sum()),
                "conflicting": 0,
            },
            "distributions": {
                "field_prior_reliability_valid": distribution(
                    v2_vectors["field_prior_reliability"], valid
                ),
                "field_prior_coverage_valid": distribution(
                    v2_vectors["field_prior_coverage"], valid
                ),
                "field_prior_concentration_valid": distribution(
                    v2_vectors["field_prior_concentration"], valid
                ),
                "residual_evidence_concentration_observed": distribution(
                    v2_vectors["residual_evidence_concentration"], observed
                ),
            },
            "vectors_persisted": False,
        }
    return compact


def _execute_registered_forward_beta(
    engine: CanonicalQueryEngine,
    query,
    feature_banks: Mapping[str, torch.Tensor],
    feature_signatures,
    *,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    contribution_weights: torch.Tensor,
    capability_valid: torch.Tensor,
    valid_rows: torch.Tensor,
    positive_pixels: torch.Tensor,
    negative_pixels: torch.Tensor,
    unary_temperature: float,
    mode: str = "beta_coverage_v1",
    primitive_reliability: torch.Tensor | None = None,
    primitive_coverage: torch.Tensor | None = None,
    anchor_threshold: float | None = None,
):
    """Execute the field pass and beta-fused pass around one CPU primitive."""

    valid = torch.as_tensor(capability_valid).detach().bool().cpu().reshape(-1)
    rows = torch.as_tensor(valid_rows).detach().long().cpu().reshape(-1)
    if not torch.equal(rows, torch.where(valid)[0]):
        raise ValueError("valid_rows must exactly enumerate capability_valid")
    temperature = float(unary_temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("unary_temperature must be finite and positive")

    field_query = replace(query, primitive_unary_evidence=None)
    field_result = engine.execute(
        field_query,
        feature_banks,
        feature_signatures=feature_signatures,
    )
    valid_field_prior = torch.sigmoid(
        field_result.unary / temperature
    ).detach().float().cpu()
    if valid_field_prior.shape != rows.shape:
        raise ValueError("field unary does not align with capability-valid rows")
    field_prior = torch.full((valid.numel(),), 0.5, dtype=torch.float32)
    field_prior[rows] = valid_field_prior

    positive = torch.as_tensor(positive_pixels).detach().bool().cpu().reshape(-1)
    negative = torch.as_tensor(negative_pixels).detach().bool().cpu().reshape(-1)
    labeled = positive | negative
    all_pixels = torch.ones_like(labeled)
    resolved_mode = str(mode)
    common_arguments = (
        torch.as_tensor(gaussian_ids).detach().cpu(),
        torch.as_tensor(pixel_ids).detach().cpu(),
        torch.as_tensor(contribution_weights).detach().cpu(),
        valid,
        field_prior,
    )
    if resolved_mode == "beta_coverage_v1":
        if primitive_reliability is not None or primitive_coverage is not None:
            raise ValueError("beta_coverage_v1 does not consume v2 prior precision")
        forward_observation, diagnostics = registered_forward_beta_observation(
            *common_arguments,
            positive,
            negative,
            labeled,
            all_pixels,
        )
    elif resolved_mode == "beta_balanced_residual_v2":
        if primitive_reliability is None or primitive_coverage is None:
            raise ValueError(
                "beta_balanced_residual_v2 requires reliability and coverage"
            )
        if anchor_threshold is None:
            raise ValueError(
                "beta_balanced_residual_v2 requires an anchor threshold"
            )
        forward_observation, diagnostics = (
            registered_forward_beta_balanced_residual_observation(
                *common_arguments,
                torch.as_tensor(primitive_reliability).detach().cpu(),
                torch.as_tensor(primitive_coverage).detach().cpu(),
                positive,
                negative,
                labeled,
                all_pixels,
                anchor_threshold=float(anchor_threshold),
            )
        )
    else:
        raise ValueError(f"unknown registered forward unary mode {resolved_mode!r}")
    forward_valid_observation = PrimitiveUnaryEvidence(
        forward_observation.values[rows],
        forward_observation.source,
        (
            forward_observation.confidence[rows]
            if forward_observation.confidence is not None
            else None
        ),
    )
    final_query = replace(field_query, primitive_unary_evidence=forward_valid_observation)
    if resolved_mode == "beta_balanced_residual_v2":
        if (
            diagnostics.positive_anchor_mask is None
            or diagnostics.negative_anchor_mask is None
            or field_query.positive_seeds is None
            or field_query.negative_seeds is None
        ):
            raise ValueError("beta_balanced_residual_v2 anchor contract is incomplete")
        positive_anchor = diagnostics.positive_anchor_mask[rows]
        negative_anchor = diagnostics.negative_anchor_mask[rows]
        positive_seed = field_query.positive_seeds
        negative_seed = field_query.negative_seeds
        positive_weights = torch.maximum(
            positive_seed.weights,
            positive_anchor.to(
                device=positive_seed.weights.device,
                dtype=positive_seed.weights.dtype,
            ),
        )
        negative_weights = torch.maximum(
            negative_seed.weights,
            negative_anchor.to(
                device=negative_seed.weights.device,
                dtype=negative_seed.weights.dtype,
            ),
        )
        final_query = replace(
            final_query,
            positive_seeds=replace(
                positive_seed,
                weights=positive_weights,
                source=positive_seed.source + "+forward_beta_v2_anchor",
            ),
            negative_seeds=replace(
                negative_seed,
                weights=negative_weights,
                source=negative_seed.source + "+forward_beta_v2_anchor",
            ),
        )
    result = engine.execute(
        final_query,
        feature_banks,
        feature_signatures=feature_signatures,
    )
    return result, field_result, forward_observation, diagnostics


def _declared_prompt_asset_hashes(
    manifest: Mapping[str, object],
    scene: Mapping[str, object],
) -> dict[str, object]:
    """Resolve explicit hashes or the exact same frozen reference-frame asset."""

    scene_id = str(scene["scene_id"])
    protocol = dict(manifest.get("protocol", {}))
    explicit = dict(
        dict(protocol.get("prompt_asset_sha256", {})).get(scene_id, {})
    )
    if explicit:
        return explicit
    prompt = dict(scene.get("prompt", {}))
    frame_records = {
        str(frame["frame_id"]): frame for frame in scene.get("frames", [])
    }
    prompt_frame = frame_records.get(str(prompt.get("frame_id", "")))
    prompt_path = Path(str(prompt.get("mask_path", ""))).resolve()
    frame_path = (
        Path(
            str(
                prompt_frame.get(
                    "ground_truth",
                    prompt_frame.get("gt_mask_path", ""),
                )
            )
        ).resolve()
        if isinstance(prompt_frame, Mapping)
        else None
    )
    frame_digest = (
        str(prompt_frame.get("ground_truth_sha256", "")).strip()
        if isinstance(prompt_frame, Mapping)
        else ""
    )
    if (
        str(prompt.get("type", "")) != "reference_binary_mask"
        or frame_path is None
        or prompt_path != frame_path
        or len(frame_digest) != 64
        or any(character not in "0123456789abcdef" for character in frame_digest)
    ):
        raise ValueError(f"{scene_id}: prompt asset hashes are undeclared")
    return {"reference_binary_mask": frame_digest}


def _dataset_protocol_contract(
    manifest: Mapping[str, object],
    *,
    benchmark_manifest_sha256: str = "",
) -> dict[str, object]:
    """Extract dataset/prompt roles without inheriting a method's score rules."""

    protocol = dict(manifest.get("protocol", {}))
    scenes = list(manifest.get("scenes", []))
    scene_ids = [str(scene["scene_id"]) for scene in scenes]
    cohort = [str(value) for value in protocol.get("cohort", scene_ids)]
    if cohort != scene_ids:
        raise ValueError("manifest scene order differs from its declared cohort")
    scene_contracts: list[dict[str, object]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        prompt = dict(scene.get("prompt", {}))
        frame_records = {
            str(frame["frame_id"]): frame
            for frame in scene.get("frames", [])
        }
        evaluation_targets = []
        for frame_id in scene.get("evaluation_frame_ids", []):
            frame = frame_records.get(str(frame_id))
            if frame is None:
                raise ValueError(
                    f"{scene_id}: evaluation frame {frame_id!r} is undeclared"
                )
            digest = str(frame.get("ground_truth_sha256", "")).strip()
            if not digest:
                raise ValueError(
                    f"{scene_id}: evaluation frame {frame_id!r} lacks target SHA"
                )
            evaluation_targets.append(
                {
                    "frame_id": str(frame_id),
                    "ground_truth_sha256": digest,
                }
            )
        declared_prompt_hashes = _declared_prompt_asset_hashes(manifest, scene)
        scene_contracts.append(
            {
                "scene_id": scene_id,
                "base_scene_id": str(scene.get("base_scene_id") or scene_id),
                "prompt": {
                    "type": str(prompt.get("type", "")),
                    "frame_id": str(prompt.get("frame_id", "")),
                    "asset_sha256": declared_prompt_hashes,
                },
                "prompt_frame_ids": [
                    str(value) for value in scene.get("prompt_frame_ids", [])
                ],
                "calibration_frame_ids": [
                    str(value) for value in scene.get(
                        "calibration_frame_ids", []
                    )
                ],
                "evaluation_frame_ids": [
                    str(value) for value in scene.get(
                        "evaluation_frame_ids", []
                    )
                ],
                "excluded_training_frame_ids": [
                    str(value) for value in scene.get(
                        "excluded_training_frame_ids", []
                    )
                ],
                "training_frame_ids": [
                    str(value["frame_id"])
                    for value in scene.get("training_frames", [])
                ],
                "target_rgb_policy": str(
                    scene.get("target_rgb_policy", "")
                ),
                "evaluation_targets": evaluation_targets,
            }
        )
    return {
        "schema_version": 1,
        "benchmark": str(manifest.get("benchmark", "")),
        "legacy_protocol_hash": str(manifest.get("protocol_hash", "")),
        "benchmark_manifest_sha256": str(benchmark_manifest_sha256),
        "dataset_version": str(protocol.get("dataset_version", "")),
        "task": str(protocol.get("task", "")),
        "cohort": cohort,
        "prompt_type": str(protocol.get("prompt_type", "")),
        "prompt_support": str(protocol.get("prompt_support", "")),
        "target_mask_use": str(protocol.get("target_mask_use", "")),
        "target_rgb_at_query": str(protocol.get("target_rgb_at_query", "")),
        "target_rgb_during_field_training": str(
            protocol.get("target_rgb_during_field_training", "")
        ),
        "allow_reference_scoring": bool(
            protocol.get("allow_reference_scoring", False)
        ),
        "scenes": scene_contracts,
    }


def _verify_declared_sha256(
    path: Path,
    expected: object,
    *,
    label: str,
) -> str:
    declared = str(expected or "").strip()
    if len(declared) != 64:
        raise ValueError(f"{label} lacks a valid declared SHA256")
    actual = _file_sha256(path)
    if actual != declared:
        raise ValueError(f"{label} SHA256 mismatch")
    return actual


def _resolve_scene_carrier_assets(
    queue_scene: Path,
    *,
    scene_config: str = "",
    scene_checkpoint: str = "",
    camera_map: str = "",
) -> tuple[Path, Path, Path]:
    """Resolve either the queue-layout carrier or one explicit frozen bundle."""

    overrides = [
        str(scene_config).strip(),
        str(scene_checkpoint).strip(),
        str(camera_map).strip(),
    ]
    if any(overrides) and not all(overrides):
        raise ValueError(
            "explicit scene carrier requires --scene-config, "
            "--scene-checkpoint, and --camera-map together"
        )
    if all(overrides):
        paths = tuple(Path(value).expanduser().resolve() for value in overrides)
    else:
        paths = (
            queue_scene / "gaussfm_main_track.yaml",
            queue_scene / "feature_field" / "checkpoints" / "best.pth",
            queue_scene / "rgb_to_colmap_camera_mapping.json",
        )
    labels = ("scene config", "scene checkpoint", "camera map")
    missing = [label for label, path in zip(labels, paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing " + ", ".join(missing))
    return paths


def _registered_strong_unary_method_contract(
    observation_fusion: str,
    *,
    anchor_threshold: float,
) -> dict[str, object] | None:
    """Describe the registered strong-unary rule without task-specific state."""

    if observation_fusion == "hard_seed_anchored_probability":
        return {
            "policy": "unit_confidence_on_shared_hard_seed_rows",
            "anchor_threshold_source": "solver.hard_seed_threshold",
            "anchor_threshold": float(anchor_threshold),
            "formula": (
                "a=1[c>0 and abs(s)>=tau]; c_eff=a+(1-a)c; "
                "p=(1-c_eff)p_field+c_eff*q"
            ),
            "new_numeric_constant": False,
        }
    if observation_fusion == "hard_seed_anchor_only_probability":
        return {
            "policy": "anchor_only_on_shared_hard_seed_rows",
            "anchor_threshold_source": "solver.hard_seed_threshold",
            "anchor_threshold": float(anchor_threshold),
            "formula": (
                "a=1[c>0 and abs(s)>=tau]; c_eff=a; "
                "p=(1-a)p_field+a*q"
            ),
            "non_anchor_policy": "bitwise_field_unary_preservation",
            "new_numeric_constant": False,
        }
    return None


def _registered_posterior_consensus_method_contract(
    observation_fusion: str,
) -> dict[str, object] | None:
    """Describe the target-blind two-expert posterior consensus, if selected."""

    if observation_fusion not in _BERNOULLI_POE_FUSIONS:
        return None
    return {
        "policy": "symmetric_normalized_bernoulli_product_of_experts",
        "domain": "same_canonical_primitive_rows",
        "experts": ["prototype_field", "exact_raster_adjoint"],
        "formula": (
            "p=(p_field*p_exact)/(p_field*p_exact+"
            "(1-p_field)*(1-p_exact))"
        ),
        "posterior_temperature_source": "solver.unary_temperature",
        "neutral_expert_probability": 0.5,
        "neutral_policy": "exact_field_unary_preservation",
        "certain_conflict_policy": "neutral_probability_0.5",
        "new_numeric_constant": False,
        "uses_target_rgb_or_mask": False,
        "uses_scene_specific_constants": False,
        "observation_operator_coupling": (
            "independent_legacy_prototype_and_native_exact_adjoint"
            if observation_fusion == "dual_registration_bernoulli_poe"
            else "shared_native_exact_adjoint_negative_ablation"
        ),
        **(
            {
                "prototype_operator_contract": {
                    "mode": "legacy_alpha_depth",
                    "alpha_threshold": (
                        _FROZEN_LEGACY_PROTOTYPE_ALPHA_THRESHOLD
                    ),
                    "deterministic_cpu_accumulation": True,
                    "seed_provenance": DUAL_PROTOTYPE_SEED_PROVENANCE,
                },
                "exact_operator_contract": {
                    "mode": "native_front_to_back_raster_adjoint",
                    "alpha_threshold": 0.0,
                    "seed_provenance": DUAL_SOLVER_SEED_PROVENANCE,
                },
            }
            if observation_fusion == "dual_registration_bernoulli_poe"
            else {}
        ),
    }


def _candidate_method_manifest_contract(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Return the runner-facing method subset checked before any model load."""

    observation_mode = str(
        getattr(
            args,
            "registered_observation_confidence",
            "relative_joint_max",
        )
    )
    observation_fusion = str(
        getattr(args, "registered_observation_fusion", "additive")
    )
    seed_construction = str(
        getattr(args, "registered_seed_construction", "winner_take_all")
    )
    final_readout = str(
        getattr(args, "registered_readout_stage", "connected")
    )
    forward_unary = _registered_forward_unary_contract(args)
    forward_scoring = _registered_forward_scoring_contract(args)
    strong_unary = _registered_strong_unary_method_contract(
        observation_fusion,
        anchor_threshold=float(getattr(args, "hard_seed_threshold", 0.20)),
    )
    posterior_consensus = _registered_posterior_consensus_method_contract(
        observation_fusion
    )
    source_completion_unary = _source_completion_unary_contract(args)
    return {
        "support_mode": str(args.support_mode),
        "region_space": str(args.region_space),
        "prompt_registration": {
            "mode": str(
                getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
            ),
            "scale": float(
                getattr(args, "prompt_registration_scale", 1.0)
            ),
            "alpha_threshold": float(args.alpha_threshold),
            "depth_tolerance": float(args.depth_tolerance),
            "relative_depth_tolerance": float(
                args.relative_depth_tolerance
            ),
        },
        "seed_construction": seed_construction,
        "seed_normalization": (
            "none"
            if seed_construction == "joint_signed"
            else "independent_max"
        ),
        "observation_fusion": observation_fusion,
        **(
            {"source_completion_unary": source_completion_unary}
            if source_completion_unary is not None
            else {}
        ),
        "registered_seed_unary_weight": float(
            getattr(args, "registered_seed_unary_weight", 0.0)
        ),
        **({"strong_unary": strong_unary} if strong_unary is not None else {}),
        **(
            {"posterior_consensus": posterior_consensus}
            if posterior_consensus is not None
            else {}
        ),
        "observation_mass_source": (
            "exact_shared_raster_responsibility_foreground_background_over_visible_mass"
            if observation_fusion in _EXACT_RASTER_OBSERVATION_FUSIONS
            else (
                "raw_raster_adjoint_prompt_mass_times_labeled_footprint_coverage"
                if observation_mode == "poisson_mass_coverage"
                else (
                    "raw_raster_adjoint_prompt_mass"
                    if observation_mode == "poisson_mass"
                    else "conditional_labeled_footprint_fraction"
                )
            )
        ),
        "observation_confidence": (
            "exact_labeled_visible_fraction"
            if observation_fusion in _EXACT_RASTER_OBSERVATION_FUSIONS
            else observation_mode
        ),
        "observation_mass_scale": float(
            getattr(args, "registered_observation_mass_scale", 1.0)
        ),
        **(
            {
                "observation_coverage_power": float(
                    getattr(
                        args,
                        "registered_observation_coverage_power",
                        1.0,
                    )
                )
            }
            if observation_mode == "poisson_mass_coverage"
            else {}
        ),
        "observation_constructed_before_capability_filter": (
            str(args.support_mode)
            in {"prompt_gaussian", "canonical_support"}
        ),
        "prompt_support_threshold": float(args.support_threshold),
        "prototype_count": int(args.prototype_count),
        "prototype_strategy": str(args.prototype_strategy),
        "appearance_weight": float(args.appearance_weight),
        "boundary_weight": float(args.boundary_weight),
        "prototype_temperature": float(args.prototype_temperature),
        "feature_calibration": str(args.feature_calibration),
        "background_centroids": int(args.background_centroids),
        "score_calibration": str(args.score_calibration),
        "negative_spatial_mode": str(
            getattr(args, "negative_spatial_mode", "none")
        ),
        "diagnostic_selection_mode": str(
            getattr(
                args,
                "registered_selection_mode",
                SelectionMode.SEEDED_COMPONENT.value,
            )
        ),
        "selection_applied_to_main_output": final_readout == "connected",
        "final_readout": final_readout,
        **(
            {"registered_forward_unary": forward_unary}
            if forward_unary is not None
            else {}
        ),
        "graph": {
            "policy": str(args.graph_policy),
            "component_policy": str(args.component_graph_policy),
            "legacy_residual": float(args.graph_legacy_residual),
            "channel_confidence_mode": str(
                getattr(args, "channel_confidence_mode", "none")
            ),
        },
        "score_render": {
            "resolution": str(
                getattr(args, "score_render_resolution", "scaled_renderer")
            ),
            "scale": float(getattr(args, "score_render_scale", 1.0)),
            "valid_support_normalization": bool(
                getattr(args, "valid_support_normalization", False)
            ),
            "valid_support_coverage_power": float(
                getattr(args, "valid_support_coverage_power", 0.0)
            ),
            "feature_contribution_gamma": float(
                args.feature_contribution_gamma
            ),
            "score_chunk_size": int(args.score_chunk_size),
            "pixel_threshold": (
                float(forward_scoring["threshold"]["value"])
                if forward_scoring is not None
                else float(args.solver_support_threshold)
            ),
            "threshold_comparison": "greater_or_equal",
            "resize_to_ground_truth": (
                "cv2.INTER_NEAREST"
                if forward_scoring is not None
                else "cv2.INTER_LINEAR"
            ),
        },
        "solver": {
            "type": str(getattr(args, "solver_type", "diffusion")),
            "iterations": int(args.solver_iterations),
            "residual": float(args.solver_residual),
            "unary_temperature": float(args.solver_unary_temperature),
            "support_threshold": float(args.solver_support_threshold),
            "laplacian_weight": float(
                getattr(args, "laplacian_weight", 1.0)
            ),
            "cg_iterations": int(getattr(args, "cg_iterations", 64)),
            "cg_tolerance": float(getattr(args, "cg_tolerance", 1e-5)),
            "hard_seed_threshold": float(
                getattr(args, "hard_seed_threshold", 0.20)
            ),
            "hard_seed_conflict_policy": str(
                getattr(
                    args,
                    "hard_seed_conflict_policy",
                    "positive_priority",
                )
            ),
            "hard_seed_conflict_margin": float(
                getattr(args, "hard_seed_conflict_margin", 0.0)
            ),
            "component_edge_threshold": float(
                getattr(args, "component_edge_threshold", 1e-5)
            ),
            "seeded_component_min_weight": float(
                getattr(args, "seeded_component_min_weight", 0.20)
            ),
        },
        "canonical_reliability_cache": (
            "per_scene_source_artifact:canonical_primitive_reliability_v1.pt"
            if (
                str(getattr(args, "registered_forward_unary", "none"))
                == "beta_balanced_residual_v2"
            )
            else str(getattr(args, "canonical_reliability_cache", "")).strip()
        ),
        "diagnostic_graph_affinity_override": str(
            getattr(args, "diagnostic_graph_affinity_override", "")
        ).strip(),
        "asset_hash_verification_required": bool(
            getattr(args, "require_asset_hashes", False)
        ),
        "uses_target_calibration": False,
    }


def _validate_candidate_run_manifest(
    args: argparse.Namespace,
    *,
    scene_id: str,
    benchmark_manifest_path: Path,
) -> tuple[dict[str, object] | None, str]:
    """Fail closed on a candidate manifest before loading any scene model."""

    raw_path = str(getattr(args, "run_manifest", "")).strip()
    if not raw_path:
        return None, ""
    path = Path(raw_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_sha256 = _file_sha256(path)
    expected_candidate = str(
        getattr(args, "candidate_id", "registered-region-v1")
    )
    if (
        payload.get("candidate") != expected_candidate
        or scene_id not in payload.get("scenes", [])
        or Path(str(payload.get("benchmark_manifest", ""))).resolve()
        != benchmark_manifest_path
        or payload.get("benchmark_manifest_sha256")
        != _file_sha256(benchmark_manifest_path)
        or Path(str(payload.get("radio_checkpoint", ""))).resolve()
        != Path(args.radio_checkpoint).expanduser().resolve()
        or payload.get("radio_checkpoint_sha256")
        != _file_sha256(Path(args.radio_checkpoint).expanduser().resolve())
    ):
        raise ValueError("candidate run manifest benchmark/RADIO contract mismatch")
    queue_plan = Path(str(payload.get("queue_plan", ""))).resolve()
    if (
        not queue_plan.is_file()
        or payload.get("queue_plan_sha256") != _file_sha256(queue_plan)
    ):
        raise ValueError("candidate run manifest queue-plan mismatch")
    if payload.get("method_contract") != _candidate_method_manifest_contract(args):
        raise ValueError("candidate run manifest method contract mismatch")

    implementation_root = Path(__file__).resolve().parents[2]
    implementation = payload.get("implementation_sources")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("candidate run manifest lacks implementation sources")
    for relative, expected in implementation.items():
        source = implementation_root / str(relative)
        if not source.is_file() or _file_sha256(source) != str(expected):
            raise ValueError(
                f"candidate implementation source mismatch: {relative}"
            )
    if _registered_forward_scoring_contract(args) is not None:
        required_beta_sources = [
            "radio_gs/scripts/eval_nvos_gaussian_first.py",
            "radio_gs/querying/evidence_scorer.py",
            "radio_gs/rendering/contribution_compositor.py",
            "radio_gs/scripts/bind_nvos_forward_beta_protocol_authority.py",
            "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
            "radio_gs/scripts/validate_evaluation_protocol_freeze.py",
        ]
        if (
            str(getattr(args, "registered_forward_unary", "none"))
            == "beta_balanced_residual_v2"
        ):
            required_beta_sources.extend(
                [
                    "radio_gs/interfaces/capability_cache.py",
                    "radio_gs/field/primitive_reliability.py",
                    "radio_gs/scripts/build_canonical_reliability_cache.py",
                ]
            )
        for relative in required_beta_sources:
            source = implementation_root / relative
            if implementation.get(relative) != _file_sha256(source):
                raise ValueError(
                    "beta candidate manifest lacks current implementation "
                    f"authority: {relative}"
                )
    runner = Path(str(payload.get("runner", ""))).resolve()
    if (
        not runner.is_file()
        or payload.get("runner_sha256") != _file_sha256(runner)
    ):
        raise ValueError("candidate runner source mismatch")

    source_records = payload.get("source_artifacts", {}).get(scene_id)
    if not isinstance(source_records, dict):
        raise ValueError(f"{scene_id}: candidate source artifacts are absent")
    expected_paths = {
        "canonical_d256_l128_capability_first.pth": None,
        "official_dino_sam3_views.pt": str(
            Path(args.canonical_capability_cache).resolve()
        ),
        "shared_support_graph_k16.pt": str(
            Path(args.canonical_support_graph).resolve()
        ),
    }
    if (
        str(getattr(args, "registered_forward_unary", "none"))
        == "beta_balanced_residual_v2"
    ):
        expected_paths["canonical_primitive_reliability_v1.pt"] = str(
            Path(args.canonical_reliability_cache).resolve()
        )
    for name, expected_path in expected_paths.items():
        record = source_records.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"{scene_id}: missing candidate source {name}")
        artifact = Path(str(record.get("path", ""))).resolve()
        if expected_path is not None and str(artifact) != expected_path:
            raise ValueError(f"{scene_id}: candidate source path mismatch for {name}")
        if (
            not artifact.is_file()
            or int(record.get("bytes", -1)) != artifact.stat().st_size
            or record.get("sha256") != _file_sha256(artifact)
        ):
            raise ValueError(f"{scene_id}: candidate source SHA mismatch for {name}")
        metadata = Path(str(record.get("metadata_path", ""))).resolve()
        if (
            not metadata.is_file()
            or record.get("metadata_sha256") != _file_sha256(metadata)
        ):
            raise ValueError(
                f"{scene_id}: candidate source metadata mismatch for {name}"
            )
        if (
            name == "canonical_d256_l128_capability_first.pth"
            and str(args.canonical_field_sha256).strip()
            != str(record.get("sha256"))
        ):
            raise ValueError(f"{scene_id}: canonical field digest mismatch")

    queue_inputs = payload.get("queue_scene_inputs", {}).get(scene_id)
    if not isinstance(queue_inputs, dict) or not queue_inputs:
        raise ValueError(f"{scene_id}: candidate renderer/view inputs are absent")
    for raw_input, record in queue_inputs.items():
        asset = Path(str(raw_input)).resolve()
        if (
            not isinstance(record, dict)
            or not asset.is_file()
            or int(record.get("bytes", -1)) != asset.stat().st_size
            or record.get("sha256") != _file_sha256(asset)
        ):
            raise ValueError(f"{scene_id}: renderer/view input mismatch: {asset}")
    return payload, manifest_sha256


@torch.inference_mode()
def decode_region_rows(model, codec, adaptor, *, device: torch.device, chunk_size: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    count = int(model.get_xyz().shape[0])
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        indices = torch.arange(start, stop, device=device, dtype=torch.long)
        compact = model.query_gaussian_points(indices)
        radio = codec.decode_points(compact.float())
        region = adaptor(radio.float()).float() if adaptor is not None else radio.float()
        rows.append(F.normalize(region, dim=-1).half().cpu())
    return torch.cat(rows, dim=0)


def _scene_record(manifest: dict, scene_id: str) -> dict:
    matches = [scene for scene in manifest["scenes"] if scene["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest scene {scene_id!r}")
    return matches[0]


def _view_by_frame(views: list[dict], frame_id: str) -> dict:
    matches = [view for view in views if str(view["frame_id"]) == str(frame_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one protocol view for {frame_id!r}")
    return matches[0]


def _weighted_spherical_prototypes(
    rows: torch.Tensor,
    weights: torch.Tensor,
    count: int,
    *,
    iterations: int = 6,
) -> torch.Tensor:
    """Build deterministic appearance prototypes without target-set fitting.

    Prompt support can cover several object parts whose appearances should not
    be collapsed into one mean.  Weighted farthest-first initialization keeps
    tiny raster tails from becoming prototypes, followed by a small fixed
    spherical k-means refinement.  ``count=1`` exactly reduces to the previous
    weighted mean readout.
    """
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[0] != weights.shape[0]:
        raise ValueError("rows and weights must have shapes [N,D] and [N]")
    if rows.shape[0] == 0:
        raise ValueError("Cannot build prototypes from empty prompt support")
    count = min(max(1, int(count)), int(rows.shape[0]))
    weights = weights.float().clamp_min(0)
    rows = F.normalize(rows.float(), dim=-1)
    if count == 1:
        center = (rows * weights[:, None]).sum(dim=0)
        return F.normalize(center, dim=0)[None]

    selected = [int(weights.argmax())]
    min_distance = 1.0 - rows @ rows[selected[0]]
    weight_scale = weights / weights.max().clamp_min(1e-8)
    for _ in range(1, count):
        utility = min_distance.clamp_min(0) * weight_scale.sqrt()
        utility[selected] = -1
        index = int(utility.argmax())
        selected.append(index)
        min_distance = torch.minimum(min_distance, 1.0 - rows @ rows[index])
    centers = rows[selected]

    for _ in range(max(0, int(iterations))):
        assignment = (rows @ centers.T).argmax(dim=1)
        updated = []
        for index in range(count):
            member = assignment == index
            if bool(member.any()):
                center = (rows[member] * weights[member, None]).sum(dim=0)
                updated.append(F.normalize(center, dim=0))
            else:
                updated.append(centers[index])
        centers = torch.stack(updated, dim=0)
    return centers


def _load_training_poses(
    queue_scene: Path,
    evaluation_camera_names: list[str],
) -> list[torch.Tensor]:
    mapping = json.loads(
        (queue_scene / "feature_pose_mapping.json").read_text(encoding="utf-8")
    )
    train_ids = {
        int(value)
        for value in json.loads(
            (queue_scene / "train_frame_ids.json").read_text(encoding="utf-8")
        )["frame_ids"]
    }
    records = [
        record
        for record in mapping["records"]
        if int(record["feature_frame_id"]) in train_ids
    ]
    evaluation_set = {str(value) for value in evaluation_camera_names}
    # A dataset may permit target RGBs during field construction (SPIn-NeRF),
    # but query-time support remains target-view independent.  Filter those
    # cameras rather than rendering query evidence from an evaluation pose.
    records = [
        record for record in records if str(record["camera_name"]) not in evaluation_set
    ]
    poses = []
    for record in sorted(records, key=lambda value: int(value["feature_frame_id"])):
        c2w = np.loadtxt(record["pose_path"], dtype=np.float32).reshape(4, 4)
        poses.append(torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)))
    if not poses:
        raise ValueError("No protocol-permitted training support poses")
    return poses


def _resolve_observed_feature_path(queue_scene: Path, camera_name: str) -> Path:
    """Resolve a protocol camera to its saved, observed RADIO feature map."""
    mapping = json.loads(
        (queue_scene / "feature_pose_mapping.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in mapping["records"]
        if str(record.get("camera_name")) == str(camera_name)
        or str(record.get("colmap_camera_name")) == str(camera_name)
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one feature mapping for camera {camera_name!r}")
    feature_id = int(records[0]["feature_frame_id"])
    manifest = json.loads(
        (queue_scene / "radio_features" / "frame_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frames = [
        frame
        for frame in manifest["frames"]
        if int(frame.get("source_rank", -1)) == feature_id
        or int(frame.get("frame_idx", -1)) == feature_id
    ]
    if len(frames) != 1:
        raise ValueError(f"Expected one saved feature frame for id {feature_id}")
    path = queue_scene / "radio_features" / "backbone" / f"{frames[0]['saved_stem']}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.inference_mode()
def _observed_region_map(
    queue_scene: Path,
    camera_name: str,
    adaptor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Encode the registered real query view without rendering it through 3-D."""
    path = _resolve_observed_feature_path(queue_scene, camera_name)
    radio = torch.load(path, map_location="cpu").float()
    if radio.ndim != 3:
        raise ValueError(f"Expected observed RADIO feature [C,H,W], got {tuple(radio.shape)}")
    channels, height, width = radio.shape
    rows = radio.permute(1, 2, 0).reshape(-1, channels).to(device)
    if adaptor is not None:
        rows = adaptor(rows).float()
    rows = F.normalize(rows.float(), dim=-1)
    return rows.reshape(height, width, -1).permute(2, 0, 1)[None]


@torch.inference_mode()
def _screen_region_map(
    model,
    codec,
    renderer,
    sharpener,
    refiner,
    config,
    adaptor,
    pose: torch.Tensor,
    *,
    is_hybrid: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoded, aux = render_1280d(
        model,
        codec,
        renderer,
        sharpener,
        refiner,
        pose[None],
        is_hybrid=is_hybrid,
        config=config,
        device=pose.device,
        return_aux=True,
    )
    channels, height, width = decoded.shape[1:]
    rows = decoded.permute(0, 2, 3, 1).reshape(-1, channels).float()
    if adaptor is not None:
        rows = adaptor(rows).float()
    rows = F.normalize(rows, dim=-1)
    return rows.reshape(1, height, width, -1).permute(0, 3, 1, 2), aux


def run(args: argparse.Namespace) -> dict:
    _validate_registered_forward_unary_args(args)
    _validate_source_completion_unary_args(args)
    registered_forward_contract = _registered_forward_unary_contract(args)
    device = torch.device(args.device)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_run_manifest, candidate_run_manifest_sha256 = (
        _validate_candidate_run_manifest(
            args,
            scene_id=str(args.scene_id),
            benchmark_manifest_path=manifest_path,
        )
    )
    candidate_eligibility = (
        str(candidate_run_manifest.get("eligibility", "")).strip()
        if candidate_run_manifest is not None
        else "unregistered"
    )
    candidate_method_contract_sha256 = (
        _json_sha256(candidate_run_manifest["method_contract"])
        if candidate_run_manifest is not None
        else ""
    )
    registered_forward_protocol_authority = (
        _load_registered_forward_protocol_authority(
            args,
            candidate_run_manifest,
            candidate_method_contract_sha256,
        )
    )
    registered_forward_protocol_authority_sha256 = (
        _json_sha256(registered_forward_protocol_authority)
        if registered_forward_protocol_authority is not None
        else ""
    )
    scene = _scene_record(manifest, args.scene_id)
    source_observation_oof_deployment_gate = (
        _load_source_observation_oof_deployment_gate(
            args,
            scene_id=str(args.scene_id),
            protocol_hash=str(manifest["protocol_hash"]),
        )
    )
    base_scene_id = str(scene.get("base_scene_id") or args.scene_id)
    scene_root = Path(args.queue_root).resolve() / "scenes"
    queue_scene = scene_root / args.scene_id
    if not queue_scene.is_dir():
        queue_scene = scene_root / base_scene_id
    config_path, checkpoint_path, camera_map_path = _resolve_scene_carrier_assets(
        queue_scene,
        scene_config=str(getattr(args, "scene_config", "")),
        scene_checkpoint=str(getattr(args, "scene_checkpoint", "")),
        camera_map=str(getattr(args, "camera_map", "")),
    )
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    prompt_frame = str(scene["prompt_frame_ids"][0])
    prompt_view = _view_by_frame(views, prompt_frame)
    evaluation_frames = [str(value) for value in scene["evaluation_frame_ids"]]
    evaluation_views = [_view_by_frame(views, frame_id) for frame_id in evaluation_frames]
    # Protocol frame ids (for example ``image001``) need not equal their RGB /
    # COLMAP camera names (for example ``IMG_4027``).  Query-time support is
    # keyed by the latter, so exclusions must use the resolved frozen mapping.
    evaluation_camera_names = sorted(
        {
            str(view[key])
            for view in evaluation_views
            for key in ("camera_name", "colmap_camera_name")
            if view.get(key) is not None
        }
    )

    model, codec, renderer, sharpener, refiner, field_config, is_hybrid = load_render_pipeline(
        str(config_path),
        str(checkpoint_path),
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    # ``canonical_support`` gets every query feature from the independently
    # bound capability bank.  Its renderer checkpoint is only a geometry and
    # raster-responsibility carrier, so requiring an HCD/hybrid codec here
    # would incorrectly exclude frozen RGB 3DGS geometries.  The other support
    # modes still decode field features and therefore retain the fail-closed
    # hybrid-field requirement.
    if not is_hybrid and args.support_mode != "canonical_support":
        raise ValueError(
            "non-canonical Gaussian-first NVOS requires a hybrid field"
        )
    adaptor = None
    if args.region_space == "sam3" and args.support_mode != "canonical_support":
        adaptor = load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "sam3", kind="feature_projection"
        ).to(device).eval().requires_grad_(False)
    region_rows = None
    capability_bank = None
    support_graph = None
    primitive_reliability = None
    source_observation_footprint_bundle: (
        tuple[Path, str, SourceFootprintFoldAuthority] | None
    ) = None
    if args.support_mode == "canonical_support":
        graph_disabled = bool(
            getattr(args, "disable_registered_graph", False)
        )
        if not args.canonical_capability_cache or (
            not graph_disabled and not args.canonical_support_graph
        ):
            raise ValueError(
                "canonical_support requires --canonical-capability-cache and, "
                "unless graph-disabled, --canonical-support-graph"
            )
        source_contract = str(
            getattr(args, "canonical_capability_source_contract", "field")
        )
        expected_capability_source = {
            "field": _CANONICAL_FIELD_CAPABILITY_SOURCE,
            "exact_mpr": _EXACT_MPR_CAPABILITY_SOURCE,
            "exact_capability_mpr": _EXACT_CAPABILITY_MPR_SOURCE,
        }[source_contract]
        capability_bank = load_canonical_capability_bank(
            args.canonical_capability_cache,
            expected_field_checkpoint_sha256=args.canonical_field_sha256,
            expected_source=expected_capability_source,
            require_formal_projection_order=source_contract
            in {"field", "exact_capability_mpr"},
            allow_raw_mpr_projection_diagnostic=source_contract == "exact_mpr",
            legacy_projection_authority=str(
                getattr(args, "canonical_capability_projection_authority", "")
            ),
        )
        support_graph = (
            _disabled_registered_graph(int(capability_bank.valid.sum()))
            if graph_disabled
            else load_canonical_support_graph(
                args.canonical_support_graph, capability_bank
            )
        )
        if str(args.canonical_reliability_cache).strip():
            primitive_reliability = load_canonical_primitive_reliability(
                args.canonical_reliability_cache,
                expected_xyz=capability_bank.xyz,
                expected_valid=capability_bank.valid,
                expected_field_checkpoint_sha256=str(
                    capability_bank.metadata.get("field_checkpoint_sha256", "")
                ),
            )
        if str(args.diagnostic_graph_affinity_override).strip():
            override_path = Path(args.diagnostic_graph_affinity_override)
            override = torch.load(override_path, map_location="cpu")
            global_rows = torch.as_tensor(override.get("global_rows")).long().cpu()
            if not torch.equal(global_rows, capability_bank.global_rows):
                raise ValueError("diagnostic graph override nodes do not match capability rows")
            if int(override.get("num_global_rows", -1)) != capability_bank.num_gaussians:
                raise ValueError("diagnostic graph override global row count differs")
            base_edges = support_graph.edge_index.cpu()
            override_edges = torch.as_tensor(override.get("edge_index")).long().cpu()
            if not torch.equal(base_edges, override_edges):
                raise ValueError(
                    "diagnostic graph override must preserve the exact geometry topology"
                )
            support_graph = PrimitiveSupportGraph(
                edge_index=override_edges,
                edge_weight=torch.as_tensor(override["edge_weight"]).float(),
                raw_affinity=torch.as_tensor(override["raw_affinity"]).float(),
                local_sigma=torch.as_tensor(override["local_sigma"]).float(),
                num_nodes=int(global_rows.numel()),
                edge_channels={
                    str(name): torch.as_tensor(values).float()
                    for name, values in dict(override.get("edge_channels", {})).items()
                },
            )
        geometry_xyz = model.get_xyz().detach().float().cpu()
        if geometry_xyz.shape != capability_bank.xyz.shape or not torch.allclose(
            geometry_xyz, capability_bank.xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("canonical capability geometry does not match renderer geometry")
        source_observation_footprint_bundle = (
            _load_source_observation_footprint_authority(
                args,
                capability_bank,
            )
        )
    if args.support_mode == "prompt_gaussian":
        region_rows = decode_region_rows(
            model, codec, adaptor, device=device, chunk_size=max(1, args.chunk_size)
        )

    prompt = scene["prompt"]
    prompt_type = str(prompt.get("type", ""))
    declared_prompt_hashes = _declared_prompt_asset_hashes(manifest, scene)
    if prompt_type == "positive_negative_scribbles":
        if bool(getattr(args, "require_asset_hashes", False)):
            _verify_declared_sha256(
                Path(prompt["positive_path"]),
                declared_prompt_hashes.get("positive"),
                label=f"{args.scene_id} positive prompt",
            )
            _verify_declared_sha256(
                Path(prompt["negative_path"]),
                declared_prompt_hashes.get("negative"),
                label=f"{args.scene_id} negative prompt",
            )
        positive_native = load_ground_truth_mask(prompt["positive_path"]).astype(bool)
        negative_native = load_ground_truth_mask(prompt["negative_path"]).astype(bool)
        if positive_native.shape != negative_native.shape:
            raise ValueError("positive and negative prompt rasters must align")
    elif prompt_type == "reference_binary_mask":
        if bool(getattr(args, "require_asset_hashes", False)):
            _verify_declared_sha256(
                Path(prompt["mask_path"]),
                declared_prompt_hashes.get(
                    "mask",
                    declared_prompt_hashes.get(
                        "positive",
                        declared_prompt_hashes.get("reference_binary_mask"),
                    ),
                ),
                label=f"{args.scene_id} reference mask",
            )
        positive_native = load_ground_truth_mask(prompt["mask_path"]).astype(bool)
        negative_native = np.logical_not(positive_native)
    else:
        raise ValueError(f"Unsupported registered prompt type: {prompt_type!r}")
    _validate_source_observation_oof_contract(
        args,
        prompt_type=prompt_type,
    )
    source_completion_probability: torch.Tensor | None = None
    source_completion_reliability: torch.Tensor | None = None
    source_completion_evidence: dict[str, object] | None = None
    source_completion_abstained = False
    if _source_completion_unary_contract(args) is not None:
        if prompt_type != "positive_negative_scribbles":
            raise ValueError(
                "probability-preserving source completion requires signed scribbles"
            )
        (
            source_completion_probability,
            source_completion_reliability,
            source_completion_evidence,
        ) = _load_probability_preserving_source_completion(
            args,
            scene_id=str(args.scene_id),
            frame_id=prompt_frame,
            positive_path=prompt["positive_path"],
            negative_path=prompt["negative_path"],
            raw_positive=positive_native,
            raw_negative=negative_native,
        )
        source_completion_calibration = str(
            getattr(args, "source_completion_calibration", "none")
        )
        if source_completion_calibration in {
            _SOURCE_COMPLETION_LOO_CALIBRATION,
            _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION,
        }:
            source_completion_gate = load_source_completion_loo_gate(
                args.source_completion_calibration_gate,
                expected_gate_sha256=(
                    args.source_completion_calibration_gate_sha256
                ),
                completion_path=args.source_completion,
                expected_completion_sha256=args.source_completion_sha256,
                expected_completion_receipt_sha256=(
                    args.source_completion_receipt_sha256
                ),
                expected_scene_id=str(args.scene_id),
                expected_frame_id=prompt_frame,
            )
            source_completion_evidence["source_only_calibration_gate"] = {
                **source_completion_gate,
                "path": str(
                    Path(args.source_completion_calibration_gate)
                    .expanduser()
                    .resolve()
                ),
                "sha256": str(
                    args.source_completion_calibration_gate_sha256
                ),
            }
            accept_source_completion = bool(
                source_completion_gate["decision"]["accept_source_completion"]
            )
            if (
                source_completion_calibration
                == _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION
            ):
                (
                    source_completion_probability,
                    source_completion_reliability,
                    hierarchical_trust,
                ) = _apply_hierarchical_source_completion_trust(
                    source_completion_probability,
                    source_completion_reliability,
                    accept_full_completion=accept_source_completion,
                    raw_positive=positive_native,
                    raw_negative=negative_native,
                )
                source_completion_evidence["hierarchical_trust"] = (
                    hierarchical_trust
                )
            elif not accept_source_completion:
                source_completion_abstained = True
                source_completion_reliability.zero_()
    if bool(
        getattr(args, "export_registered_prompt_cycle_diagnostic", False)
    ) and prompt_type != "reference_binary_mask":
        raise ValueError(
            "registered prompt-cycle diagnostic currently requires a full "
            "reference binary mask"
        )
    if (
        str(getattr(args, "registered_observation_fusion", "additive"))
        in _EXACT_RASTER_OBSERVATION_FUSIONS
        and bool(np.logical_and(positive_native, negative_native).any())
    ):
        raise ValueError(
            "exact raster observation fusion requires disjoint "
            "foreground/background "
            "prompt rasters"
        )
    native_height, native_width = map(int, positive_native.shape)
    if (
        getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
        == "raster_adjoint"
    ):
        height, width = _scaled_raster_shape(
            native_height,
            native_width,
            float(getattr(args, "prompt_registration_scale", 1.0)),
        )
    else:
        height, width = int(renderer.image_height), int(renderer.image_width)
    positive = resize_mask_nearest(
        positive_native, (height, width)
    ).astype(bool)
    negative = resize_mask_nearest(
        negative_native, (height, width)
    ).astype(bool)
    prompt_maps = torch.from_numpy(
        np.stack([positive, negative], axis=0).astype(np.float32)
    )[None].to(device)
    prompt_pose = torch.from_numpy(prompt_view["w2c"].copy()).float().to(device)
    camera_clearance_sigma = float(getattr(args, "camera_clearance_sigma", 0.0))

    def row_confidence_for_pose(pose: torch.Tensor) -> torch.Tensor | None:
        if camera_clearance_sigma <= 0:
            return None
        return camera_plane_clearance_confidence(
            model.get_xyz(),
            model.get_rotation(),
            model.get_scaling(),
            pose,
            near_plane=float(renderer.near_plane),
            support_sigma=camera_clearance_sigma,
        ).confidence

    prompt_row_confidence = row_confidence_for_pose(prompt_pose)
    support_view_count = 1
    prediction_threshold = 0.0
    canonical_stage_gaussian_scores: dict[str, torch.Tensor] | None = None
    registered_prompt_cycle_gaussian_scores: dict[str, torch.Tensor] | None = None
    registered_prompt_evidence: dict[str, object] | None = None
    source_observation_oof_authority: SourceObservationOOFFold | None = None
    source_observation_evidence_authority: (
        SourceObservationEvidenceAuthority | None
    ) = None
    source_observation_oof_contract: dict[str, object] | None = None
    source_observation_population_positive_weight: torch.Tensor | None = None
    source_observation_population_negative_weight: torch.Tensor | None = None
    source_completion_observation: PrimitiveUnaryEvidence | None = None
    configured_observation_fusion = str(
        getattr(args, "registered_observation_fusion", "additive")
    )
    observation_fusion = configured_observation_fusion
    if args.support_mode in {"prompt_gaussian", "canonical_support"}:
        if source_completion_abstained:
            observation_fusion = "hard_seed_anchor_only_probability"
        legacy_prototype_support: tuple[torch.Tensor, torch.Tensor] | None = None
        if (
            getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
            == "raster_adjoint"
        ):
            prompt_alpha = None
            if args.alpha_threshold > 0:
                prompt_alpha = renderer.render_feature_rows(
                    model,
                    prompt_pose,
                    torch.ones(
                        model.get_xyz().shape[0],
                        1,
                        device=device,
                        dtype=torch.float32,
                    ),
                    feature_height=height,
                    feature_width=width,
                    row_confidence=prompt_row_confidence,
                )["alpha_map"]
            registration_maps = prompt_maps
            if (
                source_completion_probability is not None
                and not source_completion_abstained
            ):
                assert source_completion_reliability is not None
                registration_maps = _probability_preserving_registration_maps(
                    prompt_maps,
                    source_completion_probability,
                    source_completion_reliability,
                )
            registered_sum, support_count = raster_adjoint_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=prompt_pose,
                siglip_feat=registration_maps,
                alpha_map=prompt_alpha,
                alpha_threshold=args.alpha_threshold,
                row_confidence=prompt_row_confidence,
            )
            support_sum = registered_sum[:, :2]
            if source_completion_probability is not None:
                completion_sum = (
                    registered_sum[:, 2:]
                    if not source_completion_abstained
                    else torch.zeros(
                        (registered_sum.shape[0], 2),
                        device=registered_sum.device,
                        dtype=registered_sum.dtype,
                    )
                )
                source_completion_observation = registered_raster_adjoint_observation(
                    completion_sum[:, 0].detach().float().clamp_min(0.0),
                    completion_sum[:, 1].detach().float().clamp_min(0.0),
                    support_count.detach().float().clamp_min(0.0),
                )
            if _requires_legacy_prototype_observation(observation_fusion):
                # The two experts deliberately use independent observation
                # operators.  Reconstruct the frozen historical prototype
                # support at the renderer grid with alpha/depth registration;
                # native exact-adjoint masses above are reserved for the
                # registered Bernoulli expert and solver seeds.
                legacy_height = int(renderer.image_height)
                legacy_width = int(renderer.image_width)
                legacy_positive = resize_mask_nearest(
                    positive_native,
                    (legacy_height, legacy_width),
                ).astype(bool)
                legacy_negative = resize_mask_nearest(
                    negative_native,
                    (legacy_height, legacy_width),
                ).astype(bool)
                legacy_prompt_maps = torch.from_numpy(
                    np.stack(
                        [legacy_positive, legacy_negative], axis=0
                    ).astype(np.float32)
                )[None].to(device)
                legacy_prototype_support = _rasterize_frozen_legacy_prototype_support(
                    model=model,
                    renderer=renderer,
                    viewmat=prompt_pose,
                    prompt_maps=legacy_prompt_maps,
                    depth_tolerance=args.depth_tolerance,
                    relative_depth_tolerance=args.relative_depth_tolerance,
                )
        else:
            prompt_aux = renderer.render_features(model, prompt_pose)
            support_sum, support_count = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=prompt_pose,
                siglip_feat=prompt_maps,
                depth_map=prompt_aux["depth_map"][None],
                alpha_map=prompt_aux["alpha_map"][None],
                registration_depth_tolerance=args.depth_tolerance,
                registration_relative_depth_tolerance=args.relative_depth_tolerance,
                registration_alpha_threshold=args.alpha_threshold,
                registration_weight_mode="alpha_depth",
                deterministic_cpu_accumulation=(
                    args.support_mode == "canonical_support"
                ),
            )
        support_fraction = support_sum / support_count.clamp_min(1e-8).unsqueeze(1)
        positive_weight = support_fraction[:, 0]
        negative_weight = support_fraction[:, 1]
        raw_positive_mass = support_sum[:, 0].detach().float().clamp_min(0.0)
        raw_negative_mass = support_sum[:, 1].detach().float().clamp_min(0.0)
        if str(
            getattr(args, "source_observation_oof_output_dir", "")
        ).strip():
            assert capability_bank is not None
            oof_root = Path(
                args.source_observation_oof_output_dir
            ).expanduser().resolve()
            if source_observation_footprint_bundle is None:
                source_observation_oof_contract = (
                    _source_observation_oof_method_contract(
                        args,
                        prompt_type=prompt_type,
                    )
                )
            else:
                (
                    footprint_path,
                    footprint_file_sha256,
                    footprint_authority,
                ) = source_observation_footprint_bundle
                source_observation_oof_contract = (
                    _source_observation_footprint_oof_method_contract(
                        args,
                        prompt_type=prompt_type,
                        footprint_path=footprint_path,
                        footprint_file_sha256=footprint_file_sha256,
                        footprint_authority=footprint_authority,
                    )
                )
            evidence_provenance = {
                "scene_id": str(args.scene_id),
                "protocol_hash": str(manifest["protocol_hash"]),
                "method_contract_sha256": _json_sha256(
                    source_observation_oof_contract
                ),
                "capability_cache_sha256": _file_sha256(
                    Path(args.canonical_capability_cache)
                    .expanduser()
                    .resolve()
                ),
                "support_graph_sha256": _file_sha256(
                    Path(args.canonical_support_graph)
                    .expanduser()
                    .resolve()
                ),
            }
            if source_observation_footprint_bundle is not None:
                evidence_provenance.update(
                    {
                        "source_footprint_fold_authority": str(footprint_path),
                        "source_footprint_fold_authority_file_sha256": (
                            footprint_file_sha256
                        ),
                        "source_footprint_fold_authority_sha256": (
                            footprint_authority.authority_sha256
                        ),
                        "source_footprint_fold_authority_tensor_bundle_sha256": (
                            footprint_authority.tensor_bundle_sha256
                        ),
                    }
                )
            source_observation_evidence_authority = (
                seal_or_load_source_observation_evidence_authority(
                    oof_root / "source_observation_evidence_authority.pt",
                    heldout_fold=int(args.source_observation_oof_heldout_fold),
                    provenance=evidence_provenance,
                    valid=capability_bank.valid,
                    global_rows=capability_bank.global_rows,
                    positive_weight=positive_weight,
                    negative_weight=negative_weight,
                    raw_positive_mass=raw_positive_mass,
                    raw_negative_mass=raw_negative_mass,
                )
            )
            sealed_evidence = source_observation_evidence_authority.tensors
            source_observation_population_positive_weight = sealed_evidence[
                "positive_weight"
            ]
            source_observation_population_negative_weight = sealed_evidence[
                "negative_weight"
            ]
            if source_observation_footprint_bundle is None:
                source_observation_oof_authority = prepare_source_observation_oof_fold(
                    sealed_evidence["global_rows"],
                    sealed_evidence["valid"],
                    sealed_evidence["positive_weight"],
                    sealed_evidence["negative_weight"],
                    sealed_evidence["raw_positive_mass"],
                    sealed_evidence["raw_negative_mass"],
                    heldout_fold=int(args.source_observation_oof_heldout_fold),
                    num_folds=3,
                )
                audit_signed_cv_population(
                    capability_bank.global_rows,
                    source_observation_oof_authority.signed_reference_evidence[
                        capability_bank.valid
                    ],
                    source_observation_oof_authority.reference_weight[
                        capability_bank.valid
                    ],
                    num_folds=3,
                    minimum_class_rows=32,
                )
            else:
                source_observation_oof_authority, population_decision = (
                    prepare_source_observation_footprint_oof_fold(
                        footprint_authority,
                        sealed_evidence["global_rows"],
                        sealed_evidence["valid"],
                        sealed_evidence["positive_weight"],
                        sealed_evidence["negative_weight"],
                        sealed_evidence["raw_positive_mass"],
                        sealed_evidence["raw_negative_mass"],
                        heldout_fold=int(args.source_observation_oof_heldout_fold),
                        expected_footprint_authority_sha256=(
                            footprint_authority.authority_sha256
                        ),
                    )
                )
                if source_observation_oof_authority is None:
                    gate_path, gate_receipt = (
                        _write_source_observation_footprint_field_base_receipt(
                            oof_root,
                            scene_id=str(args.scene_id),
                            protocol_hash=str(manifest["protocol_hash"]),
                            capability_cache=args.canonical_capability_cache,
                            support_graph=args.canonical_support_graph,
                            evidence_authority=(
                                source_observation_evidence_authority
                            ),
                            footprint_path=footprint_path,
                            footprint_file_sha256=footprint_file_sha256,
                            footprint_authority=footprint_authority,
                            population_decision=population_decision,
                            method_contract=source_observation_oof_contract,
                        )
                    )
                    return {
                        "schema_version": 1,
                        "status": "source_observation_footprint_field_base_sealed",
                        "scene_id": str(args.scene_id),
                        "protocol_hash": str(manifest["protocol_hash"]),
                        "selected_action": FIELD_BASE_ACTION,
                        "gate_receipt": str(gate_path),
                        "gate_receipt_sha256": _file_sha256(gate_path),
                        "gate": gate_receipt,
                        "target_score_rendering_performed": False,
                        "target_rgb_opened": False,
                        "target_mask_opened": False,
                        "target_metric_computed": False,
                    }
            assert source_observation_oof_authority is not None
            positive_weight = (
                source_observation_oof_authority.training_positive_weight.to(
                    device=positive_weight.device,
                    dtype=positive_weight.dtype,
                )
            )
            negative_weight = (
                source_observation_oof_authority.training_negative_weight.to(
                    device=negative_weight.device,
                    dtype=negative_weight.dtype,
                )
            )
            raw_positive_mass = (
                source_observation_oof_authority.training_raw_positive_mass.to(
                    device=raw_positive_mass.device,
                    dtype=raw_positive_mass.dtype,
                )
            )
            raw_negative_mass = (
                source_observation_oof_authority.training_raw_negative_mass.to(
                    device=raw_negative_mass.device,
                    dtype=raw_negative_mass.dtype,
                )
            )
        raw_joint_mass = raw_positive_mass + raw_negative_mass
        visibility_tolerance = 1e-4 * max(
            1.0,
            float(support_count.detach().float().amax()),
        )
        if bool((raw_joint_mass > support_count + visibility_tolerance).any()):
            raise ValueError(
                "registered positive/negative prompt mass exceeds visible "
                "raster-adjoint mass"
            )
        observation_confidence_mode = str(
            getattr(
                args,
                "registered_observation_confidence",
                "relative_joint_max",
            )
        )
        observation_mass_scale = float(
            getattr(args, "registered_observation_mass_scale", 1.0)
        )
        observation_coverage_power = float(
            getattr(args, "registered_observation_coverage_power", 1.0)
        )
        if observation_fusion in _EXACT_RASTER_OBSERVATION_FUSIONS:
            direct_observation = registered_raster_adjoint_observation(
                raw_positive_mass,
                raw_negative_mass,
                support_count.detach().float().clamp_min(0.0),
            )
            direct_signed = direct_observation.values
            assert direct_observation.confidence is not None
            direct_confidence = direct_observation.confidence
            direct_mass_source = (
                "exact_shared_raster_responsibility_foreground_background_"
                "over_visible_mass"
            )
            observation_confidence_mode = "exact_labeled_visible_fraction"
        elif observation_confidence_mode in {
            "poisson_mass",
            "poisson_mass_coverage",
        }:
            direct_signed, direct_confidence = registered_seed_observation(
                raw_positive_mass,
                raw_negative_mass,
                confidence_mode=observation_confidence_mode,
                mass_scale=observation_mass_scale,
                visible_mass=(
                    support_count.detach().float().clamp_min(0.0)
                    if observation_confidence_mode
                    == "poisson_mass_coverage"
                    else None
                ),
                coverage_power=observation_coverage_power,
            )
            direct_mass_source = (
                "raw_raster_adjoint_prompt_mass_times_"
                "labeled_footprint_coverage"
                if observation_confidence_mode
                == "poisson_mass_coverage"
                else "raw_raster_adjoint_prompt_mass"
            )
        else:
            direct_signed, direct_confidence = registered_seed_observation(
                positive_weight,
                negative_weight,
                confidence_mode=observation_confidence_mode,
                mass_scale=observation_mass_scale,
            )
            direct_mass_source = "conditional_labeled_footprint_fraction"
        if observation_fusion not in _EXACT_RASTER_OBSERVATION_FUSIONS:
            direct_observation = PrimitiveUnaryEvidence(
                direct_signed,
                f"raster_adjoint_{observation_confidence_mode}",
                direct_confidence,
            )
        strong_unary_anchor_threshold = (
            float(getattr(args, "hard_seed_threshold", 0.20))
            if observation_fusion
            in {
                "hard_seed_anchored_probability",
                "hard_seed_anchor_only_probability",
            }
            else None
        )
        strong_unary_anchor_mask = (
            registered_observation_anchor_mask(
                direct_observation,
                anchor_threshold=strong_unary_anchor_threshold,
            )
            if strong_unary_anchor_threshold is not None
            else torch.zeros_like(direct_confidence, dtype=torch.bool)
        )
        strong_unary_effective_confidence = (
            (
                registered_observation_anchor_only_confidence(
                    direct_observation,
                    anchor_threshold=strong_unary_anchor_threshold,
                )
                if observation_fusion
                == "hard_seed_anchor_only_probability"
                else registered_observation_effective_confidence(
                    direct_observation,
                    anchor_threshold=strong_unary_anchor_threshold,
                )
            )
            if strong_unary_anchor_threshold is not None
            else direct_confidence
        )
        seed_construction = str(
            getattr(args, "registered_seed_construction", "winner_take_all")
        )
        if seed_construction == "joint_signed":
            positive_solver_mass, negative_solver_mass = (
                _joint_signed_observation_seeds(
                    direct_signed,
                    direct_confidence,
                    support_threshold=float(args.support_threshold),
                )
            )
            seed_normalization = "none"
        else:
            positive_solver_mass, negative_solver_mass = _registered_solver_masses(
                positive_weight,
                negative_weight,
                support_threshold=float(args.support_threshold),
                construction=seed_construction,
            )
            seed_normalization = "independent_max"
        prototype_seed_construction = str(
            getattr(
                args,
                "registered_prototype_seed_construction",
                "shared",
            )
        )
        prototype_seed_normalization = seed_normalization
        prototype_seed_provenance: str | None = None
        solver_seed_provenance: str | None = None
        prototype_positive_solver_mass = positive_solver_mass
        prototype_negative_solver_mass = negative_solver_mass
        if prototype_seed_construction == "winner_take_all":
            (
                prototype_positive_solver_mass,
                prototype_negative_solver_mass,
            ) = _registered_solver_masses(
                positive_weight,
                negative_weight,
                support_threshold=float(args.support_threshold),
                construction="winner_take_all",
            )
            prototype_seed_normalization = "independent_max"
            prototype_seed_provenance = (
                _EXACT_WINNER_TAKE_ALL_PROTOTYPE_SEED_PROVENANCE
            )
            solver_seed_provenance = (
                _EXACT_JOINT_SIGNED_SOLVER_SEED_PROVENANCE
            )
        if _requires_legacy_prototype_observation(observation_fusion):
            if legacy_prototype_support is None:
                raise RuntimeError(
                    "dual_registration_bernoulli_poe lacks its independent "
                    "legacy prototype observation"
                )
            legacy_support_sum, legacy_support_count = legacy_prototype_support
            legacy_fraction = legacy_support_sum / legacy_support_count.clamp_min(
                1e-8
            ).unsqueeze(1)
            prototype_positive_solver_mass, prototype_negative_solver_mass = (
                _registered_solver_masses(
                    legacy_fraction[:, 0],
                    legacy_fraction[:, 1],
                    support_threshold=float(args.support_threshold),
                    construction="winner_take_all",
                )
            )
            prototype_seed_construction = "legacy_winner_take_all"
            prototype_seed_normalization = "independent_max"
            prototype_seed_provenance = DUAL_PROTOTYPE_SEED_PROVENANCE
            solver_seed_provenance = DUAL_SOLVER_SEED_PROVENANCE
            _require_bipolar_solver_support(
                prototype_positive_solver_mass,
                prototype_negative_solver_mass,
                label="Legacy prototype",
            )
        positive_support = positive_solver_mass > 0
        negative_support = negative_solver_mass > 0
        _require_bipolar_solver_support(
            positive_solver_mass,
            negative_solver_mass,
            label="Global",
        )
        registered_prompt_evidence = {
            "seed_construction": seed_construction,
            "seed_normalization": seed_normalization,
            "prototype_seed_construction": prototype_seed_construction,
            "prototype_seed_normalization": prototype_seed_normalization,
            "prototype_seed_decoupled": (
                prototype_seed_construction != "shared"
            ),
            "observation_mass_source": direct_mass_source,
            "observation_confidence_mode": observation_confidence_mode,
            "observation_mass_scale": observation_mass_scale,
            "observation_coverage_power": (
                observation_coverage_power
                if observation_confidence_mode
                == "poisson_mass_coverage"
                else None
            ),
            "observation_confidence_formula": (
                "raw_joint_prompt_mass/raw_visible_mass; "
                "signed=(raw_foreground_mass-raw_background_mass)/"
                "raw_visible_mass"
                if observation_fusion in _EXACT_RASTER_OBSERVATION_FUSIONS
                else "(1-exp(-raw_joint_prompt_mass/mass_scale))*"
                "(raw_joint_prompt_mass/raw_visible_mass)^coverage_power"
                if observation_confidence_mode
                == "poisson_mass_coverage"
                else (
                    "1-exp(-raw_joint_prompt_mass/mass_scale)"
                    if observation_confidence_mode == "poisson_mass"
                    else "joint_prompt_fraction/max_joint_prompt_fraction"
                )
            ),
            "observation_constructed_before_capability_filter": True,
            "all_gaussians": int(support_count.numel()),
            "observed_gaussians": int((support_count > 0).sum()),
            "positive_prompt_mass_sum": float(positive_weight.sum()),
            "negative_prompt_mass_sum": float(negative_weight.sum()),
            "raw_positive_prompt_mass_sum": float(raw_positive_mass.sum()),
            "raw_negative_prompt_mass_sum": float(raw_negative_mass.sum()),
            "raw_visible_mass_sum": float(support_count.sum()),
            "observation_confidence_sum": float(direct_confidence.sum()),
            "strong_unary_policy": (
                "anchor_only_on_shared_hard_seed_rows"
                if observation_fusion
                == "hard_seed_anchor_only_probability"
                else "unit_confidence_on_shared_hard_seed_rows"
                if strong_unary_anchor_threshold is not None
                else "none"
            ),
            "strong_unary_fusion_scope": (
                "anchor_rows_only_with_bitwise_non_anchor_preservation"
                if observation_fusion
                == "hard_seed_anchor_only_probability"
                else "anchors_plus_weak_probability_mixture"
                if strong_unary_anchor_threshold is not None
                else "none"
            ),
            "strong_unary_anchor_threshold": strong_unary_anchor_threshold,
            "strong_unary_anchor_rows": int(strong_unary_anchor_mask.sum()),
            "strong_unary_fusion_rows": (
                int(strong_unary_anchor_mask.sum())
                if observation_fusion
                == "hard_seed_anchor_only_probability"
                else None
            ),
            "strong_unary_effective_confidence_sum": float(
                strong_unary_effective_confidence.sum()
            ),
            "strong_unary_formula": (
                (
                    "a=1[c>0 and abs(signed_observation)>=hard_seed_threshold]; "
                    "effective_confidence=a; "
                    "p=(1-a)*p_field+a*foreground_probability"
                    if observation_fusion
                    == "hard_seed_anchor_only_probability"
                    else "a=1[c>0 and "
                    "abs(signed_observation)>=hard_seed_threshold]; "
                    "effective_confidence=a+(1-a)*c; "
                    "p=(1-effective_confidence)*p_field+"
                    "effective_confidence*foreground_probability"
                )
                if strong_unary_anchor_threshold is not None
                else None
            ),
            **(
                {
                    "posterior_consensus": (
                        _registered_posterior_consensus_method_contract(
                            observation_fusion
                        )
                    )
                }
                if observation_fusion in _BERNOULLI_POE_FUSIONS
                else {}
            ),
            **(
                {
                    "prototype_observation_operator": {
                        "mode": "legacy_alpha_depth",
                        "registration_resolution": [
                            int(renderer.image_height),
                            int(renderer.image_width),
                        ],
                        "seed_construction": "winner_take_all",
                        "seed_normalization": "independent_max",
                        "depth_tolerance": float(args.depth_tolerance),
                        "relative_depth_tolerance": float(
                            args.relative_depth_tolerance
                        ),
                        "alpha_threshold": (
                            _FROZEN_LEGACY_PROTOTYPE_ALPHA_THRESHOLD
                        ),
                        "deterministic_cpu_accumulation": True,
                        "seed_provenance": (
                            DUAL_PROTOTYPE_SEED_PROVENANCE
                        ),
                        "shares_support_sum_with_exact_adjoint": False,
                    },
                    "exact_observation_operator": {
                        "mode": "native_front_to_back_raster_adjoint",
                        "registration_resolution": [height, width],
                        "alpha_threshold": float(args.alpha_threshold),
                        "seed_provenance": DUAL_SOLVER_SEED_PROVENANCE,
                        "seed_construction": seed_construction,
                        "seed_normalization": seed_normalization,
                    },
                }
                if _requires_legacy_prototype_observation(observation_fusion)
                else {}
            ),
            "positive_solver_mass_sum": float(positive_solver_mass.sum()),
            "negative_solver_mass_sum": float(negative_solver_mass.sum()),
            "positive_solver_rows": int(positive_support.sum()),
            "negative_solver_rows": int(negative_support.sum()),
            "neutral_observed_rows": int(
                (
                    (support_count > 0)
                    & ~positive_support
                    & ~negative_support
                ).sum()
            ),
            **(
                {
                    "source_completion_unary": {
                        **source_completion_evidence,
                        "probability_mean": float(
                            source_completion_probability.float().mean()
                        ),
                        "reliability_mean": float(
                            source_completion_reliability.float().mean()
                        ),
                        "primitive_confidence_sum": float(
                            source_completion_observation.confidence.sum()
                        ),
                        "primitive_signed_sum": float(
                            source_completion_observation.values.sum()
                        ),
                    }
                }
                if source_completion_observation is not None
                and source_completion_evidence is not None
                and source_completion_probability is not None
                and source_completion_reliability is not None
                else {}
            ),
        }
        if args.support_mode == "prompt_gaussian":
            assert region_rows is not None
            positive_rows = region_rows[positive_support.cpu()].to(
                device=device, dtype=torch.float32
            )
            negative_rows = region_rows[negative_support.cpu()].to(
                device=device, dtype=torch.float32
            )
            positive_prototypes = _weighted_spherical_prototypes(
                positive_rows,
                positive_solver_mass[positive_support],
                args.prototype_count,
            )
            negative_prototypes = _weighted_spherical_prototypes(
                negative_rows,
                negative_solver_mass[negative_support],
                args.prototype_count,
            )
            score_parts: list[torch.Tensor] = []
            for start in range(0, region_rows.shape[0], max(1, args.chunk_size)):
                rows_chunk = region_rows[start : start + max(1, args.chunk_size)].to(
                    device=device, dtype=torch.float32
                )
                positive_similarity = (rows_chunk @ positive_prototypes.T).amax(dim=1)
                negative_similarity = (rows_chunk @ negative_prototypes.T).amax(dim=1)
                score_parts.append((positive_similarity - negative_similarity).cpu())
            gaussian_scores = torch.cat(score_parts, dim=0).to(device)
        else:
            assert capability_bank is not None and support_graph is not None
            valid_rows = capability_bank.global_rows
            valid_rows_device = valid_rows.to(positive_solver_mass.device)
            positive_soft = (
                positive_solver_mass[valid_rows_device].detach().float().cpu()
            )
            negative_soft = (
                negative_solver_mass[valid_rows_device].detach().float().cpu()
            )
            prototype_positive_soft = (
                prototype_positive_solver_mass[valid_rows_device]
                .detach()
                .float()
                .cpu()
            )
            prototype_negative_soft = (
                prototype_negative_solver_mass[valid_rows_device]
                .detach()
                .float()
                .cpu()
            )
            _require_bipolar_solver_support(
                positive_soft,
                negative_soft,
                label="Capability-valid",
            )
            _require_bipolar_solver_support(
                prototype_positive_soft,
                prototype_negative_soft,
                label="Capability-valid prototype",
            )
            unary_observation = (
                source_completion_observation
                if (
                    source_completion_observation is not None
                    and not source_completion_abstained
                )
                else direct_observation
            )
            valid_observation = PrimitiveUnaryEvidence(
                unary_observation.values[valid_rows_device].detach().float().cpu(),
                (
                    _PROBABILITY_PRESERVING_SOURCE_UNARY
                    if (
                        source_completion_observation is not None
                        and not source_completion_abstained
                    )
                    else unary_observation.source
                ),
                (
                    unary_observation.confidence[valid_rows_device]
                    .detach()
                    .float()
                    .cpu()
                    if unary_observation.confidence is not None
                    else None
                ),
            )
            assert registered_prompt_evidence is not None
            registered_prompt_evidence.update(
                {
                    "capability_valid_gaussians": int(valid_rows.numel()),
                    "valid_positive_prompt_mass_sum": float(
                        positive_weight[valid_rows_device].sum()
                    ),
                    "valid_negative_prompt_mass_sum": float(
                        negative_weight[valid_rows_device].sum()
                    ),
                    "valid_raw_positive_prompt_mass_sum": float(
                        raw_positive_mass[valid_rows_device].sum()
                    ),
                    "valid_raw_negative_prompt_mass_sum": float(
                        raw_negative_mass[valid_rows_device].sum()
                    ),
                    "valid_observation_confidence_sum": float(
                        valid_observation.confidence.sum()
                        if valid_observation.confidence is not None
                        else 0.0
                    ),
                    "valid_strong_unary_anchor_rows": int(
                        strong_unary_anchor_mask[valid_rows_device].sum()
                    ),
                    "valid_strong_unary_fusion_rows": (
                        int(strong_unary_anchor_mask[valid_rows_device].sum())
                        if observation_fusion
                        == "hard_seed_anchor_only_probability"
                        else None
                    ),
                    "valid_strong_unary_effective_confidence_sum": float(
                        strong_unary_effective_confidence[
                            valid_rows_device
                        ].sum()
                    ),
                    "valid_positive_solver_mass_sum": float(positive_soft.sum()),
                    "valid_negative_solver_mass_sum": float(negative_soft.sum()),
                    "valid_positive_solver_rows": int((positive_soft > 0).sum()),
                    "valid_negative_solver_rows": int((negative_soft > 0).sum()),
                    **(
                        {
                            "valid_prototype_positive_solver_mass_sum": float(
                                prototype_positive_soft.sum()
                            ),
                            "valid_prototype_negative_solver_mass_sum": float(
                                prototype_negative_soft.sum()
                            ),
                            "valid_prototype_positive_solver_rows": int(
                                (prototype_positive_soft > 0).sum()
                            ),
                            "valid_prototype_negative_solver_rows": int(
                                (prototype_negative_soft > 0).sum()
                            ),
                        }
                        if prototype_seed_construction != "shared"
                        else {}
                    ),
                }
            )
            feature_banks = {
                name: values.to(device)
                for name, values in capability_bank.valid_feature_banks().items()
            }
            support_graph = support_graph.to(device)
            query = compile_registered_primitive_seeds(
                positive_soft,
                negative_soft,
                appearance_features=feature_banks["appearance"],
                boundary_features=feature_banks["boundary"],
                appearance_signature=capability_bank.signatures["appearance"],
                boundary_signature=capability_bank.signatures["boundary"],
                prototype_count=args.prototype_count,
                prototype_strategy=args.prototype_strategy,
                primitive_unary_evidence=valid_observation,
                seed_normalization=seed_normalization,
                prototype_positive_seeds=(
                    prototype_positive_soft
                    if prototype_seed_construction != "shared"
                    else None
                ),
                prototype_negative_seeds=(
                    prototype_negative_soft
                    if prototype_seed_construction != "shared"
                    else None
                ),
                prototype_seed_provenance=prototype_seed_provenance,
                solver_seed_provenance=solver_seed_provenance,
                selection_mode=SelectionMode(
                    getattr(
                        args,
                        "registered_selection_mode",
                        SelectionMode.SEEDED_COMPONENT.value,
                    )
                ),
            )
            normalized_positive = query.positive_seeds
            normalized_negative = query.negative_seeds
            assert normalized_positive is not None
            hard_seed_threshold = float(
                getattr(args, "hard_seed_threshold", 0.20)
            )
            hard_positive = normalized_positive.weights >= hard_seed_threshold
            hard_negative = (
                normalized_negative.weights >= hard_seed_threshold
                if normalized_negative is not None
                else torch.zeros_like(hard_positive)
            )
            hard_seed_conflict_policy = str(
                getattr(args, "hard_seed_conflict_policy", "positive_priority")
            )
            hard_seed_conflict_margin = float(
                getattr(args, "hard_seed_conflict_margin", 0.0)
            )
            if hard_seed_conflict_policy == "positive_priority":
                hard_negative &= ~hard_positive
            else:
                positive_values = normalized_positive.weights
                negative_values = (
                    normalized_negative.weights
                    if normalized_negative is not None
                    else torch.zeros_like(positive_values)
                )
                hard_positive &= (
                    positive_values
                    > negative_values + hard_seed_conflict_margin
                )
                hard_negative &= (
                    negative_values
                    > positive_values + hard_seed_conflict_margin
                )
            registered_prompt_evidence.update(
                {
                    "hard_positive_valid_rows": int(hard_positive.sum()),
                    "hard_negative_valid_rows": int(hard_negative.sum()),
                    "hard_seed_conflict_policy": hard_seed_conflict_policy,
                    "hard_seed_conflict_margin": hard_seed_conflict_margin,
                    "hard_seed_threshold": hard_seed_threshold,
                }
            )
            engine = CanonicalQueryEngine(
                support_graph,
                scoring_config=EvidenceScoringConfig(
                    semantic_weight=1.0,
                    appearance_weight=args.appearance_weight,
                    boundary_weight=args.boundary_weight,
                    prototype_temperature=args.prototype_temperature,
                    feature_calibration=args.feature_calibration,
                    background_centroids=args.background_centroids,
                    calibration_sample_size=args.calibration_sample_size,
                    centroid_iterations=args.centroid_iterations,
                    score_calibration=args.score_calibration,
                    score_tanh_scale=args.score_tanh_scale,
                    score_chunk_size=args.score_chunk_size,
                    negative_spatial_mode=str(
                        getattr(args, "negative_spatial_mode", "none")
                    ),
                    negative_spatial_steps=int(
                        getattr(args, "negative_spatial_steps", 4)
                    ),
                    negative_spatial_decay=float(
                        getattr(args, "negative_spatial_decay", 0.8)
                    ),
                    registered_seed_unary_weight=float(
                        getattr(args, "registered_seed_unary_weight", 0.0)
                    ),
                    registered_observation_fusion=observation_fusion,
                ),
                solver_config=SupportSolverConfig(
                    iterations=args.solver_iterations,
                    residual=args.solver_residual,
                    unary_temperature=args.solver_unary_temperature,
                    support_threshold=args.solver_support_threshold,
                    solver_type=getattr(args, "solver_type", "diffusion"),
                    laplacian_weight=getattr(args, "laplacian_weight", 1.0),
                    cg_iterations=getattr(args, "cg_iterations", 64),
                    cg_tolerance=getattr(args, "cg_tolerance", 1e-5),
                    hard_seed_threshold=hard_seed_threshold,
                    hard_seed_conflict_policy=hard_seed_conflict_policy,
                    hard_seed_conflict_margin=hard_seed_conflict_margin,
                    component_edge_threshold=float(
                        getattr(args, "component_edge_threshold", 1e-5)
                    ),
                    seeded_component_min_weight=float(
                        getattr(args, "seeded_component_min_weight", 0.20)
                    ),
                ),
                graph_policy=args.graph_policy,
                component_graph_policy=args.component_graph_policy,
                graph_legacy_residual=args.graph_legacy_residual,
                channel_confidence_mode=str(
                    getattr(args, "channel_confidence_mode", "none")
                ),
                node_reliability=(
                    primitive_reliability.valid_confidence().to(device)
                    if primitive_reliability is not None
                    else None
                ),
            )
            forward_unary_contract = _registered_forward_unary_contract(args)
            if forward_unary_contract is None:
                result = engine.execute(
                    query,
                    feature_banks,
                    feature_signatures=capability_bank.signatures,
                )
            else:
                exact_hits = rasterize_single_view_contributions(
                    model,
                    renderer,
                    prompt_pose,
                    height=height,
                    width=width,
                )
                positive_pixels = torch.from_numpy(
                    np.ascontiguousarray(positive.reshape(-1))
                ).bool()
                negative_pixels = torch.from_numpy(
                    np.ascontiguousarray(negative.reshape(-1))
                ).bool()
                (
                    result,
                    _field_result,
                    _forward_observation,
                    forward_diagnostics,
                ) = _execute_registered_forward_beta(
                    engine,
                    query,
                    feature_banks,
                    capability_bank.signatures,
                    gaussian_ids=exact_hits["gaussian_ids"],
                    pixel_ids=exact_hits["pixel_ids"],
                    contribution_weights=exact_hits["weights"],
                    capability_valid=capability_bank.valid,
                    valid_rows=valid_rows,
                    positive_pixels=positive_pixels,
                    negative_pixels=negative_pixels,
                    unary_temperature=float(args.solver_unary_temperature),
                    mode=str(args.registered_forward_unary),
                    primitive_reliability=(
                        primitive_reliability.confidence
                        if (
                            str(args.registered_forward_unary)
                            == "beta_balanced_residual_v2"
                            and primitive_reliability is not None
                        )
                        else None
                    ),
                    primitive_coverage=(
                        primitive_reliability.components["observation_evidence"]
                        if (
                            str(args.registered_forward_unary)
                            == "beta_balanced_residual_v2"
                            and primitive_reliability is not None
                        )
                        else None
                    ),
                    anchor_threshold=(
                        float(args.hard_seed_threshold)
                        if str(args.registered_forward_unary)
                        == "beta_balanced_residual_v2"
                        else None
                    ),
                )
                registered_prompt_evidence["registered_forward_unary"] = {
                    "contract": forward_unary_contract,
                    "diagnostics": _compact_registered_forward_beta_diagnostics(
                        forward_diagnostics,
                        capability_bank.valid,
                    ),
                }
            def expand_valid_rows(values: torch.Tensor) -> torch.Tensor:
                expanded = torch.zeros(
                    capability_bank.num_gaussians, dtype=torch.float32
                )
                expanded[valid_rows] = values.detach().float().cpu()
                return expanded.to(device)

            unary_prior = torch.sigmoid(
                result.unary / float(args.solver_unary_temperature)
            )
            canonical_stage_gaussian_scores = {
                "unary_prior": expand_valid_rows(unary_prior),
                "propagated": expand_valid_rows(result.probabilities),
                "connected": expand_valid_rows(result.selected_probabilities),
            }
            if source_observation_oof_authority is not None:
                assert source_observation_evidence_authority is not None
                assert source_observation_oof_contract is not None
                oof_root = Path(
                    args.source_observation_oof_output_dir
                ).expanduser().resolve()
                heldout_fold = int(args.source_observation_oof_heldout_fold)
                if source_observation_footprint_bundle is None:
                    fold_path, fold_sha256, fold_receipt = (
                        _write_source_observation_oof_artifact(
                            oof_root / f"fold_{heldout_fold}.pt",
                            scene_id=str(args.scene_id),
                            protocol_hash=str(manifest["protocol_hash"]),
                            heldout_fold=heldout_fold,
                            capability_cache=args.canonical_capability_cache,
                            support_graph=args.canonical_support_graph,
                            authority=source_observation_oof_authority,
                            evidence_authority=source_observation_evidence_authority,
                            valid=capability_bank.valid,
                            global_rows=valid_rows,
                            unary_probability=canonical_stage_gaussian_scores[
                                "unary_prior"
                            ],
                            propagated_probability=canonical_stage_gaussian_scores[
                                "propagated"
                            ],
                            method_contract=source_observation_oof_contract,
                        )
                    )
                    gate_path, gate_status = (
                        _write_source_observation_oof_gate_receipt(oof_root)
                    )
                    sealed_status = "source_observation_oof_fold_sealed"
                else:
                    assert source_observation_population_positive_weight is not None
                    assert source_observation_population_negative_weight is not None
                    footprint_path, footprint_file_sha256, footprint_authority = (
                        source_observation_footprint_bundle
                    )
                    fold_path, fold_sha256, fold_receipt = (
                        _write_source_observation_footprint_oof_artifact(
                            oof_root / f"fold_{heldout_fold}.pt",
                            scene_id=str(args.scene_id),
                            protocol_hash=str(manifest["protocol_hash"]),
                            heldout_fold=heldout_fold,
                            capability_cache=args.canonical_capability_cache,
                            support_graph=args.canonical_support_graph,
                            authority=source_observation_oof_authority,
                            evidence_authority=source_observation_evidence_authority,
                            footprint_path=footprint_path,
                            footprint_file_sha256=footprint_file_sha256,
                            footprint_authority=footprint_authority,
                            valid=capability_bank.valid,
                            global_rows=valid_rows,
                            population_positive_weight=(
                                source_observation_population_positive_weight
                            ),
                            population_negative_weight=(
                                source_observation_population_negative_weight
                            ),
                            unary_probability=canonical_stage_gaussian_scores[
                                "unary_prior"
                            ],
                            propagated_probability=canonical_stage_gaussian_scores[
                                "propagated"
                            ],
                            method_contract=source_observation_oof_contract,
                        )
                    )
                    gate_path, gate_status = (
                        _write_source_observation_footprint_oof_gate_receipt(
                            oof_root
                        )
                    )
                    sealed_status = (
                        "source_observation_footprint_oof_fold_sealed"
                    )
                return {
                    "schema_version": 1,
                    "status": sealed_status,
                    "scene_id": str(args.scene_id),
                    "protocol_hash": str(manifest["protocol_hash"]),
                    "heldout_fold": heldout_fold,
                    "fold_artifact": str(fold_path),
                    "fold_artifact_sha256": fold_sha256,
                    "fold_receipt": str(fold_receipt),
                    "fold_receipt_sha256": _file_sha256(fold_receipt),
                    "source_evidence_authority": str(
                        source_observation_evidence_authority.path
                    ),
                    "source_evidence_authority_sha256": (
                        source_observation_evidence_authority.sha256
                    ),
                    "gate_receipt": str(gate_path) if gate_path is not None else None,
                    "gate": gate_status,
                    "target_score_rendering_performed": False,
                    "target_rgb_opened": False,
                    "target_mask_opened": False,
                    "target_metric_computed": False,
                }
            query_diffusion_kernel = str(
                getattr(args, "query_conditioned_diffusion_kernel", "none")
            )
            if query_diffusion_kernel != "none":
                # The compatibility path has already persisted the legacy
                # stage scores above.  Release its 4096-D/1024-D GPU banks and
                # k16 solver before constructing a K201 graph; otherwise large
                # scenes needlessly keep both complete methods resident.
                del engine, query, feature_banks, support_graph, result
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                relation_path = Path(args.query_diffusion_feature_cache).resolve()
                support_graph_sha256 = _file_sha256(
                    Path(args.canonical_support_graph)
                )
                expected_field_hash = str(
                    capability_bank.metadata.get("field_checkpoint_sha256", "")
                )
                relation_cache = load_query_diffusion_relation_cache(
                    relation_path,
                    expected_global_rows=valid_rows,
                    expected_xyz=capability_bank.xyz[valid_rows],
                    expected_source_graph_sha256=support_graph_sha256,
                    expected_field_checkpoint_sha256=expected_field_hash,
                    expected_source_capability_cache=(
                        args.canonical_capability_cache
                    ),
                )
                relation_metadata = relation_cache.metadata
                relation_features = relation_cache.features
                relation_feature_dimension = relation_cache.feature_dimension
                del relation_cache
                knn_cache = load_query_diffusion_knn_cache(
                    args.query_diffusion_knn_cache,
                    expected_global_rows=valid_rows,
                    expected_xyz=capability_bank.xyz[valid_rows],
                    expected_source_graph_sha256=support_graph_sha256,
                    expected_num_neighbors=200,
                )
                # Retain only the row/provenance contract after the relation
                # cache has been verified.  The dense official capability
                # tensors are no longer inputs to this query-time operator.
                capability_bank = CanonicalCapabilityBank(
                    xyz=capability_bank.xyz,
                    valid=capability_bank.valid,
                    appearance=torch.empty((valid_rows.numel(), 0)),
                    boundary=torch.empty((valid_rows.numel(), 0)),
                    signatures=capability_bank.signatures,
                    metadata=capability_bank.metadata,
                    features_are_compact=True,
                )
                continuous_convex_v2 = (
                    query_diffusion_kernel == "continuous_convex_v2"
                )
                # Release compatibility uses the positive side alone.  V2
                # instead retains the complete foreground/background adjoint:
                # its full reference mask gives a calibrated Bernoulli unary
                # and confidence for every visible primitive.
                positive_initial = (
                    positive_weight[valid_rows_device].detach().float().cpu()
                )
                negative_initial = (
                    negative_weight[valid_rows_device].detach().float().cpu()
                )
                reference_weight = (
                    support_count[valid_rows_device].detach().float().cpu()
                )
                positive_rows_before_cap = int((positive_initial > 0).sum())
                if continuous_convex_v2:
                    observation_confidence = (
                        positive_initial + negative_initial
                    ).clamp(0.0, 1.0)
                    observation_probability = torch.where(
                        observation_confidence > 0,
                        positive_initial
                        / observation_confidence.clamp_min(1e-8),
                        torch.full_like(observation_confidence, 0.5),
                    ).clamp(0.0, 1.0)
                    classifier_evidence = positive_initial - negative_initial
                    logistic_fit_population = "signed_nonzero"
                else:
                    positive_initial = cap_positive_reference_evidence(
                        positive_initial,
                        max_positive_fraction=float(
                            args.query_diffusion_max_positive_fraction
                        ),
                    )
                    observation_confidence = None
                    observation_probability = None
                    classifier_evidence = positive_initial
                    logistic_fit_population = "all_nodes_positive_only"
                neighbors_device = knn_cache.neighbor_indices.to(device)
                calibration_records: list[dict[str, object]] = []
                if bool(args.query_diffusion_reference_calibration):
                    # Normalize once on CPU, where the relation cache already
                    # resides, then transfer only the fp32 normalized bank.
                    # This prevents simultaneous fp16+fp32 4096-D GPU banks in
                    # the matched-capacity diagnostic.
                    normalized_relation_cpu = normalize_node_features(
                        relation_features
                    )
                    del relation_features
                    base_compatibility = weighted_logistic_query_compatibility(
                        normalized_relation_cpu,
                        classifier_evidence,
                        reference_weight,
                        logistic_c=float(args.query_diffusion_logistic_c),
                        regularizer_bandwidth=1.0,
                        fit_population=logistic_fit_population,
                    ).to(device)
                    normalized_relation = normalized_relation_cpu.to(device)
                    del normalized_relation_cpu
                    feature_bandwidths = (
                        [1.0]
                        if continuous_convex_v2
                        else [2.0**value for value in (-1, 0, 1, 2, 3)]
                    )
                    regularizer_bandwidths = (
                        [1.0]
                        if continuous_convex_v2
                        else [2.0**value for value in (-3, -2, -1, 0)]
                    )
                    thresholds = np.arange(0.99, 0.02, -0.01, dtype=np.float64)
                    best_reference_iou = -1.0
                    best_support = None
                    best_compatibility = None
                    best_feature_bandwidth = None
                    best_regularizer_bandwidth = None
                    best_threshold = None

                    def reference_score_map(values: torch.Tensor) -> np.ndarray:
                        expanded = expand_valid_rows(values)
                        with torch.no_grad():
                            rendered = renderer.render_feature_rows(
                                model,
                                prompt_pose,
                                expanded[:, None],
                                feature_height=int(renderer.image_height),
                                feature_width=int(renderer.image_width),
                                alpha_normalize=True,
                                contribution_gamma=args.feature_contribution_gamma,
                                row_confidence=prompt_row_confidence,
                            )["feature_map"][0]
                        score = rendered.detach().float().cpu().numpy()
                        return cv2.resize(
                            score,
                            (native_width, native_height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    # Feature distances do not depend on either registered
                    # bandwidth.  Reusing this N-by-K bank is exactly
                    # equivalent to recomputation and is decisive for 4096-D
                    # diagnostic rows.
                    positive_reference_mask_device = (
                        classifier_evidence.to(device) > 0
                    )
                    threshold_source = str(
                        args.query_diffusion_reference_threshold_source
                    )
                    calibration_only = bool(
                        args.query_diffusion_reference_calibration_only
                    )
                    relation_distance_bank = (
                        None
                        if calibration_only
                        and threshold_source == "query_compatibility"
                        else knn_feature_distances(
                            normalized_relation,
                            neighbors_device,
                            distance_chunk_size=int(
                                args.query_diffusion_distance_chunk_size
                            ),
                        )
                    )
                    for feature_bandwidth in feature_bandwidths:
                        similarities = (
                            None
                            if relation_distance_bank is None
                            else rbf_similarity_from_distances(
                                relation_distance_bank,
                                feature_bandwidth=float(feature_bandwidth),
                                positive_reference_mask=(
                                    positive_reference_mask_device
                                ),
                            )
                        )
                        for regularizer_bandwidth in regularizer_bandwidths:
                            candidate_compatibility = base_compatibility.pow(
                                1.0 / float(regularizer_bandwidth)
                            )
                            candidate_config = QueryConditionedDiffusionConfig(
                                kernel=query_diffusion_kernel,
                                feature_bandwidth=float(feature_bandwidth),
                                regularizer_bandwidth=float(
                                    regularizer_bandwidth
                                ),
                                logistic_c=float(args.query_diffusion_logistic_c),
                                logistic_fit_population=logistic_fit_population,
                                iterations=int(args.query_diffusion_iterations),
                                edge_binarize_threshold=(
                                    None
                                    if continuous_convex_v2
                                    else float(
                                        args.query_diffusion_edge_binarize_threshold
                                    )
                                ),
                                distance_chunk_size=int(
                                    args.query_diffusion_distance_chunk_size
                                ),
                            )
                            if calibration_only and threshold_source == (
                                "query_compatibility"
                            ):
                                candidate_support = candidate_compatibility
                            elif continuous_convex_v2:
                                assert observation_probability is not None
                                assert observation_confidence is not None
                                assert similarities is not None
                                candidate_support = solve_continuous_query_support(
                                    observation_probability.to(device),
                                    observation_confidence.to(device),
                                    neighbors_device,
                                    similarities,
                                    candidate_compatibility,
                                    config=candidate_config,
                                )
                            else:
                                assert similarities is not None
                                candidate_support = run_query_conditioned_diffusion(
                                    positive_initial.to(device),
                                    neighbors_device,
                                    similarities,
                                    candidate_compatibility,
                                    config=candidate_config,
                                ).squeeze(1)
                            rendered_reference = reference_score_map(
                                candidate_compatibility
                                if threshold_source == "query_compatibility"
                                else candidate_support
                            )
                            candidate_best_iou = -1.0
                            candidate_threshold = None
                            for threshold in thresholds:
                                selected = rendered_reference >= float(threshold)
                                intersection = int(
                                    np.logical_and(selected, positive_native).sum()
                                )
                                union = int(
                                    np.logical_or(selected, positive_native).sum()
                                )
                                reference_iou = (
                                    float(intersection / union) if union else 1.0
                                )
                                # Descending thresholds and strict improvement
                                # reproduce the release's deterministic first tie.
                                if reference_iou > candidate_best_iou:
                                    candidate_best_iou = reference_iou
                                    candidate_threshold = float(threshold)
                            calibration_records.append(
                                {
                                    "feature_bandwidth": float(feature_bandwidth),
                                    "regularizer_bandwidth": float(
                                        regularizer_bandwidth
                                    ),
                                    "reference_iou": float(candidate_best_iou),
                                    "rendered_threshold": candidate_threshold,
                                    "threshold_source": threshold_source,
                                }
                            )
                            if candidate_best_iou > best_reference_iou:
                                best_reference_iou = float(candidate_best_iou)
                                best_support = candidate_support.detach().clone()
                                best_compatibility = (
                                    candidate_compatibility.detach().clone()
                                )
                                best_feature_bandwidth = float(feature_bandwidth)
                                best_regularizer_bandwidth = float(
                                    regularizer_bandwidth
                                )
                                best_threshold = float(candidate_threshold)
                    if best_support is None or best_compatibility is None:
                        raise RuntimeError("reference-only diffusion calibration failed")
                    support = best_support
                    query_compatibility = best_compatibility
                    selected_query_diffusion_threshold = best_threshold
                    selected_feature_bandwidth = best_feature_bandwidth
                    selected_regularizer_bandwidth = best_regularizer_bandwidth
                    selected_reference_iou = best_reference_iou
                else:
                    relation_features_device = relation_features.to(device)
                    support, query_compatibility = compute_query_conditioned_support(
                        relation_features_device,
                        neighbors_device,
                        positive_initial.to(device),
                        reference_weight,
                        config=QueryConditionedDiffusionConfig(
                            kernel=query_diffusion_kernel,
                            feature_bandwidth=float(
                                args.query_diffusion_feature_bandwidth
                            ),
                            regularizer_bandwidth=float(
                                args.query_diffusion_regularizer_bandwidth
                            ),
                            logistic_c=float(args.query_diffusion_logistic_c),
                            logistic_fit_population="all_nodes_positive_only",
                            iterations=int(args.query_diffusion_iterations),
                            edge_binarize_threshold=(
                                float(args.query_diffusion_edge_binarize_threshold)
                                if query_diffusion_kernel
                                == "ludvig_release_compat"
                                else None
                            ),
                            distance_chunk_size=int(
                                args.query_diffusion_distance_chunk_size
                            ),
                        ),
                    )
                    selected_query_diffusion_threshold = float(
                        args.solver_support_threshold
                    )
                    selected_feature_bandwidth = float(
                        args.query_diffusion_feature_bandwidth
                    )
                    selected_regularizer_bandwidth = float(
                        args.query_diffusion_regularizer_bandwidth
                    )
                    selected_reference_iou = None
                canonical_stage_gaussian_scores.update(
                    {
                        "query_compatibility": expand_valid_rows(
                            query_compatibility
                        ),
                        "propagated": expand_valid_rows(support),
                        "connected": expand_valid_rows(support),
                    }
                )
                assert registered_prompt_evidence is not None
                registered_prompt_evidence["query_conditioned_diffusion"] = {
                    "kernel": query_diffusion_kernel,
                    "execution_optimization": (
                        "implicit_symmetrized_boolean_reachability_nonnegative_v1"
                        if query_diffusion_kernel == "ludvig_release_compat"
                        else (
                            "implicit_knn_symmetric_energy_jacobi_pcg_v1"
                            if continuous_convex_v2
                            else "explicit_symmetrized_sparse_numeric_v1"
                        )
                    ),
                    "execution_optimization_changes_method_semantics": False,
                    "relation_cache": str(relation_path),
                    "relation_cache_sha256": _file_sha256(relation_path),
                    "relation_feature_semantics": relation_metadata.get(
                        "kernel_compatibility_scope"
                    ),
                    "relation_projection": relation_metadata.get("projection"),
                    "relation_feature_dimension": relation_feature_dimension,
                    "relation_storage_dtype": relation_metadata.get(
                        "storage_dtype"
                    ),
                    "lossy_relation_compression": relation_metadata.get(
                        "lossy_relation_compression"
                    ),
                    "relation_distance_bank_reused_across_feature_bandwidths": bool(
                        args.query_diffusion_reference_calibration
                    ),
                    "relation_normalization_execution": (
                        "cpu_fp32_l2_then_single_gpu_transfer_v1"
                        if bool(args.query_diffusion_reference_calibration)
                        else "end_to_end_default_v1"
                    ),
                    "native_ludvig_dinov2_pca40_exact": False,
                    "diagnostic_status": relation_metadata.get(
                        "diagnostic_status"
                    ),
                    "formal_preregistered_result": relation_metadata.get(
                        "formal_preregistered_result"
                    ),
                    "scene_selection_after_full9": relation_metadata.get(
                        "scene_selection_after_full9"
                    ),
                    "diagnostic_declaration": relation_metadata.get(
                        "diagnostic_declaration"
                    ),
                    "diagnostic_declaration_sha256": relation_metadata.get(
                        "diagnostic_declaration_sha256"
                    ),
                    "knn_cache": str(
                        Path(args.query_diffusion_knn_cache).resolve()
                    ),
                    "official_num_neighbors_parameter": 200,
                    "effective_knn_columns": int(knn_cache.effective_k),
                    "knn_includes_self": True,
                    "feature_bandwidth": selected_feature_bandwidth,
                    "regularizer_bandwidth": selected_regularizer_bandwidth,
                    "logistic_c": float(args.query_diffusion_logistic_c),
                    "logistic_fit_population": logistic_fit_population,
                    "max_positive_fraction": (
                        None
                        if continuous_convex_v2
                        else float(args.query_diffusion_max_positive_fraction)
                    ),
                    "positive_cap_rule": (
                        "none_preserve_complete_reference_observation"
                        if continuous_convex_v2
                        else "released_argsort_keep_largest_int_fraction"
                    ),
                    "positive_rows_before_cap": positive_rows_before_cap,
                    "positive_rows_after_cap": int((positive_initial > 0).sum()),
                    "iterations": (
                        None
                        if continuous_convex_v2
                        else int(args.query_diffusion_iterations)
                    ),
                    "edge_binarize_threshold": (
                        None
                        if continuous_convex_v2
                        else float(args.query_diffusion_edge_binarize_threshold)
                    ),
                    "continuous_convex_v2": (
                        {
                            "energy": "confidence_unary_plus_query_gated_symmetric_normalized_pairwise",
                            "laplacian_weight": 1.0,
                            "unobserved_fidelity": 0.01,
                            "hard_observation_confidence": 0.99,
                            "hard_positive_probability": 0.9,
                            "hard_negative_probability": 0.1,
                            "cg_iterations": 64,
                            "cg_tolerance": 1e-5,
                            "readout": "continuous_probability_all_components",
                            "hard_positive_rows": int(
                                (
                                    (observation_confidence >= 0.99)
                                    & (observation_probability >= 0.9)
                                ).sum()
                            ),
                            "hard_negative_rows": int(
                                (
                                    (observation_confidence >= 0.99)
                                    & (observation_probability <= 0.1)
                                ).sum()
                            ),
                        }
                        if continuous_convex_v2
                        else None
                    ),
                    "reference_calibration": bool(
                        args.query_diffusion_reference_calibration
                    ),
                    "reference_threshold_source": str(
                        args.query_diffusion_reference_threshold_source
                    ),
                    "reference_calibration_candidates": calibration_records,
                    "selected_reference_iou": selected_reference_iou,
                    "selected_rendered_threshold": (
                        selected_query_diffusion_threshold
                    ),
                    "reference_only": True,
                    "target_masks_opened": False,
                    "target_metrics_opened": False,
                }
                if bool(args.query_diffusion_reference_calibration_only):
                    threshold_receipt = {
                        "schema_version": (
                            "query_diffusion_reference_threshold_receipt_v1"
                        ),
                        "scene_id": str(args.scene_id),
                        "kernel": query_diffusion_kernel,
                        "threshold_source": str(
                            args.query_diffusion_reference_threshold_source
                        ),
                        "selected_rendered_threshold": float(
                            selected_query_diffusion_threshold
                        ),
                        "selected_reference_iou": float(
                            selected_reference_iou
                        ),
                        "reference_calibration_candidates": calibration_records,
                        "query_conditioned_diffusion": registered_prompt_evidence[
                            "query_conditioned_diffusion"
                        ],
                        "target_score_rendering_performed": False,
                        "target_masks_opened": False,
                        "target_metrics_opened": False,
                    }
                    threshold_receipt_path = (
                        output_root / "query_compatibility_reference_threshold.json"
                    )
                    threshold_receipt_path.write_text(
                        json.dumps(threshold_receipt, indent=2, allow_nan=False)
                        + "\n",
                        encoding="utf-8",
                    )
                    return threshold_receipt
            if _requires_legacy_prototype_observation(observation_fusion):
                # Audit the historical canonical-field expert independently
                # from the native raster-adjoint expert and their PoE.  This
                # stage must reproduce the frozen legacy unary before the
                # dual-registration method can be considered valid.
                prototype_expert_unary = torch.sigmoid(
                    result.field_unary / float(args.solver_unary_temperature)
                )
                canonical_stage_gaussian_scores[
                    "prototype_expert_unary"
                ] = expand_valid_rows(prototype_expert_unary)
                if bool(
                    getattr(
                        args,
                        "export_registered_prompt_cycle_diagnostic",
                        False,
                    )
                ):
                    exact_expert_unary = torch.sigmoid(
                        valid_observation.values
                        / float(args.solver_unary_temperature)
                    )
                    registered_prompt_cycle_gaussian_scores = {
                        "prototype_expert": expand_valid_rows(
                            prototype_expert_unary
                        ),
                        "exact_expert": expand_valid_rows(exact_expert_unary),
                    }
            gaussian_scores = canonical_stage_gaussian_scores[
                str(getattr(args, "registered_readout_stage", "connected"))
            ]
            prediction_threshold = (
                float(selected_query_diffusion_threshold)
                if query_diffusion_kernel != "none"
                else float(args.solver_support_threshold)
            )
            positive_seed_count = int((positive_soft > 0).sum())
            negative_seed_count = int((negative_soft > 0).sum())
        if args.support_mode != "canonical_support":
            positive_seed_count = int(positive_support.sum())
            negative_seed_count = int(negative_support.sum())
    else:
        if args.prompt_feature_source == "observed":
            prompt_region = _observed_region_map(
                queue_scene,
                str(prompt_view["camera_name"]),
                adaptor,
                device=device,
            )
        else:
            prompt_region, _ = _screen_region_map(
                model, codec, renderer, sharpener, refiner, field_config, adaptor,
                prompt_pose, is_hybrid=is_hybrid,
            )
        prompt_rows = prompt_region[0].permute(1, 2, 0).reshape(-1, prompt_region.shape[1])
        prompt_hw = (int(prompt_region.shape[-2]), int(prompt_region.shape[-1]))
        prompt_positive = resize_mask_nearest(positive.astype(np.uint8), prompt_hw).astype(bool)
        prompt_negative = resize_mask_nearest(negative.astype(np.uint8), prompt_hw).astype(bool)
        positive_flat = torch.from_numpy(prompt_positive.reshape(-1)).to(device)
        negative_flat = torch.from_numpy(prompt_negative.reshape(-1)).to(device)
        positive_prototypes = _weighted_spherical_prototypes(
            prompt_rows[positive_flat], torch.ones(int(positive_flat.sum()), device=device),
            args.prototype_count,
        )
        negative_prototypes = _weighted_spherical_prototypes(
            prompt_rows[negative_flat], torch.ones(int(negative_flat.sum()), device=device),
            args.prototype_count,
        )
        total_sum = torch.zeros(model.get_xyz().shape[0], 1, device=device)
        total_count = torch.zeros(model.get_xyz().shape[0], device=device)
        training_poses = _load_training_poses(queue_scene, evaluation_camera_names)
        support_view_count = len(training_poses)
        for support_pose_cpu in training_poses:
            support_pose = support_pose_cpu.to(device)
            support_region, support_aux = _screen_region_map(
                model, codec, renderer, sharpener, refiner, field_config, adaptor,
                support_pose, is_hybrid=is_hybrid,
            )
            support_rows = support_region[0].permute(1, 2, 0).reshape(
                -1, support_region.shape[1]
            )
            screen_scores = (
                (support_rows @ positive_prototypes.T).amax(dim=1)
                - (support_rows @ negative_prototypes.T).amax(dim=1)
            ).reshape(1, 1, height, width)
            lifted_sum, lifted_count = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=support_pose,
                siglip_feat=screen_scores,
                depth_map=support_aux["depth_map"],
                alpha_map=support_aux["alpha_map"],
                registration_depth_tolerance=args.depth_tolerance,
                registration_relative_depth_tolerance=args.relative_depth_tolerance,
                registration_alpha_threshold=args.alpha_threshold,
                registration_weight_mode="alpha_depth",
            )
            total_sum += lifted_sum
            total_count += lifted_count
        observed = total_count > 0
        gaussian_scores = torch.full_like(total_count, -1.0)
        gaussian_scores[observed] = total_sum[observed, 0] / total_count[observed]
        positive_seed_count = None
        negative_seed_count = None

    if registered_forward_protocol_authority is not None:
        prediction_threshold = 0.0

    output_root = Path(args.output_dir).resolve()
    primitive_unary_path: str | None = None
    primitive_unary_sha256: str | None = None
    if str(getattr(args, "primitive_unary_output", "")).strip():
        if canonical_stage_gaussian_scores is None:
            raise ValueError(
                "primitive unary export requires canonical-support compilation"
            )
        unary_artifact = _write_primitive_unary_artifact(
            args.primitive_unary_output,
            scene_id=str(args.scene_id),
            protocol_hash=str(manifest["protocol_hash"]),
            capability_cache=args.canonical_capability_cache,
            capability_source_contract=str(
                getattr(args, "canonical_capability_source_contract", "field")
            ),
            valid=capability_bank.valid,
            primitive_unary_probability=canonical_stage_gaussian_scores[
                "unary_prior"
            ],
            compiler_contract={
                "prototype_count": int(args.prototype_count),
                "prototype_strategy": str(args.prototype_strategy),
                "registered_seed_construction": str(
                    args.registered_seed_construction
                ),
                "registered_prototype_seed_construction": str(
                    args.registered_prototype_seed_construction
                ),
                "registered_observation_fusion": observation_fusion,
                "configured_registered_observation_fusion": (
                    configured_observation_fusion
                ),
                "registered_observation_confidence": str(
                    args.registered_observation_confidence
                ),
                "hard_seed_threshold": float(args.hard_seed_threshold),
                "unary_temperature": float(args.solver_unary_temperature),
                "graph_disabled": bool(args.disable_registered_graph),
                "connected_selection_applied": False,
                "readout": "unary_prior",
                **(
                    {
                        "source_completion_unary": registered_prompt_evidence[
                            "source_completion_unary"
                        ]
                    }
                    if registered_prompt_evidence is not None
                    and "source_completion_unary" in registered_prompt_evidence
                    else {}
                ),
            },
        )
        primitive_unary_path = str(unary_artifact)
        primitive_unary_sha256 = _file_sha256(unary_artifact)
    score_paths: dict[str, str] = {}
    score_sha256: dict[str, str] = {}
    predictions: dict[str, np.ndarray] = {}
    stage_score_paths: dict[str, dict[str, str]] = {}
    stage_score_sha256: dict[str, dict[str, str]] = {}
    stage_predictions: dict[str, dict[str, np.ndarray]] = {}
    score_resolution_mode = str(
        getattr(args, "score_render_resolution", "scaled_renderer")
    )
    if score_resolution_mode == "prompt_native":
        score_height, score_width = native_height, native_width
    elif score_resolution_mode == "scaled_renderer":
        score_height, score_width = _scaled_raster_shape(
            int(renderer.image_height),
            int(renderer.image_width),
            float(getattr(args, "score_render_scale", 1.0)),
        )
    else:
        raise ValueError(
            "score_render_resolution must be scaled_renderer or prompt_native"
        )
    valid_support = (
        capability_bank.valid.to(device=device, dtype=torch.float32)
        if (
            capability_bank is not None
            and bool(getattr(args, "valid_support_normalization", False))
        )
        else None
    )

    def render_scalar_scores(
        pose: torch.Tensor,
        values: torch.Tensor,
    ) -> np.ndarray:
        row_confidence = row_confidence_for_pose(pose)
        with torch.no_grad():
            if valid_support is None:
                score_map = renderer.render_feature_rows(
                    model,
                    pose,
                    values[:, None],
                    feature_height=score_height,
                    feature_width=score_width,
                    alpha_normalize=True,
                    contribution_gamma=args.feature_contribution_gamma,
                    row_confidence=row_confidence,
                )["feature_map"][0]
            else:
                channels = renderer.render_feature_rows(
                    model,
                    pose,
                    torch.stack(
                        [values * valid_support, valid_support],
                        dim=1,
                    ),
                    feature_height=score_height,
                    feature_width=score_width,
                    alpha_normalize=True,
                    contribution_gamma=args.feature_contribution_gamma,
                    row_confidence=row_confidence,
                )["feature_map"]
                score_map = _valid_normalized_score_map(
                    channels,
                    coverage_power=float(
                        getattr(args, "valid_support_coverage_power", 0.0)
                    ),
                )
        rendered_score = score_map.float().cpu().numpy()
        if registered_forward_protocol_authority is not None:
            return _center_registered_forward_score_map(rendered_score)
        return rendered_score

    registered_reference_threshold_calibration: dict[str, object] | None = None
    if bool(getattr(args, "registered_reference_threshold_calibration", False)):
        if prompt_type != "reference_binary_mask":
            raise ValueError(
                "registered reference-threshold calibration requires a full "
                "reference binary mask"
            )
        if canonical_stage_gaussian_scores is None:
            raise ValueError(
                "registered reference-threshold calibration requires canonical support"
            )
        if str(args.query_conditioned_diffusion_kernel) != "none":
            raise ValueError(
                "registered reference-threshold calibration cannot be combined "
                "with query-diffusion calibration"
            )
        if str(args.registered_readout_stage) != "unary_prior":
            raise ValueError(
                "registered reference-threshold calibration requires unary_prior"
            )
        prompt_score = render_scalar_scores(prompt_pose, gaussian_scores)
        prompt_score_native = cv2.resize(
            prompt_score,
            (native_width, native_height),
            interpolation=cv2.INTER_LINEAR,
        )
        best_reference_iou = -1.0
        best_threshold = None
        threshold_records: list[dict[str, float]] = []
        for threshold in np.arange(0.99, 0.02, -0.01, dtype=np.float64):
            selected = prompt_score_native >= float(threshold)
            intersection = int(np.logical_and(selected, positive_native).sum())
            union = int(np.logical_or(selected, positive_native).sum())
            reference_iou = float(intersection / union) if union else 1.0
            threshold_records.append(
                {
                    "threshold": float(threshold),
                    "reference_iou": reference_iou,
                }
            )
            if reference_iou > best_reference_iou:
                best_reference_iou = reference_iou
                best_threshold = float(threshold)
        if best_threshold is None:
            raise RuntimeError("reference unary threshold calibration failed")
        prediction_threshold = best_threshold
        registered_reference_threshold_calibration = {
            "policy": "reference_mask_unary_threshold_v1",
            "threshold_grid": {
                "start": 0.99,
                "stop_exclusive": 0.02,
                "step": -0.01,
                "source": "existing_release_compatible_reference_grid",
            },
            "selected_threshold": best_threshold,
            "selected_reference_iou": best_reference_iou,
            "candidates": threshold_records,
            "reference_only": True,
            "target_masks_opened": False,
            "target_metrics_opened": False,
        }
        assert registered_prompt_evidence is not None
        registered_prompt_evidence["reference_threshold_calibration"] = (
            registered_reference_threshold_calibration
        )

    registered_prompt_cycle_report: dict[str, object] | None = None
    if registered_prompt_cycle_gaussian_scores is not None:
        if registered_prompt_evidence is None:
            raise RuntimeError("prompt-cycle diagnostic lacks registered evidence")
        ordered_experts = ("prototype_expert", "exact_expert")
        with torch.no_grad():
            prompt_cycle_render = renderer.render_feature_rows(
                model,
                prompt_pose,
                torch.stack(
                    [
                        registered_prompt_cycle_gaussian_scores[name]
                        for name in ordered_experts
                    ],
                    dim=1,
                ),
                feature_height=native_height,
                feature_width=native_width,
                alpha_normalize=True,
                contribution_gamma=args.feature_contribution_gamma,
                row_confidence=prompt_row_confidence,
            )
        prompt_cycle_maps = (
            prompt_cycle_render["feature_map"].detach().float().cpu().numpy()
        )
        prompt_cycle_visibility = (
            prompt_cycle_render["alpha_map"]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(native_height, native_width)
        )
        if prompt_cycle_maps.shape != (2, native_height, native_width):
            raise RuntimeError("prompt-cycle renderer returned an invalid shape")
        cycle_root = output_root / "prompt_cycle" / args.scene_id
        cycle_root.mkdir(parents=True, exist_ok=True)
        cycle_score_paths: dict[str, str] = {}
        cycle_score_sha256: dict[str, str] = {}
        cycle_metrics: dict[str, dict[str, float]] = {}
        for index, expert_name in enumerate(ordered_experts):
            cycle_score = np.clip(prompt_cycle_maps[index], 0.0, 1.0)
            cycle_path = cycle_root / f"{expert_name}_{prompt_frame}.npy"
            np.save(
                cycle_path,
                cycle_score.astype(np.float32),
                allow_pickle=False,
            )
            cycle_score_paths[expert_name] = str(cycle_path)
            cycle_score_sha256[expert_name] = _file_sha256(cycle_path)
            cycle_metrics[expert_name] = _prompt_cycle_reconstruction_metrics(
                cycle_score,
                positive_native,
                prompt_cycle_visibility,
            )
        exact_positive_rows = int(
            registered_prompt_evidence["valid_positive_solver_rows"]
        )
        exact_negative_rows = int(
            registered_prompt_evidence["valid_negative_solver_rows"]
        )
        prototype_positive_rows = int(
            registered_prompt_evidence[
                "valid_prototype_positive_solver_rows"
            ]
        )
        prototype_negative_rows = int(
            registered_prompt_evidence[
                "valid_prototype_negative_solver_rows"
            ]
        )
        registered_prompt_cycle_report = {
            "schema": "registered_prompt_cycle_reliability_diagnostic_v1",
            "status": "target_blind_diagnostic_only",
            "scene_id": str(args.scene_id),
            "protocol_hash": str(manifest["protocol_hash"]),
            "prompt_frame_id": prompt_frame,
            "prompt_mask_path": str(prompt["mask_path"]),
            "prompt_mask_sha256": _file_sha256(Path(prompt["mask_path"])),
            "prompt_resolution": [native_height, native_width],
            "prompt_foreground_pixels": int(positive_native.sum()),
            "computed_before_any_target_mask_open": True,
            "allowed_inputs": [
                "reference_prompt_mask",
                "legacy_alpha_depth_observation",
                "native_exact_raster_adjoint_observation",
                "renderer_responsibility_and_visibility",
                "canonical_primitive_scores",
            ],
            "forbidden_inputs": [
                "target_rgb",
                "target_mask",
                "target_metric",
                "scene_specific_constants",
            ],
            "prototype_operator": registered_prompt_evidence[
                "prototype_observation_operator"
            ],
            "exact_operator": registered_prompt_evidence[
                "exact_observation_operator"
            ],
            "expert_metrics": cycle_metrics,
            "fixed_ranking": _prompt_cycle_fixed_ranking(cycle_metrics),
            "condition_indicators": {
                "exact_positive_rows": exact_positive_rows,
                "prototype_positive_rows": prototype_positive_rows,
                "exact_to_prototype_positive_row_ratio": (
                    exact_positive_rows / prototype_positive_rows
                ),
                "exact_negative_rows": exact_negative_rows,
                "prototype_negative_rows": prototype_negative_rows,
                "exact_to_prototype_negative_row_ratio": (
                    exact_negative_rows / prototype_negative_rows
                ),
                "exact_observation_confidence_sum": float(
                    registered_prompt_evidence[
                        "valid_observation_confidence_sum"
                    ]
                ),
                "mean_renderer_visibility": float(
                    prompt_cycle_visibility.mean()
                ),
                "visible_pixel_fraction": float(
                    (prompt_cycle_visibility > 0.0).mean()
                ),
            },
            "score_paths": cycle_score_paths,
            "score_sha256": cycle_score_sha256,
            "uses_target_rgb_or_mask": False,
            "learned_or_scene_tuned_constants": False,
        }
        cycle_report_path = cycle_root / "prompt_cycle_diagnostic.json"
        cycle_report_path.write_text(
            json.dumps(
                registered_prompt_cycle_report,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        registered_prompt_cycle_report = {
            **registered_prompt_cycle_report,
            "report_path": str(cycle_report_path),
            "report_sha256": _file_sha256(cycle_report_path),
        }

    for frame_id in evaluation_frames:
        view = _view_by_frame(views, frame_id)
        pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
        rendered = render_scalar_scores(pose, gaussian_scores)
        score_path = output_root / "scores" / args.scene_id / f"{frame_id}.npy"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(score_path, rendered.astype(np.float32), allow_pickle=False)
        score_paths[frame_id] = str(score_path)
        score_sha256[frame_id] = _file_sha256(score_path)
        predictions[frame_id] = rendered
        if canonical_stage_gaussian_scores is not None:
            rendered_stages = _render_registered_stage_maps(
                canonical_stage_gaussian_scores,
                final_stage=str(
                    getattr(args, "registered_readout_stage", "connected")
                ),
                final_rendered=rendered,
                render=lambda values: render_scalar_scores(pose, values),
            )
            for stage_name, stage_rendered in rendered_stages.items():
                stage_path = (
                    output_root
                    / "stage_scores"
                    / stage_name
                    / args.scene_id
                    / f"{frame_id}.npy"
                )
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(stage_path, stage_rendered.astype(np.float32), allow_pickle=False)
                stage_score_paths.setdefault(stage_name, {})[frame_id] = str(stage_path)
                stage_score_sha256.setdefault(stage_name, {})[
                    frame_id
                ] = _file_sha256(stage_path)
                stage_predictions.setdefault(stage_name, {})[frame_id] = stage_rendered

    prediction_receipt_path: str | None = None
    prediction_receipt_sha256: str | None = None
    if str(getattr(args, "prediction_receipt_output", "")).strip():
        receipt_path, receipt_sha256 = _write_pre_metric_prediction_receipt(
            args.prediction_receipt_output,
            scene_id=str(args.scene_id),
            protocol_hash=str(manifest["protocol_hash"]),
            capability_cache=args.canonical_capability_cache,
            support_graph=args.canonical_support_graph,
            score_paths=score_paths,
            score_sha256=score_sha256,
            stage_score_paths=stage_score_paths,
            stage_score_sha256=stage_score_sha256,
            method_contract={
                "support_mode": str(args.support_mode),
                "capability_source_contract": str(
                    args.canonical_capability_source_contract
                ),
                "prototype_count": int(args.prototype_count),
                "prototype_strategy": str(args.prototype_strategy),
                "registered_seed_construction": str(
                    args.registered_seed_construction
                ),
                "registered_prototype_seed_construction": str(
                    args.registered_prototype_seed_construction
                ),
                "registered_observation_fusion": observation_fusion,
                "configured_registered_observation_fusion": (
                    configured_observation_fusion
                ),
                "registered_observation_confidence": str(
                    args.registered_observation_confidence
                ),
                "registered_observation_mass_scale": float(
                    args.registered_observation_mass_scale
                ),
                "registered_observation_coverage_power": float(
                    args.registered_observation_coverage_power
                ),
                "prompt_registration_mode": str(args.prompt_registration_mode),
                "prompt_registration_scale": float(
                    args.prompt_registration_scale
                ),
                "alpha_threshold": float(args.alpha_threshold),
                "registered_selection_mode": str(
                    args.registered_selection_mode
                ),
                "registered_readout_stage": str(args.registered_readout_stage),
                "graph_policy": str(args.graph_policy),
                "component_graph_policy": str(args.component_graph_policy),
                "channel_confidence_mode": str(args.channel_confidence_mode),
                "solver_type": str(args.solver_type),
                "solver_iterations": int(args.solver_iterations),
                "solver_residual": float(args.solver_residual),
                "solver_unary_temperature": float(
                    args.solver_unary_temperature
                ),
                "solver_support_threshold": float(
                    args.solver_support_threshold
                ),
                "laplacian_weight": float(args.laplacian_weight),
                "hard_seed_threshold": float(args.hard_seed_threshold),
                "hard_seed_conflict_policy": str(
                    args.hard_seed_conflict_policy
                ),
                "hard_seed_conflict_margin": float(
                    args.hard_seed_conflict_margin
                ),
                "appearance_weight": float(args.appearance_weight),
                "boundary_weight": float(args.boundary_weight),
                "prototype_temperature": float(args.prototype_temperature),
                "feature_calibration": str(args.feature_calibration),
                "score_calibration": str(args.score_calibration),
                "query_conditioned_diffusion": (
                    {
                        "kernel": str(args.query_conditioned_diffusion_kernel),
                        "knn_cache": {
                            "path": str(
                                Path(args.query_diffusion_knn_cache)
                                .expanduser()
                                .resolve()
                            ),
                            "sha256": _file_sha256(
                                Path(args.query_diffusion_knn_cache)
                                .expanduser()
                                .resolve()
                            ),
                        },
                        "relation_cache": {
                            "path": str(
                                Path(args.query_diffusion_feature_cache)
                                .expanduser()
                                .resolve()
                            ),
                            "sha256": _file_sha256(
                                Path(args.query_diffusion_feature_cache)
                                .expanduser()
                                .resolve()
                            ),
                        },
                        "reference_calibration": bool(
                            args.query_diffusion_reference_calibration
                        ),
                        "feature_bandwidth": float(
                            args.query_diffusion_feature_bandwidth
                        ),
                        "regularizer_bandwidth": float(
                            args.query_diffusion_regularizer_bandwidth
                        ),
                        "logistic_c": float(args.query_diffusion_logistic_c),
                        "iterations": int(args.query_diffusion_iterations),
                        "edge_binarize_threshold": float(
                            args.query_diffusion_edge_binarize_threshold
                        ),
                        "max_positive_fraction": float(
                            args.query_diffusion_max_positive_fraction
                        ),
                    }
                    if str(args.query_conditioned_diffusion_kernel) != "none"
                    else None
                ),
                "score_threshold": float(prediction_threshold),
                "score_render_resolution": str(args.score_render_resolution),
                "score_render_scale": float(args.score_render_scale),
                "valid_support_normalization": bool(
                    args.valid_support_normalization
                ),
                "feature_contribution_gamma": float(
                    args.feature_contribution_gamma
                ),
                "source_observation_oof_deployment_gate": (
                    source_observation_oof_deployment_gate
                ),
                "registered_graph_disabled": bool(
                    args.disable_registered_graph
                ),
                "evaluator_sha256": _file_sha256(Path(__file__).resolve()),
                "candidate_args": {
                    str(name): value
                    for name, value in sorted(vars(args).items())
                },
                "source_completion_unary": source_completion_evidence,
                "primitive_unary_artifact": (
                    {
                        "path": primitive_unary_path,
                        "file_sha256": primitive_unary_sha256,
                        "score_tensor_sha256": tensor_sha256(
                            canonical_stage_gaussian_scores["unary_prior"]
                            .detach()
                            .float()
                            .cpu()
                            .contiguous()
                        ),
                    }
                    if primitive_unary_path is not None
                    and canonical_stage_gaussian_scores is not None
                    else None
                ),
            },
            graph_disabled=bool(args.disable_registered_graph),
        )
        prediction_receipt_path = str(receipt_path)
        prediction_receipt_sha256 = str(receipt_sha256)

    if bool(getattr(args, "prediction_only", False)):
        return {
            "schema_version": 1,
            "artifact_type": "nvos_prediction_only_completion_v1",
            "scene_id": str(args.scene_id),
            "protocol_hash": str(manifest["protocol_hash"]),
            "prediction_receipt": prediction_receipt_path,
            "prediction_receipt_sha256": prediction_receipt_sha256,
            "target_score_paths": dict(score_paths),
            "target_score_sha256": dict(score_sha256),
            "primitive_unary_path": primitive_unary_path,
            "primitive_unary_sha256": primitive_unary_sha256,
            "safety": {
                "stopped_before_target_ground_truth_open": True,
                "target_rgb_opened": False,
                "target_mask_opened": False,
                "target_metric_opened": False,
            },
        }

    # Evaluation begins only after every prediction has been persisted.
    frame_metrics: list[dict] = []
    stage_frame_metrics: dict[str, list[dict]] = {
        name: [] for name in stage_predictions
    }
    for frame_id in evaluation_frames:
        frame = next(value for value in scene["frames"] if str(value["frame_id"]) == frame_id)
        if bool(getattr(args, "require_asset_hashes", False)):
            _verify_declared_sha256(
                Path(frame["ground_truth"]),
                frame.get("ground_truth_sha256"),
                label=f"{args.scene_id} target {frame_id}",
            )
        gt = load_ground_truth_mask(frame["ground_truth"]).astype(bool)
        if registered_forward_protocol_authority is not None:
            score = _resize_nvos_score_for_evaluation(
                predictions[frame_id],
                gt.shape,
                registered_forward_unary=str(args.registered_forward_unary),
            )
        else:
            score = cv2.resize(
                predictions[frame_id],
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        pred = score >= prediction_threshold
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = float(intersection / union) if union else 1.0
        accuracy = float((pred == gt).mean())
        frame_metrics.append(
            {"frame_id": frame_id, "foreground_iou": iou, "pixel_accuracy": accuracy}
        )
        for stage_name, per_frame in stage_predictions.items():
            if registered_forward_protocol_authority is not None:
                stage_score = _resize_nvos_score_for_evaluation(
                    per_frame[frame_id],
                    gt.shape,
                    registered_forward_unary=str(args.registered_forward_unary),
                )
            else:
                stage_score = cv2.resize(
                    per_frame[frame_id],
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            stage_pred = stage_score >= prediction_threshold
            stage_intersection = np.logical_and(stage_pred, gt).sum()
            stage_union = np.logical_or(stage_pred, gt).sum()
            stage_iou = (
                float(stage_intersection / stage_union) if stage_union else 1.0
            )
            stage_accuracy = float((stage_pred == gt).mean())
            stage_frame_metrics[stage_name].append(
                {
                    "frame_id": frame_id,
                    "foreground_iou": stage_iou,
                    "pixel_accuracy": stage_accuracy,
                }
            )

    stage_metrics = {
        name: {
            "foreground_iou": float(
                np.mean([value["foreground_iou"] for value in values])
            ),
            "pixel_accuracy": float(
                np.mean([value["pixel_accuracy"] for value in values])
            ),
            "frames": values,
        }
        for name, values in stage_frame_metrics.items()
    }

    report = {
        "scene_id": args.scene_id,
        "protocol_hash": manifest["protocol_hash"],
        "method": (
            f"gaussian_first_{args.support_mode}_{args.region_space}_"
            f"{'beta_centered_posterior' if registered_forward_protocol_authority is not None else 'cosine_margin'}_"
            f"{args.prototype_count}proto_"
            f"{'raster_responsibility' if args.support_mode == 'canonical_support' else args.prompt_feature_source}_prompt"
        ),
        "positive_gaussian_seeds": positive_seed_count,
        "negative_gaussian_seeds": negative_seed_count,
        "positive_prompt_pixels": int(positive.sum()),
        "negative_prompt_pixels": int(negative.sum()),
        "positive_prompt_pixels_native": int(positive_native.sum()),
        "negative_prompt_pixels_native": int(negative_native.sum()),
        "prompt_native_resolution": [native_height, native_width],
        "prompt_registration_resolution": [height, width],
        "registered_prompt_evidence": registered_prompt_evidence,
        **(
            {"registered_prompt_cycle": registered_prompt_cycle_report}
            if registered_prompt_cycle_report is not None
            else {}
        ),
        **(
            {
                "registered_forward_protocol_authority": (
                    registered_forward_protocol_authority
                ),
                "registered_forward_protocol_authority_sha256": (
                    registered_forward_protocol_authority_sha256
                ),
                "registered_forward_protocol_authority_builder": (
                    "radio_gs/scripts/"
                    "bind_nvos_forward_beta_protocol_authority.py"
                ),
            }
            if registered_forward_protocol_authority is not None
            else {}
        ),
        "support_mode": args.support_mode,
        "support_view_count": support_view_count,
        "support_threshold": float(args.support_threshold),
        "prototype_count": int(args.prototype_count),
        "prototype_strategy": str(args.prototype_strategy),
        "prompt_feature_source": (
            "raster_responsibility"
            if args.support_mode == "canonical_support"
            else args.prompt_feature_source
        ),
        "prompt_type": prompt_type,
        "prompt_registration": (
            "exact_front_to_back_raster_adjoint"
            if getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
            == "raster_adjoint"
            else (
                "raster_responsibility_deterministic_cpu"
                if args.support_mode == "canonical_support"
                else "raster_contribution"
            )
        ),
        "feature_observation_operator": {
            "type": (
                "normalized_front_to_back_contribution_power"
                if float(args.feature_contribution_gamma) != 1.0
                else "alpha_normalized_mean"
            ),
            "gamma": float(args.feature_contribution_gamma),
            "score_render_resolution": [score_height, score_width],
            "score_render_resolution_mode": score_resolution_mode,
            "valid_support_normalization": bool(valid_support is not None),
            "valid_support_formula": (
                "sum(w*v*p)/sum(w*v) * coverage**coverage_power"
                if valid_support is not None
                else None
            ),
            "valid_support_coverage_power": (
                float(getattr(args, "valid_support_coverage_power", 0.0))
                if valid_support is not None
                else None
            ),
            "query_dependent": False,
            "changes_geometry_or_alpha": camera_clearance_sigma > 0,
            "camera_clearance": (
                {
                    "contract": CAMERA_PLANE_CLEARANCE_CONTRACT,
                    "support_sigma": camera_clearance_sigma,
                    "uses_rgb_prompt_query_or_mask": False,
                }
                if camera_clearance_sigma > 0
                else None
            ),
            **(
                {
                    "post_compositor_scoring_adapter": (
                        _registered_forward_scoring_contract(args)
                    )
                }
                if registered_forward_protocol_authority is not None
                else {}
            ),
        },
        "score_threshold": prediction_threshold,
        "primitive_unary_artifact": (
            {
                "path": primitive_unary_path,
                "sha256": primitive_unary_sha256,
                "written_before_target_ground_truth_open": True,
            }
            if primitive_unary_path is not None
            else None
        ),
        "canonical_capability_source_contract": str(
            getattr(args, "canonical_capability_source_contract", "field")
        ),
        "source_observation_oof_deployment_gate": (
            source_observation_oof_deployment_gate
        ),
        "registered_graph_disabled": bool(
            getattr(args, "disable_registered_graph", False)
        ),
        "shared_solver": (
            {
                "appearance_weight": float(args.appearance_weight),
                "boundary_weight": float(args.boundary_weight),
                "prototype_temperature": float(args.prototype_temperature),
                "iterations": int(args.solver_iterations),
                "residual": float(args.solver_residual),
                "unary_temperature": float(args.solver_unary_temperature),
                "support_threshold": float(args.solver_support_threshold),
                "solver_type": getattr(args, "solver_type", "diffusion"),
                "laplacian_weight": float(getattr(args, "laplacian_weight", 1.0)),
                "cg_iterations": int(getattr(args, "cg_iterations", 64)),
                "cg_tolerance": float(getattr(args, "cg_tolerance", 1e-5)),
                "hard_seed_threshold": float(
                    getattr(args, "hard_seed_threshold", 0.20)
                ),
                "hard_seed_conflict_policy": str(
                    getattr(
                        args,
                        "hard_seed_conflict_policy",
                        "positive_priority",
                    )
                ),
                "hard_seed_conflict_margin": float(
                    getattr(args, "hard_seed_conflict_margin", 0.0)
                ),
                "component_edge_threshold": float(
                    getattr(args, "component_edge_threshold", 1e-5)
                ),
                "seeded_component_min_weight": float(
                    getattr(args, "seeded_component_min_weight", 0.20)
                ),
                "score_chunk_size": int(args.score_chunk_size),
                "registered_seed_unary_weight": float(
                    getattr(args, "registered_seed_unary_weight", 0.0)
                ),
                "registered_observation_fusion": observation_fusion,
                "configured_registered_observation_fusion": (
                    configured_observation_fusion
                ),
                **(
                    {
                        "registered_strong_unary": {
                            "policy": (
                                "unit_confidence_on_shared_hard_seed_rows"
                            ),
                            "anchor_threshold_source": (
                                "hard_seed_threshold"
                            ),
                            "anchor_threshold": float(
                                getattr(args, "hard_seed_threshold", 0.20)
                            ),
                            "new_numeric_constant": False,
                        }
                    }
                    if observation_fusion
                    == "hard_seed_anchored_probability"
                    else {}
                ),
                **(
                    {
                        "registered_posterior_consensus": (
                            _registered_posterior_consensus_method_contract(
                                observation_fusion
                            )
                        )
                    }
                    if observation_fusion
                    in _BERNOULLI_POE_FUSIONS
                    else {}
                ),
                **(
                    {
                        "registered_strong_unary": {
                            "policy": "anchor_only_on_shared_hard_seed_rows",
                            "anchor_threshold_source": "hard_seed_threshold",
                            "anchor_threshold": float(
                                getattr(args, "hard_seed_threshold", 0.20)
                            ),
                            "formula": (
                                "a=1[c>0 and abs(s)>=tau]; c_eff=a; "
                                "p=(1-a)p_field+a*q"
                            ),
                            "non_anchor_policy": (
                                "bitwise_field_unary_preservation"
                            ),
                            "new_numeric_constant": False,
                        }
                    }
                    if observation_fusion
                    == "hard_seed_anchor_only_probability"
                    else {}
                ),
                "registered_seed_construction": str(
                    getattr(
                        args,
                        "registered_seed_construction",
                        "winner_take_all",
                    )
                ),
                "registered_prototype_seed_construction": str(
                    getattr(
                        args,
                        "registered_prototype_seed_construction",
                        "shared",
                    )
                ),
                "registered_prototype_seed_normalization": (
                    str(prototype_seed_normalization)
                    if registered_prompt_evidence is not None
                    else None
                ),
                "registered_seed_normalization": (
                    str(seed_normalization)
                    if registered_prompt_evidence is not None
                    else None
                ),
                "registered_observation_confidence": str(
                    "exact_labeled_visible_fraction"
                    if str(
                        getattr(
                            args,
                            "registered_observation_fusion",
                            "additive",
                        )
                    )
                    in _EXACT_RASTER_OBSERVATION_FUSIONS
                    else getattr(
                        args,
                        "registered_observation_confidence",
                        "relative_joint_max",
                    )
                ),
                "registered_observation_mass_scale": float(
                    getattr(args, "registered_observation_mass_scale", 1.0)
                ),
                "registered_observation_coverage_power": float(
                    getattr(
                        args,
                        "registered_observation_coverage_power",
                        1.0,
                    )
                ),
                "registered_selection_mode": str(
                    getattr(
                        args,
                        "registered_selection_mode",
                        SelectionMode.SEEDED_COMPONENT.value,
                    )
                ),
                "registered_readout_stage": str(
                    getattr(args, "registered_readout_stage", "connected")
                ),
                **(
                    {"registered_forward_unary": registered_forward_contract}
                    if registered_forward_contract is not None
                    else {}
                ),
                "graph_policy": args.graph_policy,
                "component_graph_policy": args.component_graph_policy,
                "graph_legacy_residual": float(args.graph_legacy_residual),
                "query_conditioned_diffusion": (
                    registered_prompt_evidence.get(
                        "query_conditioned_diffusion"
                    )
                    if (
                        registered_prompt_evidence is not None
                        and str(
                            getattr(
                                args,
                                "query_conditioned_diffusion_kernel",
                                "none",
                            )
                        )
                        != "none"
                    )
                    else None
                ),
                "channel_confidence_mode": str(
                    getattr(args, "channel_confidence_mode", "none")
                ),
                "negative_spatial_mode": str(
                    getattr(args, "negative_spatial_mode", "none")
                ),
                "negative_spatial_steps": int(
                    getattr(args, "negative_spatial_steps", 4)
                ),
                "negative_spatial_decay": float(
                    getattr(args, "negative_spatial_decay", 0.8)
                ),
                "spatial_log_weight": 0.25,
                "spatial_floor": 0.01,
                "feature_calibration": args.feature_calibration,
                "background_centroids": int(args.background_centroids),
                "calibration_sample_size": int(args.calibration_sample_size),
                "centroid_iterations": int(args.centroid_iterations),
                "score_calibration": args.score_calibration,
                "score_tanh_scale": float(args.score_tanh_scale),
                "calibration_uses_target_labels": False,
                "calibration_uses_target_masks": False,
                "calibration_uses_query_conditioned_scores": (
                    args.score_calibration != "none"
                ),
                "calibration_uses_unlabeled_scene_statistics": (
                    args.feature_calibration != "none"
                    or int(args.background_centroids) > 0
                    or args.score_calibration != "none"
                ),
                "primitive_reliability": (
                    {
                        "cache": str(
                            Path(args.canonical_reliability_cache).resolve()
                        ),
                        "formula": primitive_reliability.metadata.get("formula"),
                        "application": "centered_unary_shrink",
                        "prototype_precision_weighting": False,
                        "centered_unary_shrink": True,
                        "seed_constraints_shrunk": False,
                        "uses_query_or_target_labels": False,
                    }
                    if primitive_reliability is not None
                    else None
                ),
            }
            if args.support_mode == "canonical_support"
            else None
        ),
        "frames": frame_metrics,
        "foreground_iou": float(np.mean([value["foreground_iou"] for value in frame_metrics])),
        "pixel_accuracy": float(np.mean([value["pixel_accuracy"] for value in frame_metrics])),
        "score_paths": score_paths,
        "score_sha256": score_sha256,
        "pre_metric_prediction_receipt": (
            {
                "path": prediction_receipt_path,
                "sha256": prediction_receipt_sha256,
                "sealed_before_target_ground_truth_open": True,
            }
            if prediction_receipt_path is not None
            else None
        ),
        "stage_metrics": stage_metrics,
        "stage_score_paths": stage_score_paths,
        "stage_score_sha256": stage_score_sha256,
        "safety": {
            "target_ground_truth_opened_before_prediction_write": False,
            "target_rgb_opened": False,
            "registered_prompt_rgb_feature_used": (
                args.support_mode == "multiview_score_lift"
                and args.prompt_feature_source == "observed"
            ),
            "target_camera_used_as_support": False,
            "test_calibration": False,
            "reference_query_calibration": bool(
                getattr(args, "query_diffusion_reference_calibration", False)
            )
            or bool(
                getattr(
                    args,
                    "registered_reference_threshold_calibration",
                    False,
                )
            ),
            "reference_query_calibration_uses_target_masks_or_metrics": False,
            "test_calibration_definition": (
                "no target labels, target masks, or metric feedback are used; "
                "unlabeled evaluation-scene statistics are disclosed separately"
            ),
            "official_sam_decoder": False,
            "frozen_source_completion_official_sam3": (
                source_completion_evidence is not None
            ),
            "source_completion_target_rgb_or_mask_opened": (
                False if source_completion_evidence is not None else None
            ),
            "canonical_capability_cache": (
                str(Path(args.canonical_capability_cache).resolve())
                if args.canonical_capability_cache
                else ""
            ),
            "canonical_support_graph": (
                str(Path(args.canonical_support_graph).resolve())
                if args.canonical_support_graph
                else ""
            ),
            "canonical_reliability_cache": (
                str(Path(args.canonical_reliability_cache).resolve())
                if str(args.canonical_reliability_cache).strip()
                else ""
            ),
            "diagnostic_graph_affinity_override": (
                str(Path(args.diagnostic_graph_affinity_override).resolve())
                if str(args.diagnostic_graph_affinity_override).strip()
                else ""
            ),
            "scene_carrier_assets": {
                "config": str(config_path),
                "config_sha256": _file_sha256(config_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "camera_map": str(camera_map_path),
                "camera_map_sha256": _file_sha256(camera_map_path),
                "explicit_override": bool(
                    str(getattr(args, "scene_config", "")).strip()
                ),
            },
            "candidate_eligibility": candidate_eligibility,
            "frozen_diagnostic_eligible": (
                registered_forward_contract is None
                and not bool(
                    str(args.diagnostic_graph_affinity_override).strip()
                )
            ),
            "main_result_eligible": (
                registered_forward_contract is None
                and
                (
                    candidate_run_manifest is None
                    or candidate_eligibility == "main_result_eligible"
                )
                and not bool(
                    str(args.diagnostic_graph_affinity_override).strip()
                )
            ),
            **(
                {
                    "strict_unseen_eligible": False,
                    "strict_unseen_protocol_exact_match": (
                        registered_forward_protocol_authority[
                            "strict_unseen_protocol_exact_match"
                        ]
                    ),
                    "strict_unseen_scoring_binding": (
                        registered_forward_contract[
                            "strict_unseen_scoring_binding"
                        ]
                    ),
                    "registered_forward_protocol_authority_sha256": (
                        registered_forward_protocol_authority_sha256
                    ),
                }
                if registered_forward_contract is not None
                else {}
            ),
            "target_camera_names_excluded_from_support": evaluation_camera_names,
        },
    }
    solver_contract = json.loads(json.dumps(report["shared_solver"]))
    if (
        isinstance(solver_contract, dict)
        and isinstance(solver_contract.get("primitive_reliability"), dict)
    ):
        solver_contract["primitive_reliability"].pop("cache", None)
    signature_checkpoint_hashes = (
        {
            str(signature.radio_checkpoint_sha256)
            for signature in capability_bank.signatures.values()
            if str(signature.radio_checkpoint_sha256).strip()
        }
        if capability_bank is not None
        else set()
    )
    if len(signature_checkpoint_hashes) > 1:
        raise ValueError("canonical capability signatures disagree on RADIO checkpoint")
    radio_checkpoint_sha256 = (
        next(iter(signature_checkpoint_hashes))
        if signature_checkpoint_hashes
        else _file_sha256(Path(args.radio_checkpoint).expanduser().resolve())
    )
    implementation_root = Path(__file__).resolve().parents[2]
    implementation_relatives = (
        "radio_gs/evaluation/promptable_segmentation.py",
        "radio_gs/interfaces/capability_cache.py",
        "radio_gs/interfaces/query_diffusion_cache.py",
        "radio_gs/config.py",
        "radio_gs/data/lerf_dataset.py",
        "radio_gs/models/explicit_gaussian.py",
        "radio_gs/models/featsharp_3d.py",
        "radio_gs/models/hcd_codec.py",
        "radio_gs/models/hybrid_gaussian.py",
        "radio_gs/models/screen_refiner.py",
        "radio_gs/querying/query_spec.py",
        "radio_gs/querying/query_compilers.py",
        "radio_gs/querying/evidence_scorer.py",
        "radio_gs/querying/query_engine.py",
        "radio_gs/querying/query_conditioned_diffusion.py",
        "radio_gs/querying/nvos_local_positive_completion.py",
        "radio_gs/querying/sam3_reference_completion.py",
        "radio_gs/querying/score_calibration.py",
        "radio_gs/querying/support_solver.py",
        "radio_gs/rendering/feature_renderer.py",
        "radio_gs/rendering/contribution_compositor.py",
        "radio_gs/rendering/camera_clearance.py",
        "radio_gs/scripts/eval_lerf_direct_3d_selection.py",
        "radio_gs/scripts/eval_lerf_grounding.py",
        "radio_gs/scripts/render_promptable_nvs_features.py",
        "radio_gs/utils/checkpoint_io.py",
    )
    if registered_forward_protocol_authority is not None:
        implementation_relatives += (
            "radio_gs/scripts/bind_nvos_forward_beta_protocol_authority.py",
            "radio_gs/scripts/bind_evaluation_protocol_freeze.py",
            "radio_gs/scripts/validate_evaluation_protocol_freeze.py",
        )
    method_contract = {
        "schema_version": 2,
        "candidate_id": str(
            getattr(args, "candidate_id", "registered-region-v1")
        ),
        "evaluator": "radio_gs/scripts/eval_nvos_gaussian_first.py",
        "evaluator_sha256": _file_sha256(Path(__file__).resolve()),
        "implementation_sha256": {
            relative: _file_sha256(implementation_root / relative)
            for relative in implementation_relatives
        },
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        "candidate_run_manifest_sha256": candidate_run_manifest_sha256,
        "candidate_method_contract_sha256": (
            candidate_method_contract_sha256
        ),
        "candidate_eligibility": candidate_eligibility,
        **(
            {
                "registered_forward_protocol_authority": (
                    registered_forward_protocol_authority
                ),
                "registered_forward_protocol_authority_sha256": (
                    registered_forward_protocol_authority_sha256
                ),
            }
            if registered_forward_protocol_authority is not None
            else {}
        ),
        "asset_hash_verification_required": bool(
            getattr(args, "require_asset_hashes", False)
        ),
        "scene_carrier_assets": report["safety"]["scene_carrier_assets"],
        "prompt_type": prompt_type,
        "support_mode": str(args.support_mode),
        "region_space": str(args.region_space),
        "prompt_feature_source": str(report["prompt_feature_source"]),
        "prompt_registration": {
            "mode": str(
                getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
            ),
            "scale": float(getattr(args, "prompt_registration_scale", 1.0)),
            "alpha_threshold": float(args.alpha_threshold),
            "depth_tolerance": float(args.depth_tolerance),
            "relative_depth_tolerance": float(args.relative_depth_tolerance),
            "observation_mass_source": (
                registered_prompt_evidence.get("observation_mass_source")
                if registered_prompt_evidence is not None
                else None
            ),
            "observation_confidence_mode": str(
                "exact_labeled_visible_fraction"
                if str(
                    getattr(
                        args,
                        "registered_observation_fusion",
                        "additive",
                    )
                )
                in _EXACT_RASTER_OBSERVATION_FUSIONS
                else getattr(
                    args,
                    "registered_observation_confidence",
                    "relative_joint_max",
                )
            ),
            "observation_mass_scale": float(
                getattr(args, "registered_observation_mass_scale", 1.0)
            ),
            **(
                {
                    "observation_coverage_power": float(
                        getattr(
                            args,
                            "registered_observation_coverage_power",
                            1.0,
                        )
                    )
                }
                if str(
                    getattr(
                        args,
                        "registered_observation_confidence",
                        "relative_joint_max",
                    )
                )
                == "poisson_mass_coverage"
                else {}
            ),
            "observation_constructed_before_capability_filter": (
                bool(
                    registered_prompt_evidence.get(
                        "observation_constructed_before_capability_filter",
                        False,
                    )
                )
                if registered_prompt_evidence is not None
                else False
            ),
        },
        "prototype_count": int(args.prototype_count),
        "prototype_strategy": str(args.prototype_strategy),
        "prompt_support_threshold": float(args.support_threshold),
        "score_render": {
            "resolution_mode": score_resolution_mode,
            "scale": float(getattr(args, "score_render_scale", 1.0)),
            "valid_support_normalization": bool(valid_support is not None),
            "valid_support_coverage_power": float(
                getattr(args, "valid_support_coverage_power", 0.0)
            ),
            "feature_contribution_gamma": float(args.feature_contribution_gamma),
            "camera_clearance": (
                {
                    "contract": CAMERA_PLANE_CLEARANCE_CONTRACT,
                    "support_sigma": camera_clearance_sigma,
                    "query_independent": True,
                }
                if camera_clearance_sigma > 0
                else None
            ),
            "pixel_threshold": float(prediction_threshold),
            "reference_threshold_calibration": (
                registered_reference_threshold_calibration
            ),
            "threshold_comparison": "greater_or_equal",
            "resize_to_ground_truth": (
                "cv2.INTER_NEAREST"
                if registered_forward_protocol_authority is not None
                else "cv2.INTER_LINEAR"
            ),
            **(
                {
                    "scoring_adapter": _registered_forward_scoring_contract(
                        args
                    )
                }
                if registered_forward_protocol_authority is not None
                else {}
            ),
        },
        "shared_solver": solver_contract,
    }
    report["method_contract"] = method_contract
    report["method_config_sha256"] = _json_sha256(method_contract)
    report["run_manifest_sha256"] = candidate_run_manifest_sha256 or None
    dataset_protocol_contract = _dataset_protocol_contract(
        manifest,
        benchmark_manifest_sha256=_file_sha256(manifest_path),
    )
    dataset_protocol_sha256 = _json_sha256(dataset_protocol_contract)
    benchmark_protocol = dict(manifest.get("protocol", {}))
    evaluation_protocol_contract = {
        "schema_version": 1,
        "dataset_protocol_sha256": dataset_protocol_sha256,
        "method_config_sha256": report["method_config_sha256"],
        "prediction_representation": (
            _registered_forward_scoring_contract(args)[
                "prediction_representation"
            ]
            if registered_forward_protocol_authority is not None
            else (
                "coverage_weighted_foreground_posterior"
                if bool(valid_support is not None)
                else "alpha_normalized_foreground_posterior"
            )
        ),
        "score_domain": (
            [-1.0, 1.0]
            if registered_forward_protocol_authority is not None
            else [0.0, 1.0]
        ),
        "final_readout": str(
            getattr(args, "registered_readout_stage", "connected")
        ),
        "scalar_compositor": {
            "alpha_normalized": True,
            "valid_support_normalization": bool(valid_support is not None),
            "valid_support_coverage_power": float(
                getattr(args, "valid_support_coverage_power", 0.0)
            ),
            "feature_contribution_gamma": float(
                args.feature_contribution_gamma
            ),
            "camera_clearance": (
                {
                    "contract": CAMERA_PLANE_CLEARANCE_CONTRACT,
                    "support_sigma": camera_clearance_sigma,
                }
                if camera_clearance_sigma > 0
                else None
            ),
        },
        "resize_to_ground_truth": (
            "cv2.INTER_NEAREST"
            if registered_forward_protocol_authority is not None
            else "cv2.INTER_LINEAR"
        ),
        "pixel_threshold": {
            "value": float(prediction_threshold),
            "comparison": "greater_or_equal",
        },
        "metrics": list(
            benchmark_protocol.get(
                "metrics", ["foreground_iou", "pixel_accuracy"]
            )
        ),
        "within_scene_aggregation": str(
            benchmark_protocol.get(
                "within_scene_aggregation", "frame_macro"
            )
        ),
        "dataset_aggregation": str(
            benchmark_protocol.get("aggregation", "scene_macro")
        ),
        "empty_union_value": float(
            benchmark_protocol.get("empty_union_value", 1.0)
        ),
        **(
            {
                "score_semantics": _registered_forward_scoring_contract(args)[
                    "score_semantics"
                ],
                "registered_forward_protocol_authority_sha256": (
                    registered_forward_protocol_authority_sha256
                ),
                "strict_unseen_protocol_exact_match": (
                    registered_forward_protocol_authority[
                        "strict_unseen_protocol_exact_match"
                    ]
                ),
            }
            if registered_forward_protocol_authority is not None
            else {}
        ),
    }
    report["legacy_protocol_hash"] = manifest["protocol_hash"]
    report["dataset_protocol_contract"] = dataset_protocol_contract
    report["dataset_protocol_sha256"] = dataset_protocol_sha256
    report["evaluation_protocol_contract"] = evaluation_protocol_contract
    report["evaluation_protocol_sha256"] = _json_sha256(
        evaluation_protocol_contract
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"{args.scene_id}_evaluation.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    temporary_report.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-config",
        default="",
        help="Explicit frozen carrier config; requires the other two carrier overrides.",
    )
    parser.add_argument(
        "--scene-checkpoint",
        default="",
        help="Explicit frozen geometry carrier checkpoint.",
    )
    parser.add_argument(
        "--camera-map",
        default="",
        help="Explicit frozen RGB-to-COLMAP camera map.",
    )
    parser.add_argument(
        "--run-manifest",
        default="",
        help="Optional immutable candidate run manifest bound into the report.",
    )
    parser.add_argument(
        "--candidate-id",
        default="registered-region-v1",
        help="Candidate identifier required to match the immutable run manifest.",
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-attestation-output", default="")
    parser.add_argument(
        "--expected-gpu-physical-index",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Physical GPU0/GPU1 selected by UUID visibility; required only "
            "for registered forward-Beta attestation."
        ),
    )
    parser.add_argument("--expected-gpu-uuid", default="")
    parser.add_argument("--expected-gpu-bus-id", default="")
    parser.add_argument("--region-space", choices=["radio", "sam3"], default="sam3")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--support-threshold", type=float, default=0.0)
    parser.add_argument("--prototype-count", type=int, default=1)
    parser.add_argument(
        "--prototype-strategy",
        choices=("weighted_fps", "spherical_mean_fps"),
        default="spherical_mean_fps",
    )
    parser.add_argument(
        "--prompt-feature-source",
        choices=("observed", "rendered"),
        default="observed",
        help="Use the registered real query feature (default) or a rendered diagnostic.",
    )
    parser.add_argument(
        "--support-mode",
        choices=("prompt_gaussian", "multiview_score_lift", "canonical_support"),
        default="prompt_gaussian",
    )
    parser.add_argument("--canonical-capability-cache", default="")
    parser.add_argument(
        "--canonical-capability-projection-authority",
        default=(
            "paper/artifacts/"
            "formal_capability_projection_lineage_closure_20260805.json"
        ),
        help=(
            "Exact path/sidecar/field-bound authority for pre-contract compact "
            "capability caches. New caches must carry an inline formal contract."
        ),
    )
    parser.add_argument("--canonical-support-graph", default="")
    parser.add_argument("--canonical-reliability-cache", default="")
    parser.add_argument(
        "--canonical-capability-source-contract",
        choices=("field", "exact_mpr", "exact_capability_mpr"),
        default="field",
        help=(
            "Provenance contract for the capability rows. exact_mpr is "
            "diagnostic-only and must be combined with unary-only graph disable."
        ),
    )
    parser.add_argument(
        "--disable-registered-graph",
        action="store_true",
        help=(
            "Use a zero-edge graph for a strictly unary-only same-compiler "
            "diagnostic; requires unary_prior and forbids connected/diffusion."
        ),
    )
    parser.add_argument(
        "--primitive-unary-output",
        default="",
        help="Optional pre-GT dense primitive unary probability artifact.",
    )
    parser.add_argument(
        "--source-observation-oof-output-dir",
        default="",
        help=(
            "Source-only three-fold evidence gate directory. One invocation "
            "seals the requested fold and exits before target rendering; the "
            "third fold also seals the immutable gate receipt."
        ),
    )
    parser.add_argument(
        "--source-observation-oof-heldout-fold",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Held-out SplitMix64 fold for a source-observation gate invocation.",
    )
    parser.add_argument(
        "--source-observation-oof-fold-mode",
        choices=("stable_primitive_rows_v1", "source_footprint_v1"),
        default="stable_primitive_rows_v1",
        help=(
            "Explicit OOF fold authority. The default preserves the frozen "
            "global-row SplitMix64 protocol; source_footprint_v1 holds out "
            "complete query-free source-raster footprint groups."
        ),
    )
    parser.add_argument(
        "--source-footprint-fold-authority",
        default="",
        help="Immutable source-footprint fold authority artifact.",
    )
    parser.add_argument(
        "--source-footprint-fold-authority-file-sha256",
        default="",
        help="Expected SHA-256 of the complete source-footprint artifact file.",
    )
    parser.add_argument(
        "--source-footprint-fold-authority-sha256",
        default="",
        help="Expected canonical authority contract SHA-256 stored in the artifact.",
    )
    parser.add_argument(
        "--source-observation-oof-gate-receipt",
        default="",
        help=(
            "Immutable source-only OOF gate selected before target rendering; "
            "the deployed readout stage must match its selected action."
        ),
    )
    parser.add_argument(
        "--source-observation-oof-gate-receipt-sha256",
        default="",
        help="Expected complete-file SHA-256 of the source-only OOF gate.",
    )
    parser.add_argument(
        "--source-completion-unary",
        choices=("none", _PROBABILITY_PRESERVING_SOURCE_UNARY),
        default="none",
        help=(
            "Optional source/reference-only SAM3 completion fused strictly in "
            "Bernoulli probability space with entropy reliability kept separate."
        ),
    )
    parser.add_argument("--source-completion", default="")
    parser.add_argument("--source-completion-sha256", default="")
    parser.add_argument("--source-completion-receipt", default="")
    parser.add_argument("--source-completion-receipt-sha256", default="")
    parser.add_argument(
        "--source-completion-calibration",
        choices=(
            "none",
            _SOURCE_COMPLETION_LOO_CALIBRATION,
            _SOURCE_COMPLETION_HIERARCHICAL_LOCAL_POSITIVE_CALIBRATION,
        ),
        default="none",
        help=(
            "Optional immutable source-only leave-one-trial consistency gate. "
            "The v1 mode fully abstains on rejection; the hierarchical v2 "
            "mode retains only strict-majority local positive evidence."
        ),
    )
    parser.add_argument("--source-completion-calibration-gate", default="")
    parser.add_argument(
        "--source-completion-calibration-gate-sha256", default=""
    )
    parser.add_argument(
        "--prediction-receipt-output",
        default="",
        help=(
            "Optional immutable JSON receipt binding every rendered target "
            "score and graph/capability SHA before target-mask metrics. "
            "Requires --require-asset-hashes; graph-disabled unary runs bind "
            "an explicit zero-edge policy instead of a graph file."
        ),
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help=(
            "Stop immediately after sealing all target scores, the primitive "
            "unary, and the pre-metric receipt; do not open target ground truth."
        ),
    )
    parser.add_argument(
        "--query-conditioned-diffusion-kernel",
        choices=(
            "none",
            "ludvig_release_compat",
            "symmetric_normalized",
            "continuous_convex_v2",
        ),
        default="none",
        help=(
            "Optional reference-only Evidence-to-Support readout. The release "
            "compatibility kernel preserves LUDVIG's actual slotwise normalization."
        ),
    )
    parser.add_argument("--query-diffusion-knn-cache", default="")
    parser.add_argument("--query-diffusion-feature-cache", default="")
    parser.add_argument(
        "--query-diffusion-reference-calibration",
        action="store_true",
        help=(
            "Select the preregistered SPIn feature/regularizer bandwidth grid "
            "and final rendered threshold using only the declared reference mask."
        ),
    )
    parser.add_argument(
        "--query-diffusion-reference-threshold-source",
        choices=("propagated", "query_compatibility"),
        default="propagated",
        help=(
            "Reference-only score domain used to select the rendered scalar "
            "threshold; propagated preserves existing runs."
        ),
    )
    parser.add_argument(
        "--query-diffusion-reference-calibration-only",
        action="store_true",
        help=(
            "Persist a reference-only threshold receipt and exit before any "
            "target score rendering or target-mask access."
        ),
    )
    parser.add_argument("--query-diffusion-feature-bandwidth", type=float, default=1.0)
    parser.add_argument(
        "--query-diffusion-regularizer-bandwidth", type=float, default=1.0
    )
    parser.add_argument("--query-diffusion-logistic-c", type=float, default=0.01)
    parser.add_argument("--query-diffusion-iterations", type=int, default=100)
    parser.add_argument(
        "--query-diffusion-edge-binarize-threshold", type=float, default=1e-5
    )
    parser.add_argument(
        "--query-diffusion-max-positive-fraction", type=float, default=0.1
    )
    parser.add_argument("--query-diffusion-distance-chunk-size", type=int, default=32)
    parser.add_argument(
        "--prompt-registration-mode",
        choices=("legacy_alpha_depth", "raster_adjoint"),
        default="legacy_alpha_depth",
        help=(
            "Use the frozen footprint/depth proxy or the exact front-to-back "
            "compositing adjoint for registered prompt lifting."
        ),
    )
    parser.add_argument(
        "--prompt-registration-scale",
        type=float,
        default=1.0,
        help=(
            "Raster scale relative to the native prompt; used only by "
            "--prompt-registration-mode raster_adjoint."
        ),
    )
    parser.add_argument(
        "--score-render-scale",
        type=float,
        default=1.0,
        help="Scalar score raster scale relative to the frozen feature raster.",
    )
    parser.add_argument(
        "--score-render-resolution",
        choices=("scaled_renderer", "prompt_native"),
        default="scaled_renderer",
        help=(
            "Render at a scaled frozen-renderer resolution or at the observable "
            "native prompt resolution (without opening target RGB or masks)."
        ),
    )
    parser.add_argument(
        "--valid-support-normalization",
        action="store_true",
        help=(
            "For canonical support, render sum(w*v*p)/sum(w*v) so invalid "
            "capability rows abstain instead of diluting scalar scores."
        ),
    )
    parser.add_argument(
        "--valid-support-coverage-power",
        type=float,
        default=0.0,
        help=(
            "Query-independent abstention after valid normalization. Zero is "
            "pure conditional scoring and one exactly recovers total-alpha "
            "dilution; intermediate values trade score purity for coverage."
        ),
    )
    parser.add_argument(
        "--feature-contribution-gamma",
        type=float,
        default=1.0,
        help=(
            "Query-independent exponent for normalized front-to-back feature "
            "mixture weights; 1 is ordinary alpha averaging."
        ),
    )
    parser.add_argument(
        "--camera-clearance-sigma",
        type=float,
        default=0.0,
        help=(
            "Query-independent camera-plane visibility guard. A positive value "
            "rejects Gaussian rows whose corresponding axial support bound "
            "intersects the renderer near plane; 2.0 is the physical two-sigma rule."
        ),
    )
    parser.add_argument(
        "--diagnostic-graph-affinity-override",
        default="",
        help=(
            "Diagnostic only: replace edge affinities with an exact-topology graph "
            "from another canonical field; reported results are not main-table eligible."
        ),
    )
    parser.add_argument(
        "--graph-policy",
        choices=(
            "legacy",
            "typed",
            "geometry",
            "appearance",
            "boundary",
            "category_mix",
            "instance_mix",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--component-graph-policy",
        choices=(
            "same",
            "legacy",
            "typed",
            "geometry",
            "appearance",
            "boundary",
            "category_mix",
            "instance_mix",
        ),
        default="same",
    )
    parser.add_argument("--graph-legacy-residual", type=float, default=0.0)
    parser.add_argument(
        "--channel-confidence-mode",
        choices=("none", "affinity_mass", "max_affinity"),
        default="none",
        help=(
            "optional label-free capability abstention; confidence modes keep "
            "unary evidence through a self loop when all neighbour relations are weak"
        ),
    )
    parser.add_argument(
        "--negative-spatial-mode",
        choices=("none", "truncated_graph_decay", "signed_geodesic"),
        default="none",
    )
    parser.add_argument("--negative-spatial-steps", type=int, default=4)
    parser.add_argument("--negative-spatial-decay", type=float, default=0.8)
    parser.add_argument("--canonical-field-sha256", default="")
    parser.add_argument(
        "--registered-seed-unary-weight",
        type=float,
        default=0.0,
        help=(
            "Direct signed unary weight for raster-registered positive/negative "
            "primitive responsibilities; zero preserves the frozen protocol."
        ),
    )
    parser.add_argument(
        "--registered-observation-fusion",
        choices=(
            "additive",
            "probability_mixture",
            "hard_seed_anchored_probability",
            "hard_seed_anchor_only_probability",
            "direct_raster_adjoint",
            "raster_adjoint_bernoulli_poe",
            "dual_registration_bernoulli_poe",
        ),
        default="additive",
        help=(
            "Fuse registered mass by the historical additive unary or as the "
            "label-free probability mixture (1-c)*p_field+c*q, or make "
            "the same direct rows accepted as solver hard seeds unit-confidence "
            "unary anchors, or apply those anchors only while preserving every "
            "non-anchor field unary bit-for-bit. direct_raster_adjoint instead "
            "bypasses prototype "
            "cosine and uses the exact shared-responsibility foreground/background "
            "continuous primitive unary. raster_adjoint_bernoulli_poe keeps the "
            "prototype unary and pools its Bernoulli posterior with that same "
            "exact-adjoint posterior by a symmetric normalized product. "
            "dual_registration_bernoulli_poe instead reconstructs the prototype "
            "expert with the frozen legacy alpha/depth operator and keeps native "
            "exact adjoint as an independent expert."
        ),
    )
    parser.add_argument(
        "--registered-observation-confidence",
        choices=(
            "relative_joint_max",
            "poisson_mass",
            "poisson_mass_coverage",
        ),
        default="relative_joint_max",
        help=(
            "Normalize generic prompt mass by its joint maximum or interpret "
            "raw raster-adjoint mass as Poisson observation confidence; the "
            "coverage variant additionally requires labeled footprint support. "
            "With a forward-Beta mode this controls seed construction only."
        ),
    )
    parser.add_argument(
        "--registered-forward-unary",
        choices=("none", "beta_coverage_v1", "beta_balanced_residual_v2"),
        default="none",
        help=(
            "Diagnostic new-method unary from an exact registered-view forward "
            "likelihood E-step. It is authority-bound but non-exact for frozen "
            "strict-unseen scoring; none preserves the historical evaluator path."
        ),
    )
    parser.add_argument(
        "--registered-observation-mass-scale",
        type=float,
        default=1.0,
        help=(
            "Effective alpha-weighted pixel mass for Poisson observation "
            "confidence; one is the fixed native-raster unit."
        ),
    )
    parser.add_argument(
        "--registered-observation-coverage-power",
        type=float,
        default=1.0,
        help=(
            "Power applied to labeled/visible footprint coverage for "
            "poisson_mass_coverage confidence."
        ),
    )
    parser.add_argument(
        "--registered-seed-construction",
        choices=("winner_take_all", "joint_signed"),
        default="winner_take_all",
        help=(
            "Historical per-sign winner-take-all seeds or joint signed masses "
            "relu(m_pos-m_neg)/relu(m_neg-m_pos), which leave ties neutral."
        ),
    )
    parser.add_argument(
        "--registered-prototype-seed-construction",
        choices=("shared", "winner_take_all"),
        default="shared",
        help=(
            "Use the solver seed masses for prototype construction (shared), "
            "or preserve independent winner-take-all prototype coverage while "
            "the exact joint-signed path supplies solver seeds and unary anchors."
        ),
    )
    parser.add_argument(
        "--registered-selection-mode",
        choices=(
            SelectionMode.SEEDED_COMPONENT.value,
            SelectionMode.ALL_COMPONENTS.value,
        ),
        default=SelectionMode.SEEDED_COMPONENT.value,
        help=(
            "Read out only seed-connected active support (frozen behavior) or "
            "retain every active component for full-region prompts."
        ),
    )
    parser.add_argument(
        "--registered-readout-stage",
        choices=("unary_prior", "propagated", "connected"),
        default="connected",
        help=(
            "Choose the continuous unary/graph field or the component-masked "
            "support as the final scalar render; all stages remain reported."
        ),
    )
    parser.add_argument(
        "--registered-reference-threshold-calibration",
        action="store_true",
        help=(
            "For a declared full reference mask and unary_prior readout, "
            "select the rendered probability threshold on that reference view "
            "using the existing release-compatible 0.99..0.03 grid before "
            "opening any target mask."
        ),
    )
    parser.add_argument(
        "--export-registered-prompt-cycle-diagnostic",
        action="store_true",
        help=(
            "Before any target mask is opened, reproject the independently "
            "registered prototype and exact experts into the reference prompt "
            "view and persist fixed BCE/soft-IoU reliability diagnostics."
        ),
    )
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.35)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument(
        "--feature-calibration",
        choices=("none", "diagonal_robust"),
        default="none",
    )
    parser.add_argument("--background-centroids", type=int, default=0)
    parser.add_argument("--calibration-sample-size", type=int, default=8192)
    parser.add_argument("--centroid-iterations", type=int, default=4)
    parser.add_argument(
        "--score-calibration",
        choices=("none", "robust_tanh", "robust_tanh_centered", "robust_tanh_zero"),
        default="none",
    )
    parser.add_argument("--score-tanh-scale", type=float, default=2.0)
    parser.add_argument("--score-chunk-size", type=int, default=65536)
    parser.add_argument("--solver-iterations", type=int, default=12)
    parser.add_argument("--solver-residual", type=float, default=0.30)
    parser.add_argument(
        "--solver-type", choices=("diffusion", "random_walker", "confidence_random_walker"), default="diffusion"
    )
    parser.add_argument("--laplacian-weight", type=float, default=1.0)
    parser.add_argument("--cg-iterations", type=int, default=64)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--hard-seed-threshold", type=float, default=0.20)
    parser.add_argument(
        "--hard-seed-conflict-policy",
        choices=("positive_priority", "exclusive_relative"),
        default="positive_priority",
    )
    parser.add_argument("--hard-seed-conflict-margin", type=float, default=0.0)
    parser.add_argument("--component-edge-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--seeded-component-min-weight", type=float, default=0.20
    )
    parser.add_argument("--solver-unary-temperature", type=float, default=0.10)
    parser.add_argument("--solver-support-threshold", type=float, default=0.50)
    parser.add_argument(
        "--require-asset-hashes",
        action="store_true",
        help=(
            "Verify prompt assets before prediction and target masks only "
            "after every prediction has been persisted."
        ),
    )
    args = parser.parse_args()
    source_oof_enabled = bool(
        str(args.source_observation_oof_output_dir).strip()
    )
    if source_oof_enabled != (args.source_observation_oof_heldout_fold is not None):
        parser.error(
            "--source-observation-oof-output-dir and "
            "--source-observation-oof-heldout-fold are required together"
        )
    footprint_values = (
        str(args.source_footprint_fold_authority).strip(),
        str(args.source_footprint_fold_authority_file_sha256).strip(),
        str(args.source_footprint_fold_authority_sha256).strip(),
    )
    if args.source_observation_oof_fold_mode == "stable_primitive_rows_v1":
        if any(footprint_values):
            parser.error(
                "stable_primitive_rows_v1 forbids source-footprint authority inputs"
            )
    elif not source_oof_enabled or not all(footprint_values):
        parser.error(
            "source_footprint_v1 requires OOF output/fold plus authority path, "
            "file SHA-256, and authority SHA-256"
        )
    if source_oof_enabled and str(args.prediction_receipt_output).strip():
        parser.error(
            "source-observation OOF exits before target prediction receipts"
        )
    deployment_gate_values = (
        str(args.source_observation_oof_gate_receipt).strip(),
        str(args.source_observation_oof_gate_receipt_sha256).strip(),
    )
    if bool(deployment_gate_values[0]) != bool(deployment_gate_values[1]):
        parser.error(
            "source-observation OOF deployment requires gate receipt path and SHA-256"
        )
    if source_oof_enabled and any(deployment_gate_values):
        parser.error(
            "source-observation OOF fold generation and deployment are separate invocations"
        )
    if any(deployment_gate_values) and not str(args.prediction_receipt_output).strip():
        parser.error(
            "source-observation OOF deployment requires --prediction-receipt-output"
        )
    if str(args.prediction_receipt_output).strip():
        receipt_requirements = {
            "--support-mode canonical_support": (
                str(args.support_mode) == "canonical_support"
            ),
            "--canonical-capability-cache": bool(
                str(args.canonical_capability_cache).strip()
            ),
            "--require-asset-hashes": bool(args.require_asset_hashes),
            "graph asset or explicit disable": (
                bool(str(args.canonical_support_graph).strip())
                != bool(args.disable_registered_graph)
            ),
        }
        failed = [
            name for name, satisfied in receipt_requirements.items() if not satisfied
        ]
        if failed:
            parser.error(
                "--prediction-receipt-output requires " + ", ".join(failed)
            )
    if bool(args.prediction_only):
        prediction_only_requirements = {
            "--prediction-receipt-output": bool(
                str(args.prediction_receipt_output).strip()
            ),
            "--primitive-unary-output": bool(
                str(args.primitive_unary_output).strip()
            ),
            "--require-asset-hashes": bool(args.require_asset_hashes),
        }
        failed = [
            name
            for name, satisfied in prediction_only_requirements.items()
            if not satisfied
        ]
        if failed:
            parser.error("--prediction-only requires " + ", ".join(failed))
    if bool(args.disable_registered_graph):
        disabled_graph_requirements = {
            "--support-mode canonical_support": (
                str(args.support_mode) == "canonical_support"
            ),
            "--registered-readout-stage unary_prior": (
                str(args.registered_readout_stage) == "unary_prior"
            ),
            "--query-conditioned-diffusion-kernel none": (
                str(args.query_conditioned_diffusion_kernel) == "none"
            ),
            "--negative-spatial-mode none": (
                str(args.negative_spatial_mode) == "none"
            ),
            "--registered-forward-unary none": (
                str(args.registered_forward_unary) == "none"
            ),
            "no --canonical-support-graph": (
                not str(args.canonical_support_graph).strip()
            ),
        }
        failed = [
            name
            for name, satisfied in disabled_graph_requirements.items()
            if not satisfied
        ]
        if failed:
            parser.error("--disable-registered-graph requires " + ", ".join(failed))
    if args.canonical_capability_source_contract in {
        "exact_mpr",
        "exact_capability_mpr",
    }:
        exact_requirements = {
            "--disable-registered-graph": bool(args.disable_registered_graph),
            "empty --canonical-field-sha256": not str(
                args.canonical_field_sha256
            ).strip(),
            "--primitive-unary-output": bool(
                str(args.primitive_unary_output).strip()
            ),
        }
        failed = [
            name for name, satisfied in exact_requirements.items() if not satisfied
        ]
        if failed:
            parser.error(
                "non-field --canonical-capability-source-contract requires "
                + ", ".join(failed)
            )
    try:
        _validate_registered_forward_unary_args(args)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.feature_contribution_gamma) or args.feature_contribution_gamma <= 0:
        parser.error("--feature-contribution-gamma must be finite and positive")
    if not np.isfinite(args.camera_clearance_sigma) or args.camera_clearance_sigma < 0:
        parser.error("--camera-clearance-sigma must be finite and non-negative")
    if (
        not np.isfinite(args.prompt_registration_scale)
        or args.prompt_registration_scale <= 0
    ):
        parser.error("--prompt-registration-scale must be finite and positive")
    if not np.isfinite(args.score_render_scale) or args.score_render_scale <= 0:
        parser.error("--score-render-scale must be finite and positive")
    if (
        args.prompt_registration_mode == "legacy_alpha_depth"
        and args.prompt_registration_scale != 1.0
    ):
        parser.error(
            "--prompt-registration-scale applies only to raster_adjoint mode"
        )
    if args.valid_support_normalization and args.support_mode != "canonical_support":
        parser.error(
            "--valid-support-normalization requires --support-mode canonical_support"
        )
    query_diffusion_enabled = args.query_conditioned_diffusion_kernel != "none"
    if query_diffusion_enabled:
        if args.support_mode != "canonical_support":
            parser.error("query-conditioned diffusion requires canonical_support")
        if args.prompt_registration_mode != "raster_adjoint":
            parser.error("query-conditioned diffusion requires exact raster_adjoint")
        if args.registered_observation_fusion != "direct_raster_adjoint":
            parser.error(
                "query-conditioned diffusion requires direct_raster_adjoint observation"
            )
        if not args.query_diffusion_knn_cache or not args.query_diffusion_feature_cache:
            parser.error("query-conditioned diffusion requires kNN and feature caches")
        if args.registered_readout_stage not in {"propagated", "connected"}:
            parser.error("query-conditioned diffusion requires propagated/connected readout")
        if args.query_conditioned_diffusion_kernel == "continuous_convex_v2":
            if not args.query_diffusion_reference_calibration:
                parser.error(
                    "continuous_convex_v2 requires reference-only threshold calibration"
                )
            if args.registered_readout_stage != "propagated":
                parser.error(
                    "continuous_convex_v2 requires propagated all-component readout"
                )
            if float(args.query_diffusion_logistic_c) != 0.01:
                parser.error("continuous_convex_v2 freezes query logistic C at 0.01")
        if args.query_diffusion_reference_calibration_only:
            if args.query_conditioned_diffusion_kernel != "continuous_convex_v2":
                parser.error(
                    "reference-calibration-only is registered only for continuous_convex_v2"
                )
            if args.query_diffusion_reference_threshold_source != (
                "query_compatibility"
            ):
                parser.error(
                    "reference-calibration-only requires query_compatibility threshold source"
                )
        if any(
            not np.isfinite(value) or value <= 0
            for value in (
                args.query_diffusion_feature_bandwidth,
                args.query_diffusion_regularizer_bandwidth,
                args.query_diffusion_logistic_c,
                args.query_diffusion_edge_binarize_threshold,
            )
        ):
            parser.error("query-conditioned diffusion scales must be finite and positive")
        if args.query_diffusion_iterations <= 0 or args.query_diffusion_distance_chunk_size <= 0:
            parser.error("query-conditioned diffusion iterations/chunk must be positive")
        if not 0 < args.query_diffusion_max_positive_fraction <= 1:
            parser.error("query-diffusion max-positive-fraction must be in (0,1]")
    if (
        not np.isfinite(args.valid_support_coverage_power)
        or args.valid_support_coverage_power < 0
    ):
        parser.error(
            "--valid-support-coverage-power must be finite and non-negative"
        )
    if (
        args.valid_support_coverage_power != 0
        and not args.valid_support_normalization
    ):
        parser.error(
            "--valid-support-coverage-power requires --valid-support-normalization"
        )
    if (
        not np.isfinite(args.registered_seed_unary_weight)
        or args.registered_seed_unary_weight < 0
    ):
        parser.error("--registered-seed-unary-weight must be finite and non-negative")
    if (
        args.registered_observation_fusion
        in {
            "probability_mixture",
            "hard_seed_anchored_probability",
            "hard_seed_anchor_only_probability",
            "direct_raster_adjoint",
            "raster_adjoint_bernoulli_poe",
            "dual_registration_bernoulli_poe",
        }
        and args.registered_seed_unary_weight != 0
    ):
        parser.error(
            "non-additive registered-observation fusion requires "
            "--registered-seed-unary-weight 0"
        )
    try:
        _validate_direct_raster_adjoint_args(args)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.export_registered_prompt_cycle_diagnostic
        and args.registered_observation_fusion
        != "dual_registration_bernoulli_poe"
    ):
        parser.error(
            "--export-registered-prompt-cycle-diagnostic requires "
            "--registered-observation-fusion dual_registration_bernoulli_poe"
        )
    try:
        _validate_hard_seed_anchor_only_probability_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        _validate_registered_prototype_seed_construction_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        _validate_registered_reference_threshold_calibration_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.registered_observation_fusion in {
        "hard_seed_anchored_probability",
        "hard_seed_anchor_only_probability",
    }:
        if args.registered_seed_construction != "joint_signed":
            parser.error(
                "hard-seed probability fusion requires "
                "--registered-seed-construction joint_signed"
            )
        if args.hard_seed_threshold <= 0 or args.hard_seed_conflict_margin != 0:
            parser.error(
                "hard-seed probability fusion requires a positive "
                "--hard-seed-threshold and --hard-seed-conflict-margin 0"
            )
    if (
        not np.isfinite(args.registered_observation_mass_scale)
        or args.registered_observation_mass_scale <= 0
    ):
        parser.error(
            "--registered-observation-mass-scale must be finite and positive"
        )
    if (
        not np.isfinite(args.registered_observation_coverage_power)
        or args.registered_observation_coverage_power <= 0
    ):
        parser.error(
            "--registered-observation-coverage-power must be finite and positive"
        )
    if (
        args.registered_observation_confidence
        in {"poisson_mass", "poisson_mass_coverage"}
        and args.prompt_registration_mode != "raster_adjoint"
    ):
        parser.error(
            "Poisson registered observation confidence requires "
            "--prompt-registration-mode raster_adjoint"
        )
    if not 0 <= args.hard_seed_threshold <= 1:
        parser.error("--hard-seed-threshold must be in [0,1]")
    if (
        not np.isfinite(args.hard_seed_conflict_margin)
        or args.hard_seed_conflict_margin < 0
        or not np.isfinite(args.component_edge_threshold)
        or args.component_edge_threshold < 0
        or not 0 <= args.seeded_component_min_weight <= 1
    ):
        parser.error("registered hard-seed/component parameters are invalid")
    attestation_values = (
        args.gpu_attestation_output,
        args.expected_gpu_uuid,
        args.expected_gpu_bus_id,
    )
    beta_forward = args.registered_forward_unary in {
        "beta_coverage_v1",
        "beta_balanced_residual_v2",
    }
    if beta_forward:
        if not all(attestation_values) or args.expected_gpu_physical_index is None:
            parser.error(
                "registered forward Beta requires GPU attestation output, physical "
                "index, expected UUID, and expected PCI bus"
            )
        if args.device != "cuda:0":
            parser.error("GPU-attested NVOS evaluation requires --device cuda:0")
        from radio_gs.scripts.nvos_forward_beta_scene_authority import (
            write_forward_beta_cuda_child_attestation,
        )

        write_forward_beta_cuda_child_attestation(
            output=args.gpu_attestation_output,
            scene=args.scene_id,
            physical_index=args.expected_gpu_physical_index,
            expected_uuid=args.expected_gpu_uuid,
            expected_bus_id=args.expected_gpu_bus_id,
        )
    else:
        if args.expected_gpu_physical_index is not None:
            parser.error(
                "--expected-gpu-physical-index applies only to registered forward Beta"
            )
        if any(attestation_values) and not all(attestation_values):
            parser.error(
                "GPU attestation output, expected UUID, and expected PCI bus must "
                "be provided together"
            )
        if all(attestation_values):
            if args.device != "cuda:0":
                parser.error("GPU-attested NVOS evaluation requires --device cuda:0")
            write_cuda_child_attestation(
                output=args.gpu_attestation_output,
                scene=args.scene_id,
                expected_uuid=args.expected_gpu_uuid,
                expected_bus_id=args.expected_gpu_bus_id,
            )
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
