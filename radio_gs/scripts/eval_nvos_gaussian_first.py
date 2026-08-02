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
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.evidence_scorer import (
    EvidenceScoringConfig,
    RegisteredForwardBetaDiagnostics,
    registered_forward_beta_balanced_residual_observation,
    registered_forward_beta_observation,
    registered_observation_anchor_mask,
    registered_observation_effective_confidence,
    registered_seed_observation,
)
from radio_gs.querying.query_compilers import compile_registered_primitive_seeds
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_spec import PrimitiveUnaryEvidence, SelectionMode
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
    prompt_hashes = dict(protocol.get("prompt_asset_sha256", {}))
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
        declared_prompt_hashes = dict(prompt_hashes.get(scene_id, {}))
        if not declared_prompt_hashes:
            raise ValueError(f"{scene_id}: prompt asset hashes are undeclared")
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
        "registered_seed_unary_weight": float(
            getattr(args, "registered_seed_unary_weight", 0.0)
        ),
        **(
            {
                "strong_unary": {
                    "policy": "unit_confidence_on_shared_hard_seed_rows",
                    "anchor_threshold_source": "solver.hard_seed_threshold",
                    "anchor_threshold": float(
                        getattr(args, "hard_seed_threshold", 0.20)
                    ),
                    "formula": (
                        "a=1[c>0 and abs(s)>=tau]; c_eff=a+(1-a)c; "
                        "p=(1-c_eff)p_field+c_eff*q"
                    ),
                    "new_numeric_constant": False,
                }
            }
            if observation_fusion == "hard_seed_anchored_probability"
            else {}
        ),
        "observation_mass_source": (
            "raw_raster_adjoint_prompt_mass_times_labeled_footprint_coverage"
            if observation_mode == "poisson_mass_coverage"
            else (
                "raw_raster_adjoint_prompt_mass"
                if observation_mode == "poisson_mass"
                else "conditional_labeled_footprint_fraction"
            )
        ),
        "observation_confidence": observation_mode,
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
    base_scene_id = str(scene.get("base_scene_id") or args.scene_id)
    scene_root = Path(args.queue_root).resolve() / "scenes"
    queue_scene = scene_root / args.scene_id
    if not queue_scene.is_dir():
        queue_scene = scene_root / base_scene_id
    config_path = queue_scene / "gaussfm_main_track.yaml"
    checkpoint_path = queue_scene / "feature_field" / "checkpoints" / "best.pth"
    camera_map_path = queue_scene / "rgb_to_colmap_camera_mapping.json"
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
    if not is_hybrid:
        raise ValueError("Gaussian-first NVOS currently requires a hybrid field")
    adaptor = None
    if args.region_space == "sam3" and args.support_mode != "canonical_support":
        adaptor = load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "sam3", kind="feature_projection"
        ).to(device).eval().requires_grad_(False)
    region_rows = None
    capability_bank = None
    support_graph = None
    primitive_reliability = None
    if args.support_mode == "canonical_support":
        if not args.canonical_capability_cache or not args.canonical_support_graph:
            raise ValueError(
                "canonical_support requires --canonical-capability-cache and "
                "--canonical-support-graph"
            )
        capability_bank = load_canonical_capability_bank(
            args.canonical_capability_cache,
            expected_field_checkpoint_sha256=args.canonical_field_sha256,
        )
        support_graph = load_canonical_support_graph(
            args.canonical_support_graph, capability_bank
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
    if args.support_mode == "prompt_gaussian":
        region_rows = decode_region_rows(
            model, codec, adaptor, device=device, chunk_size=max(1, args.chunk_size)
        )

    prompt = scene["prompt"]
    prompt_type = str(prompt.get("type", ""))
    declared_prompt_hashes = dict(
        manifest.get("protocol", {})
        .get("prompt_asset_sha256", {})
        .get(args.scene_id, {})
    )
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
                    "mask", declared_prompt_hashes.get("positive")
                ),
                label=f"{args.scene_id} reference mask",
            )
        positive_native = load_ground_truth_mask(prompt["mask_path"]).astype(bool)
        negative_native = np.logical_not(positive_native)
    else:
        raise ValueError(f"Unsupported registered prompt type: {prompt_type!r}")
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
    support_view_count = 1
    prediction_threshold = 0.0
    canonical_stage_gaussian_scores: dict[str, torch.Tensor] | None = None
    registered_prompt_evidence: dict[str, object] | None = None
    if args.support_mode in {"prompt_gaussian", "canonical_support"}:
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
                )["alpha_map"]
            support_sum, support_count = raster_adjoint_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=prompt_pose,
                siglip_feat=prompt_maps,
                alpha_map=prompt_alpha,
                alpha_threshold=args.alpha_threshold,
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
        if observation_confidence_mode in {
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
        direct_observation = PrimitiveUnaryEvidence(
            direct_signed,
            f"raster_adjoint_{observation_confidence_mode}",
            direct_confidence,
        )
        observation_fusion = str(
            getattr(args, "registered_observation_fusion", "additive")
        )
        strong_unary_anchor_threshold = (
            float(getattr(args, "hard_seed_threshold", 0.20))
            if observation_fusion == "hard_seed_anchored_probability"
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
            registered_observation_effective_confidence(
                direct_observation,
                anchor_threshold=strong_unary_anchor_threshold,
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
                "(1-exp(-raw_joint_prompt_mass/mass_scale))*"
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
                "unit_confidence_on_shared_hard_seed_rows"
                if strong_unary_anchor_threshold is not None
                else "none"
            ),
            "strong_unary_anchor_threshold": strong_unary_anchor_threshold,
            "strong_unary_anchor_rows": int(strong_unary_anchor_mask.sum()),
            "strong_unary_effective_confidence_sum": float(
                strong_unary_effective_confidence.sum()
            ),
            "strong_unary_formula": (
                "a=1[c>0 and abs(signed_observation)>=hard_seed_threshold]; "
                "effective_confidence=a+(1-a)*c; "
                "p=(1-effective_confidence)*p_field+"
                "effective_confidence*foreground_probability"
                if strong_unary_anchor_threshold is not None
                else None
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
            _require_bipolar_solver_support(
                positive_soft,
                negative_soft,
                label="Capability-valid",
            )
            valid_observation = PrimitiveUnaryEvidence(
                direct_observation.values[valid_rows_device].detach().float().cpu(),
                direct_observation.source,
                (
                    direct_observation.confidence[valid_rows_device]
                    .detach()
                    .float()
                    .cpu()
                    if direct_observation.confidence is not None
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
                    "valid_strong_unary_effective_confidence_sum": float(
                        strong_unary_effective_confidence[
                            valid_rows_device
                        ].sum()
                    ),
                    "valid_positive_solver_mass_sum": float(positive_soft.sum()),
                    "valid_negative_solver_mass_sum": float(negative_soft.sum()),
                    "valid_positive_solver_rows": int((positive_soft > 0).sum()),
                    "valid_negative_solver_rows": int((negative_soft > 0).sum()),
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
                    registered_observation_fusion=str(
                        getattr(
                            args,
                            "registered_observation_fusion",
                            "additive",
                        )
                    ),
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
            gaussian_scores = canonical_stage_gaussian_scores[
                str(getattr(args, "registered_readout_stage", "connected"))
            ]
            prediction_threshold = float(args.solver_support_threshold)
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
            "changes_geometry_or_alpha": False,
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
                "registered_observation_fusion": str(
                    getattr(args, "registered_observation_fusion", "additive")
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
                    if str(
                        getattr(
                            args,
                            "registered_observation_fusion",
                            "additive",
                        )
                    )
                    == "hard_seed_anchored_probability"
                    else {}
                ),
                "registered_seed_construction": str(
                    getattr(
                        args,
                        "registered_seed_construction",
                        "winner_take_all",
                    )
                ),
                "registered_seed_normalization": (
                    str(seed_normalization)
                    if registered_prompt_evidence is not None
                    else None
                ),
                "registered_observation_confidence": str(
                    getattr(
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
            "test_calibration_definition": (
                "no target labels, target masks, or metric feedback are used; "
                "unlabeled evaluation-scene statistics are disclosed separately"
            ),
            "official_sam_decoder": False,
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
        "radio_gs/querying/score_calibration.py",
        "radio_gs/querying/support_solver.py",
        "radio_gs/rendering/feature_renderer.py",
        "radio_gs/rendering/contribution_compositor.py",
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
                getattr(
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
            "pixel_threshold": float(prediction_threshold),
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
    parser.add_argument("--canonical-support-graph", default="")
    parser.add_argument("--canonical-reliability-cache", default="")
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
        ),
        default="additive",
        help=(
            "Fuse registered mass by the historical additive unary or as the "
            "label-free probability mixture (1-c)*p_field+c*q, or make "
            "the same direct rows accepted as solver hard seeds unit-confidence "
            "unary anchors."
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
    try:
        _validate_registered_forward_unary_args(args)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.feature_contribution_gamma) or args.feature_contribution_gamma <= 0:
        parser.error("--feature-contribution-gamma must be finite and positive")
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
        in {"probability_mixture", "hard_seed_anchored_probability"}
        and args.registered_seed_unary_weight != 0
    ):
        parser.error(
            "probability registered-observation fusion requires "
            "--registered-seed-unary-weight 0"
        )
    if args.registered_observation_fusion == "hard_seed_anchored_probability":
        if args.registered_seed_construction != "joint_signed":
            parser.error(
                "hard_seed_anchored_probability requires "
                "--registered-seed-construction joint_signed"
            )
        if args.hard_seed_threshold <= 0 or args.hard_seed_conflict_margin != 0:
            parser.error(
                "hard_seed_anchored_probability requires a positive "
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
