"""Frozen source-trained text likelihood adapter for LERF primitive caches.

This module is deliberately benchmark-label blind.  It consumes only the
already sealed positive/canonical-negative cosine caches, a query-independent
factorized field state, and the source-ScanNet-trained monotone likelihood
head.  The output keeps likelihood ``q`` and authority ``c`` separate.  A
legacy scalar consumer may use the exact ``PrimitiveUnaryEvidence`` decoding
``0.5 + c * (q - 0.5)``; no LERF threshold or metric is part of the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from .query_likelihood_head import MonotoneQueryLikelihoodHead, QueryLikelihoodInputs
from .source_text_query_likelihood import (
    LEGACY_FIELD_PRIOR_LOGIT_SCALE,
    SOURCE_TEXT_CHECKPOINT_SCHEMA,
    sha256_file,
    source_text_likelihood_contract,
)


LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_SCHEMA = (
    "radio_gs.lerf_source_text_query_likelihood_cache.v1"
)
LERF_SOURCE_TEXT_LIKELIHOOD_EFFECTIVE_FORMULA = "0.5 + c * (q - 0.5)"
LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_V2_SCHEMA = (
    "radio_gs.lerf_source_text_query_likelihood_cache.v2"
)
LERF_SOURCE_TEXT_LIKELIHOOD_V2_EFFECTIVE_FORMULA = (
    "(1 - c) * field_prior + c * q"
)
NEUTRAL_ABSTENTION_V1 = "neutral_abstention_v1"
PRIOR_PRESERVING_MIXTURE_V2 = "prior_preserving_mixture_v2"
POST_READOUT_PRIOR_PRESERVING_MIXTURE_V3 = (
    "post_readout_prior_preserving_mixture_v3"
)
POST_READOUT_PRIOR_PRESERVING_FORMULA_V3 = (
    "(1 - c) * field_probability_final + c * q"
)
POST_READOUT_ODDS_RESIDUAL_TRANSPORT_V4 = (
    "post_readout_odds_residual_transport_v4"
)
POST_READOUT_ODDS_RESIDUAL_FORMULA_V4 = (
    "sigmoid(logit(s_legacy_final) + c * "
    "(logit(q) - logit(p_field_raw)))"
)
POST_READOUT_ODDS_RESIDUAL_EPS_V4 = 1.0e-6
EFFECTIVE_PROBABILITY_MODES = (
    NEUTRAL_ABSTENTION_V1,
    PRIOR_PRESERVING_MIXTURE_V2,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compile_effective_probability(
    q: torch.Tensor,
    c: torch.Tensor,
    *,
    field_prior: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Compile q,c for a scalar consumer under an explicit fallback contract."""

    probability = torch.as_tensor(q).detach().cpu().float().contiguous()
    confidence = torch.as_tensor(c).detach().cpu().float().reshape(-1).contiguous()
    prior = torch.as_tensor(field_prior).detach().cpu().float().contiguous()
    if (
        probability.ndim != 2
        or prior.shape != probability.shape
        or confidence.shape != (probability.shape[0],)
    ):
        raise ValueError("q [N,Q], c [N], and field_prior [N,Q] must align")
    for name, tensor in (("q", probability), ("c", confidence), ("field_prior", prior)):
        if not bool(torch.isfinite(tensor).all()) or bool(
            ((tensor < 0) | (tensor > 1)).any()
        ):
            raise ValueError(f"{name} must be finite in [0,1]")
    authority = confidence[:, None]
    if mode == NEUTRAL_ABSTENTION_V1:
        return (0.5 + authority * (probability - 0.5)).float().contiguous()
    if mode != PRIOR_PRESERVING_MIXTURE_V2:
        raise ValueError(f"unsupported effective probability mode: {mode}")
    mixed = ((1.0 - authority) * prior + authority * probability).float()
    # Make the two mixture identities an exact storage contract, not merely a
    # real-arithmetic statement subject to multiply/add rounding.
    mixed = torch.where(authority == 0, prior, mixed)
    mixed = torch.where(authority == 1, probability, mixed)
    return mixed.contiguous()


def compile_post_readout_probability(
    field_probability_final: torch.Tensor,
    q: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Fuse only after the frozen field spatial readout has fully completed.

    ``field_probability_final`` is the legacy scale-select, kNN10, and
    scene-minmax result.  The returned tensor is final: consumers must apply
    only their already frozen threshold/projection, never another spatial or
    scene-wise normalization.
    """

    field = torch.as_tensor(field_probability_final).detach().cpu().float().contiguous()
    probability = torch.as_tensor(q).detach().cpu().float().contiguous()
    confidence = torch.as_tensor(c).detach().cpu().float().reshape(-1).contiguous()
    if (
        field.ndim != 2
        or probability.shape != field.shape
        or confidence.shape != (field.shape[0],)
    ):
        raise ValueError("field_probability_final/q [N,Q] and c [N] must align")
    for name, tensor in (
        ("field_probability_final", field),
        ("q", probability),
        ("c", confidence),
    ):
        if not bool(torch.isfinite(tensor).all()) or bool(
            ((tensor < 0) | (tensor > 1)).any()
        ):
            raise ValueError(f"{name} must be finite in [0,1]")
    authority = confidence[:, None]
    mixed = ((1.0 - authority) * field + authority * probability).float()
    mixed = torch.where(authority == 0, field, mixed)
    mixed = torch.where(authority == 1, probability, mixed).contiguous()
    if not torch.equal(mixed[confidence == 0], field[confidence == 0]):
        raise RuntimeError("post-readout c=0 identity changed")
    return mixed


def compile_post_readout_odds_residual(
    legacy_probability_final: torch.Tensor,
    q: torch.Tensor,
    field_probability_raw: torch.Tensor,
    c: torch.Tensor,
    *,
    eps: float = POST_READOUT_ODDS_RESIDUAL_EPS_V4,
) -> torch.Tensor:
    """Transport source evidence as an odds residual after spatial readout.

    The source head's absolute probability is deliberately not mixed with the
    scene-relative legacy readout.  Only its evidence residual relative to the
    exact raw field prior is transported.  ``eps`` is a fixed numerical guard,
    not a tunable method parameter.  Exact neutral-residual, zero-authority,
    and absorbing endpoint identities are restored after the finite logit
    computation.
    """

    if float(eps) != POST_READOUT_ODDS_RESIDUAL_EPS_V4:
        raise ValueError(
            "post-readout odds-residual eps is frozen at "
            f"{POST_READOUT_ODDS_RESIDUAL_EPS_V4}"
        )
    legacy = (
        torch.as_tensor(legacy_probability_final)
        .detach()
        .cpu()
        .float()
        .contiguous()
    )
    probability = torch.as_tensor(q).detach().cpu().float().contiguous()
    raw = torch.as_tensor(field_probability_raw).detach().cpu().float().contiguous()
    confidence = torch.as_tensor(c).detach().cpu().float().reshape(-1).contiguous()
    if (
        legacy.ndim != 2
        or probability.shape != legacy.shape
        or raw.shape != legacy.shape
        or confidence.shape != (legacy.shape[0],)
    ):
        raise ValueError(
            "s_legacy_final/q/p_field_raw [N,Q] and c [N] must align"
        )
    for name, tensor in (
        ("s_legacy_final", legacy),
        ("q", probability),
        ("p_field_raw", raw),
        ("c", confidence),
    ):
        if not bool(torch.isfinite(tensor).all()) or bool(
            ((tensor < 0) | (tensor > 1)).any()
        ):
            raise ValueError(f"{name} must be finite in [0,1]")

    def _finite_logit(value: torch.Tensor) -> torch.Tensor:
        clipped = value.clamp(
            POST_READOUT_ODDS_RESIDUAL_EPS_V4,
            1.0 - POST_READOUT_ODDS_RESIDUAL_EPS_V4,
        )
        return torch.logit(clipped)

    residual = _finite_logit(probability) - _finite_logit(raw)
    authority = confidence[:, None]
    transported = torch.sigmoid(_finite_logit(legacy) + authority * residual).float()
    exact_identity = (authority == 0) | (probability == raw)
    absorbing_endpoint = (legacy == 0) | (legacy == 1)
    positive = (probability > raw) & (authority > 0) & ~absorbing_endpoint
    negative = (probability < raw) & (authority > 0) & ~absorbing_endpoint
    # Turn mathematical monotonicity into a storage-level contract.  A finite
    # float32 logit/sigmoid round trip can otherwise move by one ULP in the
    # wrong direction when c*delta is extremely small.
    transported = torch.where(positive, torch.maximum(transported, legacy), transported)
    transported = torch.where(negative, torch.minimum(transported, legacy), transported)
    transported = torch.where(exact_identity | absorbing_endpoint, legacy, transported)
    transported = transported.contiguous()

    if not torch.equal(transported[confidence == 0], legacy[confidence == 0]):
        raise RuntimeError("post-readout v4 c=0 identity changed")
    if not torch.equal(transported[probability == raw], legacy[probability == raw]):
        raise RuntimeError("post-readout v4 neutral residual identity changed")
    if bool((transported[positive] < legacy[positive]).any()):
        raise RuntimeError("positive odds residual decreased the legacy score")
    if bool((transported[negative] > legacy[negative]).any()):
        raise RuntimeError("negative odds residual increased the legacy score")
    return transported


def _record(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source)}


def _load_mapping(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return source, dict(payload)


def load_frozen_source_text_head(
    path: str | Path,
    *,
    expected_state_sha256: str = "",
) -> tuple[MonotoneQueryLikelihoodHead, dict[str, Any], dict[str, str]]:
    source, payload = _load_mapping(path, label="source text likelihood checkpoint")
    if (
        payload.get("schema") != SOURCE_TEXT_CHECKPOINT_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("head_class") != "MonotoneQueryLikelihoodHead"
        or payload.get("head_schema_version") != "monotone-query-likelihood-v1"
        or payload.get("contract") != source_text_likelihood_contract()
    ):
        raise ValueError("source text likelihood checkpoint contract differs")
    access = payload.get("source_access")
    if not isinstance(access, Mapping) or (
        access.get("official_scannet_train_scenes_only") is not True
        or access.get("source_train_semantic_labels_opened") is not True
        or access.get("development_labels_opened") is not False
        or access.get("test_labels_opened") is not False
        or access.get("lerf_queries_or_ground_truth_opened") is not False
        or access.get("target_rgb_or_mask_opened") is not False
        or access.get("benchmark_predictions_or_metrics_opened") is not False
    ):
        raise PermissionError("source text likelihood checkpoint crossed its fit boundary")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("source text likelihood checkpoint lacks state_dict")
    observed_state_sha256 = state_dict_sha256(state)
    if expected_state_sha256 and observed_state_sha256 != expected_state_sha256:
        raise ValueError(
            "source text likelihood state SHA256 differs: "
            f"{observed_state_sha256} vs {expected_state_sha256}"
        )
    head = MonotoneQueryLikelihoodHead(affinity_channel_count=1).cpu()
    head.load_state_dict(dict(state), strict=True)
    head.eval()
    return head, payload, {
        "path": str(source),
        "sha256": sha256_file(source),
        "state_sha256": observed_state_sha256,
    }


def _validate_raw_pair(
    positive: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_scores = torch.as_tensor(positive.get("query_scores")).detach().cpu()
    negative_scores = torch.as_tensor(negative.get("query_scores")).detach().cpu()
    if (
        positive.get("version") != 4
        or negative.get("version") != 4
        or positive.get("contract")
        != "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
        or negative.get("contract")
        != "radio_gs.ours_lerf_direct3d_multiscale_query_scores_fp32.v4"
        or positive_scores.dtype != torch.float32
        or negative_scores.dtype != torch.float32
        or not positive_scores.is_contiguous()
        or not negative_scores.is_contiguous()
        or positive_scores.ndim != 3
        or negative_scores.ndim != 3
        or positive_scores.shape[:2] != negative_scores.shape[:2]
        or positive_scores.shape[1] != 3
    ):
        raise ValueError("LERF source text adapter requires paired FP32 v4 [N,3,Q/K] caches")
    for field in (
        "scale_ids",
        "scale_radii_m",
        "field_checkpoint_sha256",
        "readout_checkpoint_sha256",
        "renderer_geometry_checkpoint_sha256",
    ):
        if positive.get(field) != negative.get(field):
            raise ValueError(f"positive/canonical-negative cache {field} differs")
    if positive.get("query_ids") is None or negative.get("query_ids") is None:
        raise ValueError("paired score caches require ordered query_ids")
    xyz = torch.as_tensor(positive.get("xyz")).detach().cpu().float().contiguous()
    negative_xyz = torch.as_tensor(negative.get("xyz")).detach().cpu().float()
    valid = torch.as_tensor(positive.get("valid")).detach().cpu()
    negative_valid = torch.as_tensor(negative.get("valid")).detach().cpu()
    if (
        xyz.shape != (positive_scores.shape[0], 3)
        or not torch.equal(xyz, negative_xyz)
        or valid.dtype != torch.bool
        or valid.shape != (positive_scores.shape[0],)
        or not torch.equal(valid, negative_valid)
        or not bool(valid.any())
    ):
        raise ValueError("paired score-cache geometry/valid axes differ")
    if not bool(torch.isfinite(positive_scores).all()) or not bool(
        torch.isfinite(negative_scores).all()
    ):
        raise ValueError("paired score caches contain NaN or infinity")
    return positive_scores, negative_scores, xyz, valid


def field_coverage_reliability(
    factorized_state: Mapping[str, Any],
    *,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_field_checkpoint_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        factorized_state.get("schema") != "radio_gs.factorized_primitive_state.v2"
        or factorized_state.get("schema_version") != 2
    ):
        raise ValueError("source text adapter requires factorized primitive state v2")
    metadata = factorized_state.get("metadata")
    if not isinstance(metadata, Mapping) or (
        metadata.get("field_checkpoint_sha256") != expected_field_checkpoint_sha256
        or metadata.get("query_independent") is not True
        or metadata.get("benchmark_masks_opened") is not False
        or metadata.get("text_queries_opened") is not False
    ):
        raise ValueError("factorized primitive state authority differs")
    xyz = torch.as_tensor(factorized_state.get("xyz")).detach().cpu().float()
    valid = torch.as_tensor(factorized_state.get("valid")).detach().cpu()
    rows = torch.as_tensor(factorized_state.get("global_rows")).detach().cpu().long()
    if (
        xyz.shape != expected_xyz.shape
        or not torch.equal(xyz, expected_xyz)
        or valid.dtype != torch.bool
        or not torch.equal(valid, expected_valid)
        or not torch.equal(rows, torch.nonzero(valid, as_tuple=False).reshape(-1))
    ):
        raise ValueError("factorized state row authority differs from score cache")
    dispersion = torch.as_tensor(factorized_state.get("directional_dispersion")).float()
    evidence = torch.as_tensor(factorized_state.get("observation_evidence")).float()
    purity = torch.as_tensor(factorized_state.get("visibility_purity_value")).float()
    purity_known = torch.as_tensor(factorized_state.get("visibility_purity_known"))
    count = int(rows.numel())
    if (
        dispersion.shape != (count,)
        or evidence.shape != (count,)
        or purity.shape != (count,)
        or purity_known.dtype != torch.bool
        or purity_known.shape != (count,)
    ):
        raise ValueError("factorized state reliability channels differ")
    coverage_valid = evidence.clamp(0, 1)
    reliability_valid = torch.stack(
        (
            1.0 - dispersion.clamp(0, 1),
            coverage_valid,
            purity.clamp(0, 1) * purity_known.float(),
        ),
        dim=1,
    ).mean(dim=1)
    coverage = torch.zeros(valid.shape, dtype=torch.float32)
    reliability = torch.zeros(valid.shape, dtype=torch.float32)
    coverage[rows] = coverage_valid
    reliability[rows] = reliability_valid.clamp(0, 1)
    return coverage.contiguous(), reliability.contiguous()


def legacy_canonical_field_coverage_reliability(
    canonical_field: Mapping[str, Any],
    *,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_field_checkpoint_sha256: str,
    observed_field_checkpoint_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode query-free confidence from a frozen schema-v1 field.

    The historical carrier is ``coverage, agreement, stability``.  It predates
    exact-marginal visibility purity, so its third channel cannot be silently
    reinterpreted as purity.  The source-head bridge therefore uses exact
    unknown-purity semantics: ``reliability=mean(agreement, coverage, 0)``.
    No descriptor, text score, field prior, or learned parameter is changed.
    """

    if (
        canonical_field.get("schema_version") != 1
        or observed_field_checkpoint_sha256 != expected_field_checkpoint_sha256
        or canonical_field.get("benchmark_masks_opened") is not False
        or canonical_field.get("text_queries_opened") is not False
        or canonical_field.get("benchmark_images_opened", False) is not False
    ):
        raise ValueError("legacy canonical field authority differs")
    architecture = canonical_field.get("architecture")
    geometry = canonical_field.get("geometry_fingerprint")
    if not isinstance(architecture, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("legacy canonical field lacks architecture/geometry authority")
    rows = int(expected_xyz.shape[0])
    expected_xyz_sha256 = hashlib.sha256(
        expected_xyz.detach().cpu().float().contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    if (
        architecture.get("num_gaussians") != rows
        or geometry.get("num_gaussians") != rows
        or geometry.get("xyz_sha256") != expected_xyz_sha256
    ):
        raise ValueError("legacy canonical field geometry differs from score cache")
    carrier = torch.as_tensor(canonical_field.get("reliability")).detach().cpu().float()
    valid = torch.as_tensor(expected_valid).detach().cpu()
    if (
        carrier.shape != (rows, 3)
        or valid.dtype != torch.bool
        or valid.shape != (rows,)
        or not bool(torch.isfinite(carrier).all())
        or bool(((carrier < 0) | (carrier > 1)).any())
        or not torch.equal(carrier.ne(0).any(dim=1), valid)
    ):
        raise ValueError("legacy canonical field reliability carrier differs")
    coverage = carrier[:, 0].contiguous()
    agreement = carrier[:, 1]
    reliability = torch.stack(
        (agreement, coverage, torch.zeros_like(coverage)), dim=1
    ).mean(dim=1)
    return coverage, reliability.contiguous()


@torch.inference_mode()
def build_lerf_source_text_likelihood_cache(
    *,
    positive_score_cache: str | Path,
    negative_score_cache: str | Path,
    factorized_state: str | Path | None = None,
    canonical_field_checkpoint: str | Path | None = None,
    source_text_head_checkpoint: str | Path,
    expected_head_state_sha256: str = "",
    effective_probability_mode: str = NEUTRAL_ABSTENTION_V1,
) -> dict[str, Any]:
    if effective_probability_mode not in EFFECTIVE_PROBABILITY_MODES:
        raise ValueError("unsupported effective probability mode")
    positive_path, positive = _load_mapping(positive_score_cache, label="positive cache")
    negative_path, negative = _load_mapping(negative_score_cache, label="negative cache")
    positive_scores, negative_scores, xyz, valid = _validate_raw_pair(positive, negative)
    field_sha256 = str(positive["field_checkpoint_sha256"])
    if _SHA256.fullmatch(field_sha256) is None:
        raise ValueError("field checkpoint SHA256 is malformed")
    if (factorized_state is None) == (canonical_field_checkpoint is None):
        raise ValueError(
            "exactly one factorized state or legacy canonical field is required"
        )
    support_record: dict[str, str]
    support_contract: str
    support_key: str
    if factorized_state is not None:
        state_path, state = _load_mapping(factorized_state, label="factorized state")
        coverage, reliability = field_coverage_reliability(
            state,
            expected_xyz=xyz,
            expected_valid=valid,
            expected_field_checkpoint_sha256=field_sha256,
        )
        support_record = _record(state_path)
        support_contract = "factorized_primitive_state_v2"
        support_key = "factorized_primitive_state"
    else:
        field_path, field_payload = _load_mapping(
            canonical_field_checkpoint, label="legacy canonical field"
        )
        observed_field_sha256 = sha256_file(field_path)
        coverage, reliability = legacy_canonical_field_coverage_reliability(
            field_payload,
            expected_xyz=xyz,
            expected_valid=valid,
            expected_field_checkpoint_sha256=field_sha256,
            observed_field_checkpoint_sha256=observed_field_sha256,
        )
        support_record = _record(field_path)
        support_contract = (
            "canonical_field_v1_coverage_agreement_unknown_purity_bridge_v1"
        )
        support_key = "canonical_field_checkpoint"
    head, checkpoint, head_record = load_frozen_source_text_head(
        source_text_head_checkpoint,
        expected_state_sha256=expected_head_state_sha256,
    )
    positive_affinity = ((positive_scores + 1.0) * 0.5).clamp(0, 1)
    negative_affinity = ((negative_scores + 1.0) * 0.5).clamp(0, 1)
    field_prior = torch.sigmoid(
        LEGACY_FIELD_PRIOR_LOGIT_SCALE
        * (positive_scores - negative_scores.amax(dim=-1, keepdim=True))
    ).amax(dim=1)
    flattened_negative = negative_affinity.reshape(negative_affinity.shape[0], -1)
    q_columns = []
    c: torch.Tensor | None = None
    for query_index in range(int(positive_scores.shape[2])):
        evidence = head(
            QueryLikelihoodInputs(
                positive_affinity=positive_affinity[:, :, query_index],
                negative_affinity=flattened_negative,
                prior_probability=field_prior[:, query_index],
                coverage=coverage,
                reliability=reliability,
            ),
            source="source_train_text_likelihood_lerf_adapter",
        )
        q_columns.append(evidence.foreground_probability)
        if c is None:
            c = evidence.confidence
        elif not torch.equal(c, evidence.confidence):
            raise RuntimeError("query-independent likelihood confidence changed by query")
    assert c is not None
    q = torch.stack(q_columns, dim=1).float().contiguous()
    c = c.float().contiguous()
    effective = compile_effective_probability(
        q,
        c,
        field_prior=field_prior,
        mode=effective_probability_mode,
    )
    tensors = {
        "q": q,
        "c": c,
        "effective_probability": effective,
        "coverage": coverage,
        "reliability": reliability,
        "valid": valid.contiguous(),
        "xyz": xyz.contiguous(),
    }
    schema = LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_SCHEMA
    schema_version = 1
    effective_formula = LERF_SOURCE_TEXT_LIKELIHOOD_EFFECTIVE_FORMULA
    mode_metadata: dict[str, str] = {}
    if effective_probability_mode == PRIOR_PRESERVING_MIXTURE_V2:
        schema = LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_V2_SCHEMA
        schema_version = 2
        effective_formula = LERF_SOURCE_TEXT_LIKELIHOOD_V2_EFFECTIVE_FORMULA
        tensors["field_prior_probability"] = field_prior.float().contiguous()
        mode_metadata["effective_probability_mode"] = effective_probability_mode
    return {
        "schema": schema,
        "schema_version": schema_version,
        "query_ids": list(positive["query_ids"]),
        "canonical_negative_query_ids": list(negative["query_ids"]),
        "scale_ids": list(positive["scale_ids"]),
        "scale_radii_m": list(positive["scale_radii_m"]),
        "field_checkpoint_sha256": field_sha256,
        "readout_checkpoint_sha256": str(positive["readout_checkpoint_sha256"]),
        "renderer_geometry_checkpoint_sha256": str(
            positive["renderer_geometry_checkpoint_sha256"]
        ),
        "head_schema_version": head.schema_version,
        "head_state_sha256": head_record["state_sha256"],
        "effective_probability_formula": effective_formula,
        **mode_metadata,
        "input_factorization": dict(checkpoint["contract"]["input_factorization"]),
        **tensors,
        "channel_sha256": {key: tensor_sha256(value) for key, value in tensors.items()},
        "source_artifacts": {
            "positive_score_cache": _record(positive_path),
            "canonical_negative_score_cache": _record(negative_path),
            support_key: support_record,
            "source_text_likelihood_head": head_record,
        },
        "field_reliability_support_contract": support_contract,
        "source_access": {
            "head_fit_scenes": list(checkpoint["source_scene_ids"]),
            "lerf_ground_truth_or_metric_opened": False,
            "target_rgb_or_mask_opened": False,
            "per_scene_or_per_query_metric_tuning": False,
        },
    }


@dataclass(frozen=True)
class LerfSourceTextLikelihoodCache:
    q: torch.Tensor
    c: torch.Tensor
    effective_probability: torch.Tensor
    field_prior_probability: torch.Tensor | None
    valid: torch.Tensor
    query_ids: tuple[str, ...]
    field_checkpoint_sha256: str
    head_state_sha256: str
    effective_probability_mode: str
    effective_probability_formula: str
    source_artifacts: Mapping[str, Mapping[str, str]]


def validate_lerf_source_text_likelihood_cache(
    value: object,
    *,
    expected_xyz: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_query_ids: Sequence[str],
    expected_positive_score_cache: str | Path,
    expected_negative_score_cache: str | Path,
    expected_renderer_geometry_checkpoint_sha256: str,
) -> LerfSourceTextLikelihoodCache:
    if not isinstance(value, Mapping):
        raise ValueError("LERF source text likelihood cache must be a mapping")
    payload = dict(value)
    cache_v1 = (
        payload.get("schema") == LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_SCHEMA
        and payload.get("schema_version") == 1
        and payload.get("effective_probability_formula")
        == LERF_SOURCE_TEXT_LIKELIHOOD_EFFECTIVE_FORMULA
        and payload.get("effective_probability_mode", NEUTRAL_ABSTENTION_V1)
        == NEUTRAL_ABSTENTION_V1
    )
    cache_v2 = (
        payload.get("schema") == LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_V2_SCHEMA
        and payload.get("schema_version") == 2
        and payload.get("effective_probability_formula")
        == LERF_SOURCE_TEXT_LIKELIHOOD_V2_EFFECTIVE_FORMULA
        and payload.get("effective_probability_mode")
        == PRIOR_PRESERVING_MIXTURE_V2
    )
    if (
        not (cache_v1 or cache_v2)
        or payload.get("head_schema_version") != "monotone-query-likelihood-v1"
    ):
        raise ValueError("LERF source text likelihood cache contract differs")
    query_ids = tuple(str(value) for value in payload.get("query_ids", []))
    if query_ids != tuple(str(value) for value in expected_query_ids):
        raise ValueError("LERF source text likelihood query order differs")
    xyz = torch.as_tensor(payload.get("xyz")).detach().cpu().float()
    expected_xyz_cpu = expected_xyz.detach().cpu().float()
    valid = torch.as_tensor(payload.get("valid")).detach().cpu()
    expected_valid_cpu = expected_valid.detach().cpu()
    rows, queries = int(expected_xyz_cpu.shape[0]), len(query_ids)
    q = torch.as_tensor(payload.get("q")).detach().cpu().float()
    c = torch.as_tensor(payload.get("c")).detach().cpu().float()
    effective = torch.as_tensor(payload.get("effective_probability")).detach().cpu().float()
    field_prior = (
        torch.as_tensor(payload.get("field_prior_probability"))
        .detach()
        .cpu()
        .float()
        if cache_v2
        else None
    )
    if (
        xyz.shape != expected_xyz_cpu.shape
        or not torch.equal(xyz, expected_xyz_cpu)
        or valid.dtype != torch.bool
        or valid.shape != (rows,)
        or not torch.equal(valid, expected_valid_cpu)
        or q.shape != (rows, queries)
        or c.shape != (rows,)
        or effective.shape != (rows, queries)
        or (cache_v2 and field_prior is not None and field_prior.shape != (rows, queries))
    ):
        raise ValueError("LERF source text likelihood axes differ")
    bounded = [("q", q), ("c", c), ("effective_probability", effective)]
    if field_prior is not None:
        bounded.append(("field_prior_probability", field_prior))
    for name, tensor in bounded:
        if not bool(torch.isfinite(tensor).all()) or bool(
            ((tensor < 0) | (tensor > 1)).any()
        ):
            raise ValueError(f"LERF source text likelihood {name} must lie in [0,1]")
    compiled = compile_effective_probability(
        q,
        c,
        field_prior=(torch.full_like(q, 0.5) if field_prior is None else field_prior),
        mode=(NEUTRAL_ABSTENTION_V1 if cache_v1 else PRIOR_PRESERVING_MIXTURE_V2),
    )
    if not torch.equal(effective, compiled):
        raise ValueError("LERF source text effective probability formula changed")
    channels = payload.get("channel_sha256")
    channel_tensors = {
        "q": q,
        "c": c,
        "effective_probability": effective,
        "coverage": torch.as_tensor(payload.get("coverage")).detach().cpu().float(),
        "reliability": torch.as_tensor(payload.get("reliability")).detach().cpu().float(),
        "valid": valid,
        "xyz": xyz,
    }
    if field_prior is not None:
        channel_tensors["field_prior_probability"] = field_prior
    if not isinstance(channels, Mapping) or set(channels) != set(channel_tensors):
        raise ValueError("LERF source text likelihood channel hashes differ")
    for name, tensor in channel_tensors.items():
        if channels.get(name) != tensor_sha256(tensor):
            raise ValueError(f"LERF source text likelihood channel changed: {name}")
    sources = payload.get("source_artifacts")
    if not isinstance(sources, Mapping):
        raise ValueError("LERF source text likelihood source records missing")
    expected_paths = {
        "positive_score_cache": Path(expected_positive_score_cache).expanduser().resolve(),
        "canonical_negative_score_cache": Path(expected_negative_score_cache).expanduser().resolve(),
    }
    for name, expected_path in expected_paths.items():
        record = sources.get(name)
        if (
            not isinstance(record, Mapping)
            or Path(str(record.get("path", ""))).expanduser().resolve() != expected_path
            or record.get("sha256") != sha256_file(expected_path)
        ):
            raise ValueError(f"LERF source text likelihood {name} binding differs")
    if (
        payload.get("renderer_geometry_checkpoint_sha256")
        != expected_renderer_geometry_checkpoint_sha256
    ):
        raise ValueError("LERF source text likelihood renderer geometry differs")
    access = payload.get("source_access")
    if not isinstance(access, Mapping) or (
        access.get("lerf_ground_truth_or_metric_opened") is not False
        or access.get("target_rgb_or_mask_opened") is not False
        or access.get("per_scene_or_per_query_metric_tuning") is not False
    ):
        raise PermissionError("LERF likelihood cache was not sealed target-blind")
    return LerfSourceTextLikelihoodCache(
        q=q,
        c=c,
        effective_probability=effective,
        field_prior_probability=field_prior,
        valid=valid,
        query_ids=query_ids,
        field_checkpoint_sha256=str(payload["field_checkpoint_sha256"]),
        head_state_sha256=str(payload["head_state_sha256"]),
        effective_probability_mode=(
            NEUTRAL_ABSTENTION_V1
            if cache_v1
            else PRIOR_PRESERVING_MIXTURE_V2
        ),
        effective_probability_formula=str(payload["effective_probability_formula"]),
        source_artifacts=dict(sources),
    )


def load_lerf_source_text_likelihood_cache(
    path: str | Path,
    **expected: Any,
) -> LerfSourceTextLikelihoodCache:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return validate_lerf_source_text_likelihood_cache(payload, **expected)


__all__ = [
    "EFFECTIVE_PROBABILITY_MODES",
    "LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_SCHEMA",
    "LERF_SOURCE_TEXT_LIKELIHOOD_CACHE_V2_SCHEMA",
    "LERF_SOURCE_TEXT_LIKELIHOOD_EFFECTIVE_FORMULA",
    "LERF_SOURCE_TEXT_LIKELIHOOD_V2_EFFECTIVE_FORMULA",
    "NEUTRAL_ABSTENTION_V1",
    "PRIOR_PRESERVING_MIXTURE_V2",
    "POST_READOUT_PRIOR_PRESERVING_FORMULA_V3",
    "POST_READOUT_PRIOR_PRESERVING_MIXTURE_V3",
    "POST_READOUT_ODDS_RESIDUAL_EPS_V4",
    "POST_READOUT_ODDS_RESIDUAL_FORMULA_V4",
    "POST_READOUT_ODDS_RESIDUAL_TRANSPORT_V4",
    "LerfSourceTextLikelihoodCache",
    "build_lerf_source_text_likelihood_cache",
    "compile_effective_probability",
    "compile_post_readout_probability",
    "compile_post_readout_odds_residual",
    "field_coverage_reliability",
    "legacy_canonical_field_coverage_reliability",
    "load_frozen_source_text_head",
    "load_lerf_source_text_likelihood_cache",
    "state_dict_sha256",
    "tensor_sha256",
    "validate_lerf_source_text_likelihood_cache",
]
