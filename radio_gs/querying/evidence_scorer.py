"""Shared primitive-domain unary scorer for every query modality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .query_spec import PrototypeSet, QuerySpec
from .score_calibration import (
    SceneSpaceCalibration,
    robust_tanh_score_calibration,
)


@dataclass(frozen=True)
class EvidenceScoringConfig:
    semantic_weight: float = 1.0
    appearance_weight: float = 1.0
    boundary_weight: float = 0.35
    prototype_temperature: float = 0.07
    feature_calibration: str = "none"
    background_centroids: int = 0
    calibration_sample_size: int = 8192
    centroid_iterations: int = 4
    score_calibration: str = "none"
    score_tanh_scale: float = 2.0
    score_chunk_size: int = 65536

    def __post_init__(self) -> None:
        if min(self.semantic_weight, self.appearance_weight, self.boundary_weight) < 0:
            raise ValueError("evidence weights must be non-negative")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.feature_calibration not in {"none", "diagonal_robust"}:
            raise ValueError("feature_calibration must be none or diagonal_robust")
        if self.score_calibration not in {
            "none",
            "robust_tanh",
            "robust_tanh_centered",
            "robust_tanh_zero",
        }:
            raise ValueError("unsupported score_calibration")
        if (
            self.background_centroids < 0
            or self.calibration_sample_size <= 0
            or self.centroid_iterations < 0
            or self.score_tanh_scale <= 0
            or self.score_chunk_size <= 0
        ):
            raise ValueError("invalid evidence calibration parameters")


def shrink_unary_by_reliability(
    unary: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Shrink signed evidence toward its neutral zero, never toward background."""

    values = torch.as_tensor(unary)
    confidence = torch.as_tensor(reliability).to(
        device=values.device, dtype=values.dtype
    )
    if values.ndim != 1 or confidence.shape != values.shape:
        raise ValueError("unary and primitive reliability must align as [N]")
    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("primitive reliability contains NaN or infinity")
    if bool((confidence < 0).any()) or bool((confidence > 1).any()):
        raise ValueError("primitive reliability must be in [0,1]")
    return values * confidence


def _score_bank(
    field_features: torch.Tensor,
    evidence: PrototypeSet,
    *,
    temperature: float,
    calibration: SceneSpaceCalibration | None = None,
    chunk_size: int = 65536,
) -> torch.Tensor:
    field = torch.as_tensor(field_features)
    if field.ndim != 2 or chunk_size <= 0:
        raise ValueError("field feature bank must be [N,D] and chunk_size positive")
    if calibration is None:
        prototypes = evidence.features.to(field.device)
    else:
        prototypes = calibration.transform(evidence.features)
    weights = evidence.weights.to(field.device).clamp_min(1e-8)
    negatives = None
    if evidence.negatives is not None:
        negatives = (
            evidence.negatives.to(field.device)
            if calibration is None
            else calibration.transform(evidence.negatives)
        )
    has_background_bank = (
        calibration is not None and calibration.background_centroids is not None
    )
    if has_background_bank:
        assert calibration is not None
        centroids = calibration.background_centroids
        negatives = centroids if negatives is None else torch.cat([negatives, centroids])
    parts: list[torch.Tensor] = []
    for start in range(0, field.shape[0], int(chunk_size)):
        rows = field[start : start + int(chunk_size)].float()
        rows = (
            F.normalize(rows, dim=-1, eps=1e-8)
            if calibration is None
            else calibration.transform(rows)
        )
        logits = rows @ prototypes.T
        positive = temperature * torch.logsumexp(
            logits / temperature + weights.log()[None], dim=1
        )
        if negatives is not None:
            negative_logits = rows @ negatives.T
            if has_background_bank:
                negative = temperature * (
                    torch.logsumexp(negative_logits / temperature, dim=1)
                    - torch.log(
                        torch.tensor(
                            negatives.shape[0],
                            device=rows.device,
                            dtype=torch.float32,
                        )
                    )
                )
            else:
                # Preserve the original explicit-negative readout exactly.
                negative = negative_logits.amax(dim=1)
            positive = positive - negative
        parts.append(positive)
    return torch.cat(parts)


def score_query_evidence(
    query: QuerySpec,
    feature_banks: Mapping[str, torch.Tensor],
    *,
    config: EvidenceScoringConfig = EvidenceScoringConfig(),
    calibrations: Mapping[str, SceneSpaceCalibration] | None = None,
    num_nodes: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    weighted: list[torch.Tensor] = []
    for name, evidence, weight in (
        ("semantic", query.semantic_evidence, config.semantic_weight),
        ("appearance", query.appearance_evidence, config.appearance_weight),
        ("boundary", query.boundary_evidence, config.boundary_weight),
    ):
        if evidence is None:
            continue
        if name not in feature_banks:
            raise KeyError(f"query requires missing {name} feature bank")
        score = _score_bank(
            feature_banks[name],
            evidence,
            temperature=config.prototype_temperature,
            calibration=(calibrations or {}).get(name),
            chunk_size=config.score_chunk_size,
        )
        if config.score_calibration != "none":
            score = robust_tanh_score_calibration(
                score,
                tanh_scale=config.score_tanh_scale,
                preserve_zero=config.score_calibration in {
                    "robust_tanh",
                    "robust_tanh_zero",
                },
            )
        components[name] = score
        weighted.append(score * float(weight))
    if not weighted:
        count = int(num_nodes) if num_nodes is not None else 0
        if count <= 0 and query.positive_seeds is not None:
            count = query.positive_seeds.weights.numel()
        if count == 0:
            raise ValueError("query has neither prototypes nor seeds")
        for seeds in (query.positive_seeds, query.negative_seeds):
            if seeds is not None and seeds.weights.numel() != count:
                raise ValueError("query seeds do not align with num_nodes")
        # A seed-only query has no global foreground evidence.  Use the
        # minimum cosine prior instead of an ambiguous 0-logit (p=0.5), then
        # let the same graph diffuse the registered/world-space seeds.
        unary = torch.full((count,), -1.0, dtype=torch.float32)
    else:
        unary = torch.stack(weighted).sum(dim=0) / max(
            1e-8,
            sum(
                weight
                for evidence, weight in (
                    (query.semantic_evidence, config.semantic_weight),
                    (query.appearance_evidence, config.appearance_weight),
                    (query.boundary_evidence, config.boundary_weight),
                )
                if evidence is not None
            ),
        )
    return unary, components
