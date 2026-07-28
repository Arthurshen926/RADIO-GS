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
    background_negative_policy: str = "pooled_mean"
    calibration_sample_size: int = 8192
    centroid_iterations: int = 4
    score_calibration: str = "none"
    score_tanh_scale: float = 2.0
    score_chunk_size: int = 65536
    negative_spatial_mode: str = "none"
    negative_spatial_steps: int = 4
    negative_spatial_decay: float = 0.8
    spatial_log_weight: float = 0.25
    spatial_floor: float = 0.01

    def __post_init__(self) -> None:
        if min(self.semantic_weight, self.appearance_weight, self.boundary_weight) < 0:
            raise ValueError("evidence weights must be non-negative")
        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.feature_calibration not in {"none", "diagonal_robust"}:
            raise ValueError("feature_calibration must be none or diagonal_robust")
        if self.background_negative_policy not in {
            "pooled_mean",
            "explicit_hard_max",
        }:
            raise ValueError(
                "background_negative_policy must be pooled_mean or explicit_hard_max"
            )
        if self.score_calibration not in {
            "none",
            "robust_tanh",
            "robust_tanh_centered",
            "robust_tanh_zero",
        }:
            raise ValueError("unsupported score_calibration")
        if self.negative_spatial_mode not in {
            "none",
            "truncated_graph_decay",
            "signed_geodesic",
        }:
            raise ValueError(
                "negative_spatial_mode must be none, truncated_graph_decay, "
                "or signed_geodesic"
            )
        if (
            self.background_centroids < 0
            or self.calibration_sample_size <= 0
            or self.centroid_iterations < 0
            or self.score_tanh_scale <= 0
            or self.score_chunk_size <= 0
            or self.negative_spatial_steps < 0
            or not 0 <= self.negative_spatial_decay <= 1
            or self.spatial_log_weight < 0
            or not 0 < self.spatial_floor <= 1
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
    background_negative_policy: str = "pooled_mean",
    chunk_size: int = 65536,
    explicit_negative_influence: torch.Tensor | None = None,
    positive_spatial_influence: torch.Tensor | None = None,
    explicit_negative_spatial: torch.Tensor | None = None,
    spatial_log_weight: float = 0.25,
    spatial_floor: float = 0.01,
) -> torch.Tensor:
    field = torch.as_tensor(field_features)
    if field.ndim != 2 or chunk_size <= 0:
        raise ValueError("field feature bank must be [N,D] and chunk_size positive")
    if background_negative_policy not in {"pooled_mean", "explicit_hard_max"}:
        raise ValueError(
            "background_negative_policy must be pooled_mean or explicit_hard_max"
        )
    if spatial_log_weight < 0 or not 0 < spatial_floor <= 1:
        raise ValueError("spatial_log_weight/spatial_floor are invalid")
    negative_influence = None
    if explicit_negative_influence is not None:
        negative_influence = torch.as_tensor(
            explicit_negative_influence, device=field.device
        ).float().reshape(-1)
        if (
            negative_influence.shape != (field.shape[0],)
            or not bool(torch.isfinite(negative_influence).all())
            or bool((negative_influence < 0).any())
            or bool((negative_influence > 1).any())
        ):
            raise ValueError(
                "explicit_negative_influence must be finite [N] values in [0,1]"
            )
    if calibration is None:
        prototypes = evidence.features.to(field.device)
    else:
        prototypes = calibration.transform(evidence.features)
    weights = evidence.weights.to(field.device).clamp_min(1e-8)
    explicit_negatives = None
    if evidence.negatives is not None:
        explicit_negatives = (
            evidence.negatives.to(field.device)
            if calibration is None
            else calibration.transform(evidence.negatives)
        )

    def prepare_spatial(
        values: torch.Tensor | None,
        prototype_count: int,
        *,
        name: str,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        matrix = torch.as_tensor(values, device=field.device).float()
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if (
            matrix.ndim != 2
            or matrix.shape[0] != field.shape[0]
            or matrix.shape[1] not in {1, int(prototype_count)}
            or not bool(torch.isfinite(matrix).all())
            or bool((matrix < 0).any())
            or bool((matrix > 1).any())
        ):
            raise ValueError(
                f"{name} must be finite [N,1] or [N,{prototype_count}] in [0,1]"
            )
        return (
            matrix.expand(-1, int(prototype_count))
            if matrix.shape[1] == 1 and int(prototype_count) > 1
            else matrix
        )

    positive_spatial = prepare_spatial(
        positive_spatial_influence,
        prototypes.shape[0],
        name="positive_spatial_influence",
    )
    negative_spatial = prepare_spatial(
        explicit_negative_spatial,
        0 if explicit_negatives is None else explicit_negatives.shape[0],
        name="explicit_negative_spatial",
    )
    has_background_bank = (
        calibration is not None and calibration.background_centroids is not None
    )
    if has_background_bank:
        assert calibration is not None
        centroids = calibration.background_centroids
    parts: list[torch.Tensor] = []
    for start in range(0, field.shape[0], int(chunk_size)):
        rows = field[start : start + int(chunk_size)].float()
        rows = (
            F.normalize(rows, dim=-1, eps=1e-8)
            if calibration is None
            else calibration.transform(rows)
        )
        logits = rows @ prototypes.T
        if positive_spatial is not None:
            local_positive_spatial = positive_spatial[
                start : start + rows.shape[0]
            ].clamp_min(float(spatial_floor))
            logits = (
                logits
                + float(spatial_log_weight) * local_positive_spatial.log()
            )
        positive = temperature * torch.logsumexp(
            logits / temperature + weights.log()[None], dim=1
        )
        if explicit_negatives is not None or has_background_bank:
            local_influence = (
                None
                if negative_influence is None
                else negative_influence[start : start + rows.shape[0]]
            )
            explicit_logits = (
                rows @ explicit_negatives.T
                if explicit_negatives is not None
                else None
            )
            local_negative_spatial = None
            if explicit_logits is not None and negative_spatial is not None:
                local_negative_spatial = negative_spatial[
                    start : start + rows.shape[0]
                ].clamp_min(float(spatial_floor))
                explicit_logits = (
                    explicit_logits
                    + float(spatial_log_weight) * local_negative_spatial.log()
                )
            explicit_negative = (
                explicit_logits.amax(dim=1)
                if explicit_logits is not None
                else None
            )
            background_negative = None
            if has_background_bank:
                assert calibration is not None and calibration.background_centroids is not None
                background_logits = rows @ calibration.background_centroids.T
                background_negative = temperature * (
                    torch.logsumexp(background_logits / temperature, dim=1)
                    - torch.log(
                        torch.tensor(
                            calibration.background_centroids.shape[0],
                            device=rows.device,
                            dtype=torch.float32,
                        )
                    )
                )
            if explicit_negative is None:
                assert background_negative is not None
                negative = background_negative
            elif background_negative is None:
                # Preserve the original explicit-negative readout exactly.
                negative = (
                    explicit_negative
                    if local_influence is None
                    else local_influence * explicit_negative
                )
            elif background_negative_policy == "pooled_mean":
                if local_influence is None:
                    # Historical behavior: pool user negatives and scene modes into
                    # one average background bank.
                    if local_negative_spatial is None:
                        negative_logits = rows @ torch.cat(
                            [
                                explicit_negatives,
                                calibration.background_centroids,
                            ],
                            dim=0,
                        ).T
                    else:
                        background_logits = (
                            rows @ calibration.background_centroids.T
                        )
                        negative_logits = torch.cat(
                            [explicit_logits, background_logits], dim=1
                        )
                    negative = temperature * (
                        torch.logsumexp(negative_logits / temperature, dim=1)
                        - torch.log(
                            torch.tensor(
                                negative_logits.shape[1],
                                device=rows.device,
                                dtype=torch.float32,
                            )
                        )
                    )
                else:
                    # Treat graph locality as the participation mass of the
                    # explicit click bank.  At zero influence the scene bank is
                    # unchanged; at one this is exactly the historical pooled
                    # log-mean-exp over both banks.
                    assert explicit_logits is not None
                    assert calibration is not None
                    assert calibration.background_centroids is not None
                    background_logits = rows @ calibration.background_centroids.T
                    log_gate = local_influence.clamp_min(1e-30).log()[:, None]
                    gated_explicit = (
                        explicit_logits / temperature + log_gate
                    ).masked_fill(local_influence[:, None] <= 0, float("-inf"))
                    joined = torch.cat(
                        [gated_explicit, background_logits / temperature], dim=1
                    )
                    count = (
                        local_influence * float(explicit_logits.shape[1])
                        + float(background_logits.shape[1])
                    )
                    negative = temperature * (
                        torch.logsumexp(joined, dim=1) - count.log()
                    )
            else:
                # Scene modes only provide a query-independent prior.  Once a
                # user supplies a concrete negative click, it must remain hard
                # contrastive evidence instead of being diluted by every mode.
                if local_influence is None:
                    localized_explicit = explicit_negative
                else:
                    localized_explicit = (
                        local_influence * explicit_negative
                        + (1.0 - local_influence) * (-1.0)
                    )
                negative = torch.maximum(localized_explicit, background_negative)
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
    explicit_negative_influence: torch.Tensor | None = None,
    positive_spatial_influence: torch.Tensor | None = None,
    explicit_negative_spatial: torch.Tensor | None = None,
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
            background_negative_policy=config.background_negative_policy,
            chunk_size=config.score_chunk_size,
            explicit_negative_influence=explicit_negative_influence,
            positive_spatial_influence=positive_spatial_influence,
            explicit_negative_spatial=explicit_negative_spatial,
            spatial_log_weight=config.spatial_log_weight,
            spatial_floor=config.spatial_floor,
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
