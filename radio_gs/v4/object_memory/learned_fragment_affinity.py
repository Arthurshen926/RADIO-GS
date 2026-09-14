"""Scene-cardinality-independent learned fragment affinity primitives."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PAIR_FEATURE_LAYOUT = (
    "surface_geometry",
    "calibrated_appearance",
    "spatial_proximity",
    "visibility_conflict",
    "geometry_x_appearance",
    "appearance_x_spatial",
    "geometry_x_nonconflict",
    "appearance_x_nonconflict",
)


def fragment_pair_features(
    geometry: torch.Tensor,
    calibrated_appearance: torch.Tensor,
    spatial: torch.Tensor,
    visibility_conflict: torch.Tensor,
) -> torch.Tensor:
    """Build the shared scalar pair contract used in training and deployment."""

    tensors = [
        torch.as_tensor(value, dtype=torch.float32)
        for value in (
            geometry,
            calibrated_appearance,
            spatial,
            visibility_conflict,
        )
    ]
    if any(value.shape != tensors[0].shape for value in tensors):
        raise ValueError("fragment pair components must have identical shapes")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("fragment pair components must be finite")
    if any(bool(((value < 0) | (value > 1)).any()) for value in tensors):
        raise ValueError("fragment pair components must lie in [0,1]")
    geometry, appearance, spatial, conflict = tensors
    nonconflict = 1.0 - conflict
    return torch.stack(
        (
            geometry,
            appearance,
            spatial,
            conflict,
            geometry * appearance,
            appearance * spatial,
            geometry * nonconflict,
            appearance * nonconflict,
        ),
        dim=-1,
    )


class FragmentAffinityMLP(nn.Module):
    """Small shared edge scorer; it has no scene or token identity parameter."""

    def __init__(self, hidden_dimension: int = 32, feature_mode: str = "full") -> None:
        super().__init__()
        if feature_mode not in ("full", "geometry_only"):
            raise ValueError("unsupported affinity feature mode")
        self.feature_mode = feature_mode
        if hidden_dimension <= 0:
            raise ValueError("fragment affinity hidden dimension must be positive")
        self.hidden_dimension = int(hidden_dimension)
        self.network = nn.Sequential(
            nn.Linear(len(PAIR_FEATURE_LAYOUT), hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(pair_features)
        if values.shape[-1] != len(PAIR_FEATURE_LAYOUT):
            raise ValueError("learned fragment pair feature dimension differs")
        if self.feature_mode == "geometry_only":
            values = values * values.new_tensor([1, 0, 1, 1, 0, 0, 1, 0])
        return self.network(values).squeeze(-1)


class SetConditionedFragmentAffinity(nn.Module):
    """Permutation-equivariant edge scorer conditioned on the whole fragment set."""

    set_conditioned = True
    affinity_source = "scene_disjoint_set_conditioned_fragment_affinity"

    def __init__(self, hidden_dimension: int = 32) -> None:
        super().__init__()
        if hidden_dimension <= 0:
            raise ValueError("fragment affinity hidden dimension must be positive")
        self.hidden_dimension = int(hidden_dimension)
        pair_dimension = len(PAIR_FEATURE_LAYOUT) + 1
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(5 * hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(
        self, pair_features: torch.Tensor, view_index: torch.Tensor
    ) -> torch.Tensor:
        values = torch.as_tensor(pair_features)
        views = torch.as_tensor(view_index, dtype=torch.long, device=values.device)
        if (
            values.ndim != 3
            or values.shape[0] != values.shape[1]
            or values.shape[2] != len(PAIR_FEATURE_LAYOUT)
        ):
            raise ValueError("set-conditioned pair features must have shape [F,F,D]")
        if views.shape != (values.shape[0],):
            raise ValueError("set-conditioned fragment views do not align")
        same_view = (views[:, None] == views[None, :]).to(values.dtype)
        encoded = self.pair_encoder(torch.cat((values, same_view[..., None]), dim=-1))
        valid = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
        count = valid.sum(-1, keepdim=True).clamp_min(1).to(encoded.dtype)
        mean_context = (encoded * valid[..., None]).sum(1) / count
        max_context = encoded.masked_fill(~valid[..., None], -torch.inf).max(1).values
        max_context = torch.where(
            torch.isfinite(max_context), max_context, torch.zeros_like(max_context)
        )
        context = self.context_encoder(torch.cat((mean_context, max_context), dim=-1))
        first = context[:, None].expand(-1, values.shape[0], -1)
        second = context[None].expand(values.shape[0], -1, -1)
        edge = torch.cat(
            (encoded, first, second, (first - second).abs(), first * second), dim=-1
        )
        logits = self.edge_head(edge).squeeze(-1)
        return 0.5 * (logits + logits.T)


def balanced_fragment_affinity_loss(
    logits: torch.Tensor, target_same_object: torch.Tensor
) -> torch.Tensor:
    """Give positive and negative edge populations equal total weight."""

    values = torch.as_tensor(logits)
    target = torch.as_tensor(
        target_same_object, dtype=torch.bool, device=values.device
    )
    if values.shape != target.shape or values.ndim != 1:
        raise ValueError("fragment affinity logits and labels must be aligned vectors")
    if not bool(target.any()) or not bool((~target).any()):
        raise ValueError("fragment affinity loss needs positive and negative pairs")
    positive = F.softplus(-values[target]).mean()
    negative = F.softplus(values[~target]).mean()
    return 0.5 * (positive + negative)


@torch.no_grad()
def fragment_affinity_metrics(
    probability: torch.Tensor,
    target_same_object: torch.Tensor,
    *,
    decision_threshold: float = 0.5,
) -> dict[str, float | int]:
    probability = torch.as_tensor(probability, dtype=torch.float32).cpu()
    target = torch.as_tensor(target_same_object, dtype=torch.bool).cpu()
    if probability.shape != target.shape or probability.ndim != 1:
        raise ValueError("fragment affinity metric inputs must be aligned vectors")
    if not probability.numel() or not bool(torch.isfinite(probability).all()):
        raise ValueError("fragment affinity probabilities must be finite and non-empty")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("fragment affinity probabilities must lie in [0,1]")
    if not 0 <= decision_threshold <= 1:
        raise ValueError("fragment affinity decision threshold must lie in [0,1]")
    prediction = probability >= decision_threshold
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & ~target).sum())
    false_negative = int((~prediction & target).sum())
    true_negative = int((~prediction & ~target).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    order = torch.argsort(probability, descending=True, stable=True)
    ranked_target = target[order]
    cumulative_positive = ranked_target.cumsum(0)
    rank = torch.arange(1, target.numel() + 1, dtype=torch.float32)
    average_precision = float(
        (cumulative_positive.float() / rank)[ranked_target].sum()
        / ranked_target.sum().clamp_min(1)
    )
    return {
        "pair_count": int(target.numel()),
        "positive_pair_count": int(target.sum()),
        "negative_pair_count": int((~target).sum()),
        "decision_threshold": float(decision_threshold),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "average_precision": average_precision,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "true_negative_count": true_negative,
    }


@torch.no_grad()
def select_fragment_affinity_threshold(
    probability: torch.Tensor,
    target_same_object: torch.Tensor,
    *,
    beta: float = 0.5,
) -> dict[str, float]:
    """Select one validation-only F-beta operating point.

    ``beta < 1`` reflects the asymmetric global-clustering cost: one false
    merge can contaminate a whole object, whereas one missed edge usually only
    leaves a recoverable split.
    """

    values = torch.as_tensor(probability, dtype=torch.float32).cpu()
    target = torch.as_tensor(target_same_object, dtype=torch.bool).cpu()
    if values.shape != target.shape or values.ndim != 1 or not values.numel():
        raise ValueError("threshold selection inputs must be aligned vectors")
    if beta <= 0 or not bool(target.any()) or not bool((~target).any()):
        raise ValueError("threshold selection needs beta and both pair classes")
    candidates = torch.unique(
        torch.cat((values, values.new_tensor([0.0, 1.0])))
    ).sort().values
    best = None
    beta2 = beta * beta
    for threshold in candidates.tolist():
        prediction = values >= threshold
        true_positive = int((prediction & target).sum())
        false_positive = int((prediction & ~target).sum())
        false_negative = int((~prediction & target).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        denominator = beta2 * precision + recall
        f_beta = (
            (1 + beta2) * precision * recall / denominator
            if denominator > 0 else 0.0
        )
        candidate = (f_beta, precision, recall, threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {
        "decision_threshold": float(best[3]),
        "f_beta": float(best[0]),
        "precision": float(best[1]),
        "recall": float(best[2]),
        "beta": float(beta),
    }


def checkpoint_model(checkpoint: dict[str, Any]) -> nn.Module:
    if checkpoint.get("schema") != "radio_gs.surface_object_memory_v4.fragment_affinity_checkpoint.v1":
        raise ValueError("fragment affinity checkpoint schema differs")
    if tuple(checkpoint.get("pair_feature_layout", ())) != PAIR_FEATURE_LAYOUT:
        raise ValueError("fragment affinity checkpoint feature layout differs")
    model_kind = str(checkpoint.get("model_kind", "pair_mlp"))
    if model_kind == "pair_mlp":
        model = FragmentAffinityMLP(int(checkpoint["hidden_dimension"]), str(checkpoint.get("feature_mode", "full")))
    elif model_kind == "set_context_mlp":
        model = SetConditionedFragmentAffinity(int(checkpoint["hidden_dimension"]))
    else:
        raise ValueError("fragment affinity checkpoint model kind differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


__all__ = [
    "PAIR_FEATURE_LAYOUT",
    "FragmentAffinityMLP",
    "SetConditionedFragmentAffinity",
    "balanced_fragment_affinity_loss",
    "checkpoint_model",
    "fragment_affinity_metrics",
    "fragment_pair_features",
    "select_fragment_affinity_threshold",
]
