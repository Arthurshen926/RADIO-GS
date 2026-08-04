"""Target-blind signed-scribble cross-validation for propagation selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


ACTION_STRONG_UNARY = "strong_unary"
ACTION_HASH256_DIFFUSION = "hash256_fixed_f2_g4_k201_diffusion"
REGISTERED_ACTIONS = (ACTION_STRONG_UNARY, ACTION_HASH256_DIFFUSION)


def select_registered_action(
    metrics: dict[str, dict[str, float]], *, metric_round_decimals: int = 12
) -> str:
    """Apply the registered log-loss/AUC/complexity lexicographic rule."""

    if set(metrics) != set(REGISTERED_ACTIONS):
        raise ValueError("selection metrics must contain exactly the registered actions")
    rounded = {}
    for action in REGISTERED_ACTIONS:
        values = metrics[action]
        if set(values) != {
            "responsibility_balanced_log_loss",
            "responsibility_weighted_auc",
        }:
            raise ValueError("selection metric schema differs")
        loss = float(values["responsibility_balanced_log_loss"])
        auc = float(values["responsibility_weighted_auc"])
        if not np.isfinite(loss) or not np.isfinite(auc) or loss < 0 or not 0 <= auc <= 1:
            raise ValueError("selection metric value is invalid")
        rounded[action] = (
            round(loss, int(metric_round_decimals)),
            -round(auc, int(metric_round_decimals)),
            0 if action == ACTION_STRONG_UNARY else 1,
        )
    return min(REGISTERED_ACTIONS, key=lambda action: rounded[action])


def stable_primitive_folds(global_rows: torch.Tensor, *, num_folds: int = 3) -> torch.Tensor:
    """Assign global primitive ids with a platform-stable SplitMix64 hash."""

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    if int(num_folds) < 2 or rows.numel() == 0 or bool((rows < 0).any()):
        raise ValueError("fold assignment requires non-negative rows and >=2 folds")
    if rows.unique().numel() != rows.numel():
        raise ValueError("fold assignment requires unique global rows")
    values = rows.numpy().astype(np.uint64, copy=True)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    values ^= values >> np.uint64(31)
    folds = (values % np.uint64(int(num_folds))).astype(np.int64, copy=False)
    return torch.from_numpy(folds.copy())


def training_evidence_for_fold(
    signed_reference_evidence: torch.Tensor,
    fold_ids: torch.Tensor,
    *,
    heldout_fold: int,
) -> torch.Tensor:
    """Remove every held-out signed anchor before classifier and diffusion fit."""

    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    folds = torch.as_tensor(fold_ids).detach().long().cpu().reshape(-1)
    if evidence.shape != folds.shape or not bool(torch.isfinite(evidence).all()):
        raise ValueError("evidence and fold ids must be finite and aligned")
    if int(heldout_fold) < 0 or int(heldout_fold) >= int(folds.max()) + 1:
        raise ValueError("held-out fold is outside the fold assignment")
    training = evidence.clone()
    training[folds == int(heldout_fold)] = 0
    return training


def audit_signed_cv_population(
    global_rows: torch.Tensor,
    signed_reference_evidence: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    num_folds: int = 3,
    minimum_class_rows: int = 32,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Fail closed unless every held-out and training fold has both classes."""

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    weights = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    if rows.shape != evidence.shape or rows.shape != weights.shape:
        raise ValueError("CV rows, evidence, and weights must align")
    if not bool(torch.isfinite(evidence).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("CV evidence and weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("CV responsibility weights cannot be negative")
    observed = evidence != 0
    if not bool(observed.any()) or not bool((weights[observed] > 0).all()):
        raise ValueError("every CV anchor needs positive responsibility")
    folds = stable_primitive_folds(rows, num_folds=int(num_folds))
    labels = evidence > 0
    minimum = int(minimum_class_rows)
    if minimum <= 0:
        raise ValueError("minimum_class_rows must be positive")
    reports: list[dict[str, object]] = []
    for fold in range(int(num_folds)):
        heldout = observed & (folds == fold)
        training = observed & (folds != fold)
        report: dict[str, object] = {"fold": fold}
        for population_name, mask in (("heldout", heldout), ("training", training)):
            positive = mask & labels
            negative = mask & ~labels
            positive_count = int(positive.sum())
            negative_count = int(negative.sum())
            positive_weight = float(weights[positive].sum())
            negative_weight = float(weights[negative].sum())
            report[f"{population_name}_positive_rows"] = positive_count
            report[f"{population_name}_negative_rows"] = negative_count
            report[f"{population_name}_positive_weight"] = positive_weight
            report[f"{population_name}_negative_weight"] = negative_weight
            if (
                positive_count < minimum
                or negative_count < minimum
                or positive_weight <= 0
                or negative_weight <= 0
            ):
                raise ValueError(
                    f"fold {fold} {population_name} lacks the registered signed CV population"
                )
        reports.append(report)
    return folds, reports


def responsibility_balanced_log_loss(
    labels: torch.Tensor,
    probability: torch.Tensor,
    reference_weight: torch.Tensor,
    *,
    probability_epsilon: float = 1e-7,
) -> float:
    """Give positive and negative responsibility mass equal total influence."""

    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    score = torch.as_tensor(probability).detach().double().cpu().reshape(-1)
    weight = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    if target.shape != score.shape or target.shape != weight.shape:
        raise ValueError("balanced log-loss inputs must align")
    if not bool(torch.isfinite(score).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("balanced log-loss inputs must be finite")
    if bool((weight <= 0).any()) or not bool(target.any()) or not bool((~target).any()):
        raise ValueError("balanced log-loss requires positive weight and both classes")
    epsilon = float(probability_epsilon)
    if not 0 < epsilon < 0.5:
        raise ValueError("probability_epsilon must be in (0,0.5)")
    score = score.clamp(epsilon, 1.0 - epsilon)
    positive_weight = weight[target]
    negative_weight = weight[~target]
    positive = -(positive_weight * score[target].log()).sum() / positive_weight.sum()
    negative = -(negative_weight * (1.0 - score[~target]).log()).sum() / negative_weight.sum()
    return float(0.5 * (positive + negative))


def responsibility_weighted_auc(
    labels: torch.Tensor,
    probability: torch.Tensor,
    reference_weight: torch.Tensor,
) -> float:
    from sklearn.metrics import roc_auc_score

    target = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    score = torch.as_tensor(probability).detach().double().cpu().reshape(-1)
    weight = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    if target.shape != score.shape or target.shape != weight.shape:
        raise ValueError("weighted AUC inputs must align")
    if (
        not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(weight).all())
        or bool((weight <= 0).any())
        or not bool(target.any())
        or not bool((~target).any())
    ):
        raise ValueError("weighted AUC requires finite positive weights and both classes")
    return float(
        roc_auc_score(
            target.numpy().astype(np.int64, copy=False),
            score.numpy(),
            sample_weight=weight.numpy(),
        )
    )


@dataclass(frozen=True)
class SignedScribbleCVResult:
    selected_action: str
    metrics: dict[str, dict[str, float]]
    folds: torch.Tensor
    fold_reports: list[dict[str, object]]
    oof_predictions: dict[str, torch.Tensor]
    observed: torch.Tensor


@torch.inference_mode()
def run_signed_scribble_cross_validation(
    global_rows: torch.Tensor,
    signed_reference_evidence: torch.Tensor,
    reference_weight: torch.Tensor,
    predictor: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    num_folds: int = 3,
    minimum_class_rows: int = 32,
    metric_round_decimals: int = 12,
    probability_epsilon: float = 1e-7,
) -> SignedScribbleCVResult:
    """Generate strict OOF predictions and select one registered action."""

    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1)
    evidence = torch.as_tensor(signed_reference_evidence).detach().float().cpu().reshape(-1)
    weights = torch.as_tensor(reference_weight).detach().double().cpu().reshape(-1)
    folds, reports = audit_signed_cv_population(
        rows,
        evidence,
        weights,
        num_folds=int(num_folds),
        minimum_class_rows=int(minimum_class_rows),
    )
    observed = evidence != 0
    oof = {
        ACTION_STRONG_UNARY: torch.full(evidence.shape, float("nan"), dtype=torch.float32),
        ACTION_HASH256_DIFFUSION: torch.full(
            evidence.shape, float("nan"), dtype=torch.float32
        ),
    }
    for fold in range(int(num_folds)):
        training = training_evidence_for_fold(evidence, folds, heldout_fold=fold)
        if bool((training[folds == fold] != 0).any()):
            raise RuntimeError("held-out evidence leaked into the fold predictor")
        unary, diffusion = predictor(training, fold)
        unary = torch.as_tensor(unary).detach().float().cpu().reshape(-1)
        diffusion = torch.as_tensor(diffusion).detach().float().cpu().reshape(-1)
        if unary.shape != evidence.shape or diffusion.shape != evidence.shape:
            raise ValueError("CV predictor outputs must align with primitive rows")
        if (
            not bool(torch.isfinite(unary).all())
            or not bool(torch.isfinite(diffusion).all())
            or bool(((unary < 0) | (unary > 1)).any())
            or bool(((diffusion < 0) | (diffusion > 1)).any())
        ):
            raise ValueError("CV predictor returned an invalid probability")
        heldout = observed & (folds == fold)
        oof[ACTION_STRONG_UNARY][heldout] = unary[heldout]
        oof[ACTION_HASH256_DIFFUSION][heldout] = diffusion[heldout]
    for action in REGISTERED_ACTIONS:
        if not bool(torch.isfinite(oof[action][observed]).all()):
            raise RuntimeError("CV did not produce every held-out prediction")
    labels = evidence[observed] > 0
    observed_weights = weights[observed]
    metrics: dict[str, dict[str, float]] = {}
    for action in REGISTERED_ACTIONS:
        probability = oof[action][observed]
        metrics[action] = {
            "responsibility_balanced_log_loss": responsibility_balanced_log_loss(
                labels,
                probability,
                observed_weights,
                probability_epsilon=float(probability_epsilon),
            ),
            "responsibility_weighted_auc": responsibility_weighted_auc(
                labels, probability, observed_weights
            ),
        }
    selected = select_registered_action(
        metrics, metric_round_decimals=int(metric_round_decimals)
    )
    return SignedScribbleCVResult(
        selected_action=selected,
        metrics=metrics,
        folds=folds,
        fold_reports=reports,
        oof_predictions=oof,
        observed=observed,
    )
