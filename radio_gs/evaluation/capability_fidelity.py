"""Label-free dense and local-relation fidelity metrics for frozen adaptors."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def dense_cosine_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target must be matching [C,H,W]")
    if valid.shape != prediction.shape[1:]:
        raise ValueError("valid mask shape mismatch")
    predicted = prediction.permute(1, 2, 0)[valid].float()
    teacher = target.permute(1, 2, 0)[valid].float()
    if predicted.numel() == 0:
        return torch.empty(0, device=prediction.device)
    return F.cosine_similarity(predicted, teacher, dim=-1, eps=1e-8)


def local_affinity_pairs(
    features: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return horizontal and vertical normalized-token affinities."""

    if features.ndim != 3 or valid.shape != features.shape[1:]:
        raise ValueError("features must be [C,H,W] with matching valid mask")
    values = F.normalize(features.float(), dim=0, eps=1e-8)
    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    horizontal = (values[:, :, :-1] * values[:, :, 1:]).sum(dim=0)[horizontal_valid]
    vertical = (values[:, :-1, :] * values[:, 1:, :]).sum(dim=0)[vertical_valid]
    return torch.cat([horizontal, vertical])


def relation_fidelity_summary(
    predicted_affinity: torch.Tensor,
    target_affinity: torch.Tensor,
    *,
    boundary_quantile: float = 0.2,
) -> dict[str, float | int | None]:
    predicted = torch.as_tensor(predicted_affinity).detach().float().cpu().reshape(-1)
    target = torch.as_tensor(target_affinity).detach().float().cpu().reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target affinity arrays must align")
    if not 0 < boundary_quantile < 0.5:
        raise ValueError("boundary_quantile must be in (0,0.5)")
    if predicted.numel() == 0:
        return {
            "pairs": 0,
            "affinity_mae": None,
            "affinity_pearson": None,
            "teacher_boundary_margin": None,
            "predicted_boundary_margin": None,
            "boundary_margin_retention": None,
        }
    pearson = None
    if float(predicted.std(unbiased=False)) > 0 and float(target.std(unbiased=False)) > 0:
        pearson = float(torch.corrcoef(torch.stack([predicted, target]))[0, 1])
    low_threshold = torch.quantile(target, boundary_quantile)
    high_threshold = torch.quantile(target, 1.0 - boundary_quantile)
    boundary = target <= low_threshold
    interior = target >= high_threshold
    teacher_margin = float(target[interior].mean() - target[boundary].mean())
    predicted_margin = float(predicted[interior].mean() - predicted[boundary].mean())
    return {
        "pairs": int(target.numel()),
        "affinity_mae": float((predicted - target).abs().mean()),
        "affinity_pearson": pearson,
        "teacher_boundary_margin": teacher_margin,
        "predicted_boundary_margin": predicted_margin,
        "boundary_margin_retention": (
            predicted_margin / teacher_margin if abs(teacher_margin) > 1e-8 else None
        ),
    }


def dense_fidelity_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    cosine = torch.as_tensor(values).detach().float().cpu().reshape(-1)
    if cosine.numel() == 0:
        return {"pixels": 0, "mean_cosine": None, "p05_cosine": None}
    return {
        "pixels": int(cosine.numel()),
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(torch.quantile(cosine, 0.05)),
    }


def select_query_free_compositor(
    variants: dict[str, dict],
    *,
    baseline: str = "alpha_mean",
    max_mean_dense_drop: float = 0.005,
    max_p05_dense_drop: float = 0.01,
    max_unsupported_fraction: float = 0.005,
    min_relation_gain: float = 0.005,
) -> dict:
    """Select a label-free compositor under frozen dense-fidelity guards.

    The objective averages official DINO/SAM local-affinity Pearson and
    boundary-margin retention.  Raw/DINO/SAM mean and lower-tail dense cosine
    are hard non-inferiority constraints relative to ordinary alpha blending.
    """

    if baseline not in variants:
        raise ValueError(f"missing baseline compositor {baseline!r}")
    spaces = ("raw_radio", "official_dino_v3", "official_sam3")
    relation_spaces = ("official_dino_v3", "official_sam3")

    def dense_value(report: dict, space: str, key: str) -> float:
        value = report[space].get(key)
        if value is None:
            raise ValueError(f"undefined {space}/{key} in compositor report")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite {space}/{key} in compositor report")
        return result

    def relation_score(report: dict) -> float:
        values: list[float] = []
        for space in relation_spaces:
            relation = report[space]["local_relation"]
            for key in ("affinity_pearson", "boundary_margin_retention"):
                value = relation.get(key)
                if value is None:
                    raise ValueError(f"undefined {space}/{key} in compositor report")
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError(
                        f"non-finite {space}/{key} in compositor report"
                    )
                values.append(result)
        return sum(values) / len(values)

    reference = variants[baseline]
    baseline_score = relation_score(reference)
    candidates: dict[str, dict] = {}
    for name, report in variants.items():
        mean_drops = {
            space: dense_value(reference, space, "mean_cosine")
            - dense_value(report, space, "mean_cosine")
            for space in spaces
        }
        p05_drops = {
            space: dense_value(reference, space, "p05_cosine")
            - dense_value(report, space, "p05_cosine")
            for space in spaces
        }
        score = relation_score(report)
        if "support_fraction_on_visible" not in report:
            raise ValueError(
                f"{name}: capability report does not declare visible support"
            )
        support_fraction = float(report["support_fraction_on_visible"])
        if not math.isfinite(support_fraction) or not 0.0 <= support_fraction <= 1.0:
            raise ValueError(f"{name}: visible support fraction is invalid")
        unsupported_fraction = 1.0 - support_fraction
        dense_guard_passed = (
            max(mean_drops.values()) <= float(max_mean_dense_drop)
            and max(p05_drops.values()) <= float(max_p05_dense_drop)
        )
        support_guard_passed = (
            unsupported_fraction <= float(max_unsupported_fraction)
        )
        eligible = dense_guard_passed and support_guard_passed
        candidates[name] = {
            "eligible": bool(eligible),
            "dense_guard_passed": bool(dense_guard_passed),
            "support_guard_passed": bool(support_guard_passed),
            "unsupported_fraction": unsupported_fraction,
            "mean_dense_drop": mean_drops,
            "p05_dense_drop": p05_drops,
            "official_relation_score": score,
            "relation_gain_over_alpha_mean": score - baseline_score,
        }
    eligible_names = [name for name, values in candidates.items() if values["eligible"]]
    selection_status = "candidate_selected"
    best: str | None = None
    if not eligible_names:
        selection_status = (
            "support_gate_failed_no_promotion"
            if all(
                not values["support_guard_passed"]
                for values in candidates.values()
            )
            else "no_eligible_candidate_no_promotion"
        )
    else:
        best = max(
            eligible_names,
            key=lambda name: (
                candidates[name]["official_relation_score"],
                -sum(candidates[name]["mean_dense_drop"].values()),
                name == baseline,
            ),
        )
        if (
            candidates[best]["relation_gain_over_alpha_mean"]
            < float(min_relation_gain)
        ):
            if candidates[baseline]["eligible"]:
                best = baseline
                selection_status = (
                    "baseline_retained_relation_gain_below_threshold"
                )
            else:
                best = None
                selection_status = (
                    "relation_gain_gate_failed_no_promotion"
                )
    return {
        "selected_variant": best,
        "baseline_variant": baseline,
        "selection_status": selection_status,
        "promotion_allowed": bool(
            best is not None and best != baseline
        ),
        "selection_uses_task_labels": False,
        "rule": (
            "mean dense drop <= max_mean_dense_drop and p05 dense drop <= "
            "max_p05_dense_drop in raw/DINO/SAM; then maximize mean official "
            "DINO/SAM affinity Pearson and boundary-margin retention; require "
            "min_relation_gain over alpha_mean"
        ),
        "thresholds": {
            "max_mean_dense_drop": float(max_mean_dense_drop),
            "max_p05_dense_drop": float(max_p05_dense_drop),
            "max_unsupported_fraction": float(max_unsupported_fraction),
            "min_relation_gain": float(min_relation_gain),
        },
        "candidates": candidates,
    }
