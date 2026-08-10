#!/usr/bin/env python3
"""Evaluate conservative directional transport admission on sealed SPIn OOF folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.querying.source_oof_transport_admission import (
    apply_source_oof_directional_admission,
    fit_conservative_directional_admission,
    method_contract,
)
from radio_gs.utils.immutable_artifacts import sha256_file, write_frozen_json


def _metrics(
    probability: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, float | int]:
    weight = positive + negative
    use = eligible & (weight > 0)
    if not bool(use.any()):
        raise ValueError("source-OOF metric population is empty")
    target = torch.where(weight > 0, positive / weight.clamp_min(1e-15), 0.0)
    score = probability[use].double().clamp(1e-7, 1.0 - 1e-7)
    label = target[use].double()
    mass = weight[use].double()
    total = mass.sum()
    brier = (mass * (score - label).square()).sum() / total
    log_loss = (
        mass
        * (-label * score.log() - (1.0 - label) * (1.0 - score).log())
    ).sum() / total
    selected = score >= 0.5
    positive_mass = mass * label
    negative_mass = mass * (1.0 - label)
    union = positive_mass.sum() + negative_mass[selected].sum()
    soft_iou = positive_mass[selected].sum() / union
    return {
        "rows": int(use.sum()),
        "responsibility_mass": float(total),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "soft_iou_at_0_5": float(soft_iou),
    }


def _load_fold(path: Path, fold: int) -> dict[str, object]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("SPIn source-OOF fold must be a mapping")
    if int(payload.get("heldout_fold", -1)) != fold:
        raise ValueError("SPIn source-OOF heldout-fold identity differs")
    if any(
        payload.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")
    ):
        raise ValueError("SPIn source-OOF fold violates target-blind safety")
    return payload


def evaluate(fold_paths: tuple[Path, Path, Path]) -> dict[str, object]:
    folds = tuple(_load_fold(path, index) for index, path in enumerate(fold_paths))
    reference = folds[0]
    invariant_keys = (
        "scene_id",
        "protocol_hash",
        "fold_assignment",
        "num_folds",
        "capability_cache_sha256",
        "support_graph_sha256",
        "source_evidence_authority_sha256",
        "source_footprint_fold_authority_sha256",
    )
    for payload in folds[1:]:
        if any(payload.get(key) != reference.get(key) for key in invariant_keys):
            raise ValueError("SPIn source-OOF authority differs across folds")
    tensor_invariants = (
        "valid",
        "observed",
        "fold_ids",
        "population_positive_weight",
        "population_negative_weight",
    )
    for key in tensor_invariants:
        first = torch.as_tensor(reference[key]).detach().cpu()
        if any(not torch.equal(first, torch.as_tensor(item[key]).detach().cpu()) for item in folds[1:]):
            raise ValueError(f"SPIn source-OOF invariant tensor differs: {key}")

    valid = torch.as_tensor(reference["valid"]).bool().cpu().reshape(-1)
    observed = torch.as_tensor(reference["observed"]).bool().cpu().reshape(-1)
    fold_ids = torch.as_tensor(reference["fold_ids"]).long().cpu().reshape(-1)
    positive = torch.as_tensor(reference["population_positive_weight"]).float().cpu().reshape(-1)
    negative = torch.as_tensor(reference["population_negative_weight"]).float().cpu().reshape(-1)
    count = valid.numel()
    anchor = torch.zeros(count, dtype=torch.float32)
    proposal = torch.zeros(count, dtype=torch.float32)
    pooled_eligible = torch.zeros(count, dtype=torch.bool)
    heldout_populations: list[torch.Tensor] = []
    for fold, payload in enumerate(folds):
        heldout = torch.as_tensor(payload["heldout"]).bool().cpu().reshape(-1)
        expected = valid & (fold_ids == fold)
        if not torch.equal(heldout, expected):
            raise ValueError("SPIn source-OOF heldout rows differ from fold authority")
        eligible = heldout & observed & ((positive + negative) > 0)
        if bool((pooled_eligible & eligible).any()):
            raise ValueError("SPIn source-OOF heldout metric populations overlap")
        unary = torch.as_tensor(payload["unary_probability"]).float().cpu().reshape(-1)
        propagated = torch.as_tensor(
            payload["surface_safe_propagated_probability"]
        ).float().cpu().reshape(-1)
        if unary.shape != valid.shape or propagated.shape != valid.shape:
            raise ValueError("SPIn source-OOF probability domain differs")
        anchor[eligible] = unary[eligible]
        proposal[eligible] = propagated[eligible]
        pooled_eligible |= eligible
        heldout_populations.append(eligible)

    calibration = fit_conservative_directional_admission(
        anchor,
        proposal,
        positive,
        negative,
        pooled_eligible,
        fold_ids,
    )
    candidate = apply_source_oof_directional_admission(
        anchor,
        proposal,
        torch.zeros_like(anchor),
        calibration,
        active_domain=pooled_eligible,
    ).probability
    macro = {
        "anchor": _metrics(anchor, positive, negative, pooled_eligible),
        "proposal": _metrics(proposal, positive, negative, pooled_eligible),
        "candidate": _metrics(candidate, positive, negative, pooled_eligible),
    }
    fold_metrics = {
        str(fold): {
            "anchor": _metrics(anchor, positive, negative, population),
            "proposal": _metrics(proposal, positive, negative, population),
            "candidate": _metrics(candidate, positive, negative, population),
        }
        for fold, population in enumerate(heldout_populations)
    }
    gate_checks = {
        "candidate_log_loss_below_anchor": (
            macro["candidate"]["log_loss"] < macro["anchor"]["log_loss"]
        ),
        "candidate_log_loss_below_proposal": (
            macro["candidate"]["log_loss"] < macro["proposal"]["log_loss"]
        ),
        "candidate_brier_below_anchor": (
            macro["candidate"]["brier"] < macro["anchor"]["brier"]
        ),
        "candidate_brier_below_proposal": (
            macro["candidate"]["brier"] < macro["proposal"]["brier"]
        ),
        "candidate_soft_iou_nonregress_proposal": (
            macro["candidate"]["soft_iou_at_0_5"]
            >= macro["proposal"]["soft_iou_at_0_5"]
        ),
    }
    return {
        "schema": "radio_gs.spin_source_oof_directional_admission_result.v1",
        "schema_version": 1,
        "scene_id": str(reference["scene_id"]),
        "method": method_contract(),
        "calibration": {
            "expansion": calibration.expansion,
            "contraction": calibration.contraction,
            "leave_one_fold_expansion": list(calibration.leave_one_fold_expansion),
            "leave_one_fold_contraction": list(calibration.leave_one_fold_contraction),
            "folds": list(calibration.folds),
            "eligible_rows": calibration.eligible_rows,
        },
        "macro": macro,
        "folds": fold_metrics,
        "gate_checks": gate_checks,
        "source_gate_passed": all(gate_checks.values()),
        "safety": {
            "source_only": True,
            "target_rgb_opened": False,
            "target_mask_opened": False,
            "target_metric_computed": False,
            "connected_selection": False,
        },
        "inputs": {
            str(index): {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for index, path in enumerate(fold_paths)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-0", type=Path, required=True)
    parser.add_argument("--fold-1", type=Path, required=True)
    parser.add_argument("--fold-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate((args.fold_0, args.fold_1, args.fold_2))
    write_frozen_json(args.output, result)
    print(
        f"{result['scene_id']}: source_gate_passed={result['source_gate_passed']} "
        f"expansion={result['calibration']['expansion']:.6f} "
        f"contraction={result['calibration']['contraction']:.6f}"
    )


if __name__ == "__main__":
    main()
