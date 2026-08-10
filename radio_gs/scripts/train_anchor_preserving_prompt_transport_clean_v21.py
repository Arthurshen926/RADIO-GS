#!/usr/bin/env python3
"""Scene0001-only risk-sensitive fit with one untouched scene0003 gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import radio_gs.querying.anchor_preserving_transport as transport_module
from radio_gs.querying.anchor_preserving_transport import method_contract
from radio_gs.querying.registered_evidence_to_unary import RegisteredEvidenceToUnaryV2
from radio_gs.scripts import train_anchor_preserving_prompt_transport_clean as base
from radio_gs.scripts import train_registered_evidence_to_unary_clean_pilot as legacy
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


SEED = 260810
CVaR_FRACTION = 0.25
UNIFORM_MIXTURE = 0.5
CHECKPOINT_SCHEMA = "radio_gs.anchor_preserving_prompt_transport.checkpoint.v2_1"


def cvar_prompt_weights(losses: torch.Tensor) -> torch.Tensor:
    """Return mean-one weights for 0.5 mean + 0.5 worst-quartile CVaR."""

    values = torch.as_tensor(losses).detach().double().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("prompt losses must be a nonempty finite vector")
    count = int(values.numel())
    tail_count = max(1, int(math.ceil(CVaR_FRACTION * count)))
    # Stable sort plus caller-defined deterministic prompt order resolves ties.
    worst = torch.argsort(values, descending=True, stable=True)[:tail_count]
    weights = torch.full((count,), UNIFORM_MIXTURE, dtype=torch.float64)
    weights[worst] += (1.0 - UNIFORM_MIXTURE) * count / tail_count
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("CVaR prompt weights must have unit mean")
    return weights.float()


def risk_sensitive_selection_key(result: dict, epoch: int) -> tuple[float, ...]:
    """Prioritize per-prompt lower-tail transfer before macro capability."""

    deltas = sorted(
        float(row["delta"]["average_precision"]) for row in result["records"]
    )
    tail_count = max(1, int(math.ceil(CVaR_FRACTION * len(deltas))))
    lower_tail_mean = float(sum(deltas[:tail_count]) / tail_count)
    macro = result["macro"]
    per_mode_ap = [
        macro[mode]["candidate"]["average_precision"]
        - macro[mode]["analytic"]["average_precision"]
        for mode in ("full_mask", "scribble")
    ]
    per_mode_iou = [
        macro[mode]["candidate"]["iou_at_0_5"]
        - macro[mode]["analytic"]["iou_at_0_5"]
        for mode in ("full_mask", "scribble")
    ]
    return (
        deltas[0],
        lower_tail_mean,
        min(per_mode_ap),
        min(per_mode_iou),
        macro["all"]["candidate"]["average_precision"],
        -macro["all"]["candidate"]["bce"],
        -int(epoch),
    )


def _example_loss(
    head: RegisteredEvidenceToUnaryV2,
    example: legacy.PromptExample,
    view: legacy.SparseView,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = legacy._view_rows(view, device)
    subset = legacy._subset_features(example.features, unique)
    output = head(subset)
    prediction, supported = legacy._render_unique(
        output.foreground_probability, view, unique, inverse, device
    )
    target = (view.instance_image.to(device) == example.instance_id).float()
    data_loss = legacy._balanced_bce(prediction[supported], target[supported])
    residual_loss = output.bounded_logit_residual.square().mean()
    rendered_confidence, _ = legacy._render_unique(
        output.confidence, view, unique, inverse, device
    )
    confidence_target = 1.0 - (prediction.detach() - target).abs()
    confidence_loss = F.mse_loss(
        rendered_confidence[supported], confidence_target[supported]
    )
    total = data_loss + 0.05 * residual_loss + 0.02 * confidence_loss
    return total, data_loss


def checkpoint_payload(
    *,
    state_dict: dict,
    best_epoch: int,
    result_path: Path,
    result_sha256: str,
    lineage: dict[str, str],
) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "state_dict": state_dict,
        "best_epoch": int(best_epoch),
        "architecture": {
            "hidden_dim": 32,
            "max_delta_logit": 4.0,
            "fully_observed_tolerance": 1e-5,
        },
        "risk_sensitive_training": {
            "objective": "0.5_prompt_mean_plus_0.5_worst_quartile_cvar",
            "cvar_fraction": CVaR_FRACTION,
            "uniform_mixture": UNIFORM_MIXTURE,
        },
        "lineage": dict(lineage),
        "result_path": str(result_path.resolve()),
        "result_sha256": result_sha256,
    }


def _verify_preregistration(args: argparse.Namespace) -> tuple[dict, str, dict[str, str]]:
    prereg, prereg_sha, _ = load_json_object(
        args.preregistration,
        expected_sha256=args.expected_preregistration_sha256,
        label="V2.1 prompt-transport preregistration",
    )
    if (
        prereg.get("schema")
        != "radio_gs.anchor_preserving_prompt_transport.clean_gate_preregistration.v2_1"
        or prereg.get("fit_scene") != "scene0001_00"
        or prereg.get("confirmation_scene") != "scene0003_00"
        or int(prereg.get("epochs", -1)) != int(args.epochs)
    ):
        raise ValueError("V2.1 preregistration differs")
    authority = prereg.get("authority")
    if not isinstance(authority, dict):
        raise ValueError("V2.1 preregistered authority differs")
    lineage = {
        "fit_authority_sha256": base._verify_preregistered_record(
            authority.get("fit_asset_authority"), args.fit_authority, label="fit asset authority"
        ),
        "confirmation_authority_sha256": base._verify_preregistered_record(
            authority.get("confirmation_asset_authority"),
            args.confirmation_authority,
            label="confirmation asset authority",
        ),
        "trainer_sha256": base._verify_preregistered_record(
            authority.get("implementation"), Path(__file__), label="V2.1 trainer"
        ),
        "base_asset_loader_sha256": base._verify_preregistered_record(
            authority.get("base_asset_loader"), Path(base.__file__), label="base asset loader"
        ),
        "transport_contract_sha256": base._verify_preregistered_record(
            authority.get("transport_contract"),
            Path(transport_module.__file__),
            label="transport contract",
        ),
        "preregistration_sha256": prereg_sha,
    }
    if prereg.get("forbidden_scene0002_use") is not True:
        raise ValueError("V2.1 scene0002 exclusion differs")
    return prereg, prereg_sha, lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-authority", type=Path, required=True)
    parser.add_argument("--confirmation-authority", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pth").exists():
        raise FileExistsError("refusing to overwrite V2.1 output")
    _prereg, prereg_sha, lineage = _verify_preregistration(args)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    fit_scene, fit_views, fit_train, fit_validation, fit_authority = base._load_scene(
        args.fit_authority, device=device, validation_only=False
    )
    if fit_scene != "scene0001_00":
        raise ValueError("V2.1 fit scene differs")
    head = RegisteredEvidenceToUnaryV2(hidden_dim=32, max_delta_logit=4.0).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline = legacy.evaluate(head=head, examples=fit_validation, views=fit_views, device=device)
    best = baseline
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        ordered = sorted(
            fit_train,
            key=lambda example: legacy._hash_text(
                "anchor_v21_train_order", epoch, example.instance_id, example.mode
            ),
        )
        selected_views = [
            fit_views[example.target_frames[(epoch - 1) % len(example.target_frames)]]
            for example in ordered
        ]
        head.eval()
        with torch.no_grad():
            risk_losses = torch.tensor(
                [
                    float(_example_loss(head, example, view, device=device)[1])
                    for example, view in zip(ordered, selected_views)
                ]
            )
        weights = cvar_prompt_weights(risk_losses)
        head.train()
        weighted_losses = []
        for weight, example, view in zip(weights.tolist(), ordered, selected_views):
            total_loss, _ = _example_loss(head, example, view, device=device)
            loss = float(weight) * total_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_losses.append(float(loss.detach()))
        validation = legacy.evaluate(head=head, examples=fit_validation, views=fit_views, device=device)
        if risk_sensitive_selection_key(validation, epoch) > risk_sensitive_selection_key(best, best_epoch):
            best = validation
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "mean_weighted_train_loss": float(sum(weighted_losses) / len(weighted_losses)),
                "risk_loss_max": float(risk_losses.max()),
                "risk_loss_worst_quartile_mean": float(
                    risk_losses.topk(max(1, math.ceil(CVaR_FRACTION * len(ordered)))).values.mean()
                ),
                "selection_key": list(risk_sensitive_selection_key(validation, epoch)),
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    head.load_state_dict(best_state)
    fit_result = legacy.evaluate(head=head, examples=fit_validation, views=fit_views, device=device)
    fit_gate = base._gate(fit_result)
    del fit_train, fit_validation, fit_views
    if device.type == "cuda":
        torch.cuda.empty_cache()

    confirmation_scene, confirmation_views, _, confirmation_validation, confirmation_authority = base._load_scene(
        args.confirmation_authority, device=device, validation_only=True
    )
    if confirmation_scene != "scene0003_00":
        raise ValueError("V2.1 confirmation scene differs")
    confirmation_result = legacy.evaluate(
        head=head,
        examples=confirmation_validation,
        views=confirmation_views,
        device=device,
    )
    confirmation_gate = base._gate(confirmation_result)
    exact_no_harm = True
    for example in confirmation_validation:
        output = head(example.features)
        immutable = (
            (example.features.labeled_coverage >= 1.0 - head.fully_observed_tolerance)
            | ~example.features.capability_valid
        )
        exact_no_harm = exact_no_harm and torch.equal(
            output.foreground_probability[immutable],
            example.features.analytic_probability[immutable],
        )
    promoted = bool(fit_gate["passed"] and confirmation_gate["passed"] and exact_no_harm)
    result = {
        "schema": "radio_gs.anchor_preserving_prompt_transport.clean_gate_result.v2_1",
        "schema_version": 1,
        "method": method_contract(),
        "checkpoint_method": "RegisteredEvidenceToUnaryV2_risk_sensitive_v2_1",
        "best_epoch": best_epoch,
        "fit_scene": fit_scene,
        "confirmation_scene": confirmation_scene,
        "fit_metrics": base._compact_metrics(fit_result),
        "confirmation_metrics": base._compact_metrics(confirmation_result),
        "fit_gate": fit_gate,
        "confirmation_gate": confirmation_gate,
        "engineering_invariants": {
            "fully_observed_and_inactive_exact_identity": bool(exact_no_harm),
            "fixed_probability_threshold": 0.5,
            "graph": False,
            "connected_selection": False,
        },
        "promotion_gate_passed": promoted,
        "decision": (
            "eligible_for_one_preregistered_target_sentinel"
            if promoted
            else "stop_before_nvos_or_spin_target_metrics"
        ),
        "history": history,
        "authority": {
            "preregistration": {"path": str(args.preregistration.resolve()), "sha256": prereg_sha},
            "fit": fit_authority,
            "confirmation": confirmation_authority,
            "checkpoint_lineage": lineage,
        },
        "source_access": {
            "scene0002_enters_fit_selection_or_confirmation": False,
            "scene0003_labels_enter_fit_or_selection": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
            "per_scene_tuning": False,
        },
    }
    write_frozen_json(args.output, result)
    write_torch_noclobber(
        args.output.with_suffix(".pth"),
        checkpoint_payload(
            state_dict=best_state,
            best_epoch=best_epoch,
            result_path=args.output,
            result_sha256=sha256_file(args.output),
            lineage=lineage,
        ),
    )
    print(json.dumps({"output": str(args.output), "promoted": promoted, "best_epoch": best_epoch}))


if __name__ == "__main__":
    main()
