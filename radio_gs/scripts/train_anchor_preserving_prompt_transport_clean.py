#!/usr/bin/env python3
"""Fit and independently confirm the shared prompt-transport unary.

The fit scene and confirmation scene are nonbenchmark ScanNet scenes.  Only
the fit scene may select an epoch; the confirmation scene is evaluated once
with the selected checkpoint.  This program never imports NVOS/SPIn assets or
selects a target threshold.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import radio_gs.querying.anchor_preserving_transport as transport_module
from radio_gs.querying.anchor_preserving_transport import method_contract
from radio_gs.querying.registered_evidence_to_unary import (
    RegisteredEvidenceToUnaryV2,
)
from radio_gs.scripts import evaluate_frozen_prompt_unary_cross_scene_confirmation as cross
from radio_gs.scripts import train_registered_evidence_to_unary_clean_pilot as legacy
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_payload,
    sha256_file,
    stable_descriptor_load,
    write_frozen_json,
    write_torch_noclobber,
)


SEED = 260809
METRIC_KEYS = (
    "average_precision",
    "iou_at_0_5",
    "precision_at_0_5",
    "recall_at_0_5",
    "area_ratio",
    "bce",
)


def _authority_inputs(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    authority, _, _ = load_json_object(path, label="clean prompt asset authority")
    scene = str(authority.get("scene_id"))
    records = authority.get("inputs")
    required = ("responsibility_manifest", "label_zip", "capability_bank", "factorized_state")
    if not scene.startswith("scene") or not isinstance(records, dict):
        raise ValueError("clean prompt asset authority differs")
    for key in required:
        record = records.get(key)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"clean prompt {key} record differs")
        if sha256_file(Path(record["path"])) != str(record["sha256"]):
            raise ValueError(f"clean prompt {key} SHA-256 differs")
    return scene, records


def _verify_preregistered_record(
    record: object,
    actual_path: Path,
    *,
    label: str,
) -> str:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"preregistered {label} record differs")
    resolved = actual_path.expanduser().resolve()
    if Path(str(record["path"])).expanduser().resolve() != resolved:
        raise ValueError(f"preregistered {label} path differs")
    digest = str(record["sha256"])
    if sha256_file(resolved) != digest:
        raise ValueError(f"preregistered {label} SHA-256 differs")
    return digest


def _load_scene(
    authority_path: Path,
    *,
    device: torch.device,
    validation_only: bool = False,
) -> tuple[str, dict[int, legacy.SparseView], list[legacy.PromptExample], list[legacy.PromptExample], dict]:
    scene, records = _authority_inputs(authority_path)
    manifest_path = Path(records["responsibility_manifest"]["path"])
    manifest, manifest_sha, _ = load_json_object(
        manifest_path,
        expected_sha256=str(records["responsibility_manifest"]["sha256"]),
        label=f"{scene} exact responsibility manifest",
    )
    label_path = Path(records["label_zip"]["path"])
    label_bytes, _, _ = stable_descriptor_load(
        label_path,
        lambda handle: handle.read(),
        expected_sha256=str(records["label_zip"]["sha256"]),
        label=f"{scene} official instance zip",
    )
    views, height, width, _ = cross._load_views(
        manifest=manifest,
        manifest_path=manifest_path,
        label_bytes=label_bytes,
    )
    legacy.SCENE_ID = scene
    eligible = cross._eligible_instances(views)
    train_ids = sorted(i for i in eligible if legacy.instance_split(i) == "train")
    validation_ids = sorted(i for i in eligible if legacy.instance_split(i) == "validation")
    if len(validation_ids) < 3 or set(train_ids) & set(validation_ids):
        raise RuntimeError(f"{scene} clean instance split is invalid")
    capability, _, _ = load_torch_payload(
        Path(records["capability_bank"]["path"]),
        expected_sha256=str(records["capability_bank"]["sha256"]),
        map_location="cpu",
        label=f"{scene} capability bank",
    )
    state, _, _ = load_torch_payload(
        Path(records["factorized_state"]["path"]),
        expected_sha256=str(records["factorized_state"]["sha256"]),
        map_location="cpu",
        label=f"{scene} factorized state",
    )
    cross_asset = legacy._validate_cross_asset_authority(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        capability=capability,
        state=state,
    )
    example_authority = (
        {instance_id: eligible[instance_id] for instance_id in validation_ids}
        if validation_only
        else eligible
    )
    examples = legacy.build_examples(
        views=views,
        eligible=example_authority,
        capability_bank=capability,
        factorized_state=state,
        device=device,
        height=height,
        width=width,
    )
    del capability, state
    return (
        scene,
        views,
        [value for value in examples if value.split == "train"],
        [value for value in examples if value.split == "validation"],
        {
            "authority_path": str(authority_path.resolve()),
            "authority_sha256": sha256_file(authority_path),
            "eligible_instances": len(eligible),
            "train_instance_ids": train_ids,
            "validation_instance_ids": validation_ids,
            "cross_asset_authority": cross_asset,
        },
    )


def _selection_key(result: dict, epoch: int) -> tuple[float, ...]:
    macro = result["macro"]
    ap_deltas = [
        macro[mode]["candidate"]["average_precision"]
        - macro[mode]["analytic"]["average_precision"]
        for mode in ("full_mask", "scribble")
    ]
    iou_deltas = [
        macro[mode]["candidate"]["iou_at_0_5"]
        - macro[mode]["analytic"]["iou_at_0_5"]
        for mode in ("full_mask", "scribble")
    ]
    return (
        min(ap_deltas),
        min(iou_deltas),
        macro["all"]["candidate"]["average_precision"],
        -macro["all"]["candidate"]["bce"],
        -int(epoch),
    )


def _compact_metrics(result: dict) -> dict:
    return {
        partition: {
            stage: {
                key: float(result["macro"][partition][stage][key])
                for key in METRIC_KEYS
            }
            for stage in ("analytic", "candidate")
        }
        for partition in ("all", "full_mask", "scribble")
    }


def _gate(result: dict) -> dict[str, object]:
    macro = result["macro"]
    per_mode = {}
    for mode in ("full_mask", "scribble"):
        candidate = macro[mode]["candidate"]
        analytic = macro[mode]["analytic"]
        per_mode[mode] = {
            "ap_delta": candidate["average_precision"] - analytic["average_precision"],
            "iou05_delta": candidate["iou_at_0_5"] - analytic["iou_at_0_5"],
            "precision_delta": candidate["precision_at_0_5"] - analytic["precision_at_0_5"],
            "area_log_error_before": abs(float(np.log(max(analytic["area_ratio"], 1e-12)))),
            "area_log_error_after": abs(float(np.log(max(candidate["area_ratio"], 1e-12)))),
        }
    worst_ap = min(
        row["delta"]["average_precision"] for row in result["records"]
    )
    passed = (
        all(value["ap_delta"] >= 0 for value in per_mode.values())
        and all(value["iou05_delta"] >= 0 for value in per_mode.values())
        and all(value["precision_delta"] >= -0.01 for value in per_mode.values())
        and all(
            value["area_log_error_after"] < value["area_log_error_before"]
            for value in per_mode.values()
        )
        and worst_ap >= -0.005
    )
    return {
        "passed": bool(passed),
        "per_mode": per_mode,
        "worst_prompt_ap_delta": float(worst_ap),
    }


def _checkpoint_payload(
    *,
    state_dict: dict,
    best_epoch: int,
    head: RegisteredEvidenceToUnaryV2,
    result_path: Path,
    result_sha256: str,
    transport_sha256: str,
    trainer_sha256: str,
    preregistration_sha256: str,
    fit_authority_sha256: str,
    confirmation_authority_sha256: str,
) -> dict:
    """Build the versioned lineage that every future target adapter verifies."""

    return {
        "schema": "radio_gs.anchor_preserving_prompt_transport.checkpoint.v1",
        "state_dict": state_dict,
        "best_epoch": int(best_epoch),
        "architecture": {
            "hidden_dim": int(head.hidden_dim),
            "max_delta_logit": float(head.max_delta_logit),
            "fully_observed_tolerance": float(head.fully_observed_tolerance),
        },
        "lineage": {
            "transport_contract_sha256": transport_sha256,
            "trainer_sha256": trainer_sha256,
            "preregistration_sha256": preregistration_sha256,
            "fit_authority_sha256": fit_authority_sha256,
            "confirmation_authority_sha256": confirmation_authority_sha256,
        },
        "result_path": str(result_path.resolve()),
        "result_sha256": result_sha256,
    }


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
        raise FileExistsError("refusing to overwrite clean prompt-transport output")
    prereg, prereg_sha, _ = load_json_object(
        args.preregistration,
        expected_sha256=args.expected_preregistration_sha256,
        label="anchor-preserving prompt-transport preregistration",
    )
    if (
        prereg.get("schema") != "radio_gs.anchor_preserving_prompt_transport.clean_gate_preregistration.v1"
        or prereg.get("fit_scene") != "scene0001_00"
        or prereg.get("confirmation_scene") != "scene0002_00"
        or int(args.epochs) != int(prereg.get("epochs", -1))
    ):
        raise ValueError("anchor-preserving preregistration differs")
    prereg_authority = prereg.get("authority")
    if not isinstance(prereg_authority, dict):
        raise ValueError("anchor-preserving preregistered authority differs")
    fit_authority_sha = _verify_preregistered_record(
        prereg_authority.get("fit_asset_authority"),
        args.fit_authority,
        label="fit asset authority",
    )
    confirmation_authority_sha = _verify_preregistered_record(
        prereg_authority.get("confirmation_asset_authority"),
        args.confirmation_authority,
        label="confirmation asset authority",
    )
    trainer_sha = _verify_preregistered_record(
        prereg_authority.get("implementation"),
        Path(__file__),
        label="trainer implementation",
    )
    transport_sha = _verify_preregistered_record(
        prereg_authority.get("transport_contract"),
        Path(transport_module.__file__),
        label="transport contract",
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    fit_scene, fit_views, fit_train, fit_validation, fit_authority = _load_scene(
        args.fit_authority, device=device, validation_only=False
    )
    if fit_scene != "scene0001_00":
        raise ValueError("fit authority must name scene0001_00")
    head = RegisteredEvidenceToUnaryV2(hidden_dim=32, max_delta_logit=4.0).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline = legacy.evaluate(
        head=head, examples=fit_validation, views=fit_views, device=device
    )
    best = baseline
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in head.state_dict().items()
    }
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        losses = []
        ordered = sorted(
            fit_train,
            key=lambda example: legacy._hash_text(
                "anchor_train_order", epoch, example.instance_id, example.mode
            ),
        )
        for example in ordered:
            target_frame = example.target_frames[(epoch - 1) % len(example.target_frames)]
            view = fit_views[target_frame]
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
            loss = data_loss + 0.05 * residual_loss + 0.02 * confidence_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = legacy.evaluate(
            head=head, examples=fit_validation, views=fit_views, device=device
        )
        if _selection_key(validation, epoch) > _selection_key(best, best_epoch):
            best = validation
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": float(sum(losses) / len(losses)),
                "selection_key": list(_selection_key(validation, epoch)),
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    head.load_state_dict(best_state)
    fit_result = legacy.evaluate(
        head=head, examples=fit_validation, views=fit_views, device=device
    )
    fit_gate = _gate(fit_result)
    del fit_train, fit_validation, fit_views
    if device.type == "cuda":
        torch.cuda.empty_cache()

    confirmation_scene, confirmation_views, _, confirmation_validation, confirmation_authority = _load_scene(
        args.confirmation_authority, device=device, validation_only=True
    )
    if confirmation_scene != "scene0002_00":
        raise ValueError("confirmation authority must name scene0002_00")
    confirmation_result = legacy.evaluate(
        head=head,
        examples=confirmation_validation,
        views=confirmation_views,
        device=device,
    )
    confirmation_gate = _gate(confirmation_result)
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
        "schema": "radio_gs.anchor_preserving_prompt_transport.clean_gate_result.v1",
        "schema_version": 1,
        "method": method_contract(),
        "checkpoint_method": "RegisteredEvidenceToUnaryV2_native",
        "best_epoch": best_epoch,
        "fit_scene": fit_scene,
        "confirmation_scene": confirmation_scene,
        "fit_metrics": _compact_metrics(fit_result),
        "confirmation_metrics": _compact_metrics(confirmation_result),
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
        },
        "source_access": {
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
        _checkpoint_payload(
            state_dict=best_state,
            best_epoch=best_epoch,
            head=head,
            result_path=args.output,
            result_sha256=sha256_file(args.output),
            transport_sha256=transport_sha,
            trainer_sha256=trainer_sha,
            preregistration_sha256=prereg_sha,
            fit_authority_sha256=fit_authority_sha,
            confirmation_authority_sha256=confirmation_authority_sha,
        ),
    )
    print(json.dumps({"output": str(args.output), "promoted": promoted, "best_epoch": best_epoch}))


if __name__ == "__main__":
    main()
