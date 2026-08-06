#!/usr/bin/env python3
"""Fit and evaluate the frozen two-parameter global prompt calibrator."""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import torch
import torch.nn.functional as F

from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)
from radio_gs.querying.registered_evidence_to_unary import (
    RegisteredEvidenceToUnaryV1,
)
from radio_gs.scripts import train_registered_evidence_to_unary_clean_pilot as v1
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_payload,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _record_path(value: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} authority record differs")
    return Path(str(value["path"])).expanduser().resolve(), str(value["sha256"])


def _validate_v2_authority(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="global prompt calibrator execution authority",
    )
    if (
        authority.get("schema")
        != "radio_gs.global_prompt_logit_calibrator.execution_authority.v2"
        or authority.get("schema_version") != 2
        or authority.get("scene_id") != v1.SCENE_ID
    ):
        raise ValueError("global prompt calibrator execution contract differs")
    inputs = authority.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("global prompt calibrator inputs differ")
    bindings = {
        "preregistration": args.preregistration,
        "source_access_correction": args.source_access_correction,
        "strict_interiorization_addendum": args.strict_interiorization_addendum,
        "v1_reproduction_tolerance_addendum": args.v1_reproduction_tolerance_addendum,
        "v1_result": args.v1_result,
        "v1_checkpoint": args.v1_checkpoint,
        "v1_execution_authority": args.v1_execution_authority,
        "v1_producer": Path(v1.__file__).resolve(),
        "v1_unary_module": Path(v1.__file__).resolve().parents[1]
        / "querying"
        / "registered_evidence_to_unary.py",
        "calibrator_module": Path(__file__).resolve().parents[1]
        / "querying"
        / "global_prompt_logit_calibrator.py",
        "implementation": Path(__file__).resolve(),
    }
    expected_cli = {
        "preregistration": args.expected_preregistration_sha256,
        "source_access_correction": args.expected_source_access_correction_sha256,
        "strict_interiorization_addendum": args.expected_strict_interiorization_addendum_sha256,
        "v1_reproduction_tolerance_addendum": args.expected_v1_reproduction_tolerance_addendum_sha256,
        "v1_result": args.expected_v1_result_sha256,
        "v1_checkpoint": args.expected_v1_checkpoint_sha256,
        "v1_execution_authority": args.expected_v1_execution_authority_sha256,
    }
    for label, supplied in bindings.items():
        recorded_path, recorded_sha = _record_path(inputs.get(label), label=label)
        supplied_path = Path(supplied).expanduser().resolve()
        if recorded_path != supplied_path:
            raise ValueError(f"{label} execution path differs")
        if label in expected_cli and expected_cli[label] != recorded_sha:
            raise ValueError(f"{label} CLI SHA differs from execution authority")
        if sha256_file(supplied_path) != recorded_sha:
            raise ValueError(f"{label} execution SHA differs")
    source = authority.get("source_access", {})
    if (
        source.get("validation_labels_enter_fit") is not False
        or source.get("target_rgb_opened") is not False
        or any(
            source.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "benchmark_queries_opened",
                "benchmark_labels_opened",
                "benchmark_metrics_opened",
            )
        )
    ):
        raise ValueError("global calibrator execution source-access contract differs")
    for output in (args.calibrator_output, args.fit_receipt, args.result_output):
        path = Path(output)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite V2 output: {path}")
    prereg, _, _ = load_json_object(
        args.preregistration,
        expected_sha256=args.expected_preregistration_sha256,
        label="global prompt calibrator preregistration",
    )
    correction, _, _ = load_json_object(
        args.source_access_correction,
        expected_sha256=args.expected_source_access_correction_sha256,
        label="global prompt calibrator source-access correction",
    )
    corrected = correction.get("corrected_contract", {})
    if (
        correction.get("schema")
        != "radio_gs.global_prompt_logit_calibrator.source_access_correction.v2"
        or correction.get("schema_version") != 2
        or correction.get("scene_id") != v1.SCENE_ID
        or corrected.get(
            "validation_label_payload_may_be_deserialized_by_shared_v1_loader_before_fit"
        )
        is not True
        or corrected.get("validation_derived_tensor_prediction_or_metric_enters_optimizer")
        is not False
        or corrected.get(
            "fit_receipt_frozen_before_validation_prediction_or_metric_evaluation"
        )
        is not True
    ):
        raise ValueError("global calibrator source-access correction differs")
    interiorization, _, _ = load_json_object(
        args.strict_interiorization_addendum,
        expected_sha256=args.expected_strict_interiorization_addendum_sha256,
        label="global prompt calibrator strict-interiorization addendum",
    )
    if (
        interiorization.get("schema")
        != "radio_gs.global_prompt_logit_calibrator.strict_interiorization_addendum.v2"
        or interiorization.get("schema_version") != 2
        or interiorization.get("scene_id") != v1.SCENE_ID
        or interiorization.get("corrected_formula", {}).get("parameter_count") != 2
        or interiorization.get("unchanged_contract", {}).get(
            "promotion_gate_changed"
        )
        is not False
    ):
        raise ValueError("global calibrator strict-interiorization addendum differs")
    reproduction_addendum, _, _ = load_json_object(
        args.v1_reproduction_tolerance_addendum,
        expected_sha256=args.expected_v1_reproduction_tolerance_addendum_sha256,
        label="V1 reproduction-tolerance addendum",
    )
    if (
        reproduction_addendum.get("schema")
        != "radio_gs.global_prompt_logit_calibrator.v1_reproduction_tolerance_addendum.v2"
        or reproduction_addendum.get("schema_version") != 2
        or reproduction_addendum.get("scene_id") != v1.SCENE_ID
        or reproduction_addendum.get("corrected_control", {}).get(
            "absolute_tolerance"
        )
        != 1e-6
        or reproduction_addendum.get("explicit_separation", {}).get(
            "promotion_gate_changed"
        )
        is not False
    ):
        raise ValueError("V1 reproduction-tolerance addendum differs")
    v1_result, _, _ = load_json_object(
        args.v1_result,
        expected_sha256=args.expected_v1_result_sha256,
        label="frozen V1 result",
    )
    authority["verified_path"] = str(authority_path)
    authority["verified_sha256"] = authority_sha
    return authority, prereg, v1_result


def _v1_preflight_args(args: argparse.Namespace, execution: dict) -> SimpleNamespace:
    inputs = execution["original_v1_inputs"]
    prereg_path, prereg_sha = _record_path(inputs["v1_preregistration"], label="V1 preregistration")
    return SimpleNamespace(
        execution_authority=args.v1_execution_authority,
        expected_execution_authority_sha256=args.expected_v1_execution_authority_sha256,
        preregistration=prereg_path,
        expected_preregistration_sha256=prereg_sha,
        responsibility_manifest=args.responsibility_manifest,
        expected_responsibility_manifest_sha256=args.expected_responsibility_manifest_sha256,
        label_zip=args.label_zip,
        expected_label_zip_sha256=args.expected_label_zip_sha256,
        capability_bank=args.capability_bank,
        expected_capability_bank_sha256=args.expected_capability_bank_sha256,
        factorized_state=args.factorized_state,
        expected_factorized_state_sha256=args.expected_factorized_state_sha256,
        output=args.result_output,
    )


def _render_full(
    primitive_probability: torch.Tensor,
    view: v1.SparseView,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gids = view.gaussian_ids.to(device)
    pids = view.pixel_ids.to(device)
    weights = view.weights.to(device)
    numerator = torch.zeros_like(view.pixel_mass, device=device)
    numerator.index_add_(0, pids, weights * primitive_probability[gids])
    mass = view.pixel_mass.to(device)
    supported = mass > 0
    prediction = torch.zeros_like(mass)
    prediction[supported] = numerator[supported] / mass[supported]
    return prediction, supported


@torch.no_grad()
def _collect_prompt_predictions(
    *,
    head: RegisteredEvidenceToUnaryV1,
    examples: list[v1.PromptExample],
    views: dict[int, v1.SparseView],
    device: torch.device,
) -> dict[tuple[int, str], dict[str, torch.Tensor | int]]:
    head.eval()
    collected: dict[tuple[int, str], dict[str, torch.Tensor | int]] = {}
    for example in examples:
        scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for frame in example.target_frames:
            view = views[frame]
            unique, inverse = v1._view_rows(view, device)
            subset = v1._subset_features(example.features, unique)
            primitive = head(subset).foreground_probability
            prediction, supported = v1._render_unique(
                primitive, view, unique, inverse, device
            )
            target = view.instance_image.to(device) == example.instance_id
            scores.append(prediction[supported].detach().cpu())
            labels.append(target[supported].detach().cpu())
        collected[(example.instance_id, example.mode)] = {
            "score": torch.cat(scores),
            "label": torch.cat(labels),
            "source_frame": example.source_frame,
            "target_view_count": len(example.target_frames),
        }
    return collected


def _ranking_and_probability_metrics(
    *,
    ranking_score: torch.Tensor,
    probability: torch.Tensor,
    label: torch.Tensor,
) -> dict[str, float]:
    rank = ranking_score.detach().double().cpu().reshape(-1)
    prob = probability.detach().double().cpu().reshape(-1)
    target = label.detach().bool().cpu().reshape(-1)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0 or rank.shape != prob.shape or rank.shape != target.shape:
        raise ValueError("calibration metrics require aligned binary classes")
    order = torch.argsort(rank, descending=True, stable=True)
    ordered = target[order].double()
    tp = ordered.cumsum(0)
    fp = (1.0 - ordered).cumsum(0)
    precision_curve = tp / (tp + fp)
    recall_curve = tp / positives
    average_precision = float((precision_curve * ordered).sum() / positives)
    tpr = torch.cat((torch.zeros(1), recall_curve, torch.ones(1)))
    fpr = torch.cat((torch.zeros(1), fp / negatives, torch.ones(1)))
    auroc = float(torch.trapz(tpr, fpr))
    union = positives + torch.arange(1, rank.numel() + 1, dtype=torch.float64) - tp
    oracle_iou = float((tp / union.clamp_min(1)).max())
    prediction = prob >= 0.5
    fixed_tp = int((prediction & target).sum())
    fixed_fp = int((prediction & ~target).sum())
    fixed_fn = int((~prediction & target).sum())
    return {
        "average_precision": average_precision,
        "auroc": auroc,
        "oracle_iou": oracle_iou,
        "iou_at_0_5": fixed_tp / max(1, fixed_tp + fixed_fp + fixed_fn),
        "precision_at_0_5": fixed_tp / max(1, fixed_tp + fixed_fp),
        "recall_at_0_5": fixed_tp / max(1, fixed_tp + fixed_fn),
        "area_ratio": int(prediction.sum()) / positives,
        "bce": float(
            F.binary_cross_entropy(
                prob.float().clamp(1e-6, 1 - 1e-6), target.float()
            )
        ),
        "pixels": int(rank.numel()),
        "positive_pixels": positives,
    }


@torch.no_grad()
def _strict_ranking_audit(
    raw_score: torch.Tensor, transformed_score: torch.Tensor
) -> dict[str, int | bool]:
    """Verify stable argsort and exact tie partitions after calibration."""

    raw = raw_score.detach().double().reshape(-1)
    transformed = transformed_score.detach().double().reshape(-1)
    if raw.shape != transformed.shape or not bool(torch.isfinite(transformed).all()):
        raise ValueError("strict ranking audit requires aligned finite scores")
    raw_order = torch.argsort(raw, stable=True)
    transformed_order = torch.argsort(transformed, stable=True)
    order_equal = bool(torch.equal(raw_order, transformed_order))
    raw_sorted = raw[raw_order]
    transformed_sorted = transformed[transformed_order]
    if raw.numel() <= 1:
        tie_partition_equal = True
        raw_tie_edges = 0
        transformed_tie_edges = 0
    else:
        raw_ties = raw_sorted[1:] == raw_sorted[:-1]
        transformed_ties = transformed_sorted[1:] == transformed_sorted[:-1]
        tie_partition_equal = bool(torch.equal(raw_ties, transformed_ties))
        raw_tie_edges = int(raw_ties.sum())
        transformed_tie_edges = int(transformed_ties.sum())
    if not order_equal or not tie_partition_equal:
        raise RuntimeError("calibration changed stable order or tie partition")
    return {
        "count": int(raw.numel()),
        "stable_argsort_equal": order_equal,
        "tie_partition_equal": tie_partition_equal,
        "raw_tie_edges": raw_tie_edges,
        "transformed_tie_edges": transformed_tie_edges,
    }


def _macro(records: list[dict], stage: str, mode: str | None = None) -> dict[str, float]:
    selected = [row for row in records if mode is None or row["mode"] == mode]
    return {
        key: float(sum(row[stage][key] for row in selected) / len(selected))
        for key in selected[0][stage]
        if key not in {"pixels", "positive_pixels"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--source-access-correction", type=Path, required=True)
    parser.add_argument("--expected-source-access-correction-sha256", required=True)
    parser.add_argument("--strict-interiorization-addendum", type=Path, required=True)
    parser.add_argument(
        "--expected-strict-interiorization-addendum-sha256", required=True
    )
    parser.add_argument(
        "--v1-reproduction-tolerance-addendum", type=Path, required=True
    )
    parser.add_argument(
        "--expected-v1-reproduction-tolerance-addendum-sha256", required=True
    )
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--expected-v1-result-sha256", required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v1-checkpoint-sha256", required=True)
    parser.add_argument("--v1-execution-authority", type=Path, required=True)
    parser.add_argument("--expected-v1-execution-authority-sha256", required=True)
    parser.add_argument("--responsibility-manifest", type=Path, required=True)
    parser.add_argument("--expected-responsibility-manifest-sha256", required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--expected-label-zip-sha256", required=True)
    parser.add_argument("--capability-bank", type=Path, required=True)
    parser.add_argument("--expected-capability-bank-sha256", required=True)
    parser.add_argument("--factorized-state", type=Path, required=True)
    parser.add_argument("--expected-factorized-state-sha256", required=True)
    parser.add_argument("--calibrator-output", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(v1.SEED)
    device = torch.device(args.device)
    execution, prereg, frozen_v1_result = _validate_v2_authority(args)
    _, label_bytes = v1._validate_execution_authority(
        _v1_preflight_args(args, execution)
    )
    manifest, manifest_sha, _ = load_json_object(
        args.responsibility_manifest,
        expected_sha256=args.expected_responsibility_manifest_sha256,
        label="exact responsibility manifest",
    )
    frame_ids = [int(value) for value in manifest["frame_indices"]]
    records = manifest["views"]
    height = int(manifest["metadata"]["feature_height"])
    width = int(manifest["metadata"]["feature_width"])
    views_root = Path(str(args.responsibility_manifest) + ".views")
    first, _, _ = load_torch_payload(
        views_root / "view_00000.pt",
        expected_sha256=str(records[0]["sha256"]),
        label="first exact responsibility view",
    )
    num_gaussians = int(first["num_gaussians"])
    with zipfile.ZipFile(io.BytesIO(label_bytes), "r") as label_archive:
        views = {
            frame: v1._load_sparse_view(
                view_path=views_root / f"view_{index:05d}.pt",
                frame_id=frame,
                label_archive=label_archive,
                expected_view_sha256=str(records[index]["sha256"]),
                height=height,
                width=width,
                num_gaussians=num_gaussians,
            )
            for index, frame in enumerate(frame_ids)
        }
    occurrences: dict[int, list[int]] = {}
    for frame, view in views.items():
        ids, counts = torch.unique(view.instance_image, return_counts=True)
        for instance_id, count in zip(ids.tolist(), counts.tolist()):
            if instance_id > 0 and count >= v1.MIN_PIXELS:
                occurrences.setdefault(int(instance_id), []).append(frame)
    eligible = {
        instance_id: frames
        for instance_id, frames in occurrences.items()
        if len(frames) >= v1.MIN_VIEWS
    }
    train_ids = sorted(
        instance_id
        for instance_id in eligible
        if v1.instance_split(instance_id) == "train"
    )
    validation_ids = sorted(
        instance_id
        for instance_id in eligible
        if v1.instance_split(instance_id) == "validation"
    )
    if train_ids != frozen_v1_result.get("train_instance_ids"):
        raise ValueError("recomputed train instance IDs differ from frozen V1")
    if validation_ids != frozen_v1_result.get("validation_instance_ids"):
        raise ValueError("recomputed validation instance IDs differ from frozen V1")

    capability, _, _ = load_torch_payload(
        args.capability_bank,
        expected_sha256=args.expected_capability_bank_sha256,
        label="capability bank",
    )
    state, _, _ = load_torch_payload(
        args.factorized_state,
        expected_sha256=args.expected_factorized_state_sha256,
        label="factorized state",
    )
    cross_asset = v1._validate_cross_asset_authority(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        capability=capability,
        state=state,
    )
    examples = v1.build_examples(
        views=views,
        eligible=eligible,
        capability_bank=capability,
        factorized_state=state,
        device=device,
        height=height,
        width=width,
    )
    del capability, state
    train_examples = [example for example in examples if example.split == "train"]
    validation_examples = [example for example in examples if example.split == "validation"]

    checkpoint, _, _ = load_torch_payload(
        args.v1_checkpoint,
        expected_sha256=args.expected_v1_checkpoint_sha256,
        label="frozen V1 checkpoint",
    )
    if (
        checkpoint.get("result_sha256") != args.expected_v1_result_sha256
        or checkpoint.get("best_epoch") != frozen_v1_result.get("best_epoch")
        or checkpoint.get("train_instance_ids") != frozen_v1_result.get("train_instance_ids")
        or checkpoint.get("validation_instance_ids")
        != frozen_v1_result.get("validation_instance_ids")
    ):
        raise ValueError("V1 checkpoint/result authority differs")
    head = RegisteredEvidenceToUnaryV1(hidden_dim=32, max_delta_logit=4.0).to(device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()

    # Validation-instance equality is not evaluated until after the calibrator
    # parameters and fit receipt are durably frozen below.
    train = _collect_prompt_predictions(
        head=head, examples=train_examples, views=views, device=device
    )
    train_scores = torch.cat([value["score"] for value in train.values()]).to(
        device=device, dtype=torch.float64
    )
    train_labels = torch.cat([value["label"] for value in train.values()]).to(
        device=device, dtype=torch.float64
    )
    calibrator = GlobalPromptLogitCalibratorV2().to(
        device=device, dtype=torch.float64
    )
    train_domain = calibrator.strict_domain_audit(train_scores)
    optimizer = torch.optim.LBFGS(
        calibrator.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=20,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            calibrator.calibrated_logit(train_scores), train_labels
        )
        loss.backward()
        return loss

    initial_train_bce = float(closure().detach())
    optimizer.step(closure)
    with torch.no_grad():
        final_train_bce = float(
            F.binary_cross_entropy_with_logits(
                calibrator.calibrated_logit(train_scores), train_labels
            )
        )
        temperature = float(calibrator.temperature)
        bias = float(calibrator.bias)
    if not (0 < temperature < float("inf")):
        raise RuntimeError("global calibration temperature is invalid")
    train_ranking_audit = _strict_ranking_audit(
        train_scores, calibrator.calibrated_logit(train_scores)
    )

    calibrator_payload = {
        "schema": "radio_gs.global_prompt_logit_calibrator.checkpoint.v2",
        "schema_version": 2,
        "state_dict": {key: value.detach().cpu() for key, value in calibrator.state_dict().items()},
        "temperature": temperature,
        "bias": bias,
        "probability_eps": calibrator.probability_eps,
        "fit_pixel_count": int(train_scores.numel()),
        "fit_positive_pixel_count": int(train_labels.sum()),
        "initial_train_bce": initial_train_bce,
        "final_train_bce": final_train_bce,
        "train_strict_domain_audit": train_domain,
        "train_strict_ranking_audit": train_ranking_audit,
        "execution_authority_sha256": execution["verified_sha256"],
        "v1_result_sha256": args.expected_v1_result_sha256,
        "v1_checkpoint_sha256": args.expected_v1_checkpoint_sha256,
        "validation_label_payload_may_be_deserialized_by_shared_v1_loader": True,
        "validation_derived_tensor_prediction_or_metric_enter_fit": False,
    }
    write_torch_noclobber(args.calibrator_output, calibrator_payload)
    fit_receipt = {
        key: value for key, value in calibrator_payload.items() if key != "state_dict"
    }
    fit_receipt.update(
        {
            "calibrator_checkpoint": {
                "path": str(args.calibrator_output.resolve()),
                "sha256": sha256_file(args.calibrator_output),
            },
            "fit_instance_ids": frozen_v1_result["train_instance_ids"],
            "validation_instance_ids_not_evaluated": frozen_v1_result[
                "validation_instance_ids"
            ],
            "validation_prediction_or_metric_evaluated_before_fit_receipt": False,
        }
    )
    write_frozen_json(args.fit_receipt, fit_receipt)
    fit_receipt_sha = sha256_file(args.fit_receipt)

    validation = _collect_prompt_predictions(
        head=head, examples=validation_examples, views=views, device=device
    )
    validation_scores = torch.cat(
        [value["score"] for value in validation.values()]
    ).to(device)
    validation_domain = calibrator.strict_domain_audit(validation_scores)
    validation_ranking_audit = _strict_ranking_audit(
        validation_scores.to(device=device, dtype=torch.float64),
        calibrator.calibrated_logit(validation_scores),
    )
    result_records: list[dict] = []
    reproduction_tolerance = 1e-6
    reproduction_keys = (
        "average_precision",
        "auroc",
        "iou_at_0_5",
        "oracle_iou",
        "precision_at_0_5",
        "recall_at_0_5",
        "area_ratio",
        "bce",
    )
    v1_reproduction_max_absolute_error = 0.0
    with torch.no_grad():
        for (instance_id, mode), values in sorted(validation.items()):
            raw = values["score"].to(device)
            label = values["label"].to(device)
            calibrated_logit = calibrator.calibrated_logit(raw)
            calibrated_probability = torch.sigmoid(calibrated_logit)
            frozen_record = next(
                row
                for row in frozen_v1_result["metrics"]["records"]
                if row["instance_id"] == instance_id and row["mode"] == mode
            )
            analytic_record = frozen_record["analytic"]
            v1_metric = _ranking_and_probability_metrics(
                ranking_score=raw,
                probability=raw,
                label=label,
            )
            record_reproduction_error = max(
                abs(v1_metric[key] - frozen_record["candidate"][key])
                for key in reproduction_keys
            )
            v1_reproduction_max_absolute_error = max(
                v1_reproduction_max_absolute_error, record_reproduction_error
            )
            if record_reproduction_error > reproduction_tolerance:
                raise RuntimeError(
                    "frozen V1 prompt metric reproduction exceeds tolerance: "
                    f"instance={instance_id}, mode={mode}, "
                    f"error={record_reproduction_error}"
                )
            candidate = _ranking_and_probability_metrics(
                ranking_score=calibrated_logit,
                probability=calibrated_probability,
                label=label,
            )
            result_records.append(
                {
                    "instance_id": instance_id,
                    "mode": mode,
                    "source_frame": int(values["source_frame"]),
                    "target_view_count": int(values["target_view_count"]),
                    "analytic": analytic_record,
                    "v1": v1_metric,
                    "candidate": candidate,
                    "delta_candidate_minus_analytic": {
                        key: candidate[key] - analytic_record[key]
                        for key in candidate
                        if key in analytic_record
                    },
                }
            )
    macro = {
        mode: {
            stage: _macro(
                result_records,
                stage,
                None if mode == "all" else mode,
            )
            for stage in ("analytic", "v1", "candidate")
        }
        for mode in ("all", "full_mask", "scribble")
    }
    tolerance = 1e-10
    ranking_keys = ("average_precision", "auroc", "oracle_iou")
    ranking_preserved = all(
        abs(row["candidate"][key] - row["v1"][key]) <= tolerance
        for row in result_records
        for key in ranking_keys
    )
    all_candidate = macro["all"]["candidate"]
    all_analytic = macro["all"]["analytic"]
    worst_ap_delta = min(
        row["delta_candidate_minus_analytic"]["average_precision"]
        for row in result_records
    )
    promoted = (
        ranking_preserved
        and 0.8 <= all_candidate["area_ratio"] <= 1.25
        and all_candidate["iou_at_0_5"] >= all_analytic["iou_at_0_5"]
        and all_candidate["precision_at_0_5"] >= all_analytic["precision_at_0_5"]
        and worst_ap_delta >= -0.05
    )
    result = {
        "schema": "radio_gs.global_prompt_logit_calibrator.clean_scannet_result.v2",
        "schema_version": 2,
        "scene_id": v1.SCENE_ID,
        "method": "GlobalPromptLogitCalibratorV2",
        "graph": "off",
        "connected_selection": "off",
        "temperature": temperature,
        "bias": bias,
        "fit": {
            "pixel_count": int(train_scores.numel()),
            "positive_pixel_count": int(train_labels.sum()),
            "initial_bce": initial_train_bce,
            "final_bce": final_train_bce,
            "strict_domain_audit": train_domain,
            "validation_labels_opened": False,
        },
        "validation_strict_domain_audit": validation_domain,
        "validation_strict_ranking_audit": validation_ranking_audit,
        "v1_metric_reproduction": {
            "keys": list(reproduction_keys),
            "absolute_tolerance": reproduction_tolerance,
            "max_absolute_error": v1_reproduction_max_absolute_error,
            "passed": True,
        },
        "ranking_preservation_tolerance": tolerance,
        "ranking_preserved": ranking_preserved,
        "metrics": {"records": result_records, "macro": macro},
        "worst_prompt_ap_delta_vs_analytic": worst_ap_delta,
        "promotion_gate_passed": promoted,
        "decision": "eligible_for_cross_scene_confirmation" if promoted else "stop_before_benchmarks",
        "authority": {
            "execution_authority": {
                "path": execution["verified_path"],
                "sha256": execution["verified_sha256"],
            },
            "preregistration_sha256": args.expected_preregistration_sha256,
            "source_access_correction": {
                "path": str(args.source_access_correction.resolve()),
                "sha256": args.expected_source_access_correction_sha256,
            },
            "strict_interiorization_addendum": {
                "path": str(args.strict_interiorization_addendum.resolve()),
                "sha256": args.expected_strict_interiorization_addendum_sha256,
            },
            "v1_reproduction_tolerance_addendum": {
                "path": str(args.v1_reproduction_tolerance_addendum.resolve()),
                "sha256": args.expected_v1_reproduction_tolerance_addendum_sha256,
            },
            "v1_result_sha256": args.expected_v1_result_sha256,
            "v1_checkpoint_sha256": args.expected_v1_checkpoint_sha256,
            "calibrator_checkpoint": {
                "path": str(args.calibrator_output.resolve()),
                "sha256": sha256_file(args.calibrator_output),
            },
            "fit_receipt": {
                "path": str(args.fit_receipt.resolve()),
                "sha256": fit_receipt_sha,
            },
            "cross_asset_authority": cross_asset,
        },
        "source_access": {
            "clean_source_training_instance_labels_opened": True,
            "validation_label_payload_may_be_deserialized_by_shared_v1_loader": True,
            "validation_instance_membership_used_to_construct_fit_tensor": False,
            "validation_prediction_and_metric_evaluation_started_after_fit_receipt_frozen": True,
            "validation_labels_enter_fit": False,
            "validation_labels_enter_early_stop": False,
            "target_rgb_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
            "per_scene_tuning": False,
        },
    }
    write_frozen_json(args.result_output, result)
    print(
        json.dumps(
            {
                "result": str(args.result_output),
                "temperature": temperature,
                "bias": bias,
                "promoted": promoted,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
