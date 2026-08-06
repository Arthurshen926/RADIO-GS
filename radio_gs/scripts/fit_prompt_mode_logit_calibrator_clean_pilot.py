#!/usr/bin/env python3
"""Fit and evaluate the four-parameter prompt-mode calibrator."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import torch
import torch.nn.functional as F

from radio_gs.querying.prompt_mode_logit_calibrator import (
    PROMPT_MODES,
    PromptModeLogitCalibratorV3,
)
from radio_gs.querying.registered_evidence_to_unary import (
    RegisteredEvidenceToUnaryV1,
)
from radio_gs.scripts import fit_global_prompt_logit_calibrator_clean_pilot as v2
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


def _validate_authority(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="prompt-mode calibrator execution authority",
    )
    if (
        authority.get("schema")
        != "radio_gs.prompt_mode_logit_calibrator.execution_authority.v3"
        or authority.get("schema_version") != 3
        or authority.get("scene_id") != v1.SCENE_ID
    ):
        raise ValueError("prompt-mode calibrator execution contract differs")
    inputs = authority.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("prompt-mode calibrator inputs differ")
    bindings = {
        "preregistration": args.preregistration,
        "v2_diagnostic_result": args.v2_result,
        "v1_result": args.v1_result,
        "v1_checkpoint": args.v1_checkpoint,
        "v1_execution_authority": args.v1_execution_authority,
        "v1_producer": Path(v1.__file__).resolve(),
        "v1_unary_module": Path(v1.__file__).resolve().parents[1]
        / "querying"
        / "registered_evidence_to_unary.py",
        "v2_numeric_module": Path(v2.__file__).resolve().parents[1]
        / "querying"
        / "global_prompt_logit_calibrator.py",
        "v2_helper_producer": Path(v2.__file__).resolve(),
        "v3_calibrator_module": Path(__file__).resolve().parents[1]
        / "querying"
        / "prompt_mode_logit_calibrator.py",
        "implementation": Path(__file__).resolve(),
    }
    expected_cli = {
        "preregistration": args.expected_preregistration_sha256,
        "v2_diagnostic_result": args.expected_v2_result_sha256,
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
        source.get("validation_label_payload_may_be_deserialized_before_fit")
        is not True
        or source.get("validation_derived_tensor_prediction_or_metric_enters_optimizer")
        is not False
        or source.get("fit_receipt_frozen_before_validation_prediction_or_metric_evaluation")
        is not True
        or any(
            source.get(key) is not False
            for key in (
                "target_rgb_opened",
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "benchmark_queries_opened",
                "benchmark_labels_opened",
                "benchmark_metrics_opened",
            )
        )
    ):
        raise ValueError("prompt-mode calibrator source-access contract differs")
    for output in (args.calibrator_output, args.fit_receipt, args.result_output):
        path = Path(output)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite V3 output: {path}")
    prereg, _, _ = load_json_object(
        args.preregistration,
        expected_sha256=args.expected_preregistration_sha256,
        label="prompt-mode calibrator preregistration",
    )
    if (
        prereg.get("schema")
        != "radio_gs.prompt_mode_logit_calibrator.clean_scannet_preregistration.v3"
        or prereg.get("schema_version") != 3
        or prereg.get("scene_id") != v1.SCENE_ID
        or prereg.get("method", {}).get("parameter_count") != 4
    ):
        raise ValueError("prompt-mode calibrator preregistration differs")
    frozen_v1_result, _, _ = load_json_object(
        args.v1_result,
        expected_sha256=args.expected_v1_result_sha256,
        label="frozen V1 result",
    )
    authority["verified_path"] = str(authority_path)
    authority["verified_sha256"] = authority_sha
    return authority, prereg, frozen_v1_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--expected-v2-result-sha256", required=True)
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
    execution, _, frozen_v1_result = _validate_authority(args)
    _, label_bytes = v1._validate_execution_authority(
        v2._v1_preflight_args(args, execution)
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
    validation_examples = [
        example for example in examples if example.split == "validation"
    ]

    checkpoint, _, _ = load_torch_payload(
        args.v1_checkpoint,
        expected_sha256=args.expected_v1_checkpoint_sha256,
        label="frozen V1 checkpoint",
    )
    if (
        checkpoint.get("result_sha256") != args.expected_v1_result_sha256
        or checkpoint.get("best_epoch") != frozen_v1_result.get("best_epoch")
        or checkpoint.get("train_instance_ids") != train_ids
        or checkpoint.get("validation_instance_ids") != validation_ids
    ):
        raise ValueError("V1 checkpoint/result authority differs")
    head = RegisteredEvidenceToUnaryV1(hidden_dim=32, max_delta_logit=4.0).to(
        device
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()

    train = v2._collect_prompt_predictions(
        head=head, examples=train_examples, views=views, device=device
    )
    calibrator = PromptModeLogitCalibratorV3().to(
        device=device, dtype=torch.float64
    )
    fit_by_mode: dict[str, dict] = {}
    for mode in PROMPT_MODES:
        scores = torch.cat(
            [value["score"] for key, value in train.items() if key[1] == mode]
        ).to(device=device, dtype=torch.float64)
        labels = torch.cat(
            [value["label"] for key, value in train.items() if key[1] == mode]
        ).to(device=device, dtype=torch.float64)
        branch = calibrator.calibrators[mode]
        domain_audit = branch.strict_domain_audit(scores)
        optimizer = torch.optim.LBFGS(
            branch.parameters(),
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
                branch.calibrated_logit(scores), labels
            )
            loss.backward()
            return loss

        initial_bce = float(closure().detach())
        optimizer.step(closure)
        with torch.no_grad():
            final_logit = branch.calibrated_logit(scores)
            final_bce = float(F.binary_cross_entropy_with_logits(final_logit, labels))
            temperature = float(branch.temperature)
            bias = float(branch.bias)
        if not (0 < temperature < float("inf")):
            raise RuntimeError(f"{mode} calibration temperature is invalid")
        fit_by_mode[mode] = {
            "pixel_count": int(scores.numel()),
            "positive_pixel_count": int(labels.sum()),
            "initial_bce": initial_bce,
            "final_bce": final_bce,
            "temperature": temperature,
            "bias": bias,
            "strict_domain_audit": domain_audit,
            "strict_ranking_audit": v2._strict_ranking_audit(scores, final_logit),
        }

    calibrator_payload = {
        "schema": "radio_gs.prompt_mode_logit_calibrator.checkpoint.v3",
        "schema_version": 3,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in calibrator.state_dict().items()
        },
        "fit_by_mode": fit_by_mode,
        "fit_instance_ids": train_ids,
        "validation_instance_ids_not_evaluated": validation_ids,
        "execution_authority_sha256": execution["verified_sha256"],
        "v1_result_sha256": args.expected_v1_result_sha256,
        "v1_checkpoint_sha256": args.expected_v1_checkpoint_sha256,
        "validation_label_payload_may_be_deserialized_by_shared_v1_loader": True,
        "validation_derived_tensor_prediction_or_metric_enters_fit": False,
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
            "validation_prediction_or_metric_evaluated_before_fit_receipt": False,
        }
    )
    write_frozen_json(args.fit_receipt, fit_receipt)
    fit_receipt_sha = sha256_file(args.fit_receipt)

    validation = v2._collect_prompt_predictions(
        head=head, examples=validation_examples, views=views, device=device
    )
    validation_audits: dict[str, dict] = {}
    for mode in PROMPT_MODES:
        raw = torch.cat(
            [value["score"] for key, value in validation.items() if key[1] == mode]
        ).to(device=device, dtype=torch.float64)
        transformed = calibrator.calibrated_logit(raw, mode=mode)
        validation_audits[mode] = {
            "strict_domain_audit": calibrator.strict_domain_audit(raw, mode=mode),
            "strict_ranking_audit": v2._strict_ranking_audit(raw, transformed),
        }

    reproduction_tolerance = 1e-6
    ranking_tolerance = 1e-10
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
    reproduction_max_error = 0.0
    result_records: list[dict] = []
    with torch.no_grad():
        for (instance_id, mode), values in sorted(validation.items()):
            raw = values["score"].to(device)
            label = values["label"].to(device)
            calibrated_logit = calibrator.calibrated_logit(raw, mode=mode)
            probability = torch.sigmoid(calibrated_logit)
            frozen_record = next(
                row
                for row in frozen_v1_result["metrics"]["records"]
                if row["instance_id"] == instance_id and row["mode"] == mode
            )
            v1_metric = v2._ranking_and_probability_metrics(
                ranking_score=raw, probability=raw, label=label
            )
            error = max(
                abs(v1_metric[key] - frozen_record["candidate"][key])
                for key in reproduction_keys
            )
            reproduction_max_error = max(reproduction_max_error, error)
            if error > reproduction_tolerance:
                raise RuntimeError(
                    "frozen V1 prompt metric reproduction exceeds tolerance: "
                    f"instance={instance_id}, mode={mode}, error={error}"
                )
            candidate = v2._ranking_and_probability_metrics(
                ranking_score=calibrated_logit,
                probability=probability,
                label=label,
            )
            analytic = frozen_record["analytic"]
            result_records.append(
                {
                    "instance_id": instance_id,
                    "mode": mode,
                    "source_frame": int(values["source_frame"]),
                    "target_view_count": int(values["target_view_count"]),
                    "analytic": analytic,
                    "v1": v1_metric,
                    "candidate": candidate,
                    "delta_candidate_minus_analytic": {
                        key: candidate[key] - analytic[key]
                        for key in candidate
                        if key in analytic
                    },
                }
            )
    macro = {
        mode: {
            stage: v2._macro(
                result_records, stage, None if mode == "all" else mode
            )
            for stage in ("analytic", "v1", "candidate")
        }
        for mode in ("all", *PROMPT_MODES)
    }
    ranking_keys = ("average_precision", "auroc", "oracle_iou")
    ranking_max_error = {
        mode: max(
            abs(row["candidate"][key] - row["v1"][key])
            for row in result_records
            if row["mode"] == mode
            for key in ranking_keys
        )
        for mode in PROMPT_MODES
    }
    ranking_preserved = all(
        value <= ranking_tolerance for value in ranking_max_error.values()
    )
    mode_gates: dict[str, dict[str, bool]] = {}
    for mode in PROMPT_MODES:
        candidate = macro[mode]["candidate"]
        baseline = macro[mode]["v1"]
        analytic = macro[mode]["analytic"]
        mode_gates[mode] = {
            "ranking_preserved": ranking_max_error[mode] <= ranking_tolerance,
            "iou_at_0_5_not_below_v1": candidate["iou_at_0_5"]
            >= baseline["iou_at_0_5"],
            "area_ratio_in_interval": 0.8 <= candidate["area_ratio"] <= 1.25,
            "precision_at_0_5_not_below_analytic": candidate["precision_at_0_5"]
            >= analytic["precision_at_0_5"],
        }
    worst_ap_delta = min(
        row["delta_candidate_minus_analytic"]["average_precision"]
        for row in result_records
    )
    promoted = (
        ranking_preserved
        and all(all(gate.values()) for gate in mode_gates.values())
        and worst_ap_delta >= -0.05
    )
    result = {
        "schema": "radio_gs.prompt_mode_logit_calibrator.clean_scannet_result.v3",
        "schema_version": 3,
        "scene_id": v1.SCENE_ID,
        "method": "PromptModeLogitCalibratorV3",
        "parameter_count": 4,
        "graph": "off",
        "connected_selection": "off",
        "fit_by_mode": fit_by_mode,
        "validation_audits": validation_audits,
        "v1_metric_reproduction": {
            "absolute_tolerance": reproduction_tolerance,
            "max_absolute_error": reproduction_max_error,
            "passed": True,
        },
        "ranking_preservation": {
            "absolute_tolerance": ranking_tolerance,
            "max_absolute_error_by_mode": ranking_max_error,
            "passed": ranking_preserved,
        },
        "metrics": {"records": result_records, "macro": macro},
        "mode_gates": mode_gates,
        "worst_prompt_ap_delta_vs_analytic": worst_ap_delta,
        "promotion_gate_passed": promoted,
        "decision": "eligible_for_cross_scene_confirmation"
        if promoted
        else "stop_before_benchmarks",
        "authority": {
            "execution_authority": {
                "path": execution["verified_path"],
                "sha256": execution["verified_sha256"],
            },
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "sha256": args.expected_preregistration_sha256,
            },
            "v2_diagnostic_result_sha256": args.expected_v2_result_sha256,
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
        "iteration_status": {
            "same_scene_validation_was_seen_for_v2_method_diagnosis": True,
            "same_scene_gate_is_development_evidence": True,
            "cross_scene_confirmation_required_before_benchmark": True,
        },
        "source_access": {
            "validation_label_payload_may_be_deserialized_by_shared_v1_loader": True,
            "validation_derived_tensor_prediction_or_metric_enters_fit": False,
            "validation_prediction_and_metric_evaluation_started_after_fit_receipt_frozen": True,
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
                "fit_by_mode": {
                    mode: {
                        "temperature": fit_by_mode[mode]["temperature"],
                        "bias": fit_by_mode[mode]["bias"],
                    }
                    for mode in PROMPT_MODES
                },
                "promoted": promoted,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
