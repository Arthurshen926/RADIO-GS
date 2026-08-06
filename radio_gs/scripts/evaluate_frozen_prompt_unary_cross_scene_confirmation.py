#!/usr/bin/env python3
"""Evaluate frozen prompt-unary candidates on one independent clean scene.

This evaluator never fits a parameter or selects a threshold.  It reuses the
source-only prompt construction of the frozen V1 pilot, applies the sealed V1
head and V2/V3/V4 monotone calibrators, and reports only clean ScanNet source
confirmation metrics.  Benchmark assets are outside this program's inputs.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import torch

from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)
from radio_gs.querying.observation_conditioned_prompt_calibrator import (
    ObservationConditionedPromptCalibratorV4,
)
from radio_gs.querying.prompt_mode_logit_calibrator import (
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
    stable_descriptor_load,
    write_frozen_json,
)


AUTHORITY_SCHEMA = "radio_gs.prompt_unary.cross_scene_execution_authority.v1"
RESULT_SCHEMA = "radio_gs.prompt_unary.cross_scene_clean_confirmation_result.v1"
METRIC_KEYS = (
    "average_precision",
    "auroc",
    "oracle_iou",
    "iou_at_0_5",
    "precision_at_0_5",
    "recall_at_0_5",
    "area_ratio",
    "bce",
)


def _verify_record(record: object, path: Path, *, label: str) -> str:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"execution authority {label} record differs")
    resolved = path.expanduser().resolve()
    if Path(str(record["path"])).expanduser().resolve() != resolved:
        raise ValueError(f"execution authority {label} path differs")
    expected = str(record["sha256"])
    if sha256_file(resolved) != expected:
        raise ValueError(f"execution authority {label} SHA-256 differs")
    return expected


def _validate_authority(args: argparse.Namespace) -> tuple[dict, bytes]:
    authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="cross-scene prompt-unary execution authority",
    )
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("scene_id") != args.scene_id
        or authority.get("fit_or_parameter_update") is not False
        or authority.get("threshold") != 0.5
    ):
        raise ValueError("cross-scene execution authority contract differs")
    records = authority.get("inputs")
    if not isinstance(records, dict):
        raise ValueError("cross-scene execution input records differ")
    paths = {
        "preregistration": args.preregistration,
        "independent_split_addendum": args.independent_split_addendum,
        "responsibility_manifest": args.responsibility_manifest,
        "label_zip": args.label_zip,
        "capability_bank": args.capability_bank,
        "factorized_state": args.factorized_state,
        "v1_checkpoint": args.v1_checkpoint,
        "v2_checkpoint": args.v2_checkpoint,
        "v3_checkpoint": args.v3_checkpoint,
        "v4_checkpoint": args.v4_checkpoint,
        "implementation": Path(__file__).resolve(),
    }
    expected = {
        "preregistration": args.expected_preregistration_sha256,
        "independent_split_addendum": args.expected_independent_split_addendum_sha256,
        "responsibility_manifest": args.expected_responsibility_manifest_sha256,
        "label_zip": args.expected_label_zip_sha256,
        "capability_bank": args.expected_capability_bank_sha256,
        "factorized_state": args.expected_factorized_state_sha256,
        "v1_checkpoint": args.expected_v1_checkpoint_sha256,
        "v2_checkpoint": args.expected_v2_checkpoint_sha256,
        "v3_checkpoint": args.expected_v3_checkpoint_sha256,
        "v4_checkpoint": args.expected_v4_checkpoint_sha256,
    }
    for label, path in paths.items():
        digest = _verify_record(records.get(label), path, label=label)
        if label in expected and digest != str(expected[label]):
            raise ValueError(f"CLI and execution authority {label} SHA-256 differ")
    for label in ("v1_source_result", "v2_source_result", "v3_source_result", "v4_source_result"):
        record = records.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"execution authority {label} record differs")
        _verify_record(record, Path(str(record.get("path", ""))), label=label)
    access = authority.get("source_access", {})
    if any(
        access.get(key) is not False
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "benchmark_queries_opened",
            "benchmark_labels_opened",
            "benchmark_metrics_opened",
            "scene0002_labels_enter_fit",
        )
    ):
        raise ValueError("cross-scene source-access contract differs")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    label_bytes, _, _ = stable_descriptor_load(
        args.label_zip,
        lambda handle: handle.read(),
        expected_sha256=args.expected_label_zip_sha256,
        label="official scene0002 instance zip",
    )
    authority["verified_authority_path"] = str(authority_path)
    authority["verified_authority_sha256"] = authority_sha
    return authority, label_bytes


def _load_source_gate_receipts(authority: dict) -> dict[str, dict]:
    records = authority["inputs"]
    loaded: dict[str, dict] = {}
    for stage in ("v1", "v2", "v3", "v4"):
        record = records[f"{stage}_source_result"]
        payload, _, _ = load_json_object(
            Path(str(record["path"])),
            expected_sha256=str(record["sha256"]),
            label=f"frozen {stage.upper()} source result",
        )
        loaded[stage.upper()] = payload
    expected_gate_states = {
        "V1": (False, "stop_v1_before_benchmarks"),
        "V2": (True, "eligible_for_cross_scene_confirmation"),
        "V3": (False, "stop_before_benchmarks"),
        "V4": (False, "stop_before_benchmarks"),
    }
    if any(
        payload.get("promotion_gate_passed") is not expected[0]
        or payload.get("decision") != expected[1]
        for stage, payload in loaded.items()
        for expected in (expected_gate_states[stage],)
    ):
        raise ValueError("frozen source candidate gate states differ")
    return loaded


def _load_views(
    *,
    manifest: dict,
    manifest_path: Path,
    label_bytes: bytes,
) -> tuple[dict[int, v1.SparseView], int, int, int]:
    height = int(manifest["metadata"]["feature_height"])
    width = int(manifest["metadata"]["feature_width"])
    frame_ids = [int(value) for value in manifest["frame_indices"]]
    records = manifest.get("views")
    if not isinstance(records, list) or len(records) != len(frame_ids):
        raise ValueError("exact responsibility view records differ")
    views_root = Path(str(manifest_path) + ".views")
    first, _, _ = load_torch_payload(
        views_root / "view_00000.pt",
        expected_sha256=str(records[0]["sha256"]),
        map_location="cpu",
        label="first exact responsibility view",
    )
    num_gaussians = int(first["num_gaussians"])
    with zipfile.ZipFile(io.BytesIO(label_bytes), "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"official instance zip CRC failure: {bad_member}")
        views = {
            frame: v1._load_sparse_view(
                view_path=views_root / f"view_{index:05d}.pt",
                frame_id=frame,
                label_archive=archive,
                expected_view_sha256=str(records[index]["sha256"]),
                height=height,
                width=width,
                num_gaussians=num_gaussians,
            )
            for index, frame in enumerate(frame_ids)
        }
    return views, height, width, num_gaussians


def _eligible_instances(views: dict[int, v1.SparseView]) -> dict[int, list[int]]:
    occurrences: dict[int, list[int]] = {}
    for frame, view in views.items():
        ids, counts = torch.unique(view.instance_image, return_counts=True)
        for instance_id, count in zip(ids.tolist(), counts.tolist()):
            if instance_id > 0 and count >= v1.MIN_PIXELS:
                occurrences.setdefault(int(instance_id), []).append(frame)
    return {
        instance_id: frames
        for instance_id, frames in occurrences.items()
        if len(frames) >= v1.MIN_VIEWS
    }


@torch.no_grad()
def _collect_predictions(
    *,
    head: RegisteredEvidenceToUnaryV1,
    examples: list[v1.PromptExample],
    views: dict[int, v1.SparseView],
    device: torch.device,
) -> dict[tuple[int, str], dict[str, torch.Tensor | int]]:
    head.eval()
    collected: dict[tuple[int, str], dict[str, torch.Tensor | int]] = {}
    for example in examples:
        analytic_parts: list[torch.Tensor] = []
        v1_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for frame in example.target_frames:
            view = views[frame]
            unique, inverse = v1._view_rows(view, device)
            subset = v1._subset_features(example.features, unique)
            output = head(subset)
            prediction, supported = v1._render_unique(
                output.foreground_probability, view, unique, inverse, device
            )
            analytic, _ = v1._render_unique(
                subset.analytic_probability, view, unique, inverse, device
            )
            target = view.instance_image.to(device) == example.instance_id
            analytic_parts.append(analytic[supported].cpu())
            v1_parts.append(prediction[supported].cpu())
            label_parts.append(target[supported].cpu())
        collected[(example.instance_id, example.mode)] = {
            "analytic": torch.cat(analytic_parts),
            "V1": torch.cat(v1_parts),
            "label": torch.cat(label_parts),
            "source_frame": int(example.source_frame),
            "target_view_count": len(example.target_frames),
        }
    return collected


def _macro(records: list[dict], stage: str, mode: str | None = None) -> dict[str, float]:
    selected = [row for row in records if mode is None or row["mode"] == mode]
    if not selected:
        raise ValueError("macro partition is empty")
    return {
        key: float(sum(float(row[stage][key]) for row in selected) / len(selected))
        for key in METRIC_KEYS
    }


def _load_models(args: argparse.Namespace) -> tuple[
    RegisteredEvidenceToUnaryV1,
    GlobalPromptLogitCalibratorV2,
    PromptModeLogitCalibratorV3,
    ObservationConditionedPromptCalibratorV4,
]:
    v1_payload, _, _ = load_torch_payload(
        args.v1_checkpoint,
        expected_sha256=args.expected_v1_checkpoint_sha256,
        map_location="cpu",
        label="frozen V1 checkpoint",
    )
    if v1_payload.get("schema") != "radio_gs.registered_evidence_to_unary.checkpoint.v1":
        raise ValueError("frozen V1 checkpoint schema differs")
    head = RegisteredEvidenceToUnaryV1(hidden_dim=32, max_delta_logit=4.0)
    head.load_state_dict(v1_payload["state_dict"], strict=True)
    models: list[torch.nn.Module] = []
    specifications = (
        (
            args.v2_checkpoint,
            args.expected_v2_checkpoint_sha256,
            "radio_gs.global_prompt_logit_calibrator.checkpoint.v2",
            GlobalPromptLogitCalibratorV2(),
        ),
        (
            args.v3_checkpoint,
            args.expected_v3_checkpoint_sha256,
            "radio_gs.prompt_mode_logit_calibrator.checkpoint.v3",
            PromptModeLogitCalibratorV3(),
        ),
        (
            args.v4_checkpoint,
            args.expected_v4_checkpoint_sha256,
            "radio_gs.observation_conditioned_prompt_calibrator.checkpoint.v4",
            ObservationConditionedPromptCalibratorV4(),
        ),
    )
    for path, expected_sha, schema, model in specifications:
        payload, _, _ = load_torch_payload(
            path,
            expected_sha256=expected_sha,
            map_location="cpu",
            label=f"frozen {schema} checkpoint",
        )
        if payload.get("schema") != schema:
            raise ValueError(f"frozen calibrator schema differs: {schema}")
        model = model.double()
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        models.append(model)
    head.eval().requires_grad_(False)
    return head, models[0], models[1], models[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0002_00")
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--independent-split-addendum", type=Path, required=True)
    parser.add_argument("--expected-independent-split-addendum-sha256", required=True)
    parser.add_argument("--execution-authority", type=Path, required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    parser.add_argument("--responsibility-manifest", type=Path, required=True)
    parser.add_argument("--expected-responsibility-manifest-sha256", required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--expected-label-zip-sha256", required=True)
    parser.add_argument("--capability-bank", type=Path, required=True)
    parser.add_argument("--expected-capability-bank-sha256", required=True)
    parser.add_argument("--factorized-state", type=Path, required=True)
    parser.add_argument("--expected-factorized-state-sha256", required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v1-checkpoint-sha256", required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v2-checkpoint-sha256", required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v3-checkpoint-sha256", required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v4-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.scene_id != "scene0002_00":
        raise ValueError("this sealed confirmation is restricted to scene0002_00")
    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError("the sealed confirmation evaluator is CPU-only")
    v1.SCENE_ID = args.scene_id
    authority, label_bytes = _validate_authority(args)
    source_receipts = _load_source_gate_receipts(authority)
    manifest, manifest_sha, _ = load_json_object(
        args.responsibility_manifest,
        expected_sha256=args.expected_responsibility_manifest_sha256,
        label="exact responsibility manifest",
    )
    views, height, width, num_gaussians = _load_views(
        manifest=manifest,
        manifest_path=args.responsibility_manifest,
        label_bytes=label_bytes,
    )
    eligible = _eligible_instances(views)
    train_ids = sorted(i for i in eligible if v1.instance_split(i) == "train")
    validation_ids = sorted(i for i in eligible if v1.instance_split(i) == "validation")
    if len(validation_ids) < 3 or set(train_ids) & set(validation_ids):
        raise RuntimeError("frozen scene0002 independent split is invalid or underpowered")

    capability, _, _ = load_torch_payload(
        args.capability_bank,
        expected_sha256=args.expected_capability_bank_sha256,
        map_location="cpu",
        label="scene0002 canonical capability bank",
    )
    state, _, _ = load_torch_payload(
        args.factorized_state,
        expected_sha256=args.expected_factorized_state_sha256,
        map_location="cpu",
        label="scene0002 factorized primitive state",
    )
    cross_asset = v1._validate_cross_asset_authority(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        capability=capability,
        state=state,
    )
    validation_eligible = {instance_id: eligible[instance_id] for instance_id in validation_ids}
    examples = v1.build_examples(
        views=views,
        eligible=validation_eligible,
        capability_bank=capability,
        factorized_state=state,
        device=device,
        height=height,
        width=width,
    )
    del capability, state
    head, calibrator_v2, calibrator_v3, calibrator_v4 = _load_models(args)
    predictions = _collect_predictions(
        head=head, examples=examples, views=views, device=device
    )

    records: list[dict] = []
    ranking_audits: dict[str, list[dict]] = {"V2": [], "V3": [], "V4": []}
    calibrators = {
        "V2": lambda score, mode: calibrator_v2.calibrated_logit(score.double()),
        "V3": lambda score, mode: calibrator_v3.calibrated_logit(score.double(), mode=mode),
        "V4": lambda score, mode: calibrator_v4.calibrated_logit(score.double(), mode=mode),
    }
    for (instance_id, mode), values in sorted(predictions.items()):
        label = torch.as_tensor(values["label"])
        analytic_score = torch.as_tensor(values["analytic"])
        v1_score = torch.as_tensor(values["V1"])
        record = {
            "instance_id": instance_id,
            "mode": mode,
            "source_frame": int(values["source_frame"]),
            "target_view_count": int(values["target_view_count"]),
            "analytic": v2._ranking_and_probability_metrics(
                ranking_score=analytic_score,
                probability=analytic_score,
                label=label,
            ),
            "V1": v2._ranking_and_probability_metrics(
                ranking_score=v1_score,
                probability=v1_score,
                label=label,
            ),
        }
        for stage, transform in calibrators.items():
            logit = transform(v1_score, mode)
            probability = torch.sigmoid(logit)
            audit = v2._strict_ranking_audit(v1_score.double(), logit)
            ranking_audits[stage].append(
                {"instance_id": instance_id, "mode": mode, **audit}
            )
            record[stage] = v2._ranking_and_probability_metrics(
                ranking_score=logit,
                probability=probability,
                label=label,
            )
        records.append(record)

    stages = ("analytic", "V1", "V2", "V3", "V4")
    macro = {
        partition: {
            stage: _macro(records, stage, None if partition == "all" else partition)
            for stage in stages
        }
        for partition in ("all", "full_mask", "scribble")
    }
    ranking_controls = {
        stage: {
            "all_stable_argsort_equal": all(row["stable_argsort_equal"] for row in audits),
            "all_tie_partitions_equal": all(row["tie_partition_equal"] for row in audits),
            "maximum_ap_absolute_error_vs_v1": max(
                abs(row[stage]["average_precision"] - row["V1"]["average_precision"])
                for row in records
            ),
            "maximum_auroc_absolute_error_vs_v1": max(
                abs(row[stage]["auroc"] - row["V1"]["auroc"])
                for row in records
            ),
            "maximum_oracle_iou_absolute_error_vs_v1": max(
                abs(row[stage]["oracle_iou"] - row["V1"]["oracle_iou"])
                for row in records
            ),
        }
        for stage, audits in ranking_audits.items()
    }
    for stage, control in ranking_controls.items():
        if (
            not control["all_stable_argsort_equal"]
            or not control["all_tie_partitions_equal"]
            or max(
                control["maximum_ap_absolute_error_vs_v1"],
                control["maximum_auroc_absolute_error_vs_v1"],
                control["maximum_oracle_iou_absolute_error_vs_v1"],
            )
            > 1e-10
        ):
            raise RuntimeError(f"{stage} violates the frozen ranking invariant")

    analytic = macro["all"]["analytic"]
    v1_macro = macro["all"]["V1"]
    worst_ap_delta = {
        stage: min(
            row[stage]["average_precision"] - row["analytic"]["average_precision"]
            for row in records
        )
        for stage in ("V1", "V2", "V3", "V4")
    }
    ranking_transfer = (
        v1_macro["average_precision"] > analytic["average_precision"]
        and v1_macro["oracle_iou"] > analytic["oracle_iou"]
        and worst_ap_delta["V1"] >= -0.05
    )
    v2_macro = macro["all"]["V2"]
    v2_cross_scene_gate = (
        ranking_transfer
        and ranking_controls["V2"]["all_stable_argsort_equal"]
        and ranking_controls["V2"]["all_tie_partitions_equal"]
        and 0.8 <= v2_macro["area_ratio"] <= 1.25
        and v2_macro["iou_at_0_5"] >= analytic["iou_at_0_5"]
        and v2_macro["precision_at_0_5"] >= analytic["precision_at_0_5"]
        and worst_ap_delta["V2"] >= -0.05
    )
    eligibility = {
        "analytic": {"eligible": False, "reason": "comparator_only"},
        "V1": {
            "eligible": False,
            "ranking_transfer_confirmed": ranking_transfer,
            "reason": "original_scene0001_whole_gate_failed",
        },
        "V2": {
            "eligible": bool(v2_cross_scene_gate),
            "reason": (
                "passed_original_whole_gate_and_independent_cross_scene_gate"
                if v2_cross_scene_gate
                else "independent_cross_scene_gate_failed"
            ),
        },
        "V3": {
            "eligible": False,
            "reason": "original_formal_whole_gate_failed_stop_before_benchmarks",
        },
        "V4": {
            "eligible": False,
            "reason": "original_formal_whole_gate_failed_and_calibration_branch_closed",
        },
    }
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "scene_id": args.scene_id,
        "status": "cross_scene_clean_confirmation_complete",
        "fit_or_parameter_update": False,
        "threshold": 0.5,
        "graph": "off",
        "connected_selection": "off",
        "eligible_instance_ids": sorted(eligible),
        "scene0002_train_instance_ids_inventory_only": train_ids,
        "scene0002_validation_instance_ids_evaluated": validation_ids,
        "independent_split": True,
        "source_target_view_disjoint": all(
            example.source_frame not in example.target_frames for example in examples
        ),
        "metrics": {"records": records, "macro": macro},
        "ranking_controls": ranking_controls,
        "worst_prompt_ap_delta_vs_analytic": worst_ap_delta,
        "candidate_eligibility": eligibility,
        "source_gate_states": {
            stage: {
                "promotion_gate_passed": source_receipts[stage].get(
                    "promotion_gate_passed"
                ),
                "decision": source_receipts[stage].get("decision"),
            }
            for stage in ("V1", "V2", "V3", "V4")
        },
        "cross_asset_authority": cross_asset,
        "authority": {
            "execution_authority": {
                "path": authority["verified_authority_path"],
                "sha256": authority["verified_authority_sha256"],
            },
            "responsibility_manifest_sha256": manifest_sha,
            "label_zip_sha256": args.expected_label_zip_sha256,
            "capability_bank_sha256": args.expected_capability_bank_sha256,
            "factorized_state_sha256": args.expected_factorized_state_sha256,
        },
        "source_access": {
            "scene0002_labels_used_only_for_clean_confirmation_metrics": True,
            "scene0002_labels_enter_fit": False,
            "scene0002_target_rgb_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_queries_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
            "per_scene_tuning": False,
        },
    }
    write_frozen_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_instances": len(validation_ids),
                "eligible_candidate": "V2" if v2_cross_scene_gate else None,
                "macro": macro,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
