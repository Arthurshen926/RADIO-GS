#!/usr/bin/env python3
"""Evaluate the preregistered Field-A representation gate without labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import torch

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from radio_gs.scripts.train_canonical_radio_field import (
    CAPABILITY_TARGET_CONTRACT_FIELD_A,
    _capability_reconstruction_metrics,
    _consensus_from_cache,
    _load_capability_mpr_target,
    _load_field_a_observation_reference,
    _reconstruction_metrics,
)
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


def evaluate_gate(initial: dict[str, float], final: dict[str, float]) -> dict:
    deltas = {key: float(final[key] - initial[key]) for key in final}
    checks = {
        "raw_mean_cosine_no_regression_gt_0.005": (
            deltas["mean_cosine"] >= -0.005
        ),
        "raw_p05_cosine_no_regression_gt_0.010": (
            deltas["p05_cosine"] >= -0.010
        ),
        "dino_v3_exact_mean_cosine_improves": (
            deltas["dino_v3_target_mean_cosine"] > 0.0
        ),
        "sam3_exact_mean_cosine_improves": (
            deltas["sam3_target_mean_cosine"] > 0.0
        ),
    }
    return {
        "initial": initial,
        "final": final,
        "delta": deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _all_field_tensors_finite(field: torch.nn.Module) -> bool:
    for value in list(field.parameters()) + list(field.buffers()):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            return False
    return True


def _canonical_architecture(value: dict) -> dict:
    """Normalize only the schema default introduced after legacy checkpoints."""
    result = dict(value)
    result.setdefault("fusion_residual_blocks", 0)
    return result


def _same_state_structure(
    initial: torch.nn.Module,
    final: torch.nn.Module,
) -> bool:
    initial_state = initial.state_dict()
    final_state = final.state_dict()
    return set(initial_state) == set(final_state) and all(
        initial_state[key].shape == final_state[key].shape for key in initial_state
    )


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _training_runtime_receipt(
    telemetry_path: str | Path,
    owner_audit_path: str | Path,
) -> dict:
    telemetry_path = Path(telemetry_path).expanduser().resolve()
    owner_audit_path = Path(owner_audit_path).expanduser().resolve()
    with telemetry_path.open(newline="", encoding="utf-8") as handle:
        telemetry = list(csv.DictReader(handle))
    with owner_audit_path.open(newline="", encoding="utf-8") as handle:
        owner_audit = list(csv.DictReader(handle))
    if not telemetry or not owner_audit:
        raise ValueError("Field-A training telemetry/audit is empty")
    required_telemetry = {
        "gpu",
        "bus_id",
        "temp_c",
        "power_w",
        "power_limit_w",
        "util_pct",
        "memory_mib",
        "event",
    }
    if not required_telemetry.issubset(telemetry[0]):
        raise ValueError("Field-A training telemetry columns differ")
    if any(row["gpu"] != "1" for row in telemetry):
        raise ValueError("Field-A training telemetry is not physical GPU1")
    if any(float(row["power_limit_w"]) > 300.5 for row in telemetry):
        raise ValueError("Field-A training exceeded the registered power limit")
    events = [row["event"] for row in telemetry]
    unsafe_events = [
        event
        for event in events
        if "abort" in event
        or "failed" in event
        or "foreign_compute_owner" in event
    ]
    if unsafe_events:
        raise ValueError(f"Field-A training guard recorded unsafe events: {unsafe_events}")
    if events[-1] != "cuda_release_verified_no_compute_owner":
        raise ValueError("Field-A training did not verify CUDA release")
    foreign_owners = [
        row.get("foreign_owner_pids", "").strip()
        for row in owner_audit
        if row.get("foreign_owner_pids", "").strip()
    ]
    owner_events = [row.get("event", "") for row in owner_audit]
    if foreign_owners:
        raise ValueError("Field-A training shared physical GPU1 with a foreign owner")
    if "prelaunch_owner_clear" not in owner_events or "postexit_owner_clear" not in (
        owner_events
    ):
        raise ValueError("Field-A training owner-clear audit is incomplete")
    return {
        "physical_gpu": 1,
        "bus_ids": sorted({row["bus_id"] for row in telemetry}),
        "samples": len(telemetry),
        "max_temperature_c": max(int(row["temp_c"]) for row in telemetry),
        "max_power_w": max(float(row["power_w"]) for row in telemetry),
        "max_power_limit_w": max(
            float(row["power_limit_w"]) for row in telemetry
        ),
        "max_utilization_pct": max(int(row["util_pct"]) for row in telemetry),
        "max_memory_mib": max(int(row["memory_mib"]) for row in telemetry),
        "soft_pause_events": sum(event.startswith("soft_pause") for event in events),
        "thermal_abort_events": sum("thermal_abort" in event for event in events),
        "unsafe_events": unsafe_events,
        "cuda_release_verified": True,
        "foreign_owner_pids": foreign_owners,
        "prelaunch_owner_clear": True,
        "postexit_owner_clear": True,
        "telemetry": {
            "path": str(telemetry_path),
            "sha256": sha256_file(telemetry_path),
        },
        "owner_audit": {
            "path": str(owner_audit_path),
            "sha256": sha256_file(owner_audit_path),
        },
    }


def run(args: argparse.Namespace) -> dict:
    registration, registration_sha, registration_path = load_json_object(
        args.experiment_registration,
        expected_sha256=args.expected_experiment_registration_sha256,
        label="Field-A experiment registration",
    )
    if registration.get("schema_version") != (
        "canonical_field_a_exact_capability_registration_v1"
    ):
        raise ValueError("Field-A experiment registration schema differs")

    primary, primary_sha, primary_path = load_mpr_cache(
        args.mpr_cache,
        expected_sha256=args.expected_mpr_cache_sha256,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=False,
    )
    primary_metadata = dict(primary["metadata"])
    consensus = _consensus_from_cache(primary)
    observation, observation_metadata, observation_provenance = (
        _load_field_a_observation_reference(
            args.capability_observation_reference_mpr_cache,
            primary_raw_cache=primary,
            primary_raw_metadata=primary_metadata,
            expected_cache_sha256=(
                args.expected_capability_observation_reference_mpr_cache_sha256
            ),
        )
    )
    radio_sha = sha256_file(args.radio_checkpoint)
    if radio_sha != args.expected_radio_checkpoint_sha256:
        raise ValueError("official RADIO checkpoint SHA-256 differs")
    targets = {}
    target_provenance = {}
    for space, path, digest in (
        ("dino_v3", args.dino_mpr_cache, args.expected_dino_v3_mpr_cache_sha256),
        ("sam3", args.sam3_mpr_cache, args.expected_sam3_mpr_cache_sha256),
    ):
        target, provenance = _load_capability_mpr_target(
            path,
            expected_space=space,
            raw_cache=observation,
            raw_metadata=observation_metadata,
            radio_checkpoint_sha256=radio_sha,
            expected_cache_sha256=digest,
            expected_feature_output_bundle_sha256=(
                args.expected_feature_output_bundle_sha256
            ),
            target_contract=CAPABILITY_TARGET_CONTRACT_FIELD_A,
        )
        targets[space] = target
        target_provenance[space] = provenance

    initial_field, initial_payload = load_canonical_field_checkpoint(
        args.initial_field_checkpoint,
        map_location="cpu",
        expected_sha256=args.expected_initial_field_checkpoint_sha256,
    )
    final_field, final_payload = load_canonical_field_checkpoint(
        args.final_field_checkpoint,
        map_location="cpu",
        expected_sha256=args.expected_final_field_checkpoint_sha256,
    )
    if _canonical_architecture(initial_payload.get("architecture", {})) != (
        _canonical_architecture(final_payload.get("architecture", {}))
    ) or not _same_state_structure(initial_field, final_field):
        raise ValueError("Field-A changed the canonical field architecture")
    expected_geometry = primary.get("geometry_fingerprint", {})
    for label, payload in (("initial", initial_payload), ("final", final_payload)):
        if payload.get("geometry_fingerprint") != expected_geometry:
            raise ValueError(f"{label} field geometry differs from primary MPR")
        if any(
            payload.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
            if key in payload or label == "final"
        ):
            raise ValueError(f"{label} field is benchmark/query contaminated")
    if not _all_field_tensors_finite(initial_field) or not _all_field_tensors_finite(
        final_field
    ):
        raise ValueError("initial/final field contains non-finite state")

    history = list(final_payload.get("history", []))
    training = dict(final_payload.get("training_config", {}))
    loss = dict(final_payload.get("loss_config", {}))
    expected_loss = {
        "mpr_weight": 1.0,
        "dino_weight": 0.2,
        "sam3_weight": 0.2,
        "relation_weight": 0.0,
        "coefficient_weight": 1e-5,
        "basis_orthogonality_weight": 1e-3,
    }
    if loss != expected_loss:
        raise ValueError("Field-A loss weights differ from registration")
    expected_training = {
        "epochs": 20,
        "min_epochs": 20,
        "batch_size": 4096,
        "eval_batch_size": 16384,
        "learning_rate": 0.002,
        "weight_decay": 1e-5,
        "validation_fraction": 0.05,
        "seed": 0,
        "observation_contract": "compatible-legacy",
        "capability_target_contract": "field_a_exact_adjoint",
    }
    mismatched_training = {
        key: [training.get(key), value]
        for key, value in expected_training.items()
        if training.get(key) != value
    }
    if mismatched_training:
        raise ValueError(f"Field-A training config differs: {mismatched_training}")
    if len(history) != 20 or [int(row.get("epoch", -1)) for row in history] != list(
        range(1, 21)
    ):
        raise ValueError("Field-A did not complete the fixed 20 epochs")
    if final_payload.get("capability_target_contract") != (
        "field_a_exact_adjoint"
    ) or final_payload.get("capability_reliability_policy") != (
        "field_a_boundary_safe"
    ):
        raise ValueError("Field-A target/reliability contract differs")

    device = torch.device(args.device)
    official = FrozenRadioViews.from_radio_checkpoint(
        args.radio_checkpoint,
        expected_sha256=radio_sha,
    ).to(device).eval()
    official.requires_grad_(False)
    rows = torch.where(consensus.valid)[0]
    metrics = {}
    for label, field in (("initial", initial_field), ("final", final_field)):
        field = field.to(device).eval()
        values = _reconstruction_metrics(
            field,
            consensus,
            rows,
            args.batch_size,
        )
        values.update(
            _capability_reconstruction_metrics(
                field,
                official,
                targets,
                rows,
                args.batch_size,
            )
        )
        metrics[label] = values
        field.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gate = evaluate_gate(metrics["initial"], metrics["final"])
    training_runtime = _training_runtime_receipt(
        args.training_telemetry,
        args.training_owner_audit,
    )
    stored = dict(final_payload.get("final_metrics", {}))
    stored.update(final_payload.get("final_capability_metrics", {}))
    for key, value in metrics["final"].items():
        if abs(float(stored.get(key, float("nan"))) - float(value)) > 2e-5:
            raise ValueError(f"stored/recomputed Field-A metric differs: {key}")

    receipt = {
        "schema_version": "canonical_field_a_label_free_gate_receipt_v1",
        "experiment_registration": {
            "path": str(registration_path),
            "sha256": registration_sha,
        },
        "initial_field": {
            "path": str(Path(args.initial_field_checkpoint).resolve()),
            "sha256": args.expected_initial_field_checkpoint_sha256,
            "file_bytes": Path(args.initial_field_checkpoint).stat().st_size,
        },
        "final_field": {
            "path": str(Path(args.final_field_checkpoint).resolve()),
            "sha256": args.expected_final_field_checkpoint_sha256,
            "file_bytes": Path(args.final_field_checkpoint).stat().st_size,
            "architecture_unchanged": True,
            "all_state_finite": True,
            "training_config_sha256": final_payload.get("training_config_sha256"),
            "history_epochs": len(history),
            "stored_final_metrics": stored,
        },
        "primary_mpr": {"path": str(primary_path), "sha256": primary_sha},
        "capability_observation_reference": observation_provenance,
        "capability_targets": target_provenance,
        "feature_output_bundle_sha256": args.expected_feature_output_bundle_sha256,
        "loss_config": loss,
        "fixed_training_config": expected_training,
        "training_runtime": training_runtime,
        "gate": gate,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "downstream_evaluation_started": False,
        "source_hashes": {
            "gate": sha256_file(Path(__file__).resolve()),
            "trainer": sha256_file(
                Path(__file__).with_name("train_canonical_radio_field.py")
            ),
        },
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, receipt)
    return {**receipt, "output": str(output), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--expected-experiment-registration-sha256", required=True)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--expected-mpr-cache-sha256", required=True)
    parser.add_argument("--capability-observation-reference-mpr-cache", required=True)
    parser.add_argument(
        "--expected-capability-observation-reference-mpr-cache-sha256",
        required=True,
    )
    parser.add_argument("--dino-mpr-cache", required=True)
    parser.add_argument("--expected-dino-v3-mpr-cache-sha256", required=True)
    parser.add_argument("--sam3-mpr-cache", required=True)
    parser.add_argument("--expected-sam3-mpr-cache-sha256", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--expected-radio-checkpoint-sha256", required=True)
    parser.add_argument("--expected-feature-output-bundle-sha256", required=True)
    parser.add_argument("--initial-field-checkpoint", required=True)
    parser.add_argument("--expected-initial-field-checkpoint-sha256", required=True)
    parser.add_argument("--final-field-checkpoint", required=True)
    parser.add_argument("--expected-final-field-checkpoint-sha256", required=True)
    parser.add_argument("--training-telemetry", required=True)
    parser.add_argument("--training-owner-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
