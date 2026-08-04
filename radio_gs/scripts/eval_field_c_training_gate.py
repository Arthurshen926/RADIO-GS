#!/usr/bin/env python3
"""Evaluate the preregistered Field-C training gate without task labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from radio_gs.scripts.train_canonical_radio_field import (
    CAPABILITY_TARGET_CONTRACT_FIELD_C,
    _capability_reconstruction_metrics,
    _consensus_from_cache,
    _load_capability_mpr_target,
    _reconstruction_metrics,
)
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


def evaluate_gate(
    initial: dict[str, float],
    final: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, object]:
    deltas = {key: float(final[key] - initial[key]) for key in final}
    tolerance = 1e-12
    checks = {
        "raw_mean_cosine": deltas["mean_cosine"]
        >= float(thresholds["raw_mean_cosine_delta_minimum"]) - tolerance,
        "raw_p05_cosine": deltas["p05_cosine"]
        >= float(thresholds["raw_p05_cosine_delta_minimum"]) - tolerance,
        "dino_v3_mean_cosine": deltas["dino_v3_target_mean_cosine"]
        >= float(thresholds["dino_v3_mean_cosine_delta_minimum"]) - tolerance,
        "sam3_mean_cosine": deltas["sam3_target_mean_cosine"]
        >= float(thresholds["sam3_mean_cosine_delta_minimum"]) - tolerance,
    }
    return {
        "initial": initial,
        "final": final,
        "delta": deltas,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _canonical_architecture(value: dict) -> dict:
    result = dict(value)
    result.setdefault("fusion_residual_blocks", 0)
    return result


def _same_state_structure(initial: torch.nn.Module, final: torch.nn.Module) -> bool:
    initial_state = initial.state_dict()
    final_state = final.state_dict()
    return set(initial_state) == set(final_state) and all(
        initial_state[key].shape == final_state[key].shape for key in initial_state
    )


def _all_field_tensors_finite(field: torch.nn.Module) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in list(field.parameters()) + list(field.buffers())
    )


def _atomic_json(path: Path, value: object) -> None:
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


def _registered_input(
    registration: dict, name: str, *, expected_sha256: str = ""
) -> tuple[Path, str]:
    record = dict(registration["immutable_inputs"][name])
    path = Path(record["path"]).expanduser().resolve()
    digest = str(record["sha256"])
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"registered {name} SHA-256 differs from CLI authority")
    if sha256_file(path) != digest:
        raise ValueError(f"registered {name} artifact differs")
    return path, digest


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    registration, registration_sha, registration_path = load_json_object(
        args.experiment_registration,
        expected_sha256=args.expected_experiment_registration_sha256,
        label="Field-C training registration",
    )
    if registration.get("schema_version") != "canonical_field_c_training_registration_v1":
        raise ValueError("Field-C training registration schema differs")
    if any(value is not False for value in dict(registration["safety"]).values()):
        raise ValueError("Field-C training registration is not target blind")

    implementation = dict(registration["implementation_sha256"])
    source_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "train_canonical_radio_field.py": source_root
        / "scripts"
        / "train_canonical_radio_field.py",
        "canonical_field_losses.py": source_root
        / "training"
        / "canonical_field_losses.py",
    }
    if set(implementation) != set(source_paths):
        raise ValueError("Field-C registered implementation authority differs")
    for name, path in source_paths.items():
        if sha256_file(path) != implementation[name]:
            raise ValueError(f"Field-C implementation changed after registration: {name}")

    initial_path, initial_sha = _registered_input(registration, "initial_field")
    raw_path, raw_sha = _registered_input(registration, "raw_radio")
    dino_path, dino_sha = _registered_input(registration, "dino_v3")
    sam_path, sam_sha = _registered_input(registration, "sam3")
    radio_path, radio_sha = _registered_input(
        registration, "official_c_radio_checkpoint"
    )
    expected_bundle = str(
        registration["immutable_inputs"]["feature_output_bundle_sha256"]
    )
    if len(expected_bundle) != 64:
        raise ValueError("Field-C feature output bundle authority is malformed")

    raw, loaded_raw_sha, raw_loaded_path = load_mpr_cache(
        raw_path,
        expected_sha256=raw_sha,
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=True,
    )
    raw_metadata = dict(raw["metadata"])
    if raw_metadata.get("aggregation_mode") != "raster_exact_center_uncertainty":
        raise ValueError("Field-C raw target is not exact-center uncertainty MPR")
    consensus = _consensus_from_cache(raw)
    targets: dict[str, object] = {}
    target_provenance: dict[str, object] = {}
    for space, path, digest in (
        ("dino_v3", dino_path, dino_sha),
        ("sam3", sam_path, sam_sha),
    ):
        target, provenance = _load_capability_mpr_target(
            path,
            expected_space=space,
            raw_cache=raw,
            raw_metadata=raw_metadata,
            radio_checkpoint_sha256=radio_sha,
            expected_cache_sha256=digest,
            expected_feature_output_bundle_sha256=expected_bundle,
            target_contract=CAPABILITY_TARGET_CONTRACT_FIELD_C,
        )
        targets[space] = target
        target_provenance[space] = provenance

    final_path = Path(args.final_field_checkpoint).expanduser().resolve()
    if final_path != Path(registration["output"]).expanduser().resolve():
        raise ValueError("Field-C final field path differs from registration")
    if sha256_file(final_path) != args.expected_final_field_checkpoint_sha256:
        raise ValueError("Field-C final field SHA-256 differs")
    initial_field, initial_payload = load_canonical_field_checkpoint(
        initial_path, map_location="cpu", expected_sha256=initial_sha
    )
    final_field, final_payload = load_canonical_field_checkpoint(
        final_path,
        map_location="cpu",
        expected_sha256=args.expected_final_field_checkpoint_sha256,
    )
    if _canonical_architecture(initial_payload.get("architecture", {})) != (
        _canonical_architecture(final_payload.get("architecture", {}))
    ) or not _same_state_structure(initial_field, final_field):
        raise ValueError("Field-C changed the canonical field architecture")
    expected_geometry = raw.get("geometry_fingerprint", {})
    for label, field, payload in (
        ("initial", initial_field, initial_payload),
        ("final", final_field, final_payload),
    ):
        if payload.get("geometry_fingerprint") != expected_geometry:
            raise ValueError(f"{label} Field-C geometry differs")
        if any(
            payload.get(key) is not False
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
            if key in payload or label == "final"
        ):
            raise ValueError(f"{label} Field-C field is task contaminated")
        if not _all_field_tensors_finite(field):
            raise ValueError(f"{label} Field-C field contains non-finite state")

    # The trainer refreshes this fixed input buffer before the first update.
    # Apply the identical Field-C uncertainty to the initial control so that
    # the gate measures learned-state changes rather than a buffer mismatch.
    with torch.no_grad():
        initial_field.reliability.copy_(consensus.reliability)
    if not torch.equal(final_field.reliability.cpu(), consensus.reliability):
        raise ValueError("final Field-C reliability differs from the registered MPR")

    contract = dict(registration["training_contract"])
    training = dict(final_payload.get("training_config", {}))
    loss = dict(final_payload.get("loss_config", {}))
    expected_training = {
        "epochs": int(contract["epochs"]),
        "min_epochs": int(contract["min_epochs"]),
        "batch_size": int(contract["batch_size"]),
        "eval_batch_size": int(contract["eval_batch_size"]),
        "learning_rate": float(contract["learning_rate"]),
        "weight_decay": float(contract["weight_decay"]),
        "validation_fraction": float(contract["validation_fraction"]),
        "seed": int(contract["seed"]),
        "observation_contract": "compatible-legacy",
        "capability_target_contract": CAPABILITY_TARGET_CONTRACT_FIELD_C,
    }
    mismatched_training = {
        key: [training.get(key), value]
        for key, value in expected_training.items()
        if training.get(key) != value
    }
    if mismatched_training:
        raise ValueError(f"Field-C training config differs: {mismatched_training}")
    expected_loss = {
        "mpr_weight": float(contract["loss_weights"]["raw_radio"]),
        "dino_weight": float(contract["loss_weights"]["dino_v3"]),
        "sam3_weight": float(contract["loss_weights"]["sam3"]),
        "relation_weight": float(contract["loss_weights"]["relation"]),
        "coefficient_weight": float(contract["loss_weights"]["coefficient"]),
        "basis_orthogonality_weight": float(
            contract["loss_weights"]["basis_orthogonality"]
        ),
    }
    if loss != expected_loss:
        raise ValueError("Field-C loss weights differ from registration")
    history = list(final_payload.get("history", []))
    if len(history) != int(contract["epochs"]) or [
        int(row.get("epoch", -1)) for row in history
    ] != list(range(1, int(contract["epochs"]) + 1)):
        raise ValueError("Field-C did not complete the fixed epoch schedule")
    if final_payload.get("capability_target_contract") != CAPABILITY_TARGET_CONTRACT_FIELD_C:
        raise ValueError("Field-C target contract differs")
    if final_payload.get("capability_reliability_policy") != "field_c_visibility_safe":
        raise ValueError("Field-C reliability policy differs")

    device = torch.device(args.device)
    official = FrozenRadioViews.from_radio_checkpoint(
        radio_path, expected_sha256=radio_sha
    ).to(device).eval()
    official.requires_grad_(False)
    rows = torch.where(consensus.valid)[0]
    metrics: dict[str, dict[str, float]] = {}
    for label, field in (("initial", initial_field), ("final", final_field)):
        field = field.to(device).eval()
        values = _reconstruction_metrics(field, consensus, rows, args.batch_size)
        values.update(
            _capability_reconstruction_metrics(
                field, official, targets, rows, args.batch_size
            )
        )
        metrics[label] = values
        field.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gate = evaluate_gate(metrics["initial"], metrics["final"], registration["label_free_gate"])
    stored = dict(final_payload.get("final_metrics", {}))
    stored.update(final_payload.get("final_capability_metrics", {}))
    for key, value in metrics["final"].items():
        if abs(float(stored.get(key, float("nan"))) - float(value)) > 2e-5:
            raise ValueError(f"stored/recomputed Field-C metric differs: {key}")

    receipt: dict[str, object] = {
        "schema_version": "canonical_field_c_training_gate_receipt_v1",
        "experiment_registration": {
            "path": str(registration_path),
            "sha256": registration_sha,
        },
        "initial_field": {"path": str(initial_path), "sha256": initial_sha},
        "final_field": {
            "path": str(final_path),
            "sha256": args.expected_final_field_checkpoint_sha256,
            "history_epochs": len(history),
            "architecture_unchanged": True,
            "all_state_finite": True,
        },
        "raw_mpr": {"path": str(raw_loaded_path), "sha256": loaded_raw_sha},
        "capability_targets": target_provenance,
        "feature_output_bundle_sha256": expected_bundle,
        "fixed_training_config": expected_training,
        "loss_config": loss,
        "gate": gate,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "downstream_evaluation_started": False,
        "source_hashes": {
            "gate": sha256_file(Path(__file__).resolve()),
            "trainer": sha256_file(source_paths["train_canonical_radio_field.py"]),
            "loss": sha256_file(source_paths["canonical_field_losses.py"]),
        },
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, receipt)
    return {**receipt, "output": str(output), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-registration", required=True)
    parser.add_argument("--expected-experiment-registration-sha256", required=True)
    parser.add_argument("--final-field-checkpoint", required=True)
    parser.add_argument("--expected-final-field-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
