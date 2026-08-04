#!/usr/bin/env python3
"""Evaluate the preregistered Field-B representation gate without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews
from radio_gs.scripts.eval_field_a_label_free_gate import (
    _all_field_tensors_finite,
    _atomic_json,
    _canonical_architecture,
    _same_state_structure,
    _training_runtime_receipt,
)
from radio_gs.scripts.train_canonical_radio_field import (
    CAPABILITY_TARGET_CONTRACT_FIELD_A,
    FIELD_B_RELATION_WEIGHT,
    RELATION_OBJECTIVE_FIELD_B,
    _capability_reconstruction_metrics,
    _consensus_from_cache,
    _load_capability_mpr_target,
    _load_field_a_observation_reference,
    _load_field_b_relation_triplets,
    _reconstruction_metrics,
)
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


def evaluate_field_b_gate(
    initial: dict[str, float],
    final: dict[str, float],
) -> dict:
    deltas = {key: float(final[key] - initial[key]) for key in final}
    checks = {
        "raw_mean_cosine_no_regression_gt_0.005": (
            deltas["mean_cosine"] >= -0.005
        ),
        "raw_p05_cosine_no_regression_gt_0.010": (
            deltas["p05_cosine"] >= -0.010
        ),
        "dino_mean_cosine_no_regression_gt_0.005": (
            deltas["dino_v3_target_mean_cosine"] >= -0.005
        ),
        "dino_p05_cosine_no_regression_gt_0.010": (
            deltas["dino_v3_target_p05_cosine"] >= -0.010
        ),
        "sam3_mean_cosine_no_regression_gt_0.005": (
            deltas["sam3_target_mean_cosine"] >= -0.005
        ),
        "sam3_p05_cosine_no_regression_gt_0.010": (
            deltas["sam3_target_p05_cosine"] >= -0.010
        ),
        "relation_teacher_margin_hinge_improves": (
            deltas["relation_teacher_margin_hinge"] < 0.0
        ),
    }
    return {
        "initial": initial,
        "final": final,
        "delta": deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


@torch.inference_mode()
def _field_b_relation_metrics(
    field: torch.nn.Module,
    official_views: FrozenRadioViews,
    relation_cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("Field-B relation metric batch size must be positive")
    count = int(relation_cache["teacher_margin"].numel())
    all_pairs = relation_cache["pair_index"]
    all_margin = relation_cache["teacher_margin"]
    gaps: list[torch.Tensor] = []
    hinges: list[torch.Tensor] = []
    device = field.local_codes.device

    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        selected = torch.arange(start, stop)
        columns = torch.cat([selected, selected + count])
        global_pairs = all_pairs[:, columns]
        unique_rows, inverse = torch.unique(
            global_pairs.reshape(-1), sorted=True, return_inverse=True
        )
        local_pairs = inverse.reshape_as(global_pairs).to(device)
        predicted_radio = field.radio_features(unique_rows.to(device))
        dino = official_views.project_dino_primitives(predicted_radio)
        sam3 = official_views.project_sam3_primitives(predicted_radio)

        def relation(values: torch.Tensor) -> torch.Tensor:
            normalized = F.normalize(values.float(), dim=-1, eps=1e-8)
            cosine = (
                normalized[local_pairs[0]] * normalized[local_pairs[1]]
            ).sum(dim=-1)
            return (0.5 * (1.0 + cosine)).clamp(0.0, 1.0)

        dino_relation = relation(dino)
        sam3_relation = relation(sam3)
        combined = torch.sqrt(
            (dino_relation * sam3_relation).clamp_min(1e-12)
        )
        batch_count = stop - start
        gap = combined[:batch_count] - combined[batch_count:]
        margin = all_margin[start:stop].to(device).float()
        gaps.append(gap.cpu())
        hinges.append(F.relu(margin - gap).cpu())

    gap = torch.cat(gaps).float()
    hinge = torch.cat(hinges).float()
    margin = all_margin.float()
    return {
        "relation_teacher_margin_hinge": float(hinge.mean()),
        "relation_order_accuracy": float((gap > 0.0).float().mean()),
        "relation_margin_satisfaction": float((gap >= margin).float().mean()),
        "relation_mean_gap": float(gap.mean()),
        "relation_p05_gap": float(torch.quantile(gap, 0.05)),
    }


def run(args: argparse.Namespace) -> dict:
    registration, registration_sha, registration_path = load_json_object(
        args.experiment_registration,
        expected_sha256=args.expected_experiment_registration_sha256,
        label="Field-B experiment registration",
    )
    if registration.get("schema_version") != (
        "canonical_field_b_boundary_relation_registration_v1"
    ):
        raise ValueError("Field-B experiment registration schema differs")
    source_hashes = dict(registration.get("source_hashes", {}))
    current_source_hashes = {
        "label_free_gate": sha256_file(Path(__file__).resolve()),
        "trainer": sha256_file(
            Path(__file__).with_name("train_canonical_radio_field.py")
        ),
        "losses": sha256_file(
            Path(__file__).parents[1] / "training" / "canonical_field_losses.py"
        ),
    }
    if any(
        source_hashes.get(key) != value
        for key, value in current_source_hashes.items()
    ):
        raise ValueError("Field-B registered gate/training source SHA-256 differs")

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

    relation_cache, relation_provenance = _load_field_b_relation_triplets(
        args.relation_triplet_cache,
        expected_sha256=args.expected_relation_triplet_cache_sha256,
        num_rows=int(primary["xyz"].shape[0]),
        capability_valid=torch.as_tensor(targets["dino_v3"].valid).bool(),
        geometry_fingerprint=dict(primary.get("geometry_fingerprint", {})),
        expected_dino_sha256=args.expected_dino_v3_mpr_cache_sha256,
        expected_sam3_sha256=args.expected_sam3_mpr_cache_sha256,
    )
    if not torch.equal(
        torch.as_tensor(targets["dino_v3"].valid).bool(),
        torch.as_tensor(targets["sam3"].valid).bool(),
    ):
        raise ValueError("Field-B gate DINO/SAM valid rows differ")

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
    immutable_inputs = dict(registration.get("immutable_inputs", {}))
    expected_inputs = {
        "field_a_checkpoint": args.expected_initial_field_checkpoint_sha256,
        "primary_raw_mpr": primary_sha,
        "exact_observation_reference": (
            args.expected_capability_observation_reference_mpr_cache_sha256
        ),
        "exact_dino_v3": args.expected_dino_v3_mpr_cache_sha256,
        "exact_sam3": args.expected_sam3_mpr_cache_sha256,
        "official_c_radio_checkpoint": radio_sha,
    }
    if any(
        dict(immutable_inputs.get(key, {})).get("sha256") != value
        for key, value in expected_inputs.items()
    ):
        raise ValueError("Field-B gate immutable inputs differ from registration")
    if _canonical_architecture(initial_payload.get("architecture", {})) != (
        _canonical_architecture(final_payload.get("architecture", {}))
    ) or not _same_state_structure(initial_field, final_field):
        raise ValueError("Field-B changed the canonical field architecture")
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
        ):
            raise ValueError(f"{label} field is benchmark/query contaminated")
    if not _all_field_tensors_finite(initial_field) or not _all_field_tensors_finite(
        final_field
    ):
        raise ValueError("Field-B initial/final field contains non-finite state")

    expected_loss = {
        "mpr_weight": 1.0,
        "dino_weight": 0.2,
        "sam3_weight": 0.2,
        "relation_weight": FIELD_B_RELATION_WEIGHT,
        "coefficient_weight": 1e-5,
        "basis_orthogonality_weight": 1e-3,
    }
    loss = dict(final_payload.get("loss_config", {}))
    history = list(final_payload.get("history", []))
    training = dict(final_payload.get("training_config", {}))
    registered_training = dict(registration.get("training_contract", {}))
    fixed_training_keys = (
        "epochs",
        "min_epochs",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "validation_fraction",
        "seed",
    )
    if loss != expected_loss:
        raise ValueError("Field-B loss weights differ from registration")
    if any(training.get(key) != registered_training.get(key) for key in fixed_training_keys):
        raise ValueError("Field-B fixed training config differs from registration")
    if len(history) != 20 or [int(row.get("epoch", -1)) for row in history] != list(
        range(1, 21)
    ):
        raise ValueError("Field-B did not complete the fixed 20 epochs")
    if final_payload.get("relation_objective") != RELATION_OBJECTIVE_FIELD_B:
        raise ValueError("Field-B relation objective differs")
    stored_relation = dict(final_payload.get("relation_triplet_cache", {}))
    if stored_relation.get("sha256") != args.expected_relation_triplet_cache_sha256:
        raise ValueError("Field-B trained relation cache SHA-256 differs")
    stored_registration = dict(
        final_payload.get("field_b_experiment_registration", {})
    )
    if stored_registration.get("sha256") != registration_sha:
        raise ValueError("Field-B trained registration SHA-256 differs")

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
            field, consensus, rows, int(args.batch_size)
        )
        values.update(
            _capability_reconstruction_metrics(
                field,
                official,
                targets,
                rows,
                int(args.batch_size),
            )
        )
        values.update(
            _field_b_relation_metrics(
                field,
                official,
                relation_cache,
                batch_size=int(args.relation_batch_size),
            )
        )
        metrics[label] = values
        field.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gate = evaluate_field_b_gate(metrics["initial"], metrics["final"])
    runtime = _training_runtime_receipt(
        args.training_telemetry,
        args.training_owner_audit,
    )
    receipt = {
        "schema_version": "canonical_field_b_label_free_gate_receipt_v1",
        "experiment_registration": {
            "path": str(registration_path),
            "sha256": registration_sha,
        },
        "initial_field": {
            "path": str(Path(args.initial_field_checkpoint).resolve()),
            "sha256": args.expected_initial_field_checkpoint_sha256,
        },
        "final_field": {
            "path": str(Path(args.final_field_checkpoint).resolve()),
            "sha256": args.expected_final_field_checkpoint_sha256,
            "file_bytes": Path(args.final_field_checkpoint).stat().st_size,
            "architecture_unchanged": True,
            "all_state_finite": True,
            "history_epochs": len(history),
            "training_config_sha256": final_payload.get("training_config_sha256"),
        },
        "primary_mpr": {"path": str(primary_path), "sha256": primary_sha},
        "capability_observation_reference": observation_provenance,
        "capability_targets": target_provenance,
        "relation_triplets": relation_provenance,
        "loss_config": loss,
        "training_runtime": runtime,
        "gate": gate,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "downstream_evaluation_started": False,
        "source_hashes": current_source_hashes,
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Field-B gate output exists: {output}")
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
    parser.add_argument("--relation-triplet-cache", required=True)
    parser.add_argument("--expected-relation-triplet-cache-sha256", required=True)
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
    parser.add_argument("--relation-batch-size", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
