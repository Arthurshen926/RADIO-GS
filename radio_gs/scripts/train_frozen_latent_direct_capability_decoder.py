#!/usr/bin/env python3
"""Restore a pre-MPR nonlinear capability statistic from frozen L512.

This source-only stage does not add per-Gaussian parameters.  It trains one
scene-global residual decoder from the frozen L512 field latent to a direct
per-view-head-then-exact-MPR SigLIP descriptor.  A fixed row holdout gates the
decoder against the already deployed D512 typed descriptor.  Benchmark masks,
labels, evaluation RGB, and benchmark text queries are never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.five_benchmark_method_v1 import METHOD_ID
from radio_gs.models.frozen_latent_capability_decoder import (
    FrozenLatentCapabilityDecoder,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_sha_bound_project_checkpoint_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


def cosine_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = (F.normalize(prediction.float(), dim=-1) * F.normalize(target.float(), dim=-1)).sum(-1)
    return {
        "mean_cosine": float(values.mean()),
        "p05_cosine": float(torch.quantile(values, 0.05)),
    }


def _decode_all(
    model: FrozenLatentCapabilityDecoder,
    latent: torch.Tensor,
    baseline: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, latent.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), latent.shape[0])
            parts.append(
                model(
                    latent[start:stop].to(device), baseline[start:stop].to(device)
                ).half().cpu()
            )
    return torch.cat(parts, dim=0).contiguous()


def train_and_materialize(args: argparse.Namespace) -> dict[str, object]:
    target, target_sha, target_path = load_sha_bound_project_checkpoint_mapping(
        args.direct_capability_target,
        expected_sha256=args.expected_direct_capability_target_sha256,
        map_location="cpu",
        label="direct pre-MPR-head capability target",
    )
    baseline_payload, baseline_sha, baseline_path = (
        load_sha_bound_project_checkpoint_mapping(
            args.baseline_query_cache,
            expected_sha256=args.expected_baseline_query_cache_sha256,
            map_location="cpu",
            label="deployed D512 typed descriptor baseline",
        )
    )
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        args.field,
        map_location="cpu",
        expected_sha256=args.expected_field_sha256,
    )
    latent = field.local_codes.detach().cpu().float().contiguous()
    xyz = torch.as_tensor(target.get("xyz")).detach().cpu().float().contiguous()
    target_features = torch.as_tensor(target.get("features")).detach().cpu().float()
    target_valid = torch.as_tensor(target.get("valid")).detach().cpu().bool().reshape(-1)
    baseline_xyz = torch.as_tensor(baseline_payload.get("xyz")).detach().cpu().float()
    baseline = torch.as_tensor(
        baseline_payload.get("summary_features", baseline_payload.get("features"))
    ).detach().cpu().float()
    if (
        latent.shape != (xyz.shape[0], 512)
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or target_features.shape != (xyz.shape[0], 1536)
        or baseline.shape != target_features.shape
        or target_valid.shape != (xyz.shape[0],)
        or not torch.equal(xyz, baseline_xyz)
    ):
        raise ValueError("field, target, and baseline row domains differ")
    if not bool(target_valid.any()):
        raise ValueError("direct capability target has no observed row")
    target_features = F.normalize(target_features, dim=-1, eps=1e-8)
    baseline = F.normalize(baseline, dim=-1, eps=1e-8)

    row_ids = torch.arange(xyz.shape[0])
    validation = target_valid & (
        torch.remainder(row_ids, int(args.holdout_stride)) == int(args.holdout_residue)
    )
    training = target_valid & ~validation
    if int(validation.sum()) < 128 or int(training.sum()) < 1024:
        raise ValueError("fixed source-row split is too small")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    model = FrozenLatentCapabilityDecoder(
        latent_dim=512,
        hidden_dim=int(args.hidden_dim),
        capability_dim=1536,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_indices = torch.where(training)[0]
    validation_indices = torch.where(validation)[0]
    generator = torch.Generator().manual_seed(int(args.seed))
    baseline_validation = cosine_metrics(
        baseline[validation_indices], target_features[validation_indices]
    )
    best_metrics = dict(baseline_validation)
    best_step = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    history: list[dict[str, float | int]] = []
    for step in range(1, int(args.steps) + 1):
        sampled = train_indices[
            torch.randint(
                train_indices.numel(),
                (min(int(args.batch_size), train_indices.numel()),),
                generator=generator,
            )
        ]
        prediction = model(
            latent[sampled].to(device), baseline[sampled].to(device)
        )
        teacher = target_features[sampled].to(device)
        cosine_loss = 1.0 - (prediction * teacher).sum(-1).mean()
        coordinate_loss = F.smooth_l1_loss(prediction, teacher)
        loss = cosine_loss + float(args.coordinate_weight) * coordinate_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % int(args.validation_interval) != 0 and step != int(args.steps):
            continue
        candidate = _decode_all(
            model,
            latent[validation_indices],
            baseline[validation_indices],
            device=device,
            chunk_size=int(args.eval_chunk_size),
        ).float()
        metrics = cosine_metrics(candidate, target_features[validation_indices])
        history.append({"step": step, "loss": float(loss.detach()), **metrics})
        if (
            metrics["mean_cosine"] > best_metrics["mean_cosine"]
            and metrics["p05_cosine"]
            >= baseline_validation["p05_cosine"] - float(args.maximum_p05_drop)
        ):
            best_metrics = metrics
            best_step = step
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state, strict=True)
    restored = _decode_all(
        model,
        latent,
        baseline,
        device=device,
        chunk_size=int(args.eval_chunk_size),
    )
    # Rows never observed by the direct source teacher retain the deployed
    # descriptor exactly; the candidate therefore has the same total row domain.
    restored[~target_valid] = baseline[~target_valid].half()
    final_validation = cosine_metrics(
        restored[validation_indices].float(), target_features[validation_indices]
    )
    passed = (
        best_step > 0
        and final_validation["mean_cosine"]
        > baseline_validation["mean_cosine"] + float(args.minimum_mean_gain)
        and final_validation["p05_cosine"]
        >= baseline_validation["p05_cosine"] - float(args.maximum_p05_drop)
    )

    input_records = {
        "field": file_record(Path(args.field).expanduser().resolve(strict=True)),
        "direct_capability_target": {"path": str(target_path), "sha256": target_sha},
        "baseline_query_cache": {"path": str(baseline_path), "sha256": baseline_sha},
    }
    model_path = Path(args.output_model).expanduser().resolve()
    model_payload = {
        "schema": "radio_gs.frozen_l512_direct_capability_decoder.v1",
        "schema_version": 1,
        "scene": str(args.scene),
        "architecture": {
            "latent_dim": 512,
            "hidden_dim": int(args.hidden_dim),
            "capability_dim": 1536,
            "skip": "deployed_d512_typed_descriptor",
        },
        "state_dict": best_state,
        "metadata": {
            "query_independent": True,
            "field_frozen": True,
            "per_gaussian_parameters_added": False,
            "teacher_order": str(args.teacher_order),
            "view_split": "source_only_teacher_then_fixed_gaussian_row_holdout",
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "evaluation_rgb_opened": False,
            "text_queries_opened": False,
            "inputs": input_records,
        },
    }
    write_torch_noclobber(model_path, model_payload)
    cache_path = Path(args.output_query_cache).expanduser().resolve()
    cache_payload = {
        "xyz": xyz,
        "summary_features": restored,
        "features": restored,
        "valid": torch.ones(xyz.shape[0], dtype=torch.bool),
        "direct_observed": target_valid,
        "metadata": {
            "schema_version": 1,
            "artifact_type": "radio_gs_method_v1_primitive_query_cache",
            "method_id": METHOD_ID,
            "feature_space": "global_decoder_restored_direct_siglip_descriptor",
            "construction": (
                "frozen_l512_scene_global_residual_decoder_of_"
                "per_view_summary_head_then_exact_mpr"
            ),
            "query_independent": True,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "text_queries_opened": False,
            "postprocessing": "none",
            "decoder": {"path": str(model_path), "sha256": file_record(model_path)["sha256"]},
            "inputs": input_records,
            "source_gate_passed": passed,
        },
    }
    write_torch_noclobber(cache_path, cache_payload)
    report = {
        "status": "source_gate_pass" if passed else "source_gate_fail",
        "scene": str(args.scene),
        "training_rows": int(training.sum()),
        "validation_rows": int(validation.sum()),
        "unobserved_fallback_rows": int((~target_valid).sum()),
        "baseline_validation": baseline_validation,
        "decoder_validation": final_validation,
        "best_step": best_step,
        "history": history,
        "model": file_record(model_path),
        "query_cache": file_record(cache_path),
    }
    write_frozen_json(cache_path.with_suffix(cache_path.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--direct-capability-target", required=True)
    parser.add_argument("--expected-direct-capability-target-sha256", required=True)
    parser.add_argument("--baseline-query-cache", required=True)
    parser.add_argument("--expected-baseline-query-cache-sha256", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-query-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coordinate-weight", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--eval-chunk-size", type=int, default=8192)
    parser.add_argument("--holdout-stride", type=int, default=8)
    parser.add_argument("--holdout-residue", type=int, default=7)
    parser.add_argument("--minimum-mean-gain", type=float, default=1e-4)
    parser.add_argument("--maximum-p05-drop", type=float, default=0.002)
    parser.add_argument(
        "--teacher-order",
        default="per_view_summary_head_then_exact_mpr",
        help="Hash-recorded source-teacher construction order for this decoder.",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    print(json.dumps(train_and_materialize(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
