#!/usr/bin/env python3
"""Complete uncovered canonical-field rows without changing primary MPR rows.

The base field remains authoritative.  A query-free fused MPR cache supplies
targets only for rows that were invalid in the base MPR.  All shared modules
and all primary local codes are frozen; only fallback local-code rows receive
gradients.  This preserves the precise dominant-registration field while
adding coverage in the same compact representation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews, sha256_file
from radio_gs.scripts.train_canonical_radio_field import (
    _consensus_from_cache,
    _reconstruction_metrics,
)
from radio_gs.training.canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
)


def support_completion_rows(
    base_valid: torch.Tensor,
    completed_valid: torch.Tensor,
) -> torch.Tensor:
    base = torch.as_tensor(base_valid).bool().cpu()
    completed = torch.as_tensor(completed_valid).bool().cpu()
    if base.shape != completed.shape:
        raise ValueError("base and completed validity masks must align")
    if bool((base & ~completed).any()):
        raise ValueError("completed MPR cache cannot drop primary rows")
    return torch.where(completed & ~base)[0]


def _xyz_digest(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _query_free_metadata(cache: dict, label: str) -> dict:
    metadata = dict(cache.get("metadata", {}))
    contaminated = [
        key
        for key in (
            "benchmark_images_opened",
            "benchmark_masks_opened",
            "text_queries_opened",
        )
        if bool(metadata.get(key, False))
    ]
    if contaminated:
        raise ValueError(f"{label} cache is benchmark-contaminated: {contaminated}")
    return metadata


@torch.no_grad()
def _feature_rows(field, rows: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    device = field.local_codes.device
    for start in range(0, rows.numel(), int(batch_size)):
        parts.append(
            field.radio_features(rows[start : start + int(batch_size)].to(device))
            .float()
            .cpu()
        )
    return torch.cat(parts) if parts else torch.empty(0, field.decoder.feature_dim)


@torch.no_grad()
def initialize_completion_local_codes(
    field,
    target_features: torch.Tensor,
    rows: torch.Tensor,
) -> str:
    """Initialize new rows through the frozen affine path of the base field.

    Compact fields use a 128-D local code followed by a learned residual
    fusion into the affine RADIO coefficient.  Randomly initialized missing
    rows take hundreds of optimizer steps merely to enter the trained code
    manifold.  The base projection provides a query-free least-squares inverse
    that is valid for any scene and leaves every authoritative row untouched.
    """

    selected = torch.as_tensor(rows, device=field.local_codes.device).long()
    target = torch.as_tensor(
        target_features, device=field.local_codes.device, dtype=torch.float32
    )
    if target.shape != (selected.numel(), field.decoder.feature_dim):
        raise ValueError("completion targets must align with selected rows")
    coefficients = field.decoder.encode(target)
    if field.fusion is None:
        if coefficients.shape[1] != field.local_codes.shape[1]:
            raise RuntimeError("direct field local/coefficient dimensions differ")
        local = coefficients
        mode = "exact_affine_encode"
    elif field.fusion.base_projection is None:
        local = coefficients
        mode = "identity_base_projection"
    else:
        projection = field.fusion.base_projection
        weight = projection.weight.detach().float()
        bias = projection.bias.detach().float()
        # For y = x W^T + b, x = (y-b) pinv(W^T).  The pseudoinverse is
        # shared and computed once; no labels, queries, or neighboring rows
        # participate in this initialization.
        local = (coefficients.float() - bias) @ torch.linalg.pinv(
            weight.transpose(0, 1)
        )
        mode = "least_squares_base_projection_inverse"
    field.local_codes.index_copy_(0, selected, local.to(field.local_codes))
    return mode


def complete(args: argparse.Namespace) -> dict:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    base_checkpoint = Path(args.base_field_checkpoint)
    field, base_payload = load_canonical_field_checkpoint(
        base_checkpoint, map_location="cpu"
    )
    completed_cache_path = Path(args.completed_mpr_cache)
    completed_cache = torch.load(completed_cache_path, map_location="cpu")
    if not isinstance(completed_cache, dict) or "features" not in completed_cache:
        raise ValueError("completed MPR cache lacks primitive features")
    completed_metadata = _query_free_metadata(completed_cache, "completed MPR")
    consensus = _consensus_from_cache(completed_cache)
    xyz = torch.as_tensor(completed_cache["xyz"]).float().cpu()
    expected_xyz = str(
        base_payload.get("geometry_fingerprint", {}).get("xyz_sha256", "")
    )
    if expected_xyz != _xyz_digest(xyz) or xyz.shape[0] != field.num_gaussians:
        raise ValueError("base field and completed MPR geometry do not align")

    base_mpr_path = Path(base_payload["mpr_cache"])
    base_mpr = torch.load(base_mpr_path, map_location="cpu")
    _query_free_metadata(base_mpr, "base MPR")
    base_valid = torch.as_tensor(base_mpr["valid"]).bool().cpu()
    fallback_rows = support_completion_rows(base_valid, consensus.valid)
    if fallback_rows.numel() == 0:
        raise ValueError("completed MPR cache adds no fallback rows")
    primary_rows = torch.where(base_valid)[0]

    resume_path = Path(args.resume_field_checkpoint) if args.resume_field_checkpoint else None
    previous_history: list[dict[str, float]] = []
    resumed_epochs = 0
    if resume_path is not None:
        resume_field, resume_payload = load_canonical_field_checkpoint(
            resume_path, map_location="cpu"
        )
        resume_completion = dict(resume_payload.get("support_completion", {}))
        if resume_completion.get("base_field_checkpoint_sha256") != sha256_file(
            base_checkpoint
        ):
            raise ValueError("resume field was not completed from this base field")
        if resume_completion.get("completed_mpr_cache_sha256") != sha256_file(
            completed_cache_path
        ):
            raise ValueError("resume field used a different completed MPR cache")
        if resume_field.local_codes.shape != field.local_codes.shape:
            raise ValueError("resume local-code table does not align with base field")
        with torch.no_grad():
            # Advanced indexing returns a temporary tensor, so ``rows.copy_``
            # would silently leave the parameter untouched.  index_copy_ is
            # the in-place row update required for a real resume.
            field.local_codes.index_copy_(
                0,
                fallback_rows,
                resume_field.local_codes[fallback_rows],
            )
        previous_history = list(resume_payload.get("history", []))
        resumed_epochs = len(previous_history)
        initialization_mode = "resume_fallback_local_codes"
    else:
        initialization_mode = initialize_completion_local_codes(
            field,
            consensus.targets[fallback_rows],
            fallback_rows,
        )

    old_reliability = field.reliability.detach().float().cpu()
    new_reliability = consensus.reliability.float().cpu()
    if old_reliability.shape != new_reliability.shape:
        raise ValueError("base and completed reliability tensors do not align")
    if not torch.equal(old_reliability[primary_rows], new_reliability[primary_rows]):
        raise ValueError("completed MPR changed primary-row reliability")

    field = field.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    audit_count = min(int(args.primary_audit_rows), int(primary_rows.numel()))
    audit_rows = primary_rows[
        torch.randperm(primary_rows.numel(), generator=generator)[:audit_count]
    ]
    primary_before = _feature_rows(field, audit_rows, int(args.eval_batch_size))
    with torch.no_grad():
        field.reliability.copy_(new_reliability.to(device))
    primary_after_reliability = _feature_rows(
        field, audit_rows, int(args.eval_batch_size)
    )
    initial_primary_error = float(
        (primary_before - primary_after_reliability).abs().max()
    )
    if initial_primary_error != 0.0:
        raise RuntimeError(
            "primary predictions changed when installing completion reliability"
        )

    field.requires_grad_(False)
    field.local_codes.requires_grad_(True)
    official_views = FrozenRadioViews.from_radio_checkpoint(
        args.radio_checkpoint
    ).to(device).eval()
    official_views.requires_grad_(False)
    loss_config = CanonicalFieldLossConfig(
        mpr_weight=float(args.mpr_weight),
        dino_weight=float(args.dino_weight),
        sam3_weight=float(args.sam3_weight),
        relation_weight=0.0,
        coefficient_weight=float(args.coefficient_weight),
        # The shared basis is frozen, so this term is constant and omitted.
        basis_orthogonality_weight=0.0,
    )
    order = fallback_rows[
        torch.randperm(fallback_rows.numel(), generator=generator)
    ]
    probe_count = max(
        1, int(round(order.numel() * float(args.validation_fraction)))
    )
    probe_rows = order[:probe_count]
    # Local codes are row-specific parameters.  A held-out row can never be
    # reconstructed by training other rows, so every fallback row must receive
    # gradients.  ``probe_rows`` is a fixed training-reconstruction monitor,
    # not a benchmark validation split or a checkpoint-selection oracle.
    training_rows = order
    # Weight decay on one dense embedding parameter would also modify rows
    # with zero gradient.  Keep it exactly zero to guarantee primary invariance.
    optimizer = torch.optim.AdamW(
        [field.local_codes], lr=float(args.learning_rate), weight_decay=0.0
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        epoch_rows = training_rows[
            torch.randperm(training_rows.numel(), generator=generator)
        ]
        totals: list[float] = []
        field.train()
        for start in range(0, epoch_rows.numel(), int(args.batch_size)):
            rows = epoch_rows[start : start + int(args.batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _stats = canonical_primitive_loss(
                field,
                consensus,
                rows,
                official_views=official_views,
                config=loss_config,
            )
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
        field.eval()
        probe = _reconstruction_metrics(
            field, consensus, probe_rows, int(args.eval_batch_size)
        )
        record = {
            "epoch": epoch + 1,
            "loss": sum(totals) / max(1, len(totals)),
            "training_probe_mean_cosine": probe["mean_cosine"],
            "training_probe_p05_cosine": probe["p05_cosine"],
            "training_probe_mean_rmse": probe["mean_rmse"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if probe["mean_cosine"] >= float(args.target_cosine):
            break

    field.eval()
    fallback_metrics = _reconstruction_metrics(
        field, consensus, fallback_rows, int(args.eval_batch_size)
    )
    primary_after = _feature_rows(field, audit_rows, int(args.eval_batch_size))
    primary_max_abs_error = float((primary_before - primary_after).abs().max())
    if primary_max_abs_error != 0.0:
        raise RuntimeError(
            f"support completion changed primary rows: {primary_max_abs_error:.3e}"
        )

    field.cpu().eval()
    completion = {
        "construction": "frozen_primary_field_with_fallback_local_code_completion",
        "base_field_checkpoint": str(base_checkpoint.resolve()),
        "base_field_checkpoint_sha256": sha256_file(base_checkpoint),
        "base_mpr_cache": str(base_mpr_path.resolve()),
        "completed_mpr_cache": str(completed_cache_path.resolve()),
        "completed_mpr_cache_sha256": sha256_file(completed_cache_path),
        "primary_rows": int(primary_rows.numel()),
        "fallback_rows": int(fallback_rows.numel()),
        "primary_audit_rows": int(audit_rows.numel()),
        "primary_max_abs_error": primary_max_abs_error,
        "shared_modules_frozen": True,
        "primary_local_codes_frozen_by_zero_gradient": True,
        "optimizer_weight_decay": 0.0,
        "resume_field_checkpoint": (
            str(resume_path.resolve()) if resume_path is not None else None
        ),
        "resumed_epochs": resumed_epochs,
        "fallback_initialization": initialization_mode,
        "probe_is_training_rows": True,
        "loss_config": asdict(loss_config),
    }
    payload = {
        **base_payload,
        "state_dict": field.state_dict(),
        "reliability": consensus.reliability.half(),
        "mpr_cache": str(completed_cache_path.resolve()),
        "mpr_cache_metadata": completed_metadata,
        "history": previous_history + history,
        "final_metrics": fallback_metrics,
        "support_completion": completion,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "output": str(output),
        **completion,
        "fallback_metrics": fallback_metrics,
        "epochs_completed": len(history),
        "total_epochs_completed": resumed_epochs + len(history),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-field-checkpoint", required=True)
    parser.add_argument("--completed-mpr-cache", required=True)
    parser.add_argument(
        "--resume-field-checkpoint",
        default="",
        help=(
            "Optional earlier support-completion field. Only its fallback "
            "local codes are resumed; primary rows and shared modules still "
            "come from the authoritative base field."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--target-cosine", type=float, default=0.98)
    parser.add_argument("--mpr-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=0.20)
    parser.add_argument("--sam3-weight", type=float, default=0.20)
    parser.add_argument("--coefficient-weight", type=float, default=1e-5)
    parser.add_argument("--primary-audit-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(complete(args), indent=2))


if __name__ == "__main__":
    main()
