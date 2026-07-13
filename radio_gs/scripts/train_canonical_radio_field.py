#!/usr/bin/env python3
"""Train one compact, query-independent canonical RADIO field from MPR targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.field import CanonicalGaussianField, FeatureSpaceSignature, fit_affine_basis
from radio_gs.interfaces.frozen_radio_views import FrozenRadioViews, sha256_file
from radio_gs.training.canonical_field_losses import (
    CanonicalFieldLossConfig,
    canonical_primitive_loss,
)
from radio_gs.training.primitive_consensus import PrimitiveConsensus


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _consensus_from_cache(cache: dict) -> PrimitiveConsensus:
    targets = torch.as_tensor(cache["features"]).float().cpu()
    valid = torch.as_tensor(cache["valid"]).bool().cpu()
    counts = torch.as_tensor(cache["view_counts"]).long().cpu()
    reliability = cache.get("reliability")
    if reliability is None:
        maximum = max(1, int(counts.max()) if counts.numel() else 1)
        reliability = torch.stack(
            [counts.float() / maximum, valid.float(), valid.float()], dim=-1
        )
    else:
        reliability = torch.as_tensor(reliability).float().cpu()
    return PrimitiveConsensus(
        targets=targets,
        valid=valid,
        observation_count=counts,
        reliability=reliability,
        per_view_agreement=torch.empty(0, targets.shape[0]),
    )


@torch.no_grad()
def _reconstruction_metrics(
    field: CanonicalGaussianField,
    consensus: PrimitiveConsensus,
    rows: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    cosines: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []
    device = field.local_codes.device
    for start in range(0, rows.numel(), batch_size):
        batch = rows[start : start + batch_size]
        predicted = field.radio_features(batch.to(device)).float().cpu()
        target = consensus.targets[batch].float()
        cosines.append(F.cosine_similarity(predicted, target, dim=-1, eps=1e-8))
        rmses.append((predicted - target).square().mean(dim=-1).sqrt())
    cosine = torch.cat(cosines)
    rmse = torch.cat(rmses)
    return {
        "mean_cosine": float(cosine.mean()),
        "p05_cosine": float(cosine.quantile(0.05)),
        "mean_rmse": float(rmse.mean()),
    }


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    cache = torch.load(Path(args.mpr_cache), map_location="cpu")
    if not isinstance(cache, dict) or "features" not in cache:
        raise ValueError("MPR cache must contain primitive features")
    metadata = dict(cache.get("metadata", {}))
    if metadata.get("benchmark_masks_opened", False) or metadata.get("text_queries_opened", False):
        raise ValueError("MPR training cache is contaminated by benchmark queries or masks")
    if str(metadata.get("feature_space", "radio")) != "radio":
        raise ValueError("canonical main field must reconstruct raw RADIO, not a query head")
    consensus = _consensus_from_cache(cache)
    valid_rows = torch.where(consensus.valid)[0]
    if valid_rows.numel() < int(args.coefficient_dim):
        raise ValueError("too few valid primitive targets for the requested basis")

    decoder, fit_report = fit_affine_basis(
        consensus.targets[valid_rows],
        int(args.coefficient_dim),
        standardize=not args.no_standardize,
        max_samples=int(args.pca_samples),
        seed=int(args.seed),
        trainable_basis=not args.freeze_basis,
    )
    radio_hash = sha256_file(args.radio_checkpoint)
    signature = FeatureSpaceSignature(
        radio_version=args.radio_version,
        radio_checkpoint_sha256=radio_hash,
        raw_feature_dim=consensus.targets.shape[1],
        adaptor_name="backbone",
        token_type="primitive",
        normalization="none",
        crop_policy="training_views_depth_alpha_checked_mpr",
        # The canonical field stores raw RADIO only.  Semantic alignment is a
        # separately selected, frozen capability view and is never part of the
        # field checkpoint contract.
        semantic_alignment="none",
    )
    field = CanonicalGaussianField(
        num_gaussians=consensus.targets.shape[0],
        decoder=decoder,
        signature=signature,
        reliability=consensus.reliability,
        hidden_dim=int(args.hidden_dim),
        use_fusion=bool(args.primitive_fusion),
    ).to(device)
    with torch.no_grad():
        encoded = decoder.encode(consensus.targets.to(device))
        field.local_codes.copy_(encoded)

    official_views = None
    if args.official_capability_loss:
        official_views = FrozenRadioViews.from_radio_checkpoint(args.radio_checkpoint).to(device)
    loss_config = CanonicalFieldLossConfig(
        mpr_weight=float(args.mpr_weight),
        dino_weight=float(args.dino_weight if official_views is not None else 0.0),
        sam3_weight=float(args.sam3_weight if official_views is not None else 0.0),
        relation_weight=0.0,
        coefficient_weight=float(args.coefficient_weight),
        basis_orthogonality_weight=float(args.basis_orthogonality_weight),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    order = valid_rows[torch.randperm(valid_rows.numel(), generator=generator)]
    validation_count = max(1, int(round(order.numel() * float(args.validation_fraction))))
    validation_rows = order[:validation_count]
    training_rows = order[validation_count:]
    if training_rows.numel() == 0:
        training_rows = validation_rows
    optimizer = torch.optim.AdamW(
        [parameter for parameter in field.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(args.epochs)):
        epoch_order = training_rows[
            torch.randperm(training_rows.numel(), generator=generator)
        ]
        totals: list[float] = []
        field.train()
        for start in range(0, epoch_order.numel(), int(args.batch_size)):
            rows = epoch_order[start : start + int(args.batch_size)].to(device)
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
        validation = _reconstruction_metrics(
            field, consensus, validation_rows, int(args.eval_batch_size)
        )
        record = {
            "epoch": epoch + 1,
            "loss": sum(totals) / max(1, len(totals)),
            **validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["mean_cosine"] >= float(args.target_cosine):
            break

    field.eval().cpu()
    final_metrics = _reconstruction_metrics(
        field, consensus, valid_rows, int(args.eval_batch_size)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    architecture = {
        "num_gaussians": field.num_gaussians,
        "feature_dim": field.decoder.feature_dim,
        "coefficient_dim": field.decoder.coefficient_dim,
        "local_dim": field.local_codes.shape[1],
        "coarse_dim": field.coarse_dim,
        "hidden_dim": int(args.hidden_dim),
        "use_fusion": bool(args.primitive_fusion),
        "trainable_basis": not args.freeze_basis,
        "trainable_statistics": False,
    }
    payload = {
        "schema_version": 1,
        "architecture": architecture,
        "feature_signature": signature.to_dict(),
        "state_dict": field.state_dict(),
        "reliability": consensus.reliability.half(),
        "geometry_fingerprint": cache.get("geometry_fingerprint", {}),
        "mpr_cache": str(Path(args.mpr_cache).resolve()),
        "mpr_cache_metadata": metadata,
        "basis_fit_report": asdict(fit_report),
        "loss_config": asdict(loss_config),
        "history": history,
        "final_metrics": final_metrics,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    torch.save(payload, output)
    report = {
        "output": str(output),
        "num_gaussians": field.num_gaussians,
        "valid_gaussians": int(valid_rows.numel()),
        "coefficient_dim": field.decoder.coefficient_dim,
        "basis_fit": asdict(fit_report),
        "final_metrics": final_metrics,
        "feature_signature": signature.to_dict(),
        "xyz_sha256": _sha256_tensor_rows(torch.as_tensor(cache["xyz"])),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coefficient-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument(
        "--primitive-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional local/coarse/reliability residual fusion; stage-1 main is direct local coefficients.",
    )
    parser.add_argument("--pca-samples", type=int, default=50000)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--freeze-basis", action="store_true")
    parser.add_argument("--official-capability-loss", action="store_true")
    parser.add_argument("--mpr-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=0.20)
    parser.add_argument("--sam3-weight", type=float, default=0.20)
    parser.add_argument("--coefficient-weight", type=float, default=1e-5)
    parser.add_argument("--basis-orthogonality-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--target-cosine", type=float, default=0.985)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
