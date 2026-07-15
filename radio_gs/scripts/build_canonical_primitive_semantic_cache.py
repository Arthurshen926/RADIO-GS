#!/usr/bin/env python3
"""Build primitive-first text-space descriptors from one canonical RADIO field.

Each descriptor is computed from density-adaptive 3-D neighbourhoods, the
frozen global region-to-summary bridge, and the frozen official SigLIP2
summary head.  No image, query string, benchmark mask, or evaluation frame is
opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.semantic_alignment import GlobalRegionSummaryBridge
from radio_gs.models.siglip_projection import SigLIP2SummaryHead


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(value) for value in raw.replace(",", " ").split()}))
    if not sizes or sizes[0] <= 0:
        raise ValueError("neighborhood sizes must be positive")
    return sizes


def query_region_neighbor_rows(
    xyz: torch.Tensor,
    valid_rows: torch.Tensor,
    maximum: int,
    *,
    domain: str,
    workers: int = -1,
) -> np.ndarray:
    """Return global Gaussian rows for fixed-cardinality region summaries.

    ``valid`` means each requested token is an observed MPR teacher row.
    Legacy ``all`` queries geometry first and may later mask most of the k
    tokens, making the effective region cardinality depend on scene coverage.
    """
    from scipy.spatial import cKDTree

    points = torch.as_tensor(xyz).float().cpu()
    rows = torch.as_tensor(valid_rows).long().cpu()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must be [N,3]")
    if rows.ndim != 1 or rows.numel() == 0:
        raise ValueError("valid_rows must contain at least one row")
    if maximum <= 0 or domain not in {"all", "valid"}:
        raise ValueError("maximum must be positive and domain all/valid")
    if domain == "valid":
        candidate_rows = rows
    else:
        candidate_rows = torch.arange(points.shape[0], dtype=torch.long)
    count = min(int(maximum), int(candidate_rows.numel()))
    _distances, local_neighbors = cKDTree(
        points[candidate_rows].numpy()
    ).query(points[rows].numpy(), k=count, workers=int(workers))
    local_neighbors = np.asarray(local_neighbors, dtype=np.int64)
    if local_neighbors.ndim == 1:
        local_neighbors = local_neighbors[:, None]
    return candidate_rows.numpy()[local_neighbors]


def build_region_token_mask(
    neighbor_rows: torch.Tensor,
    valid: torch.Tensor,
    center_rows: torch.Tensor,
    *,
    policy: str,
    primary_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve query-free token support for one region-summary batch."""
    neighbors = torch.as_tensor(neighbor_rows).long()
    centers = torch.as_tensor(center_rows).long().reshape(-1)
    observed = torch.as_tensor(valid).bool()
    if neighbors.ndim != 2 or centers.shape != (neighbors.shape[0],):
        raise ValueError("neighbor rows and center rows must be [B,K]/[B]")
    if policy == "all_valid":
        return observed[neighbors]
    if policy != "primary_plus_center":
        raise ValueError(f"unsupported region token policy: {policy}")
    if primary_valid is None:
        raise ValueError("primary_plus_center requires primary_valid")
    primary = torch.as_tensor(primary_valid).bool()
    if primary.shape != observed.shape:
        raise ValueError("primary_valid must align with valid")
    if bool((primary & ~observed).any()):
        raise ValueError("primary rows must be a subset of valid rows")
    # Primary rows see exactly the old primary support.  A fallback row adds
    # only its own observation; other fallback rows cannot contaminate the
    # region context merely because coverage increased.
    return primary[neighbors] | (neighbors == centers[:, None])


def canonical_reconstruction_confidence(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    primary_valid: torch.Tensor,
    observation_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a query-independent confidence for completed primitive rows.

    Dominant-MPR rows belong to the frozen base field and therefore retain
    unit confidence.  Newly completed rows are weighted by how faithfully the
    frozen shared field can reconstruct their multiview RADIO teacher.  This
    uses no text, query, image annotation, or benchmark mask.
    """
    predicted = torch.as_tensor(predicted).float()
    teacher = torch.as_tensor(teacher).float()
    observed = torch.as_tensor(valid, device=predicted.device).bool()
    primary = torch.as_tensor(primary_valid, device=predicted.device).bool()
    if predicted.ndim != 2 or teacher.shape != predicted.shape:
        raise ValueError("predicted and teacher features must have equal [N,D] shape")
    if observed.shape != (predicted.shape[0],) or primary.shape != observed.shape:
        raise ValueError("valid and primary masks must align with feature rows")
    if bool((primary & ~observed).any()):
        raise ValueError("primary rows must be observed")
    confidence = torch.zeros(predicted.shape[0], device=predicted.device)
    confidence[primary] = 1.0
    fallback = observed & ~primary
    if bool(fallback.any()):
        fallback_confidence = F.cosine_similarity(
            predicted[fallback], teacher[fallback], dim=-1, eps=1e-8
        ).clamp_(0.0, 1.0)
        if observation_counts is not None:
            counts = torch.as_tensor(
                observation_counts,
                device=predicted.device,
                dtype=predicted.dtype,
            ).reshape(-1)
            if counts.shape != observed.shape or bool((counts < 0).any()):
                raise ValueError("observation counts must be non-negative and row-aligned")
            # One unit pseudo-count prevents a one-view adjoint assignment
            # from receiving the same trust as a true multiview consensus.
            evidence = counts[fallback] / (counts[fallback] + 1.0)
            fallback_confidence = fallback_confidence * evidence
        confidence[fallback] = fallback_confidence
    if not bool(torch.isfinite(confidence).all()):
        raise FloatingPointError("non-finite canonical reconstruction confidence")
    return confidence


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    field, field_payload = load_canonical_field_checkpoint(
        args.field_checkpoint, map_location="cpu"
    )
    cache_path = Path(args.mpr_cache or field_payload["mpr_cache"])
    mpr = torch.load(cache_path, map_location="cpu")
    xyz = torch.as_tensor(mpr["xyz"]).float().cpu()
    valid = torch.as_tensor(mpr["valid"]).bool().cpu()
    primary_valid = None
    if args.neighbor_token_policy == "primary_plus_center":
        reliability = torch.as_tensor(mpr.get("reliability")).float().cpu()
        if reliability.ndim != 2 or reliability.shape[0] != xyz.shape[0] or (
            reliability.shape[1] < 3
        ):
            raise ValueError(
                "primary_plus_center requires the fused MPR primary indicator"
            )
        primary_valid = valid & (reliability[:, 2] > 0.5)
        if not bool(primary_valid.any()) or torch.equal(primary_valid, valid):
            raise ValueError(
                "primary_plus_center requires distinct primary and fallback rows"
            )
    primary_semantic_payload = None
    primary_semantic_features = None
    primary_semantic_path = None
    if args.primary_semantic_cache:
        if primary_valid is None:
            raise ValueError(
                "--primary-semantic-cache requires primary_plus_center"
            )
        primary_semantic_path = Path(args.primary_semantic_cache)
        primary_semantic_payload = torch.load(
            primary_semantic_path, map_location="cpu"
        )
        primary_semantic_features = torch.as_tensor(
            primary_semantic_payload.get(
                "summary_features", primary_semantic_payload.get("features")
            )
        )
        cached_valid = torch.as_tensor(
            primary_semantic_payload.get("valid")
        ).bool()
        cached_xyz = torch.as_tensor(primary_semantic_payload.get("xyz")).float()
        if not torch.equal(cached_valid, primary_valid):
            raise ValueError(
                "primary semantic cache validity does not equal fused primary rows"
            )
        if cached_xyz.shape != xyz.shape or _sha256_tensor_rows(cached_xyz) != _sha256_tensor_rows(xyz):
            raise ValueError("primary semantic cache geometry does not align")
        cached_metadata = dict(primary_semantic_payload.get("metadata", {}))
        if any(
            bool(cached_metadata.get(key, False))
            for key in (
                "benchmark_images_opened",
                "benchmark_masks_opened",
                "text_queries_opened",
            )
        ):
            raise ValueError("primary semantic cache is benchmark-contaminated")
    expected_hash = str(field_payload.get("geometry_fingerprint", {}).get("xyz_sha256", ""))
    actual_hash = _sha256_tensor_rows(xyz)
    if expected_hash != actual_hash:
        raise ValueError("canonical field and MPR geometry rows differ")
    if field.num_gaussians != xyz.shape[0] or valid.shape != (xyz.shape[0],):
        raise ValueError("canonical field, geometry, and valid rows do not align")

    bridge, bridge_manifest = GlobalRegionSummaryBridge.from_checkpoint(
        args.bridge_checkpoint, map_location="cpu"
    )
    bridge = bridge.to(device).eval()
    summary_head = SigLIP2SummaryHead.from_radio_checkpoint(
        args.radio_checkpoint
    ).to(device).eval()
    if primary_semantic_payload is not None:
        cached_metadata = dict(primary_semantic_payload.get("metadata", {}))
        if cached_metadata.get("bridge_checkpoint_sha256") != bridge_manifest.checkpoint_sha256:
            raise ValueError("primary semantic cache uses a different semantic bridge")
    for module in (field, bridge, summary_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    field = field.to(device).eval()

    count = xyz.shape[0]
    radio = torch.empty(
        count, field.decoder.feature_dim, dtype=torch.float16, device=device
    )
    if args.radio_source == "mpr":
        mpr_radio = torch.as_tensor(mpr["features"])
        if mpr_radio.shape != (count, field.decoder.feature_dim):
            raise ValueError("MPR RADIO features do not align with the canonical field")
        for start in range(0, count, int(args.radio_batch_size)):
            stop = min(count, start + int(args.radio_batch_size))
            radio[start:stop] = mpr_radio[start:stop].to(device).half()
        del mpr_radio
    else:
        for start in range(0, count, int(args.radio_batch_size)):
            stop = min(count, start + int(args.radio_batch_size))
            rows = torch.arange(start, stop, device=device)
            radio[start:stop] = field.radio_features(rows).half()

    semantic_confidence = None
    confidence_report = None
    if primary_valid is not None:
        teacher_radio = torch.as_tensor(mpr["features"])
        observation_counts = torch.as_tensor(mpr["view_counts"])
        if teacher_radio.shape != radio.shape:
            raise ValueError("MPR RADIO teacher does not align with canonical predictions")
        semantic_confidence = torch.zeros(count, dtype=torch.float16)
        fallback_values: list[torch.Tensor] = []
        for start in range(0, count, int(args.radio_batch_size)):
            stop = min(count, start + int(args.radio_batch_size))
            confidence_chunk = canonical_reconstruction_confidence(
                radio[start:stop],
                teacher_radio[start:stop].to(device),
                valid[start:stop],
                primary_valid[start:stop],
                observation_counts[start:stop],
            )
            semantic_confidence[start:stop] = confidence_chunk.half().cpu()
            fallback_chunk = valid[start:stop] & ~primary_valid[start:stop]
            if bool(fallback_chunk.any()):
                fallback_values.append(confidence_chunk[fallback_chunk.to(device)].cpu())
        fallback_confidence = torch.cat(fallback_values)
        confidence_report = {
            "policy": (
                "primary_one_fallback_canonical_teacher_cosine_times_"
                "n_over_n_plus_one"
            ),
            "fallback_mean": float(fallback_confidence.mean()),
            "fallback_p05": float(torch.quantile(fallback_confidence, 0.05)),
            "fallback_p50": float(torch.quantile(fallback_confidence, 0.50)),
        }
    encoded = torch.empty(
        count, bridge.hidden_dim, dtype=torch.float16, device=device
    )
    attention_logits = torch.empty(count, dtype=torch.float16, device=device)
    for start in range(0, count, int(args.radio_batch_size)):
        stop = min(count, start + int(args.radio_batch_size))
        token_hidden, token_logits = bridge.encode_region_tokens(radio[start:stop])
        encoded[start:stop] = token_hidden.half()
        attention_logits[start:stop] = token_logits.half()
    del field
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sizes = _parse_sizes(args.neighborhood_sizes)
    valid_rows = torch.where(valid)[0]
    neighbor_np = query_region_neighbor_rows(
        xyz,
        valid_rows,
        max(sizes),
        domain=str(args.neighbor_domain),
        workers=int(args.knn_workers),
    )

    descriptors = torch.zeros(count, int(args.output_dim), dtype=torch.float16)
    valid_device = valid.to(device)
    primary_valid_device = (
        primary_valid.to(device) if primary_valid is not None else None
    )
    for start in range(0, valid_rows.numel(), int(args.semantic_batch_size)):
        stop = min(valid_rows.numel(), start + int(args.semantic_batch_size))
        neighborhood = torch.from_numpy(neighbor_np[start:stop]).to(device)
        center_rows = valid_rows[start:stop].to(device)
        scale_descriptors: list[torch.Tensor] = []
        for requested in sizes:
            scale = min(int(requested), neighborhood.shape[1])
            rows = neighborhood[:, :scale]
            token_mask = build_region_token_mask(
                rows,
                valid_device,
                center_rows,
                policy=str(args.neighbor_token_policy),
                primary_valid=primary_valid_device,
            )
            summary = bridge.summarize_preencoded_region(
                radio[rows],
                encoded[rows],
                attention_logits[rows],
                token_mask=token_mask,
            )
            projected = summary_head(summary[:, None])[:, 0]
            scale_descriptors.append(F.normalize(projected.float(), dim=-1, eps=1e-8))
        fused = F.normalize(
            torch.stack(scale_descriptors, dim=1).mean(dim=1), dim=-1, eps=1e-8
        )
        descriptors[valid_rows[start:stop]] = fused.half().cpu()

    # The completion field freezes all dominant-MPR predictions.  Reusing the
    # already materialised primary descriptors makes that invariant bitwise at
    # the query-cache boundary as well, while only fallback rows are new.
    if primary_semantic_features is not None:
        if primary_semantic_features.shape != descriptors.shape:
            raise ValueError("primary semantic descriptor dimensions differ")
        descriptors[primary_valid] = primary_semantic_features[primary_valid].to(
            descriptors.dtype
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    radio_checkpoint_sha256 = _sha256_file(args.radio_checkpoint)
    region_policy = {
        "neighborhood_type": "density_adaptive_3d_knn",
        "neighbor_domain": str(args.neighbor_domain),
        "neighbor_token_policy": str(args.neighbor_token_policy),
        "primary_valid_count": (
            int(primary_valid.sum()) if primary_valid is not None else None
        ),
        "neighborhood_sizes": list(sizes),
        "scale_fusion": "normalized_official_descriptor_mean",
        "radio_source": str(args.radio_source),
        "semantic_confidence": confidence_report,
        "primary_semantic_cache": (
            str(primary_semantic_path.resolve())
            if primary_semantic_path is not None
            else None
        ),
        "primary_semantic_cache_sha256": (
            _sha256_file(primary_semantic_path)
            if primary_semantic_path is not None
            else None
        ),
        "primary_descriptors_bitwise_preserved": bool(
            primary_semantic_features is not None
        ),
    }
    region_policy_sha256 = hashlib.sha256(
        json.dumps(region_policy, sort_keys=True).encode("utf-8")
    ).hexdigest()
    target_space = {
        "bridge_checkpoint_sha256": bridge_manifest.checkpoint_sha256,
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        "official_summary_head": True,
        "custom_text_projection": False,
        "region_policy_sha256": region_policy_sha256,
    }
    target_space_digest = hashlib.sha256(
        json.dumps(target_space, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": 1,
        "source": f"{args.radio_source}_radio_primitive_neighborhood",
        "field_checkpoint": str(Path(args.field_checkpoint).resolve()),
        "field_checkpoint_sha256": _sha256_file(args.field_checkpoint),
        "mpr_cache": str(cache_path.resolve()),
        "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
        "bridge_checkpoint_sha256": bridge_manifest.checkpoint_sha256,
        "bridge_training_scope": bridge_manifest.training_scope,
        "radio_checkpoint": str(Path(args.radio_checkpoint).resolve()),
        "radio_checkpoint_sha256": radio_checkpoint_sha256,
        **region_policy,
        "region_policy_sha256": region_policy_sha256,
        "target_space_digest": target_space_digest,
        "official_summary_head": True,
        "custom_text_projection": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }
    torch.save(
        {
            "schema_version": 1,
            "summary_features": descriptors,
            "features": descriptors,
            "valid": valid,
            "primary_valid": primary_valid,
            "semantic_confidence": semantic_confidence,
            "xyz": xyz,
            "geometry_fingerprint": mpr.get("geometry_fingerprint", {}),
            "metadata": metadata,
        },
        output,
    )
    report = {
        **metadata,
        "output": str(output),
        "num_gaussians": count,
        "valid_gaussians": int(valid.sum()),
        "feature_dim": descriptors.shape[1],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mpr-cache", default="")
    parser.add_argument(
        "--primary-semantic-cache",
        default="",
        help=(
            "Optional query-free base cache whose primary descriptor rows are "
            "preserved bitwise while fallback rows are completed."
        ),
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--neighborhood-sizes", default="8,32,64")
    parser.add_argument(
        "--neighbor-domain",
        choices=("all", "valid"),
        default="all",
        help=(
            "all preserves the legacy geometry-first masked neighborhood; "
            "valid requests the declared number of nearest MPR-observed rows."
        ),
    )
    parser.add_argument(
        "--neighbor-token-policy",
        choices=("all_valid", "primary_plus_center"),
        default="all_valid",
        help=(
            "primary_plus_center preserves dominant-MPR region context and "
            "admits only the current fallback center from support completion."
        ),
    )
    parser.add_argument(
        "--radio-source",
        choices=["canonical", "mpr"],
        default="canonical",
        help="Use the learned field or its query-free multiview RADIO target (oracle audit).",
    )
    parser.add_argument("--radio-batch-size", type=int, default=16384)
    parser.add_argument("--semantic-batch-size", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=1536)
    parser.add_argument("--knn-workers", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
