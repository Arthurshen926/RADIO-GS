"""Lift legal source-view SAM3 proposals to Gaussian memberships with exact MPR.

This builder deliberately separates object extent from text identity.  It uses
only mapping-time SAM3 masks and a frozen, query-independent sparse renderer
authority.  Benchmark masks and evaluation RGB are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.models.foundation_cache import load_foundation_cache
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_sam3_exact_mpr_memberships.v1"
RESPONSIBILITY_SCHEMA = "radio_gs.sparse_exact_marginal_responsibility_authority.v1"


def _float32_rows_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _frame_id(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"SAM3 cache filename has no numeric frame id: {path}") from exc


def exact_mpr_target_weights(
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_pixels: int,
) -> torch.Tensor:
    """Return ``w * (w / sum_pixel(w))`` for every sparse compositor hit."""

    pixels = pixel_ids.long()
    weights = base_weights.float()
    if pixels.ndim != 1 or weights.ndim != 1 or pixels.shape != weights.shape:
        raise ValueError("pixel_ids and base_weights must be aligned vectors")
    if pixels.numel() == 0:
        return weights
    if int(pixels.min()) < 0 or int(pixels.max()) >= int(num_pixels):
        raise ValueError("pixel id is outside the declared feature grid")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("base weights must be finite and non-negative")
    pixel_mass = weights.new_zeros((int(num_pixels),))
    pixel_mass.index_add_(0, pixels, weights)
    return weights.square() / pixel_mass[pixels].clamp_min(1e-12)


def lift_masks_with_exact_mpr(
    mask_logits: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_gaussians: int,
    feature_height: int,
    feature_width: int,
    proposal_scores: torch.Tensor | None = None,
    min_membership: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift dense mask logits into sparse, occlusion-aware Gaussian memberships."""

    if mask_logits.ndim != 3:
        raise ValueError("mask_logits must have shape [M,H,W]")
    gids = gaussian_ids.long()
    pixels = pixel_ids.long()
    if gids.shape != pixels.shape or gids.shape != base_weights.shape:
        raise ValueError("sparse responsibility tensors must be aligned")
    if gids.numel() and (int(gids.min()) < 0 or int(gids.max()) >= int(num_gaussians)):
        raise ValueError("Gaussian id is outside the declared row domain")
    num_masks = int(mask_logits.shape[0])
    if proposal_scores is not None and tuple(proposal_scores.shape) != (num_masks,):
        raise ValueError("proposal_scores must have shape [M]")
    if not 0.0 <= float(min_membership) <= 1.0:
        raise ValueError("min_membership must be in [0,1]")
    if num_masks == 0 or gids.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty_long, empty_long.clone(), torch.empty(0, device=gids.device)

    probabilities = F.interpolate(
        torch.sigmoid(mask_logits.float()).unsqueeze(1),
        size=(int(feature_height), int(feature_width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).flatten(1)
    num_pixels = int(feature_height) * int(feature_width)
    target_weights = exact_mpr_target_weights(
        pixels,
        base_weights,
        num_pixels=num_pixels,
    )
    denominator = target_weights.new_zeros((int(num_gaussians),))
    denominator.index_add_(0, gids, target_weights)

    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    confidence = (
        proposal_scores.float().clamp(0.0, 1.0)
        if proposal_scores is not None
        else torch.ones(num_masks, device=mask_logits.device)
    )
    for proposal_index in range(num_masks):
        numerator = target_weights.new_zeros((int(num_gaussians),))
        numerator.index_add_(
            0,
            gids,
            target_weights * probabilities[proposal_index, pixels],
        )
        membership = numerator / denominator.clamp_min(1e-12)
        keep = membership >= float(min_membership)
        if not bool(keep.any()):
            continue
        rows = torch.nonzero(keep, as_tuple=False).flatten()
        row_chunks.append(rows)
        proposal_chunks.append(torch.full_like(rows, proposal_index))
        weight_chunks.append(membership[rows] * confidence[proposal_index])
    if not row_chunks:
        empty_long = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty_long, empty_long.clone(), torch.empty(0, device=gids.device)
    return (
        torch.cat(row_chunks),
        torch.cat(proposal_chunks),
        torch.cat(weight_chunks).float(),
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(f"output already exists: {output}")
    responsibility_path = Path(args.responsibility_authority).expanduser().resolve()
    responsibility = json.loads(responsibility_path.read_text(encoding="utf-8"))
    if responsibility.get("schema") != RESPONSIBILITY_SCHEMA:
        raise ValueError("exact-marginal responsibility schema differs")
    metadata = dict(responsibility.get("metadata", {}))
    frame_ids = [int(value) for value in responsibility.get("frame_indices", [])]
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("responsibility frame ids are empty or duplicated")
    if any(bool(metadata.get(key, False)) for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"
    )):
        raise ValueError("responsibility authority is not query-independent source-only data")

    primitive_path = Path(args.primitive_cache).expanduser().resolve()
    if args.expected_primitive_cache_sha256:
        observed = sha256_file(primitive_path)
        if observed != args.expected_primitive_cache_sha256:
            raise ValueError("primitive cache SHA-256 differs")
    primitive = torch.load(primitive_path, map_location="cpu")
    xyz = torch.as_tensor(primitive["xyz"]).float()
    num_gaussians = int(xyz.shape[0])
    xyz_sha256 = _float32_rows_sha256(xyz)
    if xyz_sha256 != str(metadata.get("xyz_sha256", "")):
        raise ValueError("primitive cache and responsibility Gaussian rows differ")
    del primitive, xyz

    feature_height = int(metadata.get("feature_height", 0))
    feature_width = int(metadata.get("feature_width", 0))
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("responsibility feature grid is invalid")
    frame_to_view = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    sam_root = Path(args.sam3_cache_root).expanduser().resolve() / args.scene
    sam_paths = sorted(sam_root.glob("frame_*.pt"), key=_frame_id)
    if not sam_paths:
        raise FileNotFoundError(f"no source SAM3 caches found: {sam_root}")

    device = torch.device(args.device)
    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    proposal_view_indices: list[int] = []
    proposal_query_names: list[str] = []
    proposal_offset = 0
    source_records: list[dict[str, Any]] = []
    views_root = Path(str(responsibility_path) + ".views")
    for source_view_index, sam_path in enumerate(sam_paths):
        frame_id = _frame_id(sam_path)
        if frame_id not in frame_to_view:
            raise ValueError(f"SAM3 source frame {frame_id} has no exact MPR authority")
        responsibility_view_index = frame_to_view[frame_id]
        view_path = views_root / f"view_{responsibility_view_index:05d}.pt"
        view = torch.load(view_path, map_location="cpu")
        if (
            int(view.get("frame_index", -1)) != frame_id
            or int(view.get("num_gaussians", -1)) != num_gaussians
            or int(view.get("num_pixels", -1)) != feature_height * feature_width
        ):
            raise ValueError(f"responsibility view binding differs: {view_path}")
        cache = load_foundation_cache(sam_path, require_official=True)
        head = cache.heads.get("sam3")
        if head is None or head.mask_logits is None:
            raise ValueError(f"official SAM3 head is absent: {sam_path}")
        mask_logits = head.mask_logits.detach().to(device=device, dtype=torch.float32)
        scores = (
            head.scores.detach().to(device=device, dtype=torch.float32)
            if head.scores is not None
            else None
        )
        rows, local_proposals, weights = lift_masks_with_exact_mpr(
            mask_logits,
            torch.as_tensor(view["gaussian_ids"]).to(device),
            torch.as_tensor(view["pixel_ids"]).to(device),
            torch.as_tensor(view["base_weights"]).to(device),
            num_gaussians=num_gaussians,
            feature_height=feature_height,
            feature_width=feature_width,
            proposal_scores=scores,
            min_membership=args.min_membership,
        )
        num_proposals = int(mask_logits.shape[0])
        if rows.numel():
            row_chunks.append(rows.cpu())
            proposal_chunks.append((local_proposals + proposal_offset).cpu())
            weight_chunks.append(weights.cpu())
        queries = list(head.queries or [])
        mask_query_indices = (
            head.mask_query_indices.detach().long().cpu()
            if head.mask_query_indices is not None
            else torch.full((num_proposals,), -1, dtype=torch.long)
        )
        if tuple(mask_query_indices.shape) != (num_proposals,):
            raise ValueError("SAM3 proposal-query binding differs")
        for local_index in range(num_proposals):
            query_index = int(mask_query_indices[local_index])
            proposal_query_names.append(
                str(queries[query_index]) if 0 <= query_index < len(queries) else ""
            )
            proposal_view_indices.append(source_view_index)
        source_records.append({
            "frame_id": frame_id,
            "source_view_index": source_view_index,
            "responsibility_view_index": responsibility_view_index,
            "sam3_cache": str(sam_path),
            "sam3_cache_sha256": sha256_file(sam_path),
            "responsibility_view": str(view_path),
            "responsibility_view_sha256": sha256_file(view_path),
            "num_proposals": num_proposals,
            "num_memberships": int(rows.numel()),
        })
        proposal_offset += num_proposals

    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": args.scene,
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "row_indices": torch.cat(row_chunks) if row_chunks else torch.empty(0, dtype=torch.long),
        "proposal_indices": (
            torch.cat(proposal_chunks) if proposal_chunks else torch.empty(0, dtype=torch.long)
        ),
        "weights": torch.cat(weight_chunks) if weight_chunks else torch.empty(0),
        "proposal_view_indices": torch.tensor(proposal_view_indices, dtype=torch.long),
        "proposal_query_names": proposal_query_names,
        "metadata": {
            "membership_lifting": "exact_front_to_back_marginal_target_weight",
            "target_weight": "base_weight_squared_divided_by_pixel_mass",
            "min_membership": float(args.min_membership),
            "feature_height": feature_height,
            "feature_width": feature_width,
            "xyz_sha256": xyz_sha256,
            "primitive_cache": str(primitive_path),
            "primitive_cache_sha256": sha256_file(primitive_path),
            "responsibility_authority": str(responsibility_path),
            "responsibility_authority_sha256": sha256_file(responsibility_path),
            "source_view_count": len(source_records),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "query_independent_geometry_lifting": True,
            "source_records": source_records,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scene": args.scene,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "num_memberships": int(payload["row_indices"].numel()),
        "source_view_count": len(source_records),
        "min_membership": float(args.min_membership),
        "xyz_sha256": xyz_sha256,
        "membership_lifting": payload["metadata"]["membership_lifting"],
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--sam3-cache-root", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--expected-primitive-cache-sha256", default="")
    parser.add_argument("--min-membership", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
