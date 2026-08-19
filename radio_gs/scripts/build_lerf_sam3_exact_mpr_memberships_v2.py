"""Lift source SAM3 *probability masks* with exact MPR (corrected v2).

The historical foundation-cache key is named ``mask_logits`` for compatibility,
but ``build_sam3_foundation_cache.make_sam3_cache_payload`` stores the official
SAM3 output masks after conversion to float.  Those values are already in
``[0,1]`` and must not pass through another sigmoid.  V1 did so and converted
every near-zero background value to approximately 0.5, producing near-global
proposal memberships.

V2 also separates proposal-wide SAM quality from conditional row membership.
The former is stored as ``proposal_scores`` and never multiplies the latter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.models.foundation_cache import load_foundation_cache
from radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships import (
    RESPONSIBILITY_SCHEMA,
    _float32_rows_sha256,
    _frame_id,
    exact_mpr_target_weights,
)
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_sam3_exact_mpr_memberships.v2"


def lift_probability_masks_with_exact_mpr(
    mask_probabilities: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    num_gaussians: int,
    feature_height: int,
    feature_width: int,
    min_membership: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift official SAM3 probabilities without applying a second sigmoid."""

    probabilities = mask_probabilities.float()
    if probabilities.ndim != 3:
        raise ValueError("mask_probabilities must have shape [M,H,W]")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("mask probabilities must be finite")
    if probabilities.numel() and (
        float(probabilities.min()) < -1e-6 or float(probabilities.max()) > 1.0 + 1e-6
    ):
        raise ValueError("official SAM3 mask cache is not in probability space")
    probabilities = probabilities.clamp(0.0, 1.0)
    gids = gaussian_ids.long()
    pixels = pixel_ids.long()
    base = base_weights.float()
    if gids.shape != pixels.shape or gids.shape != base.shape:
        raise ValueError("sparse responsibility tensors must be aligned")
    if gids.numel() and (int(gids.min()) < 0 or int(gids.max()) >= int(num_gaussians)):
        raise ValueError("Gaussian id is outside the declared row domain")
    if not 0.0 <= float(min_membership) <= 1.0:
        raise ValueError("min_membership must be in [0,1]")
    num_masks = int(probabilities.shape[0])
    if num_masks == 0 or gids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty, empty.clone(), torch.empty(0, device=gids.device)

    probabilities = F.interpolate(
        probabilities.unsqueeze(1),
        size=(int(feature_height), int(feature_width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).flatten(1)
    num_pixels = int(feature_height) * int(feature_width)
    target = exact_mpr_target_weights(pixels, base, num_pixels=num_pixels)
    denominator = target.new_zeros((int(num_gaussians),))
    denominator.index_add_(0, gids, target)
    rows_out: list[torch.Tensor] = []
    props_out: list[torch.Tensor] = []
    weights_out: list[torch.Tensor] = []
    for proposal_index in range(num_masks):
        numerator = target.new_zeros((int(num_gaussians),))
        numerator.index_add_(0, gids, target * probabilities[proposal_index, pixels])
        membership = numerator / denominator.clamp_min(1e-12)
        keep = membership >= float(min_membership)
        if bool(keep.any()):
            rows = torch.nonzero(keep, as_tuple=False).flatten()
            rows_out.append(rows)
            props_out.append(torch.full_like(rows, proposal_index))
            weights_out.append(membership[rows])
    if not rows_out:
        empty = torch.empty(0, dtype=torch.long, device=gids.device)
        return empty, empty.clone(), torch.empty(0, device=gids.device)
    return torch.cat(rows_out), torch.cat(props_out), torch.cat(weights_out).float()


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"output already exists: {output}")
    authority_path = Path(args.responsibility_authority).expanduser().resolve()
    authority = json.loads(authority_path.read_text())
    if authority.get("schema") != RESPONSIBILITY_SCHEMA:
        raise ValueError("exact-MPR responsibility schema differs")
    metadata = dict(authority.get("metadata", {}))
    frame_ids = [int(value) for value in authority.get("frame_indices", [])]
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("responsibility frame ids are empty or duplicated")
    if any(
        bool(metadata.get(key, False))
        for key in ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened")
    ):
        raise ValueError("responsibility authority is not query-independent source data")

    primitive_path = Path(args.primitive_cache).expanduser().resolve()
    primitive = torch.load(primitive_path, map_location="cpu")
    xyz = torch.as_tensor(primitive.get("xyz"), dtype=torch.float32)
    num_gaussians = int(xyz.shape[0])
    xyz_sha256 = _float32_rows_sha256(xyz)
    if xyz_sha256 != str(metadata.get("xyz_sha256", "")):
        raise ValueError("primitive and exact-MPR Gaussian rows differ")
    del primitive, xyz

    feature_height = int(metadata.get("feature_height", 0))
    feature_width = int(metadata.get("feature_width", 0))
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("exact-MPR feature grid is invalid")
    frame_to_view = {frame: index for index, frame in enumerate(frame_ids)}
    sam_root = Path(args.sam3_cache_root).expanduser().resolve() / args.scene
    sam_paths = sorted(sam_root.glob("frame_*.pt"), key=_frame_id)
    if not sam_paths:
        raise FileNotFoundError(f"no source SAM3 caches under {sam_root}")

    device = torch.device(args.device)
    row_chunks: list[torch.Tensor] = []
    proposal_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    proposal_views: list[int] = []
    proposal_names: list[str] = []
    proposal_scores: list[float] = []
    proposal_offset = 0
    records: list[dict[str, Any]] = []
    views_root = Path(str(authority_path) + ".views")
    for source_view, sam_path in enumerate(sam_paths):
        frame_id = _frame_id(sam_path)
        if frame_id not in frame_to_view:
            raise ValueError(f"SAM source frame lacks exact-MPR authority: {frame_id}")
        authority_view = frame_to_view[frame_id]
        view_path = views_root / f"view_{authority_view:05d}.pt"
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
            raise ValueError(f"official SAM3 probability mask is absent: {sam_path}")
        masks = head.mask_logits.detach().to(device=device, dtype=torch.float32)
        rows, local_props, row_membership = lift_probability_masks_with_exact_mpr(
            masks,
            torch.as_tensor(view["gaussian_ids"]).to(device),
            torch.as_tensor(view["pixel_ids"]).to(device),
            torch.as_tensor(view["base_weights"]).to(device),
            num_gaussians=num_gaussians,
            feature_height=feature_height,
            feature_width=feature_width,
            min_membership=float(args.min_membership),
        )
        num_proposals = int(masks.shape[0])
        if rows.numel():
            row_chunks.append(rows.cpu())
            proposal_chunks.append((local_props + proposal_offset).cpu())
            weight_chunks.append(row_membership.cpu())
        queries = list(head.queries or [])
        query_indices = (
            head.mask_query_indices.detach().long().cpu()
            if head.mask_query_indices is not None
            else torch.full((num_proposals,), -1, dtype=torch.long)
        )
        scores = (
            head.scores.detach().float().cpu()
            if head.scores is not None
            else torch.ones(num_proposals)
        )
        if query_indices.shape != (num_proposals,) or scores.shape != (num_proposals,):
            raise ValueError("SAM proposal identity/quality rows differ")
        for local_index in range(num_proposals):
            query_index = int(query_indices[local_index])
            proposal_names.append(
                str(queries[query_index]) if 0 <= query_index < len(queries) else ""
            )
            proposal_views.append(source_view)
            proposal_scores.append(float(scores[local_index]))
        records.append(
            {
                "frame_id": frame_id,
                "source_view_index": source_view,
                "responsibility_view_index": authority_view,
                "sam3_cache": str(sam_path),
                "sam3_cache_sha256": sha256_file(sam_path),
                "responsibility_view": str(view_path),
                "responsibility_view_sha256": sha256_file(view_path),
                "num_proposals": num_proposals,
                "num_memberships": int(rows.numel()),
            }
        )
        proposal_offset += num_proposals

    payload = {
        "schema": SCHEMA,
        "schema_version": 2,
        "scene": args.scene,
        "num_rows": num_gaussians,
        "num_proposals": proposal_offset,
        "row_indices": torch.cat(row_chunks) if row_chunks else torch.empty(0, dtype=torch.long),
        "proposal_indices": (
            torch.cat(proposal_chunks) if proposal_chunks else torch.empty(0, dtype=torch.long)
        ),
        "weights": torch.cat(weight_chunks) if weight_chunks else torch.empty(0),
        "proposal_view_indices": torch.tensor(proposal_views, dtype=torch.long),
        "proposal_query_names": proposal_names,
        "proposal_scores": torch.tensor(proposal_scores, dtype=torch.float32),
        "metadata": {
            "mask_tensor_semantics": "official_sam3_probability_not_logits",
            "mask_nonlinearity_after_cache_load": "identity",
            "membership_lifting": "exact_front_to_back_marginal_target_weight",
            "proposal_quality_role": "separate_proposal_metadata_not_row_membership_multiplier",
            "min_membership": float(args.min_membership),
            "feature_height": feature_height,
            "feature_width": feature_width,
            "xyz_sha256": xyz_sha256,
            "primitive_cache": str(primitive_path),
            "primitive_cache_sha256": sha256_file(primitive_path),
            "responsibility_authority": str(authority_path),
            "responsibility_authority_sha256": sha256_file(authority_path),
            "source_view_count": len(records),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "query_independent_geometry_lifting": True,
            "query_independent_mask_hierarchy": False,
            "capability_track": "query_conditioned_source_sam_diagnostic_not_p0",
            "source_records": records,
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
        "source_view_count": len(records),
        "mask_tensor_semantics": payload["metadata"]["mask_tensor_semantics"],
        "mask_nonlinearity_after_cache_load": "identity",
        "query_independent_mask_hierarchy": False,
        "capability_track": "query_conditioned_source_sam_diagnostic_not_p0",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--responsibility-authority", required=True)
    parser.add_argument("--sam3-cache-root", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--min-membership", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
