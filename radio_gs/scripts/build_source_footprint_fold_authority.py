#!/usr/bin/env python3
"""Build a query-free source-footprint fold authority from an exact W cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.interfaces.prompt_responsibility_cache import (
    PromptResponsibilityAuthority,
    load_prompt_responsibility_cache,
)
from radio_gs.querying.source_footprint_fold_authority import (
    build_source_raster_dominant_footprint_authority,
    save_source_footprint_fold_authority,
    splitmix64_source_group_folds,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    sha256_file,
)


def _require_false(mapping: dict, names: tuple[str, ...], *, label: str) -> None:
    for name in names:
        if mapping.get(name) is not False:
            raise ValueError(f"{label} safety flag differs: {name}")


def _filter_exact_triplets_to_primitive_rows(
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    primitive_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Restrict complete W triplets to the immutable hierarchy row domain."""

    identifiers = torch.as_tensor(gaussian_ids).long().cpu().reshape(-1)
    pixels = torch.as_tensor(pixel_ids).long().cpu().reshape(-1)
    values = torch.as_tensor(weights).cpu().reshape(-1)
    rows = torch.as_tensor(primitive_rows).long().cpu().reshape(-1)
    if identifiers.shape != pixels.shape or identifiers.shape != values.shape:
        raise ValueError("exact-W triplets do not align before row filtering")
    if rows.numel() == 0:
        raise ValueError("primitive row authority is empty")
    local = torch.searchsorted(rows, identifiers)
    in_range = local < rows.numel()
    keep = in_range & (rows[local.clamp_max(rows.numel() - 1)] == identifiers)
    retained_weights = values[keep].contiguous()
    report: dict[str, float | int] = {
        "input_triplets": int(keep.numel()),
        "retained_triplets": int(keep.sum()),
        "excluded_triplets": int((~keep).sum()),
        "input_weight_mass": float(values.double().sum()),
        "retained_weight_mass": float(retained_weights.double().sum()),
        "excluded_weight_mass": float(values[~keep].double().sum()),
    }
    return (
        identifiers[keep].contiguous(),
        pixels[keep].contiguous(),
        retained_weights,
        report,
    )


def build(args: argparse.Namespace) -> dict[str, object]:
    exact_report, exact_report_sha, exact_report_path = load_json_object(
        args.exact_w_report,
        expected_sha256=args.expected_exact_w_report_sha256,
        label="exact-W source report",
    )
    _require_false(
        exact_report,
        ("target_rgb_opened", "target_mask_opened"),
        label="exact-W report",
    )
    header = exact_report.get("reference_mask_header_authority")
    if not isinstance(header, dict):
        raise ValueError("exact-W report lacks reference-mask header authority")
    _require_false(
        header,
        (
            "source_mask_pixels_decoded",
            "source_mask_pixels_interpreted",
            "query_or_evidence_constructed",
            "target_rgb_opened",
            "target_mask_opened",
            "target_metric_computed",
        ),
        label="reference-mask header authority",
    )
    if exact_report.get("historical_top1_responsibility_opened") is not False:
        raise ValueError("exact-W export opened a historical top-1 cache")
    if exact_report.get("file_sha256") != args.expected_exact_w_sha256:
        raise ValueError("exact-W report file SHA differs from caller authority")
    authority_value = exact_report.get("authority")
    authority = PromptResponsibilityAuthority.from_dict(authority_value)
    if exact_report.get("authority_sha256") != authority.digest:
        raise ValueError("exact-W report authority digest differs")
    exact_cache = load_prompt_responsibility_cache(
        args.exact_w,
        expected_authority=authority,
        expected_file_sha256=args.expected_exact_w_sha256,
    )
    if exact_report.get("tensor_bundle_sha256") != exact_cache.tensor_bundle_sha256:
        raise ValueError("exact-W report tensor bundle differs")

    graph, graph_sha, graph_path = load_torch_mapping(
        args.primitive_row_authority,
        expected_sha256=args.expected_primitive_row_authority_sha256,
        label="primitive-row authority",
    )
    graph_metadata = graph.get("metadata")
    if not isinstance(graph_metadata, dict):
        raise ValueError("primitive-row authority lacks metadata")
    _require_false(
        graph_metadata,
        ("benchmark_images_opened", "benchmark_masks_opened", "text_queries_opened"),
        label="primitive-row authority",
    )
    if int(graph.get("num_global_rows", -1)) != authority.num_gaussians:
        raise ValueError("primitive-row authority global domain differs from exact W")
    primitive_rows = torch.as_tensor(graph.get("global_rows")).long().cpu().contiguous()
    if (
        primitive_rows.ndim != 1
        or primitive_rows.numel() == 0
        or bool((primitive_rows < 0).any())
        or bool((primitive_rows >= authority.num_gaussians).any())
        or (
            primitive_rows.numel() > 1
            and not bool((primitive_rows[1:] > primitive_rows[:-1]).all())
        )
    ):
        raise ValueError("primitive-row authority rows are not sorted unique global rows")

    filtered_ids, filtered_pixels, filtered_weights, row_filter_report = (
        _filter_exact_triplets_to_primitive_rows(
            exact_cache.gaussian_ids,
            exact_cache.pixel_ids,
            exact_cache.weights,
            primitive_rows,
        )
    )
    footprint = build_source_raster_dominant_footprint_authority(
        filtered_pixels,
        filtered_ids,
        filtered_weights,
        height=authority.height,
        width=authority.width,
        hierarchy_primitive_rows=primitive_rows,
        primitive_id_domain="global_rows",
        source_triplet_authority_sha256=authority.digest,
        expected_source_triplet_authority_sha256=authority.digest,
    )
    artifact = save_source_footprint_fold_authority(
        footprint,
        args.output,
        overwrite=bool(args.overwrite),
    )
    fold_ids = splitmix64_source_group_folds(footprint.group_ids)
    fold_rows = [int((fold_ids == fold).sum()) for fold in range(3)]
    visible = footprint.visible_mass > 0
    fold_visible_rows = [int((visible & (fold_ids == fold)).sum()) for fold in range(3)]
    report = {
        "schema_version": 1,
        "method": "source_raster_dominant_footprint_blocks_v1",
        "artifact": artifact,
        "exact_w": {
            "path": str(Path(args.exact_w).resolve()),
            "sha256": args.expected_exact_w_sha256,
            "report_path": str(exact_report_path),
            "report_sha256": exact_report_sha,
            "authority_sha256": authority.digest,
            "tensor_bundle_sha256": exact_cache.tensor_bundle_sha256,
            "triplet_count": int(exact_cache.weights.numel()),
        },
        "primitive_row_authority": {
            "path": str(graph_path),
            "sha256": graph_sha,
            "rows": int(primitive_rows.numel()),
            "exact_w_row_filter": row_filter_report,
        },
        "raster": {
            "height": authority.height,
            "width": authority.width,
            "block_rows": footprint.block_rows,
            "block_cols": footprint.block_cols,
            "block_count": footprint.block_count,
            "invisible_group_id": footprint.invisible_group_id,
        },
        "population": {
            "visible_rows": int(visible.sum()),
            "invisible_rows": int((~visible).sum()),
            "fold_rows": fold_rows,
            "fold_visible_rows": fold_visible_rows,
            "purity_mean_visible": (
                float(footprint.purity[visible].mean()) if bool(visible.any()) else 0.0
            ),
        },
        "source_mask_pixels_decoded": False,
        "query_or_evidence_constructed": False,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-w", required=True)
    parser.add_argument("--expected-exact-w-sha256", required=True)
    parser.add_argument("--exact-w-report", required=True)
    parser.add_argument("--expected-exact-w-report-sha256", required=True)
    parser.add_argument("--primitive-row-authority", required=True)
    parser.add_argument("--expected-primitive-row-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
