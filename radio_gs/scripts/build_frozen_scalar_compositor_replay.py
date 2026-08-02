#!/usr/bin/env python3
"""Build a bounded sparse replay of the frozen scalar compositor on CPU.

The input contribution cache is an exact, query-free export of
``rasterize_single_view_contributions``.  This program applies the variant
selected by ``query-free-scalar-compositor-v1``, normalizes within each
rendered pixel, retains a deterministic bounded set of nonempty pixels, and
binds a fixed prefix of the target-blind fit vocabulary.  It never invents
raster weights and cannot consume masks, labels, or benchmark queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from radio_gs.rendering.contribution_compositor import (
    build_compositing_variants,
)
from radio_gs.scripts.materialize_surface_dual_descriptor import (
    INPUT_ARTIFACT_TYPE,
    WEIGHTS_ARTIFACT_TYPE,
    _QUERY_FREE_FLAGS,
    _validate_compositor_manifest,
)
from radio_gs.scripts.train_surface_region_text_response_distill import (
    load_fit_text_embedding_bank,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    validate_file_record,
    write_torch_noclobber,
)


SCHEMA_VERSION = 2
CONTRIBUTION_SCHEMA_VERSION = 1
CONTRIBUTION_ARTIFACT_TYPE = "exact_front_to_back_scalar_contribution_cache"
REPLAY_QUERY_COUNT = 8
MAX_RENDER_ROWS_PER_SOURCE = 256
_CONTRIBUTION_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "primitive_ids",
        "render_row_keys",
        "render_row_index",
        "primitive_row_index",
        "front_to_back_contribution_weights",
        "metadata",
    }
)


def exact_contribution_cache_payload(
    *,
    primitive_input_cache: Path,
    view_hits: Sequence[tuple[str, Mapping[str, Any]]],
    geometry_checkpoint: Path,
    camera_manifest: Path,
    target_blind_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize real outputs of ``rasterize_single_view_contributions``.

    The function intentionally accepts the already computed hit dictionaries
    rather than offering a second rasterizer.  Callers therefore reuse the
    production compositor exactly and this CPU-side exporter only compacts its
    grouped pixel IDs into stable sparse row indices.
    """

    ids, primitive_record, row_authority = _load_primitive_ids(
        primitive_input_cache
    )
    _require_query_free(target_blind_provenance, label="contribution export")
    render_keys: list[str] = []
    row_parts: list[torch.Tensor] = []
    primitive_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    seen_views: set[str] = set()
    for view_key, hits in view_hits:
        if not view_key or view_key in seen_views or "/pixel-" in view_key:
            raise ValueError("exact contribution export view keys differ")
        seen_views.add(view_key)
        primitive_rows = torch.as_tensor(hits.get("gaussian_ids")).long().cpu().reshape(-1)
        pixel_rows = torch.as_tensor(hits.get("pixel_ids")).long().cpu().reshape(-1)
        weights = torch.as_tensor(hits.get("weights")).float().cpu().reshape(-1)
        if (
            primitive_rows.shape != pixel_rows.shape
            or primitive_rows.shape != weights.shape
            or primitive_rows.numel() == 0
            or bool((primitive_rows < 0).any())
            or bool((primitive_rows >= len(ids)).any())
            or bool((pixel_rows < 0).any())
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
            or bool((pixel_rows[1:] < pixel_rows[:-1]).any())
        ):
            raise ValueError("exact production contribution hits differ")
        unique_pixels, local_rows = torch.unique_consecutive(
            pixel_rows, return_inverse=True
        )
        raw_mass = torch.zeros(unique_pixels.numel(), dtype=torch.float32)
        raw_mass.index_add_(0, local_rows, weights)
        if bool((raw_mass > 1.00001).any()) or not bool((raw_mass > 0).all()):
            raise ValueError("front-to-back contribution mass lies outside (0,1]")
        prefix = len(render_keys)
        render_keys.extend(
            f"{view_key}/pixel-{int(pixel)}" for pixel in unique_pixels
        )
        row_parts.append(local_rows + prefix)
        primitive_parts.append(primitive_rows)
        weight_parts.append(weights)
    if not render_keys:
        raise ValueError("exact contribution export contains no hits")
    implementation = (
        Path(__file__).resolve().parents[1]
        / "rendering"
        / "contribution_compositor.py"
    )
    geometry_record = file_record(geometry_checkpoint)
    if row_authority.get("geometry_checkpoint") != geometry_record:
        raise ValueError("contribution geometry differs from primitive row authority")
    return {
        "schema_version": CONTRIBUTION_SCHEMA_VERSION,
        "artifact_type": CONTRIBUTION_ARTIFACT_TYPE,
        "primitive_ids": ids,
        "render_row_keys": render_keys,
        "render_row_index": torch.cat(row_parts).contiguous(),
        "primitive_row_index": torch.cat(primitive_parts).contiguous(),
        "front_to_back_contribution_weights": torch.cat(weight_parts).contiguous(),
        "metadata": {
            **dict(target_blind_provenance),
            "producer": (
                "radio_gs.rendering.contribution_compositor."
                "rasterize_single_view_contributions"
            ),
            "weight_semantics": "exact_front_to_back_alpha_contribution",
            "same_geometry_rows_as_primitive_input": True,
            "primitive_input_cache": primitive_record,
            "geometry_xyz_sha256": row_authority["geometry_xyz_sha256"],
            "contribution_compositor_implementation": file_record(implementation),
            "geometry_checkpoint": file_record(geometry_checkpoint),
            "camera_manifest": file_record(camera_manifest),
        },
    }


def _require_query_free(metadata: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label} metadata is missing")
    for key in _QUERY_FREE_FLAGS:
        if metadata.get(key) is not False:
            raise ValueError(f"{label} must explicitly certify {key}=false")
    if (
        metadata.get("target_blind") is not True
        or metadata.get("benchmark_targets_or_metrics_used") is not False
    ):
        raise ValueError(f"{label} is not target-blind")
    return metadata


def _load_primitive_ids(
    path: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    payload, digest, source = load_torch_mapping(
        path, map_location="cpu", label="dual-descriptor primitive input cache"
    )
    ids = payload.get("primitive_ids")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != INPUT_ARTIFACT_TYPE
        or not isinstance(ids, list)
        or not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("dual-descriptor primitive input cache differs")
    metadata = _require_query_free(payload.get("metadata"), label=str(source))
    row_authority = metadata.get("production_primitive_row_authority")
    required = {
        "contract",
        "scene_id",
        "geometry_checkpoint",
        "geometry_xyz_sha256",
        "total_geometry_rows",
        "row_order",
        "complete_geometry_rows_present",
    }
    if not isinstance(row_authority, Mapping) or set(row_authority) != required:
        raise ValueError("primitive input lacks complete production row authority")
    scene = str(row_authority.get("scene_id", ""))
    if (
        row_authority.get("contract")
        != "complete_single_scene_gaussian_checkpoint_row_order_v1"
        or row_authority.get("row_order")
        != "zero_based_geometry_checkpoint_row_order"
        or row_authority.get("complete_geometry_rows_present") is not True
        or row_authority.get("total_geometry_rows") != len(ids)
        or len(str(row_authority.get("geometry_xyz_sha256", ""))) != 64
        or ids != [f"{scene}/primitive-{index}" for index in range(len(ids))]
    ):
        raise ValueError("primitive input production row order differs")
    validate_file_record(
        row_authority.get("geometry_checkpoint"),
        label="primitive input production geometry",
    )
    return (
        list(ids),
        {"path": str(source), "sha256": digest},
        dict(row_authority),
    )


def _load_contribution_cache(
    path: Path,
    *,
    primitive_ids: list[str],
    primitive_record: Mapping[str, str],
    row_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    payload, digest, source = load_torch_mapping(
        path, map_location="cpu", label="exact scalar contribution cache"
    )
    if (
        set(payload) != _CONTRIBUTION_KEYS
        or payload.get("schema_version") != CONTRIBUTION_SCHEMA_VERSION
        or payload.get("artifact_type") != CONTRIBUTION_ARTIFACT_TYPE
        or payload.get("primitive_ids") != primitive_ids
    ):
        raise ValueError(f"{source}: exact contribution-cache schema differs")
    metadata = _require_query_free(payload.get("metadata"), label=str(source))
    implementation = validate_file_record(
        metadata.get("contribution_compositor_implementation"),
        label="contribution compositor implementation",
    )
    producer_runner = validate_file_record(
        metadata.get("production_capture_runner_implementation"),
        label="production contribution capture runner",
    )
    if (
        implementation.name != "contribution_compositor.py"
        or producer_runner.name != "audit_feature_compositing.py"
        or metadata.get("production_capture_runner")
        != "query_free_feature_compositing_exact_hit_export_v1"
        or metadata.get("producer")
        != (
            "radio_gs.rendering.contribution_compositor."
            "rasterize_single_view_contributions"
        )
        or metadata.get("weight_semantics")
        != "exact_front_to_back_alpha_contribution"
        or metadata.get("same_geometry_rows_as_primitive_input") is not True
        or metadata.get("primitive_input_cache") != primitive_record
    ):
        raise ValueError("contribution cache is not an exact production export")
    validate_file_record(metadata.get("geometry_checkpoint"), label="replay geometry")
    validate_file_record(metadata.get("camera_manifest"), label="replay cameras")
    if (
        metadata.get("geometry_checkpoint") != row_authority["geometry_checkpoint"]
        or metadata.get("geometry_xyz_sha256")
        != row_authority["geometry_xyz_sha256"]
    ):
        raise ValueError("contribution cache geometry/primitive row binding differs")

    row_keys = payload.get("render_row_keys")
    row_index = payload.get("render_row_index")
    primitive_index = payload.get("primitive_row_index")
    weights = payload.get("front_to_back_contribution_weights")
    if (
        not isinstance(row_keys, list)
        or not row_keys
        or any(not isinstance(value, str) or not value for value in row_keys)
        or len(set(row_keys)) != len(row_keys)
        or not isinstance(row_index, torch.Tensor)
        or row_index.device.type != "cpu"
        or row_index.is_floating_point()
        or not isinstance(primitive_index, torch.Tensor)
        or primitive_index.device.type != "cpu"
        or primitive_index.is_floating_point()
        or not isinstance(weights, torch.Tensor)
        or weights.device.type != "cpu"
        or not weights.is_floating_point()
    ):
        raise ValueError("exact contribution-cache tensors differ")
    row_index = row_index.long().reshape(-1)
    primitive_index = primitive_index.long().reshape(-1)
    weights = weights.float().reshape(-1)
    if (
        row_index.shape != primitive_index.shape
        or row_index.shape != weights.shape
        or row_index.numel() == 0
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or bool((row_index < 0).any())
        or bool((row_index >= len(row_keys)).any())
        or bool((primitive_index < 0).any())
        or bool((primitive_index >= len(primitive_ids)).any())
    ):
        raise ValueError("exact contribution-cache indices/weights differ")
    raw_mass = torch.zeros(len(row_keys), dtype=torch.float32)
    raw_mass.index_add_(0, row_index, weights)
    if bool((raw_mass > 1.00001).any()) or not bool((raw_mass > 0).all()):
        raise ValueError("exact contribution-cache row mass lies outside (0,1]")
    return {
        "render_row_keys": list(row_keys),
        "render_row_index": row_index,
        "primitive_row_index": primitive_index,
        "weights": weights,
    }, {"path": str(source), "sha256": digest}


def _load_replay_queries(
    path: Path,
    manifest: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    bank = load_fit_text_embedding_bank(path, manifest)
    embeddings = torch.as_tensor(bank["embeddings"]).float().cpu()
    if embeddings.ndim != 2 or embeddings.shape[1] != 1536:
        raise ValueError("target-blind fit bank descriptor dimension differs")
    count = min(REPLAY_QUERY_COUNT, int(embeddings.shape[0]))
    if count <= 0:
        raise ValueError("target-blind fit bank is empty")
    queries = torch.nn.functional.normalize(embeddings[:count], dim=-1).contiguous()
    return queries, {
        "artifact": file_record(path),
        "manifest": file_record(manifest),
        "selection": "first_8_rows_of_frozen_order_or_all_if_fewer",
        "selected_row_indices": list(range(count)),
    }


def _selected_raw_weights(
    *,
    selected_variant: str,
    row_index: torch.Tensor,
    weights: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    variants = build_compositing_variants(
        row_index,
        weights,
        num_pixels=num_rows,
        gammas=(1.25, 1.5, 2.0),
        topk=(4,),
    )
    if selected_variant not in variants:
        raise ValueError(f"unsupported frozen scalar compositor: {selected_variant}")
    return variants[selected_variant].float()


def _bounded_sparse_rows(
    cache: Mapping[str, Any],
    *,
    selected_variant: str,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    source_rows = cache["render_row_index"]
    raw = _selected_raw_weights(
        selected_variant=selected_variant,
        row_index=source_rows,
        weights=cache["weights"],
        num_rows=len(cache["render_row_keys"]),
    )
    mass = torch.zeros(len(cache["render_row_keys"]), dtype=torch.float32)
    mass.index_add_(0, source_rows, raw)
    nonempty = torch.where(mass > 0)[0][
        :MAX_RENDER_ROWS_PER_SOURCE
    ]
    if nonempty.numel() == 0:
        raise ValueError("selected compositor has no nonempty replay rows")
    remap = torch.full((len(cache["render_row_keys"]),), -1, dtype=torch.long)
    remap[nonempty] = torch.arange(nonempty.numel())
    keep = remap[source_rows] >= 0
    rows = remap[source_rows[keep]]
    normalized = raw[keep] / mass[source_rows[keep]]
    check = torch.zeros(nonempty.numel(), dtype=torch.float32)
    check.index_add_(0, rows, normalized)
    if not torch.allclose(check, torch.ones_like(check), atol=1e-6, rtol=0.0):
        raise RuntimeError("normalized sparse compositor rows do not sum to one")
    return (
        [cache["render_row_keys"][int(index)] for index in nonempty],
        rows,
        cache["primitive_row_index"][keep],
        normalized.contiguous(),
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"scalar replay output must be new: {output}")
    primitive_ids, primitive_record, row_authority = _load_primitive_ids(
        Path(args.primitive_input_cache)
    )
    compositor_path = Path(args.scalar_compositor_manifest).resolve(strict=True)
    compositor = _validate_compositor_manifest(compositor_path)
    queries, query_source = _load_replay_queries(
        Path(args.fit_text_bank), Path(args.fit_text_bank_manifest)
    )
    loaded = [
        _load_contribution_cache(
            Path(path),
            primitive_ids=primitive_ids,
            primitive_record=primitive_record,
            row_authority=row_authority,
        )
        for path in args.contribution_cache
    ]
    all_keys: list[str] = []
    row_parts: list[torch.Tensor] = []
    primitive_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    for cache, _record in loaded:
        keys, rows, primitive_rows, weights = _bounded_sparse_rows(
            cache, selected_variant=compositor["selected_variant"]
        )
        prefix = len(all_keys)
        all_keys.extend(keys)
        row_parts.append(rows + prefix)
        primitive_parts.append(primitive_rows)
        weight_parts.append(weights)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("contribution caches contain duplicate render-row keys")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": WEIGHTS_ARTIFACT_TYPE,
        "primitive_ids": primitive_ids,
        "render_row_keys": all_keys,
        "render_row_index": torch.cat(row_parts).contiguous(),
        "primitive_row_index": torch.cat(primitive_parts).contiguous(),
        "contribution_weights": torch.cat(weight_parts).contiguous(),
        "query_bank": queries,
        "scalar_compositor_manifest": {
            "path": compositor["path"], "sha256": compositor["sha256"]
        },
        "source_contribution_caches": [record for _, record in loaded],
        "metadata": {
            **{key: False for key in _QUERY_FREE_FLAGS},
            "target_blind": True,
            "benchmark_targets_or_metrics_used": False,
            "frozen_before_materialization": True,
            "query_bank_source": "target_blind_replay_only",
            "query_source": query_source,
            "selected_variant": compositor["selected_variant"],
            "weights_semantics": (
                "frozen_selected_normalized_sparse_contribution_weights"
            ),
            "primitive_input_cache": primitive_record,
            "replay_builder_implementation": file_record(Path(__file__).resolve()),
            "render_row_selection": (
                "first_256_nonempty_rows_per_source_in_frozen_source_order"
            ),
            "render_rows": len(all_keys),
            "sparse_hits": int(sum(value.numel() for value in weight_parts)),
        },
    }
    write_torch_noclobber(output, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": f"{WEIGHTS_ARTIFACT_TYPE}_report",
        "output": file_record(output),
        "primitive_input_cache": primitive_record,
        "scalar_compositor_manifest": compositor,
        "source_contribution_caches": [record for _, record in loaded],
        "selected_variant": compositor["selected_variant"],
        "render_rows": len(all_keys),
        "sparse_hits": int(payload["contribution_weights"].numel()),
        "query_rows": int(queries.shape[0]),
        "target_blind": True,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive-input-cache", required=True)
    parser.add_argument("--scalar-compositor-manifest", required=True)
    parser.add_argument("--contribution-cache", action="append", required=True)
    parser.add_argument("--fit-text-bank", required=True)
    parser.add_argument("--fit-text-bank-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = build(build_arg_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
