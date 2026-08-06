#!/usr/bin/env python3
"""Source-only probe for a frozen V2 fallback on V3 deployment regions.

This diagnostic deliberately does not run an evaluator or load labels.  It
replays the current V3 region selection, removes support-fill tokens, restores
the compact field's raw RADIO gauge, and applies the immutable accepted V2
readout plus the official SigLIP2 summary head.  The resulting descriptors and
query responses are compared with existing source caches only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV3
from radio_gs.interfaces.surface_region_selection import (
    surface_region_contract_from_metadata,
)
from radio_gs.interfaces.surface_region_summary import (
    surface_region_effective_reliability_v3,
    surface_region_geometry_v3,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.build_surface_region_semantic_cache import (
    completion_primary_valid,
    expand_surface_region_v3_batch_at_radius,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_surface_region_summary_readout_v2,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


def _load_raw_resume_features(
    resume_dir: Path,
    *,
    expected_rows: int,
    expected_contract_sha256: str,
) -> torch.Tensor:
    """Load exact field-decode RADIO shards and require contiguous coverage."""

    shards: list[tuple[int, int, Path]] = []
    for path in Path(resume_dir).glob("radio_*.pt"):
        fields = path.stem.split("_")
        if len(fields) != 3:
            continue
        try:
            start, stop = int(fields[1]), int(fields[2])
        except ValueError:
            continue
        terminal = path.with_suffix(".complete.json")
        if not terminal.is_file():
            raise ValueError(f"raw RADIO shard lacks completion receipt: {path}")
        shards.append((start, stop, path))
    shards.sort()
    if not shards:
        raise FileNotFoundError(f"no raw RADIO shards found in {resume_dir}")
    cursor = 0
    result = torch.empty(expected_rows, 1280, dtype=torch.float16)
    for start, stop, path in shards:
        if start != cursor or stop <= start or stop > expected_rows:
            raise ValueError("raw RADIO resume shards are not exact contiguous coverage")
        receipt, _, _ = load_json_object(
            path.with_suffix(".complete.json"),
            label="raw RADIO shard receipt",
        )
        expected_receipt = {
            "artifact_type": "surface_semantic_resume_batch",
            "contract_sha256": str(expected_contract_sha256),
            "dtype": "torch.float16",
            "phase": "radio",
            "schema_version": 1,
            "shape": [stop - start, 1280],
            "start": start,
            "stop": stop,
            "tensor": file_record(path),
        }
        if receipt != expected_receipt:
            raise ValueError(f"raw RADIO shard receipt differs: {path}")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.shape != (
            stop - start,
            1280,
        ) or value.dtype != torch.float16:
            raise ValueError(f"malformed raw RADIO shard: {path}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite raw RADIO shard: {path}")
        result[start:stop] = value
        cursor = stop
    if cursor != expected_rows:
        raise ValueError(
            f"raw RADIO resume coverage stops at {cursor}, expected {expected_rows}"
        )
    return result


def _v3_contract_from_checkpoint(payload: Mapping) -> SurfaceRegionContractV3:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("V3 checkpoint lacks provenance")
    region = provenance.get("region_contract")
    if not isinstance(region, Mapping):
        raise ValueError("V3 checkpoint lacks its region contract")
    contract = surface_region_contract_from_metadata(
        {
            **dict(provenance),
            "region_contract_version": provenance.get(
                "region_contract_version", region.get("version")
            ),
        }
    )
    if not isinstance(contract, SurfaceRegionContractV3):
        raise ValueError("candidate checkpoint does not bind SurfaceRegionContractV3")
    if str(provenance.get("region_contract_sha256", "")) != contract.digest:
        raise ValueError("V3 checkpoint contract digest differs")
    return contract


def _sample_rows(row_count: int, sample_count: int, seed: int) -> torch.Tensor:
    if row_count <= 0 or sample_count <= 0:
        raise ValueError("row and sample counts must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(row_count, generator=generator)[: min(row_count, sample_count)]


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = torch.as_tensor(left).double().reshape(-1)
    y = torch.as_tensor(right).double().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x @ y / denominator.clamp_min(1e-12)).item())


def _rankdata(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(torch.as_tensor(value).reshape(-1), stable=True)
    rank = torch.empty_like(order, dtype=torch.float64)
    rank[order] = torch.arange(len(order), dtype=torch.float64)
    return rank


def _top_fraction_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    fraction: float = 0.01,
) -> float:
    x = torch.as_tensor(left).reshape(-1)
    y = torch.as_tensor(right).reshape(-1)
    count = max(1, int(round(len(x) * float(fraction))))
    a = set(torch.topk(x, count).indices.tolist())
    b = set(torch.topk(y, count).indices.tolist())
    return len(a & b) / count


def _response_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    """Compare [rows, scales, queries] response tensors without labels."""

    left = torch.as_tensor(candidate).float().cpu()
    right = torch.as_tensor(reference).float().cpu()
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("response tensors must align as [rows,scales,queries]")
    pearsons, spearmans, overlaps = [], [], []
    for scale in range(left.shape[1]):
        for query in range(left.shape[2]):
            x, y = left[:, scale, query], right[:, scale, query]
            pearsons.append(_pearson(x, y))
            spearmans.append(_pearson(_rankdata(x), _rankdata(y)))
            overlaps.append(_top_fraction_overlap(x, y))
    return {
        "pearson_flat": _pearson(left, right),
        "pearson_per_scale_query_mean": float(torch.tensor(pearsons).mean()),
        "pearson_per_scale_query_min": float(torch.tensor(pearsons).min()),
        "spearman_per_scale_query_mean": float(torch.tensor(spearmans).mean()),
        "top1pct_overlap_mean": float(torch.tensor(overlaps).mean()),
        "mean_absolute_error": float((left - right).abs().mean()),
        "root_mean_square_error": float((left - right).square().mean().sqrt()),
        "scale_argmax_agreement": float(
            (left.argmax(dim=1) == right.argmax(dim=1)).float().mean()
        ),
    }


def _descriptor_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    left = F.normalize(torch.as_tensor(candidate).float().cpu(), dim=-1)
    right = F.normalize(torch.as_tensor(reference).float().cpu(), dim=-1)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("descriptors must align as [rows,scales,channels]")
    cosine = (left * right).sum(-1)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(torch.quantile(cosine.reshape(-1), 0.05)),
        "cosine_min": float(cosine.min()),
    }


def _v3_to_v2_adapter_inputs(
    geometry_v3: torch.Tensor,
    token_mask: torch.Tensor,
    support_fill_mask: torch.Tensor,
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose the exact V2 geometry prefix while removing V3-only fill."""

    geometry = torch.as_tensor(geometry_v3)
    mask = torch.as_tensor(token_mask, device=geometry.device).bool()
    fill = torch.as_tensor(support_fill_mask, device=geometry.device).bool()
    confidence = torch.as_tensor(reliability, device=geometry.device)
    if geometry.ndim != 3 or geometry.shape[-1] != 16:
        raise ValueError("V3 geometry must align as [batch,tokens,16]")
    if mask.shape != geometry.shape[:2] or fill.shape != mask.shape:
        raise ValueError("V3 token/fill masks must align with geometry")
    if confidence.shape != (*mask.shape, 1):
        raise ValueError("V3 reliability must align as [batch,tokens,1]")
    if bool((fill & ~mask).any()):
        raise ValueError("V3 support-fill must be a subset of token_mask")
    v2_mask = mask & ~fill
    if not bool(v2_mask.any(dim=1).all()):
        raise ValueError("V2 fallback must retain at least one non-fill token")
    geometry_v2 = geometry[..., :14].masked_fill(~v2_mask[..., None], 0.0)
    v2_reliability = confidence.masked_fill(~v2_mask[..., None], 0.0)
    return geometry_v2, v2_mask, v2_reliability


def _load_reference_descriptors(
    path: Path,
    *,
    global_rows: torch.Tensor,
    sample_local_rows: torch.Tensor,
) -> torch.Tensor:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=None,
        map_location="cpu",
        label="reference descriptor cache",
    )
    if not torch.equal(torch.as_tensor(payload.get("global_rows")).long(), global_rows):
        raise ValueError(f"reference descriptor global rows differ: {path}")
    descriptors = torch.as_tensor(payload.get("features_by_scale"))
    if descriptors.ndim != 3 or descriptors.shape[0] != len(global_rows):
        raise ValueError(f"malformed reference descriptors: {path}")
    return descriptors[sample_local_rows].float().clone()


def _load_sampled_score_cache(
    path: Path,
    *,
    sample_global_rows: torch.Tensor,
    expected_query_ids: Sequence[str],
) -> torch.Tensor:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=None,
        map_location="cpu",
        label="reference query score cache",
    )
    query_ids = [str(value) for value in payload.get("query_ids", [])]
    if query_ids != [str(value) for value in expected_query_ids]:
        raise ValueError(f"reference score query axis differs: {path}")
    scores = torch.as_tensor(payload.get("query_scores"))
    if scores.ndim != 3:
        raise ValueError(f"malformed reference score cache: {path}")
    return scores[sample_global_rows].float().clone()


def _load_text_axis(path: Path, query_ids: Sequence[str]) -> torch.Tensor:
    payload, _, _ = load_torch_mapping(
        path,
        expected_sha256=None,
        map_location="cpu",
        label="frozen text embedding cache",
    )
    available = [str(value) for value in payload.get("queries", [])]
    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in query_ids if name not in lookup]
    if missing:
        raise ValueError(f"text embedding cache misses queries: {missing}")
    embeddings = torch.as_tensor(payload.get("embeddings")).float()
    return F.normalize(
        embeddings[torch.tensor([lookup[name] for name in query_ids])],
        dim=-1,
    )


def _score_descriptors(
    descriptors: torch.Tensor,
    text: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    values = torch.as_tensor(descriptors)
    queries = F.normalize(torch.as_tensor(text).float().cpu(), dim=-1)
    if values.ndim != 3 or values.shape[-1] != queries.shape[-1]:
        raise ValueError("descriptor and text axes do not align")
    chunks = []
    for start in range(0, len(values), int(chunk_size)):
        unit = F.normalize(values[start : start + chunk_size].float(), dim=-1)
        chunks.append((unit @ queries.T).half())
    return torch.cat(chunks)


def _full_output_paths(output_dir: Path) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    return {
        "descriptor": root / "figurines_v3_regions_accepted_v2_raw_descriptor.pt",
        "positive_scores": root
        / "figurines_v3_regions_accepted_v2_raw_positive_scores.pt",
        "negative_scores": root
        / "figurines_v3_regions_accepted_v2_raw_negative_scores.pt",
    }


def _write_full_outputs(
    *,
    paths: Mapping[str, Path],
    candidate: torch.Tensor,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    contract: SurfaceRegionContractV3,
    global_rows: torch.Tensor,
    output_valid: torch.Tensor,
    primary_valid: torch.Tensor | None,
    v3_descriptor_path: Path,
    accepted_v2_checkpoint: Path,
    v3_checkpoint: Path,
    support_graph: Path,
    mpr_cache: Path,
    radio_checkpoint: Path,
    raw_resume_contract: Path,
    score_geometry_authority: Mapping,
) -> dict[str, dict[str, str]]:
    """Publish full source caches with explicit mixed-contract authority."""

    if len(candidate) != len(global_rows):
        raise ValueError("full output requires every V3 support-graph row")
    if positive_scores.shape[:2] != candidate.shape[:2] or (
        negative_scores.shape[:2] != candidate.shape[:2]
    ):
        raise ValueError("full output scores and descriptors do not align")
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"full fallback output already exists: {path}")
    source, _, _ = load_torch_mapping(
        v3_descriptor_path,
        expected_sha256=None,
        map_location="cpu",
        label="V3 descriptor structural authority",
    )
    if not torch.equal(torch.as_tensor(source.get("global_rows")).long(), global_rows):
        raise ValueError("V3 descriptor and fallback global rows differ")
    if not torch.equal(torch.as_tensor(source.get("valid")).bool(), output_valid):
        raise ValueError("V3 descriptor and fallback validity differ")
    if not torch.equal(
        torch.as_tensor(score_geometry_authority.get("xyz")).float(),
        torch.as_tensor(source.get("xyz")).float(),
    ) or not torch.equal(
        torch.as_tensor(score_geometry_authority.get("valid")).bool(), output_valid
    ):
        raise ValueError("historical accepted score geometry authority differs")
    source_metadata = source.get("metadata")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("V3 descriptor lacks metadata")
    authorities = {
        "v3_descriptor": file_record(v3_descriptor_path),
        "v3_checkpoint": file_record(v3_checkpoint),
        "accepted_v2_checkpoint": file_record(accepted_v2_checkpoint),
        "support_graph": file_record(support_graph),
        "mpr_cache": file_record(mpr_cache),
        "radio_checkpoint": file_record(radio_checkpoint),
        "raw_radio_resume_contract": file_record(raw_resume_contract),
    }
    metadata = {
        **dict(source_metadata),
        "schema_version": 6,
        # Keep the canonical source identity required by the official score
        # materializer.  The adapter is still query-independent and ends in
        # the immutable SurfaceRegion readout plus official summary head.
        "source": "canonical_radio_surface_region_readout",
        "construction": (
            "exact_v3_regions_remove_support_fill_restore_raw_radio_"
            "accepted_v2_readout_official_summary_head"
        ),
        "surface_region_adapter": {
            "name": "v3_regions_to_accepted_v2_raw_fallback_v1",
            "region_selection": "exact_frozen_v3",
            "support_fill": "excluded_from_v2_token_mask",
            "geometry": "surface_region_geometry_v3_prefix_0_13_exact",
            "radio_gauge": "legacy_raw_exact_field_decode_resume_shards",
            "readout": "immutable_accepted_surface_region_summary_readout_v2",
        },
        "readout_checkpoint": str(accepted_v2_checkpoint.resolve()),
        "readout_checkpoint_sha256": authorities["accepted_v2_checkpoint"][
            "sha256"
        ],
        "bridge_checkpoint_sha256": authorities["accepted_v2_checkpoint"][
            "sha256"
        ],
        "radio_feature_normalization": "legacy_raw",
        "radio_feature_normalization_authority": (
            "exact_field_decode_resume_shards_with_contract_receipts"
        ),
        "support_fill_semantics": "excluded_from_frozen_v2_fallback_token_mask",
        "geometry_adapter": "surface_region_geometry_v3_prefix_0_13_exact",
        "region_contract": contract.to_dict(),
        "region_contract_version": contract.version,
        "region_contract_sha256": contract.digest,
        "adapter_authorities": authorities,
        "query_set_invariant": True,
        "text_queries_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_evaluator_run": False,
        "capacity_diagnostic_only": True,
        "official_score_materialization_route": (
            "materialize_lerf_multiscale_query_score_cache.py"
        ),
    }
    descriptors = torch.as_tensor(candidate).half().contiguous()
    mean_descriptor = F.normalize(descriptors.float().mean(1), dim=-1).half()
    descriptor_payload = {
        "xyz": torch.as_tensor(source["xyz"]),
        "features": mean_descriptor,
        "summary_features": mean_descriptor,
        "global_rows": global_rows,
        "features_by_scale": descriptors,
        "valid": output_valid,
        "metadata": metadata,
    }
    if primary_valid is not None:
        descriptor_payload["primary_valid"] = primary_valid
    if "semantic_confidence" in source:
        descriptor_payload["semantic_confidence"] = source["semantic_confidence"]
    paths["descriptor"].parent.mkdir(parents=True, exist_ok=True)
    write_torch_noclobber(paths["descriptor"], descriptor_payload)
    descriptor_record = file_record(paths["descriptor"])

    geometry_fingerprint = dict(
        score_geometry_authority.get("geometry_fingerprint", {})
    )
    score_authority = {
        "artifact_type": "v3_regions_accepted_v2_raw_source_score_authority",
        "schema_version": 1,
        "descriptor": descriptor_record,
        "adapter_authorities": authorities,
        "region_contract_sha256": contract.digest,
        "support_fill": "excluded",
        "radio_gauge": "legacy_raw",
        "geometry": "v3_prefix_0_13",
        "score_formula": "l2_normalize(descriptor) @ l2_normalize(text).T",
        "score_dtype": "torch.float16",
        "scale_reduction_applied": False,
        "threshold_applied": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_evaluator_run": False,
        "text_queries_opened": True,
        "capacity_diagnostic_only": True,
        "frozen_evaluator_input_authorized": False,
        "role": "diagnostic_only_official_materializer_output_is_authoritative",
    }
    for key, scores, query_ids in (
        ("positive_scores", positive_scores, positive_ids),
        ("negative_scores", negative_scores, negative_ids),
    ):
        dense = torch.zeros(
            len(output_valid), len(contract.radii_m), len(query_ids), dtype=torch.float16
        )
        dense[global_rows] = torch.as_tensor(scores).half()
        payload = {
            "version": 2,
            "contract": "radio_gs.ours_lerf_direct3d_multiscale_query_scores.v2",
            "query_scores": dense,
            "query_ids": list(query_ids),
            "scale_ids": [str(value) for value in contract.radii_m],
            "scale_radii_m": list(contract.radii_m),
            "xyz": torch.as_tensor(source["xyz"]),
            "valid": output_valid,
            "geometry_fingerprint": geometry_fingerprint,
            "field_checkpoint_sha256": score_geometry_authority.get(
                "field_checkpoint_sha256"
            ),
            "readout_checkpoint_sha256": authorities["accepted_v2_checkpoint"][
                "sha256"
            ],
            "renderer_geometry_checkpoint_sha256": score_geometry_authority.get(
                "renderer_geometry_checkpoint_sha256"
            ),
            "authority": {**score_authority, "query_role": key},
        }
        write_torch_noclobber(paths[key], payload)
    records = {name: file_record(path) for name, path in paths.items()}
    for name, record in records.items():
        write_frozen_json(
            paths[name].with_suffix(paths[name].suffix + ".json"),
            {
                "schema_version": 1,
                "artifact_type": "v3_regions_accepted_v2_raw_full_output_receipt",
                "role": name,
                "artifact": record,
                "descriptor": descriptor_record,
                "adapter_authorities": authorities,
                "region_contract_sha256": contract.digest,
                "benchmark_evaluator_run": False,
                "benchmark_labels_opened": False,
            },
        )
    return records


@torch.inference_mode()
def _candidate_descriptors(
    *,
    contract: SurfaceRegionContractV3,
    support: PrimitiveSupportGraph,
    prepared_graph,
    xyz: torch.Tensor,
    local_scale: torch.Tensor,
    reliability: torch.Tensor,
    primary_local: torch.Tensor | None,
    raw_radio: torch.Tensor,
    sample_local_rows: torch.Tensor,
    readout: torch.nn.Module,
    head: SigLIP2SummaryHead,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict]:
    outputs: list[torch.Tensor] = []
    support_counts: dict[str, dict[str, float]] = {}
    for radius in contract.radii_m:
        scale_output: list[torch.Tensor] = []
        active_counts, fill_counts = [], []
        for start in range(0, len(sample_local_rows), batch_size):
            centers = sample_local_rows[start : start + batch_size]
            selections = expand_surface_region_v3_batch_at_radius(
                contract,
                support,
                xyz,
                centers,
                float(radius),
                prepared_graph=prepared_graph,
                primary_local=primary_local,
            )
            rows = torch.stack([value.rows for value in selections])
            token_mask = torch.stack([value.token_mask for value in selections])
            core = torch.stack([value.core_mask for value in selections])
            context = torch.stack([value.context_mask for value in selections])
            support_fill = torch.stack(
                [value.support_fill_mask for value in selections]
            )
            recovery = torch.stack(
                [value.recovery_distance for value in selections]
            )
            anchor = torch.tensor(
                [value.anchor_index for value in selections], dtype=torch.long
            )
            primitive_reliability = reliability[rows, None]
            effective_reliability = surface_region_effective_reliability_v3(
                primitive_reliability,
                recovery,
                float(radius),
                support_fill_mask=support_fill,
                token_mask=token_mask,
            )
            raw_tokens = raw_radio[rows].float()
            raw_norm = torch.linalg.vector_norm(raw_tokens, dim=-1, keepdim=True)
            geometry_v3 = surface_region_geometry_v3(
                xyz[rows],
                local_scale[rows, None].expand(-1, -1, 3),
                effective_reliability,
                float(radius),
                raw_radio_l2_norm=raw_norm,
                anchor_index=anchor,
                core_mask=core,
                context_mask=context,
                support_fill_mask=support_fill,
                token_mask=token_mask,
            )
            # The fallback is intentionally an adapter, not a silent V3
            # mutation: preserve the frozen V3 selection and geometry, remove
            # only support-fill, and expose the exact V2 prefix layout.
            geometry_v2, v2_mask, effective_reliability = (
                _v3_to_v2_adapter_inputs(
                    geometry_v3,
                    token_mask,
                    support_fill,
                    effective_reliability,
                )
            )
            active_counts.extend(v2_mask.sum(1).tolist())
            fill_counts.extend(support_fill.sum(1).tolist())
            summary = readout(
                raw_tokens.to(device),
                geometry_v2.to(device),
                anchor_index=anchor.to(device),
                token_mask=v2_mask.to(device),
                reliability=effective_reliability.to(device),
            )
            descriptor = F.normalize(head(summary), dim=-1)
            scale_output.append(descriptor.cpu())
        outputs.append(torch.cat(scale_output))
        counts = torch.tensor(active_counts, dtype=torch.float32)
        fills = torch.tensor(fill_counts, dtype=torch.float32)
        support_counts[str(radius)] = {
            "non_fill_token_mean": float(counts.mean()),
            "non_fill_token_p05": float(torch.quantile(counts, 0.05)),
            "support_fill_token_mean": float(fills.mean()),
            "support_fill_fraction_of_original": float(
                (fills / (fills + counts).clamp_min(1)).mean()
            ),
            "rows_with_support_fill_fraction": float((fills > 0).float().mean()),
        }
    return torch.stack(outputs, dim=1), support_counts


@torch.inference_mode()
def probe(args: argparse.Namespace) -> dict:
    if int(args.batch_size) <= 0:
        raise ValueError("batch size must be positive")
    full_output_dir = str(getattr(args, "full_output_dir", "")).strip()
    full_paths = (
        _full_output_paths(Path(full_output_dir)) if full_output_dir else None
    )
    if full_paths is not None:
        collisions = [str(path) for path in full_paths.values() if path.exists()]
        collisions.extend(
            str(path.with_suffix(path.suffix + ".json"))
            for path in full_paths.values()
            if path.with_suffix(path.suffix + ".json").exists()
        )
        if collisions:
            raise FileExistsError(f"full fallback output collisions: {collisions}")
    graph, _, _ = load_torch_mapping(
        args.support_graph,
        expected_sha256=args.support_graph_sha256 or None,
        map_location="cpu",
        label="surface support graph",
    )
    mpr, _, _ = load_torch_mapping(
        args.mpr_cache,
        expected_sha256=args.mpr_cache_sha256 or None,
        map_location="cpu",
        label="completed MPR cache",
    )
    v3_payload, _, _ = load_torch_mapping(
        args.v3_checkpoint,
        expected_sha256=args.v3_checkpoint_sha256 or None,
        map_location="cpu",
        label="V3 readout checkpoint",
    )
    contract = _v3_contract_from_checkpoint(v3_payload)
    resume_contract, _, _ = load_json_object(
        Path(args.raw_radio_resume_dir) / "contract.json",
        label="raw RADIO resume contract",
    )
    resume_inputs = resume_contract.get("inputs")
    if not isinstance(resume_inputs, Mapping):
        raise ValueError("raw RADIO resume contract lacks input authorities")
    expected_resume_inputs = {
        "mpr": sha256_file(args.mpr_cache),
        "support_graph": sha256_file(args.support_graph),
        "readout": sha256_file(args.v3_checkpoint),
        "radio": sha256_file(args.radio_checkpoint),
    }
    for key, digest in expected_resume_inputs.items():
        record = resume_inputs.get(key)
        if not isinstance(record, Mapping) or str(record.get("sha256", "")) != digest:
            raise ValueError(f"raw RADIO resume {key} authority differs")
    if (
        resume_contract.get("canonical_radio_source", "field_decode")
        != "field_decode"
        or resume_contract.get("radio_feature_normalization")
        != "l2_direction_plus_log_raw_norm_v1"
        or resume_contract.get("region_contract_sha256") != contract.digest
    ):
        raise ValueError("raw RADIO resume gauge or V3 contract differs")
    resume_contract_sha256 = canonical_json_sha256(resume_contract)
    readout, v2_payload, _, _ = load_surface_region_summary_readout_v2(
        args.accepted_v2_checkpoint,
        expected_sha256=args.accepted_v2_checkpoint_sha256 or None,
        map_location="cpu",
    )
    if v2_payload.get("architecture", {}).get("name") != (
        "surface_region_summary_readout_v2"
    ):
        raise ValueError("accepted fallback is not a V2 readout")
    global_rows = torch.as_tensor(graph["global_rows"]).long()
    xyz = torch.as_tensor(graph["xyz"]).float()
    xyz_global = torch.as_tensor(mpr["xyz"]).float()
    if not torch.equal(xyz, xyz_global[global_rows]):
        raise ValueError("support graph and MPR geometry differ")
    output_valid = torch.zeros(len(xyz_global), dtype=torch.bool)
    output_valid[global_rows] = True
    primary_valid = completion_primary_valid(mpr, output_valid)
    primary_local = (
        None if primary_valid is None else primary_valid[global_rows]
    )
    reliability_source = torch.as_tensor(mpr["reliability"]).float()[global_rows]
    reliability = reliability_source[:, :2].clamp_min(1e-6).log().mean(-1).exp()
    reliability[(reliability_source[:, :2] <= 0).any(-1)] = 0.0
    raw_radio = _load_raw_resume_features(
        Path(args.raw_radio_resume_dir),
        expected_rows=len(global_rows),
        expected_contract_sha256=resume_contract_sha256,
    )
    sample_local_rows = (
        torch.arange(len(global_rows), dtype=torch.long)
        if full_paths is not None
        else _sample_rows(len(global_rows), int(args.sample_count), int(args.seed))
    )
    sample_global_rows = global_rows[sample_local_rows]
    support = PrimitiveSupportGraph(
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        raw_affinity=graph["raw_affinity"],
        local_sigma=graph["local_sigma"],
        num_nodes=len(xyz),
        edge_channels=graph.get("edge_channels", {}),
    )
    prepared_graph = contract.prepare_graph(support, xyz)
    device = torch.device(args.device)
    readout = readout.to(device).eval().requires_grad_(False)
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        args.radio_checkpoint,
        **(
            {"expected_sha256": args.radio_checkpoint_sha256}
            if args.radio_checkpoint_sha256
            else {}
        ),
    ).to(device).eval().requires_grad_(False)
    candidate, support_stats = _candidate_descriptors(
        contract=contract,
        support=support,
        prepared_graph=prepared_graph,
        xyz=xyz,
        local_scale=torch.as_tensor(graph["local_sigma"]).float().clamp_min(1e-4),
        reliability=reliability,
        primary_local=primary_local,
        raw_radio=raw_radio,
        sample_local_rows=sample_local_rows,
        readout=readout,
        head=head,
        device=device,
        batch_size=int(args.batch_size),
    )
    references = {
        "accepted_v2_raw_physical": (
            Path(args.accepted_descriptor),
            Path(args.accepted_positive_scores),
            Path(args.accepted_negative_scores),
        ),
        "full_mpr_teacher_physical": (
            Path(args.full_mpr_descriptor),
            Path(args.full_mpr_positive_scores),
            Path(args.full_mpr_negative_scores),
        ),
        "current_v3": (
            Path(args.current_v3_descriptor),
            Path(args.current_v3_positive_scores),
            Path(args.current_v3_negative_scores),
        ),
    }
    accepted_positive_payload = torch.load(
        args.accepted_positive_scores, map_location="cpu", weights_only=False
    )
    accepted_negative_payload = torch.load(
        args.accepted_negative_scores, map_location="cpu", weights_only=False
    )
    positive_ids = [str(value) for value in accepted_positive_payload["query_ids"]]
    negative_ids = [str(value) for value in accepted_negative_payload["query_ids"]]
    positive_text = _load_text_axis(Path(args.positive_text_cache), positive_ids)
    negative_text = _load_text_axis(Path(args.negative_text_cache), negative_ids)
    candidate_positive = _score_descriptors(candidate, positive_text).float()
    candidate_negative = _score_descriptors(candidate, negative_text).float()
    candidate_probability = torch.sigmoid(
        10.0 * (candidate_positive - candidate_negative.max(-1, keepdim=True).values)
    )
    comparisons: dict[str, dict] = {}
    cached_response_consistency: dict[str, dict[str, float]] = {}
    for name, (descriptor_path, positive_path, negative_path) in references.items():
        descriptor = _load_reference_descriptors(
            descriptor_path,
            global_rows=global_rows,
            sample_local_rows=sample_local_rows,
        )
        positive = _load_sampled_score_cache(
            positive_path,
            sample_global_rows=sample_global_rows,
            expected_query_ids=positive_ids,
        )
        negative = _load_sampled_score_cache(
            negative_path,
            sample_global_rows=sample_global_rows,
            expected_query_ids=negative_ids,
        )
        descriptor_unit = F.normalize(descriptor.float(), dim=-1)
        recomputed_positive = descriptor_unit @ positive_text.T
        recomputed_negative = descriptor_unit @ negative_text.T
        cached_response_consistency[name] = {
            "positive_max_absolute_error": float(
                (positive - recomputed_positive).abs().max()
            ),
            "negative_max_absolute_error": float(
                (negative - recomputed_negative).abs().max()
            ),
        }
        probability = torch.sigmoid(
            10.0 * (positive - negative.max(-1, keepdim=True).values)
        )
        comparisons[name] = {
            "descriptor": _descriptor_comparison(candidate, descriptor),
            "positive_cosine": _response_comparison(candidate_positive, positive),
            "hard_negative_cosine": _response_comparison(
                candidate_negative.max(-1, keepdim=True).values.expand_as(
                    candidate_positive
                ),
                negative.max(-1, keepdim=True).values.expand_as(positive),
            ),
            "margin": _response_comparison(
                candidate_positive
                - candidate_negative.max(-1, keepdim=True).values,
                positive - negative.max(-1, keepdim=True).values,
            ),
            "relevancy_probability": _response_comparison(
                candidate_probability, probability
            ),
        }
    full_output_records = None
    if full_paths is not None:
        full_output_records = _write_full_outputs(
            paths=full_paths,
            candidate=candidate,
            positive_scores=candidate_positive,
            negative_scores=candidate_negative,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            contract=contract,
            global_rows=global_rows,
            output_valid=output_valid,
            primary_valid=primary_valid,
            v3_descriptor_path=Path(args.current_v3_descriptor),
            accepted_v2_checkpoint=Path(args.accepted_v2_checkpoint),
            v3_checkpoint=Path(args.v3_checkpoint),
            support_graph=Path(args.support_graph),
            mpr_cache=Path(args.mpr_cache),
            radio_checkpoint=Path(args.radio_checkpoint),
            raw_resume_contract=(
                Path(args.raw_radio_resume_dir) / "contract.json"
            ),
            score_geometry_authority=accepted_positive_payload,
        )
    report = {
        "schema_version": 1,
        "artifact_type": "surface_region_v3_to_accepted_v2_source_only_probe",
        "status": "complete_without_evaluator_or_labels",
        "method": {
            "region_selection": "exact_frozen_v3_tiered_eligible_expansion",
            "support_fill": "excluded_from_v2_token_mask",
            "radio_gauge": "legacy_raw_exact_field_decode_resume_shards",
            "geometry": "surface_region_geometry_v3_prefix_indices_0_through_13",
            "readout": "immutable_accepted_surface_region_summary_readout_v2",
            "projection": "official_frozen_c_radio_siglip2_g_summary_head",
            "benchmark_evaluator_run": False,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
        },
        "sample": {
            "population_local_rows": len(global_rows),
            "sample_rows": len(sample_local_rows),
            "seed": int(args.seed),
            "full_population_materialized": full_paths is not None,
            "sample_local_rows_sha256": _tensor_sha256(sample_local_rows),
            "sample_global_rows_sha256": _tensor_sha256(sample_global_rows),
        },
        "authorities": {
            "support_graph": file_record(args.support_graph),
            "mpr_cache": file_record(args.mpr_cache),
            "v3_checkpoint": file_record(args.v3_checkpoint),
            "accepted_v2_checkpoint": file_record(args.accepted_v2_checkpoint),
            "radio_checkpoint": file_record(args.radio_checkpoint),
            "raw_radio_resume_contract": file_record(
                Path(args.raw_radio_resume_dir) / "contract.json"
            ),
            "references": {
                name: {
                    "descriptor": file_record(paths[0]),
                    "positive_scores": file_record(paths[1]),
                    "negative_scores": file_record(paths[2]),
                }
                for name, paths in references.items()
            },
        },
        "contract": {
            "version": contract.version,
            "sha256": contract.digest,
            "radii_m": list(contract.radii_m),
        },
        "raw_radio_gauge_statistics": {
            "l2_norm_mean": float(torch.linalg.vector_norm(raw_radio.float(), dim=-1).mean()),
            "l2_norm_p05": float(
                torch.quantile(torch.linalg.vector_norm(raw_radio.float(), dim=-1), 0.05)
            ),
            "l2_norm_p95": float(
                torch.quantile(torch.linalg.vector_norm(raw_radio.float(), dim=-1), 0.95)
            ),
            "unit_norm_fraction_at_1e_minus_3": float(
                (
                    torch.linalg.vector_norm(raw_radio.float(), dim=-1).sub(1).abs()
                    <= 1e-3
                ).float().mean()
            ),
        },
        "candidate_descriptor_sha256": _tensor_sha256(candidate.half()),
        "support_statistics": support_stats,
        "cached_response_consistency": cached_response_consistency,
        "comparisons": comparisons,
        **(
            {"full_output_records": full_output_records}
            if full_output_records is not None
            else {}
        ),
    }
    output = Path(args.output).resolve()
    write_frozen_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--support-graph-sha256", default="")
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--mpr-cache-sha256", default="")
    parser.add_argument("--v3-checkpoint", required=True)
    parser.add_argument("--v3-checkpoint-sha256", default="")
    parser.add_argument("--accepted-v2-checkpoint", required=True)
    parser.add_argument("--accepted-v2-checkpoint-sha256", default="")
    parser.add_argument("--radio-checkpoint", required=True)
    parser.add_argument("--radio-checkpoint-sha256", default="")
    parser.add_argument("--raw-radio-resume-dir", required=True)
    parser.add_argument("--accepted-descriptor", required=True)
    parser.add_argument("--accepted-positive-scores", required=True)
    parser.add_argument("--accepted-negative-scores", required=True)
    parser.add_argument("--full-mpr-descriptor", required=True)
    parser.add_argument("--full-mpr-positive-scores", required=True)
    parser.add_argument("--full-mpr-negative-scores", required=True)
    parser.add_argument("--current-v3-descriptor", required=True)
    parser.add_argument("--current-v3-positive-scores", required=True)
    parser.add_argument("--current-v3-negative-scores", required=True)
    parser.add_argument("--positive-text-cache", required=True)
    parser.add_argument("--negative-text-cache", required=True)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--full-output-dir",
        default="",
        help=(
            "Optional no-clobber directory for complete descriptor, positive "
            "score, and canonical-negative score caches. Supplying it forces "
            "the probe population to every valid V3 support-graph row."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    report = probe(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
