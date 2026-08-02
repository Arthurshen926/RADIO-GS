#!/usr/bin/env python3
"""Materialize frozen ScanNet-OVS scores on optimized Gaussian rows.

The default route is the promoted canonical-mpr-v3 field.  Graph-observed rows
use the frozen h128 surface-region readout at 0.20/0.40/0.70 m and max their
independent query cosines.  Rows with no graph evidence use the exact canonical
primitive feature through the same frozen official SigLIP2 summary head.  The
legacy v67 hybrid route is retained only behind an explicit diagnostic method
family.  Neither route opens a ScanNet mesh, label PLY, pseudo-GT, benchmark
prediction, or metric artifact.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.interfaces.surface_region_summary import (
    SurfaceRegionSummaryReadout,
    surface_region_geometry,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.scripts.eval_ours_scannet_vala_gaussian_protocol import (
    ARTIFACT_TYPE,
    CLASS_ORDER_SHA256,
    CANONICAL_MAINLINE_NAME,
    CANONICAL_MAINLINE_SHA256,
    CANONICAL_METHOD_FREEZE_NAME,
    CANONICAL_METHOD_FREEZE_SHA256,
    CANONICAL_READOUT_SHA256,
    CANONICAL_REGION_RADII_M,
    CANONICAL_TOTALITY_CONTRACT,
    CURRENT_MATERIALIZER_CONTRACT,
    CURRENT_METHOD_FAMILY,
    EXTERNAL_PROTOCOL_FREEZE_ID,
    EXTERNAL_PROTOCOL_FREEZE_SHA256,
    EXTERNAL_PROTOCOL_FREEZE_TASK,
    EXTERNAL_PROTOCOL_REGISTRY_ROW,
    LEGACY_MATERIALIZER_CONTRACT,
    LEGACY_METHOD_FAMILY,
    OFFICIAL_RADIO_SHA256,
    PAPER_CLASS_IDS,
    PAPER_CLASS_NAMES,
    PREDICTION_DOMAIN,
    PROTOCOL_CONTRACT,
    QUERY_CLASS_ORDER_SHA256,
    QUERY_TEXT_SHA256,
    ROW_ORDER,
    SCHEMA_VERSION,
    SEMANTIC_READOUT,
    SPATIAL_TRANSFER,
    _tensor_sha256,
    validate_ours_gaussian_semantic_score_cache,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _decode_gaussian_indices_1280,
)
from radio_gs.scripts.build_surface_region_semantic_cache import (
    _adjacency,
    two_hop_physical_regions,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_torch_mapping,
    sha256_file,
    write_frozen_json,
    write_torch_noclobber,
)


MATERIALIZER_CONTRACT = LEGACY_MATERIALIZER_CONTRACT
LEGACY_SCORE_FORMULA = (
    "l2_normalize(siglip2_summary_head(codec(field.gaussian_row)))) "
    "@ l2_normalize(text_embedding).T"
)
SIGLIP2_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
COMPACT_FEATURE_KEY = "features"
PROTOCOL_FREEZE_ID = "evaluation_protocols_20260801_v1"
PROTOCOL_FREEZE_TASK = "concept_scannet_ovs_vala_paper8"
PROTOCOL_REGISTRY_ROW = "scannet_ovs_vala_compatibility_20260611"
PROTOCOL_FREEZE_SHA256 = (
    "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916"
)
CANONICAL_SCORE_FORMULA = (
    "l2_normalize(canonical_mpr_v3_surface_region_descriptor) @ "
    "l2_normalize(exact_split19_text_embedding).T"
)
DEFAULT_CANONICAL_METHOD_FREEZE = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/canonical_mpr_v3_evaluation_freeze_20260716.yaml"
)
DEFAULT_CANONICAL_MAINLINE = (
    Path(__file__).resolve().parents[2] / "paper/artifacts/canonical_mainline_v3.yaml"
)
DEFAULT_PROTOCOL_FREEZE = (
    Path(__file__).resolve().parents[2]
    / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"
)


def _canonical_output(path: str | Path) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    raw.parent.mkdir(parents=True, exist_ok=True)
    return raw.parent.resolve(strict=True) / raw.name


def _preflight_outputs(output: Path, receipt: Path) -> None:
    if output == receipt:
        raise ValueError("score cache and receipt outputs must be distinct")
    for path in (output, receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable output already exists: {path}")


def _geometry_authority_device(
    method_family: str, scoring_device: torch.device
) -> torch.device:
    """Canonicalize activated geometry on the evaluator's CPU domain.

    The checkpoint stores raw log-scale, quaternion, and opacity parameters.
    Activating them with CUDA kernels can differ from the CPU evaluator by a
    few ULPs even though the raw rows are identical.  Current-mainline score
    caches therefore load geometry on CPU, while semantic field/readout
    scoring remains on ``scoring_device``.  The legacy diagnostic keeps its
    historical device because it queries the geometry model for semantics.
    """

    if method_family == CURRENT_METHOD_FAMILY:
        return torch.device("cpu")
    return torch.device(scoring_device)


def _require_same_file_record(
    before: Mapping[str, str], after: Mapping[str, str], *, label: str
) -> None:
    if dict(before) != dict(after):
        raise ValueError(f"{label} changed while semantic scores were materialized")


def load_frozen_split19_text_bank(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, str, Path]:
    """Load, validate, and normalize one immutable official SigLIP2 bank."""

    payload, digest, source = load_torch_mapping(
        path,
        map_location="cpu",
        label="frozen ScanNet split19 SigLIP2 text bank",
    )
    if [str(value) for value in payload.get("queries", [])] != list(PAPER_CLASS_NAMES):
        raise ValueError("text bank query/order differs from frozen ScanNet split19")
    model_name = payload.get("model_name", payload.get("model"))
    if model_name != SIGLIP2_MODEL_NAME:
        raise ValueError("text bank is not the frozen official SigLIP2-Giant model")
    if payload.get("text_encoder") not in (None, "siglip2"):
        raise ValueError("text bank encoder differs from SigLIP2")
    if payload.get("exact_scannet_nyu40") is not True:
        raise ValueError("text bank is not marked as exact ScanNet NYU40 authority")
    if payload.get("head_fixed") is not True:
        raise ValueError("text bank is not bound to the fixed SigLIP2 text head")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or not embeddings.is_floating_point():
        raise ValueError("text bank embeddings must be a floating tensor")
    embeddings = embeddings.detach().cpu().float().contiguous()
    if tuple(embeddings.shape) != (len(PAPER_CLASS_IDS), 1536):
        raise ValueError("text bank embeddings must have frozen shape [19,1536]")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("text bank embeddings contain NaN or infinity")
    norms = embeddings.norm(dim=1)
    if bool((norms <= 1e-8).any()):
        raise ValueError("text bank contains a zero-norm query embedding")
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)):
        raise ValueError("text bank query embeddings must already be unit normalized")
    return F.normalize(embeddings, dim=-1).to(device), digest, source


@torch.no_grad()
def compute_gaussian_semantic_scores(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    summary_head: torch.nn.Module,
    text_embeddings: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Compute calibration-free split19 cosine scores in checkpoint row order."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    count = int(model.get_xyz().shape[0])
    if count <= 0:
        raise ValueError("geometry checkpoint contains no Gaussian rows")
    text = torch.as_tensor(text_embeddings, device=device).float()
    if tuple(text.shape) != (len(PAPER_CLASS_IDS), 1536):
        raise ValueError("text_embeddings must have frozen shape [19,1536]")
    text = F.normalize(text, dim=-1)
    parts: list[torch.Tensor] = []
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        rows = torch.arange(start, end, device=device, dtype=torch.long)
        decoded = _decode_gaussian_indices_1280(
            model,
            codec,
            rows,
            points_xyz=None,
            return_aux=False,
            compact_feature_key=COMPACT_FEATURE_KEY,
        )
        if not isinstance(decoded, torch.Tensor) or tuple(decoded.shape) != (
            end - start,
            1280,
        ):
            observed = tuple(decoded.shape) if isinstance(decoded, torch.Tensor) else type(decoded)
            raise ValueError(f"decoded Gaussian features must be [B,1280], got {observed}")
        visual = summary_head(decoded.float().unsqueeze(0)).squeeze(0)
        if tuple(visual.shape) != (end - start, 1536):
            raise ValueError("SigLIP2 summary head output must be [B,1536]")
        if not bool(torch.isfinite(visual).all()):
            raise ValueError("SigLIP2 summary features contain NaN or infinity")
        parts.append((F.normalize(visual.float(), dim=-1) @ text.T).cpu())
    scores = torch.cat(parts, dim=0).float().contiguous()
    if tuple(scores.shape) != (count, len(PAPER_CLASS_IDS)):
        raise AssertionError("internal semantic score shape differs")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("semantic scores contain NaN or infinity")
    return scores


def _require_argument(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"--{name.replace('_', '-')} is required for this method family")
    return value


def _require_canonical_field_architecture(payload: Mapping[str, Any]) -> None:
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("canonical field lacks architecture authority")
    expected = {
        "feature_dim": 1280,
        "coefficient_dim": 256,
        "local_dim": 128,
        "coarse_dim": 0,
        "spatial_hash": None,
        "fusion_reliability": True,
        "hidden_dim": 192,
        "use_fusion": True,
        "trainable_basis": True,
        "trainable_statistics": False,
    }
    for key, value in expected.items():
        if architecture.get(key) != value:
            raise ValueError(
                f"canonical field architecture.{key} differs from canonical-mpr-v3"
            )
    if payload.get("benchmark_masks_opened") is not False:
        raise ValueError("canonical field benchmark-mask authority differs")
    if payload.get("text_queries_opened") is not False:
        raise ValueError("canonical field text-query authority differs")


@torch.no_grad()
def load_canonical_mpr_v3_authority(
    args: argparse.Namespace,
    *,
    expected_xyz: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    Mapping[str, Any],
    Mapping[str, Any],
    torch.nn.Module,
    torch.nn.Module,
    torch.Tensor,
    dict[str, dict[str, str]],
]:
    """Load the frozen field/readout graph and bind it to renderer rows."""

    field_path = _require_argument(args, "field_checkpoint")
    graph_path = _require_argument(args, "support_graph")
    readout_path = _require_argument(args, "readout_checkpoint")
    radio_path = _require_argument(args, "radio_checkpoint")
    canonical_freeze_path = (
        getattr(args, "canonical_method_freeze", None)
        or str(DEFAULT_CANONICAL_METHOD_FREEZE)
    )
    mainline_path = (
        getattr(args, "canonical_mainline", None) or str(DEFAULT_CANONICAL_MAINLINE)
    )
    records = {
        "canonical_field_source": file_record(field_path),
        "support_graph_source": file_record(graph_path),
        "surface_region_readout_source": file_record(readout_path),
        "official_radio_source": file_record(radio_path),
        "canonical_method_freeze": file_record(canonical_freeze_path),
        "canonical_mainline": file_record(mainline_path),
    }
    for role, expected_sha in {
        "surface_region_readout_source": CANONICAL_READOUT_SHA256,
        "official_radio_source": OFFICIAL_RADIO_SHA256,
        "canonical_method_freeze": CANONICAL_METHOD_FREEZE_SHA256,
        "canonical_mainline": CANONICAL_MAINLINE_SHA256,
    }.items():
        if records[role]["sha256"] != expected_sha:
            raise ValueError(f"{role} SHA256 differs from canonical-mpr-v3 freeze")

    field, field_payload = load_canonical_field_checkpoint(
        field_path,
        map_location="cpu",
        expected_sha256=records["canonical_field_source"]["sha256"],
    )
    _require_canonical_field_architecture(field_payload)
    reference_xyz = torch.as_tensor(expected_xyz).detach().cpu().float().contiguous()
    if field.num_gaussians != reference_xyz.shape[0]:
        raise ValueError("canonical field row count differs from renderer geometry")
    mpr_path = field_payload.get("mpr_cache")
    if not isinstance(mpr_path, str) or not mpr_path:
        raise ValueError("canonical field lacks MPR source binding")
    records["mpr_source"] = file_record(mpr_path)
    mpr, _, _ = load_torch_mapping(
        mpr_path,
        expected_sha256=records["mpr_source"]["sha256"],
        map_location="cpu",
        label="canonical-mpr-v3 MPR source",
    )
    mpr_xyz = torch.as_tensor(mpr.get("xyz")).detach().cpu().float().contiguous()
    mpr_valid = torch.as_tensor(mpr.get("valid")).detach().cpu()
    if not torch.equal(mpr_xyz, reference_xyz):
        raise ValueError("canonical field MPR xyz/row-order differs from renderer geometry")
    if mpr_valid.dtype != torch.bool or tuple(mpr_valid.shape) != (len(reference_xyz),):
        raise ValueError("canonical field MPR valid rows are malformed")
    if not bool(mpr_valid.any()):
        raise ValueError("canonical field MPR has no graph-observed rows")

    graph, _, _ = load_torch_mapping(
        graph_path,
        expected_sha256=records["support_graph_source"]["sha256"],
        map_location="cpu",
        label="canonical-mpr-v3 support graph",
    )
    graph_rows = torch.as_tensor(graph.get("global_rows")).detach().cpu().long()
    graph_xyz = torch.as_tensor(graph.get("xyz")).detach().cpu().float().contiguous()
    if (
        graph_rows.ndim != 1
        or graph_rows.numel() != torch.unique(graph_rows).numel()
        or not torch.equal(graph_rows, torch.where(mpr_valid)[0])
        or not torch.equal(graph_xyz, reference_xyz[graph_rows])
    ):
        raise ValueError("canonical support graph rows differ from MPR observed rows")

    readout, readout_payload = SurfaceRegionSummaryReadout.from_checkpoint(
        readout_path, map_location="cpu"
    )
    architecture = readout_payload.get("architecture")
    provenance = readout_payload.get("provenance")
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("name") != "surface_region_summary_readout_v1"
        or architecture.get("feature_dim") != 1280
        or architecture.get("geometry_dim") != 12
        or architecture.get("hidden_dim") != 128
        or architecture.get("digest")
        != "2ea0107b914ed2e4498893e75475f5597f55b6a1a2d6bdf7bcc5aefab2545a88"
    ):
        raise ValueError("surface-region readout architecture differs from frozen h128")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("frozen") is not True
        or provenance.get("uses_benchmark_scenes") is not False
        or provenance.get("uses_benchmark_test_vocabulary") is not False
        or provenance.get("scene_disjoint") is not True
        or provenance.get("custom_text_projection") is not False
    ):
        raise ValueError("surface-region readout provenance differs from canonical-mpr-v3")
    head = SigLIP2SummaryHead.from_radio_checkpoint(
        radio_path, expected_sha256=OFFICIAL_RADIO_SHA256
    )
    for module in (field, readout, head):
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return field, mpr, graph, readout, head, mpr_valid, records


@torch.no_grad()
def compute_canonical_mpr_v3_semantic_scores(
    field: torch.nn.Module,
    mpr: Mapping[str, Any],
    graph: Mapping[str, Any],
    readout: torch.nn.Module,
    summary_head: torch.nn.Module,
    text_embeddings: torch.Tensor,
    *,
    device: torch.device,
    radio_batch_size: int,
    semantic_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply frozen h128 regions on observed rows and exact-field totality elsewhere."""

    if radio_batch_size <= 0 or semantic_batch_size <= 0:
        raise ValueError("canonical materialization batch sizes must be positive")
    count = int(field.num_gaussians)
    graph_rows = torch.as_tensor(graph["global_rows"]).detach().cpu().long()
    graph_xyz = torch.as_tensor(graph["xyz"]).detach().cpu().float().contiguous()
    observed = torch.zeros(count, dtype=torch.bool)
    observed[graph_rows] = True
    text = F.normalize(
        torch.as_tensor(text_embeddings, device=device).float(), dim=-1
    )
    if tuple(text.shape) != (len(PAPER_CLASS_IDS), 1536):
        raise ValueError("text embeddings must be [19,1536]")

    radio = torch.empty(len(graph_rows), 1280, dtype=torch.float16, device=device)
    for start in range(0, len(graph_rows), radio_batch_size):
        stop = min(start + radio_batch_size, len(graph_rows))
        values = field.radio_features(graph_rows[start:stop].to(device)).float()
        if tuple(values.shape) != (stop - start, 1280) or not bool(
            torch.isfinite(values).all()
        ):
            raise ValueError("canonical field observed RADIO features are malformed")
        radio[start:stop] = values.half()

    reliability_source = torch.as_tensor(mpr.get("reliability")).float()[graph_rows]
    if reliability_source.ndim != 2 or reliability_source.shape[1] < 2:
        raise ValueError("canonical MPR reliability needs coverage/agreement channels")
    reliability = reliability_source[:, :2].clamp_min(1e-6).log().mean(-1).exp()
    reliability[(reliability_source[:, :2] <= 0).any(-1)] = 0.0
    reliability = reliability.to(device)
    local_scale = torch.as_tensor(graph["local_sigma"]).float().clamp_min(1e-4).to(device)
    xyz_device = graph_xyz.to(device)
    adjacency = _adjacency(dict(graph), 16).to(device)
    scores = torch.empty(count, len(PAPER_CLASS_IDS), dtype=torch.float32)
    for start in range(0, len(graph_rows), semantic_batch_size):
        stop = min(start + semantic_batch_size, len(graph_rows))
        centers = torch.arange(start, stop, device=device)
        scale_scores = []
        for radius in CANONICAL_REGION_RADII_M:
            rows, mask = two_hop_physical_regions(
                centers, adjacency, xyz_device, float(radius)
            )
            token_reliability = reliability[rows, None]
            geometry = surface_region_geometry(
                xyz_device[rows],
                local_scale[rows, None].expand(-1, -1, 3),
                torch.ones_like(token_reliability),
                token_reliability,
                float(radius),
                token_mask=mask,
            )
            summary = readout(
                radio[rows],
                geometry,
                token_mask=mask,
                reliability=token_reliability,
            )
            projected = summary_head(summary[:, None])[:, 0].float()
            if not bool(torch.isfinite(projected).all()) or bool(
                (projected.norm(dim=1) <= 1e-8).any()
            ):
                raise ValueError("canonical surface-region summaries are invalid")
            descriptor = F.normalize(projected, dim=-1)
            scale_scores.append(descriptor @ text.T)
        observed_scores = torch.stack(scale_scores, dim=1).amax(dim=1)
        scores[graph_rows[start:stop]] = observed_scores.cpu()

    fallback_rows = torch.where(~observed)[0]
    for start in range(0, len(fallback_rows), radio_batch_size):
        stop = min(start + radio_batch_size, len(fallback_rows))
        rows = fallback_rows[start:stop]
        primitive_radio = field.radio_features(rows.to(device)).float()
        if tuple(primitive_radio.shape) != (stop - start, 1280) or not bool(
            torch.isfinite(primitive_radio).all()
        ):
            raise ValueError("canonical no-evidence RADIO features are malformed")
        projected = summary_head(primitive_radio[:, None])[:, 0].float()
        if not bool(torch.isfinite(projected).all()) or bool(
            (projected.norm(dim=1) <= 1e-8).any()
        ):
            raise ValueError("canonical no-evidence summaries are invalid")
        descriptor = F.normalize(projected, dim=-1)
        scores[rows] = (descriptor @ text.T).cpu()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("canonical semantic scores contain NaN or infinity")
    return scores.contiguous(), observed


def build_cache_payload(
    *,
    scene_id: str,
    xyz: torch.Tensor,
    scale: torch.Tensor,
    quaternion: torch.Tensor,
    opacity: torch.Tensor,
    semantic_scores: torch.Tensor,
    geometry_checkpoint_record: Mapping[str, str],
    config_record: Mapping[str, str],
    query_source_record: Mapping[str, str],
    protocol_freeze_record: Mapping[str, str],
    producer_source_record: Mapping[str, str],
    method_family: str,
    semantic_source_record: Mapping[str, str],
    authority_metadata: Mapping[str, Any],
    summary_head_record: Mapping[str, str] | None = None,
    region_observed: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the exact payload consumed by the frozen ScanNet evaluator."""

    if not scene_id:
        raise ValueError("scene_id must be non-empty")
    tensors = {
        "xyz": torch.as_tensor(xyz).detach().cpu().float().contiguous(),
        "scale": torch.as_tensor(scale).detach().cpu().float().contiguous(),
        "quaternion": torch.as_tensor(quaternion).detach().cpu().float().contiguous(),
        "opacity": torch.as_tensor(opacity).detach().cpu().float().reshape(-1).contiguous(),
        "valid": torch.ones(int(torch.as_tensor(xyz).shape[0]), dtype=torch.bool),
        "semantic_scores": torch.as_tensor(semantic_scores)
        .detach()
        .cpu()
        .float()
        .contiguous(),
    }
    if region_observed is not None:
        observed = torch.as_tensor(region_observed).detach().cpu().contiguous()
        if observed.dtype != torch.bool or tuple(observed.shape) != (
            tensors["xyz"].shape[0],
        ):
            raise ValueError("region_observed must be row-aligned bool")
        tensors["region_observed"] = observed
    geometry_record = dict(geometry_checkpoint_record)
    query_record = dict(query_source_record)
    freeze_record = dict(protocol_freeze_record)
    producer_record = dict(producer_source_record)
    semantic_record = dict(semantic_source_record)
    authority = dict(authority_metadata)
    protected = {
        "protocol_contract",
        "method_family",
        "protocol_freeze_id",
        "protocol_freeze_task",
        "protocol_registry_row",
        "protocol_freeze",
        "protocol_freeze_sha256",
        "producer_source",
        "producer_source_sha256",
        "geometry_checkpoint",
        "geometry_checkpoint_sha256",
        "semantic_source",
        "semantic_source_sha256",
        "query_source",
        "query_source_sha256",
    }
    if protected.intersection(authority):
        raise ValueError("authority_metadata attempts to override protected cache fields")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        **tensors,
        "class_ids": list(PAPER_CLASS_IDS),
        "class_names": list(PAPER_CLASS_NAMES),
        "query_ids": list(PAPER_CLASS_NAMES),
        "metadata": {
            "protocol_contract": PROTOCOL_CONTRACT,
            "method_family": method_family,
            "protocol_freeze_id": EXTERNAL_PROTOCOL_FREEZE_ID,
            "protocol_freeze_task": EXTERNAL_PROTOCOL_FREEZE_TASK,
            "protocol_registry_row": EXTERNAL_PROTOCOL_REGISTRY_ROW,
            "protocol_freeze": freeze_record,
            "protocol_freeze_sha256": freeze_record["sha256"],
            "producer_source": producer_record,
            "producer_source_sha256": producer_record["sha256"],
            "scene_id": scene_id,
            "prediction_domain": PREDICTION_DOMAIN,
            "row_order": ROW_ORDER,
            "semantic_readout": SEMANTIC_READOUT,
            "spatial_transfer": SPATIAL_TRANSFER,
            "mesh_vertices_used": False,
            "knn_used": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_labels_opened": False,
            "benchmark_metrics_opened": False,
            "compact_feature_key": COMPACT_FEATURE_KEY,
            "gaussian_query_position": "optimized_gaussian_center",
            "geometry_checkpoint": geometry_record,
            "geometry_checkpoint_sha256": geometry_record["sha256"],
            "semantic_source": semantic_record,
            "semantic_source_sha256": semantic_record["sha256"],
            "config_source": dict(config_record),
            "query_source": query_record,
            "query_source_sha256": query_record["sha256"],
            **(
                {"summary_head_source": dict(summary_head_record)}
                if summary_head_record is not None
                else {}
            ),
            "query_text_sha256": QUERY_TEXT_SHA256,
            "class_order_sha256": CLASS_ORDER_SHA256,
            "query_class_order_sha256": QUERY_CLASS_ORDER_SHA256,
            "row_tensor_sha256": {
                key: _tensor_sha256(tensor) for key, tensor in tensors.items()
            },
            **authority,
        },
    }


def materialize(args: argparse.Namespace) -> tuple[Path, Path]:
    output = _canonical_output(args.output)
    receipt = _canonical_output(
        args.receipt or output.with_name(f"{output.name}.receipt.json")
    )
    _preflight_outputs(output, receipt)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    method_family = getattr(args, "method_family", CURRENT_METHOD_FAMILY)
    if method_family not in {CURRENT_METHOD_FAMILY, LEGACY_METHOD_FAMILY}:
        raise ValueError("unsupported method family")
    config_record = file_record(_require_argument(args, "config"))
    geometry_record = file_record(_require_argument(args, "geometry_checkpoint"))
    query_record = file_record(_require_argument(args, "query_source"))
    protocol_freeze_record = file_record(
        getattr(args, "protocol_freeze", None) or DEFAULT_PROTOCOL_FREEZE
    )
    if protocol_freeze_record["sha256"] != PROTOCOL_FREEZE_SHA256:
        raise ValueError("protocol freeze SHA256 differs from the frozen 20260801 authority")
    producer_source_record = file_record(Path(__file__))
    config = load_config(str(Path(args.config).expanduser().resolve(strict=True)))
    configured_scene = str(getattr(config, "scene", ""))
    if configured_scene and configured_scene != args.scene:
        raise ValueError(
            f"config scene differs: expected {args.scene!r}, got {configured_scene!r}"
        )
    geometry_device = _geometry_authority_device(method_family, device)
    model, codec = _build_hybrid_model(
        config,
        str(Path(args.geometry_checkpoint).expanduser().resolve(strict=True)),
        geometry_device,
    )
    text, query_digest, query_source = load_frozen_split19_text_bank(
        args.query_source,
        device=device,
    )
    if query_digest != query_record["sha256"] or str(query_source) != query_record["path"]:
        raise ValueError("query source identity differs across authority reads")
    xyz = model.get_xyz().detach().cpu().float().contiguous()
    scale = model.get_scaling().detach().cpu().float().contiguous()
    quaternion = model.get_rotation().detach().cpu().float().contiguous()
    opacity = model.get_opacity().detach().cpu().float().reshape(-1).contiguous()
    source_records: dict[str, dict[str, str]] = {}
    summary_head_record = None
    region_observed = None
    if method_family == LEGACY_METHOD_FAMILY:
        summary_head_path = _require_argument(args, "summary_head_weights")
        summary_head_record = file_record(summary_head_path)
        summary_head = SigLIP2SummaryHead.from_extracted_weights(
            str(Path(summary_head_path).expanduser().resolve(strict=True))
        ).to(device).eval()
        scores = compute_gaussian_semantic_scores(
            model,
            codec,
            summary_head,
            text,
            device=device,
            chunk_size=args.chunk_size,
        )
        semantic_source_record = geometry_record
        authority_metadata = {
            "materializer_contract": LEGACY_MATERIALIZER_CONTRACT,
            "diagnostic_only": True,
            "compact_feature_key": COMPACT_FEATURE_KEY,
            "gaussian_query_position": "optimized_gaussian_center",
            "score_formula": LEGACY_SCORE_FORMULA,
            "query_set_calibration": False,
            "logit_calibration": "none",
            "logit_smoothing": "none",
        }
        source_records["summary_head_source"] = summary_head_record
    else:
        (
            field,
            mpr,
            graph,
            readout,
            summary_head,
            _mpr_valid,
            canonical_records,
        ) = load_canonical_mpr_v3_authority(
            args,
            expected_xyz=xyz,
            device=device,
        )
        scores, region_observed = compute_canonical_mpr_v3_semantic_scores(
            field,
            mpr,
            graph,
            readout,
            summary_head,
            text,
            device=device,
            radio_batch_size=getattr(args, "radio_batch_size", args.chunk_size),
            semantic_batch_size=getattr(
                args, "semantic_batch_size", min(args.chunk_size, 256)
            ),
        )
        semantic_source_record = canonical_records["canonical_field_source"]
        source_records.update(canonical_records)
        authority_metadata = {
            "materializer_contract": CURRENT_MATERIALIZER_CONTRACT,
            "diagnostic_only": False,
            "canonical_mainline_name": CANONICAL_MAINLINE_NAME,
            "canonical_mainline_sha256": CANONICAL_MAINLINE_SHA256,
            "canonical_method_freeze_name": CANONICAL_METHOD_FREEZE_NAME,
            "canonical_method_freeze_sha256": CANONICAL_METHOD_FREEZE_SHA256,
            "surface_region_readout_sha256": CANONICAL_READOUT_SHA256,
            "official_radio_checkpoint_sha256": OFFICIAL_RADIO_SHA256,
            "region_radii_m": list(CANONICAL_REGION_RADII_M),
            "score_formula": CANONICAL_SCORE_FORMULA,
            "query_set_calibration": False,
            "logit_calibration": "none",
            "logit_smoothing": "none",
            "canonical_field_geometry_row_match": True,
            "region_graph_geometry_row_match": True,
            "geometry_authority_activation_device": "cpu",
            "region_scale_aggregation": (
                "max_independent_cosine_over_0.20_0.40_0.70"
            ),
            "totality_semantics": (
                "graph_observed_surface_region_h128_else_exact_canonical_field_primitive"
            ),
            "totality_contract": CANONICAL_TOTALITY_CONTRACT,
            "no_evidence_fallback": (
                "canonical_field_primitive_official_summary_head_independent_cosine"
            ),
            "region_observed_count": int(region_observed.sum()),
            "no_evidence_fallback_count": int((~region_observed).sum()),
            **canonical_records,
        }
    payload = build_cache_payload(
        scene_id=args.scene,
        xyz=xyz,
        scale=scale,
        quaternion=quaternion,
        opacity=opacity,
        semantic_scores=scores,
        geometry_checkpoint_record=geometry_record,
        config_record=config_record,
        query_source_record=query_record,
        protocol_freeze_record=protocol_freeze_record,
        producer_source_record=producer_source_record,
        method_family=method_family,
        semantic_source_record=semantic_source_record,
        authority_metadata=authority_metadata,
        summary_head_record=summary_head_record,
        region_observed=region_observed,
    )
    validate_ours_gaussian_semantic_score_cache(
        payload,
        expected_scene_id=args.scene,
        expected_xyz=xyz,
        expected_scale=scale,
        expected_quaternion=quaternion,
        expected_opacity=opacity,
        expected_valid=torch.ones(xyz.shape[0], dtype=torch.bool),
        expected_geometry_checkpoint_sha256=geometry_record["sha256"],
        expected_method_family=method_family,
    )

    stable_records = [
        ("config", config_record),
        ("geometry checkpoint", geometry_record),
        ("query source", query_record),
        ("protocol freeze", protocol_freeze_record),
        ("producer source", producer_source_record),
        *[(role, record) for role, record in source_records.items()],
    ]
    for label, before in stable_records:
        _require_same_file_record(before, file_record(before["path"]), label=label)
    write_torch_noclobber(output, payload)
    cache_record = file_record(output)
    write_frozen_json(
        receipt,
        {
            "status": "complete_immutable_gaussian_semantic_score_cache",
            "method_family": method_family,
            "materializer_contract": authority_metadata["materializer_contract"],
            "protocol_freeze_id": PROTOCOL_FREEZE_ID,
            "protocol_freeze_task": PROTOCOL_FREEZE_TASK,
            "protocol_registry_row": PROTOCOL_REGISTRY_ROW,
            "scene_id": args.scene,
            "num_gaussians": int(xyz.shape[0]),
            "num_classes": len(PAPER_CLASS_IDS),
            "semantic_score_cache": cache_record,
            "geometry_checkpoint": geometry_record,
            "config_source": config_record,
            "query_source": query_record,
            **(
                {"summary_head_source": summary_head_record}
                if summary_head_record is not None
                else {}
            ),
            **source_records,
            "protocol_freeze": protocol_freeze_record,
            "producer_source": producer_source_record,
            "row_tensor_sha256": payload["metadata"]["row_tensor_sha256"],
        },
    )
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-family",
        choices=(CURRENT_METHOD_FAMILY, LEGACY_METHOD_FAMILY),
        default=CURRENT_METHOD_FAMILY,
        help="Canonical mainline by default; legacy hybrid is diagnostic-only",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--query-source", required=True)
    parser.add_argument("--summary-head-weights", default=None)
    parser.add_argument("--field-checkpoint", default=None)
    parser.add_argument("--support-graph", default=None)
    parser.add_argument("--readout-checkpoint", default=None)
    parser.add_argument("--radio-checkpoint", default=None)
    parser.add_argument(
        "--canonical-method-freeze",
        default=str(DEFAULT_CANONICAL_METHOD_FREEZE),
    )
    parser.add_argument(
        "--canonical-mainline",
        default=str(DEFAULT_CANONICAL_MAINLINE),
    )
    parser.add_argument(
        "--protocol-freeze",
        default=str(DEFAULT_PROTOCOL_FREEZE),
        help="Immutable 20260801 protocol-freeze authority",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--radio-batch-size", type=int, default=4096)
    parser.add_argument("--semantic-batch-size", type=int, default=256)
    args = parser.parse_args()
    output, receipt = materialize(args)
    print(f"Saved score cache: {output}")
    print(f"Saved receipt: {receipt}")
    print(f"Score cache SHA256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
