#!/usr/bin/env python3
"""Evaluate Ours on frozen ScanNet-OVS in the optimized-Gaussian domain.

This is a CPU-only adapter for an already-materialized semantic score cache.
It deliberately does not reuse the historical mesh-vertex / kNN8 readout.
Geometry is loaded from the Ours checkpoint, every semantic-score row is
matched bitwise to that optimized-Gaussian row, and the final metrics reuse
the VALA Gaussian-domain pseudo-GT and weighted metric implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import (
    _build_hybrid_model,
    _read_label_ply,
)
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    _load_or_build_pseudo_gt,
    volume_weighted_split_metrics,
)
from radio_gs.scripts.eval_vala_scannet_checkpoint_gaussian_protocol import (
    _resolve_cohort_scenes,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_torch_mapping,
    sha256_file,
    validate_file_record,
    write_frozen_json,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "radio_gs_scannet_ovs_gaussian_semantic_score_cache"
PROTOCOL_CONTRACT = "radio_gs.ours_scannet_ovs_gaussian_scores.v1"
CURRENT_METHOD_FAMILY = "canonical_mpr_v3"
LEGACY_METHOD_FAMILY = "legacy_hybrid_v67"
CURRENT_MATERIALIZER_CONTRACT = (
    "radio_gs.ours_scannet_ovs_canonical_mpr_v3_score_materializer.v2"
)
LEGACY_MATERIALIZER_CONTRACT = (
    "radio_gs.ours_scannet_ovs_legacy_hybrid_score_materializer.v1"
)
PREDICTION_DOMAIN = "optimized_gaussian_checkpoint_rows"
ROW_ORDER = "zero_based_geometry_checkpoint_row_order"
SEMANTIC_READOUT = "direct_per_gaussian_class_argmax"
SPATIAL_TRANSFER = "none"
PAPER_CLASS_IDS = tuple(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
PAPER_CLASS_NAMES = tuple(NYU40_ID_TO_NAME[class_id] for class_id in PAPER_CLASS_IDS)
QUERY_TEXT_SHA256 = canonical_json_sha256(list(PAPER_CLASS_NAMES))
CLASS_ORDER_SHA256 = canonical_json_sha256(list(PAPER_CLASS_IDS))
QUERY_CLASS_ORDER_SHA256 = canonical_json_sha256(
    [
        {"class_id": class_id, "query": class_name}
        for class_id, class_name in zip(PAPER_CLASS_IDS, PAPER_CLASS_NAMES)
    ]
)
EXTERNAL_PROTOCOL_FREEZE_ID = "evaluation_protocols_20260801_v1"
EXTERNAL_PROTOCOL_FREEZE_TASK = "concept_scannet_ovs_vala_paper8"
EXTERNAL_PROTOCOL_REGISTRY_ROW = "scannet_ovs_vala_compatibility_20260611"
EXTERNAL_PROTOCOL_FREEZE_SHA256 = (
    "af91f0861d3a15354063579e78f64898801c41f2543d1cf9b352a0a123820916"
)
CANONICAL_MAINLINE_NAME = "canonical-mpr-v3"
CANONICAL_MAINLINE_SHA256 = (
    "3d6d36ab91ef3b2c406a6fad02bc9bf085a680690f84e35896751d414d57dbae"
)
CANONICAL_METHOD_FREEZE_NAME = "canonical-mpr-v3-evaluation-freeze"
CANONICAL_METHOD_FREEZE_SHA256 = (
    "9ea6fc8d79ee11ae2ccb3b6ef738580983cae406f586db61833182780587c009"
)
CANONICAL_READOUT_SHA256 = (
    "06c1d9c3bfaf674de54fa1167aa2d156e183e0a751c87d0e7c09733928464b5b"
)
OFFICIAL_RADIO_SHA256 = (
    "bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9"
)
CANONICAL_REGION_RADII_M = [0.2, 0.4, 0.7]
CANONICAL_TOTALITY_CONTRACT = (
    "radio_gs.canonical_mpr_v3_gaussian_semantic_totality.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and bytes in stable CPU row order."""
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.layout != torch.strided:
        raise ValueError("row authority tensors must have strided layout")
    digest = __import__("hashlib").sha256()
    digest.update(
        json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.ndim == 0:
        digest.update(tensor.contiguous().numpy().tobytes(order="C"))
    else:
        for start in range(0, int(tensor.shape[0]), 4096):
            digest.update(
                tensor[start : start + 4096]
                .contiguous()
                .numpy()
                .tobytes(order="C")
            )
    return digest.hexdigest()


def _as_floating_tensor(value: object, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{label} must be a floating tensor")
    tensor = value.detach().cpu().contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains NaN or infinity")
    return tensor


def _validate_row_tensor(
    payload: Mapping[str, Any],
    key: str,
    expected: torch.Tensor,
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    observed = _as_floating_tensor(payload.get(key), label=f"semantic cache {key}")
    if tuple(observed.shape) != shape:
        raise ValueError(
            f"semantic cache {key} must be row-aligned {list(shape)}, "
            f"got {list(observed.shape)}"
        )
    reference = torch.as_tensor(expected).detach().cpu().contiguous()
    if observed.dtype != reference.dtype or not torch.equal(observed, reference):
        raise ValueError(f"semantic cache {key}/row-order differs from geometry checkpoint")
    return observed


def validate_ours_gaussian_semantic_score_cache(
    payload: Mapping[str, Any],
    *,
    expected_scene_id: str,
    expected_xyz: torch.Tensor,
    expected_scale: torch.Tensor,
    expected_quaternion: torch.Tensor,
    expected_opacity: torch.Tensor,
    expected_valid: torch.Tensor,
    expected_geometry_checkpoint_sha256: str,
    expected_method_family: str = CURRENT_METHOD_FAMILY,
) -> dict[str, Any]:
    """Fail closed unless one score cache exactly binds production GS rows."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("semantic cache schema_version differs")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("unsupported Ours semantic score cache artifact type")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("semantic cache metadata must be a mapping")
    exact_metadata = {
        "protocol_contract": PROTOCOL_CONTRACT,
        "scene_id": expected_scene_id,
        "prediction_domain": PREDICTION_DOMAIN,
        "row_order": ROW_ORDER,
        "semantic_readout": SEMANTIC_READOUT,
        "spatial_transfer": SPATIAL_TRANSFER,
        "mesh_vertices_used": False,
        "knn_used": False,
        "query_text_sha256": QUERY_TEXT_SHA256,
        "class_order_sha256": CLASS_ORDER_SHA256,
        "query_class_order_sha256": QUERY_CLASS_ORDER_SHA256,
        "method_family": expected_method_family,
        "protocol_freeze_id": EXTERNAL_PROTOCOL_FREEZE_ID,
        "protocol_freeze_task": EXTERNAL_PROTOCOL_FREEZE_TASK,
        "protocol_registry_row": EXTERNAL_PROTOCOL_REGISTRY_ROW,
        "protocol_freeze_sha256": EXTERNAL_PROTOCOL_FREEZE_SHA256,
    }
    for key, expected in exact_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"semantic cache metadata.{key} differs: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    if expected_method_family == CURRENT_METHOD_FAMILY:
        canonical_metadata = {
            "materializer_contract": CURRENT_MATERIALIZER_CONTRACT,
            "canonical_mainline_name": CANONICAL_MAINLINE_NAME,
            "canonical_mainline_sha256": CANONICAL_MAINLINE_SHA256,
            "canonical_method_freeze_name": CANONICAL_METHOD_FREEZE_NAME,
            "canonical_method_freeze_sha256": CANONICAL_METHOD_FREEZE_SHA256,
            "surface_region_readout_sha256": CANONICAL_READOUT_SHA256,
            "official_radio_checkpoint_sha256": OFFICIAL_RADIO_SHA256,
            "region_radii_m": CANONICAL_REGION_RADII_M,
            "score_formula": (
                "l2_normalize(canonical_mpr_v3_surface_region_descriptor) @ "
                "l2_normalize(exact_split19_text_embedding).T"
            ),
            "query_set_calibration": False,
            "logit_calibration": "none",
            "logit_smoothing": "none",
            "canonical_field_geometry_row_match": True,
            "region_graph_geometry_row_match": True,
            "region_scale_aggregation": "max_independent_cosine_over_0.20_0.40_0.70",
            "totality_semantics": (
                "graph_observed_surface_region_h128_else_exact_canonical_field_primitive"
            ),
            "totality_contract": CANONICAL_TOTALITY_CONTRACT,
            "no_evidence_fallback": (
                "canonical_field_primitive_official_summary_head_independent_cosine"
            ),
            "diagnostic_only": False,
        }
        for key, expected in canonical_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"semantic cache metadata.{key} differs: "
                    f"expected {expected!r}, got {metadata.get(key)!r}"
                )
    elif expected_method_family == LEGACY_METHOD_FAMILY:
        legacy_metadata = {
            "materializer_contract": LEGACY_MATERIALIZER_CONTRACT,
            "diagnostic_only": True,
        }
        for key, expected in legacy_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"semantic cache metadata.{key} differs: "
                    f"expected {expected!r}, got {metadata.get(key)!r}"
                )
    else:
        raise ValueError("unsupported expected method family")
    # Reject compatibility payloads even if somebody adds new-looking metadata
    # around their mesh/neighbor tensors.  The required negative booleans above
    # are intentional and therefore are not searched as free text here.
    forbidden_keys = {
        "mesh_xyz",
        "mesh_vertices",
        "vertex_xyz",
        "knn_indices",
        "knn_distances",
        "knn_k",
        "mesh_checkpoint",
    }
    if forbidden_keys.intersection(payload) or forbidden_keys.intersection(metadata):
        raise ValueError("legacy mesh/kNN8 provenance is forbidden in Gaussian protocol")

    geometry_sha = _require_sha256(
        metadata.get("geometry_checkpoint_sha256"),
        label="semantic cache geometry_checkpoint_sha256",
    )
    expected_geometry_sha = _require_sha256(
        expected_geometry_checkpoint_sha256,
        label="expected geometry checkpoint SHA256",
    )
    if geometry_sha != expected_geometry_sha:
        raise ValueError("semantic cache geometry checkpoint SHA256 differs")
    geometry_record = metadata.get("geometry_checkpoint")
    geometry_path = validate_file_record(
        geometry_record,
        label="semantic cache geometry checkpoint",
    )
    if not isinstance(geometry_record, Mapping) or geometry_record.get("sha256") != expected_geometry_sha:
        raise ValueError("semantic cache geometry checkpoint file record differs")
    external_freeze_record = metadata.get("protocol_freeze")
    validate_file_record(
        external_freeze_record,
        label="semantic cache external protocol freeze",
    )
    if (
        not isinstance(external_freeze_record, Mapping)
        or external_freeze_record.get("sha256") != EXTERNAL_PROTOCOL_FREEZE_SHA256
    ):
        raise ValueError("semantic cache external protocol freeze file record differs")
    producer_record = metadata.get("producer_source")
    validate_file_record(producer_record, label="semantic cache producer source")
    producer_sha = _require_sha256(
        metadata.get("producer_source_sha256"),
        label="semantic cache producer_source_sha256",
    )
    if not isinstance(producer_record, Mapping) or producer_record.get("sha256") != producer_sha:
        raise ValueError("semantic cache producer source SHA256 differs")
    if Path(str(producer_record.get("path", ""))).name != (
        "materialize_ours_scannet_gaussian_semantic_score_cache.py"
    ):
        raise ValueError("semantic cache producer source is not the frozen materializer")
    query_record = metadata.get("query_source")
    semantic_record = metadata.get("semantic_source")
    query_source_path = validate_file_record(
        query_record, label="semantic cache query source"
    )
    semantic_source_path = validate_file_record(
        semantic_record, label="semantic cache semantic source"
    )
    query_source_sha = _require_sha256(
        metadata.get("query_source_sha256"),
        label="semantic cache query_source_sha256",
    )
    semantic_source_sha = _require_sha256(
        metadata.get("semantic_source_sha256"),
        label="semantic cache semantic_source_sha256",
    )
    if not isinstance(query_record, Mapping) or query_source_sha != query_record.get("sha256"):
        raise ValueError("semantic cache query source SHA256 differs")
    if not isinstance(semantic_record, Mapping) or semantic_source_sha != semantic_record.get("sha256"):
        raise ValueError("semantic cache semantic source SHA256 differs")
    if expected_method_family == CURRENT_METHOD_FAMILY:
        authority_records = {
            "canonical_mainline": CANONICAL_MAINLINE_SHA256,
            "canonical_method_freeze": CANONICAL_METHOD_FREEZE_SHA256,
            "canonical_field_source": None,
            "mpr_source": None,
            "support_graph_source": None,
            "surface_region_readout_source": CANONICAL_READOUT_SHA256,
            "official_radio_source": OFFICIAL_RADIO_SHA256,
        }
        for key, expected_sha in authority_records.items():
            record = metadata.get(key)
            validate_file_record(record, label=f"semantic cache {key}")
            if expected_sha is not None and (
                not isinstance(record, Mapping) or record.get("sha256") != expected_sha
            ):
                raise ValueError(f"semantic cache {key} SHA256 differs")
        field_record = metadata.get("canonical_field_source")
        if dict(field_record) != dict(semantic_record):
            raise ValueError("semantic cache canonical field/semantic source records differ")

    count = int(torch.as_tensor(expected_xyz).shape[0])
    xyz = _validate_row_tensor(
        payload, "xyz", expected_xyz, shape=(count, 3)
    )
    scale = _validate_row_tensor(
        payload, "scale", expected_scale, shape=(count, 3)
    )
    quaternion = _validate_row_tensor(
        payload, "quaternion", expected_quaternion, shape=(count, 4)
    )
    opacity = _validate_row_tensor(
        payload, "opacity", expected_opacity, shape=(count,)
    )
    if bool((scale <= 0).any()):
        raise ValueError("semantic cache activated Gaussian scales must be positive")
    if bool(((opacity < 0) | (opacity > 1)).any()):
        raise ValueError("semantic cache activated Gaussian opacity must be in [0,1]")
    norms = quaternion.float().norm(dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)):
        raise ValueError("semantic cache quaternion must be normalized GraphDECO (w,x,y,z)")
    valid = payload.get("valid")
    reference_valid = torch.as_tensor(expected_valid).detach().cpu().contiguous()
    if (
        not isinstance(valid, torch.Tensor)
        or valid.dtype != torch.bool
        or tuple(valid.shape) != (count,)
    ):
        raise ValueError(f"semantic cache valid must be row-aligned bool [{count}]")
    valid = valid.detach().cpu().contiguous()
    if reference_valid.dtype != torch.bool or not torch.equal(valid, reference_valid):
        raise ValueError("semantic cache valid/row-order differs from geometry checkpoint")
    if not bool(valid.all()):
        raise ValueError(
            "frozen Gaussian protocol requires a semantic score for every geometry row"
        )

    class_ids = payload.get("class_ids")
    class_names = payload.get("class_names")
    query_ids = payload.get("query_ids")
    if class_ids != list(PAPER_CLASS_IDS):
        raise ValueError("semantic cache class id/order differs from frozen split19")
    if class_names != list(PAPER_CLASS_NAMES) or query_ids != list(PAPER_CLASS_NAMES):
        raise ValueError("semantic cache query/class text order differs")
    scores = _as_floating_tensor(
        payload.get("semantic_scores"), label="semantic cache semantic_scores"
    )
    if tuple(scores.shape) != (count, len(PAPER_CLASS_IDS)):
        raise ValueError(
            "semantic cache semantic_scores must be row-aligned [N,19]"
        )
    row_hashes = metadata.get("row_tensor_sha256")
    expected_row_hashes = {
        "xyz",
        "scale",
        "quaternion",
        "opacity",
        "valid",
        "semantic_scores",
    }
    region_observed = None
    if expected_method_family == CURRENT_METHOD_FAMILY:
        expected_row_hashes.add("region_observed")
        region_observed = payload.get("region_observed")
        if (
            not isinstance(region_observed, torch.Tensor)
            or region_observed.dtype != torch.bool
            or tuple(region_observed.shape) != (count,)
        ):
            raise ValueError(
                f"semantic cache region_observed must be row-aligned bool [{count}]"
            )
        region_observed = region_observed.detach().cpu().contiguous()
        observed_count = int(region_observed.sum())
        fallback_count = count - observed_count
        if metadata.get("region_observed_count") != observed_count:
            raise ValueError("semantic cache region_observed_count differs")
        if metadata.get("no_evidence_fallback_count") != fallback_count:
            raise ValueError("semantic cache no_evidence_fallback_count differs")
        if observed_count <= 0 or fallback_count < 0:
            raise ValueError("semantic cache canonical totality row counts are invalid")
    if not isinstance(row_hashes, Mapping) or set(row_hashes) != expected_row_hashes:
        raise ValueError("semantic cache row_tensor_sha256 contract differs")
    tensors = {
        "xyz": xyz,
        "scale": scale,
        "quaternion": quaternion,
        "opacity": opacity,
        "valid": valid,
        "semantic_scores": scores,
    }
    if region_observed is not None:
        tensors["region_observed"] = region_observed
    for key, tensor in tensors.items():
        if row_hashes.get(key) != _tensor_sha256(tensor):
            raise ValueError(f"semantic cache {key} SHA256 differs")
    return {
        **tensors,
        "metadata": dict(metadata),
        "query_source": str(query_source_path),
        "semantic_source": str(semantic_source_path),
    }


def load_ours_gaussian_semantic_score_cache(
    path: str | Path,
    **expected: Any,
) -> tuple[dict[str, Any], str, Path]:
    payload, digest, source = load_torch_mapping(
        path, map_location="cpu", label="Ours ScanNet Gaussian semantic score cache"
    )
    return validate_ours_gaussian_semantic_score_cache(payload, **expected), digest, source


def predict_frozen_splits(semantic_scores: torch.Tensor) -> dict[str, np.ndarray]:
    """Restrict the one split19 score bank to each frozen split, then argmax."""
    scores = torch.as_tensor(semantic_scores).detach().cpu().float()
    if tuple(scores.shape[1:]) != (len(PAPER_CLASS_IDS),):
        raise ValueError("semantic_scores must have frozen split19 columns")
    column = {class_id: index for index, class_id in enumerate(PAPER_CLASS_IDS)}
    predictions: dict[str, np.ndarray] = {}
    for split in ("19", "15", "10"):
        class_ids = OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        indices = torch.tensor([column[class_id] for class_id in class_ids], dtype=torch.long)
        local = scores.index_select(1, indices)
        predictions[split] = np.asarray(class_ids, dtype=np.int32)[
            local.argmax(dim=1).numpy()
        ]
    return predictions


def _format_path(pattern: str, scene: str) -> Path:
    return Path(pattern.format(scene=scene)).expanduser().resolve()


def _scene_macro(scene_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        split: {
            metric: float(
                np.mean(
                    [float(scene["splits"][split][metric]) for scene in scene_results.values()]
                )
            )
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("paper8", "custom"), default="paper8")
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--config", required=True, help="Path pattern containing {scene}")
    parser.add_argument(
        "--geometry-checkpoint", required=True, help="Path pattern containing {scene}"
    )
    parser.add_argument(
        "--semantic-score-cache", required=True, help="Path pattern containing {scene}"
    )
    parser.add_argument("--label-ply", required=True, help="Path pattern containing {scene}")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pseudo-chunk-size", type=int, default=512)
    parser.add_argument("--radius-factor", type=float, default=5.0)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--fallback-k", type=int, default=1)
    parser.add_argument("--force-pseudo-gt", action="store_true")
    args = parser.parse_args()

    if args.cohort == "paper8":
        scenes, cohort_status = _resolve_cohort_scenes("paper8", args.scenes)
    else:
        scenes, cohort_status = _resolve_cohort_scenes("custom", args.scenes)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_results: dict[str, dict[str, Any]] = {}
    report_authority: dict[str, Any] | None = None
    for scene in scenes:
        print(f"\n=== {scene} ===", flush=True)
        config_path = _format_path(args.config, scene)
        checkpoint_path = _format_path(args.geometry_checkpoint, scene)
        score_path = _format_path(args.semantic_score_cache, scene)
        label_path = _format_path(args.label_ply, scene)
        geometry_sha = sha256_file(checkpoint_path)
        model, codec = _build_hybrid_model(
            load_config(config_path), str(checkpoint_path), torch.device("cpu")
        )
        xyz = model.get_xyz().detach().cpu().float().contiguous()
        scale = model.get_scaling().detach().cpu().float().contiguous()
        quaternion = model.get_rotation().detach().cpu().float().contiguous()
        opacity = model.get_opacity().detach().cpu().float().reshape(-1).contiguous()
        valid = torch.ones(xyz.shape[0], dtype=torch.bool)
        cache, score_sha, score_source = load_ours_gaussian_semantic_score_cache(
            score_path,
            expected_scene_id=scene,
            expected_xyz=xyz,
            expected_scale=scale,
            expected_quaternion=quaternion,
            expected_opacity=opacity,
            expected_valid=valid,
            expected_geometry_checkpoint_sha256=geometry_sha,
            expected_method_family=CURRENT_METHOD_FAMILY,
        )
        scene_authority = {
            "external_protocol_freeze": cache["metadata"]["protocol_freeze"],
            "canonical_mainline": cache["metadata"]["canonical_mainline"],
            "canonical_method_freeze": cache["metadata"]["canonical_method_freeze"],
            "surface_region_readout_source": cache["metadata"][
                "surface_region_readout_source"
            ],
            "official_radio_source": cache["metadata"]["official_radio_source"],
            "producer_source": cache["metadata"]["producer_source"],
        }
        if report_authority is None:
            report_authority = scene_authority
        elif report_authority != scene_authority:
            raise ValueError("scene semantic caches bind different frozen authorities")
        point_xyz, point_labels = _read_label_ply(str(label_path))
        pseudo_path = output_dir / "pseudo_gt" / f"{scene}.npz"
        pseudo_labels, pseudo_stats = _load_or_build_pseudo_gt(
            pseudo_path,
            xyz.numpy(),
            scale.numpy(),
            quaternion.numpy(),
            point_xyz,
            point_labels,
            radius_factor=args.radius_factor,
            candidate_k=args.candidate_k,
            fallback_k=args.fallback_k,
            class_balance=True,
            chunk_size=args.pseudo_chunk_size,
            force=args.force_pseudo_gt,
        )
        if pseudo_labels.shape != (xyz.shape[0],):
            raise ValueError("VALA pseudo-GT rows differ from optimized Gaussian rows")
        significance = opacity.numpy() * scale.numpy().prod(axis=1)
        if not np.isfinite(significance).all() or np.any(significance < 0):
            raise ValueError("opacity-times-volume significance is invalid")
        predictions = predict_frozen_splits(cache["semantic_scores"])
        split_results: dict[str, Any] = {}
        prediction_payload: dict[str, np.ndarray] = {
            "xyz": xyz.numpy(),
            "pseudo_labels": pseudo_labels,
            "significance": significance,
        }
        for split in ("19", "15", "10"):
            split_results[split] = volume_weighted_split_metrics(
                pseudo_labels,
                predictions[split],
                significance,
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            )
            prediction_payload[f"pred_split_{split}"] = predictions[split]
        prediction_path = output_dir / "predictions" / f"{scene}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if prediction_path.exists() or prediction_path.is_symlink():
            raise FileExistsError(f"prediction output must be new: {prediction_path}")
        np.savez_compressed(prediction_path, **prediction_payload)
        scene_results[scene] = {
            "num_gaussians": int(xyz.shape[0]),
            "geometry_checkpoint": str(checkpoint_path),
            "geometry_checkpoint_sha256": geometry_sha,
            "semantic_score_cache": str(score_source),
            "semantic_score_cache_sha256": score_sha,
            "region_observed_count": cache["metadata"]["region_observed_count"],
            "no_evidence_fallback_count": cache["metadata"][
                "no_evidence_fallback_count"
            ],
            "query_source": cache["query_source"],
            "query_source_sha256": cache["metadata"]["query_source_sha256"],
            "semantic_source": cache["semantic_source"],
            "semantic_source_sha256": cache["metadata"]["semantic_source_sha256"],
            "label_ply": str(label_path),
            "label_ply_sha256": sha256_file(label_path),
            "pseudo_gt": pseudo_stats,
            "pseudo_gt_cache": str(pseudo_path.resolve()),
            "pseudo_gt_cache_sha256": sha256_file(pseudo_path),
            "splits": split_results,
            "prediction_npz": str(prediction_path.resolve()),
        }
        del model, codec

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "RADIO-GS Ours explicit Gaussian semantic-score cache",
        "protocol": {
            "contract": PROTOCOL_CONTRACT,
            "task": "ScanNet OVS text-query semantic segmentation",
            "prediction_domain": PREDICTION_DOMAIN,
            "semantic_readout": SEMANTIC_READOUT,
            "spatial_transfer": SPATIAL_TRANSFER,
            "legacy_mesh_knn8": "forbidden",
            "pseudo_gt": "VALA anisotropic Mahalanobis-density vote",
            "metric_weights": "activated_opacity * activated_scale.prod()",
            "class_splits": ["19", "15", "10"],
            "class_aggregation": "present classes within each scene",
            "scene_aggregation": "unweighted scene macro",
            "cohort": args.cohort,
            "cohort_scenes": scenes,
            "cohort_status": cohort_status,
            "cpu_only": True,
            "query_text_sha256": QUERY_TEXT_SHA256,
            "class_order_sha256": CLASS_ORDER_SHA256,
            "query_class_order_sha256": QUERY_CLASS_ORDER_SHA256,
            "method_family": CURRENT_METHOD_FAMILY,
            "materializer_contract": CURRENT_MATERIALIZER_CONTRACT,
            "external_protocol_freeze_id": EXTERNAL_PROTOCOL_FREEZE_ID,
            "external_protocol_freeze_task": EXTERNAL_PROTOCOL_FREEZE_TASK,
            "external_protocol_registry_row": EXTERNAL_PROTOCOL_REGISTRY_ROW,
            "external_protocol_freeze_sha256": EXTERNAL_PROTOCOL_FREEZE_SHA256,
            "canonical_mainline_name": CANONICAL_MAINLINE_NAME,
            "canonical_mainline_sha256": CANONICAL_MAINLINE_SHA256,
            "canonical_method_freeze_name": CANONICAL_METHOD_FREEZE_NAME,
            "canonical_method_freeze_sha256": CANONICAL_METHOD_FREEZE_SHA256,
            "canonical_totality_contract": CANONICAL_TOTALITY_CONTRACT,
        },
        "authority": {
            **(report_authority or {}),
            "evaluator_source": file_record(Path(__file__).resolve()),
        },
        "args": {key: str(value) for key, value in vars(args).items()},
        "macro": _scene_macro(scene_results),
        "scenes": scene_results,
    }
    report_path = write_frozen_json(
        output_dir / "ours_scannet_gaussian_protocol_results.json", report
    )
    print(json.dumps(report["macro"], indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
