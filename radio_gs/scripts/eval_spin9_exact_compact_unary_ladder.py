#!/usr/bin/env python3
"""Graph-free SPIn exact-MPR/compact-field unary ladder.

The three representation variants in this audit share exactly one registered
K4 prototype compiler and one scalar Gaussian renderer:

``exact_capability``
    official DINOv3/SAM3 projection before MPR;
``exact_raw_adapted``
    raw RADIO MPR followed by the same frozen official adaptors;
``compact_field``
    compact canonical-field RADIO followed by those same adaptor modules.

Only the declared reference mask is opened before every primitive and rendered
unary has been persisted.  No graph, diffusion, component selection, target
RGB, or compatible-track SAM output is an input to this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import (
    load_ground_truth_mask,
    resize_mask_nearest,
)
from radio_gs.field import FeatureSpaceSignature, load_canonical_field_checkpoint
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.evidence_scorer import (
    EvidenceScoringConfig,
    score_query_evidence,
)
from radio_gs.querying.query_compilers import compile_registered_primitive_seeds
from radio_gs.querying.query_spec import SelectionMode
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    raster_adjoint_registered_view_features,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _registered_solver_masses,
    _resolve_scene_carrier_assets,
    _scene_record,
    _view_by_frame,
)
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache


SCHEMA = "spin9_exact_compact_same_compiler_unary_ladder_v1"
VARIANTS = ("exact_capability", "exact_raw_adapted", "compact_field")
COMPILER = {
    "graph": "disabled_not_constructed",
    "prototype_count": 4,
    "prototype_strategy": "spherical_mean_fps",
    "seed_construction": "winner_take_all",
    "seed_normalization": "independent_max",
    "appearance_weight": 1.0,
    "boundary_weight": 0.35,
    "prototype_temperature": 0.07,
    "score_calibration": "none",
    "registered_seed_unary_weight": 0.0,
    "unary_probability": "sigmoid(logit)",
}
RENDERER = {
    "mode": "alpha_normalized_scalar",
    "feature_contribution_gamma": 1.0,
    "invalid_row_policy": "zero_abstention_without_valid_renormalization",
    "resize_for_scoring": "bilinear_continuous_score",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values: torch.Tensor) -> dict[str, float]:
    rows = torch.as_tensor(values).detach().float().cpu().reshape(-1)
    if rows.numel() == 0 or not bool(torch.isfinite(rows).all()):
        raise ValueError("distribution input must be non-empty and finite")
    quantiles = torch.quantile(
        rows, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99])
    )
    return {
        "mean": float(rows.mean()),
        "std": float(rows.std(unbiased=False)),
        "min": float(rows.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(rows.max()),
    }


def _mpr_rows(
    cache: Mapping[str, object] | ShardedMPRCache,
    rows: torch.Tensor,
) -> torch.Tensor:
    if isinstance(cache, ShardedMPRCache):
        return cache.fetch_rows(rows)
    features = torch.as_tensor(cache["features"])
    return features.index_select(0, rows)


def _assert_same_rows(
    cache: Mapping[str, object] | ShardedMPRCache,
    *,
    xyz: torch.Tensor,
    valid: torch.Tensor,
    label: str,
) -> None:
    candidate_xyz = torch.as_tensor(cache["xyz"]).float().cpu()
    candidate_valid = torch.as_tensor(cache["valid"]).bool().cpu()
    if candidate_xyz.shape != xyz.shape or not torch.equal(candidate_xyz, xyz):
        raise ValueError(f"{label} MPR geometry differs from raw RADIO MPR")
    if not torch.equal(candidate_valid, valid):
        raise ValueError(f"{label} MPR valid rows differ from raw RADIO MPR")


@torch.no_grad()
def _normalized_mpr_bank(
    cache: Mapping[str, object] | ShardedMPRCache,
    rows: torch.Tensor,
    *,
    feature_dim: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    output = torch.empty(
        rows.numel(), int(feature_dim), dtype=torch.float16, device=device
    )
    for start in range(0, rows.numel(), int(chunk_size)):
        selected = rows[start : start + int(chunk_size)]
        source = _mpr_rows(cache, selected).to(device=device, dtype=torch.float32)
        if source.shape[1] != int(feature_dim):
            raise ValueError("MPR feature dimension differs")
        output[start : start + selected.numel()].copy_(
            F.normalize(source, dim=-1, eps=1e-8).half()
        )
        del source
    return output


@torch.no_grad()
def _adapt_raw_mpr(
    cache: Mapping[str, object] | ShardedMPRCache,
    rows: torch.Tensor,
    adaptors: Mapping[str, torch.nn.Module],
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    output = {
        name: torch.empty(
            rows.numel(), int(adaptor.output_dim), dtype=torch.float16, device=device
        )
        for name, adaptor in adaptors.items()
    }
    for start in range(0, rows.numel(), int(chunk_size)):
        selected = rows[start : start + int(chunk_size)]
        raw = _mpr_rows(cache, selected).to(device=device, dtype=torch.float32)
        for name, adaptor in adaptors.items():
            output[name][start : start + selected.numel()].copy_(
                F.normalize(adaptor(raw).float(), dim=-1, eps=1e-8).half()
            )
        del raw
    return output


@torch.no_grad()
def _adapt_compact_field(
    field_checkpoint: str | Path,
    rows: torch.Tensor,
    adaptors: Mapping[str, torch.nn.Module],
    *,
    expected_sha256: str,
    expected_num_gaussians: int,
    device: torch.device,
    chunk_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    actual_sha256 = sha256_file(field_checkpoint)
    if actual_sha256 != str(expected_sha256):
        raise ValueError("compact field checkpoint SHA-256 differs")
    field, payload = load_canonical_field_checkpoint(
        field_checkpoint, map_location="cpu"
    )
    if int(field.num_gaussians) != int(expected_num_gaussians):
        raise ValueError("compact field rows differ from exact MPR")
    field = field.to(device).eval().requires_grad_(False)
    output = {
        name: torch.empty(
            rows.numel(), int(adaptor.output_dim), dtype=torch.float16, device=device
        )
        for name, adaptor in adaptors.items()
    }
    for start in range(0, rows.numel(), int(chunk_size)):
        selected_cpu = rows[start : start + int(chunk_size)]
        selected = selected_cpu.to(device)
        raw = field.radio_features(selected).float()
        for name, adaptor in adaptors.items():
            output[name][start : start + selected.numel()].copy_(
                F.normalize(adaptor(raw).float(), dim=-1, eps=1e-8).half()
            )
        del raw
    provenance = {
        "path": str(Path(field_checkpoint).resolve()),
        "sha256": actual_sha256,
        "mpr_cache_sha256": str(payload.get("mpr_cache_sha256", "")),
        "signature": field.signature.to_dict(),
    }
    del field, payload
    torch.cuda.empty_cache()
    return output, provenance


def compile_k4_probability(
    feature_banks: Mapping[str, torch.Tensor],
    positive_seeds: torch.Tensor,
    negative_seeds: torch.Tensor,
    signatures: Mapping[str, FeatureSpaceSignature],
    *,
    score_chunk_size: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return graph-free K4 primitive logits and probabilities."""

    if set(feature_banks) != {"appearance", "boundary"}:
        raise ValueError("ladder requires exactly appearance and boundary banks")
    query = compile_registered_primitive_seeds(
        positive_seeds,
        negative_seeds,
        appearance_features=feature_banks["appearance"],
        boundary_features=feature_banks["boundary"],
        appearance_signature=signatures["appearance"],
        boundary_signature=signatures["boundary"],
        prototype_count=int(COMPILER["prototype_count"]),
        prototype_strategy=str(COMPILER["prototype_strategy"]),
        seed_normalization=str(COMPILER["seed_normalization"]),
        selection_mode=SelectionMode.ALL_COMPONENTS,
    )
    unary, components = score_query_evidence(
        query,
        feature_banks,
        config=EvidenceScoringConfig(
            appearance_weight=float(COMPILER["appearance_weight"]),
            boundary_weight=float(COMPILER["boundary_weight"]),
            prototype_temperature=float(COMPILER["prototype_temperature"]),
            score_calibration=str(COMPILER["score_calibration"]),
            score_chunk_size=int(score_chunk_size),
            registered_seed_unary_weight=float(
                COMPILER["registered_seed_unary_weight"]
            ),
            registered_observation_fusion="additive",
        ),
        num_nodes=int(positive_seeds.numel()),
    )
    probability = torch.sigmoid(unary)
    return unary, probability, components


def reference_only_threshold(
    rendered_probability: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[float, float, list[dict[str, float]]]:
    score = cv2.resize(
        np.asarray(rendered_probability, dtype=np.float32),
        (int(reference_mask.shape[1]), int(reference_mask.shape[0])),
        interpolation=cv2.INTER_LINEAR,
    )
    foreground = np.asarray(reference_mask, dtype=bool)
    best_iou = -1.0
    best_threshold = 0.5
    records: list[dict[str, float]] = []
    for threshold in np.arange(0.99, 0.02, -0.01, dtype=np.float64):
        selected = score >= float(threshold)
        intersection = int(np.logical_and(selected, foreground).sum())
        union = int(np.logical_or(selected, foreground).sum())
        iou = float(intersection / union) if union else 1.0
        records.append({"threshold": float(threshold), "reference_iou": iou})
        if iou > best_iou:
            best_iou = iou
            best_threshold = float(threshold)
    return best_threshold, best_iou, records


def _metric(score: np.ndarray, ground_truth: np.ndarray, threshold: float) -> dict:
    resized = cv2.resize(
        np.asarray(score, dtype=np.float32),
        (int(ground_truth.shape[1]), int(ground_truth.shape[0])),
        interpolation=cv2.INTER_LINEAR,
    )
    selected = resized >= float(threshold)
    foreground = np.asarray(ground_truth, dtype=bool)
    intersection = int(np.logical_and(selected, foreground).sum())
    union = int(np.logical_or(selected, foreground).sum())
    return {
        "foreground_iou": float(intersection / union) if union else 1.0,
        "pixel_accuracy": float((selected == foreground).mean()),
    }


def _aggregate(records: list[dict]) -> dict[str, float | int]:
    return {
        "foreground_iou": float(
            np.mean([float(value["foreground_iou"]) for value in records])
        ),
        "pixel_accuracy": float(
            np.mean([float(value["pixel_accuracy"]) for value in records])
        ),
        "num_frames": len(records),
    }


def _load_signatures(sidecar_path: str | Path) -> tuple[dict, dict]:
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    raw = sidecar.get("capability_signatures")
    if not isinstance(raw, Mapping):
        raise ValueError("compact capability sidecar lacks signatures")
    signatures = {
        name: FeatureSpaceSignature.from_mapping(value)
        for name, value in dict(raw).items()
    }
    if set(signatures) != {"appearance", "boundary"}:
        raise ValueError("compact capability signatures differ")
    return sidecar, signatures


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    if str(args.device) != "cuda:0":
        raise ValueError("registered GPU1 launch must expose physical GPU1 as cuda:0")
    device = torch.device(args.device)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = _scene_record(manifest, str(args.scene_id))
    protocol = dict(manifest.get("protocol", {}))
    if protocol.get("target_rgb_at_query") != (
        "forbidden_for_reusable_feature_field_track"
    ):
        raise ValueError("SPIn strict target-RGB policy differs")
    if str(scene.get("target_rgb_policy", "")) != (
        "allowed_for_field_training_but_forbidden_at_query"
    ):
        raise ValueError("scene target-RGB policy differs")
    if str(dict(scene.get("prompt", {})).get("type")) != "reference_binary_mask":
        raise ValueError("ladder requires the declared full reference mask")

    queue_scene = Path(args.queue_root).resolve() / "scenes" / str(args.scene_id)
    config_path, checkpoint_path, camera_map_path = _resolve_scene_carrier_assets(
        queue_scene,
        scene_config=str(args.scene_config),
        scene_checkpoint=str(args.scene_checkpoint),
        camera_map=str(args.camera_map),
    )
    config = load_config(str(config_path))
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    views = resolve_protocol_views(
        manifest,
        scene_id=str(args.scene_id),
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    prompt_frame = str(scene["prompt_frame_ids"][0])
    prompt_view = _view_by_frame(views, prompt_frame)
    evaluation_frames = [str(value) for value in scene["evaluation_frame_ids"]]

    model, _codec, renderer, _sharpener, _refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(config_path),
            str(checkpoint_path),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    raw_mpr, raw_sha256, raw_path = load_mpr_cache(
        args.raw_mpr,
        expected_sha256=str(args.raw_mpr_sha256),
        expected_feature_space="radio",
        require_reliability=True,
        require_formal_safety=False,
    )
    xyz = torch.as_tensor(raw_mpr["xyz"]).float().cpu()
    valid = torch.as_tensor(raw_mpr["valid"]).bool().cpu()
    rows = torch.where(valid)[0]
    model_xyz = model.get_xyz().detach().float().cpu()
    if model_xyz.shape != xyz.shape or not torch.allclose(
        model_xyz, xyz, atol=1e-6, rtol=0.0
    ):
        raise ValueError("renderer geometry differs from exact MPR")

    capability_sidecar, signatures = _load_signatures(
        args.compact_capability_sidecar
    )
    field_sha256 = sha256_file(args.field_checkpoint)
    if field_sha256 != str(args.field_checkpoint_sha256):
        raise ValueError("compact field checkpoint SHA-256 differs")
    radio_sha256 = sha256_file(args.radio_checkpoint)
    if radio_sha256 != str(args.radio_checkpoint_sha256):
        raise ValueError("official RADIO adaptor SHA-256 differs")
    if capability_sidecar.get("field_checkpoint_sha256") != field_sha256:
        raise ValueError("compact capability sidecar field hash differs")
    if capability_sidecar.get("radio_checkpoint_sha256") != radio_sha256:
        raise ValueError("compact capability sidecar adaptor hash differs")

    adaptors = {
        "appearance": load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "dino_v3_7b", kind="feature_projection"
        ).to(device).eval().requires_grad_(False),
        "boundary": load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "sam3", kind="feature_projection"
        ).to(device).eval().requires_grad_(False),
    }
    if adaptors["appearance"].output_dim != signatures["appearance"].adaptor_output_dim:
        raise ValueError("DINO adaptor/signature dimensions differ")
    if adaptors["boundary"].output_dim != signatures["boundary"].adaptor_output_dim:
        raise ValueError("SAM adaptor/signature dimensions differ")

    reference_mask = load_ground_truth_mask(scene["prompt"]["mask_path"]).astype(bool)
    reference_negative = np.logical_not(reference_mask)
    height, width = map(int, reference_mask.shape)
    prompt_maps = torch.from_numpy(
        np.stack([reference_mask, reference_negative], axis=0).astype(np.float32)
    )[None].to(device)
    prompt_pose = torch.from_numpy(prompt_view["w2c"].copy()).float().to(device)
    support_sum, support_count = raster_adjoint_registered_view_features(
        model=model,
        renderer=renderer,
        viewmat=prompt_pose,
        siglip_feat=prompt_maps,
        alpha_map=None,
        alpha_threshold=0.0,
        row_confidence=None,
    )
    fractions = support_sum / support_count.clamp_min(1e-8).unsqueeze(1)
    positive, negative = _registered_solver_masses(
        fractions[:, 0],
        fractions[:, 1],
        support_threshold=0.0,
        construction=str(COMPILER["seed_construction"]),
    )
    positive_valid = positive[rows.to(device)].detach().float().cpu()
    negative_valid = negative[rows.to(device)].detach().float().cpu()
    if not bool((positive_valid > 0).any()) or not bool((negative_valid > 0).any()):
        raise RuntimeError("reference prompt does not produce bipolar K4 support")
    del prompt_maps, support_sum, fractions, positive, negative

    def render_probability(probability: torch.Tensor, pose: torch.Tensor) -> np.ndarray:
        full = torch.zeros(xyz.shape[0], dtype=torch.float32, device=device)
        full[rows.to(device)] = probability.to(device=device, dtype=torch.float32)
        rendered = renderer.render_feature_rows(
            model,
            pose,
            full[:, None],
            feature_height=int(renderer.image_height),
            feature_width=int(renderer.image_width),
            alpha_normalize=True,
            contribution_gamma=1.0,
            row_confidence=None,
        )["feature_map"][0]
        return rendered.detach().float().cpu().numpy()

    source_provenance: dict[str, object] = {
        "raw_mpr": {"path": str(raw_path), "sha256": raw_sha256},
        "official_adaptor": {
            "path": str(Path(args.radio_checkpoint).resolve()),
            "sha256": radio_sha256,
        },
    }
    prediction_receipts: dict[str, dict] = {}
    primitive_paths: dict[str, Path] = {}
    reference_thresholds: dict[str, dict] = {}

    # Every target rendered unary is sealed before any target GT mask is read.
    for variant in VARIANTS:
        variant_provenance: dict[str, object]
        if variant == "exact_capability":
            banks: dict[str, torch.Tensor] = {}
            exact_records: dict[str, object] = {}
            for name, feature_space, path, expected, dim in (
                (
                    "appearance",
                    "dino_v3",
                    args.dino_mpr,
                    args.dino_mpr_sha256,
                    signatures["appearance"].adaptor_output_dim,
                ),
                (
                    "boundary",
                    "sam3",
                    args.sam_mpr,
                    args.sam_mpr_sha256,
                    signatures["boundary"].adaptor_output_dim,
                ),
            ):
                cache, digest, source = load_mpr_cache(
                    path,
                    expected_sha256=str(expected),
                    expected_feature_space=feature_space,
                    require_reliability=True,
                    require_formal_safety=False,
                )
                _assert_same_rows(cache, xyz=xyz, valid=valid, label=feature_space)
                metadata = dict(cache.get("metadata", {}))
                if metadata.get("official_adaptor_checkpoint_sha256") != radio_sha256:
                    raise ValueError(f"{feature_space} MPR official adaptor differs")
                banks[name] = _normalized_mpr_bank(
                    cache,
                    rows,
                    feature_dim=int(dim),
                    device=device,
                    chunk_size=int(args.feature_chunk_size),
                )
                exact_records[name] = {
                    "path": str(source),
                    "sha256": digest,
                    "projection_order": "official_adaptor_then_mpr",
                }
                del cache
                gc.collect()
            variant_provenance = {"exact_capability_mpr": exact_records}
        elif variant == "exact_raw_adapted":
            banks = _adapt_raw_mpr(
                raw_mpr,
                rows,
                adaptors,
                device=device,
                chunk_size=int(args.feature_chunk_size),
            )
            variant_provenance = {
                "raw_mpr": source_provenance["raw_mpr"],
                "projection_order": "mpr_then_same_official_adaptors",
            }
        else:
            banks, compact_provenance = _adapt_compact_field(
                args.field_checkpoint,
                rows,
                adaptors,
                expected_sha256=field_sha256,
                expected_num_gaussians=int(xyz.shape[0]),
                device=device,
                chunk_size=int(args.feature_chunk_size),
            )
            variant_provenance = {
                "compact_field": compact_provenance,
                "projection_order": "compact_field_radio_then_same_official_adaptors",
            }

        unary, probability, components = compile_k4_probability(
            banks,
            positive_valid,
            negative_valid,
            signatures,
            score_chunk_size=int(args.score_chunk_size),
        )
        full_unary = torch.zeros(xyz.shape[0], dtype=torch.float32)
        full_probability = torch.zeros(xyz.shape[0], dtype=torch.float32)
        full_unary[rows] = unary.detach().float().cpu()
        full_probability[rows] = probability.detach().float().cpu()
        primitive_path = output_root / "primitive_unary" / f"{variant}.pt"
        primitive_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "schema": SCHEMA,
                "scene_id": str(args.scene_id),
                "variant": variant,
                "xyz": xyz,
                "valid": valid,
                "primitive_unary_logit": full_unary,
                "primitive_probability": full_probability,
                "metadata": {
                    "compiler": COMPILER,
                    "compiler_sha256": json_sha256(COMPILER),
                    "source": variant_provenance,
                    "query_time_target_rgb_opened": False,
                    "target_masks_opened": False,
                },
            },
            primitive_path,
        )
        primitive_paths[variant] = primitive_path

        prompt_render = render_probability(probability, prompt_pose)
        prompt_path = output_root / "rendered_unary" / variant / f"{prompt_frame}.npy"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prompt_path, prompt_render.astype(np.float32), allow_pickle=False)
        selected_threshold, reference_iou, candidates = reference_only_threshold(
            prompt_render, reference_mask
        )
        reference_thresholds[variant] = {
            "selected_threshold": selected_threshold,
            "reference_iou": reference_iou,
            "threshold_grid": "0.99_to_0.03_step_minus_0.01",
            "candidates": candidates,
        }
        rendered_records: dict[str, dict] = {}
        for frame_id in evaluation_frames:
            view = _view_by_frame(views, frame_id)
            pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
            score = render_probability(probability, pose)
            score_path = output_root / "rendered_unary" / variant / f"{frame_id}.npy"
            np.save(score_path, score.astype(np.float32), allow_pickle=False)
            rendered_records[frame_id] = {
                "path": str(score_path),
                "sha256": sha256_file(score_path),
                "shape": list(score.shape),
            }
        receipt = {
            "schema": SCHEMA,
            "scene_id": str(args.scene_id),
            "variant": variant,
            "primitive_unary": {
                "path": str(primitive_path),
                "sha256": sha256_file(primitive_path),
                "valid_logit_distribution": _distribution(unary),
                "valid_probability_distribution": _distribution(probability),
                "component_distributions": {
                    name: _distribution(values)
                    for name, values in components.items()
                },
            },
            "rendered_reference_unary": {
                "path": str(prompt_path),
                "sha256": sha256_file(prompt_path),
                "shape": list(prompt_render.shape),
            },
            "rendered_target_unary": rendered_records,
            "reference_only_calibration": reference_thresholds[variant],
            "source": variant_provenance,
            "compiler": COMPILER,
            "renderer": RENDERER,
            "graph_constructed": False,
            "target_rgb_opened_at_query": False,
            "target_masks_opened": False,
        }
        receipt_path = output_root / "receipts" / f"{variant}_prediction_receipt.json"
        _write_json(receipt_path, receipt)
        prediction_receipts[variant] = {
            **receipt,
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
        del banks, unary, probability, components, full_unary, full_probability
        torch.cuda.empty_cache()
        gc.collect()

    target_masks = {
        frame_id: load_ground_truth_mask(
            next(
                frame["ground_truth"]
                for frame in scene["frames"]
                if str(frame["frame_id"]) == frame_id
            )
        ).astype(bool)
        for frame_id in evaluation_frames
    }
    variant_metrics: dict[str, dict] = {}
    primitive_values = {
        variant: torch.load(path, map_location="cpu")["primitive_probability"][valid]
        for variant, path in primitive_paths.items()
    }
    for variant in VARIANTS:
        fixed_records: list[dict] = []
        calibrated_records: list[dict] = []
        selected_threshold = float(
            reference_thresholds[variant]["selected_threshold"]
        )
        for frame_id in evaluation_frames:
            score = np.load(
                prediction_receipts[variant]["rendered_target_unary"][frame_id][
                    "path"
                ],
                allow_pickle=False,
            )
            fixed_records.append(
                {"frame_id": frame_id, **_metric(score, target_masks[frame_id], 0.5)}
            )
            calibrated_records.append(
                {
                    "frame_id": frame_id,
                    **_metric(score, target_masks[frame_id], selected_threshold),
                }
            )
        variant_metrics[variant] = {
            "fixed_probability_0p5": {
                **_aggregate(fixed_records),
                "frames": fixed_records,
            },
            "reference_only_calibrated": {
                **_aggregate(calibrated_records),
                "selected_threshold": selected_threshold,
                "reference_iou": float(
                    reference_thresholds[variant]["reference_iou"]
                ),
                "frames": calibrated_records,
            },
        }

    exact = primitive_values["exact_capability"].float()
    representation_gap: dict[str, dict] = {}
    for variant in ("exact_raw_adapted", "compact_field"):
        candidate = primitive_values[variant].float()
        centered_exact = exact - exact.mean()
        centered_candidate = candidate - candidate.mean()
        pearson = float(
            (centered_exact * centered_candidate).sum()
            / (
                centered_exact.norm().clamp_min(1e-8)
                * centered_candidate.norm().clamp_min(1e-8)
            )
        )
        representation_gap[variant] = {
            "primitive_probability_mae_vs_exact_capability": float(
                (candidate - exact).abs().mean()
            ),
            "primitive_probability_rmse_vs_exact_capability": float(
                (candidate - exact).square().mean().sqrt()
            ),
            "primitive_probability_pearson_vs_exact_capability": pearson,
            "fixed_iou_delta_vs_exact_capability": float(
                variant_metrics[variant]["fixed_probability_0p5"]["foreground_iou"]
                - variant_metrics["exact_capability"]["fixed_probability_0p5"][
                    "foreground_iou"
                ]
            ),
            "reference_calibrated_iou_delta_vs_exact_capability": float(
                variant_metrics[variant]["reference_only_calibrated"][
                    "foreground_iou"
                ]
                - variant_metrics["exact_capability"]["reference_only_calibrated"][
                    "foreground_iou"
                ]
            ),
        }

    report = {
        "schema": SCHEMA,
        "scene_id": str(args.scene_id),
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "protocol_hash": str(manifest.get("protocol_hash", "")),
            "claim_scope": "strict_local9_full_reference_mask_diagnostic",
            "target_rgb_at_query": "forbidden",
        },
        "assets": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "camera_map": {
                "path": str(camera_map_path),
                "sha256": sha256_file(camera_map_path),
            },
            "reference_mask": {
                "path": str(Path(scene["prompt"]["mask_path"]).resolve()),
                "sha256": sha256_file(scene["prompt"]["mask_path"]),
            },
            **source_provenance,
            "compact_field": {
                "path": str(Path(args.field_checkpoint).resolve()),
                "sha256": field_sha256,
            },
        },
        "compiler": COMPILER,
        "compiler_sha256": json_sha256(COMPILER),
        "renderer": RENDERER,
        "renderer_sha256": json_sha256(RENDERER),
        "valid_gaussians": int(valid.sum()),
        "total_gaussians": int(valid.numel()),
        "positive_seed_rows": int((positive_valid > 0).sum()),
        "negative_seed_rows": int((negative_valid > 0).sum()),
        "variants": variant_metrics,
        "representation_gap": representation_gap,
        "prediction_receipts": {
            name: {
                "path": value["receipt_path"],
                "sha256": value["receipt_sha256"],
            }
            for name, value in prediction_receipts.items()
        },
        "safety": {
            "graph_constructed": False,
            "diffusion_or_connected_selection_used": False,
            "target_rgb_opened_at_query": False,
            "compatible_track_information_used": False,
            "all_prediction_receipts_sealed_before_target_masks_opened": True,
            "target_masks_used_only_for_final_scoring": True,
        },
    }
    report_path = output_root / f"{args.scene_id}_unary_ladder_evaluation.json"
    _write_json(report_path, report)
    return {
        **report,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", choices=("lego", "room"), required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument("--camera-map", required=True)
    parser.add_argument("--raw-mpr", required=True)
    parser.add_argument("--raw-mpr-sha256", required=True)
    parser.add_argument("--dino-mpr", required=True)
    parser.add_argument("--dino-mpr-sha256", required=True)
    parser.add_argument("--sam-mpr", required=True)
    parser.add_argument("--sam-mpr-sha256", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument("--field-checkpoint-sha256", required=True)
    parser.add_argument("--compact-capability-sidecar", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--radio-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-chunk-size", type=int, default=1024)
    parser.add_argument("--score-chunk-size", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
