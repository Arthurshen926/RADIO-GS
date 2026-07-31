#!/usr/bin/env python3
"""Evaluate a protocol-locked Gaussian-first NVOS readout.

Prediction generation opens only the declared reference scribbles.  The target
ground-truth mask is opened only after the continuous score has been written,
so it cannot affect prototypes, thresholds, or the 3-D support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.evaluation.promptable_segmentation import (
    load_ground_truth_mask,
    resize_mask_nearest,
)
from radio_gs.interfaces.capability_cache import (
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint
from radio_gs.querying.evidence_scorer import EvidenceScoringConfig
from radio_gs.querying.query_compilers import compile_registered_primitive_seeds
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.query_spec import SelectionMode
from radio_gs.querying.support_solver import SupportSolverConfig
from radio_gs.querying.support_solver import PrimitiveSupportGraph
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    raster_adjoint_registered_view_features,
    rasterize_registered_view_features,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_lerf_grounding import render_1280d
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


def _scaled_raster_shape(
    height: int,
    width: int,
    scale: float,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("raster dimensions must be positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("raster scale must be finite and positive")
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _valid_normalized_score_map(
    rendered_channels: torch.Tensor,
    *,
    eps: float = 1e-6,
    coverage_power: float = 0.0,
) -> torch.Tensor:
    """Return a valid-conditioned score with optional coverage abstention.

    ``coverage_power=0`` is ``E[p | valid]`` while ``coverage_power=1``
    exactly recovers the total-alpha score ``E[v*p]``. Intermediate values
    keep the conditional score but lower confidence where few visible
    contributions have a valid capability row.
    """

    channels = torch.as_tensor(rendered_channels)
    if channels.ndim != 3 or channels.shape[0] != 2:
        raise ValueError("valid-normalized render must contain [numerator,validity]")
    if not np.isfinite(coverage_power) or coverage_power < 0:
        raise ValueError("coverage_power must be finite and non-negative")
    numerator, valid_mass = channels
    supported = valid_mass > float(eps)
    conditional = torch.where(
        supported,
        numerator / valid_mass.clamp_min(float(eps)),
        torch.zeros_like(numerator),
    )
    if coverage_power == 0:
        return conditional
    return conditional * valid_mass.clamp(0.0, 1.0).pow(float(coverage_power))


@torch.inference_mode()
def decode_region_rows(model, codec, adaptor, *, device: torch.device, chunk_size: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    count = int(model.get_xyz().shape[0])
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        indices = torch.arange(start, stop, device=device, dtype=torch.long)
        compact = model.query_gaussian_points(indices)
        radio = codec.decode_points(compact.float())
        region = adaptor(radio.float()).float() if adaptor is not None else radio.float()
        rows.append(F.normalize(region, dim=-1).half().cpu())
    return torch.cat(rows, dim=0)


def _scene_record(manifest: dict, scene_id: str) -> dict:
    matches = [scene for scene in manifest["scenes"] if scene["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest scene {scene_id!r}")
    return matches[0]


def _view_by_frame(views: list[dict], frame_id: str) -> dict:
    matches = [view for view in views if str(view["frame_id"]) == str(frame_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one protocol view for {frame_id!r}")
    return matches[0]


def _weighted_spherical_prototypes(
    rows: torch.Tensor,
    weights: torch.Tensor,
    count: int,
    *,
    iterations: int = 6,
) -> torch.Tensor:
    """Build deterministic appearance prototypes without target-set fitting.

    Prompt support can cover several object parts whose appearances should not
    be collapsed into one mean.  Weighted farthest-first initialization keeps
    tiny raster tails from becoming prototypes, followed by a small fixed
    spherical k-means refinement.  ``count=1`` exactly reduces to the previous
    weighted mean readout.
    """
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[0] != weights.shape[0]:
        raise ValueError("rows and weights must have shapes [N,D] and [N]")
    if rows.shape[0] == 0:
        raise ValueError("Cannot build prototypes from empty prompt support")
    count = min(max(1, int(count)), int(rows.shape[0]))
    weights = weights.float().clamp_min(0)
    rows = F.normalize(rows.float(), dim=-1)
    if count == 1:
        center = (rows * weights[:, None]).sum(dim=0)
        return F.normalize(center, dim=0)[None]

    selected = [int(weights.argmax())]
    min_distance = 1.0 - rows @ rows[selected[0]]
    weight_scale = weights / weights.max().clamp_min(1e-8)
    for _ in range(1, count):
        utility = min_distance.clamp_min(0) * weight_scale.sqrt()
        utility[selected] = -1
        index = int(utility.argmax())
        selected.append(index)
        min_distance = torch.minimum(min_distance, 1.0 - rows @ rows[index])
    centers = rows[selected]

    for _ in range(max(0, int(iterations))):
        assignment = (rows @ centers.T).argmax(dim=1)
        updated = []
        for index in range(count):
            member = assignment == index
            if bool(member.any()):
                center = (rows[member] * weights[member, None]).sum(dim=0)
                updated.append(F.normalize(center, dim=0))
            else:
                updated.append(centers[index])
        centers = torch.stack(updated, dim=0)
    return centers


def _load_training_poses(
    queue_scene: Path,
    evaluation_camera_names: list[str],
) -> list[torch.Tensor]:
    mapping = json.loads(
        (queue_scene / "feature_pose_mapping.json").read_text(encoding="utf-8")
    )
    train_ids = {
        int(value)
        for value in json.loads(
            (queue_scene / "train_frame_ids.json").read_text(encoding="utf-8")
        )["frame_ids"]
    }
    records = [
        record
        for record in mapping["records"]
        if int(record["feature_frame_id"]) in train_ids
    ]
    evaluation_set = {str(value) for value in evaluation_camera_names}
    # A dataset may permit target RGBs during field construction (SPIn-NeRF),
    # but query-time support remains target-view independent.  Filter those
    # cameras rather than rendering query evidence from an evaluation pose.
    records = [
        record for record in records if str(record["camera_name"]) not in evaluation_set
    ]
    poses = []
    for record in sorted(records, key=lambda value: int(value["feature_frame_id"])):
        c2w = np.loadtxt(record["pose_path"], dtype=np.float32).reshape(4, 4)
        poses.append(torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)))
    if not poses:
        raise ValueError("No protocol-permitted training support poses")
    return poses


def _resolve_observed_feature_path(queue_scene: Path, camera_name: str) -> Path:
    """Resolve a protocol camera to its saved, observed RADIO feature map."""
    mapping = json.loads(
        (queue_scene / "feature_pose_mapping.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in mapping["records"]
        if str(record.get("camera_name")) == str(camera_name)
        or str(record.get("colmap_camera_name")) == str(camera_name)
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one feature mapping for camera {camera_name!r}")
    feature_id = int(records[0]["feature_frame_id"])
    manifest = json.loads(
        (queue_scene / "radio_features" / "frame_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frames = [
        frame
        for frame in manifest["frames"]
        if int(frame.get("source_rank", -1)) == feature_id
        or int(frame.get("frame_idx", -1)) == feature_id
    ]
    if len(frames) != 1:
        raise ValueError(f"Expected one saved feature frame for id {feature_id}")
    path = queue_scene / "radio_features" / "backbone" / f"{frames[0]['saved_stem']}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.inference_mode()
def _observed_region_map(
    queue_scene: Path,
    camera_name: str,
    adaptor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Encode the registered real query view without rendering it through 3-D."""
    path = _resolve_observed_feature_path(queue_scene, camera_name)
    radio = torch.load(path, map_location="cpu").float()
    if radio.ndim != 3:
        raise ValueError(f"Expected observed RADIO feature [C,H,W], got {tuple(radio.shape)}")
    channels, height, width = radio.shape
    rows = radio.permute(1, 2, 0).reshape(-1, channels).to(device)
    if adaptor is not None:
        rows = adaptor(rows).float()
    rows = F.normalize(rows.float(), dim=-1)
    return rows.reshape(height, width, -1).permute(2, 0, 1)[None]


@torch.inference_mode()
def _screen_region_map(
    model,
    codec,
    renderer,
    sharpener,
    refiner,
    config,
    adaptor,
    pose: torch.Tensor,
    *,
    is_hybrid: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    decoded, aux = render_1280d(
        model,
        codec,
        renderer,
        sharpener,
        refiner,
        pose[None],
        is_hybrid=is_hybrid,
        config=config,
        device=pose.device,
        return_aux=True,
    )
    channels, height, width = decoded.shape[1:]
    rows = decoded.permute(0, 2, 3, 1).reshape(-1, channels).float()
    if adaptor is not None:
        rows = adaptor(rows).float()
    rows = F.normalize(rows, dim=-1)
    return rows.reshape(1, height, width, -1).permute(0, 3, 1, 2), aux


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = _scene_record(manifest, args.scene_id)
    base_scene_id = str(scene.get("base_scene_id") or args.scene_id)
    scene_root = Path(args.queue_root).resolve() / "scenes"
    queue_scene = scene_root / args.scene_id
    if not queue_scene.is_dir():
        queue_scene = scene_root / base_scene_id
    config_path = queue_scene / "gaussfm_main_track.yaml"
    checkpoint_path = queue_scene / "feature_field" / "checkpoints" / "best.pth"
    camera_map_path = queue_scene / "rgb_to_colmap_camera_mapping.json"
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))
    views = resolve_protocol_views(
        manifest,
        scene_id=args.scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    prompt_frame = str(scene["prompt_frame_ids"][0])
    prompt_view = _view_by_frame(views, prompt_frame)
    evaluation_frames = [str(value) for value in scene["evaluation_frame_ids"]]
    evaluation_views = [_view_by_frame(views, frame_id) for frame_id in evaluation_frames]
    # Protocol frame ids (for example ``image001``) need not equal their RGB /
    # COLMAP camera names (for example ``IMG_4027``).  Query-time support is
    # keyed by the latter, so exclusions must use the resolved frozen mapping.
    evaluation_camera_names = sorted(
        {
            str(view[key])
            for view in evaluation_views
            for key in ("camera_name", "colmap_camera_name")
            if view.get(key) is not None
        }
    )

    model, codec, renderer, sharpener, refiner, field_config, is_hybrid = load_render_pipeline(
        str(config_path),
        str(checkpoint_path),
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    if not is_hybrid:
        raise ValueError("Gaussian-first NVOS currently requires a hybrid field")
    adaptor = None
    if args.region_space == "sam3" and args.support_mode != "canonical_support":
        adaptor = load_radio_adaptor_from_checkpoint(
            args.radio_checkpoint, "sam3", kind="feature_projection"
        ).to(device).eval().requires_grad_(False)
    region_rows = None
    capability_bank = None
    support_graph = None
    primitive_reliability = None
    if args.support_mode == "canonical_support":
        if not args.canonical_capability_cache or not args.canonical_support_graph:
            raise ValueError(
                "canonical_support requires --canonical-capability-cache and "
                "--canonical-support-graph"
            )
        capability_bank = load_canonical_capability_bank(
            args.canonical_capability_cache,
            expected_field_checkpoint_sha256=args.canonical_field_sha256,
        )
        support_graph = load_canonical_support_graph(
            args.canonical_support_graph, capability_bank
        )
        if str(args.canonical_reliability_cache).strip():
            primitive_reliability = load_canonical_primitive_reliability(
                args.canonical_reliability_cache,
                expected_xyz=capability_bank.xyz,
                expected_valid=capability_bank.valid,
                expected_field_checkpoint_sha256=str(
                    capability_bank.metadata.get("field_checkpoint_sha256", "")
                ),
            )
        if str(args.diagnostic_graph_affinity_override).strip():
            override_path = Path(args.diagnostic_graph_affinity_override)
            override = torch.load(override_path, map_location="cpu")
            global_rows = torch.as_tensor(override.get("global_rows")).long().cpu()
            if not torch.equal(global_rows, capability_bank.global_rows):
                raise ValueError("diagnostic graph override nodes do not match capability rows")
            if int(override.get("num_global_rows", -1)) != capability_bank.num_gaussians:
                raise ValueError("diagnostic graph override global row count differs")
            base_edges = support_graph.edge_index.cpu()
            override_edges = torch.as_tensor(override.get("edge_index")).long().cpu()
            if not torch.equal(base_edges, override_edges):
                raise ValueError(
                    "diagnostic graph override must preserve the exact geometry topology"
                )
            support_graph = PrimitiveSupportGraph(
                edge_index=override_edges,
                edge_weight=torch.as_tensor(override["edge_weight"]).float(),
                raw_affinity=torch.as_tensor(override["raw_affinity"]).float(),
                local_sigma=torch.as_tensor(override["local_sigma"]).float(),
                num_nodes=int(global_rows.numel()),
                edge_channels={
                    str(name): torch.as_tensor(values).float()
                    for name, values in dict(override.get("edge_channels", {})).items()
                },
            )
        geometry_xyz = model.get_xyz().detach().float().cpu()
        if geometry_xyz.shape != capability_bank.xyz.shape or not torch.allclose(
            geometry_xyz, capability_bank.xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError("canonical capability geometry does not match renderer geometry")
    if args.support_mode == "prompt_gaussian":
        region_rows = decode_region_rows(
            model, codec, adaptor, device=device, chunk_size=max(1, args.chunk_size)
        )

    prompt = scene["prompt"]
    prompt_type = str(prompt.get("type", ""))
    if prompt_type == "positive_negative_scribbles":
        positive_native = load_ground_truth_mask(prompt["positive_path"]).astype(bool)
        negative_native = load_ground_truth_mask(prompt["negative_path"]).astype(bool)
        if positive_native.shape != negative_native.shape:
            raise ValueError("positive and negative prompt rasters must align")
    elif prompt_type == "reference_binary_mask":
        positive_native = load_ground_truth_mask(prompt["mask_path"]).astype(bool)
        negative_native = np.logical_not(positive_native)
    else:
        raise ValueError(f"Unsupported registered prompt type: {prompt_type!r}")
    native_height, native_width = map(int, positive_native.shape)
    if (
        getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
        == "raster_adjoint"
    ):
        height, width = _scaled_raster_shape(
            native_height,
            native_width,
            float(getattr(args, "prompt_registration_scale", 1.0)),
        )
    else:
        height, width = int(renderer.image_height), int(renderer.image_width)
    positive = resize_mask_nearest(
        positive_native, (height, width)
    ).astype(bool)
    negative = resize_mask_nearest(
        negative_native, (height, width)
    ).astype(bool)
    prompt_maps = torch.from_numpy(
        np.stack([positive, negative], axis=0).astype(np.float32)
    )[None].to(device)
    prompt_pose = torch.from_numpy(prompt_view["w2c"].copy()).float().to(device)
    support_view_count = 1
    prediction_threshold = 0.0
    canonical_stage_gaussian_scores: dict[str, torch.Tensor] | None = None
    if args.support_mode in {"prompt_gaussian", "canonical_support"}:
        if (
            getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
            == "raster_adjoint"
        ):
            prompt_alpha = None
            if args.alpha_threshold > 0:
                prompt_alpha = renderer.render_feature_rows(
                    model,
                    prompt_pose,
                    torch.ones(
                        model.get_xyz().shape[0],
                        1,
                        device=device,
                        dtype=torch.float32,
                    ),
                    feature_height=height,
                    feature_width=width,
                )["alpha_map"]
            support_sum, support_count = raster_adjoint_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=prompt_pose,
                siglip_feat=prompt_maps,
                alpha_map=prompt_alpha,
                alpha_threshold=args.alpha_threshold,
            )
        else:
            prompt_aux = renderer.render_features(model, prompt_pose)
            support_sum, support_count = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=prompt_pose,
                siglip_feat=prompt_maps,
                depth_map=prompt_aux["depth_map"][None],
                alpha_map=prompt_aux["alpha_map"][None],
                registration_depth_tolerance=args.depth_tolerance,
                registration_relative_depth_tolerance=args.relative_depth_tolerance,
                registration_alpha_threshold=args.alpha_threshold,
                registration_weight_mode="alpha_depth",
                deterministic_cpu_accumulation=(
                    args.support_mode == "canonical_support"
                ),
            )
        support_fraction = support_sum / support_count.clamp_min(1e-8).unsqueeze(1)
        positive_weight = support_fraction[:, 0]
        negative_weight = support_fraction[:, 1]
        positive_seed = (positive_weight > args.support_threshold) & (
            positive_weight >= negative_weight
        )
        negative_seed = (negative_weight > args.support_threshold) & (
            negative_weight > positive_weight
        )
        if not bool(positive_seed.any()) or not bool(negative_seed.any()):
            raise RuntimeError(
                f"Empty Gaussian prompt support: pos={int(positive_seed.sum())}, "
                f"neg={int(negative_seed.sum())}"
            )
        if args.support_mode == "prompt_gaussian":
            assert region_rows is not None
            positive_rows = region_rows[positive_seed.cpu()].to(
                device=device, dtype=torch.float32
            )
            negative_rows = region_rows[negative_seed.cpu()].to(
                device=device, dtype=torch.float32
            )
            positive_prototypes = _weighted_spherical_prototypes(
                positive_rows, positive_weight[positive_seed], args.prototype_count
            )
            negative_prototypes = _weighted_spherical_prototypes(
                negative_rows, negative_weight[negative_seed], args.prototype_count
            )
            score_parts: list[torch.Tensor] = []
            for start in range(0, region_rows.shape[0], max(1, args.chunk_size)):
                rows_chunk = region_rows[start : start + max(1, args.chunk_size)].to(
                    device=device, dtype=torch.float32
                )
                positive_similarity = (rows_chunk @ positive_prototypes.T).amax(dim=1)
                negative_similarity = (rows_chunk @ negative_prototypes.T).amax(dim=1)
                score_parts.append((positive_similarity - negative_similarity).cpu())
            gaussian_scores = torch.cat(score_parts, dim=0).to(device)
        else:
            assert capability_bank is not None and support_graph is not None
            valid_rows = capability_bank.global_rows
            positive_soft = torch.where(
                positive_seed.cpu(), positive_weight.detach().float().cpu(), 0.0
            )[valid_rows]
            negative_soft = torch.where(
                negative_seed.cpu(), negative_weight.detach().float().cpu(), 0.0
            )[valid_rows]
            feature_banks = {
                name: values.to(device)
                for name, values in capability_bank.valid_feature_banks().items()
            }
            support_graph = support_graph.to(device)
            query = compile_registered_primitive_seeds(
                positive_soft,
                negative_soft,
                appearance_features=feature_banks["appearance"],
                boundary_features=feature_banks["boundary"],
                appearance_signature=capability_bank.signatures["appearance"],
                boundary_signature=capability_bank.signatures["boundary"],
                prototype_count=args.prototype_count,
                prototype_strategy=args.prototype_strategy,
                positive_prompt_mass=positive_weight.detach().float().cpu()[
                    valid_rows
                ],
                negative_prompt_mass=negative_weight.detach().float().cpu()[
                    valid_rows
                ],
                selection_mode=SelectionMode(
                    getattr(
                        args,
                        "registered_selection_mode",
                        SelectionMode.SEEDED_COMPONENT.value,
                    )
                ),
            )
            engine = CanonicalQueryEngine(
                support_graph,
                scoring_config=EvidenceScoringConfig(
                    semantic_weight=1.0,
                    appearance_weight=args.appearance_weight,
                    boundary_weight=args.boundary_weight,
                    prototype_temperature=args.prototype_temperature,
                    feature_calibration=args.feature_calibration,
                    background_centroids=args.background_centroids,
                    calibration_sample_size=args.calibration_sample_size,
                    centroid_iterations=args.centroid_iterations,
                    score_calibration=args.score_calibration,
                    score_tanh_scale=args.score_tanh_scale,
                    score_chunk_size=args.score_chunk_size,
                    negative_spatial_mode=str(
                        getattr(args, "negative_spatial_mode", "none")
                    ),
                    negative_spatial_steps=int(
                        getattr(args, "negative_spatial_steps", 4)
                    ),
                    negative_spatial_decay=float(
                        getattr(args, "negative_spatial_decay", 0.8)
                    ),
                    registered_seed_unary_weight=float(
                        getattr(args, "registered_seed_unary_weight", 0.0)
                    ),
                ),
                solver_config=SupportSolverConfig(
                    iterations=args.solver_iterations,
                    residual=args.solver_residual,
                    unary_temperature=args.solver_unary_temperature,
                    support_threshold=args.solver_support_threshold,
                    solver_type=getattr(args, "solver_type", "diffusion"),
                    laplacian_weight=getattr(args, "laplacian_weight", 1.0),
                    cg_iterations=getattr(args, "cg_iterations", 64),
                    cg_tolerance=getattr(args, "cg_tolerance", 1e-5),
                ),
                graph_policy=args.graph_policy,
                component_graph_policy=args.component_graph_policy,
                graph_legacy_residual=args.graph_legacy_residual,
                channel_confidence_mode=str(
                    getattr(args, "channel_confidence_mode", "none")
                ),
                node_reliability=(
                    primitive_reliability.valid_confidence().to(device)
                    if primitive_reliability is not None
                    else None
                ),
            )
            result = engine.execute(
                query,
                feature_banks,
                feature_signatures=capability_bank.signatures,
            )
            def expand_valid_rows(values: torch.Tensor) -> torch.Tensor:
                expanded = torch.zeros(
                    capability_bank.num_gaussians, dtype=torch.float32
                )
                expanded[valid_rows] = values.detach().float().cpu()
                return expanded.to(device)

            unary_prior = torch.sigmoid(
                result.unary / float(args.solver_unary_temperature)
            )
            canonical_stage_gaussian_scores = {
                "unary_prior": expand_valid_rows(unary_prior),
                "propagated": expand_valid_rows(result.probabilities),
                "connected": expand_valid_rows(result.selected_probabilities),
            }
            gaussian_scores = canonical_stage_gaussian_scores[
                str(getattr(args, "registered_readout_stage", "connected"))
            ]
            prediction_threshold = float(args.solver_support_threshold)
        positive_seed_count = int(positive_seed.sum())
        negative_seed_count = int(negative_seed.sum())
    else:
        if args.prompt_feature_source == "observed":
            prompt_region = _observed_region_map(
                queue_scene,
                str(prompt_view["camera_name"]),
                adaptor,
                device=device,
            )
        else:
            prompt_region, _ = _screen_region_map(
                model, codec, renderer, sharpener, refiner, field_config, adaptor,
                prompt_pose, is_hybrid=is_hybrid,
            )
        prompt_rows = prompt_region[0].permute(1, 2, 0).reshape(-1, prompt_region.shape[1])
        prompt_hw = (int(prompt_region.shape[-2]), int(prompt_region.shape[-1]))
        prompt_positive = resize_mask_nearest(positive.astype(np.uint8), prompt_hw).astype(bool)
        prompt_negative = resize_mask_nearest(negative.astype(np.uint8), prompt_hw).astype(bool)
        positive_flat = torch.from_numpy(prompt_positive.reshape(-1)).to(device)
        negative_flat = torch.from_numpy(prompt_negative.reshape(-1)).to(device)
        positive_prototypes = _weighted_spherical_prototypes(
            prompt_rows[positive_flat], torch.ones(int(positive_flat.sum()), device=device),
            args.prototype_count,
        )
        negative_prototypes = _weighted_spherical_prototypes(
            prompt_rows[negative_flat], torch.ones(int(negative_flat.sum()), device=device),
            args.prototype_count,
        )
        total_sum = torch.zeros(model.get_xyz().shape[0], 1, device=device)
        total_count = torch.zeros(model.get_xyz().shape[0], device=device)
        training_poses = _load_training_poses(queue_scene, evaluation_camera_names)
        support_view_count = len(training_poses)
        for support_pose_cpu in training_poses:
            support_pose = support_pose_cpu.to(device)
            support_region, support_aux = _screen_region_map(
                model, codec, renderer, sharpener, refiner, field_config, adaptor,
                support_pose, is_hybrid=is_hybrid,
            )
            support_rows = support_region[0].permute(1, 2, 0).reshape(
                -1, support_region.shape[1]
            )
            screen_scores = (
                (support_rows @ positive_prototypes.T).amax(dim=1)
                - (support_rows @ negative_prototypes.T).amax(dim=1)
            ).reshape(1, 1, height, width)
            lifted_sum, lifted_count = rasterize_registered_view_features(
                model=model,
                renderer=renderer,
                viewmat=support_pose,
                siglip_feat=screen_scores,
                depth_map=support_aux["depth_map"],
                alpha_map=support_aux["alpha_map"],
                registration_depth_tolerance=args.depth_tolerance,
                registration_relative_depth_tolerance=args.relative_depth_tolerance,
                registration_alpha_threshold=args.alpha_threshold,
                registration_weight_mode="alpha_depth",
            )
            total_sum += lifted_sum
            total_count += lifted_count
        observed = total_count > 0
        gaussian_scores = torch.full_like(total_count, -1.0)
        gaussian_scores[observed] = total_sum[observed, 0] / total_count[observed]
        positive_seed_count = None
        negative_seed_count = None

    output_root = Path(args.output_dir).resolve()
    score_paths: dict[str, str] = {}
    predictions: dict[str, np.ndarray] = {}
    stage_score_paths: dict[str, dict[str, str]] = {}
    stage_predictions: dict[str, dict[str, np.ndarray]] = {}
    score_height, score_width = _scaled_raster_shape(
        int(renderer.image_height),
        int(renderer.image_width),
        float(getattr(args, "score_render_scale", 1.0)),
    )
    valid_support = (
        capability_bank.valid.to(device=device, dtype=torch.float32)
        if (
            capability_bank is not None
            and bool(getattr(args, "valid_support_normalization", False))
        )
        else None
    )

    def render_scalar_scores(
        pose: torch.Tensor,
        values: torch.Tensor,
    ) -> np.ndarray:
        with torch.no_grad():
            if valid_support is None:
                score_map = renderer.render_feature_rows(
                    model,
                    pose,
                    values[:, None],
                    feature_height=score_height,
                    feature_width=score_width,
                    alpha_normalize=True,
                    contribution_gamma=args.feature_contribution_gamma,
                )["feature_map"][0]
            else:
                channels = renderer.render_feature_rows(
                    model,
                    pose,
                    torch.stack(
                        [values * valid_support, valid_support],
                        dim=1,
                    ),
                    feature_height=score_height,
                    feature_width=score_width,
                    alpha_normalize=True,
                    contribution_gamma=args.feature_contribution_gamma,
                )["feature_map"]
                score_map = _valid_normalized_score_map(
                    channels,
                    coverage_power=float(
                        getattr(args, "valid_support_coverage_power", 0.0)
                    ),
                )
        return score_map.float().cpu().numpy()

    for frame_id in evaluation_frames:
        view = _view_by_frame(views, frame_id)
        pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
        rendered = render_scalar_scores(pose, gaussian_scores)
        score_path = output_root / "scores" / args.scene_id / f"{frame_id}.npy"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(score_path, rendered.astype(np.float32), allow_pickle=False)
        score_paths[frame_id] = str(score_path)
        predictions[frame_id] = rendered
        if canonical_stage_gaussian_scores is not None:
            for stage_name, stage_gaussian_scores in canonical_stage_gaussian_scores.items():
                stage_rendered = (
                    rendered
                    if stage_name == "connected"
                    else render_scalar_scores(pose, stage_gaussian_scores)
                )
                stage_path = (
                    output_root
                    / "stage_scores"
                    / stage_name
                    / args.scene_id
                    / f"{frame_id}.npy"
                )
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(stage_path, stage_rendered.astype(np.float32), allow_pickle=False)
                stage_score_paths.setdefault(stage_name, {})[frame_id] = str(stage_path)
                stage_predictions.setdefault(stage_name, {})[frame_id] = stage_rendered

    # Evaluation begins only after every prediction has been persisted.
    frame_metrics: list[dict] = []
    stage_frame_metrics: dict[str, list[dict]] = {
        name: [] for name in stage_predictions
    }
    for frame_id in evaluation_frames:
        frame = next(value for value in scene["frames"] if str(value["frame_id"]) == frame_id)
        gt = load_ground_truth_mask(frame["ground_truth"]).astype(bool)
        score = cv2.resize(
            predictions[frame_id], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        pred = score >= prediction_threshold
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = float(intersection / union) if union else 1.0
        accuracy = float((pred == gt).mean())
        frame_metrics.append(
            {"frame_id": frame_id, "foreground_iou": iou, "pixel_accuracy": accuracy}
        )
        for stage_name, per_frame in stage_predictions.items():
            stage_score = cv2.resize(
                per_frame[frame_id],
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            stage_pred = stage_score >= prediction_threshold
            stage_intersection = np.logical_and(stage_pred, gt).sum()
            stage_union = np.logical_or(stage_pred, gt).sum()
            stage_iou = (
                float(stage_intersection / stage_union) if stage_union else 1.0
            )
            stage_accuracy = float((stage_pred == gt).mean())
            stage_frame_metrics[stage_name].append(
                {
                    "frame_id": frame_id,
                    "foreground_iou": stage_iou,
                    "pixel_accuracy": stage_accuracy,
                }
            )

    stage_metrics = {
        name: {
            "foreground_iou": float(
                np.mean([value["foreground_iou"] for value in values])
            ),
            "pixel_accuracy": float(
                np.mean([value["pixel_accuracy"] for value in values])
            ),
            "frames": values,
        }
        for name, values in stage_frame_metrics.items()
    }

    report = {
        "scene_id": args.scene_id,
        "protocol_hash": manifest["protocol_hash"],
        "method": (
            f"gaussian_first_{args.support_mode}_{args.region_space}_cosine_margin_"
            f"{args.prototype_count}proto_"
            f"{'raster_responsibility' if args.support_mode == 'canonical_support' else args.prompt_feature_source}_prompt"
        ),
        "positive_gaussian_seeds": positive_seed_count,
        "negative_gaussian_seeds": negative_seed_count,
        "positive_prompt_pixels": int(positive.sum()),
        "negative_prompt_pixels": int(negative.sum()),
        "positive_prompt_pixels_native": int(positive_native.sum()),
        "negative_prompt_pixels_native": int(negative_native.sum()),
        "prompt_native_resolution": [native_height, native_width],
        "prompt_registration_resolution": [height, width],
        "support_mode": args.support_mode,
        "support_view_count": support_view_count,
        "support_threshold": float(args.support_threshold),
        "prototype_count": int(args.prototype_count),
        "prototype_strategy": str(args.prototype_strategy),
        "prompt_feature_source": (
            "raster_responsibility"
            if args.support_mode == "canonical_support"
            else args.prompt_feature_source
        ),
        "prompt_type": prompt_type,
        "prompt_registration": (
            "exact_front_to_back_raster_adjoint"
            if getattr(args, "prompt_registration_mode", "legacy_alpha_depth")
            == "raster_adjoint"
            else (
                "raster_responsibility_deterministic_cpu"
                if args.support_mode == "canonical_support"
                else "raster_contribution"
            )
        ),
        "feature_observation_operator": {
            "type": (
                "normalized_front_to_back_contribution_power"
                if float(args.feature_contribution_gamma) != 1.0
                else "alpha_normalized_mean"
            ),
            "gamma": float(args.feature_contribution_gamma),
            "score_render_resolution": [score_height, score_width],
            "valid_support_normalization": bool(valid_support is not None),
            "valid_support_formula": (
                "sum(w*v*p)/sum(w*v) * coverage**coverage_power"
                if valid_support is not None
                else None
            ),
            "valid_support_coverage_power": (
                float(getattr(args, "valid_support_coverage_power", 0.0))
                if valid_support is not None
                else None
            ),
            "query_dependent": False,
            "changes_geometry_or_alpha": False,
        },
        "score_threshold": prediction_threshold,
        "shared_solver": (
            {
                "appearance_weight": float(args.appearance_weight),
                "boundary_weight": float(args.boundary_weight),
                "prototype_temperature": float(args.prototype_temperature),
                "iterations": int(args.solver_iterations),
                "residual": float(args.solver_residual),
                "unary_temperature": float(args.solver_unary_temperature),
                "support_threshold": float(args.solver_support_threshold),
                "solver_type": getattr(args, "solver_type", "diffusion"),
                "laplacian_weight": float(getattr(args, "laplacian_weight", 1.0)),
                "cg_iterations": int(getattr(args, "cg_iterations", 64)),
                "hard_seed_threshold": 0.20,
                "registered_seed_unary_weight": float(
                    getattr(args, "registered_seed_unary_weight", 0.0)
                ),
                "registered_selection_mode": str(
                    getattr(
                        args,
                        "registered_selection_mode",
                        SelectionMode.SEEDED_COMPONENT.value,
                    )
                ),
                "registered_readout_stage": str(
                    getattr(args, "registered_readout_stage", "connected")
                ),
                "graph_policy": args.graph_policy,
                "component_graph_policy": args.component_graph_policy,
                "graph_legacy_residual": float(args.graph_legacy_residual),
                "channel_confidence_mode": str(
                    getattr(args, "channel_confidence_mode", "none")
                ),
                "negative_spatial_mode": str(
                    getattr(args, "negative_spatial_mode", "none")
                ),
                "negative_spatial_steps": int(
                    getattr(args, "negative_spatial_steps", 4)
                ),
                "negative_spatial_decay": float(
                    getattr(args, "negative_spatial_decay", 0.8)
                ),
                "spatial_log_weight": 0.25,
                "spatial_floor": 0.01,
                "feature_calibration": args.feature_calibration,
                "background_centroids": int(args.background_centroids),
                "calibration_sample_size": int(args.calibration_sample_size),
                "centroid_iterations": int(args.centroid_iterations),
                "score_calibration": args.score_calibration,
                "score_tanh_scale": float(args.score_tanh_scale),
                "calibration_uses_target_labels": False,
                "calibration_uses_target_masks": False,
                "calibration_uses_query_conditioned_scores": (
                    args.score_calibration != "none"
                ),
                "calibration_uses_unlabeled_scene_statistics": (
                    args.feature_calibration != "none"
                    or int(args.background_centroids) > 0
                    or args.score_calibration != "none"
                ),
                "primitive_reliability": (
                    {
                        "cache": str(
                            Path(args.canonical_reliability_cache).resolve()
                        ),
                        "formula": primitive_reliability.metadata.get("formula"),
                        "application": "centered_unary_shrink",
                        "prototype_precision_weighting": False,
                        "centered_unary_shrink": True,
                        "seed_constraints_shrunk": False,
                        "uses_query_or_target_labels": False,
                    }
                    if primitive_reliability is not None
                    else None
                ),
            }
            if args.support_mode == "canonical_support"
            else None
        ),
        "frames": frame_metrics,
        "foreground_iou": float(np.mean([value["foreground_iou"] for value in frame_metrics])),
        "pixel_accuracy": float(np.mean([value["pixel_accuracy"] for value in frame_metrics])),
        "score_paths": score_paths,
        "stage_metrics": stage_metrics,
        "stage_score_paths": stage_score_paths,
        "safety": {
            "target_ground_truth_opened_before_prediction_write": False,
            "target_rgb_opened": False,
            "registered_prompt_rgb_feature_used": (
                args.support_mode == "multiview_score_lift"
                and args.prompt_feature_source == "observed"
            ),
            "target_camera_used_as_support": False,
            "test_calibration": False,
            "test_calibration_definition": (
                "no target labels, target masks, or metric feedback are used; "
                "unlabeled evaluation-scene statistics are disclosed separately"
            ),
            "official_sam_decoder": False,
            "canonical_capability_cache": (
                str(Path(args.canonical_capability_cache).resolve())
                if args.canonical_capability_cache
                else ""
            ),
            "canonical_support_graph": (
                str(Path(args.canonical_support_graph).resolve())
                if args.canonical_support_graph
                else ""
            ),
            "canonical_reliability_cache": (
                str(Path(args.canonical_reliability_cache).resolve())
                if str(args.canonical_reliability_cache).strip()
                else ""
            ),
            "diagnostic_graph_affinity_override": (
                str(Path(args.diagnostic_graph_affinity_override).resolve())
                if str(args.diagnostic_graph_affinity_override).strip()
                else ""
            ),
            "main_result_eligible": not bool(
                str(args.diagnostic_graph_affinity_override).strip()
            ),
            "target_camera_names_excluded_from_support": evaluation_camera_names,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{args.scene_id}_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--region-space", choices=["radio", "sam3"], default="sam3")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--support-threshold", type=float, default=0.0)
    parser.add_argument("--prototype-count", type=int, default=1)
    parser.add_argument(
        "--prototype-strategy",
        choices=("weighted_fps", "spherical_mean_fps"),
        default="spherical_mean_fps",
    )
    parser.add_argument(
        "--prompt-feature-source",
        choices=("observed", "rendered"),
        default="observed",
        help="Use the registered real query feature (default) or a rendered diagnostic.",
    )
    parser.add_argument(
        "--support-mode",
        choices=("prompt_gaussian", "multiview_score_lift", "canonical_support"),
        default="prompt_gaussian",
    )
    parser.add_argument("--canonical-capability-cache", default="")
    parser.add_argument("--canonical-support-graph", default="")
    parser.add_argument("--canonical-reliability-cache", default="")
    parser.add_argument(
        "--prompt-registration-mode",
        choices=("legacy_alpha_depth", "raster_adjoint"),
        default="legacy_alpha_depth",
        help=(
            "Use the frozen footprint/depth proxy or the exact front-to-back "
            "compositing adjoint for registered prompt lifting."
        ),
    )
    parser.add_argument(
        "--prompt-registration-scale",
        type=float,
        default=1.0,
        help=(
            "Raster scale relative to the native prompt; used only by "
            "--prompt-registration-mode raster_adjoint."
        ),
    )
    parser.add_argument(
        "--score-render-scale",
        type=float,
        default=1.0,
        help="Scalar score raster scale relative to the frozen feature raster.",
    )
    parser.add_argument(
        "--valid-support-normalization",
        action="store_true",
        help=(
            "For canonical support, render sum(w*v*p)/sum(w*v) so invalid "
            "capability rows abstain instead of diluting scalar scores."
        ),
    )
    parser.add_argument(
        "--valid-support-coverage-power",
        type=float,
        default=0.0,
        help=(
            "Query-independent abstention after valid normalization. Zero is "
            "pure conditional scoring and one exactly recovers total-alpha "
            "dilution; intermediate values trade score purity for coverage."
        ),
    )
    parser.add_argument(
        "--feature-contribution-gamma",
        type=float,
        default=1.0,
        help=(
            "Query-independent exponent for normalized front-to-back feature "
            "mixture weights; 1 is ordinary alpha averaging."
        ),
    )
    parser.add_argument(
        "--diagnostic-graph-affinity-override",
        default="",
        help=(
            "Diagnostic only: replace edge affinities with an exact-topology graph "
            "from another canonical field; reported results are not main-table eligible."
        ),
    )
    parser.add_argument(
        "--graph-policy",
        choices=(
            "legacy",
            "typed",
            "geometry",
            "appearance",
            "boundary",
            "category_mix",
            "instance_mix",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--component-graph-policy",
        choices=(
            "same",
            "legacy",
            "typed",
            "geometry",
            "appearance",
            "boundary",
            "category_mix",
            "instance_mix",
        ),
        default="same",
    )
    parser.add_argument("--graph-legacy-residual", type=float, default=0.0)
    parser.add_argument(
        "--channel-confidence-mode",
        choices=("none", "affinity_mass", "max_affinity"),
        default="none",
        help=(
            "optional label-free capability abstention; confidence modes keep "
            "unary evidence through a self loop when all neighbour relations are weak"
        ),
    )
    parser.add_argument(
        "--negative-spatial-mode",
        choices=("none", "truncated_graph_decay", "signed_geodesic"),
        default="none",
    )
    parser.add_argument("--negative-spatial-steps", type=int, default=4)
    parser.add_argument("--negative-spatial-decay", type=float, default=0.8)
    parser.add_argument("--canonical-field-sha256", default="")
    parser.add_argument(
        "--registered-seed-unary-weight",
        type=float,
        default=0.0,
        help=(
            "Direct signed unary weight for raster-registered positive/negative "
            "primitive responsibilities; zero preserves the frozen protocol."
        ),
    )
    parser.add_argument(
        "--registered-selection-mode",
        choices=(
            SelectionMode.SEEDED_COMPONENT.value,
            SelectionMode.ALL_COMPONENTS.value,
        ),
        default=SelectionMode.SEEDED_COMPONENT.value,
        help=(
            "Read out only seed-connected active support (frozen behavior) or "
            "retain every active component for full-region prompts."
        ),
    )
    parser.add_argument(
        "--registered-readout-stage",
        choices=("unary_prior", "propagated", "connected"),
        default="connected",
        help=(
            "Choose the continuous unary/graph field or the component-masked "
            "support as the final scalar render; all stages remain reported."
        ),
    )
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.35)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument(
        "--feature-calibration",
        choices=("none", "diagonal_robust"),
        default="none",
    )
    parser.add_argument("--background-centroids", type=int, default=0)
    parser.add_argument("--calibration-sample-size", type=int, default=8192)
    parser.add_argument("--centroid-iterations", type=int, default=4)
    parser.add_argument(
        "--score-calibration",
        choices=("none", "robust_tanh", "robust_tanh_centered", "robust_tanh_zero"),
        default="none",
    )
    parser.add_argument("--score-tanh-scale", type=float, default=2.0)
    parser.add_argument("--score-chunk-size", type=int, default=65536)
    parser.add_argument("--solver-iterations", type=int, default=12)
    parser.add_argument("--solver-residual", type=float, default=0.30)
    parser.add_argument(
        "--solver-type", choices=("diffusion", "random_walker", "confidence_random_walker"), default="diffusion"
    )
    parser.add_argument("--laplacian-weight", type=float, default=1.0)
    parser.add_argument("--cg-iterations", type=int, default=64)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--solver-unary-temperature", type=float, default=0.10)
    parser.add_argument("--solver-support-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if not np.isfinite(args.feature_contribution_gamma) or args.feature_contribution_gamma <= 0:
        parser.error("--feature-contribution-gamma must be finite and positive")
    if (
        not np.isfinite(args.prompt_registration_scale)
        or args.prompt_registration_scale <= 0
    ):
        parser.error("--prompt-registration-scale must be finite and positive")
    if not np.isfinite(args.score_render_scale) or args.score_render_scale <= 0:
        parser.error("--score-render-scale must be finite and positive")
    if (
        args.prompt_registration_mode == "legacy_alpha_depth"
        and args.prompt_registration_scale != 1.0
    ):
        parser.error(
            "--prompt-registration-scale applies only to raster_adjoint mode"
        )
    if args.valid_support_normalization and args.support_mode != "canonical_support":
        parser.error(
            "--valid-support-normalization requires --support-mode canonical_support"
        )
    if (
        not np.isfinite(args.valid_support_coverage_power)
        or args.valid_support_coverage_power < 0
    ):
        parser.error(
            "--valid-support-coverage-power must be finite and non-negative"
        )
    if (
        args.valid_support_coverage_power != 0
        and not args.valid_support_normalization
    ):
        parser.error(
            "--valid-support-coverage-power requires --valid-support-normalization"
        )
    if (
        not np.isfinite(args.registered_seed_unary_weight)
        or args.registered_seed_unary_weight < 0
    ):
        parser.error("--registered-seed-unary-weight must be finite and non-negative")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
