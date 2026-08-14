#!/usr/bin/env python3
"""Solve a canonical coefficient field with query-free multiview render loss."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.interfaces.semantic_alignment import (
    GlobalRegionSummaryBridge,
    project_dense_region_semantics,
)
from radio_gs.losses.radio_adaptor_loss import (
    compute_radio_adaptor_masked_render_losses,
)
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.models.siglip_projection import (
    SigLIP2FeatureProjection,
    SigLIP2SummaryHead,
)
from radio_gs.rendering.coefficient_renderer import render_canonical_radio
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _resolve_extracted_capability_source,
)
from radio_gs.training.gauge_separated_capability import gauge_separated_radio
from radio_gs.training.canonical_field_losses import normalized_render_reconstruction_loss
from radio_gs.training.feature_training_utils import SimpleRadioDataset
from radio_gs.training.primitive_consensus import (
    PrimitiveConsensus,
    primitive_reconstruction_loss,
)


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_consensus(path: str) -> tuple[PrimitiveConsensus, dict]:
    cache = torch.load(path, map_location="cpu")
    factorized = cache.get("factorized_radio")
    if isinstance(factorized, Mapping):
        targets = torch.as_tensor(factorized["canonical_feature"]).float()
        valid = torch.as_tensor(factorized["valid"]).bool()
        reliability = torch.as_tensor(factorized["reliability"]).float()
    else:
        targets = torch.as_tensor(cache["features"]).float()
        valid = torch.as_tensor(cache["valid"]).bool()
        reliability = torch.as_tensor(cache["reliability"]).float()
    counts = torch.as_tensor(cache["view_counts"]).long()
    return (
        PrimitiveConsensus(
            targets=targets,
            valid=valid,
            observation_count=counts,
            reliability=reliability,
            per_view_agreement=torch.empty(0, targets.shape[0]),
        ),
        cache,
    )


def _dataset(
    config,
    renderer,
    frame_ids: list[int] | None = None,
    *,
    feature_subdir: str = "backbone",
) -> SimpleRadioDataset:
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    pose_file = raw_pose_file if raw_pose_file and Path(raw_pose_file).is_file() else None
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    fallback = feature_dir / "poses_w2c"
    pose_dir = (
        raw_pose_dir
        if raw_pose_dir and Path(raw_pose_dir).is_dir()
        else str(fallback) if fallback.is_dir() else None
    )
    return SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(
            int(getattr(config, "feature_height", renderer.image_height)),
            int(getattr(config, "feature_width", renderer.image_width)),
        ),
        feature_subdir=feature_subdir,
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=frame_ids,
    )


def _load_official_capability_teacher_datasets(
    config,
    renderer,
    raw_dataset: SimpleRadioDataset,
    capability_names: list[str],
) -> tuple[
    dict[str, SimpleRadioDataset],
    dict[str, dict[int, int]],
    dict[str, dict[str, object]],
]:
    """Load registered maps emitted by the official C-RADIO adaptor runtime.

    The extractor stores adaptor maps on their native token grid.  The dataset
    is intentionally responsible for the *only* subsequent interpolation to
    the fixed Gaussian render grid.  This preserves the required ordering
    ``resample(A_official(raw))`` and avoids applying a nonlinear adaptor to an
    interpolated raw feature map.
    """

    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    frame_ids = [int(frame) for frame in raw_dataset.frame_indices]
    datasets: dict[str, SimpleRadioDataset] = {}
    frame_to_index: dict[str, dict[int, int]] = {}
    provenance: dict[str, dict[str, object]] = {}
    for name in capability_names:
        source = _resolve_extracted_capability_source(feature_dir, name)
        dataset = _dataset(
            config,
            renderer,
            frame_ids,
            feature_subdir=str(source["subdir"]),
        )
        capability_frames = [int(frame) for frame in dataset.frame_indices]
        if capability_frames != frame_ids:
            raise ValueError(
                f"official {name} frame order differs from raw RADIO render targets"
            )
        datasets[name] = dataset
        frame_to_index[name] = {
            int(frame): index for index, frame in enumerate(capability_frames)
        }
        provenance[name] = dict(source)
    return datasets, frame_to_index, provenance


def _official_capability_teacher_map(
    dataset: SimpleRadioDataset,
    index: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one normalized official map after only registration resampling."""

    values = dataset[index]["radio_features"].to(device=device, dtype=torch.float32)
    if values.ndim != 3:
        raise ValueError("official capability teacher must be [C,H,W]")
    return F.normalize(values[None], dim=1, eps=1e-8)


def _even_subset(values: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(values):
        return list(values)
    positions = torch.linspace(0, len(values) - 1, count).round().long().tolist()
    return [values[int(index)] for index in positions]


def _excluded_mpr_frame_ids(metadata: Mapping[str, object]) -> set[int]:
    """Recover the frozen exclusion set from current or nested MPR metadata."""

    direct = metadata.get("excluded_frame_ids")
    registration = metadata.get("registration_responsibility_contract", {})
    nested = (
        registration.get("excluded_frame_ids")
        if isinstance(registration, Mapping)
        else None
    )
    declared = direct if direct is not None else nested
    if declared is None:
        raise ValueError("MPR metadata does not declare excluded benchmark frames")
    values = {int(frame) for frame in declared}
    if not values:
        raise ValueError("MPR benchmark exclusion set must not be empty")
    return values


def _parse_frame_ids(raw: str) -> set[int]:
    value = str(raw or "").strip()
    if not value:
        return set()
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return {int(token) for token in tokens}


def _semantic_teacher_path(root: Path, scene: str, frame: int) -> Path:
    candidates = (
        root / scene / f"rgb_{frame}.pt",
        root / f"rgb_{frame}.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"semantic teacher is missing for frame {frame}: {candidates}")


def _load_semantic_teacher(root: Path, scene: str, frame: int, device) -> torch.Tensor:
    teacher = torch.load(
        _semantic_teacher_path(root, scene, frame), map_location="cpu"
    )
    teacher = torch.as_tensor(teacher).float()
    if teacher.ndim == 4 and teacher.shape[0] == 1:
        teacher = teacher[0]
    if teacher.ndim != 3:
        raise ValueError("semantic teacher must be [C,H,W]")
    return teacher.to(device)


def _semantic_fidelity_losses(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    alpha_map: torch.Tensor,
    *,
    alpha_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if predicted.ndim != 4 or predicted.shape[0] != 1:
        raise ValueError("predicted semantics must be [1,C,H,W]")
    if teacher.shape != predicted.shape[1:]:
        raise ValueError(
            f"semantic teacher/prediction mismatch: {tuple(teacher.shape)} vs "
            f"{tuple(predicted.shape[1:])}"
        )
    valid = alpha_map >= float(alpha_threshold)
    predicted_pixels = predicted[0].permute(1, 2, 0)[valid]
    teacher_pixels = teacher.permute(1, 2, 0)[valid]
    if predicted_pixels.numel() == 0:
        zero = predicted.sum() * 0.0
        return zero, zero, 0
    cosine = F.cosine_similarity(predicted_pixels, teacher_pixels, dim=-1, eps=1e-8)
    predicted_centered = predicted_pixels - predicted_pixels.mean(dim=0, keepdim=True)
    teacher_centered = teacher_pixels - teacher_pixels.mean(dim=0, keepdim=True)
    centered_cosine = F.cosine_similarity(
        predicted_centered, teacher_centered, dim=-1, eps=1e-8
    )
    return (
        1.0 - cosine.mean(),
        1.0 - centered_cosine.mean(),
        int(cosine.numel()),
    )


@torch.no_grad()
def _view_cosine(field, model, renderer, sample, device, *, reliability_splat: bool) -> tuple[float, int]:
    result = render_canonical_radio(
        renderer,
        model,
        field,
        sample["pose_w2c"].to(device),
        feature_height=sample["radio_features"].shape[1],
        feature_width=sample["radio_features"].shape[2],
        use_reliability=reliability_splat,
    )
    predicted = result["feature_map"].permute(1, 2, 0).float()
    teacher = sample["radio_features"].to(device).permute(1, 2, 0).float()
    valid = result["alpha_map"] >= 0.02
    cosine = F.cosine_similarity(predicted[valid], teacher[valid], dim=-1)
    return (float(cosine.mean()) if cosine.numel() else 0.0, int(cosine.numel()))


@torch.no_grad()
def _mean_view_cosine(
    field,
    model,
    renderer,
    dataset,
    frame_to_index: dict[int, int],
    frames: list[int],
    device,
    *,
    reliability_splat: bool,
) -> float:
    weighted = 0.0
    count = 0
    field.eval()
    for frame in frames:
        value, pixels = _view_cosine(
            field,
            model,
            renderer,
            dataset[frame_to_index[frame]],
            device,
            reliability_splat=reliability_splat,
        )
        weighted += value * pixels
        count += pixels
    return weighted / max(1, count)


@torch.no_grad()
def _mean_multicapability_fidelity(
    field, model, renderer, dataset, frame_to_index, frames, device, *,
    adaptors: dict[str, torch.nn.Module], reliability_splat: bool,
    alpha_threshold: float,
    capability_teacher_datasets: Mapping[str, SimpleRadioDataset] | None = None,
    capability_teacher_frame_to_index: Mapping[str, Mapping[int, int]] | None = None,
) -> dict[str, float]:
    if (capability_teacher_datasets is None) != (
        capability_teacher_frame_to_index is None
    ):
        raise ValueError(
            "official capability datasets and frame indices must be supplied together"
        )
    if capability_teacher_datasets is not None:
        missing = sorted(set(adaptors) - set(capability_teacher_datasets))
        if missing:
            raise ValueError(f"official capability datasets are missing: {missing}")
        missing_indices = sorted(
            set(adaptors) - set(capability_teacher_frame_to_index or {})
        )
        if missing_indices:
            raise ValueError(
                f"official capability frame indices are missing: {missing_indices}"
            )
    totals = {"raw_radio": 0.0, **{name: 0.0 for name in adaptors}}
    count = 0
    field.eval()
    for frame in frames:
        sample = dataset[frame_to_index[frame]]
        result = render_canonical_radio(
            renderer, model, field, sample["pose_w2c"].to(device),
            feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2],
            use_reliability=reliability_splat,
        )
        predicted = result["feature_map"][None].float()
        target = sample["radio_features"].to(device)[None].float()
        valid = result["alpha_map"] >= float(alpha_threshold)
        pixels = int(valid.sum())
        if not pixels:
            continue
        totals["raw_radio"] += float(
            F.cosine_similarity(predicted, target, dim=1)[0][valid].mean()
        ) * pixels
        for name, adaptor in adaptors.items():
            projected = project_feature_map_with_adaptor(predicted, adaptor)
            if capability_teacher_datasets is None:
                teacher = project_feature_map_with_adaptor(target, adaptor)
            else:
                teacher_index = capability_teacher_frame_to_index[name].get(int(frame))
                if teacher_index is None:
                    raise ValueError(
                        f"official {name} teacher does not contain frame {int(frame)}"
                    )
                teacher = _official_capability_teacher_map(
                    capability_teacher_datasets[name], teacher_index, device
                )
                if teacher.shape != projected.shape:
                    raise ValueError(
                        f"official {name} teacher/render shape mismatch: "
                        f"{tuple(teacher.shape)} vs {tuple(projected.shape)}"
                    )
            totals[name] += float((projected * teacher).sum(dim=1)[0][valid].mean()) * pixels
        count += pixels
    if count <= 0:
        raise RuntimeError("multicapability validation rendered no visible pixels")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def _primitive_probe_cosine(
    field,
    consensus: PrimitiveConsensus,
    rows: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 16384,
) -> float:
    values: list[torch.Tensor] = []
    field.eval()
    for start in range(0, rows.numel(), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        predicted = field.radio_features(batch.to(device)).float().cpu()
        target = consensus.targets[batch].float()
        values.append(F.cosine_similarity(predicted, target, dim=-1, eps=1e-8))
    return float(torch.cat(values).mean())


@torch.no_grad()
def _mean_semantic_view_metrics(
    field,
    model,
    renderer,
    dataset,
    frame_to_index: dict[int, int],
    frames: list[int],
    device,
    *,
    bridge: GlobalRegionSummaryBridge,
    summary_head: torch.nn.Module,
    teacher_root: Path,
    scene: str,
    kernel_sizes: tuple[int, ...],
    projection_batch_size: int,
    reliability_splat: bool,
    alpha_threshold: float,
) -> tuple[float, float]:
    absolute_weighted = 0.0
    centered_weighted = 0.0
    count = 0
    field.eval()
    for frame in frames:
        sample = dataset[frame_to_index[frame]]
        result = render_canonical_radio(
            renderer,
            model,
            field,
            sample["pose_w2c"].to(device),
            feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2],
            use_reliability=reliability_splat,
        )
        predicted = project_dense_region_semantics(
            bridge,
            summary_head,
            gauge_separated_radio(result["feature_map"][None], feature_dim=1),
            kernel_sizes=kernel_sizes,
            projection_batch_size=projection_batch_size,
        )
        teacher = _load_semantic_teacher(teacher_root, scene, frame, device)
        absolute_loss, centered_loss, pixels = _semantic_fidelity_losses(
            predicted,
            teacher,
            result["alpha_map"],
            alpha_threshold=alpha_threshold,
        )
        absolute_weighted += (1.0 - float(absolute_loss)) * pixels
        centered_weighted += (1.0 - float(centered_loss)) * pixels
        count += pixels
    return (
        absolute_weighted / max(1, count),
        centered_weighted / max(1, count),
    )


def _trainable_parameters(field, args: argparse.Namespace) -> list[torch.nn.Parameter]:
    for parameter in field.parameters():
        parameter.requires_grad_(False)
    field.local_codes.requires_grad_(True)
    selected: list[torch.nn.Parameter] = [field.local_codes]
    if bool(args.train_basis):
        field.decoder.basis.requires_grad_(True)
        selected.append(field.decoder.basis)
    if bool(args.train_fusion) and field.fusion is not None:
        for parameter in field.fusion.parameters():
            parameter.requires_grad_(True)
            selected.append(parameter)
    return selected


def finetune(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    field, payload, _factorized_signature = (
        load_factorized_canonical_field_checkpoint(
            args.field_checkpoint, map_location="cpu"
        )
    )
    expected_hash = str(payload.get("geometry_fingerprint", {}).get("xyz_sha256", ""))
    if expected_hash != _sha256_tensor_rows(model.get_xyz()):
        raise ValueError("canonical field and geometry rows differ")
    field = field.to(device)
    mpr_cache_path = str(args.mpr_cache or payload["mpr_cache"])
    consensus, mpr_cache = _load_consensus(mpr_cache_path)
    mpr_geometry_hash = str(
        mpr_cache.get("geometry_fingerprint", {}).get("xyz_sha256", "")
    )
    if mpr_geometry_hash != expected_hash:
        raise ValueError("MPR override and canonical field geometry rows differ")
    mpr_metadata = dict(mpr_cache.get("metadata", {}))
    if field.reliability.shape[1] > 0:
        if field.reliability.shape != consensus.reliability.shape:
            raise ValueError("MPR override reliability rows do not match canonical field")
        with torch.no_grad():
            field.reliability.copy_(consensus.reliability.to(device))
    included_frames = sorted(_parse_frame_ids(args.include_frame_ids))
    dataset = _dataset(config, renderer, included_frames or None)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    mpr_frames = [
        int(frame)
        for frame in mpr_metadata.get("selected_frame_indices", [])
        if int(frame) in frame_to_index
    ]
    if not mpr_frames:
        raise RuntimeError("field checkpoint has no row-aligned MPR training frames")
    excluded_from_field_training = _excluded_mpr_frame_ids(mpr_metadata)
    declared_validation = _parse_frame_ids(args.validation_frame_ids)
    if declared_validation:
        missing = declared_validation - excluded_from_field_training
        if missing:
            raise ValueError(
                "validation frames must have been excluded from MPR construction: "
                f"{sorted(missing)}"
            )
        validation_frames = sorted(
            frame for frame in declared_validation if frame in frame_to_index
        )
    else:
        training_pool = [
            int(frame)
            for frame in dataset.frame_indices
            if int(frame) not in excluded_from_field_training
        ]
        validation_candidates = [
            frame for frame in training_pool if frame not in set(mpr_frames)
        ]
        validation_frames = _even_subset(validation_candidates, int(args.validation_views))
    if not validation_frames:
        raise RuntimeError("no non-benchmark validation views remain")
    benchmark_frames = excluded_from_field_training - set(validation_frames)
    raw_training_pool = [
        int(frame)
        for frame in dataset.frame_indices
        if int(frame) not in excluded_from_field_training
    ]
    if args.render_view_policy == "mpr":
        render_train_frames = list(mpr_frames)
    else:
        held_out = set(validation_frames)
        render_train_frames = [frame for frame in raw_training_pool if frame not in held_out]
    if not render_train_frames:
        raise RuntimeError("no raw render training views remain")

    semantic_enabled = bool(str(args.semantic_teacher_root).strip())
    if float(args.semantic_weight) > 0.0 and not semantic_enabled:
        raise ValueError("--semantic-teacher-root is required when semantic loss is enabled")
    if args.selection_policy == "semantic_capability" and not semantic_enabled:
        raise ValueError("semantic_capability selection requires semantic teachers")
    semantic_bridge = None
    semantic_summary_head = None
    semantic_teacher_root = Path(args.semantic_teacher_root) if semantic_enabled else None
    semantic_scene = str(args.semantic_scene).strip() or Path(config.scene_root).name
    semantic_kernel_sizes = tuple(
        int(value) for value in str(args.semantic_kernel_sizes).split(",") if value.strip()
    )
    if semantic_enabled:
        if not args.semantic_bridge_checkpoint:
            raise ValueError("--semantic-bridge-checkpoint is required for semantic teachers")
        semantic_bridge, _semantic_manifest = GlobalRegionSummaryBridge.from_checkpoint(
            args.semantic_bridge_checkpoint, map_location="cpu"
        )
        semantic_bridge = semantic_bridge.to(device).eval()
        semantic_summary_head = SigLIP2SummaryHead.from_radio_checkpoint(
            args.radio_checkpoint
        ).to(device).eval()
        for module in (semantic_bridge, semantic_summary_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for frame in set(render_train_frames) | set(validation_frames):
            _semantic_teacher_path(semantic_teacher_root, semantic_scene, frame)

    capability_weights = {
        "siglip2-g": float(args.siglip_spatial_render_weight),
        "dino_v3": float(args.dino_render_weight),
        "sam3": float(args.sam3_render_weight),
    }
    capability_weights = {
        name: weight for name, weight in capability_weights.items() if weight > 0
    }
    capability_adaptors: dict[str, torch.nn.Module] = {}
    for name in capability_weights:
        adaptor = (
            SigLIP2FeatureProjection.from_radio_checkpoint(args.radio_checkpoint)
            if name == "siglip2-g"
            else load_radio_adaptor_from_checkpoint(
                args.radio_checkpoint, name, kind="feature_projection"
            )
        ).to(device).eval()
        for parameter in adaptor.parameters():
            parameter.requires_grad_(False)
        capability_adaptors[name] = adaptor
    capability_teacher_datasets: dict[str, SimpleRadioDataset] | None = None
    capability_teacher_frame_to_index: dict[str, dict[int, int]] | None = None
    capability_teacher_provenance: dict[str, dict[str, object]] = {}
    if (
        capability_adaptors
        and args.capability_map_source == "official_extracted"
    ):
        (
            capability_teacher_datasets,
            capability_teacher_frame_to_index,
            capability_teacher_provenance,
        ) = _load_official_capability_teacher_datasets(
            config,
            renderer,
            dataset,
            list(capability_adaptors),
        )

    optimizer = torch.optim.AdamW(
        _trainable_parameters(field, args),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    valid_rows = torch.where(consensus.valid)[0]
    probe_count = min(valid_rows.numel(), int(args.mpr_validation_rows))
    probe_order = torch.randperm(valid_rows.numel(), generator=generator)[:probe_count]
    mpr_probe_rows = valid_rows[probe_order]
    reliability_splat = bool(args.reliability_splat)
    if reliability_splat and field.reliability.shape[1] == 0:
        raise ValueError(
            "factorized field has no persistent reliability columns; "
            "--reliability-splat is unavailable"
        )
    initial_validation = _mean_view_cosine(
        field,
        model,
        renderer,
        dataset,
        frame_to_index,
        validation_frames,
        device,
        reliability_splat=reliability_splat,
    )
    best_validation = initial_validation
    initial_capability_validation = _mean_multicapability_fidelity(
        field, model, renderer, dataset, frame_to_index, validation_frames, device,
        adaptors=capability_adaptors, reliability_splat=reliability_splat,
        alpha_threshold=float(args.alpha_threshold),
        capability_teacher_datasets=capability_teacher_datasets,
        capability_teacher_frame_to_index=capability_teacher_frame_to_index,
    )
    best_capability_validation = dict(initial_capability_validation)
    initial_mpr_probe = _primitive_probe_cosine(
        field, consensus, mpr_probe_rows, device
    )
    best_mpr_probe = initial_mpr_probe
    initial_semantic_validation = None
    initial_semantic_absolute = None
    initial_semantic_centered = None
    best_semantic_validation = None
    if semantic_enabled:
        initial_semantic_absolute, initial_semantic_centered = _mean_semantic_view_metrics(
            field,
            model,
            renderer,
            dataset,
            frame_to_index,
            validation_frames,
            device,
            bridge=semantic_bridge,
            summary_head=semantic_summary_head,
            teacher_root=semantic_teacher_root,
            scene=semantic_scene,
            kernel_sizes=semantic_kernel_sizes,
            projection_batch_size=int(args.semantic_projection_batch_size),
            reliability_splat=reliability_splat,
            alpha_threshold=float(args.alpha_threshold),
        )
        initial_semantic_validation = 0.5 * (
            initial_semantic_absolute + initial_semantic_centered
        )
        best_semantic_validation = initial_semantic_validation
    best_step = 0
    best_state = copy.deepcopy(field.state_dict())
    history: list[dict] = []
    shuffled: list[int] = []

    for step in range(int(args.steps)):
        if not shuffled:
            order = torch.randperm(len(render_train_frames), generator=generator).tolist()
            shuffled = [render_train_frames[index] for index in order]
        frame = shuffled.pop()
        sample = dataset[frame_to_index[frame]]
        optimizer.zero_grad(set_to_none=True)
        field.train()
        result = render_canonical_radio(
            renderer,
            model,
            field,
            sample["pose_w2c"].to(device),
            feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2],
            use_reliability=reliability_splat,
        )
        teacher = sample["radio_features"].to(device)[None]
        render_loss = normalized_render_reconstruction_loss(
            result["feature_map"][None],
            teacher,
            result["alpha_map"][None],
            alpha_threshold=float(args.alpha_threshold),
            cosine_weight=1.0,
            huber_weight=float(args.render_huber_weight),
        )
        chosen = valid_rows[
            torch.randint(
                valid_rows.numel(),
                (int(args.mpr_batch_size),),
                generator=generator,
            )
        ]
        predicted_rows = field.radio_features(chosen.to(device))
        mpr_loss, _mpr_stats = primitive_reconstruction_loss(
            predicted_rows, consensus, row_indices=chosen
        )
        capability_alignment_loss = render_loss.detach() * 0.0
        capability_local_affinity_loss = render_loss.detach() * 0.0
        capability_stats: dict[str, dict[str, torch.Tensor]] = {}
        if capability_adaptors:
            teacher_capability_maps = None
            if capability_teacher_datasets is not None:
                teacher_capability_maps = {}
                for name, teacher_dataset in capability_teacher_datasets.items():
                    teacher_index = capability_teacher_frame_to_index[name].get(int(frame))
                    if teacher_index is None:
                        raise ValueError(
                            f"official {name} teacher does not contain frame {int(frame)}"
                        )
                    teacher_capability_maps[name] = _official_capability_teacher_map(
                        teacher_dataset, teacher_index, device
                    )
            (
                capability_alignment_loss,
                capability_local_affinity_loss,
                capability_stats,
            ) = compute_radio_adaptor_masked_render_losses(
                result["feature_map"][None],
                teacher,
                capability_adaptors,
                result["alpha_map"][None] >= float(args.alpha_threshold),
                adaptor_weights=capability_weights,
                local_radius=int(args.capability_local_radius),
                local_balance_quantile=float(args.capability_local_balance_quantile),
                teacher_capability_maps=teacher_capability_maps,
            )
        semantic_absolute_loss = render_loss.detach() * 0.0
        semantic_centered_loss = render_loss.detach() * 0.0
        semantic_loss = render_loss.detach() * 0.0
        if semantic_enabled and float(args.semantic_weight) > 0.0:
            predicted_semantics = project_dense_region_semantics(
                semantic_bridge,
                semantic_summary_head,
                gauge_separated_radio(
                    result["feature_map"][None], feature_dim=1
                ),
                kernel_sizes=semantic_kernel_sizes,
                projection_batch_size=int(args.semantic_projection_batch_size),
            )
            semantic_teacher = _load_semantic_teacher(
                semantic_teacher_root, semantic_scene, frame, device
            )
            (
                semantic_absolute_loss,
                semantic_centered_loss,
                _semantic_pixels,
            ) = _semantic_fidelity_losses(
                predicted_semantics,
                semantic_teacher,
                result["alpha_map"],
                alpha_threshold=float(args.alpha_threshold),
            )
            semantic_loss = (
                semantic_absolute_loss
                + float(args.semantic_centered_weight) * semantic_centered_loss
            )
        capability_scale = sum(capability_weights.values())
        loss = (
            render_loss
            + float(args.mpr_weight) * mpr_loss
            + capability_scale * capability_alignment_loss
            + capability_scale
            * float(args.capability_local_affinity_weight)
            * capability_local_affinity_loss
            + float(args.semantic_weight) * semantic_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), float(args.grad_clip))
        optimizer.step()

        should_validate = (
            (step + 1) % int(args.log_every) == 0
            or step == 0
            or step + 1 == int(args.steps)
        )
        if should_validate:
            validation_cosine = _mean_view_cosine(
                field,
                model,
                renderer,
                dataset,
                frame_to_index,
                validation_frames,
                device,
                reliability_splat=reliability_splat,
            )
            mpr_probe_cosine = _primitive_probe_cosine(
                field, consensus, mpr_probe_rows, device
            )
            capability_validation = _mean_multicapability_fidelity(
                field, model, renderer, dataset, frame_to_index, validation_frames,
                device, adaptors=capability_adaptors,
                reliability_splat=reliability_splat,
                alpha_threshold=float(args.alpha_threshold),
                capability_teacher_datasets=capability_teacher_datasets,
                capability_teacher_frame_to_index=capability_teacher_frame_to_index,
            )
            semantic_validation = None
            semantic_validation_absolute = None
            semantic_validation_centered = None
            if semantic_enabled:
                (
                    semantic_validation_absolute,
                    semantic_validation_centered,
                ) = _mean_semantic_view_metrics(
                    field,
                    model,
                    renderer,
                    dataset,
                    frame_to_index,
                    validation_frames,
                    device,
                    bridge=semantic_bridge,
                    summary_head=semantic_summary_head,
                    teacher_root=semantic_teacher_root,
                    scene=semantic_scene,
                    kernel_sizes=semantic_kernel_sizes,
                    projection_batch_size=int(args.semantic_projection_batch_size),
                    reliability_splat=reliability_splat,
                    alpha_threshold=float(args.alpha_threshold),
                )
                semantic_validation = 0.5 * (
                    semantic_validation_absolute + semantic_validation_centered
                )
            if args.selection_policy == "final":
                selected = True
            elif args.selection_policy == "validation":
                selected = validation_cosine > best_validation
            elif args.selection_policy == "raw_fidelity":
                selected = (
                    validation_cosine > best_validation
                    and mpr_probe_cosine
                    >= initial_mpr_probe - float(args.max_mpr_drop)
                )
            elif args.selection_policy == "pareto_mpr":
                selected = (
                    validation_cosine
                    >= initial_validation - float(args.max_validation_drop)
                    and mpr_probe_cosine > best_mpr_probe
                )
            elif args.selection_policy == "capability_pareto":
                current_score = sum(capability_validation.values())
                best_score = sum(best_capability_validation.values())
                selected = (
                    current_score > best_score
                    and all(
                        capability_validation[name]
                        >= initial_capability_validation[name]
                        - float(args.max_capability_drop)
                        for name in initial_capability_validation
                    )
                    and mpr_probe_cosine
                    >= initial_mpr_probe - float(args.max_mpr_drop)
                )
            else:
                selected = (
                    semantic_validation is not None
                    and semantic_validation > best_semantic_validation
                    and validation_cosine
                    >= initial_validation - float(args.max_validation_drop)
                    and mpr_probe_cosine
                    >= initial_mpr_probe - float(args.max_mpr_drop)
                )
            if selected:
                best_validation = validation_cosine
                best_mpr_probe = mpr_probe_cosine
                if semantic_validation is not None:
                    best_semantic_validation = semantic_validation
                best_capability_validation = dict(capability_validation)
                best_step = step + 1
                best_state = copy.deepcopy(field.state_dict())
            record = {
                "step": step + 1,
                "frame_id": frame,
                "loss": float(loss.detach()),
                "render_loss": float(render_loss.detach()),
                "mpr_loss": float(mpr_loss.detach()),
                "capability_alignment_loss": float(
                    capability_alignment_loss.detach()
                ),
                "capability_local_affinity_loss": float(
                    capability_local_affinity_loss.detach()
                ),
                "capability_adaptors": {
                    name: {
                        "alignment": float(values["alignment"].detach()),
                        "local_affinity": float(
                            values["local_affinity"].detach()
                        ),
                    }
                    for name, values in capability_stats.items()
                },
                "semantic_loss": float(semantic_loss.detach()),
                "semantic_absolute_loss": float(semantic_absolute_loss.detach()),
                "semantic_centered_loss": float(semantic_centered_loss.detach()),
                "validation_cosine": validation_cosine,
                "mpr_probe_cosine": mpr_probe_cosine,
                "capability_validation_cosine": capability_validation,
                "best_capability_validation_cosine": best_capability_validation,
                "semantic_validation_cosine": semantic_validation,
                "semantic_validation_absolute_cosine": semantic_validation_absolute,
                "semantic_validation_centered_cosine": semantic_validation_centered,
                "best_validation_cosine": best_validation,
                "best_mpr_probe_cosine": best_mpr_probe,
                "best_semantic_validation_cosine": best_semantic_validation,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    field.load_state_dict(best_state, strict=True)
    train_probe_frames = _even_subset(render_train_frames, len(validation_frames))
    final_train_probe = _mean_view_cosine(
        field,
        model,
        renderer,
        dataset,
        frame_to_index,
        train_probe_frames,
        device,
        reliability_splat=reliability_splat,
    )
    field.eval().cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["state_dict"] = field.state_dict()
    # Schema-v2 factorized fields deliberately keep target reliability out of
    # persistent scene state.  It may weight the MPR training objective above,
    # but must never be copied into the saved deployment checkpoint.
    payload["reliability"] = field.reliability.detach().cpu()
    payload["mpr_cache"] = str(Path(mpr_cache_path).resolve())
    payload["mpr_cache_metadata"] = mpr_metadata
    payload["render_optimization"] = {
        "config": str(Path(args.config).resolve()),
        "geometry_checkpoint": str(Path(args.geometry_checkpoint).resolve()),
        "render_view_policy": args.render_view_policy,
        "render_train_frames": render_train_frames,
        "mpr_anchor_frames": mpr_frames,
        "validation_frames": validation_frames,
        "excluded_benchmark_frames": sorted(benchmark_frames),
        "excluded_from_field_training": sorted(excluded_from_field_training),
        "initial_validation_cosine": initial_validation,
        "initial_mpr_probe_cosine": initial_mpr_probe,
        "initial_capability_validation_cosine": initial_capability_validation,
        "initial_semantic_validation_cosine": initial_semantic_validation,
        "initial_semantic_absolute_cosine": initial_semantic_absolute,
        "initial_semantic_centered_cosine": initial_semantic_centered,
        "best_validation_cosine": best_validation,
        "best_mpr_probe_cosine": best_mpr_probe,
        "best_capability_validation_cosine": best_capability_validation,
        "best_semantic_validation_cosine": best_semantic_validation,
        "best_step": best_step,
        "selection_policy": args.selection_policy,
        "max_validation_drop": float(args.max_validation_drop),
        "max_mpr_drop": float(args.max_mpr_drop),
        "max_capability_drop": float(args.max_capability_drop),
        "semantic_capability": {
            "enabled": semantic_enabled,
            "teacher_root": str(semantic_teacher_root.resolve()) if semantic_enabled else "",
            "scene": semantic_scene,
            "bridge_checkpoint": str(Path(args.semantic_bridge_checkpoint).resolve())
            if semantic_enabled
            else "",
            "radio_checkpoint": str(Path(args.radio_checkpoint).resolve())
            if semantic_enabled
            else "",
            "kernel_sizes": list(semantic_kernel_sizes),
            "weight": float(args.semantic_weight),
            "centered_weight": float(args.semantic_centered_weight),
            "selection_score": "mean(absolute_cosine, centered_cosine)",
            "uses_benchmark_masks": False,
            "uses_text_queries": False,
        },
        "official_render_capability": {
            "enabled": bool(capability_adaptors),
            "radio_checkpoint": (
                str(Path(args.radio_checkpoint).resolve())
                if capability_adaptors
                else ""
            ),
            "adaptor_weights": capability_weights,
            "teacher_map_source": args.capability_map_source,
            "teacher_map_provenance": capability_teacher_provenance,
            "dense_alignment": bool(capability_adaptors),
            "local_affinity_weight": float(
                args.capability_local_affinity_weight
            ),
            "local_radius": int(args.capability_local_radius),
            "local_balance_quantile": float(
                args.capability_local_balance_quantile
            ),
            "projection_order": (
                "complete_rendered_2d_grid_vs_resample(official_runtime_adaptor_output)"
                if args.capability_map_source == "official_extracted"
                else "complete_rendered_2d_grid_then_official_adaptor"
            ),
            "visibility_domain": f"rendered_alpha>={float(args.alpha_threshold)}",
            "custom_adaptor_head": False,
            "uses_benchmark_masks": False,
            "uses_text_queries": False,
        },
        "train_probe_frames": train_probe_frames,
        "train_probe_cosine": final_train_probe,
        "reliability_splat": reliability_splat,
        "train_basis": bool(args.train_basis),
        "train_fusion": bool(args.train_fusion),
        "history": history,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "text_queries_opened": False,
        "canonical_contract": {
            "name": "canonical-mpr-v2",
            "definition": "canonical-mpr-v1 initialization + exact render replay + render-matched coefficient fitting + primitive MPR prior",
            "exact_training_renderer_matches_inference": True,
            "primitive_prior_weight": float(args.mpr_weight),
            "selection_uses_nonbenchmark_validation_only": True,
        },
    }
    torch.save(payload, output)
    report = {
        "output": str(output),
        "steps": int(args.steps),
        "initial_validation_cosine": initial_validation,
        "initial_mpr_probe_cosine": initial_mpr_probe,
        "initial_capability_validation_cosine": initial_capability_validation,
        "initial_semantic_validation_cosine": initial_semantic_validation,
        "initial_semantic_absolute_cosine": initial_semantic_absolute,
        "initial_semantic_centered_cosine": initial_semantic_centered,
        "best_validation_cosine": best_validation,
        "best_mpr_probe_cosine": best_mpr_probe,
        "best_capability_validation_cosine": best_capability_validation,
        "best_semantic_validation_cosine": best_semantic_validation,
        "best_step": best_step,
        "train_probe_cosine": final_train_probe,
        "num_render_train_frames": len(render_train_frames),
        "validation_frames": validation_frames,
        "excluded_benchmark_frames": sorted(benchmark_frames),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--field-checkpoint", required=True)
    parser.add_argument(
        "--mpr-cache",
        default="",
        help="Optional row-verified MPR target override for staged raw→MPR training.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--mpr-weight", type=float, default=0.10)
    parser.add_argument("--max-mpr-drop", type=float, default=0.0)
    parser.add_argument("--mpr-batch-size", type=int, default=4096)
    parser.add_argument("--mpr-validation-rows", type=int, default=32768)
    parser.add_argument("--render-huber-weight", type=float, default=0.0)
    parser.add_argument(
        "--siglip-spatial-render-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for query-free full-grid alignment through the official "
            "SigLIP2 spatial projection; this never applies the summary head "
            "to individual pixels or Gaussians."
        ),
    )
    parser.add_argument(
        "--dino-render-weight",
        type=float,
        default=0.0,
        help="Weight for query-free dense alignment through the official DINOv3 adaptor.",
    )
    parser.add_argument(
        "--sam3-render-weight",
        type=float,
        default=0.0,
        help="Weight for query-free dense alignment through the official SAM3 adaptor.",
    )
    parser.add_argument(
        "--capability-local-affinity-weight",
        type=float,
        default=0.25,
        help="Local-relation loss relative to each enabled capability alignment term.",
    )
    parser.add_argument("--capability-local-radius", type=int, default=1)
    parser.add_argument(
        "--capability-local-balance-quantile",
        type=float,
        default=0.0,
        help=(
            "Query-free teacher-affinity tail fraction. Values above zero "
            "balance discontinuity and interior relation errors."
        ),
    )
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--semantic-centered-weight", type=float, default=1.0)
    parser.add_argument("--semantic-teacher-root", default="")
    parser.add_argument("--semantic-scene", default="")
    parser.add_argument("--semantic-bridge-checkpoint", default="")
    parser.add_argument("--semantic-kernel-sizes", default="3,7,15")
    parser.add_argument("--semantic-projection-batch-size", type=int, default=2048)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument(
        "--capability-map-source",
        choices=["project_raw", "official_extracted"],
        default="project_raw",
        help=(
            "Teacher source for SigLIP2/DINO/SAM render losses. 'official_extracted' "
            "requires extractor-produced native official adaptor maps and "
            "only resamples them to the render grid."
        ),
    )
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--validation-views", type=int, default=4)
    parser.add_argument(
        "--validation-frame-ids",
        default="",
        help="Comma list or commented text file; every dev frame must already be excluded from MPR.",
    )
    parser.add_argument(
        "--include-frame-ids",
        default="",
        help="Optional registered-frame allowlist applied before pose loading.",
    )
    parser.add_argument(
        "--render-view-policy", choices=["all_nonbenchmark", "mpr"], default="all_nonbenchmark"
    )
    parser.add_argument("--reliability-splat", action="store_true")
    parser.add_argument("--train-basis", action="store_true")
    parser.add_argument("--train-fusion", action="store_true")
    parser.add_argument(
        "--selection-policy",
        choices=[
            "final",
            "validation",
            "raw_fidelity",
            "pareto_mpr",
            "semantic_capability",
            "capability_pareto",
        ],
        default="validation",
    )
    parser.add_argument("--max-validation-drop", type=float, default=0.0)
    parser.add_argument("--max-capability-drop", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(
        args.siglip_spatial_render_weight,
        args.dino_render_weight,
        args.sam3_render_weight,
        args.capability_local_affinity_weight,
    ) < 0:
        parser.error("capability render weights cannot be negative")
    if args.capability_local_radius <= 0:
        parser.error("--capability-local-radius must be positive")
    print(json.dumps(finetune(args), indent=2))


if __name__ == "__main__":
    main()
