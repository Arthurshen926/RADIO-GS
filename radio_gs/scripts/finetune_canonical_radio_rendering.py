#!/usr/bin/env python3
"""Solve a canonical coefficient field with query-free multiview render loss."""

from __future__ import annotations

import argparse
import gc
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
    align_full_extent_feature_grid,
    project_dense_region_semantics,
)
from radio_gs.losses.radio_adaptor_loss import (
    compute_radio_adaptor_masked_render_losses,
)
from radio_gs.losses.generic_region_text_response import (
    FrozenGenericRegionTextBundle,
    generic_region_text_response_loss,
    load_frozen_generic_region_text_bundle,
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
from radio_gs.training.canonical_field_losses import (
    normalized_render_reconstruction_loss,
)
from radio_gs.training.feature_training_utils import SimpleRadioDataset
from radio_gs.training.primitive_consensus import (
    PrimitiveConsensus,
    primitive_reconstruction_loss,
)
from radio_gs.utils.immutable_artifacts import load_torch_mapping, sha256_file


METHOD_V1_STAGES = [
    "factorized_d512_l512",
    "official_siglip2_full_grid",
    "genuine_source_crop_region_summary",
    "target_blind_generic_text_response",
]


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot selection state off device without sharing mutable storage."""

    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _copy_state_dict_to_cpu_(
    snapshot: dict[str, torch.Tensor], module: torch.nn.Module
) -> None:
    """Refresh a fixed-shape CPU selection snapshot without duplicating it."""

    current = module.state_dict()
    if set(snapshot) != set(current):
        raise ValueError("selection snapshot keys differ from module state")
    for name, value in current.items():
        target = snapshot[name]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"selection snapshot tensor differs for {name}")
        target.copy_(value.detach(), non_blocking=False)


def _metadata_only_parent_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Drop parent tensor copies once their field has been reconstructed."""

    metadata = dict(payload)
    metadata.pop("state_dict", None)
    metadata.pop("reliability", None)
    return metadata


def _release_validation_cuda_cache(
    device: torch.device,
    *,
    enabled: bool,
    should_validate: bool,
) -> None:
    """Return validation-only cached blocks without changing tensor values."""

    if bool(enabled) and bool(should_validate) and device.type == "cuda":
        torch.cuda.empty_cache()


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move Adam state tensors without changing their values or update rule."""

    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value) and value.device != device:
                state[name] = value.to(device=device)


@torch.no_grad()
def _offloaded_adamw_step(
    optimizer: torch.optim.AdamW,
    *,
    chunk_elements: int,
) -> None:
    """Apply AdamW exactly while staging CPU moments through bounded chunks."""

    if int(chunk_elements) <= 0:
        raise ValueError("optimizer-state chunk size must be positive")
    for group in optimizer.param_groups:
        if (
            bool(group.get("amsgrad", False))
            or bool(group.get("maximize", False))
            or bool(group.get("capturable", False))
            or bool(group.get("differentiable", False))
            or bool(group.get("fused", False))
        ):
            raise ValueError("chunked offloaded AdamW received unsupported options")
        beta1, beta2 = group["betas"]
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        epsilon = float(group["eps"])
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            if parameter.is_complex():
                raise ValueError("chunked offloaded AdamW forbids complex parameters")
            state = optimizer.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(
                    parameter,
                    device="cpu",
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    device="cpu",
                    memory_format=torch.preserve_format,
                )
            step_tensor = state["step"]
            if not torch.is_tensor(step_tensor) or step_tensor.device.type != "cpu":
                raise ValueError("offloaded AdamW step must remain on CPU")
            step_tensor.add_(1)
            step = float(step_tensor.item())
            bias_correction1 = 1.0 - beta1**step
            bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
            step_size = learning_rate / bias_correction1

            parameter_flat = parameter.view(-1)
            gradient_flat = gradient.view(-1)
            exp_avg_cpu = state["exp_avg"].view(-1)
            exp_avg_sq_cpu = state["exp_avg_sq"].view(-1)
            if (
                exp_avg_cpu.device.type != "cpu"
                or exp_avg_sq_cpu.device.type != "cpu"
            ):
                raise ValueError("AdamW moments must remain on CPU between steps")
            for start in range(0, parameter_flat.numel(), int(chunk_elements)):
                stop = min(start + int(chunk_elements), parameter_flat.numel())
                parameter_chunk = parameter_flat[start:stop]
                gradient_chunk = gradient_flat[start:stop]
                exp_avg = exp_avg_cpu[start:stop].to(parameter.device)
                exp_avg_sq = exp_avg_sq_cpu[start:stop].to(parameter.device)

                parameter_chunk.mul_(1.0 - learning_rate * weight_decay)
                exp_avg.mul_(beta1).add_(gradient_chunk, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient_chunk,
                    gradient_chunk,
                    value=1.0 - beta2,
                )
                denominator = (
                    exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(epsilon)
                )
                parameter_chunk.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )
                exp_avg_cpu[start:stop].copy_(exp_avg)
                exp_avg_sq_cpu[start:stop].copy_(exp_avg_sq)


def _zero_optimizer_gradients(
    optimizer: torch.optim.Optimizer,
    *,
    preserve_buffers: bool,
) -> None:
    """Reuse giant local-code gradients when reallocating them cannot fit."""

    optimizer.zero_grad(set_to_none=not bool(preserve_buffers))


def _optimizer_state_offload_enabled(
    *, requested: bool, semantic_enabled: bool
) -> bool:
    """Protect dense semantic stages from Adam-state/teacher GPU overlap."""

    return bool(requested) or bool(semantic_enabled)


def _staged_feature_branch_gradient(
    branch_loss: torch.Tensor,
    feature_map: torch.Tensor,
) -> torch.Tensor:
    """Collapse a large frozen-head branch to its feature-map gradient.

    The returned tensor is the exact first-order gradient.  Once this call
    completes, autograd can release the dense semantic projection graph before
    allocating the much larger primitive-local gradient.
    """

    (gradient,) = torch.autograd.grad(
        branch_loss,
        feature_map,
        retain_graph=False,
        create_graph=False,
    )
    return gradient.detach()


def _backward_base_with_feature_gradient(
    base_loss: torch.Tensor,
    feature_map: torch.Tensor,
    feature_gradient: torch.Tensor,
) -> None:
    """Backpropagate a base loss plus a precomputed feature-map gradient."""

    if feature_gradient.shape != feature_map.shape:
        raise ValueError("staged feature gradient shape differs")
    torch.autograd.backward(
        (base_loss, feature_map),
        grad_tensors=(None, feature_gradient),
    )


def _payload_method_v1_stage(payload: Mapping[str, object]) -> str:
    """Infer the most advanced completed Method-v1 stage in one field."""

    render = payload.get("render_optimization")
    if not isinstance(render, Mapping):
        architecture = payload.get("architecture")
        if not isinstance(architecture, Mapping) or (
            int(architecture.get("coefficient_dim", -1)) != 512
            or int(architecture.get("local_dim", -1)) != 512
        ):
            raise ValueError("Method-v1 base field is not D512/L512")
        return "factorized_d512_l512"
    generic = render.get("generic_text_response")
    if isinstance(generic, Mapping) and generic.get("enabled") is True:
        return "target_blind_generic_text_response"
    semantic = render.get("semantic_capability")
    if isinstance(semantic, Mapping) and semantic.get("enabled") is True:
        return "genuine_source_crop_region_summary"
    official = render.get("official_render_capability")
    if isinstance(official, Mapping) and (
        float(official.get("adaptor_weights", {}).get("siglip2-g", 0.0)) > 0.0
    ):
        return "official_siglip2_full_grid"
    raise ValueError("cannot infer Method-v1 stage from field payload")


def _lineage_record(
    path: Path,
    digest: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    render = payload.get("render_optimization")
    return {
        "stage": _payload_method_v1_stage(payload),
        "field": str(path),
        "sha256": digest,
        "selection_policy": (
            str(render.get("selection_policy", ""))
            if isinstance(render, Mapping)
            else "mapping_only_checkpoint_rule"
        ),
        "best_step": (
            int(render.get("best_step", 0)) if isinstance(render, Mapping) else 0
        ),
    }


def _method_v1_predecessor_lineage(
    args: argparse.Namespace,
    parent_payload: Mapping[str, object],
    *,
    current_stage: str,
    parent_sha256: str | None = None,
) -> list[dict[str, object]]:
    """Bind every predecessor field by content before saving a new stage."""

    parent_path = Path(args.field_checkpoint).expanduser().resolve()
    parent_sha256 = parent_sha256 or sha256_file(parent_path)
    existing = parent_payload.get("method_v1_construction_lineage")
    if existing is None:
        lineage: list[dict[str, object]] = []
    elif isinstance(existing, list) and all(
        isinstance(record, Mapping) for record in existing
    ):
        lineage = [dict(record) for record in existing]
    else:
        raise ValueError("Method-v1 construction lineage is malformed")

    if not lineage:
        training = parent_payload.get("training_config")
        if not isinstance(training, Mapping):
            raise ValueError("Method-v1 field lacks base training identity")
        base_value = str(training.get("output", "")).strip()
        if not base_value:
            raise ValueError("Method-v1 field lacks base checkpoint path")
        base_path = Path(base_value).expanduser().resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"Method-v1 base field is missing: {base_path}")
        lineage.append(
            {
                "stage": "factorized_d512_l512",
                "field": str(base_path),
                "sha256": sha256_file(base_path),
                "selection_policy": "mapping_only_checkpoint_rule",
                "best_step": 0,
            }
        )
        for value in args.construction_prior_field:
            prior_path = Path(value).expanduser().resolve()
            prior_payload, prior_sha256, _ = load_torch_mapping(
                prior_path,
                map_location="cpu",
                label="Method-v1 prior field",
            )
            lineage.append(_lineage_record(prior_path, prior_sha256, prior_payload))

    parent_stage = _payload_method_v1_stage(parent_payload)
    if parent_stage != "factorized_d512_l512":
        lineage.append(_lineage_record(parent_path, parent_sha256, parent_payload))
    elif Path(lineage[0]["field"]) != parent_path:
        raise ValueError("Method-v1 base lineage and parent field differ")

    current_index = METHOD_V1_STAGES.index(current_stage)
    observed_stages = [str(record.get("stage", "")) for record in lineage]
    if observed_stages != METHOD_V1_STAGES[:current_index]:
        raise ValueError(
            "Method-v1 predecessor stages differ: "
            f"expected={METHOD_V1_STAGES[:current_index]}, observed={observed_stages}"
        )
    digests = [str(record.get("sha256", "")) for record in lineage]
    if len(set(digests)) != len(digests) or any(
        len(digest) != 64 for digest in digests
    ):
        raise ValueError("Method-v1 predecessor hashes are missing or duplicated")
    return lineage


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
    pose_file = (
        raw_pose_file if raw_pose_file and Path(raw_pose_file).is_file() else None
    )
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
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, Mapping):
                items = next(
                    (
                        payload[key]
                        for key in ("frame_ids", "frames", "indices")
                        if key in payload
                    ),
                    None,
                )
                if items is None:
                    raise ValueError(
                        f"unsupported JSON frame-id authority: {path}"
                    )
            else:
                raise ValueError(f"unsupported JSON frame-id authority: {path}")
            if not isinstance(items, list):
                raise ValueError(f"JSON frame-id authority is not a list: {path}")
            return {int(item) for item in items}
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return {int(token) for token in tokens}


def _resolve_training_frame_ids(config, cli_value: str) -> set[int]:
    """Resolve the registered source-frame cohort without widening it silently."""

    declared = _parse_frame_ids(cli_value)
    if declared:
        return declared
    configured = str(getattr(config, "train_frame_ids_path", "") or "").strip()
    if not configured:
        return set()
    path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"configured training frame authority is missing: {path}"
        )
    declared = _parse_frame_ids(str(path))
    if not declared:
        raise ValueError(f"configured training frame authority is empty: {path}")
    return declared


def _semantic_teacher_path(root: Path, scene: str, frame: int) -> Path:
    candidates = (
        root / scene / f"rgb_{frame}.pt",
        root / f"rgb_{frame}.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"semantic teacher is missing for frame {frame}: {candidates}"
    )


def _load_semantic_teacher(root: Path, scene: str, frame: int, device) -> torch.Tensor:
    teacher = torch.load(_semantic_teacher_path(root, scene, frame), map_location="cpu")
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
    if teacher.ndim != 3 or teacher.shape[0] != predicted.shape[1]:
        raise ValueError(
            f"semantic teacher/prediction mismatch: {tuple(teacher.shape)} vs "
            f"{tuple(predicted.shape[1:])}"
        )
    predicted_size = tuple(int(value) for value in predicted.shape[-2:])
    teacher = align_full_extent_feature_grid(
        teacher,
        predicted_size,
        label="semantic teacher/prediction mismatch",
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
def _view_cosine(
    field, model, renderer, sample, device, *, reliability_splat: bool
) -> tuple[float, int]:
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
    field,
    model,
    renderer,
    dataset,
    frame_to_index,
    frames,
    device,
    *,
    adaptors: dict[str, torch.nn.Module],
    reliability_splat: bool,
    alpha_threshold: float,
    capability_teacher_datasets: Mapping[str, SimpleRadioDataset] | None = None,
    capability_teacher_frame_to_index: Mapping[str, Mapping[int, int]] | None = None,
    projection_amp: bool = False,
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
            renderer,
            model,
            field,
            sample["pose_w2c"].to(device),
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
        totals["raw_radio"] += (
            float(F.cosine_similarity(predicted, target, dim=1)[0][valid].mean())
            * pixels
        )
        for name, adaptor in adaptors.items():
            projected = project_feature_map_with_adaptor(
                predicted, adaptor, amp=bool(projection_amp)
            )
            if capability_teacher_datasets is None:
                teacher = project_feature_map_with_adaptor(
                    target, adaptor, amp=bool(projection_amp)
                )
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
            totals[name] += (
                float((projected * teacher).sum(dim=1)[0][valid].mean()) * pixels
            )
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


@torch.no_grad()
def _mean_generic_text_response_metrics(
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
    text_bundle: FrozenGenericRegionTextBundle,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "profile": 0.0,
        "profile_cosine": 0.0,
        "listwise": 0.0,
        "sibling": 0.0,
        "synonym": 0.0,
    }
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
        teacher = _load_semantic_teacher(teacher_root, scene, frame, device)[None]
        loss, stats = generic_region_text_response_loss(
            predicted,
            teacher,
            result["alpha_map"],
            text_bundle,
            alpha_threshold=alpha_threshold,
        )
        regions = int(stats["regions"])
        if regions < 2:
            continue
        totals["loss"] += float(loss) * regions
        for name in totals:
            if name != "loss":
                totals[name] += float(stats[name]) * regions
        count += regions
    if count <= 0:
        raise RuntimeError(
            "generic text-response validation has fewer than two regions"
        )
    return {name: value / count for name, value in totals.items()}


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
    parent_field_path = Path(args.field_checkpoint).expanduser().resolve()
    # Hash before loading the multi-GiB state so lineage I/O never extends the
    # period in which both the checkpoint payload and reconstructed field are
    # resident in host memory.
    parent_field_sha256 = sha256_file(parent_field_path)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    field, payload, _factorized_signature = load_factorized_canonical_field_checkpoint(
        args.field_checkpoint, map_location="cpu"
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
            raise ValueError(
                "MPR override reliability rows do not match canonical field"
            )
        with torch.no_grad():
            field.reliability.copy_(consensus.reliability.to(device))
    included_frames = sorted(
        _resolve_training_frame_ids(config, args.include_frame_ids)
    )
    dataset = _dataset(config, renderer, included_frames or None)
    frame_to_index = {
        int(frame): index for index, frame in enumerate(dataset.frame_indices)
    }
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
        validation_frames = _even_subset(
            validation_candidates, int(args.validation_views)
        )
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
        render_train_frames = [
            frame for frame in raw_training_pool if frame not in held_out
        ]
    if not render_train_frames:
        raise RuntimeError("no raw render training views remain")

    semantic_enabled = bool(str(args.semantic_teacher_root).strip())
    optimizer_state_offload = _optimizer_state_offload_enabled(
        requested=bool(args.offload_optimizer_state),
        semantic_enabled=semantic_enabled,
    )
    generic_text_response_enabled = float(args.generic_text_response_weight) > 0.0
    if generic_text_response_enabled:
        current_method_v1_stage = "target_blind_generic_text_response"
    elif semantic_enabled and float(args.semantic_weight) > 0.0:
        current_method_v1_stage = "genuine_source_crop_region_summary"
    elif float(args.siglip_spatial_render_weight) > 0.0:
        current_method_v1_stage = "official_siglip2_full_grid"
    else:
        raise ValueError("fine-tuning invocation does not identify a Method-v1 stage")
    method_v1_predecessor_lineage = _method_v1_predecessor_lineage(
        args,
        payload,
        current_stage=current_method_v1_stage,
        parent_sha256=parent_field_sha256,
    )
    # ``load_state_dict`` reconstructed the field before it moved to CUDA, so
    # the checkpoint payload still owns a second full CPU copy.  Retain only
    # metadata; the selected field state is reattached when the child stage is
    # sealed.  This is material for million-row D512/L512 scenes.
    payload = _metadata_only_parent_payload(payload)
    gc.collect()
    if float(args.semantic_weight) > 0.0 and not semantic_enabled:
        raise ValueError(
            "--semantic-teacher-root is required when semantic loss is enabled"
        )
    if generic_text_response_enabled and not semantic_enabled:
        raise ValueError(
            "generic text-response preservation requires --semantic-teacher-root"
        )
    if args.selection_policy == "semantic_capability" and not semantic_enabled:
        raise ValueError("semantic_capability selection requires semantic teachers")
    if (
        args.selection_policy == "text_response_capability"
        and not generic_text_response_enabled
    ):
        raise ValueError(
            "text_response_capability selection requires a positive response weight"
        )
    semantic_bridge = None
    semantic_summary_head = None
    generic_text_bundle = None
    semantic_teacher_root = (
        Path(args.semantic_teacher_root) if semantic_enabled else None
    )
    semantic_scene = str(args.semantic_scene).strip() or Path(config.scene_root).name
    semantic_kernel_sizes = tuple(
        int(value)
        for value in str(args.semantic_kernel_sizes).split(",")
        if value.strip()
    )
    if semantic_enabled:
        if not args.semantic_bridge_checkpoint:
            raise ValueError(
                "--semantic-bridge-checkpoint is required for semantic teachers"
            )
        semantic_bridge, _semantic_manifest = GlobalRegionSummaryBridge.from_checkpoint(
            args.semantic_bridge_checkpoint, map_location="cpu"
        )
        semantic_bridge = semantic_bridge.to(device).eval()
        semantic_summary_head = (
            SigLIP2SummaryHead.from_radio_checkpoint(args.radio_checkpoint)
            .to(device)
            .eval()
        )
        for module in (semantic_bridge, semantic_summary_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for frame in set(render_train_frames) | set(validation_frames):
            _semantic_teacher_path(semantic_teacher_root, semantic_scene, frame)
    if generic_text_response_enabled:
        if not args.generic_text_relation_authority:
            raise ValueError(
                "--generic-text-relation-authority is required when response loss is enabled"
            )
        if not args.expected_generic_text_relation_authority_sha256:
            raise ValueError(
                "--expected-generic-text-relation-authority-sha256 is required"
            )
        generic_text_bundle = load_frozen_generic_region_text_bundle(
            args.generic_text_relation_authority,
            expected_relation_authority_sha256=(
                args.expected_generic_text_relation_authority_sha256
            ),
        ).to(device)

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
        if name == "siglip2-g":
            adaptor = SigLIP2FeatureProjection.from_radio_checkpoint(
                args.radio_checkpoint
            )
            if bool(args.capability_projection_xformers):
                if not bool(args.capability_projection_amp):
                    raise ValueError(
                        "xFormers capability projection requires CUDA AMP"
                    )
                adaptor.enable_xformers_memory_efficient_attention()
        else:
            adaptor = load_radio_adaptor_from_checkpoint(
                args.radio_checkpoint, name, kind="feature_projection"
            )
        adaptor = adaptor.to(device).eval()
        for parameter in adaptor.parameters():
            parameter.requires_grad_(False)
        capability_adaptors[name] = adaptor
    capability_teacher_datasets: dict[str, SimpleRadioDataset] | None = None
    capability_teacher_frame_to_index: dict[str, dict[int, int]] | None = None
    capability_teacher_provenance: dict[str, dict[str, object]] = {}
    if capability_adaptors and args.capability_map_source == "official_extracted":
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
        field,
        model,
        renderer,
        dataset,
        frame_to_index,
        validation_frames,
        device,
        adaptors=capability_adaptors,
        reliability_splat=reliability_splat,
        alpha_threshold=float(args.alpha_threshold),
        capability_teacher_datasets=capability_teacher_datasets,
        capability_teacher_frame_to_index=capability_teacher_frame_to_index,
        projection_amp=bool(args.capability_projection_amp),
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
        initial_semantic_absolute, initial_semantic_centered = (
            _mean_semantic_view_metrics(
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
        )
        initial_semantic_validation = 0.5 * (
            initial_semantic_absolute + initial_semantic_centered
        )
        best_semantic_validation = initial_semantic_validation
    initial_generic_text_response_validation = None
    best_generic_text_response_validation = None
    if generic_text_response_enabled:
        initial_generic_text_response_validation = _mean_generic_text_response_metrics(
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
            text_bundle=generic_text_bundle,
        )
        best_generic_text_response_validation = dict(
            initial_generic_text_response_validation
        )
    best_step = 0
    # D512/L512 fields can have more than one million primitive-local rows.
    # Keeping the selection snapshot on the accelerator duplicates several
    # GiB and does not affect optimization; retain the exact tensors on CPU.
    best_state = _cpu_state_dict(field)
    history: list[dict] = []
    shuffled: list[int] = []

    for step in range(int(args.steps)):
        if not shuffled:
            order = torch.randperm(
                len(render_train_frames), generator=generator
            ).tolist()
            shuffled = [render_train_frames[index] for index in order]
        frame = shuffled.pop()
        sample = dataset[frame_to_index[frame]]
        _zero_optimizer_gradients(
            optimizer,
            # Dense full-grid semantic supervision needs the memory occupied
            # by the previous 1.4M-row local-code gradient during the next
            # forward pass. Adam moments remain CPU-offloaded; recreating the
            # gradient preserves the exact update while lowering forward peak.
            preserve_buffers=False,
        )
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
        teacher_capability_maps = None
        capability_alignment_loss = render_loss.detach() * 0.0
        capability_local_affinity_loss = render_loss.detach() * 0.0
        capability_stats: dict[str, dict[str, torch.Tensor]] = {}
        if capability_adaptors:
            if capability_teacher_datasets is not None:
                teacher_capability_maps = {}
                for name, teacher_dataset in capability_teacher_datasets.items():
                    teacher_index = capability_teacher_frame_to_index[name].get(
                        int(frame)
                    )
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
                projection_amp=bool(args.capability_projection_amp),
            )
        predicted_semantics = None
        semantic_teacher = None
        semantic_absolute_loss = render_loss.detach() * 0.0
        semantic_centered_loss = render_loss.detach() * 0.0
        semantic_loss = render_loss.detach() * 0.0
        generic_response_loss = render_loss.detach() * 0.0
        generic_text_response_stats: dict[str, torch.Tensor | int] = {}
        if semantic_enabled and (
            float(args.semantic_weight) > 0.0 or generic_text_response_enabled
        ):
            predicted_semantics = project_dense_region_semantics(
                semantic_bridge,
                semantic_summary_head,
                gauge_separated_radio(result["feature_map"][None], feature_dim=1),
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
            if generic_text_response_enabled:
                (
                    generic_response_loss,
                    generic_text_response_stats,
                ) = generic_region_text_response_loss(
                    predicted_semantics,
                    semantic_teacher[None],
                    result["alpha_map"],
                    generic_text_bundle,
                    alpha_threshold=float(args.alpha_threshold),
                )
        capability_scale = sum(capability_weights.values())
        base_loss = (
            render_loss
            + float(args.mpr_weight) * mpr_loss
            + capability_scale * capability_alignment_loss
            + capability_scale
            * float(args.capability_local_affinity_weight)
            * capability_local_affinity_loss
        )
        feature_branch_loss = (
            float(args.semantic_weight) * semantic_loss
            + float(args.generic_text_response_weight) * generic_response_loss
        )
        loss = (base_loss + feature_branch_loss).detach()
        if semantic_enabled and (
            float(args.semantic_weight) > 0.0 or generic_text_response_enabled
        ):
            feature_gradient = _staged_feature_branch_gradient(
                feature_branch_loss,
                result["feature_map"],
            )
            # Preserve only scalar diagnostics. The full descriptor and
            # teacher would otherwise survive until after the local-code
            # gradient allocation.
            semantic_absolute_loss = semantic_absolute_loss.detach()
            semantic_centered_loss = semantic_centered_loss.detach()
            semantic_loss = semantic_loss.detach()
            generic_response_loss = generic_response_loss.detach()
            generic_text_response_stats = {
                name: value.detach() if torch.is_tensor(value) else value
                for name, value in generic_text_response_stats.items()
            }
            predicted_semantics = None
            semantic_teacher = None
            del feature_branch_loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _backward_base_with_feature_gradient(
                base_loss,
                result["feature_map"],
                feature_gradient,
            )
            del feature_gradient
        else:
            base_loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), float(args.grad_clip))
        if optimizer_state_offload:
            _offloaded_adamw_step(
                optimizer,
                chunk_elements=int(args.optimizer_state_chunk_elements),
            )
        else:
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
                field,
                model,
                renderer,
                dataset,
                frame_to_index,
                validation_frames,
                device,
                adaptors=capability_adaptors,
                reliability_splat=reliability_splat,
                alpha_threshold=float(args.alpha_threshold),
                capability_teacher_datasets=capability_teacher_datasets,
                capability_teacher_frame_to_index=capability_teacher_frame_to_index,
                projection_amp=bool(args.capability_projection_amp),
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
            generic_text_response_validation = None
            if generic_text_response_enabled:
                generic_text_response_validation = _mean_generic_text_response_metrics(
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
                    text_bundle=generic_text_bundle,
                )
            if args.selection_policy == "final":
                selected = True
            elif args.selection_policy == "validation":
                selected = validation_cosine > best_validation
            elif args.selection_policy == "raw_fidelity":
                selected = (
                    validation_cosine > best_validation
                    and mpr_probe_cosine >= initial_mpr_probe - float(args.max_mpr_drop)
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
                    and mpr_probe_cosine >= initial_mpr_probe - float(args.max_mpr_drop)
                )
            elif args.selection_policy == "semantic_capability":
                selected = (
                    semantic_validation is not None
                    and semantic_validation > best_semantic_validation
                    and validation_cosine
                    >= initial_validation - float(args.max_validation_drop)
                    and mpr_probe_cosine >= initial_mpr_probe - float(args.max_mpr_drop)
                )
            else:
                selected = (
                    generic_text_response_validation is not None
                    and generic_text_response_validation["loss"]
                    < best_generic_text_response_validation["loss"]
                    and semantic_validation is not None
                    and semantic_validation
                    >= initial_semantic_validation - float(args.max_capability_drop)
                    and validation_cosine
                    >= initial_validation - float(args.max_validation_drop)
                    and mpr_probe_cosine >= initial_mpr_probe - float(args.max_mpr_drop)
                )
            if selected:
                best_validation = validation_cosine
                best_mpr_probe = mpr_probe_cosine
                if semantic_validation is not None:
                    best_semantic_validation = semantic_validation
                if generic_text_response_validation is not None:
                    best_generic_text_response_validation = dict(
                        generic_text_response_validation
                    )
                best_capability_validation = dict(capability_validation)
                best_step = step + 1
                _copy_state_dict_to_cpu_(best_state, field)
            record = {
                "step": step + 1,
                "frame_id": frame,
                "loss": float(loss.detach()),
                "render_loss": float(render_loss.detach()),
                "mpr_loss": float(mpr_loss.detach()),
                "capability_alignment_loss": float(capability_alignment_loss.detach()),
                "capability_local_affinity_loss": float(
                    capability_local_affinity_loss.detach()
                ),
                "capability_adaptors": {
                    name: {
                        "alignment": float(values["alignment"].detach()),
                        "local_affinity": float(values["local_affinity"].detach()),
                    }
                    for name, values in capability_stats.items()
                },
                "semantic_loss": float(semantic_loss.detach()),
                "semantic_absolute_loss": float(semantic_absolute_loss.detach()),
                "semantic_centered_loss": float(semantic_centered_loss.detach()),
                "generic_text_response_loss": float(generic_response_loss.detach()),
                "generic_text_response_components": {
                    name: int(value) if isinstance(value, int) else float(value)
                    for name, value in generic_text_response_stats.items()
                },
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
                "generic_text_response_validation": (generic_text_response_validation),
                "best_generic_text_response_validation": (
                    best_generic_text_response_validation
                ),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

        # Do not retain the previous step's dense render/capability/semantic
        # outputs while constructing the next full-resolution graph.  This is
        # especially important after validation, which temporarily exercises
        # all declared held-out views and several frozen projection spaces.
        del (
            result,
            teacher,
            chosen,
            predicted_rows,
            render_loss,
            mpr_loss,
            teacher_capability_maps,
            capability_alignment_loss,
            capability_local_affinity_loss,
            capability_stats,
            predicted_semantics,
            semantic_teacher,
            semantic_absolute_loss,
            semantic_centered_loss,
            semantic_loss,
            generic_response_loss,
            generic_text_response_stats,
            base_loss,
            loss,
        )
        _release_validation_cuda_cache(
            device,
            enabled=bool(args.release_validation_cuda_cache),
            should_validate=should_validate,
        )

    field.load_state_dict(best_state, strict=True)
    del best_state
    gc.collect()
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
    payload["method_v1_construction_lineage"] = method_v1_predecessor_lineage
    payload["render_optimization"] = {
        "method_v1_stage": current_method_v1_stage,
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
        "initial_generic_text_response_validation": (
            initial_generic_text_response_validation
        ),
        "best_validation_cosine": best_validation,
        "best_mpr_probe_cosine": best_mpr_probe,
        "best_capability_validation_cosine": best_capability_validation,
        "best_semantic_validation_cosine": best_semantic_validation,
        "best_generic_text_response_validation": (
            best_generic_text_response_validation
        ),
        "best_step": best_step,
        "selection_policy": args.selection_policy,
        "max_validation_drop": float(args.max_validation_drop),
        "max_mpr_drop": float(args.max_mpr_drop),
        "max_capability_drop": float(args.max_capability_drop),
        "semantic_capability": {
            "enabled": semantic_enabled,
            "teacher_root": (
                str(semantic_teacher_root.resolve()) if semantic_enabled else ""
            ),
            "scene": semantic_scene,
            "bridge_checkpoint": (
                str(Path(args.semantic_bridge_checkpoint).resolve())
                if semantic_enabled
                else ""
            ),
            "radio_checkpoint": (
                str(Path(args.radio_checkpoint).resolve()) if semantic_enabled else ""
            ),
            "kernel_sizes": list(semantic_kernel_sizes),
            "weight": float(args.semantic_weight),
            "centered_weight": float(args.semantic_centered_weight),
            "selection_score": "mean(absolute_cosine, centered_cosine)",
            "uses_benchmark_masks": False,
            "uses_text_queries": False,
        },
        "generic_text_response": {
            "enabled": generic_text_response_enabled,
            "weight": float(args.generic_text_response_weight),
            "relation_authority": (
                str(Path(args.generic_text_relation_authority).resolve())
                if generic_text_response_enabled
                else ""
            ),
            "relation_authority_sha256": (
                generic_text_bundle.relation_authority_sha256
                if generic_text_response_enabled
                else ""
            ),
            "relation_content_authority_sha256": (
                generic_text_bundle.relation_content_authority_sha256
                if generic_text_response_enabled
                else ""
            ),
            "primary_file_sha256": (
                generic_text_bundle.primary_file_sha256
                if generic_text_response_enabled
                else ""
            ),
            "primary_embedding_sha256": (
                generic_text_bundle.primary_embedding_sha256
                if generic_text_response_enabled
                else ""
            ),
            "synonym_file_sha256": (
                generic_text_bundle.synonym_file_sha256
                if generic_text_response_enabled
                else ""
            ),
            "synonym_embedding_sha256": (
                generic_text_bundle.synonym_embedding_sha256
                if generic_text_response_enabled
                else ""
            ),
            "region_operator": "fixed_adaptive_average_8x8_visible_cells",
            "components": ["profile", "listwise", "sibling", "synonym"],
            "generic_target_blind_text_bank_opened": (generic_text_response_enabled),
            "benchmark_text_queries_opened": False,
            "uses_benchmark_masks": False,
            "uses_target_metrics_for_selection": False,
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
            "local_affinity_weight": float(args.capability_local_affinity_weight),
            "local_radius": int(args.capability_local_radius),
            "local_balance_quantile": float(args.capability_local_balance_quantile),
            "projection_order": (
                "complete_rendered_2d_grid_vs_resample(official_runtime_adaptor_output)"
                if args.capability_map_source == "official_extracted"
                else "complete_rendered_2d_grid_then_official_adaptor"
            ),
            "visibility_domain": f"rendered_alpha>={float(args.alpha_threshold)}",
            "custom_adaptor_head": False,
            "projection_precision": (
                "cuda_amp_matching_official_extractor"
                if bool(args.capability_projection_amp)
                else "float32"
            ),
            "attention_runtime": (
                "xformers_memory_efficient_exact_global"
                if bool(args.capability_projection_xformers)
                else "torch_default_sdpa"
            ),
            "allocator_boundary": (
                "release_unused_cuda_cache_after_validation"
                if bool(args.release_validation_cuda_cache)
                else "default_cuda_cache"
            ),
            "optimizer_state_residency": (
                "cpu_moments_bounded_chunk_gpu_adamw_update"
                if optimizer_state_offload
                else "optimizer_default"
            ),
            "optimizer_state_chunk_elements": (
                int(args.optimizer_state_chunk_elements)
                if optimizer_state_offload
                else None
            ),
            "gradient_buffer_residency": (
                "zero_in_place_and_reuse_between_steps"
                if optimizer_state_offload
                else "set_to_none_each_step"
            ),
            "optimizer_state_offload_requested": bool(
                args.offload_optimizer_state
            ),
            "optimizer_state_offload_auto_semantic": bool(
                semantic_enabled and not bool(args.offload_optimizer_state)
            ),
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
        "initial_generic_text_response_validation": (
            initial_generic_text_response_validation
        ),
        "best_validation_cosine": best_validation,
        "best_mpr_probe_cosine": best_mpr_probe,
        "best_capability_validation_cosine": best_capability_validation,
        "best_semantic_validation_cosine": best_semantic_validation,
        "best_generic_text_response_validation": (
            best_generic_text_response_validation
        ),
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
        "--construction-prior-field",
        action="append",
        default=[],
        help=(
            "Recovery-only predecessor field between the base and immediate parent; "
            "repeat in Method-v1 stage order when upgrading a pre-lineage checkpoint."
        ),
    )
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
        "--generic-text-response-weight",
        type=float,
        default=0.0,
        help=(
            "Single weight for fixed target-blind response-profile, listwise, "
            "sibling, and synonym preservation on genuine source region summaries."
        ),
    )
    parser.add_argument("--generic-text-relation-authority", default="")
    parser.add_argument("--expected-generic-text-relation-authority-sha256", default="")
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
    parser.add_argument(
        "--capability-projection-amp",
        action="store_true",
        help=(
            "Project rendered grids through frozen capability adaptors under "
            "the same CUDA AMP runtime used by the official feature extractor."
        ),
    )
    parser.add_argument(
        "--capability-projection-xformers",
        action="store_true",
        help=(
            "Use xFormers memory-efficient exact global attention for the "
            "frozen SigLIP2 spatial adaptor."
        ),
    )
    parser.add_argument(
        "--release-validation-cuda-cache",
        action="store_true",
        help=(
            "Release unused CUDA allocator blocks after heavy validation "
            "boundaries; this does not change tensors or selection metrics."
        ),
    )
    parser.add_argument(
        "--offload-optimizer-state",
        action="store_true",
        help=(
            "Keep AdamW moments on CPU during forward/backward and return "
            "them to the field device only for optimizer.step()."
        ),
    )
    parser.add_argument(
        "--optimizer-state-chunk-elements",
        type=int,
        default=16777216,
        help=(
            "Maximum number of AdamW moment elements staged on the field "
            "device at once when optimizer-state offload is active."
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
        "--render-view-policy",
        choices=["all_nonbenchmark", "mpr"],
        default="all_nonbenchmark",
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
            "text_response_capability",
            "capability_pareto",
        ],
        default="validation",
    )
    parser.add_argument("--max-validation-drop", type=float, default=0.0)
    parser.add_argument("--max-capability-drop", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        min(
            args.siglip_spatial_render_weight,
            args.dino_render_weight,
            args.sam3_render_weight,
            args.capability_local_affinity_weight,
            args.semantic_weight,
            args.generic_text_response_weight,
        )
        < 0
    ):
        parser.error("render, semantic, and response weights cannot be negative")
    if args.capability_local_radius <= 0:
        parser.error("--capability-local-radius must be positive")
    print(json.dumps(finetune(args), indent=2))


if __name__ == "__main__":
    main()
