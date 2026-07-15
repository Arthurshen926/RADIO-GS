#!/usr/bin/env python3
"""Train a small zero-mean view residual over a frozen canonical RADIO field."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.evaluation.view_consistency import (
    consistency_from_sums,
    merge_training_partials,
)
from radio_gs.field import ZeroMeanViewResidual, load_canonical_field_checkpoint
from radio_gs.field.observation_lifting_contract import (
    validate_observation_contract_metadata,
)
from radio_gs.losses.radio_adaptor_loss import (
    compute_radio_adaptor_masked_render_losses,
)
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.rendering.coefficient_renderer import render_view_conditioned_radio
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.canonical_field_losses import normalized_render_reconstruction_loss
from radio_gs.training.feature_training_utils import SimpleRadioDataset


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parse_frame_ids(raw: str) -> list[int]:
    value = str(raw or "").strip()
    if not value:
        return []
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return sorted({int(token) for token in tokens})


def _dataset(config, renderer) -> SimpleRadioDataset:
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
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
    )


def _load_training_disagreement(
    raw_paths: str,
    *,
    geometry_hash: str,
    mpr_path: Path,
    expected_frames: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    paths = [Path(value) for value in raw_paths.split(",") if value.strip()]
    if not paths:
        raise ValueError("--training-consistency-partials is empty")
    partials = [torch.load(path, map_location="cpu") for path in paths]
    for partial in partials:
        if partial.get("kind") != "mpr_training_view_consistency_partial":
            raise ValueError("view residual accepts training-only consistency partials")
        if str(partial.get("geometry_xyz_sha256", "")) != geometry_hash:
            raise ValueError("consistency partial geometry mismatch")
        if Path(str(partial.get("mpr_cache", ""))).resolve() != mpr_path.resolve():
            raise ValueError("consistency partial MPR mismatch")
    frames = [
        int(frame)
        for partial in partials
        for frame in partial.get("selected_frame_ids", [])
    ]
    if len(frames) != len(set(frames)) or sorted(frames) != sorted(expected_frames):
        raise ValueError("consistency partials do not form the exact MPR training split")
    sums = merge_training_partials(partials)
    consistency = consistency_from_sums(sums)
    return (
        consistency["view_disagreement"],
        torch.as_tensor(sums["observation_count"]).long(),
        [str(path.resolve()) for path in paths],
    )


@torch.no_grad()
def _mean_view_fidelity(
    residual,
    field,
    model,
    renderer,
    dataset,
    frame_to_index: dict[int, int],
    frames: list[int],
    device: torch.device,
    *,
    alpha_threshold: float,
    adaptors: dict[str, torch.nn.Module],
) -> tuple[dict[str, float], list[dict]]:
    weighted_sum = {"raw_radio": 0.0, **{name: 0.0 for name in adaptors}}
    total_pixels = 0
    per_view: list[dict] = []
    residual.eval()
    for frame in frames:
        sample = dataset[frame_to_index[frame]]
        result = render_view_conditioned_radio(
            renderer,
            model,
            field,
            residual,
            sample["pose_w2c"].to(device),
            feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2],
        )
        predicted = result["feature_map"].permute(1, 2, 0).float()
        teacher = sample["radio_features"].to(device).permute(1, 2, 0).float()
        valid = result["alpha_map"] >= float(alpha_threshold)
        maps = {"raw_radio": (predicted, teacher)}
        for name, adaptor in adaptors.items():
            projected = project_feature_map_with_adaptor(
                predicted.permute(2, 0, 1)[None], adaptor
            )[0].permute(1, 2, 0)
            target = project_feature_map_with_adaptor(
                teacher.permute(2, 0, 1)[None], adaptor
            )[0].permute(1, 2, 0)
            maps[name] = (projected, target)
        values = {}
        pixels = int(valid.sum())
        for name, (prediction_map, target_map) in maps.items():
            cosine = F.cosine_similarity(
                prediction_map[valid], target_map[valid], dim=-1, eps=1e-8
            )
            value = float(cosine.mean()) if pixels else 0.0
            values[name] = value
            weighted_sum[name] += value * pixels
        total_pixels += pixels
        per_view.append({"frame_id": frame, "pixels": pixels, "mean_cosine": values})
    if total_pixels <= 0:
        raise RuntimeError("view residual rendered no validation pixels")
    return {name: value / total_pixels for name, value in weighted_sum.items()}, per_view


def _selection_score(metrics: dict[str, float]) -> float:
    return sum(metrics.values()) / max(1, len(metrics))


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _hybrid = load_render_pipeline(
        args.config,
        args.geometry_checkpoint,
        device,
        strict_checkpoint_contract=True,
        load_ply_rgb_features=False,
    )
    geometry_hash = _sha256_tensor_rows(model.get_xyz())
    field_path = Path(args.field_checkpoint)
    field, field_payload = load_canonical_field_checkpoint(field_path, map_location="cpu")
    fingerprint = dict(field_payload.get("geometry_fingerprint", {}))
    if str(fingerprint.get("xyz_sha256", "")) != geometry_hash:
        raise ValueError("canonical field/geometry row fingerprint mismatch")
    field = field.to(device).eval()
    for parameter in field.parameters():
        parameter.requires_grad_(False)

    mpr_path = Path(args.mpr_cache)
    mpr = torch.load(mpr_path, map_location="cpu")
    mpr_metadata = dict(mpr.get("metadata", {}))
    contract = validate_observation_contract_metadata(
        mpr_metadata,
        require_declaration=not bool(args.allow_compatible_legacy_mpr),
    )
    if str(mpr_metadata.get("xyz_sha256", "")) != geometry_hash:
        raise ValueError("MPR/geometry row fingerprint mismatch")
    if bool(mpr_metadata.get("benchmark_masks_opened", False)) or bool(
        mpr_metadata.get("text_queries_opened", False)
    ):
        raise ValueError("MPR is not query/label free")
    training_frames = [int(value) for value in mpr_metadata.get("selected_frame_indices", [])]
    excluded_frames = {int(value) for value in mpr_metadata.get("excluded_frame_ids", [])}
    validation_frames = _parse_frame_ids(args.validation_frame_ids)
    benchmark_frames = _parse_frame_ids(args.benchmark_frame_ids)
    if not validation_frames or not set(validation_frames).issubset(excluded_frames):
        raise ValueError("validation frames must be a non-empty subset excluded from MPR")
    if not benchmark_frames or not set(benchmark_frames).issubset(excluded_frames):
        raise ValueError("benchmark frames must be a non-empty subset excluded from MPR")
    if set(validation_frames).intersection(benchmark_frames):
        raise ValueError("validation and benchmark frames must be disjoint")

    disagreement, replay_counts, partial_paths = _load_training_disagreement(
        args.training_consistency_partials,
        geometry_hash=geometry_hash,
        mpr_path=mpr_path,
        expected_frames=training_frames,
    )
    direction_path = Path(args.direction_stats)
    direction_stats = torch.load(direction_path, map_location="cpu")
    if direction_stats.get("kind") != "mpr_training_view_direction_stats":
        raise ValueError("direction stats are not training-only MPR statistics")
    if str(direction_stats.get("geometry_xyz_sha256", "")) != geometry_hash:
        raise ValueError("direction-stat geometry mismatch")
    if Path(str(direction_stats.get("mpr_cache", ""))).resolve() != mpr_path.resolve():
        raise ValueError("direction-stat MPR mismatch")
    if sorted(map(int, direction_stats.get("training_frame_ids", []))) != sorted(training_frames):
        raise ValueError("direction stats use a different training split")
    if not bool(direction_stats.get("protocol", {}).get("training_views_only", False)):
        raise ValueError("direction stats do not certify training-only construction")

    mpr_valid = torch.as_tensor(mpr["valid"]).bool()
    eligible = mpr_valid & (replay_counts >= int(args.minimum_context_views))
    if not bool(eligible.any()):
        raise RuntimeError("no rows satisfy the view-context gate")
    gate_scale = float(torch.quantile(disagreement[eligible], float(args.gate_quantile)))
    row_gate = torch.zeros_like(disagreement)
    row_gate[eligible] = (
        disagreement[eligible] / max(gate_scale, 1e-6)
    ).clamp(0.0, 1.0).pow(float(args.gate_power))
    residual = ZeroMeanViewResidual(
        num_gaussians=int(model.get_xyz().shape[0]),
        coefficient_dim=field.decoder.coefficient_dim,
        rank=int(args.rank),
        mean_view_direction=direction_stats["mean_view_direction"],
        row_gate=row_gate,
        residual_scale=float(args.residual_scale),
    ).to(device)
    if residual.rank > int(args.maximum_rank):
        raise ValueError("view residual rank exceeds the declared low-capacity budget")

    adaptors: dict[str, torch.nn.Module] = {}
    if float(args.capability_alignment_weight) > 0 or float(
        args.capability_local_weight
    ) > 0:
        for name in ("dino_v3", "sam3"):
            module = load_radio_adaptor_from_checkpoint(
                args.radio_checkpoint, name, kind="feature_projection"
            ).to(device).eval()
            module.requires_grad_(False)
            adaptors[name] = module

    dataset = _dataset(config, renderer)
    frame_to_index = {int(frame): index for index, frame in enumerate(dataset.frame_indices)}
    required = set(training_frames) | set(validation_frames) | set(benchmark_frames)
    missing = sorted(required - set(frame_to_index))
    if missing:
        raise ValueError(f"required raw feature frames unavailable: {missing}")
    # Deliberately do not index benchmark samples before the checkpoint is frozen.
    base_coefficients = field.coefficients().detach()
    optimizer = torch.optim.AdamW(
        [
            {"params": [residual.local_codes], "lr": float(args.learning_rate_local)},
            {
                "params": [residual.direction_projection, residual.output_basis],
                "lr": float(args.learning_rate_global),
            },
        ],
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    initial_validation, initial_per_view = _mean_view_fidelity(
        residual,
        field,
        model,
        renderer,
        dataset,
        frame_to_index,
        validation_frames,
        device,
        alpha_threshold=float(args.alpha_threshold),
        adaptors=adaptors,
    )
    best_validation = dict(initial_validation)
    best_score = _selection_score(initial_validation)
    best_step = 0
    best_state = copy.deepcopy(residual.state_dict())
    history: list[dict] = []
    shuffled: list[int] = []

    for step in range(int(args.steps)):
        if not shuffled:
            order = torch.randperm(len(training_frames), generator=generator).tolist()
            shuffled = [training_frames[index] for index in order]
        frame = shuffled.pop()
        sample = dataset[frame_to_index[frame]]
        pose = sample["pose_w2c"].to(device)
        optimizer.zero_grad(set_to_none=True)
        residual.train()
        delta = residual(model.get_xyz(), pose)
        rendered = renderer.render_feature_rows(
            model,
            pose,
            base_coefficients + delta,
            feature_height=sample["radio_features"].shape[1],
            feature_width=sample["radio_features"].shape[2],
            alpha_normalize=True,
        )
        predicted = field.decoder.decode_map(rendered["feature_map"])[None]
        teacher = sample["radio_features"].to(device)[None]
        render_loss = normalized_render_reconstruction_loss(
            predicted,
            teacher,
            rendered["alpha_map"][None],
            alpha_threshold=float(args.alpha_threshold),
            cosine_weight=1.0,
            huber_weight=float(args.huber_weight),
        )
        if adaptors:
            capability_alignment, capability_local, _capability_details = (
                compute_radio_adaptor_masked_render_losses(
                    predicted,
                    teacher,
                    adaptors,
                    rendered["alpha_map"][None] >= float(args.alpha_threshold),
                    adaptor_weights={name: 1.0 for name in adaptors},
                    local_radius=int(args.capability_local_radius),
                )
            )
        else:
            capability_alignment = predicted.sum() * 0.0
            capability_local = predicted.sum() * 0.0
        regularization = residual.regularization()
        delta_l2 = delta.square().mean()
        loss = (
            render_loss
            + float(args.local_regularization) * regularization["local_l2"]
            + float(args.basis_regularization) * regularization["basis_orthogonality"]
            + float(args.delta_regularization) * delta_l2
            + float(args.capability_alignment_weight) * capability_alignment
            + float(args.capability_local_weight) * capability_local
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), float(args.grad_clip))
        optimizer.step()

        should_validate = (step + 1) % int(args.eval_every) == 0 or step == 0
        if should_validate:
            validation, validation_per_view = _mean_view_fidelity(
                residual,
                field,
                model,
                renderer,
                dataset,
                frame_to_index,
                validation_frames,
                device,
                alpha_threshold=float(args.alpha_threshold),
                adaptors=adaptors,
            )
            score = _selection_score(validation)
            noninferior = all(
                validation[name]
                >= initial_validation[name] - float(args.maximum_validation_drop)
                for name in initial_validation
            )
            selected = noninferior and score > best_score + float(
                args.minimum_selection_gain
            )
            if selected:
                best_validation = dict(validation)
                best_score = score
                best_step = step + 1
                best_state = copy.deepcopy(residual.state_dict())
            record = {
                "step": step + 1,
                "frame_id": frame,
                "loss": float(loss.detach()),
                "render_loss": float(render_loss.detach()),
                "delta_l2": float(delta_l2.detach()),
                "capability_alignment_loss": float(capability_alignment.detach()),
                "capability_local_loss": float(capability_local.detach()),
                "validation_cosine": validation,
                "best_validation_cosine": best_validation,
                "validation_score": score,
                "validation_noninferior": noninferior,
                "selected": selected,
                "validation_per_view": validation_per_view,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    residual.load_state_dict(best_state, strict=True)
    final_validation, final_per_view = _mean_view_fidelity(
        residual,
        field,
        model,
        renderer,
        dataset,
        frame_to_index,
        validation_frames,
        device,
        alpha_threshold=float(args.alpha_threshold),
        adaptors=adaptors,
    )
    residual = residual.eval().cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "zero_mean_view_residual",
        "architecture": {
            "num_gaussians": residual.num_gaussians,
            "coefficient_dim": residual.coefficient_dim,
            "rank": residual.rank,
            "residual_scale": residual.residual_scale,
            "parameter_ratio_vs_dense_coefficients": (
                residual.rank / residual.coefficient_dim
            ),
            "formula": "(local * ((view_direction - training_mean_direction) @ A)) @ B",
        },
        "state_dict": residual.state_dict(),
        "mean_view_direction": residual.mean_view_direction,
        "row_gate": residual.row_gate,
        "geometry_fingerprint": fingerprint,
        "base_field_checkpoint": str(field_path.resolve()),
        "base_field_sha256": _sha256_file(field_path),
        "mpr_cache": str(mpr_path.resolve()),
        "training_provenance": {
            "training_frames": training_frames,
            "validation_frames": validation_frames,
            "excluded_benchmark_frames": benchmark_frames,
            "consistency_partials": partial_paths,
            "direction_stats": str(direction_path.resolve()),
            "direction_stats_sha256": _sha256_file(direction_path),
            "minimum_context_views": int(args.minimum_context_views),
            "gate_quantile": float(args.gate_quantile),
            "gate_scale": gate_scale,
            "gate_power": float(args.gate_power),
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "observation_lifting_contract": contract,
            "compatible_legacy_contract_certification": bool(
                args.allow_compatible_legacy_mpr
                and "observation_lifting_contract" not in mpr_metadata
            ),
        },
        "selection": {
            "policy": "frozen_nonbenchmark_multicapability_noninferiority",
            "initial_validation_cosine": initial_validation,
            "initial_validation_per_view": initial_per_view,
            "best_validation_cosine": final_validation,
            "best_validation_per_view": final_per_view,
            "best_step": best_step,
            "minimum_selection_gain": float(args.minimum_selection_gain),
            "maximum_validation_drop": float(args.maximum_validation_drop),
            "selection_score": "mean(raw_radio,dino_v3,sam3)_dense_cosine",
            "history": history,
        },
        "invariants": {
            "canonical_field_frozen": True,
            "primitive_query_reads_canonical_only": True,
            "residual_used_only_for_view_rendering": True,
            "weighted_training_view_residual_mean": "zero_by_linear_centering",
        },
    }
    torch.save(payload, output)
    report = {
        "output": str(output.resolve()),
        "initial_validation_cosine": initial_validation,
        "best_validation_cosine": final_validation,
        "gain": {
            name: final_validation[name] - initial_validation[name]
            for name in initial_validation
        },
        "best_step": best_step,
        "benchmark_images_opened": False,
        "architecture": payload["architecture"],
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
    parser.add_argument("--mpr-cache", required=True)
    parser.add_argument("--training-consistency-partials", required=True)
    parser.add_argument("--direction-stats", required=True)
    parser.add_argument("--validation-frame-ids", required=True)
    parser.add_argument("--benchmark-frame-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--maximum-rank", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--minimum-context-views", type=int, default=2)
    parser.add_argument("--gate-quantile", type=float, default=0.90)
    parser.add_argument("--gate-power", type=float, default=0.5)
    parser.add_argument("--learning-rate-local", type=float, default=0.02)
    parser.add_argument("--learning-rate-global", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--huber-weight", type=float, default=0.25)
    parser.add_argument("--local-regularization", type=float, default=1e-5)
    parser.add_argument("--basis-regularization", type=float, default=1e-4)
    parser.add_argument("--delta-regularization", type=float, default=1e-4)
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--capability-alignment-weight", type=float, default=0.5)
    parser.add_argument("--capability-local-weight", type=float, default=0.1)
    parser.add_argument("--capability-local-radius", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--minimum-selection-gain", type=float, default=1e-5)
    parser.add_argument("--maximum-validation-drop", type=float, default=0.002)
    parser.add_argument(
        "--allow-compatible-legacy-mpr",
        action="store_true",
        help="Certify an old cache by checking every canonical-v1 policy field.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
