#!/usr/bin/env python3
"""Diagnose CTF-GS direct primitive readout against a VPR summary cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.config import load_config
from radio_gs.scripts.eval_lerf_direct_3d_selection import (
    _blend_point_summary_adapter_features,
    _build_point_summary_adapter,
    load_summary_head,
)
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _build_hybrid_model
from radio_gs.models.point_summary_adapter import append_point_summary_context
from radio_gs.utils.checkpoint_io import load_trusted_checkpoint


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.as_tensor(x).float(), dim=-1)


def compute_cosine_stats(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred_n = _normalize(pred)
    target_n = _normalize(target).to(pred_n.device)
    if pred_n.shape != target_n.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_n.shape)} vs {tuple(target_n.shape)}")
    cos = (pred_n * target_n).sum(dim=-1).detach().cpu()
    return {
        "mean_cos_to_vpr": float(cos.mean().item()) if cos.numel() else 0.0,
        "p10": float(torch.quantile(cos, 0.10).item()) if cos.numel() else 0.0,
        "p50": float(torch.quantile(cos, 0.50).item()) if cos.numel() else 0.0,
        "p90": float(torch.quantile(cos, 0.90).item()) if cos.numel() else 0.0,
        "min": float(cos.min().item()) if cos.numel() else 0.0,
        "max": float(cos.max().item()) if cos.numel() else 0.0,
        "count": int(cos.numel()),
    }


def compute_rank_agreement(student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> dict[str, Any]:
    student = torch.as_tensor(student_scores).float()
    teacher = torch.as_tensor(teacher_scores).float().to(student.device)
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("student_scores and teacher_scores must both be [N,C] with matching shape")
    if student.shape[1] < 2:
        return {"text_rank_agreement": 1.0, "text_rank_pairs": 0}
    teacher_diff = teacher.unsqueeze(2) - teacher.unsqueeze(1)
    student_diff = student.unsqueeze(2) - student.unsqueeze(1)
    mask = teacher_diff > 0
    pairs = int(mask.sum().item())
    if pairs == 0:
        return {"text_rank_agreement": 1.0, "text_rank_pairs": 0}
    agree = ((student_diff > 0) == mask)[mask].float().mean()
    return {"text_rank_agreement": float(agree.item()), "text_rank_pairs": pairs}


def build_diagnostic_adapter_input(
    compact: torch.Tensor,
    *,
    context_features: str = "",
    opacity: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    view_counts: torch.Tensor | None = None,
    view_count_max: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the same compact+context vector used by direct 3D eval/training."""
    return append_point_summary_context(
        compact,
        context_features=context_features,
        opacity=opacity,
        scales=scales,
        view_counts=view_counts,
        view_count_max=view_count_max,
    )


def build_adapter_metadata_status(
    checkpoint: dict[str, Any],
    *,
    use_point_summary_adapter: bool,
    adapter_loaded: bool,
) -> dict[str, Any]:
    has_adapter = "point_summary_adapter_state_dict" in checkpoint
    warnings: list[str] = []
    if has_adapter and not use_point_summary_adapter:
        warnings.append("checkpoint_has_point_summary_adapter_but_use_point_summary_adapter_is_false")
    if use_point_summary_adapter and not adapter_loaded:
        warnings.append("use_point_summary_adapter_true_but_adapter_not_loaded")
    return {
        "checkpoint_has_point_summary_adapter": bool(has_adapter),
        "adapter_loaded": bool(adapter_loaded),
        "eval_adapter_disabled_with_checkpoint_adapter": bool(has_adapter and not use_point_summary_adapter),
        "metadata_warnings": warnings,
        "point_summary_adapter_metadata": checkpoint.get("point_summary_adapter_metadata") or {},
    }


def _indices_sha256(indices: torch.Tensor) -> str:
    arr = torch.as_tensor(indices, dtype=torch.long).detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load_teacher_cache(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"teacher cache must be dict: {path}")
    if "summary_features" not in payload and "features" not in payload:
        raise KeyError("teacher cache must contain summary_features or features")
    if "valid" not in payload:
        raise KeyError("teacher cache must contain valid")
    return payload


def _sample_indices(valid: torch.Tensor, *, max_points: int, seed: int) -> torch.Tensor:
    source = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    if max_points > 0 and source.numel() > max_points:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(source.numel(), generator=generator)[:max_points]
        source = source[order]
    return source.long()


def _query_compact(model: torch.nn.Module, indices: torch.Tensor, *, feature_key: str) -> torch.Tensor:
    result = model.query_gaussian_points(indices, return_aux=feature_key != "features")
    if isinstance(result, dict):
        if feature_key not in result:
            raise KeyError(f"feature_key={feature_key!r} not in {sorted(result.keys())}")
        return result[feature_key]
    if feature_key != "features":
        raise KeyError(f"feature_key={feature_key!r} requires return_aux dict")
    return result


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    config = load_config(args.config)
    model, codec = _build_hybrid_model(config, args.checkpoint, device)
    checkpoint = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    teacher = _load_teacher_cache(args.teacher_cache)
    valid = torch.as_tensor(teacher["valid"]).bool()
    indices_cpu = _sample_indices(valid, max_points=args.max_points, seed=args.seed)
    indices = indices_cpu.to(device=device, dtype=torch.long)
    target = teacher.get("summary_features")
    if target is None:
        raise KeyError("diagnose_direct_head_consistency currently expects summary_features cache")
    target_summary = _normalize(torch.as_tensor(target)[indices_cpu]).to(device)
    compact = _query_compact(model, indices, feature_key=args.compact_feature_key)

    summary_head = load_summary_head(args.summary_head_weights, device)
    with torch.no_grad():
        decoded = codec.decode_points(compact.float())
        head_param = next(summary_head.parameters(), None)
        tokens = decoded.unsqueeze(0)
        if head_param is not None:
            tokens = tokens.to(dtype=head_param.dtype)
        decoded_summary = _normalize(summary_head(tokens).squeeze(0).float())

    adapter_loaded = False
    adapter_summary = None
    final_summary = decoded_summary
    if args.use_point_summary_adapter:
        adapter = _build_point_summary_adapter(config, args.checkpoint, device)
        adapter_loaded = True
        metadata = checkpoint.get("point_summary_adapter_metadata") or {}
        context_features = str(
            metadata.get(
                "point_summary_adapter_context_features",
                getattr(config, "point_summary_adapter_context_features", ""),
            )
            or ""
        )
        view_count_max = metadata.get("point_summary_adapter_view_count_max")
        opacity = None
        scales = None
        view_counts = None
        if "opacity" in context_features:
            opacity = model.get_opacity()[indices].to(device=device)
        if "scale_log" in context_features:
            scales = model.get_scaling()[indices].to(device=device)
        if "view_count" in context_features:
            raw_view_counts = teacher.get("view_counts")
            if not isinstance(raw_view_counts, torch.Tensor):
                raise KeyError(
                    "point_summary_adapter_context_features includes view_count, "
                    "but the teacher cache has no view_counts tensor"
                )
            view_counts = raw_view_counts[indices_cpu].to(device=device)
        adapter_input = build_diagnostic_adapter_input(
            compact.float(),
            context_features=context_features,
            opacity=opacity,
            scales=scales,
            view_counts=view_counts,
            view_count_max=view_count_max,
        )
        with torch.no_grad():
            adapter_summary = _normalize(adapter(adapter_input).float())
        final_summary = _blend_point_summary_adapter_features(
            decoded_summary,
            adapter_summary,
            alpha=args.point_summary_adapter_blend_alpha,
            valid_mask=None,
        )

    report: dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "teacher_cache": args.teacher_cache,
        "sample_count": int(indices_cpu.numel()),
        "indices_sha256": _indices_sha256(indices_cpu),
        "compact_feature_key": args.compact_feature_key,
        "use_point_summary_adapter": bool(args.use_point_summary_adapter),
        "point_summary_adapter_blend_alpha": float(args.point_summary_adapter_blend_alpha),
        "point_summary_adapter_context_features": str(
            (checkpoint.get("point_summary_adapter_metadata") or {}).get(
                "point_summary_adapter_context_features",
                getattr(config, "point_summary_adapter_context_features", ""),
            )
            or ""
        ),
        "decoded_summary": compute_cosine_stats(decoded_summary, target_summary),
        "final_summary": compute_cosine_stats(final_summary, target_summary),
    }
    if adapter_summary is not None:
        report["adapter_summary"] = compute_cosine_stats(adapter_summary, target_summary)
    report["adapter_status"] = build_adapter_metadata_status(
        checkpoint,
        use_point_summary_adapter=bool(args.use_point_summary_adapter),
        adapter_loaded=adapter_loaded,
    )
    if args.text_embeddings:
        text_payload = torch.load(args.text_embeddings, map_location="cpu")
        text = text_payload.get("embeddings") if isinstance(text_payload, dict) else text_payload
        text = _normalize(torch.as_tensor(text)).to(device)
        report["rank_agreement"] = compute_rank_agreement(final_summary @ text.T, target_summary @ text.T)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--summary_head_weights", default="checkpoints/siglip2_summary_head.pth")
    parser.add_argument("--compact_feature_key", default="features", choices=("features", "fused", "semantic", "geometry"))
    parser.add_argument("--use_point_summary_adapter", action="store_true")
    parser.add_argument("--point_summary_adapter_blend_alpha", type=float, default=1.0)
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text_embeddings", default="")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_diagnostic(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
