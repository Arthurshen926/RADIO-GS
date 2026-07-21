#!/usr/bin/env python3
"""Compile an unregistered real-image crop into canonical 3-D support."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces import (
    OfficialCropSummaryRuntime,
    OfficialRadioRuntime,
    load_canonical_capability_bank,
    load_canonical_primitive_reliability,
    load_canonical_support_graph,
)
from radio_gs.querying.evidence_scorer import EvidenceScoringConfig
from radio_gs.querying.query_compilers import compile_image_query
from radio_gs.querying.query_engine import CanonicalQueryEngine
from radio_gs.querying.support_solver import SupportSolverConfig
from radio_gs.interfaces.frozen_radio_views import sha256_file


def parse_bbox(raw: str, width: int, height: int) -> tuple[int, int, int, int]:
    if not str(raw).strip():
        return 0, 0, int(width), int(height)
    values = [int(value) for value in str(raw).replace(",", " ").split()]
    if len(values) != 4:
        raise ValueError("bbox must be x0,y0,x1,y1")
    x0, y0, x1, y1 = values
    x0, x1 = max(0, x0), min(int(width), x1)
    y0, y1 = max(0, y0), min(int(height), y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox is empty after image clipping")
    return x0, y0, x1, y1


def semantic_feature_rows(
    payload: dict,
    *,
    bank_xyz: torch.Tensor,
    bank_valid: torch.Tensor,
    bank_global_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Validate a dense or sparse semantic cache and return graph-local rows.

    Surface-region caches intentionally store descriptors only for valid
    canonical primitives.  The support graph is defined on that same compact
    row set, so expanding to all Gaussian rows is both unnecessary and was an
    incorrect index operation in the original pose-free path.
    """

    semantic_features = torch.as_tensor(payload["features"])
    semantic_valid = torch.as_tensor(payload["valid"]).bool().cpu()
    semantic_xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("semantic cache metadata must be a mapping")
    bank_xyz = torch.as_tensor(bank_xyz).float().cpu()
    bank_valid = torch.as_tensor(bank_valid).bool().cpu()
    bank_global_rows = torch.as_tensor(bank_global_rows).long().cpu()
    if (
        semantic_features.ndim != 2
        or semantic_valid.shape != bank_valid.shape
        or not torch.equal(semantic_valid, bank_valid)
        or semantic_xyz.shape != bank_xyz.shape
        or not torch.allclose(semantic_xyz, bank_xyz, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("semantic cache does not align with canonical capability rows")
    sparse_rows = payload.get("global_rows")
    if sparse_rows is None:
        if semantic_features.shape[0] != bank_xyz.shape[0]:
            raise ValueError("dense semantic cache does not align with canonical geometry")
        local_features = semantic_features[bank_global_rows]
    else:
        sparse_rows = torch.as_tensor(sparse_rows).long().cpu()
        if (
            sparse_rows.ndim != 1
            or semantic_features.shape[0] != sparse_rows.numel()
            or not torch.equal(sparse_rows, bank_global_rows)
        ):
            raise ValueError("sparse semantic cache global_rows do not align with support graph")
        local_features = semantic_features
    if not bool(torch.isfinite(local_features).all()):
        raise ValueError("semantic cache contains NaN or infinity")
    return local_features, metadata


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    bank = load_canonical_capability_bank(
        args.capability_cache,
        expected_field_checkpoint_sha256=args.field_checkpoint_sha256,
    )
    appearance_field_signature = bank.signatures["appearance"]
    if args.radio_version != appearance_field_signature.radio_version:
        raise ValueError("requested RADIO version differs from capability cache")
    if sha256_file(args.radio_checkpoint) != bank.metadata.get(
        "radio_checkpoint_sha256"
    ):
        raise ValueError("requested RADIO checkpoint differs from capability cache")
    graph = load_canonical_support_graph(args.support_graph, bank).to(device)
    primitive_reliability = None
    if str(args.reliability_cache).strip():
        primitive_reliability = load_canonical_primitive_reliability(
            args.reliability_cache,
            expected_xyz=bank.xyz,
            expected_valid=bank.valid,
            expected_field_checkpoint_sha256=str(
                bank.metadata.get("field_checkpoint_sha256", "")
            ),
        )
    node_reliability = (
        primitive_reliability.valid_confidence()
        if primitive_reliability is not None
        else None
    )
    semantic = torch.load(args.semantic_cache, map_location="cpu")
    semantic_features, semantic_metadata = semantic_feature_rows(
        semantic,
        bank_xyz=bank.xyz,
        bank_valid=bank.valid,
        bank_global_rows=bank.global_rows,
    )
    if (
        semantic_metadata.get("field_checkpoint_sha256") != bank.metadata.get(
            "field_checkpoint_sha256"
        )
    ):
        raise ValueError("semantic/capability canonical-field hashes differ")
    if semantic_metadata.get("radio_checkpoint_sha256") != bank.metadata.get(
        "radio_checkpoint_sha256"
    ):
        raise ValueError("semantic/capability RADIO checkpoint hashes differ")
    if (
        semantic_metadata.get("bridge_training_scope") != "global_cross_scene"
        or semantic_metadata.get("official_summary_head") is not True
        or semantic_metadata.get("custom_text_projection") is not False
    ):
        raise ValueError(
            "semantic cache is not the frozen global Level-3 official-summary variant"
        )

    image = Image.open(args.image).convert("RGB")
    box = parse_bbox(args.bbox, *image.size)
    crop = image.crop(box)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    crop_tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
    semantic_runtime = OfficialCropSummaryRuntime.load(
        checkpoint_path=args.radio_checkpoint,
        radio_repo=args.radio_repo,
        version=args.radio_version,
        device=device,
    )
    with torch.inference_mode():
        semantic_summary = semantic_runtime.encode(crop_tensor)
    del semantic_runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    appearance_runtime = OfficialRadioRuntime.load(
        radio_repo=args.radio_repo,
        version=args.radio_version,
        adaptor_names=("dino_v3_7b",),
        device=device,
    )
    with torch.inference_mode():
        _, appearance_map = appearance_runtime.encode_adaptor_images(
            crop_tensor, "dino_v3_7b", feature_fmt="NCHW"
        )
    del appearance_runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    semantic_summary = F.normalize(semantic_summary.float(), dim=-1)
    appearance_map = appearance_map.float()
    appearance_tokens = appearance_map.permute(0, 2, 3, 1).reshape(
        -1, appearance_map.shape[1]
    )

    semantic_query_signature = FeatureSpaceSignature(
        radio_version=args.radio_version,
        radio_checkpoint_sha256=bank.metadata["radio_checkpoint_sha256"],
        raw_feature_dim=1280,
        adaptor_name="siglip2-g.summary",
        adaptor_sha256=bank.metadata["radio_checkpoint_sha256"],
        adaptor_output_dim=int(semantic_features.shape[1]),
        token_type="summary",
        normalization="l2",
        crop_policy="official_real_image_crop",
        field_checkpoint_sha256="",
        semantic_alignment="none",
        semantic_alignment_sha256="",
    )
    semantic_field_signature = FeatureSpaceSignature(
        radio_version=args.radio_version,
        radio_checkpoint_sha256=bank.metadata["radio_checkpoint_sha256"],
        raw_feature_dim=1280,
        adaptor_name="siglip2-g.summary",
        adaptor_sha256=bank.metadata["radio_checkpoint_sha256"],
        adaptor_output_dim=int(semantic_features.shape[1]),
        token_type="primitive",
        normalization="l2",
        crop_policy="density_adaptive_3d_region_summary",
        field_checkpoint_sha256=bank.metadata["field_checkpoint_sha256"],
        semantic_alignment="global_region_summary_bridge",
        semantic_alignment_sha256=str(
            semantic_metadata.get("bridge_checkpoint_sha256", "")
        ),
    )
    appearance_query_signature = FeatureSpaceSignature(
        **{
            **appearance_field_signature.to_dict(),
            "token_type": "spatial",
            "crop_policy": "official_real_image_crop",
            "field_checkpoint_sha256": "",
            "semantic_alignment": "none",
            "semantic_alignment_sha256": "",
        }
    )
    query = compile_image_query(
        semantic_summary,
        appearance_tokens,
        semantic_signature=semantic_query_signature,
        appearance_signature=appearance_query_signature,
        semantic_negatives=F.normalize(
            semantic_features.float().mean(dim=0, keepdim=True),
            dim=-1,
        ).to(device),
        appearance_negatives=F.normalize(
            bank.appearance[bank.global_rows].float().mean(dim=0, keepdim=True),
            dim=-1,
        ).to(device),
        prototype_count=args.prototype_count,
    )
    feature_banks = {
        "semantic": semantic_features.to(device),
        "appearance": bank.appearance[bank.global_rows].to(device),
    }
    engine = CanonicalQueryEngine(
        graph,
        scoring_config=EvidenceScoringConfig(
            semantic_weight=args.semantic_weight,
            appearance_weight=args.appearance_weight,
            boundary_weight=0.0,
            prototype_temperature=args.prototype_temperature,
        ),
        solver_config=SupportSolverConfig(
            iterations=args.iterations,
            residual=args.residual,
            unary_temperature=args.unary_temperature,
            support_threshold=args.support_threshold,
        ),
        node_reliability=(
            node_reliability.to(device) if node_reliability is not None else None
        ),
    )
    result = engine.execute(
        query,
        feature_banks,
        feature_signatures={
            "semantic": semantic_field_signature,
            "appearance": appearance_field_signature,
        },
    )
    probabilities = torch.zeros(bank.num_gaussians, dtype=torch.float16)
    probabilities[bank.global_rows] = result.selected_probabilities.half().cpu()
    unary = torch.full(
        (bank.num_gaussians,), -float("inf"), dtype=torch.float16
    )
    unary[bank.global_rows] = result.unary.half().cpu()
    metadata = {
        "schema_version": 1,
        "source": "posefree_official_image_crop_to_canonical_3d_support",
        "image": str(Path(args.image).resolve()),
        "bbox_xyxy": list(box),
        "query_pose_used": False,
        "official_siglip2_summary": True,
        "official_dino_v3_7b_spatial": True,
        "custom_image_head": False,
        "semantic_alignment_level": 3,
        "query_signatures": {
            "semantic": semantic_query_signature.to_dict(),
            "appearance": appearance_query_signature.to_dict(),
        },
        "field_signatures": {
            "semantic": semantic_field_signature.to_dict(),
            "appearance": appearance_field_signature.to_dict(),
        },
        "negative_baseline": "query_independent_canonical_scene_mean",
        "primitive_reliability": (
            {
                "cache": str(Path(args.reliability_cache).resolve()),
                "formula": primitive_reliability.metadata.get("formula"),
                "scene_mean_precision_weighting": False,
                "centered_unary_shrink": True,
                "uses_query_or_target_labels": False,
            }
            if primitive_reliability is not None
            else None
        ),
        "target_masks_opened": False,
        "test_calibration": False,
        "capability_cache": str(Path(args.capability_cache).resolve()),
        "semantic_cache": str(Path(args.semantic_cache).resolve()),
        "support_graph": str(Path(args.support_graph).resolve()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "xyz": bank.xyz,
            "valid": bank.valid,
            "features": probabilities[:, None],
            "unary": unary[:, None],
            "metadata": metadata,
        },
        output_path,
    )
    report = {
        **metadata,
        "output": str(output_path),
        "valid_gaussians": int(bank.valid.sum()),
        "selected_gaussians": int((probabilities >= args.support_threshold).sum()),
        "track_a_score": "fused_primitive_unary_before_graph_or_threshold",
        "track_b_score": "selected_support_probability_after_frozen_solver",
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", default="")
    parser.add_argument("--capability-cache", required=True)
    parser.add_argument("--semantic-cache", required=True)
    parser.add_argument("--support-graph", required=True)
    parser.add_argument("--reliability-cache", default="")
    parser.add_argument("--field-checkpoint-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prototype-count", type=int, default=4)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--residual", type=float, default=0.30)
    parser.add_argument("--unary-temperature", type=float, default=0.10)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
