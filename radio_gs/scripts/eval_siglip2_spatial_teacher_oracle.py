#!/usr/bin/env python3
"""Level-1 oracle: official SigLIP2 spatial output versus official text encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radio_gs.interfaces.frozen_radio_views import sha256_file
from radio_gs.interfaces.semantic_alignment import (
    SemanticAlignmentStage,
    SemanticOracleResult,
)
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection
from radio_gs.scripts.eval_lerf_grounding import (
    encode_text_siglip2,
    evaluate_scene,
    load_lerf_ovs_labels,
)


def _protocol_hash(protocol: dict) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _aggregate(scene_results: dict[str, dict]) -> dict[str, float | int]:
    sample_count = sum(int(value["n_iou_samples"]) for value in scene_results.values())
    localization_count = sum(int(value["loc_total"]) for value in scene_results.values())
    if sample_count <= 0 or localization_count <= 0:
        raise RuntimeError("oracle produced no evaluation samples")
    return {
        "miou": sum(
            float(value["miou"]) * int(value["n_iou_samples"])
            for value in scene_results.values()
        )
        / sample_count,
        "localization_accuracy": sum(
            int(value["loc_correct"]) for value in scene_results.values()
        )
        / localization_count,
        "sample_count": sample_count,
        "localization_count": localization_count,
        "scene_macro_miou": sum(float(value["miou"]) for value in scene_results.values())
        / len(scene_results),
        "scene_macro_localization_accuracy": sum(
            float(value["loc_acc"]) for value in scene_results.values()
        )
        / len(scene_results),
    }


def evaluate_oracle(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    available: list[str] = []
    categories: set[str] = set()
    for scene in scenes:
        feature_dir = Path(args.feature_root) / scene
        label_scene = Path(args.label_dir) / scene
        if not feature_dir.exists() or not label_scene.exists():
            if args.require_all_scenes:
                raise FileNotFoundError(
                    f"missing oracle input for {scene}: {feature_dir} or {label_scene}"
                )
            continue
        _annotations, scene_categories, _height, _width = load_lerf_ovs_labels(
            args.label_dir, scene
        )
        available.append(scene)
        categories.update(scene_categories)
    if not available:
        raise RuntimeError("no scene has both raw teacher features and LERF labels")
    ordered_categories = sorted(categories)

    with torch.inference_mode():
        # This is the same official SigLIP2-G text tower/tokenizer referenced
        # by C-RADIO's siglip2-g adaptor.  The helper also restores the
        # upstream 1536-D projection whose HF config is currently malformed.
        text_embeddings = encode_text_siglip2(ordered_categories, device).cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.teacher_space == "official_spatial":
        projection = SigLIP2FeatureProjection.from_radio_checkpoint(
            args.radio_checkpoint
        ).to(device)
        oracle_stage = SemanticAlignmentStage.OFFICIAL_SPATIAL
        visual_encoder = "official C-RADIOv4 siglip2-g spatial feature projection"
    elif args.teacher_space == "official_crop_summary":
        projection = torch.nn.Identity().to(device)
        oracle_stage = SemanticAlignmentStage.OFFICIAL_CROP_SUMMARY
        visual_encoder = "official C-RADIOv4 siglip2-g visual crop summary"
    else:
        projection = torch.nn.Identity().to(device)
        oracle_stage = SemanticAlignmentStage.GLOBAL_FROZEN_BRIDGE
        visual_encoder = (
            "global frozen region-to-RADIO-summary aligner followed by the "
            "official C-RADIOv4 siglip2-g visual summary head"
        )
    projection.eval()
    for parameter in projection.parameters():
        parameter.requires_grad_(False)

    protocols = {
        "direct_cosine": {
            "scoring": "cosine",
            "temperature": 1.0,
            "formula": f"cos({args.teacher_space}, official_siglip2_text)",
        },
        "protocol_aligned": {
            "scoring": "softmax_scene",
            "temperature": float(args.temperature),
            "formula": "scene_softmax(cosine)",
        },
        "vala_relevancy": {
            "scoring": "relevancy",
            "temperature": float(args.vala_logit_scale),
            "formula": "softmax([query_cosine, max_generic_negative_cosine])",
        },
    }
    canonical_embeddings = None
    canonical_path = Path(args.canonical_embedding_cache)
    if canonical_path.is_file():
        canonical_payload = torch.load(canonical_path, map_location="cpu")
        canonical_embeddings = torch.as_tensor(canonical_payload["embeddings"]).float()
    common = {
        "dataset": "LERF-OVS",
        "scenes": available,
        "query_text": "exact annotation category string",
        "text_prompt_templates": ["{query}"],
        "text_encoder": "official SigLIP2-G tokenizer/text tower used by C-RADIOv4 siglip2-g",
        "visual_encoder": visual_encoder,
        "iou_threshold_mode": "fixed_peak_relative",
        "iou_threshold": float(args.iou_threshold),
        "localization_mode": args.localization_mode,
        "localization_smoothing_kernel": int(args.localization_smoothing_kernel),
        "eval_at_image_resolution": bool(args.eval_at_image_resolution),
        "mask_refinement": "none",
        "test_set_calibration": False,
        "benchmark_masks_used_for_model_or_threshold_selection": False,
        "radio_version": args.radio_version,
        "radio_checkpoint_sha256": sha256_file(args.radio_checkpoint),
    }
    readout_results: dict[str, dict] = {}
    for readout_name, readout in protocols.items():
        protocol = {**common, **readout}
        per_scene: dict[str, dict] = {}
        for scene in available:
            if readout["scoring"] == "relevancy" and canonical_embeddings is None:
                raise FileNotFoundError(
                    "vala_relevancy requires --canonical-embedding-cache"
                )
            result = evaluate_scene(
                scene=scene,
                label_dir=args.label_dir,
                proj=projection,
                text_embeddings=text_embeddings,
                categories=ordered_categories,
                device=device,
                gt_feature_dir=str(Path(args.feature_root) / scene),
                iou_threshold=float(args.iou_threshold),
                threshold_mode="fixed",
                temperature=float(readout["temperature"]),
                scoring=str(readout["scoring"]),
                canonical_emb=(
                    canonical_embeddings.to(device)
                    if readout["scoring"] == "relevancy"
                    else None
                ),
                eval_at_image_resolution=bool(args.eval_at_image_resolution),
                localization_mode=args.localization_mode,
                localization_smoothing_kernel=int(args.localization_smoothing_kernel),
                mask_refinement="none",
            )
            per_scene[scene] = result["teacher"]
        aggregate = _aggregate(per_scene)
        readout_results[readout_name] = {
            "protocol": protocol,
            "protocol_hash": _protocol_hash(protocol),
            "aggregate": aggregate,
            "per_scene": per_scene,
        }

    decision = readout_results[args.decision_readout]
    aggregate = decision["aggregate"]
    oracle = SemanticOracleResult(
        stage=oracle_stage,
        dataset=f"LERF-OVS_{args.source_kind}",
        miou=float(aggregate["miou"]),
        localization_accuracy=float(aggregate["localization_accuracy"]),
        sample_count=int(aggregate["sample_count"]),
        protocol_hash=str(decision["protocol_hash"]),
        metadata={
            "decision_readout": args.decision_readout,
            "available_scenes": available,
            "source_kind": args.source_kind,
            "no_3d_field_used": args.source_kind == "teacher",
        },
    )
    report = {
        "schema_version": 1,
        "oracle": {
            "stage": oracle.stage.value,
            "dataset": oracle.dataset,
            "miou": oracle.miou,
            "localization_accuracy": oracle.localization_accuracy,
            "sample_count": oracle.sample_count,
            "protocol_hash": oracle.protocol_hash,
            "metadata": dict(oracle.metadata),
        },
        "readouts": readout_results,
        "categories": ordered_categories,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--feature-root", default="output/radio_features_lerf")
    parser.add_argument(
        "--source-kind",
        choices=["teacher", "canonical_field_render"],
        default="teacher",
        help="Provenance label only; the evaluator and frozen protocol are unchanged.",
    )
    parser.add_argument(
        "--teacher-space",
        choices=[
            "official_spatial",
            "official_crop_summary",
            "global_region_summary_bridge",
        ],
        default="official_spatial",
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
    )
    parser.add_argument("--radio-repo", default="/root/RADIO")
    parser.add_argument("--radio-version", default="c-radio_v4-h")
    parser.add_argument(
        "--scenes", default="figurines,ramen,teatime,waldo_kitchen"
    )
    parser.add_argument("--require-all-scenes", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou-threshold", type=float, default=0.60)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--vala-logit-scale", type=float, default=10.0)
    parser.add_argument(
        "--canonical-embedding-cache",
        default="checkpoints/siglip2_lerf_generic_negatives_exact_official.pt",
    )
    parser.add_argument(
        "--decision-readout",
        choices=["direct_cosine", "protocol_aligned", "vala_relevancy"],
        default="protocol_aligned",
    )
    parser.add_argument(
        "--localization-mode",
        choices=["polygon_argmax", "bbox_smoothed_peak"],
        default="polygon_argmax",
    )
    parser.add_argument("--localization-smoothing-kernel", type=int, default=30)
    parser.add_argument(
        "--eval-at-image-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        default="output/semantic_oracles/siglip2_spatial_teacher_lerf.json",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_oracle(args), indent=2))


if __name__ == "__main__":
    main()
