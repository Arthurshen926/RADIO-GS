#!/usr/bin/env python3
"""Fit source-only text scores to canonical object-membership log-odds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.models.query_native_gaussian_memory import CanonicalIdentityEvidenceCalibrator
from radio_gs.querying.object_track_extent_authority import compile_object_track_extent_authority
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import _load_mapping
from radio_gs.scripts.train_lerf_anchor_conditioned_extent import _track_split
from radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder import _load_scene
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json, write_torch_noclobber
from radio_gs.utils.immutable_artifacts import load_torch_payload


def _radius_key(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _sample_rows(pool: torch.Tensor, cap: int, generator: torch.Generator) -> torch.Tensor:
    if pool.numel() <= cap:
        return pool
    return pool[torch.randperm(pool.numel(), generator=generator)[:cap]]


def _class_balanced_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, logits: bool,
) -> torch.Tensor:
    """Proper binary loss under the calibrator's explicit equal-class prior.

    Episode construction caps positives and negatives independently, so their
    sampled ratio is not a population prior.  Averaging the two conditional
    risks estimates an equal-prior likelihood ratio and keeps zero logit as the
    fixed membership decision boundary.
    """
    positive = target >= 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("balanced canonical identity loss requires both classes")
    if logits:
        positive_loss = F.binary_cross_entropy_with_logits(prediction[positive], target[positive])
        negative_loss = F.binary_cross_entropy_with_logits(prediction[negative], target[negative])
    else:
        positive_loss = F.mse_loss(prediction[positive], target[positive])
        negative_loss = F.mse_loss(prediction[negative], target[negative])
    return 0.5 * (positive_loss + negative_loss)


def _extract_scene(spec: dict[str, Any], args: argparse.Namespace, generator: torch.Generator) -> dict[str, Any]:
    data = _load_scene(
        spec["seed_model"], spec["seed_model_sha256"], spec["episodes"],
        spec["episodes_sha256"], args.evaluation_membership_threshold,
        spec["instance_text"],
    )
    anchor, anchor_record = _load_mapping(
        spec["instance_anchor_cache"], spec["instance_anchor_cache_sha256"],
        "instance text AnchorPacket cache",
    )
    tracks = compile_object_track_extent_authority(
        data["episode_object"], data["episode_query"], data["episode_target"],
        data["soft_support"], data["soft_values"], data["negative_support"],
    )
    split_masks = _track_split(data)
    split_names = ("fit", "dev", "audit")
    output: dict[str, list[dict[str, torch.Tensor]]] = {name: [] for name in split_names}
    suffix = _radius_key(args.radius_fraction)
    observed = data["observed"].any(0)
    observed_rows = torch.where(observed)[0]
    for split_name, split_mask in zip(split_names, split_masks):
        eligible = split_mask & data["generic_text"][split_name]["eligible"]
        for sample in torch.where(eligible)[0].tolist():
            track = tracks[int(data["episode_object"][sample])]
            positive = track.positive_rows[track.positive_probability >= args.evaluation_membership_threshold]
            negative = track.explicit_negative_rows
            if not positive.numel() or not negative.numel():
                continue
            positive = _sample_rows(positive, args.positive_cap, generator)
            negative = _sample_rows(negative, args.negative_cap, generator)
            random = _sample_rows(observed_rows, args.random_rows, generator)
            peak = int(anchor[f"text_{split_name}_peak_rows_r{suffix}"][sample])
            anchor_rows = torch.as_tensor(anchor[f"text_{split_name}_anchor_rows_r{suffix}"][sample]).long()
            rows = torch.unique(torch.cat((positive, negative, random, anchor_rows, torch.tensor([peak]))), sorted=True)
            query = data["generic_text"][split_name]["embedding"][sample].float()
            raw = data["baseline"][rows].float() @ F.normalize(query, dim=0)
            radius = float(anchor[f"text_{split_name}_local_radius_r{suffix}"][sample])
            distance = torch.linalg.vector_norm(data["xyz"][rows] - data["xyz"][peak], dim=1) / max(radius, 1e-6)
            target = torch.zeros(rows.numel())
            known = torch.zeros(rows.numel(), dtype=torch.bool)
            positive_position = torch.searchsorted(rows, positive)
            negative_position = torch.searchsorted(rows, negative)
            target[positive_position] = 1.0
            known[positive_position] = True
            known[negative_position] = True
            output[split_name].append({
                "raw": raw, "distance": distance,
                "reliability": data["reliability"][rows].float(),
                "target": target, "known": known,
            })
    return {"scene": data["scene"], "episodes": output, "anchor_cache": anchor_record}


def _metrics(
    model: CanonicalIdentityEvidenceCalibrator,
    scenes: list[dict[str, Any]], split: str, device: torch.device,
    baseline_center: float, baseline_scale: float,
    *, use_distance: bool = True, use_reliability: bool = True,
) -> dict[str, Any]:
    scene_result: dict[str, Any] = {}
    with torch.inference_mode():
        for scene in scenes:
            candidate_iou, baseline_iou, candidate_brier, baseline_brier = [], [], [], []
            for episode in scene["episodes"][split]:
                known = episode["known"].to(device)
                target = episode["target"].to(device)[known]
                distance = episode["distance"].to(device)
                reliability = episode["reliability"].to(device)
                logits = model(
                    episode["raw"].to(device),
                    distance if use_distance else torch.zeros_like(distance),
                    reliability if use_reliability else torch.zeros_like(reliability),
                )[known]
                candidate = torch.sigmoid(logits)
                baseline = torch.sigmoid((episode["raw"].to(device)[known] - baseline_center) / baseline_scale)
                def iou(value: torch.Tensor) -> float:
                    prediction = value >= 0.5
                    truth = target >= 0.5
                    return float((prediction & truth).sum() / (prediction | truth).sum().clamp_min(1))
                candidate_iou.append(iou(candidate)); baseline_iou.append(iou(baseline))
                candidate_brier.append(float(_class_balanced_loss(candidate, target, logits=False)))
                baseline_brier.append(float(_class_balanced_loss(baseline, target, logits=False)))
            if not candidate_iou:
                raise ValueError(f"{scene['scene']} {split} calibrator episodes are empty")
            scene_result[scene["scene"]] = {
                "candidate_iou": sum(candidate_iou) / len(candidate_iou),
                "baseline_iou": sum(baseline_iou) / len(baseline_iou),
                "candidate_brier": sum(candidate_brier) / len(candidate_brier),
                "baseline_brier": sum(baseline_brier) / len(baseline_brier),
                "episodes": len(candidate_iou),
            }
    for value in scene_result.values():
        value["delta_iou"] = value["candidate_iou"] - value["baseline_iou"]
        value["delta_brier"] = value["candidate_brier"] - value["baseline_brier"]
    return scene_result


def _fit_baseline(scenes: list[dict[str, Any]]) -> tuple[float, float]:
    episodes = [episode for scene in scenes for episode in scene["episodes"]["dev"]]
    raw = torch.cat([episode["raw"][episode["known"]] for episode in episodes])
    centers = torch.quantile(raw, torch.linspace(0.1, 0.9, 17))
    best: tuple[float, float, float] | None = None
    for center in centers.tolist():
        for scale in (0.01, 0.02, 0.05, 0.1, 0.2):
            episode_brier = []
            for episode in episodes:
                known = episode["known"]
                target = episode["target"][known]
                probability = torch.sigmoid((episode["raw"][known] - center) / scale)
                episode_brier.append(float(_class_balanced_loss(probability, target, logits=False)))
            brier = sum(episode_brier) / len(episode_brier)
            key = (brier, center, scale)
            if best is None or key < best:
                best = key
    if best is None:
        raise RuntimeError("baseline identity calibration failed")
    return best[1], best[2]


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = json.loads(Path(args.scene_specs).read_text())
    generator = torch.Generator().manual_seed(args.seed + 1)
    episode_cache_record = None
    if args.episode_cache and Path(args.episode_cache).exists():
        cached, digest, source = load_torch_payload(
            args.episode_cache,
            expected_sha256=args.expected_episode_cache_sha256 or None,
            label="LERF canonical identity episode cache",
        )
        if not isinstance(cached, dict) or cached.get("schema") != "radio_gs.lerf_canonical_identity_episodes.v1":
            raise ValueError("canonical identity episode cache differs")
        scenes = cached["scenes"]
        episode_cache_record = {"path": str(source), "sha256": digest}
    else:
        scenes = [_extract_scene(spec, args, generator) for spec in specs]
        if args.episode_cache:
            cache_path = Path(args.episode_cache).resolve()
            write_torch_noclobber(cache_path, {
                "schema": "radio_gs.lerf_canonical_identity_episodes.v1",
                "schema_version": 1, "scenes": scenes,
                "metadata": {"source_only": True, "track_complete": True, "scene_specs": str(Path(args.scene_specs).resolve())},
            })
            episode_cache_record = file_record(cache_path)
    if any(not scene["episodes"]["fit"] for scene in scenes):
        raise ValueError("canonical identity fit split is empty")
    baseline_center, baseline_scale = _fit_baseline(scenes)
    device = torch.device(args.device)
    model = CanonicalIdentityEvidenceCalibrator(
        reliability_dim=scenes[0]["episodes"]["fit"][0]["reliability"].shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    buckets = [scene["episodes"]["fit"] for scene in scenes]
    history = []
    best_key = None
    best_state = None
    best_dev = None
    torch.manual_seed(args.seed)
    for step in range(args.steps):
        scene_index = step % len(buckets)
        episode = buckets[scene_index][(step // len(buckets)) % len(buckets[scene_index])]
        known = episode["known"].to(device)
        target = episode["target"].to(device)[known]
        distance = episode["distance"].to(device)
        reliability = episode["reliability"].to(device)
        logits = model(
            episode["raw"].to(device),
            distance if args.use_distance else torch.zeros_like(distance),
            reliability if args.use_reliability else torch.zeros_like(reliability),
        )[known]
        bce = _class_balanced_loss(logits, target, logits=True)
        brier = _class_balanced_loss(torch.sigmoid(logits), target, logits=False)
        positive, negative = logits[target >= 0.5], logits[target < 0.5]
        ranking = F.softplus(negative.max() - positive.max() + args.ranking_margin)
        loss = bce + args.brier_weight * brier + args.ranking_weight * ranking
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (step + 1) % args.log_interval == 0:
            model.eval()
            dev_metrics = _metrics(
                model, scenes, "dev", device, baseline_center, baseline_scale,
                use_distance=args.use_distance, use_reliability=args.use_reliability,
            )
            minimum_iou = min(value["delta_iou"] for value in dev_metrics.values())
            maximum_brier = max(value["delta_brier"] for value in dev_metrics.values())
            macro_iou_dev = sum(value["delta_iou"] for value in dev_metrics.values()) / len(dev_metrics)
            macro_brier_dev = sum(value["delta_brier"] for value in dev_metrics.values()) / len(dev_metrics)
            key = (
                float(minimum_iou >= 0 and maximum_brier <= 0),
                -maximum_brier, minimum_iou, macro_iou_dev, -macro_brier_dev,
            )
            history.append({
                "step": step + 1, "loss": float(loss.detach()),
                "minimum_scene_delta_iou": minimum_iou,
                "maximum_scene_delta_brier": maximum_brier,
                "macro_delta_iou": macro_iou_dev,
                "macro_delta_brier": macro_brier_dev,
            })
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_dev = dev_metrics
            model.train()
    if best_state is None:
        raise RuntimeError("canonical identity checkpoint selection did not run")
    model.load_state_dict(best_state)
    model.eval()
    dev = best_dev
    audit = _metrics(
        model, scenes, "audit", device, baseline_center, baseline_scale,
        use_distance=args.use_distance, use_reliability=args.use_reliability,
    )
    all_noninferior = all(value["delta_iou"] >= 0 and value["delta_brier"] <= 0 for value in audit.values())
    macro_iou = sum(value["delta_iou"] for value in audit.values()) / len(audit)
    macro_brier = sum(value["delta_brier"] for value in audit.values()) / len(audit)
    passed = all_noninferior and macro_iou >= args.minimum_macro_iou_gain and macro_brier < 0
    output = Path(args.output).resolve()
    write_torch_noclobber(output, {
        "schema": "radio_gs.lerf_canonical_identity_calibrator.v1",
        "schema_version": 1,
        "calibrator_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "baseline_center": baseline_center, "baseline_scale": baseline_scale,
        "metadata": {
            "modality": "text", "output_semantics": "canonical_membership_log_odds",
            "track_complete": True, "unknown_excluded_from_loss": True,
            "source_only": True, "benchmark_vocabulary_opened": False,
            "benchmark_images_opened": False, "benchmark_masks_opened": False,
            "use_distance": args.use_distance,
            "use_reliability": args.use_reliability,
            "identity_prior": "equal_class_prior",
            "proper_score": "class_balanced_brier",
            "scene_records": [{"scene": scene["scene"], "anchor_cache": scene["anchor_cache"]} for scene in scenes],
        },
    })
    report = {
        "status": "source_canonical_identity_gate_pass" if passed else "source_canonical_identity_gate_fail",
        "baseline_calibration": {"center": baseline_center, "scale": baseline_scale},
        "dev": dev, "audit": audit,
        "summary": {"all_scenes_noninferior": all_noninferior, "macro_delta_iou": macro_iou, "macro_delta_brier": macro_brier},
        "history": history, "output": file_record(output),
        "episode_cache": episode_cache_record,
    }
    write_frozen_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-specs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episode-cache", default="")
    parser.add_argument("--expected-episode-cache-sha256", default="")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--radius-fraction", type=float, default=0.04)
    parser.add_argument("--positive-cap", type=int, default=1024)
    parser.add_argument("--negative-cap", type=int, default=2048)
    parser.add_argument("--random-rows", type=int, default=4096)
    parser.add_argument("--use-distance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-reliability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-membership-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-macro-iou-gain", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
