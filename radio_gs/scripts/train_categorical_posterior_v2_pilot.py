#!/usr/bin/env python3
"""Development-only cross-scene pilot for CategoricalPosteriorV2.

This experiment deliberately opens VALA pseudo labels for two declared
development scenes.  It answers one narrow question: can one globally shared
categorical calibration head improve Primitive Readout-v0 on held-out scenes?
The resulting checkpoint is diagnostic evidence and is never paper-eligible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from radio_gs.querying.typed_posteriors import CategoricalPosteriorV2
from radio_gs.querying.unified_query import cosine_bank_torch
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    volume_weighted_split_metrics,
)
from radio_gs.universal_field_v1 import validate_universal_field_payload
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
    write_torch_noclobber,
)


PAPER_SCENES = (
    "scene0000_00",
    "scene0062_00",
    "scene0070_00",
    "scene0097_00",
    "scene0140_00",
    "scene0347_00",
    "scene0400_00",
    "scene0590_00",
)
PAPER_CLASS_IDS = tuple(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
DEFAULT_TRAIN_SCENES = ("scene0000_00", "scene0070_00")


def encode_background_targets(
    raw_labels: np.ndarray,
    class_ids: Sequence[int] = PAPER_CLASS_IDS,
) -> np.ndarray:
    """Map raw NYU40 ids to categorical columns and all others to background."""

    labels = np.asarray(raw_labels, dtype=np.int32).reshape(-1)
    background = len(class_ids)
    encoded = np.full(labels.shape, background, dtype=np.int64)
    for column, raw_id in enumerate(class_ids):
        encoded[labels == int(raw_id)] = int(column)
    return encoded


def posterior_prediction_to_raw(
    prediction: torch.Tensor,
    class_ids: Sequence[int] = PAPER_CLASS_IDS,
) -> np.ndarray:
    """Map global posterior columns to raw NYU40 ids; abstention is id zero."""

    selected = torch.as_tensor(prediction).detach().cpu().long().numpy()
    raw = np.zeros(selected.shape, dtype=np.int32)
    class_array = np.asarray(class_ids, dtype=np.int32)
    valid = (selected >= 0) & (selected < len(class_ids))
    raw[valid] = class_array[selected[valid]]
    return raw


def scene_macro(
    scenes: Mapping[str, Mapping[str, Mapping[str, float]]],
    selected: Sequence[str],
) -> dict[str, dict[str, float]]:
    return {
        split: {
            metric: float(
                np.mean([float(scenes[scene][split][metric]) for scene in selected])
            )
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }


def _parse_scenes(value: str) -> tuple[str, ...]:
    scenes = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(scenes) == 0 or len(set(scenes)) != len(scenes):
        raise ValueError("training scenes must be non-empty and unique")
    if not set(scenes).issubset(PAPER_SCENES):
        raise ValueError("training scenes must be in the frozen paper-eight cohort")
    if len(scenes) >= len(PAPER_SCENES):
        raise ValueError("the pilot requires at least one held-out scene")
    return scenes


def _cosine_logits(
    features: torch.Tensor,
    text: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    output = torch.empty(features.shape[0], text.shape[0], dtype=torch.float32)
    # Match the frozen evaluator's operation order exactly: the cache loader
    # normalizes once on CPU and cosine_bank_torch normalizes again on device.
    # The frozen 8192-row GEMM chunk is also material for deterministic tie
    # breaking at one scene0347 row (the top-two margin is 1.49e-8).
    normalized_features = F.normalize(features.float(), dim=-1, eps=1e-8)
    text_device = F.normalize(text.float(), dim=-1, eps=1e-8).to(device)
    for start in range(0, features.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), features.shape[0])
        output[start:end] = cosine_bank_torch(
            normalized_features[start:end].to(device), text_device
        ).cpu()
    return output


def _split_prediction(
    model: CategoricalPosteriorV2,
    logits: torch.Tensor,
    reliability: torch.Tensor,
    valid: torch.Tensor,
    split: str,
    *,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    global_column = {class_id: index for index, class_id in enumerate(PAPER_CLASS_IDS)}
    active = torch.tensor(
        [
            global_column[class_id]
            for class_id in OPENGAUSSIAN_NYU40_CLASS_SPLITS[split]
        ],
        dtype=torch.long,
        device=device,
    )
    raw = np.zeros(logits.shape[0], dtype=np.int32)
    for start in range(0, logits.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), logits.shape[0])
        with torch.inference_mode():
            output = model(
                logits[start:end].to(device),
                reliability=reliability[start:end].to(device),
                valid=valid[start:end].to(device),
                active_class_indices=active,
            )
        raw[start:end] = posterior_prediction_to_raw(output.prediction)
    return raw


def _load_scene(
    scene: str,
    *,
    inventory: Mapping[str, Any],
    universal_root: Path,
    text: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    require_exact_replay: bool = True,
) -> dict[str, Any]:
    record = inventory["scannet_ovs_paper8"]["scene_results"][scene]
    query, query_sha, query_path = load_torch_mapping(
        record["primitive_query_cache"],
        expected_sha256=record["primitive_query_cache_sha256"],
        map_location="cpu",
        label=f"{scene} Primitive Readout-v0 cache",
    )
    universal_path = universal_root / scene / "universal_field_v1.pth"
    universal, universal_sha, universal_path = load_torch_mapping(
        universal_path,
        map_location="cpu",
        label=f"{scene} Universal Field v1",
    )
    validate_universal_field_payload(universal)
    migration = universal["universal_field_migration"]
    source = query.get("metadata", {}).get("field_checkpoint", {})
    if source.get("sha256") != migration.get("source_field_sha256"):
        raise ValueError(f"{scene} query cache is not bound to migration source")
    features = torch.as_tensor(query["summary_features"])
    valid = torch.as_tensor(query["valid"]).bool().reshape(-1)
    reliability = torch.as_tensor(universal["reliability"]).float()
    if features.shape[0] != valid.numel() or reliability.shape[0] != valid.numel():
        raise ValueError(f"{scene} row domains differ")
    logits = _cosine_logits(
        features,
        text,
        device=device,
        chunk_size=chunk_size,
    )

    evaluation, _, _ = load_json_object(
        record["evaluation_report"],
        expected_sha256=record["evaluation_report_sha256"],
        label=f"{scene} frozen evaluation report",
    )
    prediction_path = Path(evaluation["scenes"][scene]["prediction_npz"])
    with np.load(prediction_path, allow_pickle=False) as frozen:
        pseudo_labels = np.asarray(frozen["pseudo_labels"], dtype=np.int32)
        significance = np.asarray(frozen["significance"], dtype=np.float32)
        frozen_predictions = {
            split: np.asarray(frozen[f"pred_split_{split}"], dtype=np.int32)
            for split in ("19", "15", "10")
        }
    if (
        pseudo_labels.shape != (valid.numel(),)
        or significance.shape != pseudo_labels.shape
    ):
        raise ValueError(f"{scene} frozen evaluation rows differ")
    if not np.isfinite(significance).all() or np.any(significance < 0):
        raise ValueError(f"{scene} significance is invalid")

    class_array = np.asarray(PAPER_CLASS_IDS, dtype=np.int32)
    baseline = class_array[logits.argmax(dim=-1).numpy()]
    replay_mismatch_count = int(np.count_nonzero(baseline != frozen_predictions["19"]))
    if require_exact_replay and replay_mismatch_count:
        raise ValueError(f"{scene} Primitive Readout-v0 replay differs")
    return {
        "logits": logits,
        "reliability": reliability,
        "valid": valid,
        "pseudo_labels": pseudo_labels,
        "significance": significance,
        "frozen_predictions": frozen_predictions,
        "primitive_replay": {
            "exact": replay_mismatch_count == 0,
            "mismatch_count": replay_mismatch_count,
            "row_count": int(valid.numel()),
        },
        "provenance": {
            "primitive_query_cache": {"path": str(query_path), "sha256": query_sha},
            "universal_field": {"path": str(universal_path), "sha256": universal_sha},
            "frozen_evaluation_report": {
                "path": str(Path(record["evaluation_report"]).resolve()),
                "sha256": record["evaluation_report_sha256"],
            },
            "frozen_prediction_npz": file_record(prediction_path),
        },
    }


def _balanced_weights(target: torch.Tensor, significance: torch.Tensor) -> torch.Tensor:
    mass = torch.zeros(int(target.max().item()) + 1, dtype=torch.float64)
    mass.scatter_add_(0, target.cpu(), significance.cpu().double())
    row_mass = mass[target.cpu()]
    if bool((row_mass <= 0).any()):
        raise ValueError("an observed pilot training target has zero significance")
    # Classes absent from the two declared development scenes remain at their
    # zero-initialized Primitive Readout-v0 parameters; they are not an error.
    weights = significance.double() / row_mass
    return (weights / weights.mean()).float()


def _metric_rows(
    scenes: Mapping[str, Mapping[str, Any]],
    model: CategoricalPosteriorV2,
    *,
    device: torch.device,
    chunk_size: int,
    zero_reliability: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_rows: dict[str, Any] = {}
    calibrated_rows: dict[str, Any] = {}
    for scene, data in scenes.items():
        baseline_rows[scene] = {
            split: volume_weighted_split_metrics(
                data["pseudo_labels"],
                data["frozen_predictions"][split],
                data["significance"],
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            )
            for split in ("19", "15", "10")
        }
        calibrated_rows[scene] = {}
        reliability = (
            torch.zeros_like(data["reliability"])
            if zero_reliability
            else data["reliability"]
        )
        for split in ("19", "15", "10"):
            prediction = _split_prediction(
                model,
                data["logits"],
                reliability,
                data["valid"],
                split,
                device=device,
                chunk_size=chunk_size,
            )
            calibrated_rows[scene][split] = volume_weighted_split_metrics(
                data["pseudo_labels"],
                prediction,
                data["significance"],
                OPENGAUSSIAN_NYU40_CLASS_SPLITS[split],
            )
    return baseline_rows, calibrated_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_scenes = _parse_scenes(args.train_scenes)
    heldout_scenes = tuple(scene for scene in PAPER_SCENES if scene not in train_scenes)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)

    inventory, inventory_sha, inventory_path = load_json_object(
        args.inventory,
        label="Method-v1 asset inventory",
    )
    text_payload, text_sha, text_path = load_torch_mapping(
        args.text_embeddings,
        map_location="cpu",
        label="ScanNet split19 text embeddings",
    )
    expected_queries = [NYU40_ID_TO_NAME[class_id] for class_id in PAPER_CLASS_IDS]
    if text_payload.get("queries") != expected_queries:
        raise ValueError("text query order differs from the frozen split19 order")
    text = torch.as_tensor(text_payload["embeddings"]).float()
    if text.shape[0] != len(PAPER_CLASS_IDS):
        raise ValueError("text embedding count differs")

    scenes = {
        scene: _load_scene(
            scene,
            inventory=inventory,
            universal_root=Path(args.universal_root).expanduser().resolve(),
            text=text,
            device=device,
            chunk_size=args.chunk_size,
        )
        for scene in PAPER_SCENES
    }
    train_logits = torch.cat([scenes[scene]["logits"] for scene in train_scenes])
    train_reliability = torch.cat(
        [scenes[scene]["reliability"] for scene in train_scenes]
    )
    train_valid = torch.cat([scenes[scene]["valid"] for scene in train_scenes])
    train_target = torch.from_numpy(
        np.concatenate(
            [
                encode_background_targets(scenes[scene]["pseudo_labels"])
                for scene in train_scenes
            ]
        )
    ).long()
    train_significance = torch.from_numpy(
        np.concatenate([scenes[scene]["significance"] for scene in train_scenes])
    ).float()
    train_weights = _balanced_weights(train_target, train_significance)

    model = CategoricalPosteriorV2(num_classes=len(PAPER_CLASS_IDS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=0.0
    )
    logits_device = train_logits.to(device)
    reliability_device = train_reliability.to(device)
    valid_device = train_valid.to(device)
    target_device = train_target.to(device)
    weights_device = train_weights.to(device)
    loss_history: list[float] = []
    model.train()
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            logits_device,
            reliability=reliability_device,
            valid=valid_device,
        )
        cross_entropy = F.cross_entropy(
            output.logits,
            target_device,
            reduction="none",
        )
        calibration_loss = (cross_entropy * weights_device).mean()
        regularization = sum(
            parameter.square().mean() for parameter in model.parameters()
        )
        loss = calibration_loss + float(args.regularization) * regularization
        loss.backward()
        optimizer.step()
        if step in {0, int(args.steps) - 1} or (step + 1) % 50 == 0:
            loss_history.append(float(loss.detach().cpu()))
    del logits_device, reliability_device, valid_device, target_device, weights_device
    model.eval()

    baseline_rows, calibrated_rows = _metric_rows(
        scenes,
        model,
        device=device,
        chunk_size=args.chunk_size,
    )
    _, calibrated_without_reliability_rows = _metric_rows(
        scenes,
        model,
        device=device,
        chunk_size=args.chunk_size,
        zero_reliability=True,
    )
    baseline_macro = scene_macro(baseline_rows, PAPER_SCENES)
    calibrated_macro = scene_macro(calibrated_rows, PAPER_SCENES)
    baseline_train = scene_macro(baseline_rows, train_scenes)
    calibrated_train = scene_macro(calibrated_rows, train_scenes)
    baseline_heldout = scene_macro(baseline_rows, heldout_scenes)
    calibrated_heldout = scene_macro(calibrated_rows, heldout_scenes)
    calibrated_without_reliability_heldout = scene_macro(
        calibrated_without_reliability_rows, heldout_scenes
    )
    deltas = {
        split: {
            metric: calibrated_heldout[split][metric] - baseline_heldout[split][metric]
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }
    gate_pass = all(
        deltas[split]["miou"] > 0.0 and deltas[split]["macc"] >= -0.01
        for split in ("19", "15", "10")
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = output_dir / "categorical_posterior_v2_development_only.pth"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = {
        "schema_version": 1,
        "artifact_type": "radio_gs_categorical_posterior_v2_development_only",
        "model_schema": model.schema,
        "num_classes": len(PAPER_CLASS_IDS),
        "class_ids": list(PAPER_CLASS_IDS),
        "state_dict": state,
        "training": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "regularization": float(args.regularization),
            "train_scenes": list(train_scenes),
            "benchmark_labels_opened": True,
            "paper_eligible": False,
        },
        "producer_source": file_record(Path(__file__)),
    }
    write_torch_noclobber(checkpoint_path, checkpoint)
    checkpoint_record = file_record(checkpoint_path)
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_categorical_posterior_v2_cross_scene_pilot",
        "status": "complete",
        "decision": "mechanism_supported" if gate_pass else "mechanism_not_supported",
        "gate": {
            "pass": gate_pass,
            "rule": "held-out mIoU improves on 19/15/10 and held-out mAcc drop is <=0.01 on each split",
        },
        "eligibility": {
            "development_only": True,
            "benchmark_labels_opened": True,
            "heldout_labels_used_for_optimization": False,
            "paper_or_sota_claim_eligible": False,
        },
        "cohort": {
            "train_scenes": list(train_scenes),
            "heldout_scenes": list(heldout_scenes),
            "all_scenes": list(PAPER_SCENES),
        },
        "training": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "regularization": float(args.regularization),
            "loss_checkpoints": loss_history,
            "weighting": "per-class balanced opacity-times-volume significance",
        },
        "metrics": {
            "baseline_all8": baseline_macro,
            "calibrated_all8": calibrated_macro,
            "baseline_train2": baseline_train,
            "calibrated_train2": calibrated_train,
            "baseline_heldout6": baseline_heldout,
            "calibrated_heldout6": calibrated_heldout,
            "heldout_delta": deltas,
            "calibrated_without_reliability_heldout6": (
                calibrated_without_reliability_heldout
            ),
            "reliability_inference_delta_heldout6": {
                split: {
                    metric: calibrated_heldout[split][metric]
                    - calibrated_without_reliability_heldout[split][metric]
                    for metric in ("miou", "macc")
                }
                for split in ("19", "15", "10")
            },
            "baseline_by_scene": baseline_rows,
            "calibrated_by_scene": calibrated_rows,
        },
        "checkpoint": checkpoint_record,
        "provenance": {
            "pilot_source": file_record(Path(__file__)),
            "inventory": {"path": str(inventory_path), "sha256": inventory_sha},
            "text_embeddings": {"path": str(text_path), "sha256": text_sha},
            "scenes": {scene: scenes[scene]["provenance"] for scene in PAPER_SCENES},
            "universal_field_authority": file_record(
                "paper/artifacts/universal_field_v1_authority_20260816.json"
            ),
        },
    }
    report_path = output_dir / "categorical_posterior_v2_cross_scene_pilot.json"
    write_frozen_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        default="paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json",
    )
    parser.add_argument(
        "--universal-root",
        default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/universal_field_v1",
    )
    parser.add_argument(
        "--text-embeddings",
        default="checkpoints/siglip2_scannet_og_text_embeddings_ens5_split19.pt",
    )
    parser.add_argument("--train-scenes", default=",".join(DEFAULT_TRAIN_SCENES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
