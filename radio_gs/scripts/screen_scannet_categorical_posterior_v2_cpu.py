#!/usr/bin/env python3
"""CPU-only development screen of minimal ScanNet categorical readouts.

This script opens paper-eight pseudo labels to compare a small, declared set
of shrinkage/background ablations of an existing development checkpoint.  Its
output is target-selected development evidence and is never blind or paper
eligible.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.querying.typed_posteriors import CategoricalPosteriorV2
from radio_gs.scannet_constants import NYU40_ID_TO_NAME
from radio_gs.scripts.train_categorical_posterior_v2_pilot import (
    PAPER_CLASS_IDS,
    PAPER_SCENES,
    _load_scene,
    _metric_rows,
    scene_macro,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    write_frozen_json,
)


TRAIN_SCENES = ("scene0000_00", "scene0070_00")
HELDOUT_SCENES = tuple(scene for scene in PAPER_SCENES if scene not in TRAIN_SCENES)
SHRINKAGE = (0.25, 0.5, 0.75, 1.0)


def variant_state(
    trained: dict[str, torch.Tensor], *, alpha: float, background: bool
) -> dict[str, torch.Tensor]:
    """Shrink a trained head toward the exact Primitive Readout-v0 identity."""

    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    state = {key: torch.as_tensor(item).clone() for key, item in trained.items()}
    state["class_log_temperature"].mul_(value)
    state["class_bias"].mul_(value)
    state["background_reliability.weight"].mul_(value)
    state["background_ambiguity.weight"].mul_(value)
    if background:
        state["background_bias"] = torch.tensor(-8.0) + value * (
            state["background_bias"] - torch.tensor(-8.0)
        )
    else:
        state["background_bias"] = torch.tensor(-80.0)
        state["background_reliability.weight"].zero_()
        state["background_ambiguity.weight"].zero_()
    return state


def _macro_for_model(
    scenes: dict[str, dict[str, Any]],
    model: CategoricalPosteriorV2,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    baseline, calibrated = _metric_rows(
        scenes, model, device=device, chunk_size=chunk_size
    )
    return {
        "baseline_all8": scene_macro(baseline, PAPER_SCENES),
        "candidate_all8": scene_macro(calibrated, PAPER_SCENES),
        "baseline_heldout6": scene_macro(baseline, HELDOUT_SCENES),
        "candidate_heldout6": scene_macro(calibrated, HELDOUT_SCENES),
        "candidate_by_scene": calibrated,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA exact replay requested but CUDA is unavailable")
    shrinkage = tuple(float(value) for value in args.shrinkage.split(","))
    if not shrinkage or any(value not in SHRINKAGE for value in shrinkage):
        raise ValueError(f"shrinkage must be a subset of {SHRINKAGE}")
    inventory, inventory_sha, inventory_path = load_json_object(
        args.inventory, label="Method-v1 asset inventory"
    )
    text_payload, text_sha, text_path = load_torch_mapping(
        args.text_embeddings, map_location="cpu", label="ScanNet text embeddings"
    )
    expected = [NYU40_ID_TO_NAME[class_id] for class_id in PAPER_CLASS_IDS]
    if text_payload.get("queries") != expected:
        raise ValueError("text query order differs")
    text = torch.as_tensor(text_payload["embeddings"]).float()
    scenes = {
        scene: _load_scene(
            scene,
            inventory=inventory,
            universal_root=Path(args.universal_root).expanduser().resolve(),
            text=text,
            device=device,
            chunk_size=int(args.chunk_size),
            require_exact_replay=device.type == "cuda",
        )
        for scene in PAPER_SCENES
    }
    checkpoint, checkpoint_sha, checkpoint_path = load_torch_mapping(
        args.checkpoint, map_location="cpu", label="development categorical head"
    )
    if checkpoint.get("model_schema") != CategoricalPosteriorV2.schema:
        raise ValueError("checkpoint model schema differs")
    trained = checkpoint["state_dict"]
    variants: dict[str, Any] = {}
    for background in (False, True):
        for alpha in shrinkage:
            name = f"shrink_{alpha:g}_{'with_background' if background else 'class_only'}"
            model = CategoricalPosteriorV2(num_classes=len(PAPER_CLASS_IDS)).to(device)
            model.load_state_dict(
                variant_state(trained, alpha=alpha, background=background), strict=True
            )
            model.eval()
            variants[name] = _macro_for_model(
                scenes, model, int(args.chunk_size), device
            )
    def objective(item: tuple[str, Any]) -> tuple[float, float, str]:
        name, metrics = item
        heldout = metrics["candidate_heldout6"]
        return (
            min(heldout[split]["miou"] for split in ("19", "15", "10")),
            sum(heldout[split]["miou"] for split in ("19", "15", "10")),
            name,
        )
    selected_name, selected = max(variants.items(), key=objective)
    report = {
        "schema_version": 1,
        "artifact_type": "radio_gs_scannet_categorical_minimal_cpu_screen",
        "status": "complete_target_selected_development_only",
        "eligibility": {
            "paper8_pseudo_labels_opened": True,
            "heldout6_metrics_used_for_variant_selection": True,
            "prediction_uses_target_labels": False,
            "paper_or_sota_claim_eligible": False,
        },
        "candidate_family": {
            "base": "CategoricalPosteriorV2 trained on scene0000_00 and scene0070_00",
            "variants": "four fixed shrinkage strengths crossed with background enabled/disabled",
            "persistent_scene_state_added": False,
            "selection_objective": "maximize worst heldout6 mIoU across 19/15/10, then sum mIoU",
        },
        "execution": {
            "device": str(device),
            "cuda_exact_primitive_replay_required": device.type == "cuda",
            "chunk_size": int(args.chunk_size),
        },
        "cpu_replay_audit": {
            scene: copy.deepcopy(data["primitive_replay"])
            for scene, data in scenes.items()
        },
        "selected_variant": selected_name,
        "selected_metrics": selected,
        "variants": variants,
        "provenance": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
            "inventory": {"path": str(inventory_path), "sha256": inventory_sha},
            "text_embeddings": {"path": str(text_path), "sha256": text_sha},
            "source": file_record(Path(__file__)),
            "scenes": {scene: copy.deepcopy(data["provenance"]) for scene, data in scenes.items()},
        },
    }
    output = Path(args.output).expanduser().resolve()
    write_frozen_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json")
    parser.add_argument("--universal-root", default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/universal_field_v1")
    parser.add_argument("--text-embeddings", default="checkpoints/siglip2_scannet_og_text_embeddings_ens5_split19.pt")
    parser.add_argument("--checkpoint", default="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/categorical_posterior_v2_pilot_train0000_0070_sourcebound_v2/categorical_posterior_v2_development_only.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shrinkage", default=",".join(str(value) for value in SHRINKAGE))
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 16:
        parser.error("--cpu-threads must be in [1,16]")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
