"""Train and source-gate the preregistered signed/null-centered v4 head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from radio_gs.benchmarks.agile3d_scannet40.build_capability_likelihood_training_dataset import (
    CAPABILITY_CHANNELS,
)
from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import (
    FIXED_FIT_SCENES,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneSignedLikelihoodRatioHead,
)
from radio_gs.scripts.train_capability_likelihood_ratio_head import (
    _aggregate_scenes,
    _compile_scene_training,
    _evaluate_scene,
    _load_manifests,
    _load_v2,
    _scene_readout_context,
    _train_fold,
)
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    _sha256,
    _write_json_no_clobber,
    _write_torch_no_clobber,
)


RECIPE_ID = (
    "capability-signed-null-likelihood-ratio-bce-prior-correction-"
    "rank025-adam-seed0-e100-lr0.02-v4"
)
CHECKPOINT_SCHEMA = "monotone-signed-null-likelihood-ratio-head-checkpoint-v4"


def _checkpoint_payload(
    head: MonotoneSignedLikelihoodRatioHead,
    *,
    train_scenes: list[str],
    manifests: Mapping[str, Mapping[str, object]],
    preregistration: Path,
    recipe: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 4,
        "artifact_type": CHECKPOINT_SCHEMA,
        "head_class": "MonotoneSignedLikelihoodRatioHead",
        "head_schema_version": head.schema_version,
        "affinity_channels": list(CAPABILITY_CHANNELS),
        "affinity_transform": "signed_scene_centered_cosine_s_equals_2a_minus_1",
        "bias_trainable": False,
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "recipe": dict(recipe),
        "train_scene_ids": list(train_scenes),
        "dataset_manifests": {
            scene: {
                "path": manifests[scene]["_path"],
                "sha256": manifests[scene]["_sha256"],
            }
            for scene in train_scenes
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
        "safety": {
            "fit_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).expanduser().resolve()
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    recipe = prereg["training_recipe"]
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("v4 recipe differs from sealed preregistration")
    manifests = _load_manifests(args.dataset_manifest)
    device = torch.device(args.device)
    v2_path = Path(args.v2_checkpoint).expanduser().resolve()
    v2 = _load_v2(v2_path, device=device)
    readout_contexts = {
        scene: _scene_readout_context(manifest, device=device)
        for scene, manifest in manifests.items()
    }
    compiled_scenes = {}
    inventories = {}
    for scene, manifest in manifests.items():
        compiled_scenes[scene], inventories[scene] = _compile_scene_training(
            manifest, null_centered=True
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        head, trace, training_inventory = _train_fold(
            compiled_scenes,
            inventories,
            train_scenes=train_scenes,
            recipe=recipe,
            device=device,
            head_class=MonotoneSignedLikelihoodRatioHead,
        )
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.pt",
            _checkpoint_payload(
                head,
                train_scenes=train_scenes,
                manifests=manifests,
                preregistration=preregistration,
                recipe=recipe,
            ),
        )
        v2_metric = _evaluate_scene(
            v2,
            manifests[heldout],
            device=device,
            context=readout_contexts[heldout],
        )
        v4_metric = _evaluate_scene(
            head.eval(),
            manifests[heldout],
            device=device,
            context=readout_contexts[heldout],
        )
        folds.append(
            {
                "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "epoch_trace": trace,
                "training_inventory": training_inventory,
                "v2": v2_metric,
                "v4": v4_metric,
                "mean_iou_gain": (
                    v4_metric["mean"]["iou_at_0.5"]
                    - v2_metric["mean"]["iou_at_0.5"]
                ),
                "absolute_log_mass_error_gain": (
                    v2_metric["mean"]["absolute_log_probability_mass_ratio_error"]
                    - v4_metric["mean"]["absolute_log_probability_mass_ratio_error"]
                ),
            }
        )

    all_head, all_trace, all_training_inventory = _train_fold(
        compiled_scenes,
        inventories,
        train_scenes=list(FIXED_FIT_SCENES),
        recipe=recipe,
        device=device,
        head_class=MonotoneSignedLikelihoodRatioHead,
    )
    all_checkpoint = _write_torch_no_clobber(
        output_dir / "all_fit_scene0000_0002_0005.pt",
        _checkpoint_payload(
            all_head,
            train_scenes=list(FIXED_FIT_SCENES),
            manifests=manifests,
            preregistration=preregistration,
            recipe=recipe,
        ),
    )
    all_v2 = _aggregate_scenes(
        [
            _evaluate_scene(
                v2,
                manifests[scene],
                device=device,
                context=readout_contexts[scene],
            )
            for scene in FIXED_FIT_SCENES
        ]
    )
    all_v4 = _aggregate_scenes(
        [
            _evaluate_scene(
                all_head.eval(),
                manifests[scene],
                device=device,
                context=readout_contexts[scene],
            )
            for scene in FIXED_FIT_SCENES
        ]
    )

    loo_mean_gain = sum(float(row["mean_iou_gain"]) for row in folds) / len(folds)
    loo_scene_passes = sum(float(row["mean_iou_gain"]) > 0.02 for row in folds)
    loo_click_pass = all(
        float(row["v4"]["click10_minus_click1_iou"]) > 0.02 for row in folds
    )
    loo_mass_pass = (
        sum(
            float(row["v4"]["mean"]["absolute_log_probability_mass_ratio_error"])
            for row in folds
        )
        < sum(
            float(row["v2"]["mean"]["absolute_log_probability_mass_ratio_error"])
            for row in folds
        )
    )
    all_iou_gain = (
        all_v4["scene_macro_mean"]["iou_at_0.5"]
        - all_v2["scene_macro_mean"]["iou_at_0.5"]
    )
    all_precision_gain = (
        all_v4["scene_macro_mean"]["precision_at_0.5"]
        - all_v2["scene_macro_mean"]["precision_at_0.5"]
    )
    gates = {
        "loo_macro_mean_iou_gain": loo_mean_gain,
        "loo_macro_mean_iou_gain_pass": loo_mean_gain > 0.02,
        "loo_scenes_with_iou_gain_over_0.02": loo_scene_passes,
        "loo_scene_count_pass": loo_scene_passes >= 2,
        "loo_each_scene_click_response_over_0.02_pass": loo_click_pass,
        "loo_mass_calibration_improves_pass": loo_mass_pass,
        "all_fit_mean_iou_gain": all_iou_gain,
        "all_fit_mean_iou_gain_pass": all_iou_gain > 0.02,
        "all_fit_precision_gain": all_precision_gain,
        "all_fit_precision_gain_pass": all_precision_gain > 0.02,
        "all_fit_click10_minus_click1_iou": all_v4[
            "scene_macro_click10_minus_click1_iou"
        ],
        "all_fit_click_response_pass": all_v4[
            "scene_macro_click10_minus_click1_iou"
        ]
        > 0.05,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 4,
        "artifact_type": "capability-signed-null-likelihood-ratio-source-gate-v4",
        "status": (
            "pass_source_gates_development_authorized"
            if passed
            else "fail_source_gates_do_not_open_development"
        ),
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
        "dataset_manifests": {
            scene: {"path": row["_path"], "sha256": row["_sha256"]}
            for scene, row in manifests.items()
        },
        "reference_v2_checkpoint": {"path": str(v2_path), "sha256": _sha256(v2_path)},
        "leave_one_scene_out": folds,
        "all_fit": {
            "checkpoint": {"path": str(all_checkpoint), "sha256": _sha256(all_checkpoint)},
            "epoch_trace": all_trace,
            "training_inventory": all_training_inventory,
            "v2": all_v2,
            "v4": all_v4,
        },
        "gates": {**gates, "all_pass": passed, "development_authorized": passed},
        "safety": {
            "fit_scene_ids": list(FIXED_FIT_SCENES),
            "development_scene_id": "scene0003_00",
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
        },
    }
    receipt_path = _write_json_no_clobber(args.receipt, receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--device", default="cpu")
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
