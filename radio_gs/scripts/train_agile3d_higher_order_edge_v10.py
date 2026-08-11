"""Run the preregistered AGILE v10 edge-only source feasibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import FIXED_FIT_SCENES
from radio_gs.querying.higher_order_edge_likelihood import SymmetricHigherOrderEdgeLikelihood
from radio_gs.scripts.train_agile3d_instance_edge_likelihood_v9 import _primitive_instance_labels
from radio_gs.scripts.train_capability_likelihood_ratio_head import _load_manifests
from radio_gs.scripts.train_query_likelihood_head_fixed import _sha256, _write_json_no_clobber, _write_torch_no_clobber


FEATURE_ARTIFACT = "agile3d-query-independent-higher-order-edge-features-v10"
CHECKPOINT_ARTIFACT = "agile3d-higher-order-topology-edge-checkpoint-v10"
RECEIPT_ARTIFACT = "agile3d-higher-order-topology-edge-source-gate-v10"
RECIPE_ID = "higher-order-edge-balanced-density-ratio-mlp16x8-adam-seed0-e100-lr0.01-v10"


def _load_scene_data(
    manifest: Mapping[str, object],
    authority: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor | int], dict[str, object]]:
    path = Path(authority["path"]).resolve()
    if _sha256(path) != authority["sha256"]:
        raise ValueError("v10 query-independent feature artifact changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    safety = payload.get("safety", {})
    if (
        payload.get("artifact_type") != FEATURE_ARTIFACT
        or payload.get("scene_id") != manifest["scene_id"]
        or safety.get("query_independent") is not True
        or safety.get("labels_opened") is not False
        or safety.get("development_labels_opened") is not False
        or safety.get("test_labels_opened") is not False
        or safety.get("point_as_primitive_used") is not False
    ):
        raise PermissionError("v10 feature artifact violates sealed contract")
    node_count = int(payload["inventory"]["node_count"])
    primitive_labels, label_inventory = _primitive_instance_labels(
        manifest, node_count=node_count
    )
    edge_index = torch.as_tensor(payload["edge_index"]).long()
    features = torch.as_tensor(payload["features"]).half()
    row, col = edge_index
    authoritative = (primitive_labels[row] > 0) & (primitive_labels[col] > 0)
    target = primitive_labels[row[authoritative]] == primitive_labels[col[authoritative]]
    selected = features[authoritative]
    positive_count = int(target.sum())
    negative_count = int((~target).sum())
    if not positive_count or not negative_count:
        raise ValueError("v10 edge labels require both classes")
    label_inventory.update(
        {
            "authoritative_edge_count": int(authoritative.sum()),
            "same_instance_edge_count": positive_count,
            "different_instance_edge_count": negative_count,
            "omitted_edge_count": int((~authoritative).sum()),
            "feature_artifact": {"path": str(path), "sha256": _sha256(path)},
        }
    )
    return {"features": selected, "target": target}, label_inventory


def _evenly_spaced(rows: torch.Tensor, count: int) -> torch.Tensor:
    if len(rows) <= count:
        return rows
    positions = torch.div(
        torch.arange(count, dtype=torch.long) * len(rows),
        count,
        rounding_mode="floor",
    )
    return rows[positions]


def _training_rows(data: Mapping[str, torch.Tensor | int]) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.as_tensor(data["target"]).bool()
    positive = torch.nonzero(target, as_tuple=False).flatten()
    negative = torch.nonzero(~target, as_tuple=False).flatten()
    positive = _evenly_spaced(positive, len(negative))
    features = torch.as_tensor(data["features"])
    return features[positive].float(), features[negative].float()


def _train(
    data: Mapping[str, Mapping[str, torch.Tensor | int]],
    *,
    train_scenes: list[str],
    recipe: Mapping[str, object],
    device: torch.device,
) -> tuple[SymmetricHigherOrderEdgeLikelihood, list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(int(recipe["seed"]))
    feature_count = int(torch.as_tensor(data[train_scenes[0]]["features"]).shape[1])
    head = SymmetricHigherOrderEdgeLikelihood(
        feature_count=feature_count, hidden_width=16
    ).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    samples = {}
    inventory = {}
    for scene in train_scenes:
        positive, negative = _training_rows(data[scene])
        samples[scene] = (positive.to(device), negative.to(device))
        inventory[scene] = {
            "positive_training_rows": len(positive),
            "negative_training_rows": len(negative),
            "deterministic_evenly_spaced_positive_subsample": True,
        }
    trace = []
    for epoch in range(int(recipe["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        rankings = []
        for scene in train_scenes:
            positive, negative = samples[scene]
            positive_score = head.log_likelihood_ratio(positive)
            negative_score = head.log_likelihood_ratio(negative)
            losses.append(
                0.5 * F.softplus(-positive_score).mean()
                + 0.5 * F.softplus(negative_score).mean()
            )
            rankings.append(
                F.softplus(-(positive_score - negative_score)).mean()
            )
        balanced_bce = torch.stack(losses).mean()
        ranking = torch.stack(rankings).mean()
        objective = balanced_bce + 0.10 * ranking
        objective.backward()
        optimizer.step()
        if epoch in {0, 1, 2, 4, 9, 19, 49, 99}:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "objective": float(objective.detach()),
                    "balanced_bce": float(balanced_bce.detach()),
                    "ranking": float(ranking.detach()),
                }
            )
    return head.eval(), trace, inventory


@torch.inference_mode()
def _metrics(
    head: SymmetricHigherOrderEdgeLikelihood,
    data: Mapping[str, torch.Tensor | int],
    *,
    device: torch.device,
    chunk_size: int = 262144,
) -> dict[str, float]:
    features = torch.as_tensor(data["features"])
    target = torch.as_tensor(data["target"]).bool()
    score_parts = []
    for start in range(0, len(features), chunk_size):
        score_parts.append(
            head.log_likelihood_ratio(
                features[start : start + chunk_size].float().to(device)
            ).cpu()
        )
    score = torch.cat(score_parts)
    prediction = score >= 0
    tp = int((prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    positive_recall = tp / max(1, tp + fn)
    negative_recall = tn / max(1, tn + fp)
    return {
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "same_instance_precision": tp / max(1, tp + fp),
        "same_instance_recall": positive_recall,
        "different_instance_recall": negative_recall,
        "edge_keep_fraction": float(prediction.float().mean()),
        "mean_same_instance_llr": float(score[target].mean()),
        "mean_different_instance_llr": float(score[~target].mean()),
    }


def _checkpoint(
    head: SymmetricHigherOrderEdgeLikelihood,
    *,
    train_scenes: list[str],
    trace: list[dict[str, object]],
    preregistration: Path,
) -> dict[str, object]:
    return {
        "schema_version": 10,
        "artifact_type": CHECKPOINT_ARTIFACT,
        "head_class": "SymmetricHigherOrderEdgeLikelihood",
        "head_schema_version": head.schema_version,
        "feature_count": head.feature_count,
        "hidden_width": head.hidden_width,
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "train_scene_ids": train_scenes,
        "trace": trace,
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "safety": {
            "source_fit_edge_labels_opened": True,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "trajectory_run": False,
        },
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    preregistration = Path(args.preregistration).resolve()
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    recipe = prereg["training_recipe"]
    if recipe.get("recipe_id") != RECIPE_ID:
        raise ValueError("v10 recipe differs from preregistration")
    manifests = _load_manifests(args.dataset_manifest)
    data = {}
    inventories = {}
    for scene in FIXED_FIT_SCENES:
        data[scene], inventories[scene] = _load_scene_data(
            manifests[scene], prereg["query_independent_feature_authority"][scene]
        )
        print(json.dumps({"edge_data": scene, **inventories[scene]}), flush=True)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for heldout in FIXED_FIT_SCENES:
        train_scenes = [scene for scene in FIXED_FIT_SCENES if scene != heldout]
        head, trace, training_inventory = _train(
            data, train_scenes=train_scenes, recipe=recipe, device=device
        )
        checkpoint = _write_torch_no_clobber(
            output_dir / f"loo_holdout_{heldout}.pt",
            _checkpoint(
                head,
                train_scenes=train_scenes,
                trace=trace,
                preregistration=preregistration,
            ),
        )
        metric = _metrics(head, data[heldout], device=device)
        folds.append(
            {
                "heldout_scene": heldout,
                "train_scenes": train_scenes,
                "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                "training_inventory": training_inventory,
                "epoch_trace": trace,
                "metrics": metric,
            }
        )
        print(json.dumps({"fold_complete": heldout, "metrics": metric}), flush=True)
    macro_balanced = sum(row["metrics"]["balanced_accuracy"] for row in folds) / 3
    v9_reference = float(prereg["edge_only_source_validation"]["v9_loso_macro_balanced_accuracy_reference"])
    gates = {
        "macro_balanced_accuracy": macro_balanced,
        "macro_balanced_accuracy_pass": macro_balanced >= 0.70,
        "every_scene_different_instance_recall_pass": all(row["metrics"]["different_instance_recall"] >= 0.65 for row in folds),
        "every_scene_same_instance_recall_pass": all(row["metrics"]["same_instance_recall"] >= 0.60 for row in folds),
        "macro_balanced_accuracy_gain_over_v9": macro_balanced - v9_reference,
        "significant_gain_over_v9_pass": macro_balanced - v9_reference >= 0.10,
    }
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    receipt = {
        "schema_version": 10,
        "artifact_type": RECEIPT_ARTIFACT,
        "status": "edge_gate_pass_trajectory_authorized" if passed else "edge_gate_failed_stop_agile_head",
        "preregistration": {"path": str(preregistration), "sha256": _sha256(preregistration)},
        "feature_and_label_inventory": inventories,
        "leave_one_scene_out": folds,
        "gates": gates,
        "trajectory_authorized": passed,
        "development_authorized": False,
        "safety": {
            "source_fit_labels_opened": True,
            "trajectory_run": False,
            "development_labels_opened": False,
            "test_labels_opened": False,
            "test312_run": False,
            "point_as_primitive_used": False,
        },
    }
    receipt_path = _write_json_no_clobber(output_dir / "edge_source_gate_receipt.json", receipt)
    return receipt_path, receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    path, receipt = run(parse_args())
    print(json.dumps({"receipt": str(path), "status": receipt["status"], "gates": receipt["gates"]}, indent=2))


if __name__ == "__main__":
    main()
