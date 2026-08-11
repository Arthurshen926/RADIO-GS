"""Train the six-parameter monotone likelihood head with one frozen recipe.

This source-train sentinel is deliberately in-sample.  It reports calibration
and click-step diagnostics but cannot be used to select a method variant; the
separate frozen development-scene command is the first generalization check.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_training_dataset import (
    DATASET_SCHEMA,
    iter_head_training_examples,
)
from radio_gs.querying.query_compilers import continuous_gaussian_readout
from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead


RECIPE = {
    "recipe_id": "monotone-query-likelihood-adam-seed0-e3-lr0.05-v1",
    "seed": 0,
    "optimizer": "Adam",
    "epochs": 3,
    "learning_rate": 0.05,
    "example_order": "sealed_manifest_then_click_ascending_no_shuffle",
    "objective": "unweighted_primitive_binary_cross_entropy",
    "probability_clamp": [1e-6, 1.0 - 1e-6],
}
CHECKPOINT_SCHEMA = "monotone-query-likelihood-head-checkpoint-v1"
RECEIPT_SCHEMA = "monotone-query-likelihood-fixed-source-train-receipt-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_no_clobber(path: str | Path, payload: Mapping[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different receipt: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_torch_no_clobber(path: str | Path, payload: Mapping[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"refusing to replace likelihood checkpoint: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _binary_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    q = torch.as_tensor(probability).float().reshape(-1)
    y = torch.as_tensor(target).bool().reshape(-1)
    if q.shape != y.shape or not q.numel():
        raise ValueError("metric probability and target must be aligned")
    prediction = q >= 0.5
    intersection = int((prediction & y).sum())
    union = int((prediction | y).sum())
    true_positive = intersection
    false_positive = int((prediction & ~y).sum())
    false_negative = int((~prediction & y).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "bce": float(
            F.binary_cross_entropy(q.clamp(1e-6, 1 - 1e-6), y.float()).item()
        ),
        "iou_at_0.5": float(intersection / union) if union else 0.0,
        "precision_at_0.5": float(precision),
        "recall_at_0.5": float(recall),
        "f1_at_0.5": float(2 * precision * recall / max(1e-12, precision + recall)),
        "accuracy_at_0.5": float((prediction == y).float().mean().item()),
        "predicted_foreground_fraction": float(prediction.float().mean().item()),
        "target_foreground_fraction": float(y.float().mean().item()),
    }


def _mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty metric rows")
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0]
    }


def _load_dataset(manifest_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != DATASET_SCHEMA:
        raise ValueError("unexpected likelihood dataset manifest")
    safety = manifest.get("safety", {})
    if (
        safety.get("labels_opened") is not True
        or safety.get("source_train_labels_opened") is not True
        or safety.get("label_scope") != "official_source_train_scene_only"
        or safety.get("test_labels_opened") is not False
        or safety.get("full312_evaluation_authorized") is not False
    ):
        raise ValueError("dataset does not disclose a sealed source-train label boundary")
    payloads = []
    for record in manifest["records"]:
        if record.get("partition") != "fit" or record.get("test_labels_opened") is not False:
            raise PermissionError("fixed head training accepts fit scenes only")
        shard = record["shard"]
        if _sha256(shard["path"]) != shard["sha256"]:
            raise ValueError("sealed source-train shard changed")
        payload = torch.load(shard["path"], map_location="cpu", weights_only=True)
        if payload.get("safety", {}).get("source_train_labels_opened") is not True:
            raise ValueError("training shard does not disclose source label access")
        if payload.get("safety", {}).get("test_labels_opened") is not False:
            raise PermissionError("training shard crosses the test-label boundary")
        payloads.append(payload)
    if not payloads:
        raise ValueError("sealed dataset has no source-train shards")
    return manifest, payloads


def _readout_context(
    payload: Mapping[str, object], primitive_bundle: str | Path | None
) -> dict[str, torch.Tensor] | None:
    adapter = str(payload.get("adapter", ""))
    if adapter == "released_5cm_point_identity_smoke_v1":
        return None
    if adapter != "canonical_primitive_bundle_v1":
        raise ValueError(f"unsupported source-train adapter: {adapter}")
    if primitive_bundle is None:
        raise ValueError("canonical Gaussian diagnostics require --primitive-bundle")
    path = Path(primitive_bundle).expanduser().resolve()
    authority = payload.get("source_authority", {}).get("primitive_bundle")
    if not isinstance(authority, Mapping) or authority.get("sha256") != _sha256(path):
        raise ValueError("primitive bundle differs from the sealed training shard")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "primitive_xyz",
        "primitive_covariance",
        "primitive_opacity",
        "official_point_xyz",
        "point_candidate_indices",
    }
    if not required <= set(bundle):
        raise ValueError("canonical bundle lacks continuous point-readout tensors")
    covariance = torch.as_tensor(bundle["primitive_covariance"]).float()
    identity = torch.eye(3, dtype=torch.float32)
    return {
        "primitive_xyz": torch.as_tensor(bundle["primitive_xyz"]).float(),
        "primitive_covariance": covariance,
        "primitive_precision": torch.linalg.pinv(covariance + 1e-6 * identity),
        "primitive_opacity": torch.as_tensor(bundle["primitive_opacity"]).float(),
        "official_point_xyz": torch.as_tensor(bundle["official_point_xyz"]).float(),
        "point_candidate_indices": torch.as_tensor(
            bundle["point_candidate_indices"]
        ).long(),
    }


@torch.inference_mode()
def _evaluate(
    head: MonotoneQueryLikelihoodHead,
    payloads: list[dict[str, object]],
    readout_contexts: list[dict[str, torch.Tensor] | None],
) -> dict[str, object]:
    primitive_rows = []
    point_rows = []
    by_click: dict[int, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {"primitive": [], "official_point": []}
    )
    example_count = 0
    for payload, context in zip(payloads, readout_contexts):
        point_target = torch.as_tensor(payload["point_target"]).bool()
        for observations, primitive_target, step in iter_head_training_examples(payload):
            evidence = head(observations, source="world_click_source_train")
            probability = evidence.foreground_probability
            primitive_metric = _binary_metrics(probability, primitive_target)
            if context is None:
                point_probability = probability
            else:
                point_probability, _support = continuous_gaussian_readout(
                    context["primitive_xyz"],
                    context["primitive_covariance"],
                    probability,
                    context["official_point_xyz"],
                    gaussian_precision=context["primitive_precision"],
                    opacity=context["primitive_opacity"],
                    candidate_indices=context["point_candidate_indices"],
                )
            point_metric = _binary_metrics(point_probability, point_target)
            click_count = int(step["click_count"])
            primitive_rows.append(primitive_metric)
            point_rows.append(point_metric)
            by_click[click_count]["primitive"].append(primitive_metric)
            by_click[click_count]["official_point"].append(point_metric)
            example_count += 1
    return {
        "example_count": example_count,
        "mean": {
            "primitive": _mean_metric_rows(primitive_rows),
            "official_point": _mean_metric_rows(point_rows),
        },
        "by_click_count": {
            str(click): {
                domain: _mean_metric_rows(rows)
                for domain, rows in domains.items()
            }
            for click, domains in sorted(by_click.items())
        },
    }


def _parameter_summary(head: MonotoneQueryLikelihoodHead) -> dict[str, object]:
    return {
        "bias": float(head.bias.detach()),
        "positive_weights": F.softplus(head.raw_positive_weights.detach()).tolist(),
        "negative_weights": F.softplus(head.raw_negative_weights.detach()).tolist(),
        "prior_weight": float(F.softplus(head.raw_prior_weight.detach())),
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("fixed source-train recipe must start before CUDA initialization")
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    manifest, payloads = _load_dataset(manifest_path)
    if len(payloads) != 1 or str(payloads[0].get("scene_id")) != "scene0000_00":
        raise ValueError("Stage-A fixed sentinel is sealed to fit scene0000_00")
    bundle_path = (
        Path(args.primitive_bundle).expanduser().resolve()
        if str(args.primitive_bundle).strip()
        else None
    )
    contexts = [_readout_context(payload, bundle_path) for payload in payloads]
    examples = [
        example
        for payload in payloads
        for example in iter_head_training_examples(payload)
    ]

    torch.manual_seed(int(RECIPE["seed"]))
    head = MonotoneQueryLikelihoodHead().cpu()
    initial = _evaluate(head, payloads, contexts)
    initial_parameters = _parameter_summary(head)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=float(RECIPE["learning_rate"])
    )
    epoch_mean_bce = []
    for _epoch in range(int(RECIPE["epochs"])):
        losses = []
        for observations, target, _step in examples:
            optimizer.zero_grad(set_to_none=True)
            evidence = head(observations, source="world_click_source_train")
            loss = F.binary_cross_entropy(
                evidence.foreground_probability.clamp(1e-6, 1 - 1e-6),
                target.float(),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_mean_bce.append(float(sum(losses) / len(losses)))
    final = _evaluate(head, payloads, contexts)
    final_parameters = _parameter_summary(head)

    checkpoint_payload = {
        "schema_version": 1,
        "artifact_type": CHECKPOINT_SCHEMA,
        "head_schema_version": head.schema_version,
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "recipe": RECIPE,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "source_scene_ids": ["scene0000_00"],
        "safety": {
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "test_labels_opened": False,
            "development_labels_opened": False,
            "full312_evaluation_run": False,
            "cuda_initialized": torch.cuda.is_initialized(),
        },
    }
    checkpoint = _write_torch_no_clobber(args.checkpoint, checkpoint_payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": RECEIPT_SCHEMA,
        "status": "complete",
        "device": "cpu",
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "adapter": str(payloads[0]["adapter"]),
        "primitive_bundle": (
            {"path": str(bundle_path), "sha256": _sha256(bundle_path)}
            if bundle_path is not None
            else None
        ),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "recipe": RECIPE,
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "epoch_mean_training_bce": epoch_mean_bce,
        "no_training": initial,
        "trained": final,
        "initial_parameters": initial_parameters,
        "trained_parameters": final_parameters,
        "safety": checkpoint_payload["safety"],
    }
    receipt_path = _write_json_no_clobber(args.receipt, receipt)
    return receipt_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--primitive-bundle", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", required=True)
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
