"""Evaluate one frozen likelihood checkpoint once on sealed scene0003 dev data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_training_dataset import (
    DATASET_SCHEMA,
)
from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    CHECKPOINT_SCHEMA,
    RECIPE,
    _evaluate,
    _readout_context,
    _sha256,
    _write_json_no_clobber,
)


@torch.inference_mode()
def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("development sentinel must start before CUDA initialization")
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != DATASET_SCHEMA or manifest.get("scene_count") != 1:
        raise ValueError("development sentinel requires one sealed likelihood scene")
    safety = manifest.get("safety", {})
    if (
        safety.get("labels_opened") is not True
        or safety.get("label_scope") != "official_source_train_scene_only"
        or safety.get("test_labels_opened") is not False
        or safety.get("full312_evaluation_authorized") is not False
    ):
        raise PermissionError("development dataset crosses the frozen label boundary")
    record = manifest["records"][0]
    if (
        record.get("scene_id") != "scene0003_00"
        or record.get("partition") != "development_validation"
        or record.get("test_labels_opened") is not False
    ):
        raise PermissionError("one-shot sentinel is sealed to development scene0003_00")
    shard = record["shard"]
    if _sha256(shard["path"]) != shard["sha256"]:
        raise ValueError("sealed development shard changed")
    payload = torch.load(shard["path"], map_location="cpu", weights_only=True)
    if payload.get("safety", {}).get("test_labels_opened") is not False:
        raise PermissionError("development payload crosses the test-label boundary")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if _sha256(checkpoint_path) != args.checkpoint_sha256:
        raise ValueError("frozen likelihood checkpoint SHA-256 differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("artifact_type") != CHECKPOINT_SCHEMA
        or checkpoint.get("recipe") != RECIPE
        or checkpoint.get("source_scene_ids") != ["scene0000_00"]
        or checkpoint.get("safety", {}).get("development_labels_opened") is not False
        or checkpoint.get("safety", {}).get("test_labels_opened") is not False
    ):
        raise ValueError("likelihood checkpoint is not the frozen scene0000 fit model")
    head = MonotoneQueryLikelihoodHead().cpu()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    bundle_path = Path(args.primitive_bundle).expanduser().resolve()
    context = _readout_context(payload, bundle_path)
    metrics = _evaluate(head, [payload], [context])
    receipt = {
        "schema_version": 1,
        "artifact_type": "monotone-query-likelihood-development-one-shot-v1",
        "status": "complete_no_selection_or_refit",
        "scene_id": "scene0003_00",
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "primitive_bundle": {
            "path": str(bundle_path),
            "sha256": _sha256(bundle_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": args.checkpoint_sha256,
        },
        "metrics": metrics,
        "safety": {
            "fit_scene": "scene0000_00",
            "development_scene": "scene0003_00",
            "development_labels_opened": True,
            "test_labels_opened": False,
            "checkpoint_refit": False,
            "threshold_selection": False,
            "full312_evaluation_run": False,
            "cuda_initialized": torch.cuda.is_initialized(),
        },
    }
    output = _write_json_no_clobber(args.receipt, receipt)
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--primitive-bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    path, receipt = run(parser.parse_args())
    print(json.dumps({"receipt": str(path), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
