"""CPU-only consumer smoke for a sealed query-likelihood training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_training_dataset import (
    DATASET_SCHEMA,
    iter_head_training_examples,
)
from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_no_clobber(path: str | Path, payload: dict[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different smoke receipt: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _losses(
    head: MonotoneQueryLikelihoodHead,
    examples: list[tuple[object, torch.Tensor, object]],
) -> list[torch.Tensor]:
    result = []
    for observations, target, _step in examples:
        evidence = head(observations, source="world_click_source_train")
        result.append(
            F.binary_cross_entropy(
                evidence.foreground_probability.clamp(1e-6, 1 - 1e-6),
                target.float(),
            )
        )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if torch.cuda.is_initialized():
        raise RuntimeError("consumer smoke must start before CUDA initialization")
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != DATASET_SCHEMA:
        raise ValueError("unexpected likelihood dataset manifest")
    safety = manifest.get("safety", {})
    if safety.get("test_labels_opened") is not False or safety.get(
        "full312_evaluation_authorized"
    ) is not False:
        raise ValueError("dataset manifest does not preserve the test boundary")
    examples = []
    shard_records = []
    for record in manifest["records"]:
        shard = record["shard"]
        if _sha256(shard["path"]) != shard["sha256"]:
            raise ValueError("sealed training shard changed")
        payload = torch.load(shard["path"], map_location="cpu", weights_only=True)
        examples.extend(iter_head_training_examples(payload))
        shard_records.append(dict(shard))
    if not examples:
        raise ValueError("sealed training dataset has no trajectory examples")

    torch.manual_seed(0)
    head = MonotoneQueryLikelihoodHead().cpu()
    with torch.no_grad():
        initial = float(torch.stack(_losses(head, examples)).mean().item())
    optimizer = torch.optim.Adam(head.parameters(), lr=float(args.learning_rate))
    for _epoch in range(int(args.epochs)):
        for observations, target, _step in examples:
            optimizer.zero_grad(set_to_none=True)
            evidence = head(observations, source="world_click_source_train")
            loss = F.binary_cross_entropy(
                evidence.foreground_probability.clamp(1e-6, 1 - 1e-6),
                target.float(),
            )
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        final = float(torch.stack(_losses(head, examples)).mean().item())
    payload = {
        "schema_version": 1,
        "artifact_type": "monotone-query-likelihood-source-train-consumer-smoke-v1",
        "status": "complete" if final < initial else "failed_no_loss_decrease",
        "device": "cpu",
        "dataset_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "shards": shard_records,
        "scene_count": int(manifest["scene_count"]),
        "trajectory_example_count": len(examples),
        "head_schema_version": head.schema_version,
        "trainable_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "initial_mean_bce": initial,
        "final_mean_bce": final,
        "loss_decreased": final < initial,
        "safety": {
            "cuda_initialized": torch.cuda.is_initialized(),
            "test_labels_opened": False,
            "full312_evaluation_run": False,
        },
    }
    output = _write_no_clobber(args.output, payload)
    return {"output": str(output), "sha256": _sha256(output), **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
