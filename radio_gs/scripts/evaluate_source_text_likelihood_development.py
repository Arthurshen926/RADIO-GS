#!/usr/bin/env python3
"""One-shot held-out source-development check for the frozen text calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.data.scannet_source_region_semantics import (
    validate_development_region_semantic_sidecar,
)
from radio_gs.querying.query_likelihood_head import (
    MonotoneQueryLikelihoodHead,
    QueryLikelihoodInputs,
)
from radio_gs.querying.source_text_query_likelihood import (
    LEGACY_FIELD_PRIOR_LOGIT_SCALE,
    SOURCE_TEXT_CHECKPOINT_SCHEMA,
    confidence_weighted_balanced_bce,
    sha256_file,
)
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.build_scannet_source_region_semantic_sidecar import (
    _write_json_noclobber,
)
from radio_gs.scripts.build_scannet_source_text_scene_input import (
    _field_coverage_reliability,
)


RESULT_SCHEMA = "radio_gs.source_text_query_likelihood_development_result.v1"


def _state_sha256(head: MonotoneQueryLikelihoodHead) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _weighted_summary(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    loss, details = confidence_weighted_balanced_bce(probability, target, weight)
    positive = weight * target
    negative = weight * (1.0 - target)
    positive_probability = (probability * positive).sum() / positive.sum()
    negative_probability = (probability * negative).sum() / negative.sum()
    return {
        "balanced_bce": float(loss),
        "positive_probability": float(positive_probability),
        "negative_probability": float(negative_probability),
        "positive_minus_negative_probability": float(
            positive_probability - negative_probability
        ),
        "positive_weight": float(details["positive_weight"]),
        "negative_weight": float(details["negative_weight"]),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("source development evaluator must remain CPU-only")
    paths = {
        "accepted_region_authority": Path(args.accepted_region_authority)
        .expanduser()
        .resolve(strict=True),
        "development_semantic_sidecar": Path(args.development_semantic_sidecar)
        .expanduser()
        .resolve(strict=True),
        "full_scalar_training_shard": Path(args.full_scalar_training_shard)
        .expanduser()
        .resolve(strict=True),
        "class_text_cache": Path(args.class_text_cache).expanduser().resolve(strict=True),
        "canonical_negative_text_cache": Path(args.canonical_negative_text_cache)
        .expanduser()
        .resolve(strict=True),
        "checkpoint": Path(args.checkpoint).expanduser().resolve(strict=True),
        "source_gate_receipt": Path(args.source_gate_receipt)
        .expanduser()
        .resolve(strict=True),
    }
    sidecar = validate_development_region_semantic_sidecar(
        torch.load(
            paths["development_semantic_sidecar"],
            map_location="cpu",
            weights_only=True,
        )
    )
    if sidecar["scene_id"] != "scene0003_00":
        raise ValueError("development evaluator accepts only frozen scene0003_00")
    accepted = torch.load(
        paths["accepted_region_authority"], map_location="cpu", weights_only=True
    )
    scalar = torch.load(
        paths["full_scalar_training_shard"], map_location="cpu", weights_only=True
    )
    descriptor = torch.as_tensor(accepted.get("accepted_v2_e0")).float()
    scalar_descriptor = torch.as_tensor(scalar.get("accepted_v2_e0")).float()
    if descriptor.shape != (4096, 1536) or not torch.equal(
        descriptor, scalar_descriptor
    ):
        raise ValueError("development descriptor authorities differ")
    if not torch.equal(
        torch.as_tensor(accepted.get("canonical_region_indices")).long(),
        sidecar["canonical_region_indices"].long(),
    ):
        raise ValueError("development semantic/descriptor row order differs")
    eligible = torch.as_tensor(scalar.get("eligible"))
    if eligible.shape != (4096,) or eligible.dtype != torch.bool:
        raise ValueError("development field eligibility differs")
    coverage, reliability = _field_coverage_reliability(
        scalar.get("raw_full_scalar_summary")
    )
    valid = sidecar["valid"] & eligible
    training_weight = (
        valid.float()
        * sidecar["semantic_coverage"].float()
        * coverage
        * reliability
    )

    class_ids = list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
    class_names = [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
    class_text = torch.load(
        paths["class_text_cache"], map_location="cpu", weights_only=True
    )
    negative_text = torch.load(
        paths["canonical_negative_text_cache"], map_location="cpu", weights_only=True
    )
    if class_text.get("queries") != class_names:
        raise ValueError("development class text authority differs")
    descriptor = F.normalize(descriptor, dim=-1)
    class_embedding = F.normalize(torch.as_tensor(class_text["embeddings"]).float(), dim=-1)
    negative_embedding = F.normalize(
        torch.as_tensor(negative_text["embeddings"]).float(), dim=-1
    )
    positive_cosine = descriptor @ class_embedding.T
    negative_cosine = descriptor @ negative_embedding.T
    negative_max = negative_cosine.amax(dim=1, keepdim=True)
    legacy_probability = torch.sigmoid(
        LEGACY_FIELD_PRIOR_LOGIT_SCALE * (positive_cosine - negative_max)
    )
    positive_affinity = ((positive_cosine + 1.0) * 0.5).clamp(0, 1)
    negative_affinity = ((negative_cosine + 1.0) * 0.5).clamp(0, 1)

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SOURCE_TEXT_CHECKPOINT_SCHEMA:
        raise ValueError("development checkpoint schema differs")
    if checkpoint.get("class_ids") != class_ids or checkpoint.get("class_names") != class_names:
        raise ValueError("development checkpoint vocabulary differs")
    head = MonotoneQueryLikelihoodHead(affinity_channel_count=1).cpu()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    before_sha = _state_sha256(head)
    target_distribution = sidecar["nyu40_class_distribution"][:, class_ids]
    rows = []
    for class_index, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
        target = target_distribution[:, class_index]
        positive_weight = float((training_weight * target).sum())
        negative_weight = float((training_weight * (1.0 - target)).sum())
        if positive_weight <= 0 or negative_weight <= 0:
            continue
        observations = QueryLikelihoodInputs(
            positive_affinity=positive_affinity[:, class_index, None],
            negative_affinity=negative_affinity,
            prior_probability=legacy_probability[:, class_index],
            coverage=coverage,
            reliability=reliability,
        ).validated()
        learned = head(observations, source="heldout_source_development")
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "legacy": _weighted_summary(
                    legacy_probability[:, class_index], target, training_weight
                ),
                "learned": _weighted_summary(
                    learned.foreground_probability, target, training_weight
                ),
            }
        )
    after_sha = _state_sha256(head)
    if before_sha != after_sha:
        raise RuntimeError("development evaluation mutated the frozen head")
    if not rows:
        raise ValueError("development scene contains no present split19 class")

    def macro(method: str, key: str) -> float:
        return float(sum(row[method][key] for row in rows) / len(rows))

    legacy_bce = macro("legacy", "balanced_bce")
    learned_bce = macro("learned", "balanced_bce")
    legacy_gap = macro("legacy", "positive_minus_negative_probability")
    learned_gap = macro("learned", "positive_minus_negative_probability")
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "complete_one_shot_heldout_source_development_no_callback",
        "scene_id": sidecar["scene_id"],
        "checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": sha256_file(paths["checkpoint"]),
            "state_sha256_before": before_sha,
            "state_sha256_after": after_sha,
        },
        "source_gate_receipt": {
            "path": str(paths["source_gate_receipt"]),
            "sha256": sha256_file(paths["source_gate_receipt"]),
        },
        "development_semantic_sidecar": {
            "path": str(paths["development_semantic_sidecar"]),
            "sha256": sha256_file(paths["development_semantic_sidecar"]),
        },
        "development_objective": {
            "present_class_count": len(rows),
            "legacy_macro_balanced_bce": legacy_bce,
            "learned_macro_balanced_bce": learned_bce,
            "balanced_bce_delta_learned_minus_legacy": learned_bce - legacy_bce,
            "legacy_macro_positive_minus_negative_probability": legacy_gap,
            "learned_macro_positive_minus_negative_probability": learned_gap,
            "probability_gap_delta_learned_minus_legacy": learned_gap - legacy_gap,
            "rows": rows,
        },
        "source_access": dict(sidecar["source_access"]),
        "execution": {
            "cuda_initialized": torch.cuda.is_initialized(),
            "parameter_callback_allowed": False,
            "parameter_callback_performed": False,
            "benchmark_metric_run": False,
            "scannet_exact_default_changed": False,
            "lerf_evaluator_changed": False,
        },
        "lineage": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--development-semantic-sidecar", required=True)
    parser.add_argument("--full-scalar-training-shard", required=True)
    parser.add_argument("--class-text-cache", required=True)
    parser.add_argument("--canonical-negative-text-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-gate-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run(args)
    output = _write_json_noclobber(args.output, payload)
    print(json.dumps({"output": str(output), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
