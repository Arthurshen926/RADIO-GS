#!/usr/bin/env python3
"""One-shot scene0003 source-development gate for the frozen spatial head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.data.scannet_source_region_semantics import (
    validate_development_region_semantic_sidecar,
)
from radio_gs.querying.source_spatial_text_likelihood import (
    BoundedSourceSpatialLikelihoodHead,
    SOURCE_SPATIAL_CHECKPOINT_SCHEMA,
    fixed_knn_indices,
    fixed_spatial_logit_statistics,
    sha256_file,
    state_dict_sha256,
)
from radio_gs.scannet_constants import (
    NYU40_ID_TO_NAME,
    OPENGAUSSIAN_NYU40_CLASS_SPLITS,
)
from radio_gs.scripts.build_scannet_source_text_scene_input import (
    _field_coverage_reliability,
)
from radio_gs.scripts.build_source_spatial_text_likelihood_shard import (
    canonical_region_xyz,
)
from radio_gs.scripts.train_source_spatial_text_likelihood import (
    evaluate_source_spatial_objective,
)


RESULT_SCHEMA = "radio_gs.source_spatial_text_likelihood_development_result.v1"


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("source spatial development gate must remain CPU-only")
    paths = {
        name: Path(value).expanduser().resolve(strict=True)
        for name, value in {
            "accepted_region_authority": args.accepted_region_authority,
            "query_independent_support_graph": args.query_independent_support_graph,
            "development_semantic_sidecar": args.development_semantic_sidecar,
            "full_scalar_training_shard": args.full_scalar_training_shard,
            "class_text_cache": args.class_text_cache,
            "canonical_negative_text_cache": args.canonical_negative_text_cache,
            "checkpoint": args.checkpoint,
            "source_gate_receipt": args.source_gate_receipt,
        }.items()
    }
    gate_receipt = json.loads(paths["source_gate_receipt"].read_text())
    if (
        gate_receipt.get("diagnostics", {}).get("gate", {}).get("all_passed")
        is not True
        or gate_receipt.get("heldout_development_opened") is not False
    ):
        raise PermissionError("source-fit gates did not authorize development open")
    sidecar = validate_development_region_semantic_sidecar(
        torch.load(
            paths["development_semantic_sidecar"],
            map_location="cpu",
            weights_only=True,
        )
    )
    if sidecar["scene_id"] != "scene0003_00":
        raise ValueError("development gate accepts only scene0003_00")
    accepted = torch.load(
        paths["accepted_region_authority"], map_location="cpu", weights_only=False
    )
    graph = torch.load(
        paths["query_independent_support_graph"], map_location="cpu", weights_only=False
    )
    scalar = torch.load(
        paths["full_scalar_training_shard"], map_location="cpu", weights_only=True
    )
    descriptor = torch.as_tensor(accepted.get("accepted_v2_e0")).float()
    if (
        accepted.get("scene_id") != "scene0003_00"
        or descriptor.shape != (4096, 1536)
        or not torch.equal(descriptor, torch.as_tensor(scalar.get("accepted_v2_e0")).float())
        or not torch.equal(
            torch.as_tensor(accepted.get("canonical_region_indices")).long(),
            sidecar["canonical_region_indices"].long(),
        )
    ):
        raise ValueError("development MPR/semantic/scalar row authorities differ")
    eligible = torch.as_tensor(scalar.get("eligible"))
    if eligible.shape != (4096,) or eligible.dtype != torch.bool:
        raise ValueError("development scalar eligibility differs")
    coverage, reliability = _field_coverage_reliability(
        scalar.get("raw_full_scalar_summary")
    )
    valid = sidecar["valid"] & eligible
    training_weight = sidecar["semantic_coverage"].float().contiguous()

    class_ids = list(OPENGAUSSIAN_NYU40_CLASS_SPLITS["19"])
    class_names = [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
    class_text = torch.load(
        paths["class_text_cache"], map_location="cpu", weights_only=True
    )
    negative_text = torch.load(
        paths["canonical_negative_text_cache"], map_location="cpu", weights_only=True
    )
    if class_text.get("queries") != class_names:
        raise ValueError("development class vocabulary differs")
    descriptor = F.normalize(descriptor, dim=-1)
    class_embedding = F.normalize(
        torch.as_tensor(class_text["embeddings"]).float(), dim=-1
    )
    negative_embedding = F.normalize(
        torch.as_tensor(negative_text["embeddings"]).float(), dim=-1
    )
    positive_cosine = descriptor @ class_embedding.T
    negative_cosine = descriptor @ negative_embedding.T
    raw_logit = (
        10.0 * (positive_cosine - negative_cosine.amax(dim=1, keepdim=True))
    )[:, None, :].contiguous()
    region_xyz = canonical_region_xyz(accepted, graph)
    neighbors = fixed_knn_indices(region_xyz)
    mean, maximum, contrast = fixed_spatial_logit_statistics(
        raw_logit, neighbors, valid=valid
    )
    payload = {
        "scene_id": "scene0003_00",
        "physical_space_id": "scene0003",
        "raw_logit": raw_logit,
        "neighbor_mean_logit": mean,
        "neighbor_max_logit": maximum,
        "neighbor_contrast_logit": contrast,
        "neighbor_indices": neighbors,
        "semantic_class_distribution": sidecar["nyu40_class_distribution"][:, class_ids]
        .float()
        .contiguous(),
        "valid": valid.contiguous(),
        "coverage": coverage,
        "reliability": reliability,
        "training_label_weight": training_weight,
        "class_ids": class_ids,
        "class_names": class_names,
    }
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("schema") != SOURCE_SPATIAL_CHECKPOINT_SCHEMA
        or checkpoint.get("source_scene_ids")
        != ["scene0001_00", "scene0002_00", "scene0005_00"]
        or checkpoint.get("class_ids") != class_ids
        or checkpoint.get("class_names") != class_names
    ):
        raise ValueError("development checkpoint contract differs")
    head = BoundedSourceSpatialLikelihoodHead().cpu()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    before = state_dict_sha256(head.state_dict())
    objective = evaluate_source_spatial_objective(head, [payload])
    after = state_dict_sha256(head.state_dict())
    if before != after or before != checkpoint.get("state_dict_sha256"):
        raise RuntimeError("development gate mutated or mismatched the frozen head")
    gate = {
        "balanced_bce_improved": objective["balanced_bce_delta"] < 0,
        "positive_negative_gap_improved": objective["positive_negative_gap_delta"] > 0,
        "local_relation_improved": objective["local_relation_loss_delta"] < 0,
    }
    gate["all_passed"] = all(gate.values())
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "complete_one_shot_heldout_source_development_no_callback",
        "scene_id": "scene0003_00",
        "checkpoint": {
            **_record(paths["checkpoint"]),
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "source_gate_receipt": _record(paths["source_gate_receipt"]),
        "development_objective": objective,
        "gate": gate,
        "lineage": {name: _record(path) for name, path in paths.items()},
        "source_access": dict(sidecar["source_access"]),
        "execution": {
            "cuda_initialized": torch.cuda.is_initialized(),
            "parameter_callback_allowed": False,
            "parameter_callback_performed": False,
            "benchmark_metric_run": False,
            "scannet_exact_default_changed": False,
            "lerf_metric_run": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-region-authority", required=True)
    parser.add_argument("--query-independent-support-graph", required=True)
    parser.add_argument("--development-semantic-sidecar", required=True)
    parser.add_argument("--full-scalar-training-shard", required=True)
    parser.add_argument("--class-text-cache", required=True)
    parser.add_argument("--canonical-negative-text-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-gate-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
