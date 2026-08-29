"""Install a clean selected D128 semantic memory into the retained D512+R5."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.contracts.method import validate_scene_state
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-scene-state", required=True)
    parser.add_argument("--semantic-memory", required=True)
    parser.add_argument("--semantic-codec", required=True)
    parser.add_argument("--confidence-scaled", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("parent_scene_state", args.parent_scene_state),
            ("semantic_memory", args.semantic_memory),
            ("semantic_codec", args.semantic_codec),
        )
    }
    parent = torch.load(paths["parent_scene_state"], map_location="cpu")
    memory = torch.load(paths["semantic_memory"], map_location="cpu")
    codec = torch.load(paths["semantic_codec"], map_location="cpu")
    if (
        parent.get("schema") != "radio_gs.sugm_v3.unknown_aware_scene_state.v1"
        or not parent.get("metadata", {}).get("source_only")
        or parent.get("metadata", {}).get("target_rgb_opened")
        or parent.get("metadata", {}).get("benchmark_metrics_opened")
    ):
        raise ValueError("clean semantic parent scene-state lineage differs")
    if memory.get("schema") not in (
        "radio_gs.sugm_v3.conflict_aware_semantic_memory.v1",
        "radio_gs.sugm_v3.conflict_aware_semantic_memory.v2",
    ) or codec.get("schema") != "radio_gs.sugm_v3.query_discriminative_semantic_codec.v1":
        raise ValueError("clean semantic state input lineage differs")
    latent = torch.as_tensor(parent["latent"]).float()
    semantic = torch.as_tensor(memory["semantic"]).float()
    if parent["scene"] != memory["scene"] or semantic.shape != (latent.shape[0], 128):
        raise ValueError("clean semantic scene or row domain differs")
    if args.confidence_scaled:
        confidence = torch.as_tensor(memory["write_confidence"]).float().clamp(0, 1)
        semantic = semantic * confidence[:, None]
    candidate = latent.clone()
    candidate[:, 320:448] = semantic
    reliability = torch.as_tensor(parent["reliability"]).float()
    membership = parent["metadata"]["inputs"]["membership"]
    validate_scene_state(
        candidate, reliability, source_authority_sha256=membership["sha256"]
    )
    global_state = dict(parent["global_state_dict"])
    global_state["codec.siglip_mean"] = torch.as_tensor(
        codec["state_dict"]["siglip_mean"]
    ).float()
    global_state["codec.siglip_basis"] = torch.as_tensor(
        codec["state_dict"]["siglip_basis"]
    ).float()
    payload = {
        **parent, "latent": candidate, "global_state_dict": global_state,
        "metadata": {
            **parent["metadata"],
            "clean_semantic_rewrite": {
                "confidence_scaled": bool(args.confidence_scaled),
                "known_rows": int((semantic.norm(dim=1) > 1e-6).sum()),
                "parent": {"path": str(paths["parent_scene_state"]), "sha256": sha256_file(paths["parent_scene_state"])},
                "semantic_memory": {"path": str(paths["semantic_memory"]), "sha256": sha256_file(paths["semantic_memory"])},
                "semantic_codec": {"path": str(paths["semantic_codec"]), "sha256": sha256_file(paths["semantic_codec"])},
            },
            "historical_language_authority_opened": False,
            "source_only": True, "target_rgb_opened": False,
            "benchmark_metrics_opened": False,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({
        "output": str(output), "sha256": sha256_file(output),
        "known_rows": int((semantic.norm(dim=1) > 1e-6).sum()),
        "d320_max_abs_delta": float((candidate[:, :320] - latent[:, :320]).abs().max()),
        "d48_d16_max_abs_delta": float((candidate[:, 448:] - latent[:, 448:]).abs().max()),
        "r5_max_abs_delta": float((reliability - torch.as_tensor(parent["reliability"]).float()).abs().max()),
    })


if __name__ == "__main__":
    main()
