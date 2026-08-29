"""Fill only unknown D128 rows using the shared masked semantic writer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-state", required=True)
    parser.add_argument("--semantic-memory", required=True)
    parser.add_argument("--masked-writer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve(strict=True)
        for name, value in (
            ("visual_state", args.visual_state), ("semantic_memory", args.semantic_memory),
            ("masked_writer", args.masked_writer),
        )
    }
    visual_payload = torch.load(paths["visual_state"], map_location="cpu")
    memory_payload = torch.load(paths["semantic_memory"], map_location="cpu")
    writer = torch.load(paths["masked_writer"], map_location="cpu")
    if writer.get("schema") != "radio_gs.sugm_v3.masked_semantic_writer.v1":
        raise ValueError("masked semantic memory writer differs")
    visual = torch.as_tensor(visual_payload["state_dict"]["memory"])[:, :320].float()
    semantic = torch.as_tensor(memory_payload["semantic"]).float().clone()
    direct = semantic.norm(dim=1) > 1e-6
    visual_known = visual.norm(dim=1) > 1e-6
    fill = ~direct & visual_known
    state = writer["state_dict"]
    x_mean = torch.as_tensor(state["x_mean"]).float()
    y_mean = torch.as_tensor(state["y_mean"]).float()
    weight = torch.as_tensor(state["weight"]).float()
    for chunk in torch.where(fill)[0].split(32768):
        semantic[chunk] = F.normalize(
            (F.normalize(visual[chunk], dim=-1, eps=1e-8) - x_mean) @ weight + y_mean,
            dim=-1, eps=1e-8,
        )
    known = semantic.norm(dim=1) > 1e-6
    validation_confidence = min(
        value["median_cosine"] for value in writer["heldout_reports"]
    )
    direct_confidence = torch.as_tensor(memory_payload["write_confidence"]).float()
    confidence = direct_confidence.clone()
    confidence[fill] = float(validation_confidence)
    payload = {
        "schema": "radio_gs.sugm_v3.conflict_aware_semantic_memory.v2",
        "scene": memory_payload["scene"], "semantic": semantic,
        "write_mass": memory_payload["write_mass"], "write_confidence": confidence,
        "direct_write_mask": direct, "predicted_write_mask": fill,
        "metadata": {
            **memory_payload["metadata"],
            "coverage_repair": "shared_masked_frozen_D320_to_D128_writer",
            "direct_rows": int(direct.sum()), "predicted_rows": int(fill.sum()),
            "known_rows": int(known.sum()), "unknown_rows": int((~known).sum()),
            "masked_validation_confidence_floor": float(validation_confidence),
            "inputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({
        "output": str(output), "sha256": sha256_file(output),
        "direct_rows": int(direct.sum()), "predicted_rows": int(fill.sum()),
        "known_rows": int(known.sum()), "unknown_rows": int((~known).sum()),
    })


if __name__ == "__main__":
    main()
