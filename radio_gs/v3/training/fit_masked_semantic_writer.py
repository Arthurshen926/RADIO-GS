"""Fit a shared frozen-D320 to query-discriminative-D128 masked writer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _parse_scene(value: str) -> tuple[Path, Path]:
    parts = value.split("::")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("scene must be VISUAL_STATE::SEMANTIC_MEMORY")
    return tuple(Path(item).resolve(strict=True) for item in parts)  # type: ignore[return-value]


def _heldout_mask(rows: torch.Tensor, scene_index: int) -> torch.Tensor:
    return ((rows * 1103515247 + scene_index * 2654435761) % 5) == 0


def _load_pair(visual_path: Path, semantic_path: Path) -> tuple[dict, torch.Tensor, torch.Tensor]:
    visual_payload = torch.load(visual_path, map_location="cpu")
    semantic_payload = torch.load(semantic_path, map_location="cpu")
    metadata = visual_payload.get("metadata", {})
    if (
        not metadata.get("source_only") or metadata.get("historical_field_opened")
        or metadata.get("target_rgb_opened") or metadata.get("benchmark_metrics_opened")
    ):
        raise ValueError("masked semantic writer visual lineage differs")
    if semantic_payload.get("schema") != "radio_gs.sugm_v3.conflict_aware_semantic_memory.v1":
        raise ValueError("masked semantic writer target lineage differs")
    memory = torch.as_tensor(visual_payload["state_dict"]["memory"]).float()
    semantic = torch.as_tensor(semantic_payload["semantic"]).float()
    if memory.shape[0] != semantic.shape[0] or memory.shape[1] != 512:
        raise ValueError("masked semantic writer row axes differ")
    return semantic_payload, memory[:, :320], semantic


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", type=_parse_scene, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.scene) < 2 or args.ridge <= 0:
        raise ValueError("masked semantic writer fit budget differs")
    x_sum = torch.zeros(320, dtype=torch.float64)
    y_sum = torch.zeros(128, dtype=torch.float64)
    count = 0
    receipts = []
    for scene_index, (visual_path, semantic_path) in enumerate(args.scene):
        payload, visual, semantic = _load_pair(visual_path, semantic_path)
        rows = torch.arange(visual.shape[0])
        known = (visual.norm(dim=1) > 1e-6) & (semantic.norm(dim=1) > 1e-6)
        train = known & ~_heldout_mask(rows, scene_index)
        x = F.normalize(visual[train], dim=-1, eps=1e-8).double()
        y = F.normalize(semantic[train], dim=-1, eps=1e-8).double()
        x_sum += x.sum(0); y_sum += y.sum(0); count += x.shape[0]
        receipts.append({
            "scene": payload["scene"], "known_rows": int(known.sum()),
            "fit_rows": int(train.sum()),
            "visual_state": {"path": str(visual_path), "sha256": sha256_file(visual_path)},
            "semantic_memory": {"path": str(semantic_path), "sha256": sha256_file(semantic_path)},
        })
    x_mean, y_mean = x_sum / count, y_sum / count
    gram = torch.zeros((320, 320), dtype=torch.float64)
    cross = torch.zeros((320, 128), dtype=torch.float64)
    for scene_index, (visual_path, semantic_path) in enumerate(args.scene):
        _, visual, semantic = _load_pair(visual_path, semantic_path)
        rows = torch.arange(visual.shape[0])
        known = (visual.norm(dim=1) > 1e-6) & (semantic.norm(dim=1) > 1e-6)
        train_rows = torch.where(known & ~_heldout_mask(rows, scene_index))[0]
        for chunk in train_rows.split(32768):
            x = F.normalize(visual[chunk], dim=-1, eps=1e-8).double() - x_mean
            y = F.normalize(semantic[chunk], dim=-1, eps=1e-8).double() - y_mean
            gram += x.T @ x
            cross += x.T @ y
    regularizer = args.ridge * gram.diag().mean().clamp_min(1e-12)
    weight = torch.linalg.solve(
        gram + torch.eye(320, dtype=torch.float64) * regularizer, cross
    ).float()
    reports = []
    for scene_index, (visual_path, semantic_path) in enumerate(args.scene):
        payload, visual, semantic = _load_pair(visual_path, semantic_path)
        rows = torch.arange(visual.shape[0])
        known = (visual.norm(dim=1) > 1e-6) & (semantic.norm(dim=1) > 1e-6)
        heldout = known & _heldout_mask(rows, scene_index)
        prediction = F.normalize(
            (F.normalize(visual[heldout], dim=-1, eps=1e-8) - x_mean.float()) @ weight
            + y_mean.float(), dim=-1, eps=1e-8,
        )
        cosine = (prediction * F.normalize(semantic[heldout], dim=-1, eps=1e-8)).sum(1)
        reports.append({
            "scene": payload["scene"], "heldout_rows": int(heldout.sum()),
            "mean_cosine": float(cosine.mean()), "p10_cosine": float(torch.quantile(cosine, 0.1)),
            "median_cosine": float(cosine.median()),
        })
    payload = {
        "schema": "radio_gs.sugm_v3.masked_semantic_writer.v1",
        "state_dict": {"x_mean": x_mean.float(), "y_mean": y_mean.float(), "weight": weight},
        "heldout_reports": reports, "scene_receipts": receipts,
        "metadata": {
            "source_only": True, "source_train_residues": [1, 2],
            "masked_row_modulus": 5, "masked_row_residue": 0,
            "input": "frozen_source_only_D320", "target": "clean_query_discriminative_D128",
            "historical_field_opened": False, "target_rgb_opened": False,
            "benchmark_metrics_opened": False, "shared_across_scenes": True,
            "gaussian_indexed_high_dimensional_sidecars": 0, "ridge": args.ridge,
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    print({"output": str(output), "sha256": sha256_file(output), "reports": reports})


if __name__ == "__main__":
    main()


__all__ = ["_heldout_mask"]
