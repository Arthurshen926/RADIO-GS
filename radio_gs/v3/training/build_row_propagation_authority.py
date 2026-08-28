"""Build source-only region and multiscale-geometry row propagation authority."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import write_torch_noclobber
from radio_gs.v3.training.instance_upper_bound import sha256_file


def _voxel_keys(xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    lower, upper = xyz.amin(0), xyz.amax(0)
    normalized = (xyz - lower) / (upper - lower).clamp_min(1e-8)
    index = (normalized * resolution).floor().long().clamp(0, resolution - 1)
    return index[:, 0] * resolution * resolution + index[:, 1] * resolution + index[:, 2]


def _strongest_source_per_group(
    groups: torch.Tensor, rows: torch.Tensor, evidence: torch.Tensor
) -> dict[int, int]:
    authority: dict[int, int] = {}
    for group, row in zip(groups.tolist(), rows.tolist()):
        previous = authority.get(group)
        if previous is None or evidence[row] > evidence[previous]:
            authority[group] = row
    return authority


def _assign_group_best(
    *, target_rows: torch.Tensor, target_groups: torch.Tensor,
    source_rows: torch.Tensor, source_groups: torch.Tensor,
    xyz: torch.Tensor, evidence_unit: torch.Tensor, distance_scale: float,
    assigned_before: torch.Tensor, source_row: torch.Tensor, tier: torch.Tensor,
    confidence: torch.Tensor, tier_value: int, target_weight: torch.Tensor | None = None,
    mixture_source: torch.Tensor | None = None, mixture_weight: torch.Tensor | None = None,
) -> None:
    grouped: dict[int, list[int]] = {}
    for group, row in zip(source_groups.tolist(), source_rows.tolist()):
        grouped.setdefault(group, []).append(row)
    for index, (group, row) in enumerate(zip(target_groups.tolist(), target_rows.tolist())):
        if assigned_before[row]:
            continue
        candidates = grouped.get(group)
        if not candidates:
            continue
        candidate_rows = torch.tensor(candidates)
        distance = torch.linalg.vector_norm(xyz[candidate_rows] - xyz[row], dim=1)
        score = torch.exp(-distance / max(distance_scale, 1e-8)) * evidence_unit[candidate_rows]
        if target_weight is not None:
            score = score * target_weight[index]
        best = int(score.argmax())
        if score[best] > confidence[row]:
            source_row[row], tier[row], confidence[row] = candidate_rows[best], tier_value, score[best]
            if mixture_source is not None and mixture_weight is not None:
                count = min(mixture_source.shape[1], candidate_rows.numel())
                top_score, top = torch.topk(score, count)
                mixture_source[row].fill_(-1)
                mixture_weight[row].zero_()
                mixture_source[row, :count] = candidate_rows[top]
                mixture_weight[row, :count] = top_score / top_score.sum().clamp_min(1e-8)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    membership_path = Path(args.membership).resolve(strict=True)
    graph_path = Path(args.overlap_graph).resolve(strict=True)
    primitive_path = Path(args.primitive_cache).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu", weights_only=False)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    primitive = torch.load(primitive_path, map_location="cpu", weights_only=False)
    metadata = membership["metadata"]
    if graph["metadata"]["inputs"]["membership"]["sha256"] != sha256_file(membership_path):
        raise ValueError("overlap graph and membership authority differ")
    if str(metadata.get("primitive_cache")) != str(primitive_path):
        raise ValueError("primitive cache is not membership-bound")
    if any(metadata.get(key) is not False for key in (
        "benchmark_images_opened", "benchmark_masks_opened", "evaluation_rgb_opened"
    )):
        raise ValueError("row propagation requires sealed source-only inputs")

    xyz = torch.as_tensor(primitive["xyz"]).float()
    num_rows = int(membership["num_rows"])
    if xyz.shape != (num_rows, 3):
        raise ValueError("primitive geometry row mismatch")
    records = {int(value["source_view_index"]): value for value in metadata["source_records"]}
    evidence = torch.zeros(num_rows)
    for view in graph["metadata"]["selected_views"]:
        shard = torch.load(Path(records[int(view)]["responsibility_view"]), map_location="cpu", weights_only=False)
        evidence.index_add_(
            0, torch.as_tensor(shard["gaussian_ids"]).long(),
            torch.as_tensor(shard["base_weights"]).float(),
        )
    observed = evidence > 0
    positive_evidence = evidence[observed]
    evidence_scale = positive_evidence.median().clamp_min(1e-8)
    evidence_unit = 1 - torch.exp(-evidence / evidence_scale)
    assigned = observed.clone()
    source_row = torch.full((num_rows,), -1, dtype=torch.long)
    tier = torch.full((num_rows,), -1, dtype=torch.long)
    confidence = torch.zeros(num_rows)
    mixture_source = torch.full((num_rows, args.propagation_top_k), -1, dtype=torch.long)
    mixture_weight = torch.zeros(num_rows, args.propagation_top_k)
    source_row[observed] = torch.where(observed)[0]
    tier[observed] = 0
    confidence[observed] = 1
    mixture_source[observed, 0] = torch.where(observed)[0]
    mixture_weight[observed, 0] = 1

    member_rows = torch.as_tensor(membership["row_indices"]).long()
    proposals = torch.as_tensor(membership["proposal_indices"]).long()
    weights = torch.as_tensor(membership["weights"]).float()
    eligible = weights >= args.min_membership
    fine_resolution = int(args.voxel_resolutions[0])
    fine_keys = _voxel_keys(xyz, fine_resolution)
    composite_group = proposals * (fine_resolution ** 3) + fine_keys[member_rows]
    anchored = eligible & observed[member_rows]
    fine_cell_diagonal = 3 ** 0.5 / fine_resolution
    scene_diagonal = float(torch.linalg.vector_norm(xyz.amax(0) - xyz.amin(0)))
    _assign_group_best(
        target_rows=member_rows[eligible], target_groups=composite_group[eligible],
        source_rows=member_rows[anchored], source_groups=composite_group[anchored],
        xyz=xyz, evidence_unit=evidence_unit,
        distance_scale=scene_diagonal * fine_cell_diagonal,
        assigned_before=assigned.clone(), source_row=source_row, tier=tier,
        confidence=confidence, tier_value=1, target_weight=weights[eligible],
        mixture_source=mixture_source, mixture_weight=mixture_weight,
    )
    assigned |= source_row >= 0

    coverage_reports = [{"tier": "observed", "union_coverage": float(observed.float().mean())}, {
        "tier": "region", "union_coverage": float(assigned.float().mean())
    }]
    for tier_index, resolution in enumerate(args.voxel_resolutions, start=2):
        keys = _voxel_keys(xyz, resolution)
        observed_rows = torch.where(observed)[0]
        cell_diagonal = 3 ** 0.5 / resolution
        target_rows = torch.where(~assigned)[0]
        _assign_group_best(
            target_rows=target_rows, target_groups=keys[target_rows],
            source_rows=observed_rows, source_groups=keys[observed_rows],
            xyz=xyz, evidence_unit=evidence_unit,
            distance_scale=scene_diagonal * cell_diagonal,
            assigned_before=assigned.clone(), source_row=source_row, tier=tier,
            confidence=confidence, tier_value=tier_index,
            mixture_source=mixture_source, mixture_weight=mixture_weight,
        )
        assigned |= source_row >= 0
        coverage_reports.append({
            "tier": f"voxel_{resolution}", "union_coverage": float(assigned.float().mean())
        })

    target = torch.where(assigned)[0]
    payload = {
        "schema": "radio_gs.sugm_v3.row_propagation_authority.v1",
        "scene": membership["scene"],
        "columns": ["target_row", "source_row", "tier", "confidence", "xyz_distance"],
        "assignments": torch.stack((
            target, source_row[target], tier[target], confidence[target],
            torch.linalg.vector_norm(xyz[target] - xyz[source_row[target]], dim=1),
        ), dim=1),
        "mixture_source_rows": mixture_source[target],
        "mixture_weights": mixture_weight[target],
        "metadata": {
            "tier_definition": {"0": "observed", "1": "same anchored region and finest voxel", **{
                str(index): f"same {resolution}-voxel strongest observed source"
                for index, resolution in enumerate(args.voxel_resolutions, start=2)
            }},
            "coverage_reports": coverage_reports,
            "unassigned_rows": int((~assigned).sum()),
            "min_membership": args.min_membership,
            "voxel_resolutions": args.voxel_resolutions,
            "propagation_top_k": args.propagation_top_k,
            "source_only": True, "historical_field_opened": False,
            "benchmark_metrics_opened": False, "target_rgb_opened": False,
            "inputs": {
                "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
                "overlap_graph": {"path": str(graph_path), "sha256": sha256_file(graph_path)},
                "primitive_cache": {"path": str(primitive_path), "sha256": sha256_file(primitive_path)},
            },
        },
    }
    output = Path(args.output).resolve()
    write_torch_noclobber(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), "coverage_reports": coverage_reports, "unassigned_rows": int((~assigned).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--overlap-graph", required=True)
    parser.add_argument("--primitive-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-membership", type=float, default=0.5)
    parser.add_argument("--voxel-resolutions", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--propagation-top-k", type=int, default=4)
    args = parser.parse_args()
    if (
        not 0 <= args.min_membership <= 1 or not args.voxel_resolutions
        or any(value <= 0 for value in args.voxel_resolutions) or args.propagation_top_k <= 0
    ):
        raise ValueError("propagation authority budgets are invalid")
    print(run(args))


if __name__ == "__main__":
    main()
