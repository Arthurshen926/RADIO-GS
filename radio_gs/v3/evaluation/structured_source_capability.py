"""Real source-heldout capability diagnostics for fresh structured D512."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.source_visual_no_regression import render_decoded_field
from radio_gs.v3.query.membership import pool_prototype
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file
from radio_gs.v3.training.run_instance_upper_bound import materialize_canonical_memory
from radio_gs.v3.training.structured_initialization import fixed_jl_projection


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    candidate_path = Path(args.candidate).resolve(strict=True)
    siglip_path = Path(args.siglip_teacher).resolve(strict=True)
    teacher_root = Path(args.radio_teacher_root).resolve(strict=True)
    membership = torch.load(membership_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    metadata = candidate["metadata"]
    if (
        metadata.get("state_representation")
        != "single_structured_shared_private_d512_plus_global_heads"
        or metadata.get("historical_field_opened") is not False
        or metadata.get("gaussian_indexed_sidecars") != 0
    ):
        raise ValueError("candidate is not a fresh structured single-D512")
    layout = metadata["layout"]
    memory = torch.as_tensor(candidate["state_dict"]["memory"]).float()
    if memory.ndim != 2 or memory.shape[1] != 512:
        raise ValueError("structured candidate memory axes differ")
    shared = memory[:, : int(layout["shared"])]
    semantic_start = int(layout["shared"])
    semantic = memory[:, semantic_start : semantic_start + int(layout["semantic"])]
    architecture = metadata.get("architecture", "hard_block_shared_private")
    state_dict = candidate["state_dict"]

    def target_image_view(value: torch.Tensor) -> torch.Tensor:
        if architecture != "learned_orthogonal_product":
            return value
        return F.pad(value, (0, 512 - value.shape[-1]))

    def candidate_image_view(value: torch.Tensor) -> torch.Tensor:
        output = target_image_view(value)
        if architecture != "learned_orthogonal_product":
            return output
        angles = torch.as_tensor(state_dict["basis_angles"], device=output.device)
        left_indices = torch.as_tensor(
            state_dict["basis_left"], device=output.device, dtype=torch.long
        )
        right_indices = torch.as_tensor(
            state_dict["basis_right"], device=output.device, dtype=torch.long
        )
        left = output[..., left_indices]
        right = output[..., right_indices]
        rotated = output.clone()
        rotated[..., left_indices] = angles.cos() * left - angles.sin() * right
        rotated[..., right_indices] = angles.sin() * left + angles.cos() * right
        return rotated
    device = torch.device(args.device)
    baseline_memory = baseline_decoder = baseline_path = None
    if args.baseline_field:
        baseline_path = Path(args.baseline_field).resolve(strict=True)
        field, _payload, _signature = load_factorized_canonical_field_checkpoint(
            baseline_path,
            map_location="cpu",
            expected_sha256=args.expected_baseline_field_sha256,
        )
        baseline_memory = materialize_canonical_memory(field)
        baseline_decoder = field.decoder.to(device).eval()

    records = [
        value for value in membership["metadata"]["source_records"]
        if int(value["source_view_index"]) % 4 == 3
    ]
    radio_values = []
    baseline_radio_values = []
    valid_pixels = 0
    for record in records:
        teacher = torch.load(
            teacher_root / "backbone" / f"rgb_{int(record['frame_id'])}.pt",
            map_location="cpu",
        ).float()
        projection = fixed_jl_projection(
            teacher.shape[0], int(layout["shared"]),
            int(metadata["initialization"]["radio_projection"]["seed"]),
        )
        target = F.normalize(
            teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0]) @ projection,
            dim=-1,
            eps=1e-8,
        ).to(device)
        target = target_image_view(target)
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
        ids = torch.as_tensor(shard["gaussian_ids"]).long()
        pixels = torch.as_tensor(shard["pixel_ids"]).long()
        weights = torch.as_tensor(shard["base_weights"]).float()
        rendered = torch.zeros(target.shape[0], shared.shape[1], device=device)
        alpha = torch.zeros(target.shape[0], device=device)
        for start in range(0, ids.numel(), args.hit_chunk):
            stop = min(start + args.hit_chunk, ids.numel())
            target_pixels = pixels[start:stop].to(device)
            mass = weights[start:stop].to(device)
            features = shared[ids[start:stop]].to(device)
            rendered.index_add_(0, target_pixels, features * mass[:, None])
            alpha.index_add_(0, target_pixels, mass)
        valid = alpha >= args.alpha_threshold
        rendered_view = candidate_image_view(rendered)
        radio_values.append(
            F.cosine_similarity(rendered_view[valid], target[valid], dim=-1)
        )
        if baseline_memory is not None and baseline_decoder is not None:
            baseline_render, baseline_alpha = render_decoded_field(
                baseline_memory,
                baseline_decoder,
                ids,
                pixels,
                weights,
                num_pixels=target.shape[0],
                device=device,
                chunk_size=args.hit_chunk,
            )
            if not torch.allclose(alpha, baseline_alpha, atol=1e-6, rtol=1e-6):
                raise RuntimeError("structured and historical exact compositor geometry differ")
            projected_baseline = target_image_view(
                baseline_render @ projection.to(device)
            )
            baseline_radio_values.append(
                F.cosine_similarity(projected_baseline[valid], target[valid], dim=-1)
            )
        valid_pixels += int(valid.sum())
    radio_cosine = float(torch.cat(radio_values).mean())
    baseline_radio_cosine = (
        float(torch.cat(baseline_radio_values).mean())
        if baseline_radio_values else None
    )

    pooled = []
    if int(layout["semantic"]):
        siglip = torch.load(siglip_path, map_location="cpu")
        descriptors = torch.as_tensor(siglip["descriptors"]).float()
        semantic_projection = fixed_jl_projection(
            descriptors.shape[1], int(layout["semantic"]),
            int(metadata["initialization"]["siglip_projection"]["seed"]),
        )
        target_semantic = F.normalize(descriptors @ semantic_projection, dim=-1, eps=1e-8)
        supports = proposal_supports(
            membership["row_indices"], membership["proposal_indices"],
            membership["weights"], int(membership["num_proposals"]),
        )
        dev = torch.where(torch.as_tensor(membership["proposal_view_indices"]).long() % 4 == 3)[0]
        targets = []
        for proposal in dev.tolist():
            rows, weights = supports[proposal]
            if rows.numel():
                pooled.append(pool_prototype(semantic[rows], weights))
                targets.append(target_semantic[proposal])
        pooled_tensor = torch.stack(pooled)
        target_tensor = torch.stack(targets)
        semantic_cosine = float(F.cosine_similarity(pooled_tensor, target_tensor, dim=-1).mean())
        similarity = pooled_tensor @ target_tensor.T
        retrieval_top1 = float((similarity.argmax(-1) == torch.arange(len(pooled))).float().mean())
    else:
        semantic_cosine = retrieval_top1 = None
    report = {
        "schema": "radio_gs.sugm_v3.structured_source_capability.v1",
        "architecture": architecture,
        "split": "source_dev_view_residue_3",
        "image_correspondence": {
            "projected_radio_render_cosine": radio_cosine,
            "historical_projected_radio_render_cosine": baseline_radio_cosine,
            "delta_from_historical": (
                radio_cosine - baseline_radio_cosine
                if baseline_radio_cosine is not None else None
            ),
            "valid_pixels": valid_pixels,
        },
        "semantic_identity": {
            "projected_siglip_proposal_cosine": semantic_cosine,
            "proposal_retrieval_top1": retrieval_top1,
            "proposals": len(pooled),
        },
        "protected_by_partition_owned_writes": True,
        "raw_radio_is_hard_gate": False,
        "category_source_metric": "pending_scannet_authority",
        "text_query_metric": "pending_registered_source_query_suite",
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "siglip_teacher": {"path": str(siglip_path), "sha256": sha256_file(siglip_path)},
            "historical_field_comparator": (
                {"path": str(baseline_path), "sha256": sha256_file(baseline_path)}
                if baseline_path is not None else None
            ),
        },
    }
    write_frozen_json(Path(args.output).resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--radio-teacher-root", required=True)
    parser.add_argument("--siglip-teacher", required=True)
    parser.add_argument("--baseline-field")
    parser.add_argument("--expected-baseline-field-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hit-chunk", type=int, default=32768)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.baseline_field) != bool(args.expected_baseline_field_sha256):
        raise ValueError("historical comparator requires a hash-bound field")
    print(run(args))


if __name__ == "__main__":
    main()
