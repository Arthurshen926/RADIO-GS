"""Real source-heldout capability diagnostics for fresh structured D512."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.evaluation.source_visual_no_regression import render_decoded_field
from radio_gs.v3.evaluation.comparator_view_audit import (
    audit_historical_comparator_views,
)
from radio_gs.v3.query.membership import pool_prototype
from radio_gs.v3.training.instance_upper_bound import proposal_supports, sha256_file
from radio_gs.v3.training.run_instance_upper_bound import materialize_canonical_memory
from radio_gs.v3.training.structured_initialization import fixed_jl_projection
from radio_gs.v3.training.learned_source_codec import apply_codec
from radio_gs.v3.training.native_visual_codec import GatedResidualVisualCodec, _load_dino


def _same_pixel_retrieval(
    query: torch.Tensor, target: torch.Tensor, sample_budget: int
) -> tuple[float, float, float]:
    """Retrieve the matching held-out source pixel among same-view candidates."""

    count = min(int(sample_budget), query.shape[0])
    if count <= 0:
        raise ValueError("same-pixel retrieval has no valid pixels")
    indices = torch.linspace(0, query.shape[0] - 1, count, device=query.device).long()
    query = F.normalize(query[indices], dim=-1, eps=1e-8)
    target = F.normalize(target[indices], dim=-1, eps=1e-8)
    similarity = query @ target.T
    diagonal = similarity.diag()
    rank = 1 + (similarity > diagonal[:, None]).sum(-1)
    if count == 1:
        margin = diagonal.new_zeros(())
    else:
        other = similarity.clone()
        other.fill_diagonal_(-torch.inf)
        margin = diagonal - other.max(-1).values
    return (
        float((rank == 1).float().mean()),
        float((rank <= min(5, count)).float().mean()),
        float(margin.mean()),
    )


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
    projection_type = metadata["initialization"]["radio_projection"]["type"]
    native_codec = projection_type == "native_gated_residual_radio_dino"
    learned_codec = projection_type == "learned_cross_scene_pca"
    semantic_learned = metadata["initialization"]["siglip_projection"]["type"] != "fixed_jl"
    if learned_codec:
        radio_mean = torch.as_tensor(state_dict["codec.radio_mean"]).float()
        radio_basis = torch.as_tensor(state_dict["codec.radio_basis"]).float()
    if semantic_learned:
        siglip_mean = torch.as_tensor(state_dict["codec.siglip_mean"]).float()
        siglip_basis = torch.as_tensor(state_dict["codec.siglip_basis"]).float()

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
    visual_model = None
    dino_root = None
    if native_codec:
        if args.baseline_field:
            raise ValueError("native RADIO+DINO codec has no historical RADIO-only comparator")
        dino_root = Path(metadata["dino_teacher_root"]).resolve(strict=True)
        visual_model = GatedResidualVisualCodec().to(device).eval()
        visual_model.load_state_dict({
            name.removeprefix("visual_codec."): torch.as_tensor(value)
            for name, value in state_dict.items() if name.startswith("visual_codec.")
        }, strict=True)
    baseline_memory = baseline_decoder = baseline_path = comparator_view_audit = None
    if args.baseline_field:
        baseline_path = Path(args.baseline_field).resolve(strict=True)
        field, payload, _signature = load_factorized_canonical_field_checkpoint(
            baseline_path,
            map_location="cpu",
            expected_sha256=args.expected_baseline_field_sha256,
        )
        baseline_memory = materialize_canonical_memory(field)
        baseline_decoder = field.decoder.to(device).eval()
        split = metadata.get("view_split", {})
        comparator_view_audit = audit_historical_comparator_views(
            payload,
            membership["metadata"]["source_records"],
            train_residues=split.get("train_residues", (1, 2)),
            dev_residue=int(split.get("dev_residue", 3)),
            audit_residue=int(split.get("audit_residue", 0)),
        )

    records = [
        value for value in membership["metadata"]["source_records"]
        if int(value["source_view_index"]) % 4 == 3
    ]
    radio_values = []
    baseline_radio_values = []
    retrieval_values = []
    baseline_retrieval_values = []
    teacher_ceiling_values = []
    valid_pixels = 0
    for record in records:
        teacher = torch.load(
            teacher_root / "backbone" / f"rgb_{int(record['frame_id'])}.pt",
            map_location="cpu",
        ).float()
        teacher_flat = teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0])
        if native_codec:
            assert visual_model is not None and dino_root is not None
            dino = _load_dino(
                dino_root / f"frame_{int(record['frame_id']):05d}.pt"
            ).float().permute(1, 2, 0).reshape(-1, 768)
            target_parts = []
            for start in range(0, teacher_flat.shape[0], args.teacher_pixel_chunk):
                stop = min(start + args.teacher_pixel_chunk, teacher_flat.shape[0])
                target_parts.append(visual_model.encode(
                    teacher_flat[start:stop].to(device), dino[start:stop].to(device)
                ))
            target = torch.cat(target_parts)
            projection = None
        else:
            projection = (
                radio_basis if learned_codec else fixed_jl_projection(
                    teacher.shape[0], int(layout["shared"]),
                    int(metadata["initialization"]["radio_projection"]["seed"]),
                )
            )
            target = F.normalize(
                apply_codec(
                    teacher_flat,
                    radio_mean if learned_codec else torch.zeros(teacher.shape[0]),
                    projection,
                ),
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
        retrieval_values.append(
            _same_pixel_retrieval(
                rendered_view[valid], target[valid], args.retrieval_samples_per_view
            )
        )
        teacher_ceiling_values.append(
            _same_pixel_retrieval(
                target[valid], target[valid], args.retrieval_samples_per_view
            )
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
            projected_baseline = target_image_view(apply_codec(
                baseline_render,
                radio_mean.to(device) if learned_codec else baseline_render.new_zeros(teacher.shape[0]),
                projection.to(device),
            ))
            baseline_radio_values.append(
                F.cosine_similarity(projected_baseline[valid], target[valid], dim=-1)
            )
            baseline_retrieval_values.append(
                _same_pixel_retrieval(
                    projected_baseline[valid], target[valid], args.retrieval_samples_per_view
                )
            )
        valid_pixels += int(valid.sum())
    radio_cosine = float(torch.cat(radio_values).mean())
    baseline_radio_cosine = (
        float(torch.cat(baseline_radio_values).mean())
        if baseline_radio_values else None
    )
    def mean_retrieval(values):
        return {
            "top1": sum(value[0] for value in values) / len(values),
            "top5": sum(value[1] for value in values) / len(values),
            "positive_margin": sum(value[2] for value in values) / len(values),
        } if values else None

    pooled = []
    if int(layout["semantic"]):
        siglip = torch.load(siglip_path, map_location="cpu")
        descriptors = torch.as_tensor(siglip["descriptors"]).float()
        if semantic_learned:
            target_semantic = F.normalize(
                apply_codec(descriptors, siglip_mean, siglip_basis), dim=-1, eps=1e-8
            )
        else:
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
            "same_view_same_pixel_retrieval": mean_retrieval(retrieval_values),
            "native_teacher_ceiling": mean_retrieval(teacher_ceiling_values),
            "historical_same_view_same_pixel_retrieval": mean_retrieval(
                baseline_retrieval_values
            ),
            "historical_delta_is_heldout_gate": (
                comparator_view_audit["eligible_as_heldout_gate"]
                if comparator_view_audit is not None else None
            ),
        },
        "historical_comparator_view_audit": comparator_view_audit,
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
    parser.add_argument("--retrieval-samples-per-view", type=int, default=512)
    parser.add_argument("--teacher-pixel-chunk", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.baseline_field) != bool(args.expected_baseline_field_sha256):
        raise ValueError("historical comparator requires a hash-bound field")
    print(run(args))


if __name__ == "__main__":
    main()
