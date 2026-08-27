"""Source-dev RADIO fidelity gate for a folded SUGM-v3 D512 candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.field import load_factorized_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.run_instance_upper_bound import materialize_canonical_memory


@torch.no_grad()
def render_decoded_field(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_pixels: int,
    device: torch.device,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one decoded field through immutable exact compositor hits."""

    rows = torch.as_tensor(gaussian_ids).long().reshape(-1)
    pixels = torch.as_tensor(pixel_ids).long().reshape(-1)
    mass = torch.as_tensor(weights).float().reshape(-1)
    if not (rows.shape == pixels.shape == mass.shape) or chunk_size <= 0:
        raise ValueError("exact source visual hit axes differ")
    feature_dim = int(decoder.feature_dim)
    output = torch.zeros(int(num_pixels), feature_dim, device=device)
    alpha = torch.zeros(int(num_pixels), device=device)
    for start in range(0, rows.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), rows.numel())
        ids = rows[start:stop]
        target_pixels = pixels[start:stop].to(device)
        target_mass = mass[start:stop].to(device)
        coefficients = latent[ids].to(device)
        features = decoder(coefficients)
        output.index_add_(0, target_pixels, features * target_mass[:, None])
        alpha.index_add_(0, target_pixels, target_mass)
    return output, alpha


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    membership_path = Path(args.membership).resolve(strict=True)
    candidate_path = Path(args.candidate).resolve(strict=True)
    field_path = Path(args.field).resolve(strict=True)
    teacher_root = Path(args.teacher_root).resolve(strict=True)
    teacher_manifest = teacher_root / "frame_manifest.json"
    membership = torch.load(membership_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    latent = torch.as_tensor(candidate["state_dict"]["latent"]).float()
    metadata = candidate.get("metadata", {})
    if (
        metadata.get("state_representation")
        not in {
            "folded_single_d512_plus_global_head",
            "single_joint_d512_plus_global_head",
        }
        or (
            metadata.get("state_representation")
            == "folded_single_d512_plus_global_head"
            and metadata.get("low_rank_training_parameterization_discarded") is not True
        )
    ):
        raise ValueError("candidate is not a folded single-D512 artifact")
    field, _payload, _signature = load_factorized_canonical_field_checkpoint(
        field_path,
        map_location="cpu",
        expected_sha256=args.expected_field_sha256,
    )
    base = materialize_canonical_memory(field)
    if latent.shape != base.shape:
        raise ValueError("candidate and canonical field axes differ")
    device = torch.device(args.device)
    decoder = field.decoder.to(device).eval()
    records = [
        value
        for value in membership["metadata"]["source_records"]
        if int(value["source_view_index"]) % 4 == 3
    ]
    if not records:
        raise ValueError("source visual gate has no dev views")
    base_weighted = candidate_weighted = preservation_weighted = 0.0
    valid_pixels = 0
    per_view = []
    for record in records:
        shard_path = Path(record["responsibility_view"]).resolve(strict=True)
        shard = torch.load(shard_path, map_location="cpu")
        frame_id = int(record["frame_id"])
        teacher_path = teacher_root / "backbone" / f"rgb_{frame_id}.pt"
        teacher = torch.load(teacher_path, map_location="cpu").float()
        height, width = int(teacher.shape[1]), int(teacher.shape[2])
        num_pixels = height * width
        common = {
            "gaussian_ids": shard["gaussian_ids"],
            "pixel_ids": shard["pixel_ids"],
            "weights": shard["base_weights"],
            "num_pixels": num_pixels,
            "device": device,
            "chunk_size": args.chunk_size,
        }
        rendered_base, alpha = render_decoded_field(base, decoder, **common)
        rendered_candidate, candidate_alpha = render_decoded_field(
            latent, decoder, **common
        )
        alpha_max_difference = float((alpha - candidate_alpha).abs().max())
        if not torch.allclose(alpha, candidate_alpha, atol=1e-6, rtol=1e-6):
            raise RuntimeError("candidate changed exact compositor geometry")
        target = teacher.permute(1, 2, 0).reshape(num_pixels, -1).to(device)
        valid = alpha >= float(args.alpha_threshold)
        count = int(valid.sum())
        if not count:
            continue
        base_cosine = float(
            F.cosine_similarity(rendered_base[valid], target[valid], dim=-1).mean()
        )
        candidate_cosine = float(
            F.cosine_similarity(rendered_candidate[valid], target[valid], dim=-1).mean()
        )
        preservation = float(
            F.cosine_similarity(
                rendered_candidate[valid], rendered_base[valid], dim=-1
            ).mean()
        )
        base_weighted += base_cosine * count
        candidate_weighted += candidate_cosine * count
        preservation_weighted += preservation * count
        valid_pixels += count
        per_view.append({
            "frame_id": frame_id,
            "valid_pixels": count,
            "base_radio_cosine": base_cosine,
            "candidate_radio_cosine": candidate_cosine,
            "delta": candidate_cosine - base_cosine,
            "candidate_base_cosine": preservation,
            "repeated_alpha_max_abs_difference": alpha_max_difference,
            "responsibility_view_sha256": sha256_file(shard_path),
            "teacher_sha256": sha256_file(teacher_path),
        })
    if not valid_pixels:
        raise ValueError("source visual gate has no valid exact-render pixels")
    base_cosine = base_weighted / valid_pixels
    candidate_cosine = candidate_weighted / valid_pixels
    report = {
        "schema": "radio_gs.sugm_v3.source_visual_no_regression.v1",
        "split": "source_dev_view_residue_3",
        "valid_pixels": valid_pixels,
        "base_radio_cosine": base_cosine,
        "candidate_radio_cosine": candidate_cosine,
        "delta": candidate_cosine - base_cosine,
        "candidate_base_render_cosine": preservation_weighted / valid_pixels,
        "passed": candidate_cosine >= base_cosine,
        "per_view": per_view,
        "inputs": {
            "membership": {"path": str(membership_path), "sha256": sha256_file(membership_path)},
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "field": {"path": str(field_path), "sha256": sha256_file(field_path)},
            "teacher_manifest": {"path": str(teacher_manifest), "sha256": sha256_file(teacher_manifest)},
        },
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
    }
    write_frozen_json(Path(args.output).resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--expected-field-sha256", required=True)
    parser.add_argument("--teacher-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()


__all__ = ["render_decoded_field", "run"]
