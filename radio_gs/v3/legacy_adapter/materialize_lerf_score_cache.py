"""Bridge a frozen v3 posterior to the immutable historical LERF renderer schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radio_gs.v3.query.interface import load_query_interface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.training.instance_upper_bound import sha256_file


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--geometry-cache", required=True)
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--text-alignment")
    parser.add_argument("--text-negatives")
    parser.add_argument("--text-logit-scale", type=float, default=10.0)
    parser.add_argument("--membership-margin", type=float, default=0.0)
    parser.add_argument("--membership-margin-authority")
    parser.add_argument("--center-semantic-identity", action="store_true")
    parser.add_argument("--identity-extent-weight", type=float, default=0.0)
    parser.add_argument("--identity-extent-authority")
    parser.add_argument("--posterior-chunk-size", type=int, default=65536)
    parser.add_argument("--extent-fraction", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state_path = Path(args.scene_state).resolve(strict=True)
    geometry_path = Path(args.geometry_cache).resolve(strict=True)
    text_path = Path(args.text_embeddings).resolve(strict=True)
    geometry = torch.load(geometry_path, map_location="cpu")
    text = torch.load(text_path, map_location="cpu")
    queries = [str(value) for value in geometry["metadata"]["query_names"]]
    lookup = {str(value).casefold(): index for index, value in enumerate(text["queries"])}
    if any(query.casefold() not in lookup for query in queries):
        raise ValueError("frozen text cache lacks renderer query axis")
    interface = load_query_interface(
        state_path, device=args.device, text_alignment_path=args.text_alignment,
        text_negative_path=args.text_negatives, text_logit_scale=args.text_logit_scale,
        center_semantic_identity=args.center_semantic_identity,
    )
    xyz = torch.as_tensor(geometry["xyz"]).float()
    if xyz.shape != (interface.model.memory.shape[0], 3):
        raise ValueError("renderer geometry row domain differs from v3 state")
    posterior, identity = [], []
    embeddings = torch.as_tensor(text["embeddings"]).float()
    for query in queries:
        token = embeddings[lookup[query.casefold()]]
        value, score = interface.posterior_from_packet(
            QueryPacket("text", token=token), scale=args.scale,
            topk=args.topk, temperature=args.temperature,
            membership_margin=args.membership_margin,
            identity_extent_weight=args.identity_extent_weight,
            posterior_chunk_size=args.posterior_chunk_size,
        )
        if args.extent_fraction:
            if not 0.0 < args.extent_fraction <= 1.0:
                raise ValueError("extent fraction differs")
            budget = max(1, round(value.numel() * args.extent_fraction))
            keep = value.topk(budget).indices
            bounded = torch.zeros_like(value)
            bounded[keep] = value[keep]
            value = bounded
        posterior.append(value.cpu())
        identity.append(score.cpu())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "radio_gs.lerf_sam_siglip_object_posterior_scores.v1",
        "schema_version": 1,
        "scene": torch.load(state_path, map_location="cpu")["scene"],
        "query_scores": torch.stack(posterior, dim=1),
        "identity_query_scores": torch.stack(identity, dim=1),
        "valid": torch.ones(xyz.shape[0], dtype=torch.bool),
        "xyz": xyz,
        "metadata": {
            "query_names": queries,
            "query_family": "text_object_extent",
            "construction": "shared_3d_support_solver_probabilities",
            "typed_posterior": "sugm_v3_shared_gaussian_posterior",
            "persistent_second_semantic_field": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
            "separate_identity_localization": True,
            "score_input": "exactly_one_d512_plus_five_scalars",
            "fixed_downstream_threshold": 0.6,
            "same_gaussian_posterior_for_2d_and_3d": True,
            "target_rgb_opened": False,
            "benchmark_labels_opened": False,
            "method_selection_from_benchmark": False,
            "scene_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
            "geometry_only_cache": {"path": str(geometry_path), "sha256": sha256_file(geometry_path)},
            "text_embeddings": {"path": str(text_path), "sha256": sha256_file(text_path)},
            "topk": args.topk, "scale": args.scale, "temperature": args.temperature,
            "membership_margin": args.membership_margin,
            "center_semantic_identity": args.center_semantic_identity,
            "identity_extent_weight": args.identity_extent_weight,
            "posterior_chunk_size": args.posterior_chunk_size,
            "identity_extent_authority": (
                {"path": str(Path(args.identity_extent_authority).resolve(strict=True)),
                 "sha256": sha256_file(Path(args.identity_extent_authority).resolve(strict=True))}
                if args.identity_extent_authority else None
            ),
            "membership_margin_authority": (
                {"path": str(Path(args.membership_margin_authority).resolve(strict=True)),
                 "sha256": sha256_file(Path(args.membership_margin_authority).resolve(strict=True))}
                if args.membership_margin_authority else None
            ),
            "extent_fraction": args.extent_fraction,
            "extent_fraction_authority": (
                "source_train_mean_proposal_gaussian_fraction"
                if args.extent_fraction else None
            ),
            "text_alignment": (
                {"path": str(Path(args.text_alignment).resolve(strict=True)),
                 "sha256": sha256_file(Path(args.text_alignment).resolve(strict=True))}
                if args.text_alignment else None
            ),
            "text_negatives": (
                {"path": str(Path(args.text_negatives).resolve(strict=True)),
                 "sha256": sha256_file(Path(args.text_negatives).resolve(strict=True)),
                 "logit_scale": args.text_logit_scale}
                if args.text_negatives else None
            ),
        },
    }
    torch.save(payload, output)
    print({"output": str(output), "sha256": sha256_file(output), "shape": list(payload["query_scores"].shape)})


if __name__ == "__main__":
    main()
