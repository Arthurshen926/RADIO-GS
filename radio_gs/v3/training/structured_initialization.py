"""Fresh source-only exact-MPR initialization for structured D512 memory."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch.nn import functional as F

from radio_gs.v3.memory.structured_memory import SharedPrivateLayout
from radio_gs.v3.training.instance_upper_bound import sha256_file
from radio_gs.v3.training.learned_source_codec import apply_codec


def fixed_jl_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    value = torch.randn(input_dim, output_dim, generator=generator)
    return value / float(output_dim) ** 0.5


@torch.no_grad()
def initialize_structured_memory(
    membership: Mapping[str, object],
    *,
    radio_teacher_root: str | Path,
    siglip_teacher_path: str | Path,
    layout: SharedPrivateLayout = SharedPrivateLayout(),
    seed: int = 20260826,
    hit_chunk: int = 32768,
    codec_path: str | Path | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build all D512 blocks without reading a historical Gaussian field."""

    rows = int(membership["num_rows"])
    metadata = membership["metadata"]
    records = [
        value for value in metadata["source_records"]
        if int(value["source_view_index"]) % 4 in (1, 2)
    ]
    if not records or hit_chunk <= 0:
        raise ValueError("structured initialization lacks source-train records")
    teacher_root = Path(radio_teacher_root).resolve(strict=True)
    siglip_path = Path(siglip_teacher_path).resolve(strict=True)

    first = torch.load(
        teacher_root / "backbone" / f"rgb_{int(records[0]['frame_id'])}.pt",
        map_location="cpu",
    ).float()
    codec = None
    if codec_path is not None:
        resolved_codec = Path(codec_path).resolve(strict=True)
        codec = torch.load(resolved_codec, map_location="cpu")
        codec_metadata = codec.get("metadata", {})
        if (
            codec.get("schema") != "radio_gs.sugm_v3.cross_scene_source_codec.v1"
            or codec_metadata.get("source_only") is not True
            or codec_metadata.get("historical_field_opened") is not False
            or codec_metadata.get("source_train_residues") != [1, 2]
        ):
            raise ValueError("learned codec violates fresh source-only lineage")
        codec_state = codec["state_dict"]
        radio_mean = torch.as_tensor(codec_state["radio_mean"]).float()
        radio_projection = torch.as_tensor(codec_state["radio_basis"]).float()
        if radio_projection.shape != (first.shape[0], layout.shared):
            raise ValueError("learned RADIO codec axes differ")
    else:
        resolved_codec = None
        radio_mean = torch.zeros(first.shape[0])
        radio_projection = fixed_jl_projection(first.shape[0], layout.shared, seed)
    shared_sum = torch.zeros(rows, layout.shared)
    shared_mass = torch.zeros(rows)
    for record in records:
        teacher = torch.load(
            teacher_root / "backbone" / f"rgb_{int(record['frame_id'])}.pt",
            map_location="cpu",
        ).float()
        pixels = teacher.permute(1, 2, 0).reshape(-1, teacher.shape[0])
        projected = apply_codec(pixels, radio_mean, radio_projection)
        if codec is None:
            projected = F.normalize(projected, dim=-1, eps=1e-8)
        shard = torch.load(Path(record["responsibility_view"]), map_location="cpu")
        gaussian_ids = torch.as_tensor(shard["gaussian_ids"]).long()
        pixel_ids = torch.as_tensor(shard["pixel_ids"]).long()
        weights = torch.as_tensor(shard["base_weights"]).float()
        for start in range(0, gaussian_ids.numel(), int(hit_chunk)):
            stop = min(start + int(hit_chunk), gaussian_ids.numel())
            ids = gaussian_ids[start:stop]
            mass = weights[start:stop]
            shared_sum.index_add_(
                0, ids, projected[pixel_ids[start:stop]] * mass[:, None]
            )
            shared_mass.index_add_(0, ids, mass)
    observed_shared = shared_mass > 0
    shared = shared_sum / shared_mass.clamp_min(1e-8)[:, None]
    shared[observed_shared] = F.normalize(shared[observed_shared], dim=-1, eps=1e-8)

    if layout.semantic:
        siglip = torch.load(siglip_path, map_location="cpu")
        descriptors = torch.as_tensor(siglip["descriptors"]).float()
        if descriptors.shape[0] != int(membership["num_proposals"]):
            raise ValueError("SigLIP and membership proposal axes differ")
        if codec is not None:
            semantic_mean = torch.as_tensor(codec_state["siglip_mean"]).float()
            semantic_projection = torch.as_tensor(codec_state["siglip_basis"]).float()
            if semantic_projection.shape != (descriptors.shape[1], layout.semantic):
                raise ValueError("learned SigLIP codec axes differ")
            proposal_semantic = apply_codec(
                descriptors, semantic_mean, semantic_projection
            )
        else:
            semantic_projection = fixed_jl_projection(
                descriptors.shape[1], layout.semantic, seed + 1
            )
            proposal_semantic = F.normalize(
                descriptors @ semantic_projection, dim=-1, eps=1e-8
            )
        proposal_views = torch.as_tensor(membership["proposal_view_indices"]).long()
        train_proposals = (proposal_views % 4 == 1) | (proposal_views % 4 == 2)
        membership_proposals = torch.as_tensor(membership["proposal_indices"]).long()
        selected = train_proposals[membership_proposals]
        membership_rows = torch.as_tensor(membership["row_indices"]).long()[selected]
        membership_proposals = membership_proposals[selected]
        membership_weights = torch.as_tensor(membership["weights"]).float()[selected]
        semantic_sum = torch.zeros(rows, layout.semantic)
        semantic_mass = torch.zeros(rows)
        semantic_sum.index_add_(
            0,
            membership_rows,
            proposal_semantic[membership_proposals] * membership_weights[:, None],
        )
        semantic_mass.index_add_(0, membership_rows, membership_weights)
        observed_semantic = semantic_mass > 0
        semantic = semantic_sum / semantic_mass.clamp_min(1e-8)[:, None]
        semantic[observed_semantic] = F.normalize(
            semantic[observed_semantic], dim=-1, eps=1e-8
        )
    else:
        semantic = torch.empty(rows, 0)
        observed_semantic = torch.zeros(rows, dtype=torch.bool)

    generator = torch.Generator(device="cpu").manual_seed(int(seed + 2))
    instance = torch.randn(rows, layout.instance, generator=generator)
    instance = instance / float(layout.instance) ** 0.5
    boundary = torch.zeros(rows, layout.boundary)
    memory = torch.cat((shared, semantic, instance, boundary), dim=-1)
    return memory, {
        "source_train_residues": [1, 2],
        "historical_field_opened": False,
        "radio_projection": (
            {"type": "learned_cross_scene_pca", "dim": layout.shared}
            if codec is not None else
            {"type": "fixed_jl", "seed": int(seed), "dim": layout.shared}
        ),
        "siglip_projection": (
            ({"type": "learned_cross_scene_pca", "dim": layout.semantic}
             if codec is not None else
             {"type": "fixed_jl", "seed": int(seed + 1), "dim": layout.semantic})
            if layout.semantic else None
        ),
        "shared_observed_rows": int(observed_shared.sum()),
        "semantic_observed_rows": int(observed_semantic.sum()),
        "num_rows": rows,
        "codec": (
            {"path": str(resolved_codec), "sha256": sha256_file(resolved_codec)}
            if resolved_codec is not None else None
        ),
    }


__all__ = ["fixed_jl_projection", "initialize_structured_memory"]
