#!/usr/bin/env python3
"""Build a depth-checked multiview RADIO teacher cache on Gaussian rows.

The cache is training-only: it opens extracted RADIO maps and camera poses,
but never benchmark masks or text queries.  Each Gaussian receives the mean
of the views in which its center is inside the camera, depth-consistent with
the rendered field, and supported by non-trivial rendered alpha.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from radio_gs.config import load_config
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
    CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    CANONICAL_OBSERVATION_CONTRACT_NAME,
    apply_canonical_observation_contract,
    observation_contract_sha256,
    select_full_observation_coverage_ranked_dataset_indices,
)
from radio_gs.models.radio_adaptors import (
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import (
    SimpleRadioDataset,
    sample_multiview_radio_targets,
)
from radio_gs.training.primitive_consensus import robust_multiview_consensus
from radio_gs.training.tensor_cache_io import validate_mpr_cache_payload
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    load_torch_payload,
    sha256_file,
)
from radio_gs.scripts.extract_radio_features import (
    _validate_final_output_bundle,
)


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return sha256_file(path)


def raster_fusion_reliability(
    features: torch.Tensor,
    valid: torch.Tensor,
    view_counts: torch.Tensor,
    *,
    num_views: int,
    mode: str,
    normalized_observations: bool,
) -> torch.Tensor:
    """Build query-free reliability for a raster-fused primitive target.

    For unit-normalized observations, the norm of their weighted mean is the
    mean resultant length: one for perfect directional agreement and lower as
    observations disagree. The legacy mode is retained for frozen contracts.
    """

    values = torch.as_tensor(features).float()
    active = torch.as_tensor(valid).bool().reshape(-1)
    counts = torch.as_tensor(view_counts).float().reshape(-1)
    if values.ndim != 2 or active.shape[0] != values.shape[0]:
        raise ValueError("raster features and validity must align by primitive")
    if counts.shape != active.shape or num_views <= 0:
        raise ValueError("raster view counts must align and num_views be positive")
    if mode not in {"legacy_valid", "mean_resultant"}:
        raise ValueError("unsupported raster reliability mode")
    coverage = (counts / float(num_views)).clamp(0.0, 1.0)
    if mode == "legacy_valid":
        agreement = active.float()
    else:
        if not normalized_observations:
            raise ValueError(
                "mean-resultant reliability requires normalized observations"
            )
        agreement = values.norm(dim=-1).clamp(0.0, 1.0)
        agreement = agreement.masked_fill(~active, 0.0)
    return torch.stack([coverage, agreement, active.float()], dim=-1).half()


# These names and dimensions are those emitted by the official C-RADIOv4-H
# runtime.  The direct path deliberately consumes the saved official adaptor
# output, rather than applying an MLP to an already interpolated raw map.
_EXTRACTED_CAPABILITY_SPECS: dict[str, dict[str, object]] = {
    "dino_v3": {
        "adaptor_name": "dino_v3_7b",
        "subdir": "dino_v3_7b",
        "output_dim": 4096,
    },
    "sam3": {
        "adaptor_name": "sam3",
        "subdir": "sam3",
        "output_dim": 1024,
    },
}


def _validated_feature_bundle(
    feature_dir: str | Path,
    *,
    expected_output_bundle_sha256: str = "",
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    """Reopen a complete feature bundle and index every declared tensor."""

    root = Path(feature_dir).expanduser().resolve()
    manifest, manifest_sha256, _source = load_json_object(
        root / "frame_manifest.json",
        label="RADIO feature frame manifest",
    )
    validation = _validate_final_output_bundle(
        root,
        manifest,
        expected_output_bundle_sha256=(
            str(expected_output_bundle_sha256) or None
        ),
        expected_manifest_sha256=manifest_sha256,
    )
    bundle = manifest.get("output_bundle")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("frames"), list):
        raise ValueError("RADIO feature output bundle has no frame records")
    records: dict[str, dict[str, object]] = {}
    for frame in bundle["frames"]:
        if not isinstance(frame, dict) or not isinstance(frame.get("tensors"), list):
            raise ValueError("RADIO feature output bundle frame is malformed")
        for record in frame["tensors"]:
            if not isinstance(record, dict):
                raise ValueError("RADIO feature tensor record is malformed")
            relative_path = str(record.get("relative_path", ""))
            if (
                not relative_path
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
                or relative_path in records
            ):
                raise ValueError("RADIO feature tensor path is unsafe or repeated")
            records[relative_path] = record
    return manifest, {**validation, "manifest_sha256": manifest_sha256}, records


def _load_bundle_feature_maps(
    *,
    feature_dir: str | Path,
    selected_frame_indices: list[int],
    subdir: str,
    expected_dim: int,
    feature_size: tuple[int, int],
    tensor_records: dict[str, dict[str, object]],
    normalize: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Consume selected feature maps through their bundle-bound descriptors."""

    root = Path(feature_dir).expanduser().resolve()
    maps: torch.Tensor | None = None
    for output_index, frame_index in enumerate(selected_frame_indices):
        relative_path = f"{subdir}/rgb_{int(frame_index)}.pt"
        record = tensor_records.get(relative_path)
        if record is None:
            raise ValueError(f"feature output bundle lacks {relative_path}")
        value, _digest, _source = load_torch_payload(
            root / relative_path,
            expected_sha256=str(record.get("sha256", "")),
            map_location="cpu",
            label=f"RADIO feature tensor {relative_path}",
        )
        if not torch.is_tensor(value):
            raise ValueError(f"{relative_path} is not a tensor")
        item = value.detach().cpu()
        if item.ndim == 4 and item.shape[0] == 1:
            item = item[0]
        if item.ndim != 3 or int(item.shape[0]) != int(expected_dim):
            raise ValueError(
                f"{relative_path} has unexpected shape {tuple(item.shape)}"
            )
        if item.dtype not in {torch.float16, torch.float32}:
            raise ValueError(f"{relative_path} has unsupported dtype {item.dtype}")
        if not bool(torch.isfinite(item).all()):
            raise ValueError(f"{relative_path} contains non-finite values")
        target_height, target_width = (int(value) for value in feature_size)
        if target_height > item.shape[1] or target_width > item.shape[2]:
            item = F.interpolate(
                item.float().unsqueeze(0),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        if normalize:
            item = F.normalize(item.float(), dim=0, eps=1e-8)
        item = item.to(dtype=output_dtype)
        if maps is None:
            maps = torch.empty(
                (len(selected_frame_indices), *item.shape),
                dtype=output_dtype,
            )
        elif item.shape != maps.shape[1:]:
            raise ValueError(
                "selected feature maps do not share one spatial shape"
            )
        maps[output_index].copy_(item)
    if maps is None:
        raise ValueError("feature map selection is empty")
    return maps


def validate_raster_reliability_policy(args: argparse.Namespace) -> None:
    """Resolve the observation contract before checking reliability options."""

    memory_fraction = float(
        getattr(args, "max_estimated_cpu_memory_fraction", 0.85)
    )
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(
            "--max-estimated-cpu-memory-fraction must lie in (0,1]"
        )
    if str(getattr(args, "observation_contract", "legacy")) in {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }:
        apply_canonical_observation_contract(args)
    if (
        args.raster_reliability_mode != "legacy_valid"
        and args.aggregation_mode == "center"
    ):
        raise ValueError(
            "--raster-reliability-mode applies only to raster aggregation"
        )
    if (
        args.raster_reliability_mode == "mean_resultant"
        and not args.normalize_each_view
    ):
        raise ValueError(
            "--raster-reliability-mode mean_resultant requires "
            "--normalize-each-view"
        )


def _resolve_extracted_capability_source(
    feature_dir: str | Path,
    feature_space: str,
    *,
    expected_radio_checkpoint_sha256: str = "",
    expected_scene: str = "",
    expected_image_dir: str | Path = "",
    expected_frame_indices: list[int] | None = None,
    expected_output_bundle_sha256: str = "",
    include_tensor_records: bool = False,
) -> dict[str, object]:
    """Verify an official C-RADIO adaptor map advertised by a frame manifest.

    A capability MPR may only consume this route when the same feature
    extraction run explicitly recorded the official adaptor output.  This is
    intentionally stricter than accepting a directory with compatible tensors:
    a forgotten ``--extract_adaptors`` flag must fail closed instead of
    silently reintroducing the legacy project-after-interpolation route.
    """

    space = str(feature_space).lower()
    if space not in _EXTRACTED_CAPABILITY_SPECS:
        raise ValueError(f"no extracted official capability source for {feature_space!r}")
    expected = dict(_EXTRACTED_CAPABILITY_SPECS[space])
    root = Path(feature_dir)
    manifest_path = root / "frame_manifest.json"
    manifest, bundle_validation, tensor_records = _validated_feature_bundle(
        root,
        expected_output_bundle_sha256=expected_output_bundle_sha256,
    )
    scene = str(manifest.get("scene", ""))
    image_dir = str(manifest.get("image_dir", ""))
    if str(expected_scene) and scene != str(expected_scene):
        raise ValueError(
            "official capability manifest belongs to a different scene"
        )
    if str(expected_image_dir):
        if not image_dir or Path(image_dir).resolve() != Path(
            expected_image_dir
        ).resolve():
            raise ValueError(
                "official capability manifest belongs to a different image "
                "directory"
            )
    frame_records = manifest.get("frames")
    if not isinstance(frame_records, list) or not frame_records:
        raise ValueError("feature manifest does not declare extracted frames")
    frame_indices: list[int] = []
    saved_stems: list[str] = []
    for frame in frame_records:
        if not isinstance(frame, dict):
            raise ValueError("feature manifest contains an invalid frame record")
        frame_index = int(frame.get("frame_idx", -1))
        saved_stem = str(frame.get("saved_stem", ""))
        if frame_index < 0 or saved_stem != f"rgb_{frame_index}":
            raise ValueError("feature manifest frame identity is invalid")
        frame_indices.append(frame_index)
        saved_stems.append(saved_stem)
    if (
        len(set(frame_indices)) != len(frame_indices)
        or len(set(saved_stems)) != len(saved_stems)
    ):
        raise ValueError("feature manifest contains duplicate frames")
    expected_frames = {
        int(value) for value in (expected_frame_indices or [])
    }
    if expected_frames and not expected_frames.issubset(frame_indices):
        raise ValueError(
            "official capability manifest lacks selected raw RADIO frames"
        )
    radio = manifest.get("radio")
    if not isinstance(radio, dict):
        raise ValueError("feature manifest does not declare its RADIO runtime")
    checkpoint_sha256 = str(radio.get("checkpoint_sha256", ""))
    checkpoint_provenance = str(radio.get("checkpoint_provenance", ""))
    checkpoint_load_contract = str(
        radio.get("checkpoint_load_contract", "")
    )
    expected_checkpoint = str(expected_radio_checkpoint_sha256 or "")
    if expected_checkpoint and (
        checkpoint_provenance != "explicit_file_sha256"
        or checkpoint_sha256 != expected_checkpoint
        or checkpoint_load_contract
        != "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
    ):
        raise ValueError(
            f"official {space} maps are not bound to the requested RADIO "
            "checkpoint SHA256"
        )
    features = manifest.get("features")
    adapters = features.get("adaptors") if isinstance(features, dict) else None
    if not isinstance(adapters, list):
        raise ValueError(
            f"feature manifest does not declare official adaptor maps for {space}"
        )
    matches = [
        value
        for value in adapters
        if isinstance(value, dict)
        and value.get("name") == expected["adaptor_name"]
        and value.get("subdir") == expected["subdir"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"feature manifest does not declare the matching official {space} adaptor"
        )
    record = matches[0]
    if int(record.get("dim", -1)) != int(expected["output_dim"]):
        raise ValueError(
            f"official {space} adaptor dimension differs from C-RADIO contract"
        )
    grid = record.get("grid")
    if (
        not isinstance(grid, list)
        or len(grid) != 2
        or any(int(value) <= 0 for value in grid)
    ):
        raise ValueError(f"official {space} adaptor manifest has an invalid spatial grid")
    subdir = root / str(expected["subdir"])
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"feature manifest declares {space}, but its map directory is missing: {subdir}"
        )
    expected_files = {
        f"{stem}.pt"
        for stem in saved_stems
    }
    actual_files = {
        path.name
        for path in subdir.iterdir()
        if path.is_file() and path.suffix == ".pt"
    }
    if actual_files != expected_files:
        raise ValueError(
            f"official {space} map files differ from the frame manifest"
        )
    frame_indices_sha256 = hashlib.sha256(
        json.dumps(
            frame_indices,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result: dict[str, object] = {
        **expected,
        "native_grid": [int(grid[0]), int(grid[1])],
        "frame_manifest": str(manifest_path.resolve()),
        "frame_manifest_sha256": str(bundle_validation["manifest_sha256"]),
        "output_bundle_sha256": str(
            bundle_validation["output_bundle_sha256"]
        ),
        "radio_version": str(radio.get("version", "")),
        "radio_checkpoint": str(radio.get("checkpoint", "")),
        "radio_checkpoint_sha256": checkpoint_sha256,
        "radio_checkpoint_provenance": checkpoint_provenance,
        "radio_checkpoint_load_contract": checkpoint_load_contract,
        "scene": scene,
        "image_dir": str(Path(image_dir).resolve()) if image_dir else "",
        "frame_indices": frame_indices,
        "frame_indices_sha256": frame_indices_sha256,
        "execution": "official_c_radio_runtime_adaptor_output",
    }
    if include_tensor_records:
        result["tensor_records"] = tensor_records
    return result


def _load_extracted_capability_maps(
    *,
    feature_dir: str | Path,
    feature_space: str,
    pose_file: str | None,
    pose_dir: str | None,
    feature_size: tuple[int, int],
    dataset_type: str,
    selected_frame_indices: list[int],
    expected_radio_checkpoint_sha256: str = "",
    expected_scene: str = "",
    expected_image_dir: str | Path = "",
    expected_output_bundle_sha256: str = "",
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load selected native official-adaptor maps then resample for registration.

    The frozen adaptor is evaluated by the official C-RADIO runtime at its
    native token locations.  Interpolation is applied only afterwards to match
    the fixed Gaussian raster grid, which preserves the intended ordering
    ``MPR(resample(A_official(f_v)))``.
    """

    source = _resolve_extracted_capability_source(
        feature_dir,
        feature_space,
        expected_radio_checkpoint_sha256=expected_radio_checkpoint_sha256,
        expected_scene=expected_scene,
        expected_image_dir=expected_image_dir,
        expected_frame_indices=selected_frame_indices,
        expected_output_bundle_sha256=expected_output_bundle_sha256,
        include_tensor_records=True,
    )
    maps = _load_bundle_feature_maps(
        feature_dir=feature_dir,
        selected_frame_indices=selected_frame_indices,
        subdir=str(source["subdir"]),
        expected_dim=int(source["output_dim"]),
        feature_size=feature_size,
        tensor_records=dict(source["tensor_records"]),
        normalize=True,
        output_dtype=torch.float16,
    )
    if maps.ndim != 4 or maps.shape[1] != int(source["output_dim"]):
        raise ValueError(
            f"official {feature_space} maps have unexpected shape {tuple(maps.shape)}"
        )
    source["in_memory_dtype"] = "float16"
    source["normalization"] = "per_pixel_float32_then_float16_storage"
    source.pop("tensor_records", None)
    return maps, source


@torch.no_grad()
def project_official_capability_maps(
    teacher_maps: torch.Tensor,
    adaptor: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Project complete 2-D RADIO grids before any 3-D aggregation.

    This order is intentional: for a nonlinear official adaptor,
    ``MPR(A(f_v))`` is not equivalent to ``A(MPR(f_v))``.  Returned maps are
    normalized official spatial features and contain no query information.
    """

    maps = torch.as_tensor(teacher_maps)
    if maps.ndim != 4 or maps.shape[1] != 1280:
        raise ValueError("teacher RADIO maps must be [V,1280,H,W]")
    if int(batch_size) <= 0:
        raise ValueError("projection batch size must be positive")
    parts: list[torch.Tensor] = []
    for start in tqdm(
        range(0, maps.shape[0], int(batch_size)),
        desc="project teacher capability space",
    ):
        projected = project_feature_map_with_adaptor(
            maps[start : start + int(batch_size)].to(device),
            adaptor,
            normalize=True,
        )
        parts.append(projected.half().cpu())
    return torch.cat(parts, dim=0)


def _select_indices(length: int, max_views: int) -> list[int]:
    if max_views <= 0 or max_views >= length:
        return list(range(length))
    positions = np.linspace(0, length - 1, num=max_views)
    return sorted({int(round(value)) for value in positions})


def _parse_frame_ids(raw: str) -> set[int]:
    value = str(raw or "").strip()
    if not value:
        return set()
    path = Path(value)
    try:
        is_dir = path.is_dir()
        is_file = path.is_file()
    except OSError:
        # Long comma-separated CLI lists are values, not filesystem paths.
        is_dir = False
        is_file = False
    if is_dir:
        return {
            int(item.stem.split("_")[-1])
            for item in path.glob("frame_*.json")
        }
    if is_file:
        value = path.read_text(encoding="utf-8")
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(line.split("#", 1)[0].replace(",", " ").split())
    return {int(item) for item in tokens}


def _select_candidate_indices(candidates: list[int], max_views: int) -> list[int]:
    chosen = _select_indices(len(candidates), max_views)
    return [candidates[index] for index in chosen]


def _select_full_observation_coverage_views(
    *,
    scene_root: str | Path,
    dataset_frame_ids: list[int],
    candidates: list[int],
    maximum_views: int,
    minimum_source_views: int = 0,
) -> tuple[list[int], dict[str, object]]:
    """Restore a full-.sens field's label-free greedy coverage order.

    A numeric-sorted RGB-D directory is deliberately convenient for normal
    loaders, but it must not erase the coverage ranking that created it.  The
    source manifest has no target/object/click information and is checked here
    before it can influence MPR frame selection.
    """

    manifest_path = Path(scene_root) / "field_source_contract.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "canonical full-observation MPR requires "
            f"{manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("full-observation field-source contract must be an object")
    allowed_versions = {
        "scannet_full_observation_v1",
        "scannet_full_observation_pfpr_queryheldout_v1",
    }
    if str(payload.get("field_contract_version", "")) not in allowed_versions:
        raise ValueError(
            "full-observation MPR source contract has an unsupported provenance version"
        )
    for key in (
        "uses_private_anchor",
        "uses_private_depth_pixel",
        "uses_instances_or_semantic_labels",
        "contains_instance_or_label_directories",
    ):
        if bool(payload.get(key, False)):
            raise ValueError(
                "full-observation MPR source contract is not query/label free: "
                f"{key}"
            )
    selected_frame_ids = [int(value) for value in payload.get("selected_frame_indices", [])]
    ranked_frame_ids = [int(value) for value in payload.get("selection_order_frame_indices", [])]
    if len(set(selected_frame_ids)) != len(selected_frame_ids) or set(
        selected_frame_ids
    ) != set(ranked_frame_ids):
        raise ValueError(
            "full-observation source contract has an incomplete coverage ranking"
        )
    minimum_source_views = int(minimum_source_views)
    if minimum_source_views > 0 and len(selected_frame_ids) < minimum_source_views:
        raise ValueError(
            "full-observation MPR contract requires an independently "
            f"materialized {minimum_source_views}-view source prefix"
        )
    selected = select_full_observation_coverage_ranked_dataset_indices(
        dataset_frame_ids=dataset_frame_ids,
        candidate_dataset_indices=candidates,
        ranked_frame_ids=ranked_frame_ids,
        maximum_views=int(maximum_views),
    )
    return selected, {
        "full_observation_source_contract": str(manifest_path.resolve()),
        "full_observation_source_contract_sha256": _sha256_file(manifest_path),
        "full_observation_source_contract_version": str(
            payload["field_contract_version"]
        ),
        "full_observation_coverage_order_applied": True,
        "full_observation_source_view_count": int(len(selected_frame_ids)),
    }


def merge_topk_view_observations(
    current_features: torch.Tensor,
    current_responsibility: torch.Tensor,
    observation: torch.Tensor,
    responsibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain whole view vectors ranked by compositing responsibility.

    Selecting top-k independently for each feature channel creates a synthetic
    descriptor whose channels come from different views.  The ranking scalar
    is therefore kept separately and its selected slot indexes are applied to
    every channel of the corresponding view observation.
    """
    if current_features.ndim != 3 or current_responsibility.ndim != 2:
        raise ValueError("current top-k state must be [N,D,K] and [N,K]")
    if current_features.shape[0] != current_responsibility.shape[0] or (
        current_features.shape[2] != current_responsibility.shape[1]
    ):
        raise ValueError("top-k feature and responsibility slots do not align")
    if observation.shape != current_features.shape[:2]:
        raise ValueError("new view observation must be [N,D]")
    if responsibility.shape != (current_features.shape[0],):
        raise ValueError("new view responsibility must be [N]")
    candidates = torch.cat(
        [current_responsibility, responsibility[:, None]], dim=1
    )
    selected_responsibility, selected_slots = torch.topk(
        candidates, k=current_responsibility.shape[1], dim=1
    )
    feature_candidates = torch.cat(
        [current_features, observation.unsqueeze(-1)], dim=-1
    )
    selected_features = torch.gather(
        feature_candidates,
        2,
        selected_slots[:, None, :].expand(-1, current_features.shape[1], -1),
    )
    return selected_features, selected_responsibility


def accumulate_contribution_mean_channel_chunked(
    feature_map: torch.Tensor,
    gaussian_ids: torch.Tensor,
    pixel_ids: torch.Tensor,
    weights: torch.Tensor,
    registered_sum: torch.Tensor,
    registered_counts: torch.Tensor,
    *,
    channel_chunk_size: int,
    cpu_sum_staging: torch.Tensor | None = None,
    cpu_count_staging: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate one view without materializing a dense ``N x C`` CUDA tensor.

    The operation is mathematically identical to full-channel weighted
    ``index_add``.  Only channel blocks are materialized on the GPU; the global
    contribution sum remains on CPU.  This is required for official 4096-D
    DINO targets on million-primitive scenes.
    """

    features = feature_map[0] if feature_map.ndim == 4 else feature_map
    if features.ndim != 3:
        raise ValueError("feature_map must be [C,H,W] or [1,C,H,W]")
    num_rows, channels = registered_sum.shape
    if registered_counts.shape != (num_rows,) or channels != features.shape[0]:
        raise ValueError("registered accumulation tensors do not align")
    if channel_chunk_size <= 0:
        raise ValueError("channel_chunk_size must be positive")
    maximum_chunk = min(int(channel_chunk_size), int(channels))
    if cpu_sum_staging is not None and (
        cpu_sum_staging.device.type != "cpu"
        or cpu_sum_staging.dtype != torch.float32
        or cpu_sum_staging.shape != (num_rows, maximum_chunk)
    ):
        raise ValueError("cpu_sum_staging must be float32 [N,min(C,chunk)] on CPU")
    if cpu_count_staging is not None and (
        cpu_count_staging.device.type != "cpu"
        or cpu_count_staging.dtype != torch.float32
        or cpu_count_staging.shape != (num_rows,)
    ):
        raise ValueError("cpu_count_staging must be float32 [N] on CPU")
    device = features.device
    gids = torch.as_tensor(gaussian_ids, device=device).long()
    pids = torch.as_tensor(pixel_ids, device=device).long()
    weight = torch.as_tensor(weights, device=device).float()
    height, width = features.shape[1:]
    valid = (
        (gids >= 0)
        & (gids < num_rows)
        & (pids >= 0)
        & (pids < height * width)
        & (weight > 0)
    )
    gids, pids, weight = gids[valid], pids[valid], weight[valid]
    frame_counts = torch.zeros(num_rows, dtype=torch.float32, device=device)
    if gids.numel():
        frame_counts.index_add_(0, gids, weight)
    if cpu_count_staging is None:
        counts_cpu = frame_counts.cpu()
    else:
        cpu_count_staging.copy_(frame_counts)
        counts_cpu = cpu_count_staging
    registered_counts.add_(counts_cpu)
    if not gids.numel():
        return counts_cpu
    for start in range(0, channels, int(channel_chunk_size)):
        stop = min(start + int(channel_chunk_size), channels)
        flat = features[start:stop].float().reshape(stop - start, height * width).t()
        sampled = flat[pids] * weight[:, None]
        sums = torch.zeros(
            num_rows, stop - start, dtype=torch.float32, device=device
        )
        sums.index_add_(0, gids, sampled)
        if cpu_sum_staging is None:
            sums_cpu = sums.cpu()
        else:
            sums_cpu = cpu_sum_staging[:, : stop - start]
            sums_cpu.copy_(sums)
        registered_sum[:, start:stop].add_(sums_cpu)
        del flat, sampled, sums
    return counts_cpu


def finalize_registered_mean_chunked(
    registered_sum: torch.Tensor,
    registered_counts: torch.Tensor,
    *,
    row_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a large CPU MPR without full-size advanced-index temporaries."""

    sums = torch.as_tensor(registered_sum)
    counts = torch.as_tensor(registered_counts)
    if (
        sums.device.type != "cpu"
        or sums.dtype != torch.float32
        or sums.ndim != 2
        or counts.device.type != "cpu"
        or counts.dtype != torch.float32
        or counts.shape != (sums.shape[0],)
    ):
        raise ValueError(
            "registered sums/counts must be aligned float32 CPU tensors"
        )
    if int(row_chunk_size) <= 0:
        raise ValueError("row_chunk_size must be positive")
    valid = counts > 0
    features = torch.zeros_like(sums, dtype=torch.float16)
    for start in range(0, sums.shape[0], int(row_chunk_size)):
        stop = min(start + int(row_chunk_size), sums.shape[0])
        chunk_valid = valid[start:stop]
        normalized = sums[start:stop] / counts[
            start:stop, None
        ].clamp_min(1e-8)
        normalized[~chunk_valid] = 0.0
        features[start:stop].copy_(normalized.half())
        del normalized
    return features, valid


def estimate_capability_mpr_cpu_bytes(
    *,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    num_gaussians: int,
    aggregation_mode: str,
    raster_view_fusion: str,
    raster_topk: int,
    raster_channel_chunk_size: int,
    responsibility_cache_bytes: int = 0,
) -> dict[str, int]:
    """Conservative overlapping CPU allocation estimate for a capability MPR."""

    values = (num_views, channels, height, width, num_gaussians)
    if any(int(value) <= 0 for value in values):
        raise ValueError("MPR memory dimensions must be positive")
    teacher_maps = (
        int(num_views)
        * int(channels)
        * int(height)
        * int(width)
        * 2
    )
    components = {
        "teacher_maps_float16": teacher_maps,
        "responsibility_cache_file_allowance": max(
            0, int(responsibility_cache_bytes)
        ),
    }
    if str(aggregation_mode) != "center":
        rows, dims = int(num_gaussians), int(channels)
        components["registered_sum_float32"] = rows * dims * 4
        components["final_features_float16"] = rows * dims * 2
        if str(raster_view_fusion) == "contribution_mean":
            components["channel_staging_float32"] = (
                rows
                * min(dims, int(raster_channel_chunk_size))
                * 4
            )
        elif str(raster_view_fusion) == "topk_mean":
            components["topk_features_float32"] = (
                rows * dims * max(1, int(raster_topk)) * 4
            )
    components["estimated_peak_bytes"] = sum(components.values())
    return components


def _available_cpu_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return 0


def _gaussian_state_sha256(model: torch.nn.Module) -> str:
    """Hash every primitive attribute that affects raster responsibility."""

    digest = hashlib.sha256()
    for name, values in (
        ("xyz", model.get_xyz()),
        ("rotation", model.get_rotation()),
        ("scaling", model.get_scaling()),
        ("opacity", model.get_opacity()),
    ):
        tensor = values.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _responsibility_contract(
    *,
    args: argparse.Namespace,
    selected: list[int],
    selected_frame_indices: list[int],
    poses: torch.Tensor,
    renderer,
    model: torch.nn.Module,
    feature_height: int,
    feature_width: int,
) -> dict:
    """Build the exact feature-independent registration contract."""

    return {
        "schema_version": 1,
        "assignment_mode": "raster_gaussian_top1",
        "registration_weight_mode": str(args.registration_weight_mode),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_dataset_indices": list(selected),
        "selected_frame_indices": list(selected_frame_indices),
        "excluded_frame_ids": sorted(_parse_frame_ids(args.exclude_frame_ids)),
        "feature_height": int(feature_height),
        "feature_width": int(feature_width),
        "depth_tolerance": float(args.depth_tolerance),
        "relative_depth_tolerance": float(args.relative_depth_tolerance),
        "alpha_threshold": float(args.alpha_threshold),
        "pose_sha256": _sha256_tensor_rows(poses),
        "intrinsics_sha256": _sha256_tensor_rows(
            renderer.scaled_intrinsics(feature_width, feature_height)
        ),
        "xyz_sha256": _sha256_tensor_rows(model.get_xyz()),
        "gaussian_state_sha256": _gaussian_state_sha256(model),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
    }


def _load_responsibility_cache(
    path: str | Path,
    *,
    expected_contract: dict,
    num_gaussians: int,
    expected_sha256: str = "",
) -> tuple[list[dict[str, torch.Tensor]], str]:
    """Load a shared registration sidecar and fail closed on any mismatch."""

    payload, observed_sha256, cache_path = load_torch_mapping(
        path,
        expected_sha256=str(expected_sha256) or None,
        map_location="cpu",
        label="registration responsibility cache",
    )
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("responsibility cache must use schema version 1")
    metadata = dict(payload.get("metadata", {}))
    # ``config`` and dataset-local row indices are feature-source aliases, not
    # registration identity.  A semantic directory may deliberately omit held
    # out images, shifting local indices.  Frame IDs plus pose/intrinsics and
    # complete Gaussian-state hashes are the fail-closed geometric identity.
    alias_fields = {"config", "selected_dataset_indices"}
    mismatched = [
        key
        for key, expected in expected_contract.items()
        if key not in alias_fields and metadata.get(key) != expected
    ]
    if mismatched:
        raise ValueError(
            f"responsibility cache contract differs: {sorted(mismatched)}"
        )
    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(
        expected_contract["selected_frame_indices"]
    ):
        raise ValueError("responsibility cache does not cover the selected views")
    checked: list[dict[str, torch.Tensor]] = []
    for view_index, item in enumerate(assignments):
        if not isinstance(item, dict):
            raise ValueError(f"responsibility view {view_index} is malformed")
        raw_gaussian_ids = item.get("gaussian_ids")
        raw_pixel_ids = item.get("pixel_ids")
        raw_weights = item.get("weights")
        if (
            not torch.is_tensor(raw_gaussian_ids)
            or raw_gaussian_ids.dtype
            not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            or not torch.is_tensor(raw_pixel_ids)
            or raw_pixel_ids.dtype
            not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            or not torch.is_tensor(raw_weights)
            or raw_weights.dtype not in {torch.float16, torch.float32, torch.float64}
        ):
            raise ValueError(
                f"responsibility view {view_index} has invalid tensor dtypes"
            )
        gaussian_ids = raw_gaussian_ids.long().cpu()
        pixel_ids = raw_pixel_ids.long().cpu()
        weights = raw_weights.float().cpu()
        if (
            gaussian_ids.ndim != 1
            or pixel_ids.shape != gaussian_ids.shape
            or weights.shape != gaussian_ids.shape
        ):
            raise ValueError(f"responsibility view {view_index} tensors do not align")
        if gaussian_ids.numel() and (
            int(gaussian_ids.min()) < 0
            or int(gaussian_ids.max()) >= int(num_gaussians)
        ):
            raise ValueError(f"responsibility view {view_index} has invalid Gaussian IDs")
        num_pixels = int(expected_contract["feature_height"]) * int(
            expected_contract["feature_width"]
        )
        if pixel_ids.numel() and (
            int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= num_pixels
        ):
            raise ValueError(f"responsibility view {view_index} has invalid pixel IDs")
        if pixel_ids.numel() > num_pixels or pixel_ids.unique().numel() != pixel_ids.numel():
            raise ValueError(
                f"responsibility view {view_index} repeats top-1 pixel IDs"
            )
        if (
            not bool(torch.isfinite(weights).all())
            or bool((weights <= 0).any())
            or bool((weights > 1.001).any())
        ):
            raise ValueError(f"responsibility view {view_index} has invalid weights")
        checked.append(
            {
                "gaussian_ids": gaussian_ids,
                "pixel_ids": pixel_ids,
                "weights": weights,
            }
        )
    return checked, observed_sha256


def build_cache(args: argparse.Namespace) -> dict:
    observation_contract = None
    if str(getattr(args, "observation_contract", "legacy")) in {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }:
        observation_contract = apply_canonical_observation_contract(args)
    expected_feature_bundle_sha256 = str(
        getattr(args, "expected_feature_output_bundle_sha256", "")
    )
    expected_geometry_sha256 = str(
        getattr(args, "expected_geometry_checkpoint_sha256", "")
    )
    if (
        observation_contract is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_feature_bundle_sha256)
        is None
    ):
        raise ValueError(
            "formal MPR requires --expected-feature-output-bundle-sha256"
        )
    if observation_contract is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_geometry_sha256
    ) is None:
        raise ValueError(
            "formal MPR requires --expected-geometry-checkpoint-sha256"
        )
    if expected_geometry_sha256 and (
        _sha256_file(args.checkpoint) != expected_geometry_sha256
    ):
        raise ValueError("geometry checkpoint differs from caller authority")
    if (
        observation_contract is not None
        and str(getattr(args, "responsibility_cache", "")).strip()
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(getattr(args, "expected_responsibility_cache_sha256", "")),
        )
        is None
    ):
        raise ValueError(
            "formal MPR responsibility reuse requires its expected SHA-256"
        )
    device = torch.device(args.device)
    config = load_config(args.config)
    model, _codec, renderer, _sharpener, _refiner, _cfg, _is_hybrid = (
        load_render_pipeline(
            args.config,
            args.checkpoint,
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
            expected_checkpoint_sha256=(
                expected_geometry_sha256 or None
            ),
        )
    )
    feature_height = int(getattr(config, "feature_height", renderer.image_height))
    feature_width = int(getattr(config, "feature_width", renderer.image_width))
    feature_dir = Path(str(getattr(config, "feature_dir", "")))
    (
        _feature_manifest,
        feature_bundle_validation,
        feature_tensor_records,
    ) = _validated_feature_bundle(
        feature_dir,
        expected_output_bundle_sha256=str(
            expected_feature_bundle_sha256
        ),
    )
    raw_pose_file = str(getattr(config, "pose_file", "") or "").strip()
    configured_pose_file = Path(raw_pose_file) if raw_pose_file else None
    pose_file = (
        str(configured_pose_file)
        if configured_pose_file is not None and configured_pose_file.is_file()
        else None
    )
    raw_pose_dir = str(getattr(config, "pose_dir", "") or "").strip()
    configured_pose_dir = Path(raw_pose_dir) if raw_pose_dir else None
    fallback_pose_dir = feature_dir / "poses_w2c"
    pose_dir = (
        str(configured_pose_dir)
        if configured_pose_dir is not None and configured_pose_dir.is_dir()
        else str(fallback_pose_dir) if fallback_pose_dir.is_dir() else None
    )
    included_frame_ids = _parse_frame_ids(
        str(getattr(args, "include_frame_ids", "") or "")
    )
    dataset = SimpleRadioDataset(
        feature_dir=str(feature_dir),
        pose_file=pose_file,
        pose_dir=pose_dir,
        feature_size=(feature_height, feature_width),
        split="train",
        dataset_type=str(getattr(config, "dataset_type", "lerf")),
        frame_ids=sorted(included_frame_ids) if included_frame_ids else None,
    )
    excluded_frame_ids = _parse_frame_ids(args.exclude_frame_ids)
    candidates = [
        index
        for index, frame_id in enumerate(dataset.frame_indices)
        if int(frame_id) not in excluded_frame_ids
    ]
    if not candidates:
        raise RuntimeError("all available feature frames were excluded")
    full_observation_source_metadata: dict[str, object] = {}
    if observation_contract is not None and bool(
        observation_contract["requires_full_observation_source_contract"]
    ):
        selected, full_observation_source_metadata = (
            _select_full_observation_coverage_views(
                scene_root=str(getattr(config, "scene_root", "")),
                dataset_frame_ids=[int(value) for value in dataset.frame_indices],
                candidates=candidates,
                maximum_views=int(args.max_views),
                minimum_source_views=int(
                    observation_contract.get("minimum_source_views", 0)
                ),
            )
        )
    else:
        selected = _select_candidate_indices(candidates, int(args.max_views))
    poses = torch.stack(
        [
            torch.from_numpy(dataset.poses_w2c[index]).float().cpu()
            for index in selected
        ],
        dim=0,
    )
    feature_space = str(args.feature_space).lower()
    capability_map_source = str(args.capability_map_source).lower()
    if capability_map_source not in {"project_raw", "official_extracted"}:
        raise ValueError(
            "capability_map_source must be project_raw or official_extracted"
        )
    selected_frame_indices = [
        int(dataset.frame_indices[index]) for index in selected
    ]
    summary_head_path = ""
    adaptor_name = ""
    adaptor_checkpoint_path = ""
    adaptor_checkpoint_sha256 = ""
    adaptor_checkpoint_provenance = ""
    capability_source_metadata: dict[str, object] = {
        "capability_map_source": "not_applicable",
        "capability_native_map_manifest": "",
        "capability_native_map_manifest_sha256": "",
        "capability_native_map_grid": [],
        "capability_native_map_scene": "",
        "capability_native_map_image_dir": "",
        "capability_native_map_frame_indices_sha256": "",
        "capability_native_map_output_bundle_sha256": "",
        "capability_native_map_radio_checkpoint_load_contract": "",
        "capability_adaptor_execution": "not_applicable",
    }
    capability_cpu_memory_preflight: dict[str, int | float] = {}
    if feature_space in {"dino_v3", "sam3"} and capability_map_source == "official_extracted":
        expected_radio_checkpoint_sha256 = _sha256_file(
            args.radio_checkpoint
        )
        responsibility_cache_bytes = 0
        if str(args.responsibility_cache).strip():
            responsibility_path = Path(
                args.responsibility_cache
            ).expanduser()
            if responsibility_path.is_file():
                responsibility_cache_bytes = (
                    2 * responsibility_path.stat().st_size
                )
        capability_cpu_memory_preflight = (
            estimate_capability_mpr_cpu_bytes(
                num_views=len(selected),
                channels=int(
                    _EXTRACTED_CAPABILITY_SPECS[feature_space][
                        "output_dim"
                    ]
                ),
                height=feature_height,
                width=feature_width,
                num_gaussians=int(model.get_xyz().shape[0]),
                aggregation_mode=str(args.aggregation_mode),
                raster_view_fusion=str(args.raster_view_fusion),
                raster_topk=int(args.raster_topk),
                raster_channel_chunk_size=int(
                    args.raster_channel_chunk_size
                ),
                responsibility_cache_bytes=(
                    responsibility_cache_bytes
                ),
            )
        )
        available_memory = _available_cpu_memory_bytes()
        capability_cpu_memory_preflight.update(
            {
                "available_memory_bytes": available_memory,
                "maximum_fraction": float(
                    getattr(
                        args,
                        "max_estimated_cpu_memory_fraction",
                        0.85,
                    )
                ),
            }
        )
        print(
            json.dumps(
                {
                    "capability_cpu_memory_preflight": (
                        capability_cpu_memory_preflight
                    )
                }
            ),
            flush=True,
        )
        if (
            available_memory > 0
            and capability_cpu_memory_preflight[
                "estimated_peak_bytes"
            ]
            > available_memory
            * float(
                getattr(
                    args,
                    "max_estimated_cpu_memory_fraction",
                    0.85,
                )
            )
        ):
            raise MemoryError(
                "estimated official capability MPR CPU peak exceeds "
                "the configured fraction of available memory"
            )
        teacher_maps, extracted_source = _load_extracted_capability_maps(
            feature_dir=feature_dir,
            feature_space=feature_space,
            pose_file=pose_file,
            pose_dir=pose_dir,
            feature_size=(feature_height, feature_width),
            dataset_type=str(getattr(config, "dataset_type", "lerf")),
            selected_frame_indices=selected_frame_indices,
            expected_radio_checkpoint_sha256=(
                expected_radio_checkpoint_sha256
            ),
            expected_scene=str(
                getattr(args, "expected_feature_scene", "")
            ),
            expected_image_dir=str(
                getattr(args, "expected_feature_image_dir", "")
            ),
            expected_output_bundle_sha256=str(
                feature_bundle_validation["output_bundle_sha256"]
            ),
        )
        adaptor_name = str(extracted_source["adaptor_name"])
        adaptor_checkpoint_path = str(extracted_source["radio_checkpoint"])
        adaptor_checkpoint_sha256 = str(
            extracted_source["radio_checkpoint_sha256"]
        )
        adaptor_checkpoint_provenance = str(
            extracted_source["radio_checkpoint_provenance"]
        )
        capability_source_metadata = {
            "capability_map_source": "official_extracted",
            "capability_native_map_manifest": str(
                extracted_source["frame_manifest"]
            ),
            "capability_native_map_manifest_sha256": str(
                extracted_source["frame_manifest_sha256"]
            ),
            "capability_native_map_grid": list(extracted_source["native_grid"]),
            "capability_native_map_scene": str(
                extracted_source["scene"]
            ),
            "capability_native_map_image_dir": str(
                extracted_source["image_dir"]
            ),
            "capability_native_map_frame_indices_sha256": str(
                extracted_source["frame_indices_sha256"]
            ),
            "capability_native_map_output_bundle_sha256": str(
                extracted_source["output_bundle_sha256"]
            ),
            "capability_native_map_radio_checkpoint_load_contract": str(
                extracted_source["radio_checkpoint_load_contract"]
            ),
            "capability_adaptor_execution": str(extracted_source["execution"]),
        }
    else:
        teacher_maps = _load_bundle_feature_maps(
            feature_dir=feature_dir,
            selected_frame_indices=selected_frame_indices,
            subdir="backbone",
            expected_dim=1280,
            feature_size=(feature_height, feature_width),
            tensor_records=feature_tensor_records,
            normalize=False,
            output_dtype=torch.float32,
        )
    if feature_space == "siglip_summary":
        summary_head_path = str(Path(args.summary_head_weights).expanduser().resolve())
        summary_head = SigLIP2SummaryHead.from_extracted_weights(summary_head_path).to(device)
        summary_head.eval()
        projected_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, teacher_maps.shape[0], int(args.projection_batch_size)),
                desc="project teacher query space",
            ):
                maps = teacher_maps[
                    start : start + int(args.projection_batch_size)
                ].to(device)
                batch, channels, height, width = maps.shape
                tokens = maps.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                projected = F.normalize(summary_head(tokens.float()), dim=-1)
                projected_parts.append(
                    projected.reshape(batch, height, width, -1)
                    .permute(0, 3, 1, 2)
                    .half()
                    .cpu()
                )
        teacher_maps = torch.cat(projected_parts, dim=0)
        del summary_head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif (
        feature_space in {"dino_v3", "sam3"}
        and capability_map_source != "official_extracted"
    ):
        adaptor_name = "dino_v3_7b" if feature_space == "dino_v3" else "sam3"
        adaptor_checkpoint_path = str(
            Path(args.radio_checkpoint).expanduser().resolve()
        )
        adaptor_checkpoint_sha256 = _sha256_file(adaptor_checkpoint_path)
        adaptor_checkpoint_provenance = "runtime_cli_checkpoint_sha256"
        adaptor = load_radio_adaptor_from_checkpoint(
            adaptor_checkpoint_path,
            adaptor_name,
            kind="feature_projection",
            expected_sha256=adaptor_checkpoint_sha256,
        ).to(device).eval()
        adaptor.requires_grad_(False)
        teacher_maps = project_official_capability_maps(
            teacher_maps,
            adaptor,
            device=device,
            batch_size=int(args.projection_batch_size),
        )
        del adaptor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        capability_source_metadata = {
            "capability_map_source": "project_raw",
            "capability_native_map_manifest": "",
            "capability_native_map_manifest_sha256": "",
            "capability_native_map_grid": [],
            "capability_native_map_scene": "",
            "capability_native_map_image_dir": "",
            "capability_native_map_frame_indices_sha256": "",
            "capability_native_map_output_bundle_sha256": "",
            "capability_native_map_radio_checkpoint_load_contract": "",
            "capability_adaptor_execution": "frozen_official_feature_projection_after_raw_resampling",
        }
    elif feature_space not in {"radio", "semantic_descriptor", "dino_v3", "sam3"}:
        raise ValueError(f"Unsupported feature space: {feature_space}")

    xyz_cpu = model.get_xyz().detach().float().cpu().contiguous()
    responsibility_assignments: list[dict[str, torch.Tensor]] | None = None
    responsibility_cache_path = ""
    responsibility_cache_sha256 = ""
    responsibility_contract: dict = {}
    if args.responsibility_cache or args.save_responsibility_cache:
        if args.aggregation_mode != "raster_gaussian_top1":
            raise ValueError(
                "shared responsibility caches require raster_gaussian_top1 aggregation"
            )
        if args.responsibility_cache and args.save_responsibility_cache:
            raise ValueError(
                "load and save responsibility cache options are mutually exclusive"
            )
        responsibility_contract = _responsibility_contract(
            args=args,
            selected=selected,
            selected_frame_indices=selected_frame_indices,
            poses=poses,
            renderer=renderer,
            model=model,
            feature_height=feature_height,
            feature_width=feature_width,
        )
        if args.responsibility_cache:
            responsibility_cache_path = str(
                Path(args.responsibility_cache).expanduser().resolve()
            )
            responsibility_assignments, responsibility_cache_sha256 = (
                _load_responsibility_cache(
                    responsibility_cache_path,
                    expected_contract=responsibility_contract,
                    num_gaussians=int(xyz_cpu.shape[0]),
                    expected_sha256=str(
                        getattr(
                            args,
                            "expected_responsibility_cache_sha256",
                            "",
                        )
                    ),
                )
            )

    # Visibility is already encoded in a loaded sidecar.  Otherwise render it
    # once, and optionally freeze the resulting registration assignments for
    # all feature spaces.
    depth_maps = None
    alpha_maps = None
    if responsibility_assignments is None:
        depth_parts: list[torch.Tensor] = []
        alpha_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(selected), int(args.render_batch_size)),
                desc="render visibility",
            ):
                stop = min(start + int(args.render_batch_size), len(selected))
                result = renderer.render_features_batch(
                    model,
                    poses[start:stop].to(device),
                    feature_height=feature_height,
                    feature_width=feature_width,
                )
                depth_parts.append(result["depth_map"].float().cpu())
                alpha_parts.append(result["alpha_map"].float().cpu())
        depth_maps = torch.cat(depth_parts, dim=0)
        alpha_maps = torch.cat(alpha_parts, dim=0)

    feature_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    count_parts: list[torch.Tensor] = []
    reliability_parts: list[torch.Tensor] = []
    view_chunk = max(1, int(args.view_chunk_size))
    point_chunk = max(1, int(args.point_chunk_size))
    if args.aggregation_mode == "center":
        if depth_maps is None or alpha_maps is None:
            raise RuntimeError("center aggregation requires rendered visibility maps")
        with torch.inference_mode():
            for point_start in tqdm(
                range(0, xyz_cpu.shape[0], point_chunk),
                desc="aggregate Gaussian teachers",
            ):
                point_stop = min(point_start + point_chunk, xyz_cpu.shape[0])
                points = xyz_cpu[point_start:point_stop].to(device)
                target_sum = torch.zeros(
                    points.shape[0], teacher_maps.shape[1], device=device, dtype=torch.float32
                )
                view_counts = torch.zeros(points.shape[0], device=device, dtype=torch.long)
                for view_start in range(0, len(selected), view_chunk):
                    view_stop = min(view_start + view_chunk, len(selected))
                    targets, _valid, counts = sample_multiview_radio_targets(
                        points,
                        teacher_maps[view_start:view_stop].to(device),
                        poses[view_start:view_stop].to(device),
                        renderer.scaled_intrinsics(feature_width, feature_height).float(),
                        depth_map=depth_maps[view_start:view_stop].to(device),
                        alpha_map=alpha_maps[view_start:view_stop].to(device),
                        depth_tolerance=float(args.depth_tolerance),
                        relative_depth_tolerance=float(args.relative_depth_tolerance),
                        alpha_threshold=float(args.alpha_threshold),
                        normalize_sampled_features=bool(args.normalize_each_view),
                    )
                    counts_f = counts.float().unsqueeze(1)
                    target_sum.add_(targets.float() * counts_f)
                    view_counts.add_(counts.long())
                valid = view_counts > 0
                averaged = target_sum / view_counts.clamp_min(1).float().unsqueeze(1)
                averaged[~valid] = 0.0
                if args.robust_mpr:
                    observations: list[torch.Tensor] = []
                    observation_validity: list[torch.Tensor] = []
                    for view_index in range(len(selected)):
                        view_target, view_valid, _view_count = sample_multiview_radio_targets(
                            points,
                            teacher_maps[view_index : view_index + 1].to(device),
                            poses[view_index : view_index + 1].to(device),
                            renderer.scaled_intrinsics(feature_width, feature_height).float(),
                            depth_map=depth_maps[view_index : view_index + 1].to(device),
                            alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                            depth_tolerance=float(args.depth_tolerance),
                            relative_depth_tolerance=float(args.relative_depth_tolerance),
                            alpha_threshold=float(args.alpha_threshold),
                            normalize_sampled_features=bool(args.normalize_each_view),
                        )
                        observations.append(view_target)
                        observation_validity.append(view_valid)
                    consensus = robust_multiview_consensus(
                        torch.stack(observations),
                        torch.stack(observation_validity),
                        robust_temperature=float(args.robust_temperature),
                        iterations=int(args.robust_iterations),
                        normalize_observations=bool(args.normalize_each_view),
                    )
                    averaged = consensus.targets
                    valid = consensus.valid
                    view_counts = consensus.observation_count
                    reliability = consensus.reliability
                else:
                    reliability = torch.stack(
                        [
                            view_counts.float() / max(1, len(selected)),
                            valid.float(),
                            valid.float(),
                        ],
                        dim=-1,
                    )
                feature_parts.append(averaged.half().cpu())
                valid_parts.append(valid.cpu())
                count_parts.append(view_counts.cpu())
                reliability_parts.append(reliability.half().cpu())
        del teacher_maps
        features = torch.cat(feature_parts, dim=0)
        valid = torch.cat(valid_parts, dim=0)
        view_counts = torch.cat(count_parts, dim=0)
        reliability = torch.cat(reliability_parts, dim=0)
    else:
        from radio_gs.scripts.eval_lerf_direct_3d_selection import (
            accumulate_raster_contribution_features,
            raster_adjoint_registered_view_features,
            rasterize_registered_view_assignments,
        )

        registered_sum = torch.zeros(
            xyz_cpu.shape[0], teacher_maps.shape[1], dtype=torch.float32
        )
        registered_counts = torch.zeros(xyz_cpu.shape[0], dtype=torch.float32)
        contribution_sum_staging = None
        contribution_count_staging = None
        if args.raster_view_fusion == "contribution_mean":
            contribution_sum_staging = torch.empty(
                xyz_cpu.shape[0],
                min(
                    int(args.raster_channel_chunk_size),
                    int(teacher_maps.shape[1]),
                ),
                dtype=torch.float32,
            )
            contribution_count_staging = torch.empty(
                xyz_cpu.shape[0], dtype=torch.float32
            )
        observation_counts = torch.zeros(xyz_cpu.shape[0], dtype=torch.long)
        topk_observations = None
        topk_responsibility = None
        if args.raster_view_fusion == "topk_mean":
            topk_observations = torch.zeros(
                (
                    xyz_cpu.shape[0],
                    teacher_maps.shape[1],
                    max(1, int(args.raster_topk)),
                ),
                dtype=torch.float32,
            )
            topk_responsibility = torch.full(
                (xyz_cpu.shape[0], max(1, int(args.raster_topk))),
                -float("inf"),
                dtype=torch.float32,
            )
        aggregation_context = (
            contextlib.nullcontext()
            if args.aggregation_mode == "raster_adjoint"
            else torch.inference_mode()
        )
        captured_assignments: list[dict[str, torch.Tensor]] = []
        with aggregation_context:
            for view_index in tqdm(
                range(len(selected)), desc="aggregate raster contributions"
            ):
                if args.aggregation_mode == "raster_adjoint":
                    if alpha_maps is None:
                        raise RuntimeError(
                            "raster adjoint aggregation requires rendered alpha maps"
                        )
                    frame_sum, frame_counts = raster_adjoint_registered_view_features(
                        model=model,
                        renderer=renderer,
                        viewmat=poses[view_index].to(device),
                        siglip_feat=teacher_maps[view_index : view_index + 1].to(
                            device=device, dtype=torch.float32
                        ),
                        alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                        alpha_threshold=float(args.alpha_threshold),
                        channel_chunk_size=int(args.adjoint_channel_chunk_size),
                    )
                else:
                    if responsibility_assignments is not None:
                        assignment = responsibility_assignments[view_index]
                        gaussian_ids = assignment["gaussian_ids"].to(device)
                        pixel_ids = assignment["pixel_ids"].to(device)
                        weights = assignment["weights"].to(device)
                    else:
                        if depth_maps is None or alpha_maps is None:
                            raise RuntimeError(
                                "raster registration requires visibility maps or a sidecar"
                            )
                        gaussian_ids, pixel_ids, weights = (
                            rasterize_registered_view_assignments(
                                model=model,
                                renderer=renderer,
                                viewmat=poses[view_index].to(device),
                                image_height=feature_height,
                                image_width=feature_width,
                                depth_map=depth_maps[
                                    view_index : view_index + 1
                                ].to(device),
                                alpha_map=alpha_maps[
                                    view_index : view_index + 1
                                ].to(device),
                                registration_depth_tolerance=float(
                                    args.depth_tolerance
                                ),
                                registration_relative_depth_tolerance=float(
                                    args.relative_depth_tolerance
                                ),
                                registration_alpha_threshold=float(
                                    args.alpha_threshold
                                ),
                                registration_weight_mode=args.registration_weight_mode,
                                gaussian_top1=True,
                            )
                        )
                        if args.save_responsibility_cache:
                            captured_assignments.append(
                                {
                                    "gaussian_ids": gaussian_ids.int().cpu(),
                                    "pixel_ids": pixel_ids.int().cpu(),
                                    "weights": weights.float().cpu(),
                                }
                            )
                    view_features = teacher_maps[
                        view_index : view_index + 1
                    ].to(device=device, dtype=torch.float32)
                    if bool(args.normalize_each_view):
                        view_features = F.normalize(view_features, dim=1, eps=1e-8)
                    if args.raster_view_fusion == "contribution_mean":
                        counts_cpu = accumulate_contribution_mean_channel_chunked(
                            view_features,
                            gaussian_ids,
                            pixel_ids,
                            weights,
                            registered_sum,
                            registered_counts,
                            channel_chunk_size=int(args.raster_channel_chunk_size),
                            cpu_sum_staging=contribution_sum_staging,
                            cpu_count_staging=contribution_count_staging,
                        )
                        frame_valid = counts_cpu > 0
                        observation_counts[frame_valid] += 1
                        del view_features, counts_cpu
                        continue
                    frame_sum, frame_counts = accumulate_raster_contribution_features(
                        view_features,
                        gaussian_ids,
                        pixel_ids,
                        weights,
                        n_gaussians=int(xyz_cpu.shape[0]),
                    )
                counts_cpu = frame_counts.float().cpu()
                frame_valid = counts_cpu > 0
                if bool(frame_valid.any()):
                    frame_sum_cpu = frame_sum.float().cpu()[frame_valid]
                    if args.raster_view_fusion == "contribution_mean":
                        registered_sum[frame_valid] += frame_sum_cpu
                        registered_counts[frame_valid] += counts_cpu[frame_valid]
                    else:
                        frame_observation = frame_sum_cpu / counts_cpu[
                            frame_valid, None
                        ].clamp_min(1e-8)
                        if args.raster_view_fusion == "view_mean":
                            registered_sum[frame_valid] += frame_observation
                            registered_counts[frame_valid] += 1.0
                        elif args.raster_view_fusion == "topk_mean":
                            assert topk_observations is not None
                            assert topk_responsibility is not None
                            selected_features, selected_responsibility = (
                                merge_topk_view_observations(
                                    topk_observations[frame_valid],
                                    topk_responsibility[frame_valid],
                                    frame_observation,
                                    counts_cpu[frame_valid],
                                )
                            )
                            topk_observations[frame_valid] = selected_features
                            topk_responsibility[frame_valid] = selected_responsibility
                        else:
                            raise ValueError(
                                f"unsupported raster view fusion: {args.raster_view_fusion}"
                            )
                    observation_counts[frame_valid] += 1
                del frame_sum, frame_counts
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del teacher_maps
        if args.save_responsibility_cache:
            if len(captured_assignments) != len(selected):
                raise RuntimeError("failed to capture every registration view")
            responsibility_output = Path(
                args.save_responsibility_cache
            ).expanduser()
            responsibility_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "metadata": responsibility_contract,
                    "assignments": captured_assignments,
                },
                responsibility_output,
            )
            responsibility_cache_path = str(responsibility_output.resolve())
            responsibility_cache_sha256 = _sha256_file(responsibility_output)
        if args.raster_view_fusion == "topk_mean":
            assert topk_observations is not None
            assert topk_responsibility is not None
            finite = torch.isfinite(topk_responsibility)
            valid = finite.any(dim=1)
            features = (
                (topk_observations * finite[:, None, :]).sum(dim=-1)
                / finite.sum(dim=-1, keepdim=True).clamp_min(1)
            ).half()
            features[~valid] = 0.0
            del topk_observations, topk_responsibility, finite
        else:
            features, valid = finalize_registered_mean_chunked(
                registered_sum,
                registered_counts,
                row_chunk_size=int(args.point_chunk_size),
            )
        del registered_sum, registered_counts
        del contribution_sum_staging, contribution_count_staging
        view_counts = observation_counts
        reliability = raster_fusion_reliability(
            features,
            valid,
            view_counts,
            num_views=max(1, len(selected)),
            mode=str(getattr(args, "raster_reliability_mode", "legacy_valid")),
            normalized_observations=bool(args.normalize_each_view),
        )
    metadata = {
        "schema_version": 1,
        "feature_space": feature_space,
        "construction": (
            f"{feature_space}_{args.aggregation_mode}_robust_mpr"
            if args.robust_mpr and args.aggregation_mode == "center"
            else (
                f"{feature_space}_{args.aggregation_mode}_{args.raster_view_fusion}"
                if args.aggregation_mode != "center"
                else f"{feature_space}_{args.aggregation_mode}_multiview_mean"
            )
        ),
        "aggregation_mode": args.aggregation_mode,
        "registration_weight_mode": args.registration_weight_mode,
        "raster_view_fusion": args.raster_view_fusion,
        "raster_reliability_mode": str(
            getattr(args, "raster_reliability_mode", "legacy_valid")
        ),
        "raster_topk": max(1, int(args.raster_topk)),
        "raster_topk_ranking": (
            "whole_view_compositing_responsibility"
            if args.raster_view_fusion == "topk_mean"
            else "not_applicable"
        ),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_declared_views": len(selected),
        "selected_dataset_indices": selected,
        "selected_frame_indices": selected_frame_indices,
        "excluded_frame_ids": sorted(excluded_frame_ids),
        **full_observation_source_metadata,
        "depth_tolerance": float(args.depth_tolerance),
        "relative_depth_tolerance": float(args.relative_depth_tolerance),
        "alpha_threshold": float(args.alpha_threshold),
        "normalize_each_view": bool(args.normalize_each_view),
        "feature_frame_manifest": str(
            (feature_dir / "frame_manifest.json").resolve()
        ),
        "feature_frame_manifest_sha256": str(
            feature_bundle_validation["manifest_sha256"]
        ),
        "feature_output_bundle_sha256": str(
            feature_bundle_validation["output_bundle_sha256"]
        ),
        "per_view_normalization_applied": bool(args.normalize_each_view),
        "per_view_normalization_stage": (
            "pixel_feature_before_raster_lifting"
            if args.aggregation_mode != "center" and args.normalize_each_view
            else "sampled_feature_before_center_fusion"
            if args.normalize_each_view
            else "disabled"
        ),
        "robust_mpr": bool(args.robust_mpr and args.aggregation_mode == "center"),
        "robust_temperature": float(args.robust_temperature),
        "robust_iterations": int(args.robust_iterations),
        "summary_head_weights": summary_head_path,
        "official_adaptor_name": adaptor_name,
        "official_adaptor_checkpoint": adaptor_checkpoint_path,
        "official_adaptor_checkpoint_sha256": adaptor_checkpoint_sha256,
        "official_adaptor_checkpoint_provenance": (
            adaptor_checkpoint_provenance
        ),
        **capability_source_metadata,
        "registration_responsibility_cache": responsibility_cache_path,
        "registration_responsibility_cache_sha256": responsibility_cache_sha256,
        "shared_registration_responsibility": bool(
            responsibility_cache_sha256
        ),
        "registration_responsibility_contract": responsibility_contract,
        "capability_projection_before_mpr": feature_space
        in {"dino_v3", "sam3"},
        "capability_cpu_memory_preflight": (
            capability_cpu_memory_preflight
        ),
        "custom_adaptor_head": False,
        "source": (
            "official_crop_summary_mpr"
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else ""
        ),
        "official_summary_head": (
            True
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else None
        ),
        "custom_text_projection": (
            False
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else None
        ),
        "semantic_alignment_level": (
            2
            if feature_space == "semantic_descriptor"
            and args.semantic_descriptor_source == "official_siglip2_crop_summary"
            else None
        ),
        "query_names": [
            value.strip() for value in str(args.query_names).split(",") if value.strip()
        ],
        "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
        "benchmark_masks_opened": False,
        "benchmark_images_opened": bool(args.benchmark_images_opened),
        "text_queries_opened": bool(args.text_queries_opened),
    }
    if observation_contract is not None:
        metadata["observation_lifting_contract"] = observation_contract
        metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(
            observation_contract
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "xyz": xyz_cpu,
        "geometry_fingerprint": {
            "num_gaussians": int(xyz_cpu.shape[0]),
            "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
        },
        "features": features,
        "valid": valid,
        "view_counts": view_counts,
        "reliability": reliability,
        "metadata": metadata,
    }
    validate_mpr_cache_payload(
        output_payload,
        expected_feature_space=feature_space,
        require_reliability=True,
        require_formal_safety=observation_contract is not None,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(output_payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    positive = view_counts[valid]
    report = {
        "output": str(output),
        "num_gaussians": int(xyz_cpu.shape[0]),
        "num_views": len(selected),
        "valid_count": int(valid.sum()),
        "valid_ratio": float(valid.float().mean()),
        "mean_views_if_valid": float(positive.float().mean()) if positive.numel() else 0.0,
        "median_views_if_valid": float(positive.float().median()) if positive.numel() else 0.0,
        "max_views": int(positive.max()) if positive.numel() else 0,
        "metadata": metadata,
    }
    report_path = output.with_suffix(output.suffix + ".json")
    temporary_report = report_path.with_suffix(
        report_path.suffix + ".tmp"
    )
    temporary_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--observation-contract",
        choices=[
            "legacy",
            CANONICAL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
        ],
        default=CANONICAL_OBSERVATION_CONTRACT_NAME,
        help=(
            "Versioned query-free lifting policy. canonical-full-observation "
            "variants restore the field-source coverage ranking; v1 is the "
            "frozen 240-view control; v2/v3 preserve independent 480/960-view "
            "source prefixes."
        ),
    )
    parser.add_argument("--max-views", type=int, default=32)
    parser.add_argument(
        "--exclude-frame-ids",
        default="",
        help="Comma list, text file, or LERF label directory of held-out frame IDs.",
    )
    parser.add_argument(
        "--include-frame-ids",
        default="",
        help=(
            "Optional comma list/file restricting observations before pose loading; "
            "use for RGB frames without registered cameras."
        ),
    )
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--view-chunk-size", type=int, default=8)
    parser.add_argument("--point-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--max-estimated-cpu-memory-fraction",
        type=float,
        default=0.85,
        help=(
            "Fail before loading official capability maps when the known "
            "overlapping CPU allocations exceed this fraction of currently "
            "available memory."
        ),
    )
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--relative-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-threshold", type=float, default=0.02)
    parser.add_argument("--normalize-each-view", action="store_true")
    parser.add_argument(
        "--robust-mpr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Robustly fuse per-view primitive observations (center aggregation).",
    )
    parser.add_argument("--robust-temperature", type=float, default=0.10)
    parser.add_argument("--robust-iterations", type=int, default=2)
    parser.add_argument(
        "--feature-space",
        choices=[
            "radio",
            "dino_v3",
            "sam3",
            "siglip_summary",
            "semantic_descriptor",
        ],
        default="radio",
        help=(
            "Aggregate raw RADIO, precomputed semantic descriptors, or first "
            "project every view through a frozen SigLIP2 pointwise head."
        ),
    )
    parser.add_argument(
        "--semantic-descriptor-source",
        choices=["unspecified", "official_siglip2_crop_summary"],
        default="unspecified",
        help="Auditable provenance for precomputed semantic_descriptor maps.",
    )
    parser.add_argument(
        "--summary-head-weights",
        default="checkpoints/siglip2_summary_head.pth",
    )
    parser.add_argument(
        "--radio-checkpoint",
        default="/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
        help="Official C-RADIO checkpoint used only for frozen capability projections.",
    )
    parser.add_argument(
        "--capability-map-source",
        choices=["project_raw", "official_extracted"],
        default="project_raw",
        help=(
            "For DINO/SAM MPR, either apply the frozen feature projection to "
            "the raw map (legacy) or consume the matching native official "
            "C-RADIO adaptor maps emitted by extract_radio_features."
        ),
    )
    parser.add_argument(
        "--expected-feature-scene",
        default="",
        help="Fail if an official extracted manifest names another scene.",
    )
    parser.add_argument(
        "--expected-feature-image-dir",
        default="",
        help=(
            "Fail if an official extracted manifest was built from another "
            "resolved image directory."
        ),
    )
    parser.add_argument(
        "--expected-geometry-checkpoint-sha256",
        default="",
        help=(
            "Externally trusted geometry checkpoint digest required by "
            "formal MPR contracts."
        ),
    )
    parser.add_argument(
        "--expected-feature-output-bundle-sha256",
        default="",
        help=(
            "Caller-trusted SHA-256 of the complete extracted feature output "
            "bundle. Formal runs must provide this value."
        ),
    )
    parser.add_argument("--projection-batch-size", type=int, default=2)
    parser.add_argument(
        "--responsibility-cache",
        default="",
        help=(
            "Feature-independent raster_gaussian_top1 assignment sidecar. "
            "Using the same sidecar makes raw and capability MPR observation "
            "support exactly identical."
        ),
    )
    parser.add_argument(
        "--expected-responsibility-cache-sha256",
        default="",
        help="Caller-trusted SHA-256 for a loaded responsibility sidecar.",
    )
    parser.add_argument(
        "--save-responsibility-cache",
        default="",
        help=(
            "Save the query-free pixel-to-Gaussian assignment generated by "
            "this run for exact reuse by other feature spaces."
        ),
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=["center", "raster_gaussian_top1", "raster_adjoint"],
        default="center",
    )
    parser.add_argument(
        "--registration-weight-mode",
        choices=["uniform", "alpha", "alpha_depth"],
        default="alpha_depth",
    )
    parser.add_argument("--adjoint-channel-chunk-size", type=int, default=32)
    parser.add_argument(
        "--raster-channel-chunk-size",
        type=int,
        default=256,
        help="Exact channel chunking for contribution-mean raster accumulation.",
    )
    parser.add_argument(
        "--raster-view-fusion",
        choices=["contribution_mean", "view_mean", "topk_mean"],
        default="contribution_mean",
        help="Across-view fusion after each raster registration observation.",
    )
    parser.add_argument(
        "--raster-reliability-mode",
        choices=["legacy_valid", "mean_resultant"],
        default="legacy_valid",
        help=(
            "Keep frozen [coverage,valid,valid] reliability or record the "
            "mean-resultant directional agreement of normalized observations."
        ),
    )
    parser.add_argument(
        "--raster-topk",
        type=int,
        default=3,
        help="Number of strongest view observations retained by topk_mean.",
    )
    parser.add_argument("--query-names", default="")
    parser.add_argument("--text-queries-opened", action="store_true")
    parser.add_argument(
        "--benchmark-images-opened",
        action="store_true",
        help="Mark diagnostic caches built from held-out benchmark RGB/features.",
    )
    args = parser.parse_args()
    try:
        validate_raster_reliability_policy(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(build_cache(args), indent=2))


if __name__ == "__main__":
    main()
