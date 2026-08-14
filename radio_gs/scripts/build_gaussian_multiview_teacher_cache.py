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
import fcntl
import hashlib
import inspect
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
from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
    CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    canonical_factorized_radio_contract,
)
from radio_gs.models.siglip_projection import SigLIP2SummaryHead
from radio_gs.field.observation_lifting_contract import (
    CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
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
from radio_gs.rendering.contribution_compositor import (
    EXACT_CENTER_UNCERTAINTY_CONTRACT,
    MARGINAL_RESPONSIBILITY_CONTRACT,
    marginal_responsibility_statistics,
    rasterize_single_view_contributions,
)
from radio_gs.rendering.sparse_marginal_authority import (
    SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
    SparseExactMarginalAuthorityWriter,
    load_sparse_exact_marginal_authority,
    sparse_exact_marginal_implementation_sha256,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.training.feature_training_utils import (
    SimpleRadioDataset,
    sample_multiview_radio_targets,
)
from radio_gs.training.primitive_consensus import robust_multiview_consensus
from radio_gs.training.tensor_cache_io import load_mpr_cache, validate_mpr_cache_payload
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    load_torch_mapping,
    load_torch_payload,
    sha256_file,
)
from radio_gs.scripts.extract_radio_features import (
    _validate_final_output_bundle,
)


CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA = (
    "radio_gs.canonical_factorized_radio_builder_cache.v1"
)
CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2 = (
    "radio_gs.canonical_factorized_radio_builder_cache.v2"
)
FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY = {
    "authority": "raster_gaussian_top1_sidecar_v1_missing_visible_mass",
    "measurement_available": False,
    "encoding": "exact_zero_unknown_sentinel",
    "consumer_policy": "must_not_treat_visibility_purity_as_a_measurement",
}
FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY = {
    "authority": "sparse_exact_marginal_responsibility_authority_v1",
    "measurement_available": True,
    "encoding": "positive_marginal_mass_over_exact_visible_mass",
    "consumer_policy": "measured_visibility_purity_may_weight_confidence_only",
    "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
}


def canonical_factorized_radio_builder_contract() -> dict[str, object]:
    """Return the immutable source-only lifting policy for the builder envelope."""

    return {
        "name": "canonical-factorized-radio-v1-builder",
        "schema_version": 1,
        "parent_contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "maximum_views": 120,
        "view_selection": "uniform_temporal_deterministic",
        "aggregation_mode": "raster_gaussian_top1",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": "alpha_depth",
        "normalize_each_view": False,
        "robust_mpr": False,
        "observation_unit": "positive_norm_raw_radio_pixel",
        "visibility_purity": "exact_zero_unknown_top1_sidecar_sentinel",
        "query_independent": True,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
    }


def factorized_radio_builder_contract_sha256() -> str:
    encoded = json.dumps(
        canonical_factorized_radio_builder_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_factorized_radio_builder_contract_v2() -> dict[str, object]:
    """Visibility-safe exact-marginal builder; core representation stays v1."""

    return {
        "name": "canonical-factorized-radio-v1-builder-exact-marginal-v2",
        "schema_version": 2,
        "parent_contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "maximum_views": 120,
        "view_selection": "uniform_temporal_deterministic",
        "aggregation_mode": "raster_marginal_responsibility",
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": (
            "exact_front_to_back_marginal_responsibility"
        ),
        "normalize_each_view": False,
        "robust_mpr": False,
        "observation_unit": "positive_norm_raw_radio_pixel",
        "semantic_weight": "exact_base_weight_times_pixel_marginal",
        "visibility_weight": "exact_base_weight",
        "visibility_purity": (
            "positive_amplitude_marginal_mass_over_exact_visible_mass"
        ),
        "sparse_authority_formula_sha256": (
            SPARSE_EXACT_MARGINAL_FORMULA_SHA256
        ),
        "query_independent": True,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
    }


def factorized_radio_builder_contract_v2_sha256() -> str:
    encoded = json.dumps(
        canonical_factorized_radio_builder_contract_v2(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_tensor_rows(values: torch.Tensor) -> str:
    array = values.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_int64_vector(values: torch.Tensor) -> str:
    array = values.detach().long().cpu().contiguous().numpy().astype("<i8", copy=False)
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


def prepare_raster_view_features(
    feature_map: torch.Tensor, *, normalize_each_view: bool
) -> torch.Tensor:
    """Apply the declared pixel-feature normalization before every lift.

    Keeping this outside individual raster operators prevents adjoint and
    explicit-hit paths from claiming the same metadata while consuming
    different teacher-map semantics.
    """

    values = torch.as_tensor(feature_map).float()
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("one raster view feature map must be [1,C,H,W]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("raster view feature map contains NaN or infinity")
    return F.normalize(values, dim=1, eps=1e-8) if bool(normalize_each_view) else values


def validate_factorized_radio_builder_policy(args: argparse.Namespace) -> bool:
    """Fail closed on every non-frozen factorized-builder option.

    ``canonical-factorized-radio-v1`` is intentionally a separate builder
    route.  Builder v1 consumes a caller-bound top-1 sidecar. Builder v2
    consumes or atomically creates an exact sparse marginal authority. Neither
    route permits legacy normalization, capability projection, query input, or
    target-bearing options.
    """

    if str(getattr(args, "observation_contract", "")) != (
        CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME
    ):
        return False
    aggregation_mode = str(getattr(args, "aggregation_mode", ""))
    if aggregation_mode not in {
        "raster_gaussian_top1",
        "raster_marginal_responsibility",
    }:
        raise ValueError(
            "canonical-factorized-radio-v1 requires frozen options: "
            "raster_gaussian_top1 or raster_marginal_responsibility"
        )
    required_equal = {
        "feature_space": "radio",
        "raster_view_fusion": "contribution_mean",
        "capability_map_source": "project_raw",
        "capability_storage": "dense",
    }
    mismatched = [
        name
        for name, expected in required_equal.items()
        if str(getattr(args, name, "")) != expected
    ]
    if mismatched:
        raise ValueError(
            "canonical-factorized-radio-v1 requires frozen options: "
            f"{sorted(mismatched)}"
        )
    if int(getattr(args, "max_views", -1)) != 120:
        raise ValueError("canonical-factorized-radio-v1 requires --max-views 120")
    if bool(getattr(args, "robust_mpr", True)):
        raise ValueError(
            "canonical-factorized-radio-v1 requires --no-robust-mpr; the "
            "legacy center-fusion switch is otherwise ambiguous"
        )
    memory_fraction = float(getattr(args, "max_estimated_cpu_memory_fraction", 0.85))
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("--max-estimated-cpu-memory-fraction must lie in (0,1]")
    if int(getattr(args, "point_chunk_size", 4096)) <= 0:
        raise ValueError(
            "canonical-factorized-radio-v1 requires positive --point-chunk-size"
        )
    if bool(getattr(args, "normalize_each_view", False)):
        raise ValueError(
            "canonical-factorized-radio-v1 requires raw RADIO amplitudes; "
            "--normalize-each-view is forbidden"
        )
    responsibility_cache = str(
        getattr(args, "responsibility_cache", "")
    ).strip()
    save_responsibility_cache = str(
        getattr(args, "save_responsibility_cache", "")
    ).strip()
    expected_responsibility = str(
        getattr(args, "expected_responsibility_cache_sha256", "")
    )
    if responsibility_cache and re.fullmatch(
        r"[0-9a-f]{64}", expected_responsibility
    ) is None:
        raise ValueError(
            "canonical-factorized-radio-v1 requires the responsibility "
            "authority's expected SHA-256 when loading"
        )
    if (
        aggregation_mode == "raster_marginal_responsibility"
        and save_responsibility_cache
        and expected_responsibility
    ):
        raise ValueError(
            "a newly created responsibility authority cannot have a predeclared SHA"
        )
    if aggregation_mode == "raster_gaussian_top1":
        if save_responsibility_cache:
            raise ValueError(
                "canonical-factorized-radio-v1 top1 builder forbids live "
                "responsibility creation"
            )
        if not responsibility_cache:
            raise ValueError(
                "canonical-factorized-radio-v1 requires --responsibility-cache"
            )
        if str(getattr(args, "registration_weight_mode", "")) != "alpha_depth":
            raise ValueError(
                "canonical-factorized-radio-v1 requires frozen options: "
                "alpha_depth registration"
            )
    else:
        if bool(responsibility_cache) == bool(save_responsibility_cache):
            raise ValueError(
                "canonical-factorized-radio-v1 exact marginal requires exactly "
                "one responsibility authority input or output"
            )
        if float(getattr(args, "alpha_threshold", -1.0)) != 0.0:
            raise ValueError(
                "factorized exact marginal responsibility requires alpha threshold 0"
            )
        args.registration_weight_mode = (
            "exact_front_to_back_marginal_responsibility"
        )
    for name, label in (
        ("expected_feature_output_bundle_sha256", "feature output bundle"),
        ("expected_geometry_checkpoint_sha256", "geometry checkpoint"),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name, ""))) is None:
            raise ValueError(
                "canonical-factorized-radio-v1 requires the caller-trusted "
                f"{label} SHA-256"
            )
    if (
        bool(getattr(args, "benchmark_images_opened", False))
        or bool(getattr(args, "text_queries_opened", False))
        or bool(str(getattr(args, "query_names", "")).strip())
    ):
        raise ValueError("canonical-factorized-radio-v1 is query-free and source-only")
    return True


# These names and dimensions are those emitted by the official C-RADIOv4-H
# runtime.  The direct path deliberately consumes the saved official adaptor
# output, rather than applying an MLP to an already interpolated raw map.
_EXTRACTED_CAPABILITY_SPECS: dict[str, dict[str, object]] = {
    "siglip2-g": {
        "adaptor_name": "siglip2-g",
        "subdir": "siglip2",
        "output_dim": 1536,
    },
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
        expected_output_bundle_sha256=(str(expected_output_bundle_sha256) or None),
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
            raise ValueError("selected feature maps do not share one spatial shape")
        maps[output_index].copy_(item)
    if maps is None:
        raise ValueError("feature map selection is empty")
    return maps


def validate_raster_reliability_policy(args: argparse.Namespace) -> None:
    """Resolve the observation contract before checking reliability options."""

    memory_fraction = float(getattr(args, "max_estimated_cpu_memory_fraction", 0.85))
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("--max-estimated-cpu-memory-fraction must lie in (0,1]")
    if int(getattr(args, "capability_shard_channels", 256)) <= 0:
        raise ValueError("--capability-shard-channels must be positive")
    if validate_factorized_radio_builder_policy(args):
        return
    if str(getattr(args, "observation_contract", "legacy")) in {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
        CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
    }:
        apply_canonical_observation_contract(args)
    if (
        args.raster_reliability_mode != "legacy_valid"
        and args.aggregation_mode == "center"
    ):
        raise ValueError("--raster-reliability-mode applies only to raster aggregation")
    if (
        args.raster_reliability_mode == "mean_resultant"
        and not args.normalize_each_view
    ):
        raise ValueError(
            "--raster-reliability-mode mean_resultant requires " "--normalize-each-view"
        )
    if args.aggregation_mode in {
        "raster_marginal_responsibility",
        "raster_exact_center_uncertainty",
    }:
        if args.raster_view_fusion != "contribution_mean":
            raise ValueError(
                "raster marginal responsibility requires contribution_mean fusion"
            )
        if float(args.alpha_threshold) != 0.0:
            raise ValueError(
                "raster marginal responsibility forbids post-compositor alpha filtering"
            )
        responsibility_cache = str(
            getattr(args, "responsibility_cache", "")
        ).strip()
        save_responsibility_cache = str(
            getattr(args, "save_responsibility_cache", "")
        ).strip()
        if responsibility_cache and save_responsibility_cache:
            raise ValueError(
                "marginal responsibility authority load/save are mutually exclusive"
            )
        if responsibility_cache and re.fullmatch(
            r"[0-9a-f]{64}",
            str(getattr(args, "expected_responsibility_cache_sha256", "")),
        ) is None:
            raise ValueError(
                "marginal responsibility reuse requires a caller-trusted SHA-256"
            )
        args.registration_weight_mode = (
            "exact_front_to_back_marginal_responsibility"
            if args.aggregation_mode == "raster_marginal_responsibility"
            else "exact_front_to_back_adjoint_center"
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
        raise ValueError(
            f"no extracted official capability source for {feature_space!r}"
        )
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
        raise ValueError("official capability manifest belongs to a different scene")
    if str(expected_image_dir):
        if (
            not image_dir
            or Path(image_dir).resolve() != Path(expected_image_dir).resolve()
        ):
            raise ValueError(
                "official capability manifest belongs to a different image " "directory"
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
    if len(set(frame_indices)) != len(frame_indices) or len(set(saved_stems)) != len(
        saved_stems
    ):
        raise ValueError("feature manifest contains duplicate frames")
    expected_frames = {int(value) for value in (expected_frame_indices or [])}
    if expected_frames and not expected_frames.issubset(frame_indices):
        raise ValueError("official capability manifest lacks selected raw RADIO frames")
    radio = manifest.get("radio")
    if not isinstance(radio, dict):
        raise ValueError("feature manifest does not declare its RADIO runtime")
    checkpoint_sha256 = str(radio.get("checkpoint_sha256", ""))
    checkpoint_provenance = str(radio.get("checkpoint_provenance", ""))
    checkpoint_load_contract = str(radio.get("checkpoint_load_contract", ""))
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
        raise ValueError(
            f"official {space} adaptor manifest has an invalid spatial grid"
        )
    subdir = root / str(expected["subdir"])
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"feature manifest declares {space}, but its map directory is missing: {subdir}"
        )
    expected_files = {f"{stem}.pt" for stem in saved_stems}
    actual_files = {
        path.name
        for path in subdir.iterdir()
        if path.is_file() and path.suffix == ".pt"
    }
    if actual_files != expected_files:
        raise ValueError(f"official {space} map files differ from the frame manifest")
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
        "output_bundle_sha256": str(bundle_validation["output_bundle_sha256"]),
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
        return {int(item.stem.split("_")[-1]) for item in path.glob("frame_*.json")}
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
            "canonical full-observation MPR requires " f"{manifest_path}"
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
    selected_frame_ids = [
        int(value) for value in payload.get("selected_frame_indices", [])
    ]
    ranked_frame_ids = [
        int(value) for value in payload.get("selection_order_frame_indices", [])
    ]
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
    candidates = torch.cat([current_responsibility, responsibility[:, None]], dim=1)
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
        sums = torch.zeros(num_rows, stop - start, dtype=torch.float32, device=device)
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
        raise ValueError("registered sums/counts must be aligned float32 CPU tensors")
    if int(row_chunk_size) <= 0:
        raise ValueError("row_chunk_size must be positive")
    valid = counts > 0
    features = torch.zeros_like(sums, dtype=torch.float16)
    for start in range(0, sums.shape[0], int(row_chunk_size)):
        stop = min(start + int(row_chunk_size), sums.shape[0])
        chunk_valid = valid[start:stop]
        normalized = sums[start:stop] / counts[start:stop, None].clamp_min(1e-8)
        normalized[~chunk_valid] = 0.0
        features[start:stop].copy_(normalized.half())
        del normalized
    return features, valid


def initialize_factorized_radio_accumulators(
    num_gaussians: int,
    feature_dim: int,
) -> dict[str, torch.Tensor]:
    """Allocate the one dense float32 semantic accumulator and scalar state."""

    if int(num_gaussians) <= 0 or int(feature_dim) <= 0:
        raise ValueError("factorized RADIO accumulator dimensions must be positive")
    return {
        "weighted_unit_sum": torch.zeros(
            int(num_gaussians), int(feature_dim), dtype=torch.float32
        ),
        "weighted_log_amplitude_sum": torch.zeros(
            int(num_gaussians), dtype=torch.float32
        ),
        "weighted_log_amplitude_square_sum": torch.zeros(
            int(num_gaussians), dtype=torch.float32
        ),
        "responsibility_mass": torch.zeros(int(num_gaussians), dtype=torch.float32),
        "visible_mass": torch.zeros(int(num_gaussians), dtype=torch.float32),
        "positive_view_count": torch.zeros(int(num_gaussians), dtype=torch.long),
    }


def accumulate_factorized_radio_view(
    feature_map: torch.Tensor,
    assignment: dict[str, torch.Tensor],
    accumulators: dict[str, torch.Tensor],
    *,
    observation_chunk_size: int,
) -> None:
    """Accumulate one raw view from a top-1 or exact marginal authority.

    A top-1 sidecar can retain multiple equal-weight pixels for one Gaussian.
    Each retained Gaussian/pixel pair is therefore an observation, while the
    bounded evidence counter advances at most once per Gaussian and view.
    Zero-norm pixels are excluded before division or logarithms.
    """

    values = torch.as_tensor(feature_map).detach().float().cpu()
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3 or min(values.shape) <= 0:
        raise ValueError("factorized RADIO feature map must be [C,H,W]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("factorized RADIO feature map contains non-finite values")
    if int(observation_chunk_size) <= 0:
        raise ValueError("factorized RADIO observation chunk size must be positive")
    required_state = {
        "weighted_unit_sum",
        "weighted_log_amplitude_sum",
        "weighted_log_amplitude_square_sum",
        "responsibility_mass",
        "visible_mass",
        "positive_view_count",
    }
    if set(accumulators) != required_state:
        raise ValueError("factorized RADIO accumulator fields differ")
    direction_sum = accumulators["weighted_unit_sum"]
    log_sum = accumulators["weighted_log_amplitude_sum"]
    log_square_sum = accumulators["weighted_log_amplitude_square_sum"]
    responsibility_mass = accumulators["responsibility_mass"]
    visible_mass = accumulators["visible_mass"]
    view_count = accumulators["positive_view_count"]
    num_gaussians, feature_dim = direction_sum.shape
    if (
        direction_sum.dtype != torch.float32
        or direction_sum.device.type != "cpu"
        or int(feature_dim) != int(values.shape[0])
        or log_sum.shape != (num_gaussians,)
        or log_square_sum.shape != (num_gaussians,)
        or responsibility_mass.shape != (num_gaussians,)
        or visible_mass.shape != (num_gaussians,)
        or view_count.shape != (num_gaussians,)
        or any(
            item.device.type != "cpu"
            for item in (
                log_sum,
                log_square_sum,
                responsibility_mass,
                visible_mass,
                view_count,
            )
        )
        or any(
            item.dtype != torch.float32
            for item in (
                log_sum,
                log_square_sum,
                responsibility_mass,
                visible_mass,
            )
        )
        or view_count.dtype != torch.long
    ):
        raise ValueError("factorized RADIO accumulator tensors differ")

    raw_gaussian_ids = assignment.get("gaussian_ids")
    raw_pixel_ids = assignment.get("pixel_ids")
    raw_weights = assignment.get("weights")
    raw_base_weights = assignment.get("base_weights")
    raw_marginal_weights = assignment.get("marginal_weights")
    exact_marginal = raw_base_weights is not None or raw_marginal_weights is not None
    if exact_marginal:
        if not torch.is_tensor(raw_base_weights) or not torch.is_tensor(
            raw_marginal_weights
        ):
            raise ValueError(
                "factorized exact marginal assignment requires base and marginal weights"
            )
        raw_weights = raw_marginal_weights
    if not all(
        torch.is_tensor(item) for item in (raw_gaussian_ids, raw_pixel_ids, raw_weights)
    ):
        raise ValueError("factorized RADIO assignment tensors are missing")
    gaussian_ids = raw_gaussian_ids.detach().long().cpu()
    pixel_ids = raw_pixel_ids.detach().long().cpu()
    weights = raw_weights.detach().float().cpu()
    base_weights = (
        raw_base_weights.detach().float().cpu()
        if torch.is_tensor(raw_base_weights)
        else None
    )
    if (
        gaussian_ids.ndim != 1
        or pixel_ids.shape != gaussian_ids.shape
        or weights.shape != gaussian_ids.shape
        or (base_weights is not None and base_weights.shape != gaussian_ids.shape)
    ):
        raise ValueError("factorized RADIO assignment tensors do not align")
    num_pixels = int(values.shape[1]) * int(values.shape[2])
    if gaussian_ids.numel() and (
        int(gaussian_ids.min()) < 0
        or int(gaussian_ids.max()) >= int(num_gaussians)
        or int(pixel_ids.min()) < 0
        or int(pixel_ids.max()) >= num_pixels
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
        or (
            base_weights is not None
            and (
                not bool(torch.isfinite(base_weights).all())
                or bool((base_weights <= 0).any())
                or bool((weights > base_weights + 1e-7).any())
            )
        )
    ):
        raise ValueError("factorized RADIO assignment values are invalid")

    flat_pixels = values.reshape(feature_dim, num_pixels).t()
    frame_positive = torch.zeros(num_gaussians, dtype=torch.bool)
    for start in range(0, gaussian_ids.numel(), int(observation_chunk_size)):
        stop = min(start + int(observation_chunk_size), gaussian_ids.numel())
        gids = gaussian_ids[start:stop]
        if base_weights is not None:
            # Visibility is geometric evidence. A zero-norm feature may not
            # supervise semantics, but it remains in this denominator.
            visible_mass.index_add_(0, gids, base_weights[start:stop])
        sampled = flat_pixels[pixel_ids[start:stop]]
        amplitude = torch.linalg.vector_norm(sampled, dim=-1)
        positive = amplitude > 0
        if not bool(positive.any()):
            continue
        gids = gids[positive]
        sampled = sampled[positive]
        amplitude = amplitude[positive]
        weight = weights[start:stop][positive]
        log_amplitude = torch.log(amplitude)
        direction_sum.index_add_(
            0,
            gids,
            sampled * (weight / amplitude)[:, None],
        )
        log_sum.index_add_(0, gids, weight * log_amplitude)
        log_square_sum.index_add_(0, gids, weight * log_amplitude.square())
        responsibility_mass.index_add_(0, gids, weight)
        frame_positive[gids] = True
    view_count[frame_positive] += 1


def finalize_factorized_radio_accumulators(
    accumulators: dict[str, torch.Tensor],
    *,
    row_chunk_size: int,
    visibility_purity_measured: bool = False,
) -> tuple[dict[str, object], torch.Tensor]:
    """Finalize the streaming state without persisting semantic direction."""

    if int(row_chunk_size) <= 0:
        raise ValueError("factorized RADIO row chunk size must be positive")
    direction_sum = accumulators["weighted_unit_sum"]
    log_sum = accumulators["weighted_log_amplitude_sum"]
    log_square_sum = accumulators["weighted_log_amplitude_square_sum"]
    mass = accumulators["responsibility_mass"]
    visible_mass = accumulators["visible_mass"]
    view_counts = accumulators["positive_view_count"]
    num_gaussians, feature_dim = direction_sum.shape
    canonical = torch.zeros(num_gaussians, feature_dim, dtype=torch.float16)
    log_amplitude = torch.zeros(num_gaussians, dtype=torch.float32)
    reliability = torch.zeros(
        num_gaussians,
        len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
        dtype=torch.float32,
    )
    valid = torch.zeros(num_gaussians, dtype=torch.bool)
    for start in range(0, num_gaussians, int(row_chunk_size)):
        stop = min(start + int(row_chunk_size), num_gaussians)
        chunk_mass = mass[start:stop]
        has_mass = chunk_mass > 0
        safe_mass = chunk_mass.clamp_min(torch.finfo(torch.float32).tiny)
        direction_mean = direction_sum[start:stop] / safe_mass[:, None]
        direction_norm = torch.linalg.vector_norm(direction_mean, dim=-1)
        chunk_valid = has_mass & (direction_norm > 0)
        chunk_log_amplitude = log_sum[start:stop] / safe_mass
        second_moment = log_square_sum[start:stop] / safe_mass
        log_std = torch.sqrt(
            (second_moment - chunk_log_amplitude.square()).clamp_min(0.0)
        )
        resultant = direction_norm.clamp(0.0, 1.0)
        evidence = view_counts[start:stop].float()
        evidence = evidence / (evidence + 1.0)
        purity = (
            torch.where(
                visible_mass[start:stop] > 0,
                chunk_mass
                / visible_mass[start:stop].clamp_min(
                    torch.finfo(torch.float32).tiny
                ),
                torch.zeros_like(chunk_mass),
            ).clamp(0.0, 1.0)
            if bool(visibility_purity_measured)
            else torch.zeros_like(resultant)
        )
        chunk_canonical = torch.zeros_like(direction_mean)
        if bool(chunk_valid.any()):
            unit_direction = (
                direction_mean[chunk_valid] / direction_norm[chunk_valid, None]
            )
            chunk_canonical[chunk_valid] = (
                torch.exp(chunk_log_amplitude[chunk_valid])[:, None] * unit_direction
            )
        chunk_reliability = torch.stack(
            (
                resultant,
                1.0 - resultant,
                log_std,
                evidence,
                purity,
            ),
            dim=-1,
        )
        chunk_log_amplitude[~chunk_valid] = 0.0
        chunk_reliability[~chunk_valid] = 0.0
        if not bool(torch.isfinite(chunk_canonical).all()) or not bool(
            torch.isfinite(chunk_reliability).all()
        ):
            raise ValueError("factorized RADIO finalization produced non-finite rows")
        canonical[start:stop].copy_(chunk_canonical.half())
        log_amplitude[start:stop].copy_(chunk_log_amplitude)
        reliability[start:stop].copy_(chunk_reliability)
        valid[start:stop].copy_(chunk_valid)

    core_payload: dict[str, object] = {
        "schema": CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA,
        "schema_version": 1,
        "contract": canonical_factorized_radio_contract(),
        "contract_sha256": CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256,
        "reliability_scalar_names": list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES),
        "reliability_scalar_names_sha256": (
            FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        ),
        "log_amplitude": log_amplitude,
        "canonical_feature": canonical,
        "valid": valid,
        "reliability": reliability,
    }
    return core_payload, view_counts.clone()


def _stream_factorized_radio_from_bundle(
    *,
    feature_dir: str | Path,
    selected_frame_indices: list[int],
    tensor_records: dict[str, dict[str, object]],
    feature_size: tuple[int, int],
    responsibility_assignments: list[dict[str, torch.Tensor]],
    num_gaussians: int,
    observation_chunk_size: int,
    row_chunk_size: int,
    visibility_purity_measured: bool = False,
) -> tuple[dict[str, object], torch.Tensor]:
    """Stream one raw feature view at a time through immutable assignments."""

    if len(selected_frame_indices) != len(responsibility_assignments):
        raise ValueError("factorized RADIO views and assignments do not align")
    accumulators = initialize_factorized_radio_accumulators(int(num_gaussians), 1280)
    for frame_index, assignment in tqdm(
        zip(selected_frame_indices, responsibility_assignments),
        total=len(selected_frame_indices),
        desc="factorize raw RADIO observations",
    ):
        feature_map = _load_bundle_feature_maps(
            feature_dir=feature_dir,
            selected_frame_indices=[int(frame_index)],
            subdir="backbone",
            expected_dim=1280,
            feature_size=feature_size,
            tensor_records=tensor_records,
            normalize=False,
            output_dtype=torch.float32,
        )
        if tuple(feature_map.shape[-2:]) != tuple(int(v) for v in feature_size):
            raise ValueError(
                "factorized RADIO raw map grid differs from the frozen "
                "responsibility sidecar grid"
            )
        accumulate_factorized_radio_view(
            feature_map,
            assignment,
            accumulators,
            observation_chunk_size=int(observation_chunk_size),
        )
        del feature_map
    return finalize_factorized_radio_accumulators(
        accumulators,
        row_chunk_size=int(row_chunk_size),
        visibility_purity_measured=bool(visibility_purity_measured),
    )


def _gate_exact_marginal_assignments_by_raw_amplitude(
    *,
    feature_dir: str | Path,
    selected_frame_indices: list[int],
    tensor_records: dict[str, dict[str, object]],
    feature_size: tuple[int, int],
    responsibility_assignments: list[dict[str, torch.Tensor]],
    num_gaussians: int,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    """Share one pre-adaptor raw semantic gate across RADIO/DINO/SAM."""

    if len(selected_frame_indices) != len(responsibility_assignments):
        raise ValueError("raw amplitude gate views and assignments do not align")
    geometric_view_counts = torch.zeros(int(num_gaussians), dtype=torch.long)
    semantic_assignments: list[dict[str, torch.Tensor]] = []
    for frame_index, assignment in zip(
        selected_frame_indices, responsibility_assignments
    ):
        gaussian_ids = assignment.get("gaussian_ids")
        pixel_ids = assignment.get("pixel_ids")
        if not torch.is_tensor(gaussian_ids) or not torch.is_tensor(pixel_ids):
            raise ValueError("raw amplitude gate assignment ids are missing")
        gids = gaussian_ids.detach().long().cpu().reshape(-1)
        pids = pixel_ids.detach().long().cpu().reshape(-1)
        if gids.shape != pids.shape:
            raise ValueError("raw amplitude gate assignment ids do not align")
        frame_geometric = torch.zeros(int(num_gaussians), dtype=torch.bool)
        frame_geometric[gids] = True
        geometric_view_counts[frame_geometric] += 1

        raw_map = _load_bundle_feature_maps(
            feature_dir=feature_dir,
            selected_frame_indices=[int(frame_index)],
            subdir="backbone",
            expected_dim=1280,
            feature_size=feature_size,
            tensor_records=tensor_records,
            normalize=False,
            output_dtype=torch.float32,
        )
        if tuple(raw_map.shape) != (1, 1280, *tuple(feature_size)):
            raise ValueError("raw amplitude gate feature grid differs")
        raw_amplitude = torch.linalg.vector_norm(raw_map[0], dim=0).reshape(-1)
        positive = raw_amplitude[pids] > 0
        del raw_map, raw_amplitude
        filtered: dict[str, torch.Tensor] = {}
        for name, value in assignment.items():
            if not torch.is_tensor(value) or value.reshape(-1).shape != gids.shape:
                raise ValueError("raw amplitude gate assignment tensors differ")
            filtered[name] = value.detach().cpu().reshape(-1)[positive]
        semantic_assignments.append(filtered)
    return semantic_assignments, geometric_view_counts


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    """Atomically replace a mutable, build-owned progress receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _commit_temporary_no_clobber(temporary: Path, output: Path) -> None:
    """Publish a complete file without replacing an existing artifact."""

    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"immutable output already exists: {output}") from error


class _BuildFileLock:
    """Crash-released advisory lock for one mutable progress namespace."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._descriptor = os.open(path, flags, 0o600)
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._descriptor)
            self._descriptor = -1
            raise RuntimeError("artifact progress is already being written") from error

    def __enter__(self) -> "_BuildFileLock":
        return self

    def __exit__(self, *_args: object) -> None:
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = -1


def _write_json_noclobber(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
        _commit_temporary_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_marginal_visibility_purity(
    responsibility_assignments: list[dict[str, torch.Tensor]],
    *,
    num_gaussians: int,
    valid: torch.Tensor,
    geometric_assignments: list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Derive feature-independent purity from one shared sparse authority."""

    visible_mass = torch.zeros(int(num_gaussians), dtype=torch.float32)
    semantic_mass = torch.zeros_like(visible_mass)
    for assignment in (
        responsibility_assignments
        if geometric_assignments is None
        else geometric_assignments
    ):
        gaussian_ids = assignment.get("gaussian_ids")
        base_weights = assignment.get("base_weights")
        if not all(
            torch.is_tensor(item)
            for item in (gaussian_ids, base_weights)
        ):
            raise ValueError(
                "exact-marginal visibility purity requires geometric base weights"
            )
        gids = gaussian_ids.detach().long().cpu()
        base = base_weights.detach().float().cpu()
        if gids.shape != base.shape:
            raise ValueError("exact-marginal purity tensors do not align")
        visible_mass.index_add_(0, gids, base)

    for assignment in responsibility_assignments:
        gaussian_ids = assignment.get("gaussian_ids")
        marginal_weights = assignment.get("marginal_weights")
        if not all(
            torch.is_tensor(item) for item in (gaussian_ids, marginal_weights)
        ):
            raise ValueError(
                "exact-marginal visibility purity requires semantic marginal weights"
            )
        gids = gaussian_ids.detach().long().cpu()
        marginal = marginal_weights.detach().float().cpu()
        if gids.shape != marginal.shape:
            raise ValueError("exact-marginal purity tensors do not align")
        semantic_mass.index_add_(0, gids, marginal)
    purity = torch.where(
        visible_mass > 0,
        semantic_mass / visible_mass.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(visible_mass),
    ).clamp(0.0, 1.0)
    active = torch.as_tensor(valid).detach().bool().cpu().reshape(-1)
    if active.shape != purity.shape:
        raise ValueError("exact-marginal purity support differs")
    purity[~active] = 0.0
    return purity


def _stream_channel_sharded_contribution_mean_impl(
    *,
    output: Path,
    feature_space: str,
    feature_dir: Path,
    feature_tensor_records: dict[str, dict[str, object]],
    selected_frame_indices: list[int],
    feature_size: tuple[int, int],
    responsibility_assignments: list[dict[str, torch.Tensor]],
    num_gaussians: int,
    output_dim: int,
    shard_channels: int,
    inner_channel_chunk_size: int,
    point_chunk_size: int,
    num_views: int,
    normalize_each_view: bool,
    reliability_mode: str,
    adaptor: torch.nn.Module | None,
    device: torch.device,
    resume_contract: dict[str, object],
) -> tuple[list[dict[str, object]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact contribution-mean feature channels without dense N x D state."""

    if feature_space not in {"radio", "dino_v3", "sam3"}:
        raise ValueError("channel-sharded MPR supports raw RADIO/DINO/SAM only")
    if feature_space == "radio" and adaptor is not None:
        raise ValueError("raw RADIO sharding must not receive an adaptor")
    if feature_space != "radio" and adaptor is None:
        raise ValueError("capability sharding requires a frozen official adaptor")
    if (
        len(responsibility_assignments) != len(selected_frame_indices)
        or num_views != len(selected_frame_indices)
        or num_gaussians <= 0
        or output_dim <= 0
        or shard_channels <= 0
    ):
        raise ValueError("channel-sharded MPR dimensions do not align")

    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(output.suffix + ".partial.json")
    contract_sha256 = hashlib.sha256(
        json.dumps(resume_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    completed: dict[tuple[int, int], dict[str, object]] = {}
    expected_shard_keys = {
        (start, min(start + int(shard_channels), output_dim))
        for start in range(0, output_dim, int(shard_channels))
    }
    if progress_path.exists():
        progress, _progress_sha256, _progress_source = load_json_object(
            progress_path, label="channel-sharded MPR progress"
        )
        if (
            not isinstance(progress, dict)
            or progress.get("schema") != "radio_gs.channel_sharded_mpr_progress.v1"
            or progress.get("resume_contract_sha256") != contract_sha256
            or progress.get("resume_contract") != resume_contract
            or not isinstance(progress.get("shards"), list)
        ):
            raise ValueError("channel-sharded MPR resume contract differs")
        for record in progress["shards"]:
            if not isinstance(record, dict):
                raise ValueError("channel-sharded MPR progress record is invalid")
            key = (
                int(record.get("channel_start", -1)),
                int(record.get("channel_stop", -1)),
            )
            if key not in expected_shard_keys or key in completed:
                raise ValueError(
                    "channel-sharded MPR progress has repeated or unknown channels"
                )
            completed[key] = dict(record)
    else:
        _write_json_noclobber(
            progress_path,
            {
                "schema": "radio_gs.channel_sharded_mpr_progress.v1",
                "resume_contract_sha256": contract_sha256,
                "resume_contract": resume_contract,
                "shards": [],
            },
        )

    registered_counts = torch.zeros(num_gaussians, dtype=torch.float32)
    observation_counts = torch.zeros(num_gaussians, dtype=torch.long)
    for assignment in responsibility_assignments:
        gids = assignment["gaussian_ids"].to(device).long()
        weights = assignment["weights"].to(device).float()
        frame_counts = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
        if gids.numel():
            frame_counts.index_add_(0, gids, weights)
        counts_cpu = frame_counts.cpu()
        registered_counts.add_(counts_cpu)
        observation_counts[counts_cpu > 0] += 1
        del gids, weights, frame_counts, counts_cpu
    valid = registered_counts > 0
    if not torch.equal(valid, observation_counts > 0):
        raise RuntimeError("sharded contribution support/counts disagree")

    shard_records: list[dict[str, object]] = []
    squared_norm = torch.zeros(num_gaussians, dtype=torch.float32)
    for channel_start in range(0, output_dim, int(shard_channels)):
        channel_stop = min(channel_start + int(shard_channels), output_dim)
        width = channel_stop - channel_start
        shard_name = (
            f"{output.name}.channels_{channel_start:05d}_{channel_stop:05d}.f16"
        )
        shard_path = output.parent / shard_name
        key = (channel_start, channel_stop)
        record = completed.get(key)
        expected_bytes = num_gaussians * width * 2
        reuse = False
        if record is not None:
            if (
                record.get("relative_path") != shard_name
                or record.get("dtype") != "float16"
                or record.get("shape") != [num_gaussians, width]
                or not shard_path.is_file()
                or shard_path.is_symlink()
                or shard_path.stat().st_size != expected_bytes
                or _sha256_file(shard_path) != record.get("sha256")
            ):
                raise ValueError("completed channel shard failed resume validation")
            reuse = True
        elif shard_path.exists():
            # The shard is published atomically before the mutable progress
            # receipt.  A power loss in that narrow window leaves a complete,
            # deterministic orphan which can be validated and rebound.
            if (
                not shard_path.is_file()
                or shard_path.is_symlink()
                or shard_path.stat().st_size != expected_bytes
            ):
                raise ValueError("unbound channel shard failed recovery validation")
            record = {
                "relative_path": shard_name,
                "sha256": _sha256_file(shard_path),
                "channel_start": channel_start,
                "channel_stop": channel_stop,
                "dtype": "float16",
                "shape": [num_gaussians, width],
            }
            completed[key] = record
            _atomic_json(
                progress_path,
                {
                    "schema": "radio_gs.channel_sharded_mpr_progress.v1",
                    "resume_contract_sha256": contract_sha256,
                    "resume_contract": resume_contract,
                    "shards": [completed[item] for item in sorted(completed)],
                },
            )
            reuse = True

        if not reuse:
            registered_sum = torch.zeros(num_gaussians, width, dtype=torch.float32)
            sum_staging = torch.empty(
                num_gaussians,
                min(width, int(inner_channel_chunk_size)),
                dtype=torch.float32,
            )
            count_staging = torch.empty(num_gaussians, dtype=torch.float32)
            ignored_counts = torch.zeros(num_gaussians, dtype=torch.float32)
            for view_index, frame_index in enumerate(selected_frame_indices):
                raw_map = _load_bundle_feature_maps(
                    feature_dir=feature_dir,
                    selected_frame_indices=[int(frame_index)],
                    subdir="backbone",
                    expected_dim=1280,
                    feature_size=feature_size,
                    tensor_records=feature_tensor_records,
                    normalize=False,
                    output_dtype=torch.float32,
                )
                if adaptor is None:
                    projected = raw_map
                else:
                    with torch.inference_mode():
                        projected = (
                            project_feature_map_with_adaptor(
                                raw_map.to(device), adaptor, normalize=True
                            )
                            .half()
                            .cpu()
                        )
                    del raw_map
                full_view_features = projected.to(device=device, dtype=torch.float32)
                if normalize_each_view:
                    full_view_features = F.normalize(
                        full_view_features, dim=1, eps=1e-8
                    )
                view_features = full_view_features[:, channel_start:channel_stop]
                assignment = responsibility_assignments[view_index]
                accumulate_contribution_mean_channel_chunked(
                    view_features,
                    assignment["gaussian_ids"].to(device),
                    assignment["pixel_ids"].to(device),
                    assignment["weights"].to(device),
                    registered_sum,
                    ignored_counts,
                    channel_chunk_size=int(inner_channel_chunk_size),
                    cpu_sum_staging=sum_staging,
                    cpu_count_staging=count_staging,
                )
                del projected, full_view_features, view_features
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if not torch.equal(ignored_counts, registered_counts):
                raise RuntimeError(
                    "channel-sharded accumulation support weights changed "
                    "between the frozen count pass and feature pass"
                )

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{shard_name}.", suffix=".tmp", dir=output.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                mapped = np.memmap(
                    temporary,
                    mode="w+",
                    dtype="<f2",
                    shape=(num_gaussians, width),
                    order="C",
                )
                for row_start in range(0, num_gaussians, int(point_chunk_size)):
                    row_stop = min(row_start + int(point_chunk_size), num_gaussians)
                    values = registered_sum[row_start:row_stop] / registered_counts[
                        row_start:row_stop, None
                    ].clamp_min(1e-8)
                    values[~valid[row_start:row_stop]] = 0.0
                    half = values.half()
                    mapped[row_start:row_stop] = half.numpy()
                    if reliability_mode == "mean_resultant":
                        squared_norm[row_start:row_stop].add_(
                            half.float().square().sum(dim=-1)
                        )
                    del values, half
                mapped.flush()
                del mapped
                _commit_temporary_no_clobber(temporary, shard_path)
            finally:
                temporary.unlink(missing_ok=True)
            del registered_sum, sum_staging, count_staging, ignored_counts
            record = {
                "relative_path": shard_name,
                "sha256": _sha256_file(shard_path),
                "channel_start": channel_start,
                "channel_stop": channel_stop,
                "dtype": "float16",
                "shape": [num_gaussians, width],
            }
            completed[key] = record
            _atomic_json(
                progress_path,
                {
                    "schema": "radio_gs.channel_sharded_mpr_progress.v1",
                    "resume_contract_sha256": contract_sha256,
                    "resume_contract": resume_contract,
                    "shards": [completed[item] for item in sorted(completed)],
                },
            )
        else:
            assert record is not None
            if reliability_mode == "mean_resultant":
                mapped = np.memmap(
                    shard_path,
                    mode="r",
                    dtype="<f2",
                    shape=(num_gaussians, width),
                    order="C",
                )
                for row_start in range(0, num_gaussians, int(point_chunk_size)):
                    row_stop = min(row_start + int(point_chunk_size), num_gaussians)
                    values = torch.from_numpy(
                        np.asarray(mapped[row_start:row_stop]).copy()
                    ).float()
                    squared_norm[row_start:row_stop].add_(values.square().sum(dim=-1))
                del mapped
        shard_records.append(dict(record))

    if reliability_mode == "legacy_valid":
        agreement = valid.float()
    elif reliability_mode == "mean_resultant":
        agreement = squared_norm.sqrt().clamp(0.0, 1.0)
        agreement[~valid] = 0.0
    else:
        raise ValueError("unsupported channel-sharded reliability mode")
    reliability = torch.stack(
        [
            observation_counts.float() / float(max(1, num_views)),
            agreement,
            valid.float(),
        ],
        dim=-1,
    ).half()
    return shard_records, valid, observation_counts, reliability


def _stream_channel_sharded_contribution_mean(
    **kwargs: object,
) -> tuple[list[dict[str, object]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serialize progress mutation while retaining crash-safe shard recovery."""

    output_value = kwargs.get("output")
    if output_value is None:
        raise TypeError("channel-sharded stream requires output")
    output = Path(output_value).expanduser()
    lock_path = output.with_suffix(output.suffix + ".partial.lock")
    with _BuildFileLock(lock_path):
        return _stream_channel_sharded_contribution_mean_impl(**kwargs)


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
    teacher_maps = int(num_views) * int(channels) * int(height) * int(width) * 2
    components = {
        "teacher_maps_float16": teacher_maps,
        "responsibility_cache_file_allowance": max(0, int(responsibility_cache_bytes)),
    }
    if str(aggregation_mode) != "center":
        rows, dims = int(num_gaussians), int(channels)
        components["registered_sum_float32"] = rows * dims * 4
        components["final_features_float16"] = rows * dims * 2
        if str(raster_view_fusion) == "contribution_mean":
            components["channel_staging_float32"] = (
                rows * min(dims, int(raster_channel_chunk_size)) * 4
            )
        elif str(raster_view_fusion) == "topk_mean":
            components["topk_features_float32"] = (
                rows * dims * max(1, int(raster_topk)) * 4
            )
    components["estimated_peak_bytes"] = sum(components.values())
    return components


def estimate_factorized_radio_cpu_bytes(
    *,
    num_gaussians: int,
    feature_dim: int,
    feature_height: int,
    feature_width: int,
    observation_chunk_size: int,
    row_chunk_size: int,
) -> dict[str, int]:
    """Conservative peak estimate for the dense streaming factorization."""

    values = (
        num_gaussians,
        feature_dim,
        feature_height,
        feature_width,
        observation_chunk_size,
        row_chunk_size,
    )
    if any(int(value) <= 0 for value in values):
        raise ValueError("factorized RADIO memory dimensions must be positive")
    rows = int(num_gaussians)
    dims = int(feature_dim)
    observation_rows = min(rows, int(observation_chunk_size))
    finalization_rows = min(rows, int(row_chunk_size))
    components = {
        "weighted_unit_sum_float32": rows * dims * 4,
        "canonical_feature_float16": rows * dims * 2,
        # Four float32 scalar accumulators, one finalized log amplitude,
        # five float32 reliability scalars, one int64 view count, and one
        # boolean validity flag per Gaussian.
        "scalar_accumulators_and_outputs": rows * (4 * 10 + 8 + 1),
        "single_raw_feature_map_float32": (
            dims * int(feature_height) * int(feature_width) * 4
        ),
        # sampled, weighted-direction, and one normalization work buffer
        "observation_chunk_float32_working": observation_rows * dims * 4 * 3,
        # float32 canonical validation/finalization working tensors
        "row_chunk_float32_working": finalization_rows * dims * 4 * 2,
    }
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


def _unwrapped_implementation_source(value: object) -> Path:
    """Resolve the defining source file rather than a decorator wrapper."""

    source = inspect.getsourcefile(inspect.unwrap(value))
    if source is None:
        raise RuntimeError("cannot resolve implementation source")
    return Path(source).resolve()


def _sparse_exact_marginal_authority_metadata(
    *,
    args: argparse.Namespace,
    selected: list[int],
    selected_frame_indices: list[int],
    poses: torch.Tensor,
    renderer,
    model: torch.nn.Module,
    feature_height: int,
    feature_width: int,
) -> dict[str, object]:
    """Bind the feature-independent exact compositor and geometry identity."""

    # The compositor is decorated with ``torch.no_grad``.  Inspecting the
    # wrapper directly resolves to PyTorch's ``_contextlib.py`` and binds the
    # decorator instead of our implementation, so unwrap before hashing.
    compositor_source = _unwrapped_implementation_source(
        rasterize_single_view_contributions
    )
    expected_geometry = str(
        getattr(args, "expected_geometry_checkpoint_sha256", "")
    )
    if re.fullmatch(r"[0-9a-f]{64}", expected_geometry) is None:
        raise ValueError("sparse marginal authority requires geometry checkpoint SHA")
    return {
        "schema_version": 1,
        "assignment_mode": "exact_front_to_back_sparse_marginal",
        "registration_weight_mode": (
            "exact_front_to_back_marginal_responsibility"
        ),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "geometry_checkpoint_sha256": expected_geometry,
        "selected_dataset_indices": list(selected),
        "selected_frame_indices": list(selected_frame_indices),
        "excluded_frame_ids": sorted(_parse_frame_ids(args.exclude_frame_ids)),
        "feature_height": int(feature_height),
        "feature_width": int(feature_width),
        "post_compositor_alpha_threshold": 0.0,
        "depth_filter": "not_applied_to_exact_compositor_hits",
        "pose_sha256": _sha256_tensor_rows(poses),
        "intrinsics_sha256": _sha256_tensor_rows(
            renderer.scaled_intrinsics(feature_width, feature_height)
        ),
        "xyz_sha256": _sha256_tensor_rows(model.get_xyz()),
        "gaussian_state_sha256": _gaussian_state_sha256(model),
        "formula_sha256": SPARSE_EXACT_MARGINAL_FORMULA_SHA256,
        "builder_implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "compositor_implementation_sha256": _sha256_file(compositor_source),
        "authority_implementation_sha256": (
            sparse_exact_marginal_implementation_sha256()
        ),
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
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
        raise ValueError(f"responsibility cache contract differs: {sorted(mismatched)}")
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
            int(gaussian_ids.min()) < 0 or int(gaussian_ids.max()) >= int(num_gaussians)
        ):
            raise ValueError(
                f"responsibility view {view_index} has invalid Gaussian IDs"
            )
        num_pixels = int(expected_contract["feature_height"]) * int(
            expected_contract["feature_width"]
        )
        if pixel_ids.numel() and (
            int(pixel_ids.min()) < 0 or int(pixel_ids.max()) >= num_pixels
        ):
            raise ValueError(f"responsibility view {view_index} has invalid pixel IDs")
        # ``raster_gaussian_top1`` is primitive-top-1, not pixel-top-1: every
        # Gaussian retains its strongest raster hit, so several Gaussians may
        # legitimately sample the same feature pixel.  The rasterizer emits
        # hits in non-decreasing pixel order and never emits the same
        # Gaussian/pixel pair twice; keep those producer invariants fail-closed
        # instead of incorrectly requiring pixel IDs to be unique.
        if pixel_ids.numel() > 1 and bool((pixel_ids[1:] < pixel_ids[:-1]).any()):
            raise ValueError(
                f"responsibility view {view_index} has non-canonical pixel ordering"
            )
        if gaussian_ids.numel():
            gaussian_pixel_pairs = torch.stack([gaussian_ids, pixel_ids], dim=1)
            if (
                torch.unique(gaussian_pixel_pairs, dim=0).shape[0]
                != gaussian_pixel_pairs.shape[0]
            ):
                raise ValueError(
                    f"responsibility view {view_index} repeats Gaussian/pixel pairs"
                )

            # The producer uses ``weight >= max_weight - 1e-8``.  Consequently
            # an exact or float32-near tie can retain more than one pixel for a
            # Gaussian.  Re-evaluate that same predicate here: arbitrary
            # duplicate Gaussian rows remain invalid, while existing sidecars
            # preserve byte-identical lifting across RADIO/DINO/SAM.
            maximum_weights = torch.full(
                (int(num_gaussians),),
                -float("inf"),
                dtype=torch.float32,
            )
            maximum_weights.scatter_reduce_(
                0,
                gaussian_ids,
                weights,
                reduce="amax",
                include_self=True,
            )
            if bool((weights < maximum_weights[gaussian_ids] - 1e-8).any()):
                raise ValueError(
                    f"responsibility view {view_index} repeats Gaussian IDs "
                    "outside the top-1 tie tolerance"
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


def _load_saved_responsibility_for_sharded_resume(
    *,
    output: str | Path,
    save_path: str | Path,
    expected_contract: dict,
    num_gaussians: int,
) -> tuple[list[dict[str, torch.Tensor]], str, Path] | None:
    """Reuse a save-mode sidecar only when a partial manifest binds its SHA.

    ``torch.save`` archives are not byte-deterministic across temporary file
    names, so serializing identical assignments again can change the sidecar
    digest.  A resumed raw build must reopen the exact inode content already
    frozen by its progress contract; it must never overwrite it first.
    """

    responsibility = Path(save_path).expanduser()
    progress_path = (
        Path(output).expanduser().with_suffix(Path(output).suffix + ".partial.json")
    )
    if not responsibility.exists():
        if progress_path.exists():
            raise ValueError(
                "sharded MPR progress exists but its saved responsibility "
                "cache is missing"
            )
        return None
    if responsibility.is_symlink() or not responsibility.is_file():
        raise ValueError("saved responsibility cache must be a regular file")
    if not progress_path.exists():
        raise ValueError(
            "saved responsibility cache already exists without a sharded "
            "progress binding; use --responsibility-cache with an expected "
            "SHA-256 instead of overwriting it"
        )
    progress, _progress_sha256, _progress_source = load_json_object(
        progress_path, label="channel-sharded MPR progress"
    )
    resume_contract = progress.get("resume_contract")
    if (
        progress.get("schema") != "radio_gs.channel_sharded_mpr_progress.v1"
        or not isinstance(resume_contract, dict)
        or not isinstance(progress.get("shards"), list)
    ):
        raise ValueError("channel-sharded MPR progress is malformed")
    encoded_contract = json.dumps(
        resume_contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if (
        progress.get("resume_contract_sha256")
        != hashlib.sha256(encoded_contract).hexdigest()
    ):
        raise ValueError("channel-sharded MPR progress contract hash differs")
    bound_sha256 = str(
        resume_contract.get("registration_responsibility_cache_sha256", "")
    )
    if re.fullmatch(r"[0-9a-f]{64}", bound_sha256) is None:
        raise ValueError("channel-sharded MPR progress lacks a responsibility SHA-256")
    assignments, observed_sha256 = _load_responsibility_cache(
        responsibility,
        expected_contract=expected_contract,
        num_gaussians=int(num_gaussians),
        expected_sha256=bound_sha256,
    )
    return assignments, observed_sha256, responsibility.resolve()


def validate_factorized_radio_builder_payload(payload: dict[str, object]) -> None:
    """Validate the builder envelope without allocating another N x D tensor."""

    required = {
        "schema",
        "schema_version",
        "xyz",
        "geometry_fingerprint",
        "factorized_radio",
        "view_counts",
        "metadata",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("factorized RADIO builder cache fields differ")
    schema_pair = (payload.get("schema"), payload.get("schema_version"))
    if schema_pair == (CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA, 1):
        builder_version = 1
    elif schema_pair == (CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2, 2):
        builder_version = 2
    else:
        raise ValueError("factorized RADIO builder cache schema differs")
    xyz = payload.get("xyz")
    geometry_fingerprint = payload.get("geometry_fingerprint")
    view_counts = payload.get("view_counts")
    core = payload.get("factorized_radio")
    metadata = payload.get("metadata")
    if (
        not torch.is_tensor(xyz)
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype != torch.float32
        or not torch.is_tensor(view_counts)
        or view_counts.shape != (xyz.shape[0],)
        or view_counts.dtype != torch.long
        or not isinstance(core, dict)
        or not isinstance(metadata, dict)
    ):
        raise ValueError("factorized RADIO builder support tensors differ")
    expected_geometry_fingerprint = {
        "num_gaussians": int(xyz.shape[0]),
        "xyz_sha256": _sha256_tensor_rows(xyz),
    }
    if geometry_fingerprint != expected_geometry_fingerprint:
        raise ValueError("factorized RADIO geometry fingerprint differs")
    required_core = {
        "schema",
        "schema_version",
        "contract",
        "contract_sha256",
        "reliability_scalar_names",
        "reliability_scalar_names_sha256",
        "log_amplitude",
        "canonical_feature",
        "valid",
        "reliability",
    }
    if set(core) != required_core or (
        core.get("schema") != CANONICAL_FACTORIZED_RADIO_CACHE_SCHEMA
        or core.get("schema_version") != 1
        or core.get("contract") != canonical_factorized_radio_contract()
        or core.get("contract_sha256") != CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
        or core.get("reliability_scalar_names")
        != list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES)
        or core.get("reliability_scalar_names_sha256")
        != FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        or "semantic_direction" in core
    ):
        raise ValueError("factorized RADIO core contract differs")
    canonical = core.get("canonical_feature")
    log_amplitude = core.get("log_amplitude")
    valid = core.get("valid")
    reliability = core.get("reliability")
    if (
        not torch.is_tensor(canonical)
        or canonical.shape != (xyz.shape[0], 1280)
        or canonical.dtype != torch.float16
        or not torch.is_tensor(log_amplitude)
        or log_amplitude.shape != (xyz.shape[0],)
        or log_amplitude.dtype != torch.float32
        or not torch.is_tensor(valid)
        or valid.shape != (xyz.shape[0],)
        or valid.dtype != torch.bool
        or not torch.is_tensor(reliability)
        or reliability.shape
        != (xyz.shape[0], len(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES))
        or reliability.dtype != torch.float32
    ):
        raise ValueError("factorized RADIO output tensors differ")
    if not all(
        bool(torch.isfinite(item).all())
        for item in (xyz, canonical, log_amplitude, reliability)
    ):
        raise ValueError("factorized RADIO builder cache contains non-finite values")
    invalid = ~valid
    if (
        not bool((canonical[invalid] == 0).all())
        or not bool((log_amplitude[invalid] == 0).all())
        or not bool((reliability[invalid] == 0).all())
        or not bool((view_counts[invalid] == 0).all())
    ):
        raise ValueError("factorized RADIO invalid rows must be exactly zero")
    if builder_version == 1 and not bool((reliability[:, 4] == 0).all()):
        raise ValueError("factorized RADIO unavailable purity sentinel differs")
    if builder_version == 2 and (
        bool((reliability[:, 4] < 0).any())
        or bool((reliability[:, 4] > 1).any())
    ):
        raise ValueError("factorized RADIO measured purity bounds differ")
    expected_evidence = view_counts.float() / (view_counts.float() + 1.0)
    expected_evidence[invalid] = 0.0
    if not torch.equal(reliability[:, 3], expected_evidence):
        raise ValueError("factorized RADIO view-count evidence differs")
    for start in range(0, int(xyz.shape[0]), 4096):
        stop = min(start + 4096, int(xyz.shape[0]))
        chunk_valid = valid[start:stop]
        if not bool(chunk_valid.any()):
            continue
        active_canonical = canonical[start:stop][chunk_valid].float()
        active_log_amplitude = log_amplitude[start:stop][chunk_valid]
        active_reliability = reliability[start:stop][chunk_valid]
        canonical_norm = torch.linalg.vector_norm(active_canonical, dim=-1)
        if not torch.allclose(
            canonical_norm,
            torch.exp(active_log_amplitude),
            atol=2e-3,
            rtol=2e-3,
        ):
            raise ValueError(
                "factorized RADIO canonical amplitude reconstruction differs"
            )
        resultant = active_reliability[:, 0]
        dispersion = active_reliability[:, 1]
        amplitude_std = active_reliability[:, 2]
        evidence = active_reliability[:, 3]
        purity = active_reliability[:, 4]
        if (
            bool((resultant <= 0).any())
            or bool((resultant > 1).any())
            or bool((dispersion < 0).any())
            or bool((dispersion >= 1).any())
            or bool((amplitude_std < 0).any())
            or bool((evidence <= 0).any())
            or bool((evidence >= 1).any())
            or (builder_version == 1 and bool((purity != 0).any()))
            or (builder_version == 2 and bool((purity < 0).any()))
            or (builder_version == 2 and bool((purity > 1).any()))
        ):
            raise ValueError("factorized RADIO reliability bounds differ")
        if not torch.allclose(
            dispersion,
            1.0 - resultant,
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("factorized RADIO resultant/dispersion relation differs")
    purity_authority = metadata.get("visibility_purity_authority")
    expected_authority = {
        **(
            FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY
            if builder_version == 1
            else FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
        ),
        "registration_responsibility_cache_sha256": str(
            metadata.get("registration_responsibility_cache_sha256", "")
        ),
    }
    if (
        purity_authority != expected_authority
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(metadata.get("registration_responsibility_cache_sha256", "")),
        )
        is None
    ):
        raise ValueError("factorized RADIO visibility-purity authority differs")
    expected_builder_contract = (
        canonical_factorized_radio_builder_contract()
        if builder_version == 1
        else canonical_factorized_radio_builder_contract_v2()
    )
    expected_builder_contract_sha256 = (
        factorized_radio_builder_contract_sha256()
        if builder_version == 1
        else factorized_radio_builder_contract_v2_sha256()
    )
    if metadata.get("builder_contract") != expected_builder_contract or (
        metadata.get("builder_contract_sha256")
        != expected_builder_contract_sha256
    ):
        raise ValueError("factorized RADIO builder contract differs")
    fixed_metadata = {
        "construction": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        "feature_space": "radio",
        "input_feature_space": "radio_raw_full",
        "feature_dim": 1280,
        "max_views_authority": 120,
        "aggregation_mode": (
            "raster_gaussian_top1"
            if builder_version == 1
            else "raster_marginal_responsibility"
        ),
        "raster_view_fusion": "contribution_mean",
        "registration_weight_mode": (
            "alpha_depth"
            if builder_version == 1
            else "exact_front_to_back_marginal_responsibility"
        ),
        "semantic_direction_storage": "derived_from_canonical_feature_not_persisted",
        "canonical_feature_dtype": "float16",
        "log_amplitude_dtype": "float32",
        "reliability_dtype": "float32",
        "robust_mpr": False,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
        "query_independent": True,
    }
    mismatched_metadata = sorted(
        name
        for name, expected in fixed_metadata.items()
        if metadata.get(name) != expected
    )
    if mismatched_metadata:
        raise ValueError(
            "factorized RADIO fixed metadata differs: " f"{mismatched_metadata}"
        )
    if builder_version == 2:
        semantic_valid_count = int(
            metadata.get("semantic_valid_gaussian_count", -1)
        )
        geometric_valid_count = int(
            metadata.get("geometric_visible_gaussian_count", -1)
        )
        geometric_only_count = int(
            metadata.get("geometric_visible_semantic_invalid_gaussian_count", -1)
        )
        if (
            metadata.get("valid_semantics")
            != (
                "positive_raw_radio_amplitude_responsibility_mass_and_"
                "nonzero_direction_resultant"
            )
            or metadata.get("semantic_assignment_gate")
            != "pre_adaptor_raw_radio_l2_norm_strictly_positive"
            or metadata.get("view_count_semantics")
            != "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
            or metadata.get("geometric_visibility_semantics")
            != (
                "independent_exact_base_weight_authority_includes_"
                "zero_amplitude_hits"
            )
            or metadata.get("invalid_row_purity_policy")
            != "core_v1_requires_zero_for_semantically_invalid_rows"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(metadata.get("geometric_view_counts_sha256", "")),
            )
            is None
            or semantic_valid_count != int(valid.sum())
            or geometric_valid_count < semantic_valid_count
            or geometric_only_count != geometric_valid_count - semantic_valid_count
            or geometric_valid_count > int(xyz.shape[0])
        ):
            raise ValueError("factorized RADIO semantic/geometric support differs")
    declared_views = int(metadata.get("num_declared_views", 0))
    selected_dataset = metadata.get("selected_dataset_indices")
    selected_frames = metadata.get("selected_frame_indices")
    if (
        declared_views <= 0
        or declared_views > 120
        or not isinstance(selected_dataset, list)
        or not isinstance(selected_frames, list)
        or len(selected_dataset) != declared_views
        or len(selected_frames) != declared_views
        or len(set(selected_dataset)) != declared_views
        or len(set(selected_frames)) != declared_views
    ):
        raise ValueError("factorized RADIO selected-view authority differs")
    for name in (
        "geometry_checkpoint_sha256",
        "feature_frame_manifest_sha256",
        "feature_output_bundle_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(name, ""))) is None:
            raise ValueError(f"factorized RADIO {name} authority differs")


def _atomic_torch_save(payload: object, output: Path) -> None:
    """Publish one immutable torch output without replacing existing bytes."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        _commit_temporary_no_clobber(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _build_cache(args: argparse.Namespace) -> dict:
    factorized_radio_builder = validate_factorized_radio_builder_policy(args)
    observation_contract = None
    if str(getattr(args, "observation_contract", "legacy")) in {
        CANONICAL_OBSERVATION_CONTRACT_NAME,
        CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
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
    formal_builder = observation_contract is not None or factorized_radio_builder
    if (
        formal_builder
        and re.fullmatch(r"[0-9a-f]{64}", expected_feature_bundle_sha256) is None
    ):
        raise ValueError("formal MPR requires --expected-feature-output-bundle-sha256")
    if (
        formal_builder
        and re.fullmatch(r"[0-9a-f]{64}", expected_geometry_sha256) is None
    ):
        raise ValueError("formal MPR requires --expected-geometry-checkpoint-sha256")
    if expected_geometry_sha256 and (
        _sha256_file(args.checkpoint) != expected_geometry_sha256
    ):
        raise ValueError("geometry checkpoint differs from caller authority")
    if (
        formal_builder
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
            expected_checkpoint_sha256=(expected_geometry_sha256 or None),
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
        expected_output_bundle_sha256=str(expected_feature_bundle_sha256),
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
    capability_storage = str(getattr(args, "capability_storage", "dense")).lower()
    if capability_map_source not in {"project_raw", "official_extracted"}:
        raise ValueError(
            "capability_map_source must be project_raw or official_extracted"
        )
    if capability_storage not in {"dense", "channel_sharded"}:
        raise ValueError("capability_storage must be dense or channel_sharded")
    if capability_storage == "channel_sharded" and (
        feature_space not in {"radio", "dino_v3", "sam3"}
        or capability_map_source != "project_raw"
        or args.aggregation_mode
        not in {"raster_gaussian_top1", "raster_marginal_responsibility"}
        or args.raster_view_fusion != "contribution_mean"
        or (
            not str(args.responsibility_cache).strip()
            and not (
                feature_space == "radio" and str(args.save_responsibility_cache).strip()
            )
        )
    ):
        raise ValueError(
            "channel_sharded storage requires raw/DINO/SAM project_raw, "
            "top1 or exact-marginal contribution_mean, and an existing "
            "responsibility cache (raw RADIO may create it in the same run)"
        )
    selected_frame_indices = [int(dataset.frame_indices[index]) for index in selected]
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
    sharded_adaptor: torch.nn.Module | None = None
    if (
        feature_space in {"dino_v3", "sam3"}
        and capability_map_source == "official_extracted"
    ):
        expected_radio_checkpoint_sha256 = _sha256_file(args.radio_checkpoint)
        responsibility_cache_bytes = 0
        if str(args.responsibility_cache).strip():
            responsibility_path = Path(args.responsibility_cache).expanduser()
            if responsibility_path.is_file():
                responsibility_cache_bytes = 2 * responsibility_path.stat().st_size
        capability_cpu_memory_preflight = estimate_capability_mpr_cpu_bytes(
            num_views=len(selected),
            channels=int(_EXTRACTED_CAPABILITY_SPECS[feature_space]["output_dim"]),
            height=feature_height,
            width=feature_width,
            num_gaussians=int(model.get_xyz().shape[0]),
            aggregation_mode=str(args.aggregation_mode),
            raster_view_fusion=str(args.raster_view_fusion),
            raster_topk=int(args.raster_topk),
            raster_channel_chunk_size=int(args.raster_channel_chunk_size),
            responsibility_cache_bytes=(responsibility_cache_bytes),
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
                {"capability_cpu_memory_preflight": (capability_cpu_memory_preflight)}
            ),
            flush=True,
        )
        if available_memory > 0 and capability_cpu_memory_preflight[
            "estimated_peak_bytes"
        ] > available_memory * float(
            getattr(
                args,
                "max_estimated_cpu_memory_fraction",
                0.85,
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
            expected_radio_checkpoint_sha256=(expected_radio_checkpoint_sha256),
            expected_scene=str(getattr(args, "expected_feature_scene", "")),
            expected_image_dir=str(getattr(args, "expected_feature_image_dir", "")),
            expected_output_bundle_sha256=str(
                feature_bundle_validation["output_bundle_sha256"]
            ),
        )
        adaptor_name = str(extracted_source["adaptor_name"])
        adaptor_checkpoint_path = str(extracted_source["radio_checkpoint"])
        adaptor_checkpoint_sha256 = str(extracted_source["radio_checkpoint_sha256"])
        adaptor_checkpoint_provenance = str(
            extracted_source["radio_checkpoint_provenance"]
        )
        capability_source_metadata = {
            "capability_map_source": "official_extracted",
            "capability_native_map_manifest": str(extracted_source["frame_manifest"]),
            "capability_native_map_manifest_sha256": str(
                extracted_source["frame_manifest_sha256"]
            ),
            "capability_native_map_grid": list(extracted_source["native_grid"]),
            "capability_native_map_scene": str(extracted_source["scene"]),
            "capability_native_map_image_dir": str(extracted_source["image_dir"]),
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
    elif capability_storage != "channel_sharded" and not factorized_radio_builder:
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
    else:
        teacher_maps = None
    if feature_space == "siglip_summary":
        summary_head_path = str(Path(args.summary_head_weights).expanduser().resolve())
        summary_head = SigLIP2SummaryHead.from_extracted_weights(summary_head_path).to(
            device
        )
        summary_head.eval()
        projected_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, teacher_maps.shape[0], int(args.projection_batch_size)),
                desc="project teacher query space",
            ):
                maps = teacher_maps[start : start + int(args.projection_batch_size)].to(
                    device
                )
                batch, channels, height, width = maps.shape
                tokens = maps.permute(0, 2, 3, 1).reshape(
                    batch, height * width, channels
                )
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
        adaptor = (
            load_radio_adaptor_from_checkpoint(
                adaptor_checkpoint_path,
                adaptor_name,
                kind="feature_projection",
                expected_sha256=adaptor_checkpoint_sha256,
            )
            .to(device)
            .eval()
        )
        adaptor.requires_grad_(False)
        if capability_storage == "channel_sharded":
            sharded_adaptor = adaptor
        else:
            assert teacher_maps is not None
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
    geometric_responsibility_assignments: list[
        dict[str, torch.Tensor]
    ] | None = None
    geometric_view_counts: torch.Tensor | None = None
    responsibility_cache_path = ""
    responsibility_cache_sha256 = ""
    responsibility_contract: dict = {}
    depth_maps = None
    alpha_maps = None
    if args.responsibility_cache or args.save_responsibility_cache:
        if args.responsibility_cache and args.save_responsibility_cache:
            raise ValueError(
                "load and save responsibility cache options are mutually exclusive"
            )
        if args.aggregation_mode == "raster_gaussian_top1":
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
        elif args.aggregation_mode == "raster_marginal_responsibility":
            responsibility_contract = _sparse_exact_marginal_authority_metadata(
                args=args,
                selected=selected,
                selected_frame_indices=selected_frame_indices,
                poses=poses,
                renderer=renderer,
                model=model,
                feature_height=feature_height,
                feature_width=feature_width,
            )
            expected_authority_sha256 = str(
                getattr(args, "expected_responsibility_cache_sha256", "")
            )
            authority_source = str(args.responsibility_cache).strip()
            if args.save_responsibility_cache:
                authority_source = str(args.save_responsibility_cache).strip()
                writer = SparseExactMarginalAuthorityWriter(
                    authority_source,
                    metadata=responsibility_contract,
                    frame_indices=selected_frame_indices,
                    num_gaussians=int(xyz_cpu.shape[0]),
                    num_pixels=feature_height * feature_width,
                )
                with torch.inference_mode():
                    for view_index in tqdm(
                        range(len(selected)),
                        desc="freeze exact sparse marginal responsibility",
                    ):
                        if view_index in writer.completed_view_indices:
                            continue
                        hits = rasterize_single_view_contributions(
                            model,
                            renderer,
                            poses[view_index].to(device),
                            height=feature_height,
                            width=feature_width,
                        )
                        writer.add_view(
                            view_index,
                            hits["gaussian_ids"],
                            hits["pixel_ids"],
                            hits["weights"],
                        )
                        del hits
                authority_path, expected_authority_sha256 = writer.finalize()
                authority_source = str(authority_path)
            (
                sparse_assignments,
                responsibility_cache_sha256,
                sparse_authority_path,
            ) = load_sparse_exact_marginal_authority(
                authority_source,
                expected_metadata=responsibility_contract,
                expected_frame_indices=selected_frame_indices,
                num_gaussians=int(xyz_cpu.shape[0]),
                num_pixels=feature_height * feature_width,
                expected_sha256=expected_authority_sha256,
            )
            responsibility_cache_path = str(sparse_authority_path)
            responsibility_assignments = [
                {**assignment, "weights": assignment["marginal_weights"]}
                for assignment in sparse_assignments
            ]
            geometric_responsibility_assignments = responsibility_assignments
        else:
            raise ValueError(
                "shared responsibility authorities support top1 or exact marginal"
            )

    if (
        args.aggregation_mode == "raster_marginal_responsibility"
        and responsibility_assignments is not None
    ):
        if geometric_responsibility_assignments is None:
            geometric_responsibility_assignments = responsibility_assignments
        if not factorized_radio_builder:
            responsibility_assignments, geometric_view_counts = (
                _gate_exact_marginal_assignments_by_raw_amplitude(
                    feature_dir=feature_dir,
                    selected_frame_indices=selected_frame_indices,
                    tensor_records=feature_tensor_records,
                    feature_size=(feature_height, feature_width),
                    responsibility_assignments=geometric_responsibility_assignments,
                    num_gaussians=int(xyz_cpu.shape[0]),
                )
            )

    if factorized_radio_builder:
        if responsibility_assignments is None or not responsibility_cache_sha256:
            raise RuntimeError(
                "canonical-factorized-radio-v1 did not reopen its frozen "
                "responsibility authority"
            )
        factorized_cpu_memory_preflight: dict[str, int | float] = {
            **estimate_factorized_radio_cpu_bytes(
                num_gaussians=int(xyz_cpu.shape[0]),
                feature_dim=1280,
                feature_height=feature_height,
                feature_width=feature_width,
                observation_chunk_size=max(1, int(args.point_chunk_size)),
                row_chunk_size=max(1, int(args.point_chunk_size)),
            ),
            "available_memory_bytes": _available_cpu_memory_bytes(),
            "maximum_fraction": float(args.max_estimated_cpu_memory_fraction),
        }
        authority_tensor_bytes = 0
        for assignment in responsibility_assignments:
            authority_keys = (
                ("gaussian_ids", "pixel_ids", "base_weights", "marginal_weights")
                if "base_weights" in assignment
                else ("gaussian_ids", "pixel_ids", "weights")
            )
            authority_tensor_bytes += sum(
                int(assignment[key].numel()) * int(assignment[key].element_size())
                for key in authority_keys
            )
        factorized_cpu_memory_preflight["responsibility_assignments_cpu"] = int(
            authority_tensor_bytes
        )
        factorized_cpu_memory_preflight["estimated_peak_bytes"] = int(
            factorized_cpu_memory_preflight["estimated_peak_bytes"]
        ) + int(authority_tensor_bytes)
        print(
            json.dumps(
                {
                    "factorized_radio_cpu_memory_preflight": (
                        factorized_cpu_memory_preflight
                    )
                }
            ),
            flush=True,
        )
        if int(factorized_cpu_memory_preflight["available_memory_bytes"]) > 0 and int(
            factorized_cpu_memory_preflight["estimated_peak_bytes"]
        ) > int(factorized_cpu_memory_preflight["available_memory_bytes"]) * float(
            factorized_cpu_memory_preflight["maximum_fraction"]
        ):
            raise MemoryError(
                "estimated factorized RADIO CPU peak exceeds the configured "
                "fraction of available memory"
            )
        factorized_builder_v2 = (
            args.aggregation_mode == "raster_marginal_responsibility"
        )
        factorized_core, factorized_view_counts = _stream_factorized_radio_from_bundle(
            feature_dir=feature_dir,
            selected_frame_indices=selected_frame_indices,
            tensor_records=feature_tensor_records,
            feature_size=(feature_height, feature_width),
            responsibility_assignments=responsibility_assignments,
            num_gaussians=int(xyz_cpu.shape[0]),
            observation_chunk_size=max(1, int(args.point_chunk_size)),
            row_chunk_size=max(1, int(args.point_chunk_size)),
            visibility_purity_measured=factorized_builder_v2,
        )
        if factorized_builder_v2:
            authority_view_counts = torch.zeros(
                int(xyz_cpu.shape[0]), dtype=torch.long
            )
            for assignment in geometric_responsibility_assignments or []:
                frame_support = torch.zeros(
                    int(xyz_cpu.shape[0]), dtype=torch.bool
                )
                frame_support[assignment["gaussian_ids"].long()] = True
                authority_view_counts[frame_support] += 1
            if bool((factorized_view_counts > authority_view_counts).any()):
                raise ValueError(
                    "factorized positive-amplitude supervision exceeds exact "
                    "marginal geometric visibility"
                )
        builder_contract = (
            canonical_factorized_radio_builder_contract_v2()
            if factorized_builder_v2
            else canonical_factorized_radio_builder_contract()
        )
        builder_contract_sha256 = (
            factorized_radio_builder_contract_v2_sha256()
            if factorized_builder_v2
            else factorized_radio_builder_contract_sha256()
        )
        builder_cache_schema = (
            CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA_V2
            if factorized_builder_v2
            else CANONICAL_FACTORIZED_RADIO_BUILDER_CACHE_SCHEMA
        )
        builder_cache_schema_version = 2 if factorized_builder_v2 else 1
        factorized_metadata: dict[str, object] = {
            "builder_contract": builder_contract,
            "builder_contract_sha256": builder_contract_sha256,
            "construction": CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
            "feature_space": "radio",
            "input_feature_space": "radio_raw_full",
            "feature_dim": 1280,
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "geometry_checkpoint_sha256": str(expected_geometry_sha256),
            "feature_frame_manifest": str(
                (feature_dir / "frame_manifest.json").resolve()
            ),
            "feature_frame_manifest_sha256": str(
                feature_bundle_validation["manifest_sha256"]
            ),
            "feature_output_bundle_sha256": str(
                feature_bundle_validation["output_bundle_sha256"]
            ),
            "selected_dataset_indices": list(selected),
            "selected_frame_indices": list(selected_frame_indices),
            "num_declared_views": len(selected_frame_indices),
            "max_views_authority": 120,
            "aggregation_mode": str(args.aggregation_mode),
            "raster_view_fusion": "contribution_mean",
            "registration_weight_mode": str(args.registration_weight_mode),
            "registration_responsibility_cache": responsibility_cache_path,
            "registration_responsibility_cache_sha256": (responsibility_cache_sha256),
            "registration_responsibility_contract": responsibility_contract,
            "observation_lifting_contract": canonical_factorized_radio_contract(),
            "observation_lifting_contract_sha256": (
                CANONICAL_FACTORIZED_RADIO_CONTRACT_SHA256
            ),
            "per_pixel_observation": (
                "raw_radio_then_strict_positive_norm_direction_and_log_amplitude"
            ),
            "zero_amplitude_policy": "excluded_before_division_or_logarithm",
            "semantic_direction_storage": (
                "derived_from_canonical_feature_not_persisted"
            ),
            "canonical_feature_dtype": "float16",
            "log_amplitude_dtype": "float32",
            "reliability_dtype": "float32",
            "robust_mpr": False,
            "factorized_cpu_memory_preflight": factorized_cpu_memory_preflight,
            "visibility_purity_authority": {
                **(
                    FACTORIZED_RADIO_EXACT_MARGINAL_PURITY_AUTHORITY
                    if factorized_builder_v2
                    else FACTORIZED_RADIO_TOP1_PURITY_AUTHORITY
                ),
                "registration_responsibility_cache_sha256": (
                    responsibility_cache_sha256
                ),
            },
            "benchmark_masks_opened": False,
            "benchmark_images_opened": False,
            "text_queries_opened": False,
            "query_independent": True,
        }
        if factorized_builder_v2:
            factorized_valid_for_metadata = factorized_core.get("valid")
            if not torch.is_tensor(factorized_valid_for_metadata):
                raise RuntimeError("factorized semantic validity is missing")
            geometric_valid = authority_view_counts > 0
            factorized_metadata.update(
                {
                    "shared_registration_responsibility": True,
                    "valid_semantics": (
                        "positive_raw_radio_amplitude_responsibility_mass_and_"
                        "nonzero_direction_resultant"
                    ),
                    "semantic_assignment_gate": (
                        "pre_adaptor_raw_radio_l2_norm_strictly_positive"
                    ),
                    "view_count_semantics": (
                        "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
                    ),
                    "geometric_visibility_semantics": (
                        "independent_exact_base_weight_authority_includes_"
                        "zero_amplitude_hits"
                    ),
                    "geometric_view_counts_sha256": _sha256_int64_vector(
                        authority_view_counts
                    ),
                    "geometric_visible_gaussian_count": int(geometric_valid.sum()),
                    "semantic_valid_gaussian_count": int(
                        factorized_valid_for_metadata.sum()
                    ),
                    "geometric_visible_semantic_invalid_gaussian_count": int(
                        (geometric_valid & ~factorized_valid_for_metadata).sum()
                    ),
                    "invalid_row_purity_policy": (
                        "core_v1_requires_zero_for_semantically_invalid_rows"
                    ),
                }
            )
        factorized_payload: dict[str, object] = {
            "schema": builder_cache_schema,
            "schema_version": builder_cache_schema_version,
            "xyz": xyz_cpu,
            "geometry_fingerprint": {
                "num_gaussians": int(xyz_cpu.shape[0]),
                "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
            },
            "factorized_radio": factorized_core,
            "view_counts": factorized_view_counts,
            "metadata": factorized_metadata,
        }
        validate_factorized_radio_builder_payload(factorized_payload)
        output = Path(args.output).expanduser()
        _atomic_torch_save(factorized_payload, output)
        factorized_valid = factorized_core["valid"]
        assert torch.is_tensor(factorized_valid)
        positive = factorized_view_counts[factorized_valid]
        report: dict[str, object] = {
            "output": str(output),
            "schema": builder_cache_schema,
            "num_gaussians": int(xyz_cpu.shape[0]),
            "num_views": len(selected_frame_indices),
            "valid_count": int(factorized_valid.sum()),
            "valid_ratio": float(factorized_valid.float().mean()),
            "mean_views_if_valid": (
                float(positive.float().mean()) if positive.numel() else 0.0
            ),
            "median_views_if_valid": (
                float(positive.float().median()) if positive.numel() else 0.0
            ),
            "max_views": int(positive.max()) if positive.numel() else 0,
            "metadata": factorized_metadata,
        }
        _write_json_noclobber(output.with_suffix(output.suffix + ".json"), report)
        return report

    if (
        capability_storage == "channel_sharded"
        and responsibility_assignments is None
        and str(args.save_responsibility_cache).strip()
    ):
        resumed_responsibility = _load_saved_responsibility_for_sharded_resume(
            output=args.output,
            save_path=args.save_responsibility_cache,
            expected_contract=responsibility_contract,
            num_gaussians=int(xyz_cpu.shape[0]),
        )
        if resumed_responsibility is not None:
            (
                responsibility_assignments,
                responsibility_cache_sha256,
                resumed_responsibility_path,
            ) = resumed_responsibility
            responsibility_cache_path = str(resumed_responsibility_path)

    if (
        capability_storage == "channel_sharded"
        and responsibility_assignments is None
        and str(args.save_responsibility_cache).strip()
    ):
        from radio_gs.scripts.eval_lerf_direct_3d_selection import (
            rasterize_registered_view_assignments,
        )

        depth_parts: list[torch.Tensor] = []
        alpha_parts: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(selected), int(args.render_batch_size)),
                desc="render sharded responsibility visibility",
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
        captured_assignments: list[dict[str, torch.Tensor]] = []
        with torch.inference_mode():
            for view_index in tqdm(
                range(len(selected)), desc="freeze sharded responsibility"
            ):
                gaussian_ids, pixel_ids, weights = (
                    rasterize_registered_view_assignments(
                        model=model,
                        renderer=renderer,
                        viewmat=poses[view_index].to(device),
                        image_height=feature_height,
                        image_width=feature_width,
                        depth_map=depth_maps[view_index : view_index + 1].to(device),
                        alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                        registration_depth_tolerance=float(args.depth_tolerance),
                        registration_relative_depth_tolerance=float(
                            args.relative_depth_tolerance
                        ),
                        registration_alpha_threshold=float(args.alpha_threshold),
                        registration_weight_mode=args.registration_weight_mode,
                        gaussian_top1=True,
                    )
                )
                captured_assignments.append(
                    {
                        "gaussian_ids": gaussian_ids.int().cpu(),
                        "pixel_ids": pixel_ids.int().cpu(),
                        "weights": weights.float().cpu(),
                    }
                )
        responsibility_output = Path(args.save_responsibility_cache).expanduser()
        responsibility_output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{responsibility_output.name}.",
            suffix=".tmp",
            dir=responsibility_output.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(
                {
                    "schema_version": 1,
                    "metadata": responsibility_contract,
                    "assignments": captured_assignments,
                },
                temporary,
            )
            os.replace(temporary, responsibility_output)
        finally:
            temporary.unlink(missing_ok=True)
        responsibility_cache_path = str(responsibility_output.resolve())
        responsibility_cache_sha256 = _sha256_file(responsibility_output)
        responsibility_assignments = captured_assignments
        depth_maps = None
        alpha_maps = None

    sharded_records: list[dict[str, object]] | None = None
    visibility_purity: torch.Tensor | None = None
    if capability_storage == "channel_sharded":
        if responsibility_assignments is None or not responsibility_cache_sha256:
            raise ValueError(
                "channel-sharded MPR requires a validated responsibility cache"
            )
        sharded_output_dim = (
            1280
            if feature_space == "radio"
            else int(_EXTRACTED_CAPABILITY_SPECS[feature_space]["output_dim"])
        )
        sharded_resume_contract: dict[str, object] = {
            "schema": "radio_gs.channel_sharded_mpr_resume.v1",
            "feature_space": feature_space,
            "feature_dim": sharded_output_dim,
            "num_gaussians": int(xyz_cpu.shape[0]),
            "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
            "selected_frame_indices": selected_frame_indices,
            "feature_output_bundle_sha256": str(
                feature_bundle_validation["output_bundle_sha256"]
            ),
            "geometry_checkpoint_sha256": str(
                expected_geometry_sha256 or _sha256_file(args.checkpoint)
            ),
            "registration_responsibility_cache_sha256": (responsibility_cache_sha256),
            "official_adaptor_checkpoint_sha256": adaptor_checkpoint_sha256,
            "aggregation_mode": str(args.aggregation_mode),
            "registration_weight_mode": str(args.registration_weight_mode),
            "raster_view_fusion": str(args.raster_view_fusion),
            "normalize_each_view": bool(args.normalize_each_view),
            "raster_reliability_mode": str(args.raster_reliability_mode),
            "shard_channels": int(args.capability_shard_channels),
            "inner_channel_chunk_size": int(args.raster_channel_chunk_size),
        }
        sharded_records, valid, view_counts, reliability = (
            _stream_channel_sharded_contribution_mean(
                output=Path(args.output).expanduser(),
                feature_space=feature_space,
                feature_dir=feature_dir,
                feature_tensor_records=feature_tensor_records,
                selected_frame_indices=selected_frame_indices,
                feature_size=(feature_height, feature_width),
                responsibility_assignments=responsibility_assignments,
                num_gaussians=int(xyz_cpu.shape[0]),
                output_dim=sharded_output_dim,
                shard_channels=int(args.capability_shard_channels),
                inner_channel_chunk_size=int(args.raster_channel_chunk_size),
                point_chunk_size=int(args.point_chunk_size),
                num_views=len(selected),
                normalize_each_view=bool(args.normalize_each_view),
                reliability_mode=str(args.raster_reliability_mode),
                adaptor=sharded_adaptor,
                device=device,
                resume_contract=sharded_resume_contract,
            )
        )
        if args.aggregation_mode == "raster_marginal_responsibility":
            visibility_purity = _exact_marginal_visibility_purity(
                responsibility_assignments,
                num_gaussians=int(xyz_cpu.shape[0]),
                valid=valid,
                geometric_assignments=geometric_responsibility_assignments,
            )
        if sharded_adaptor is not None:
            del sharded_adaptor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Visibility is already encoded in a loaded sidecar.  Otherwise render it
    # once, and optionally freeze the resulting registration assignments for
    # all feature spaces.
    if (
        capability_storage != "channel_sharded"
        and responsibility_assignments is None
        and args.aggregation_mode
        not in {
            "raster_marginal_responsibility",
            "raster_exact_center_uncertainty",
        }
    ):
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
    if capability_storage == "channel_sharded":
        features = None
    elif args.aggregation_mode == "center":
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
                    points.shape[0],
                    teacher_maps.shape[1],
                    device=device,
                    dtype=torch.float32,
                )
                view_counts = torch.zeros(
                    points.shape[0], device=device, dtype=torch.long
                )
                for view_start in range(0, len(selected), view_chunk):
                    view_stop = min(view_start + view_chunk, len(selected))
                    targets, _valid, counts = sample_multiview_radio_targets(
                        points,
                        teacher_maps[view_start:view_stop].to(device),
                        poses[view_start:view_stop].to(device),
                        renderer.scaled_intrinsics(
                            feature_width, feature_height
                        ).float(),
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
                        view_target, view_valid, _view_count = (
                            sample_multiview_radio_targets(
                                points,
                                teacher_maps[view_index : view_index + 1].to(device),
                                poses[view_index : view_index + 1].to(device),
                                renderer.scaled_intrinsics(
                                    feature_width, feature_height
                                ).float(),
                                depth_map=depth_maps[view_index : view_index + 1].to(
                                    device
                                ),
                                alpha_map=alpha_maps[view_index : view_index + 1].to(
                                    device
                                ),
                                depth_tolerance=float(args.depth_tolerance),
                                relative_depth_tolerance=float(
                                    args.relative_depth_tolerance
                                ),
                                alpha_threshold=float(args.alpha_threshold),
                                normalize_sampled_features=bool(
                                    args.normalize_each_view
                                ),
                            )
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
        marginal_visible_mass = (
            torch.zeros(xyz_cpu.shape[0], dtype=torch.float32)
            if args.aggregation_mode
            in {
                "raster_marginal_responsibility",
                "raster_exact_center_uncertainty",
            }
            else None
        )
        marginal_pure_mass = (
            torch.zeros(xyz_cpu.shape[0], dtype=torch.float32)
            if marginal_visible_mass is not None
            else None
        )
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
                    view_features = prepare_raster_view_features(
                        teacher_maps[view_index : view_index + 1].to(
                            device=device, dtype=torch.float32
                        ),
                        normalize_each_view=bool(args.normalize_each_view),
                    )
                    frame_sum, frame_counts = raster_adjoint_registered_view_features(
                        model=model,
                        renderer=renderer,
                        viewmat=poses[view_index].to(device),
                        siglip_feat=view_features,
                        alpha_map=alpha_maps[view_index : view_index + 1].to(device),
                        alpha_threshold=float(args.alpha_threshold),
                        channel_chunk_size=int(args.adjoint_channel_chunk_size),
                    )
                else:
                    if args.aggregation_mode in {
                        "raster_marginal_responsibility",
                        "raster_exact_center_uncertainty",
                    }:
                        if responsibility_assignments is not None:
                            assignment = responsibility_assignments[view_index]
                            gaussian_ids = assignment["gaussian_ids"].to(device)
                            pixel_ids = assignment["pixel_ids"].to(device)
                            raw_weights = assignment["base_weights"].to(device)
                            target_weights = assignment["marginal_weights"].to(device)
                            hits = None
                            marginal = None
                        else:
                            hits = rasterize_single_view_contributions(
                                model,
                                renderer,
                                poses[view_index].to(device),
                                height=feature_height,
                                width=feature_width,
                            )
                            gaussian_ids = hits["gaussian_ids"]
                            pixel_ids = hits["pixel_ids"]
                            raw_weights = hits["weights"]
                            marginal = marginal_responsibility_statistics(
                                pixel_ids,
                                raw_weights,
                                num_pixels=feature_height * feature_width,
                            )
                            target_weights = marginal.target_weight
                        weights = (
                            target_weights
                            if args.aggregation_mode == "raster_marginal_responsibility"
                            else raw_weights
                        )
                        raw_frame_mass = torch.zeros(
                            xyz_cpu.shape[0], dtype=torch.float32, device=device
                        )
                        pure_frame_mass = torch.zeros_like(raw_frame_mass)
                        raw_frame_mass.index_add_(
                            0, gaussian_ids.long(), raw_weights.float()
                        )
                        pure_frame_mass.index_add_(
                            0, gaussian_ids.long(), target_weights.float()
                        )
                        assert marginal_visible_mass is not None
                        assert marginal_pure_mass is not None
                        marginal_visible_mass.add_(raw_frame_mass.cpu())
                        marginal_pure_mass.add_(pure_frame_mass.cpu())
                        del (
                            raw_frame_mass,
                            pure_frame_mass,
                            raw_weights,
                            target_weights,
                        )
                        if hits is not None:
                            del hits
                        if marginal is not None:
                            del marginal
                    elif responsibility_assignments is not None:
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
                                depth_map=depth_maps[view_index : view_index + 1].to(
                                    device
                                ),
                                alpha_map=alpha_maps[view_index : view_index + 1].to(
                                    device
                                ),
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
                    view_features = prepare_raster_view_features(
                        teacher_maps[view_index : view_index + 1].to(
                            device=device, dtype=torch.float32
                        ),
                        normalize_each_view=bool(args.normalize_each_view),
                    )
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
        if (
            args.save_responsibility_cache
            and args.aggregation_mode == "raster_gaussian_top1"
        ):
            if len(captured_assignments) != len(selected):
                raise RuntimeError("failed to capture every registration view")
            responsibility_output = Path(args.save_responsibility_cache).expanduser()
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
        visibility_purity = None
        if marginal_visible_mass is not None:
            assert marginal_pure_mass is not None
            visibility_purity = torch.where(
                marginal_visible_mass > 0,
                marginal_pure_mass / marginal_visible_mass.clamp_min(1e-12),
                torch.zeros_like(marginal_visible_mass),
            ).clamp(0.0, 1.0)
            visibility_purity[~valid] = 0.0
        if (
            args.aggregation_mode == "raster_marginal_responsibility"
            and responsibility_assignments is not None
            and geometric_responsibility_assignments is not None
        ):
            visibility_purity = _exact_marginal_visibility_purity(
                responsibility_assignments,
                num_gaussians=int(xyz_cpu.shape[0]),
                valid=valid,
                geometric_assignments=geometric_responsibility_assignments,
            )
        del registered_sum, registered_counts
        del marginal_visible_mass, marginal_pure_mass
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
        "marginal_responsibility_contract": (
            MARGINAL_RESPONSIBILITY_CONTRACT
            if args.aggregation_mode == "raster_marginal_responsibility"
            else (
                EXACT_CENTER_UNCERTAINTY_CONTRACT
                if args.aggregation_mode == "raster_exact_center_uncertainty"
                else "not_applicable"
            )
        ),
        "visibility_uncertainty_semantics": (
            "per_primitive_sum_weight_times_responsibility_over_sum_weight"
            if args.aggregation_mode
            in {
                "raster_marginal_responsibility",
                "raster_exact_center_uncertainty",
            }
            else "not_available"
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
        "feature_frame_manifest": str((feature_dir / "frame_manifest.json").resolve()),
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
            else (
                "sampled_feature_before_center_fusion"
                if args.normalize_each_view
                else "disabled"
            )
        ),
        "robust_mpr": bool(args.robust_mpr and args.aggregation_mode == "center"),
        "robust_temperature": float(args.robust_temperature),
        "robust_iterations": int(args.robust_iterations),
        "summary_head_weights": summary_head_path,
        "official_adaptor_name": adaptor_name,
        "official_adaptor_checkpoint": adaptor_checkpoint_path,
        "official_adaptor_checkpoint_sha256": adaptor_checkpoint_sha256,
        "official_adaptor_checkpoint_provenance": (adaptor_checkpoint_provenance),
        **capability_source_metadata,
        "registration_responsibility_cache": responsibility_cache_path,
        "registration_responsibility_cache_sha256": responsibility_cache_sha256,
        "shared_registration_responsibility": bool(responsibility_cache_sha256),
        "registration_responsibility_contract": responsibility_contract,
        "capability_projection_before_mpr": feature_space in {"dino_v3", "sam3"},
        "capability_cpu_memory_preflight": (capability_cpu_memory_preflight),
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
    if (
        args.aggregation_mode == "raster_marginal_responsibility"
        and geometric_view_counts is not None
    ):
        geometric_valid = geometric_view_counts > 0
        metadata.update(
            {
                "valid_semantics": (
                    "positive_pre_adaptor_raw_radio_amplitude_responsibility_mass"
                ),
                "semantic_assignment_gate": (
                    "pre_adaptor_raw_radio_l2_norm_strictly_positive"
                ),
                "view_count_semantics": (
                    "views_with_pre_adaptor_raw_radio_l2_norm_strictly_positive"
                ),
                "geometric_visibility_semantics": (
                    "independent_exact_base_weight_authority_includes_"
                    "zero_amplitude_hits"
                ),
                "geometric_view_counts_sha256": _sha256_int64_vector(
                    geometric_view_counts
                ),
                "geometric_visible_gaussian_count": int(geometric_valid.sum()),
                "semantic_valid_gaussian_count": int(valid.sum()),
                "geometric_visible_semantic_invalid_gaussian_count": int(
                    (geometric_valid & ~valid.cpu()).sum()
                ),
                "invalid_row_purity_policy": (
                    "mpr_schema_v1_requires_zero_for_semantically_invalid_rows"
                ),
                "visibility_purity_storage": (
                    "report_summary_only"
                    if capability_storage == "channel_sharded"
                    else "output_tensor"
                ),
            }
        )
    if observation_contract is not None:
        metadata["observation_lifting_contract"] = observation_contract
        metadata["observation_lifting_contract_sha256"] = observation_contract_sha256(
            observation_contract
        )
    if capability_storage == "channel_sharded":
        metadata["feature_storage"] = "channel_sharded_fp16_row_major"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry_fingerprint = {
        "num_gaussians": int(xyz_cpu.shape[0]),
        "xyz_sha256": _sha256_tensor_rows(xyz_cpu),
    }
    if capability_storage == "channel_sharded":
        assert sharded_records is not None
        support_path = output.parent / f"{output.name}.support.pt"
        support_payload = {
            "xyz": xyz_cpu,
            "valid": valid,
            "view_counts": view_counts,
            "reliability": reliability,
        }
        if support_path.exists():
            existing_support, _support_sha256, _support_source = load_torch_mapping(
                support_path,
                map_location="cpu",
                label="resumed channel-sharded MPR support",
            )
            if set(existing_support) != set(support_payload) or any(
                not torch.equal(existing_support[name], expected)
                for name, expected in support_payload.items()
            ):
                raise ValueError("existing channel-sharded support differs")
        else:
            _atomic_torch_save(support_payload, support_path)
        manifest: dict[str, object] = {
            "schema": "radio_gs.channel_sharded_mpr.v1",
            "schema_version": 1,
            "layout": "row_major_channel_shards",
            "feature_dtype": "float16",
            "feature_shape": [
                int(xyz_cpu.shape[0]),
                (
                    1280
                    if feature_space == "radio"
                    else int(_EXTRACTED_CAPABILITY_SPECS[feature_space]["output_dim"])
                ),
            ],
            "support": {
                "relative_path": support_path.name,
                "sha256": _sha256_file(support_path),
            },
            "shards": sharded_records,
            "geometry_fingerprint": geometry_fingerprint,
            "metadata": metadata,
        }
        _write_json_noclobber(output, manifest)
        manifest_sha256 = _sha256_file(output)
        load_mpr_cache(
            output,
            expected_sha256=manifest_sha256,
            expected_feature_space=feature_space,
            require_reliability=True,
            require_formal_safety=observation_contract is not None,
        )
    else:
        output_payload = {
            "xyz": xyz_cpu,
            "geometry_fingerprint": geometry_fingerprint,
            "features": features,
            "valid": valid,
            "view_counts": view_counts,
            "reliability": reliability,
            "metadata": metadata,
        }
        if args.aggregation_mode in {
            "raster_marginal_responsibility",
            "raster_exact_center_uncertainty",
        }:
            if visibility_purity is None:
                raise RuntimeError("marginal visibility purity was not constructed")
            output_payload["visibility_purity"] = visibility_purity.half()
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
            _commit_temporary_no_clobber(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    positive = view_counts[valid]
    report = {
        "output": str(output),
        "num_gaussians": int(xyz_cpu.shape[0]),
        "num_views": len(selected),
        "valid_count": int(valid.sum()),
        "valid_ratio": float(valid.float().mean()),
        "mean_views_if_valid": (
            float(positive.float().mean()) if positive.numel() else 0.0
        ),
        "median_views_if_valid": (
            float(positive.float().median()) if positive.numel() else 0.0
        ),
        "max_views": int(positive.max()) if positive.numel() else 0,
        "metadata": metadata,
    }
    if args.aggregation_mode in {
        "raster_marginal_responsibility",
        "raster_exact_center_uncertainty",
    }:
        active_purity = visibility_purity[valid].float()
        report["visibility_purity"] = {
            "mean": float(active_purity.mean()) if active_purity.numel() else 0.0,
            "median": (float(active_purity.median()) if active_purity.numel() else 0.0),
            "minimum": float(active_purity.min()) if active_purity.numel() else 0.0,
            "maximum": float(active_purity.max()) if active_purity.numel() else 0.0,
        }
    if capability_storage == "channel_sharded":
        report["feature_storage"] = metadata["feature_storage"]
        report["channel_shards"] = len(sharded_records or [])
    report_path = output.with_suffix(output.suffix + ".json")
    _write_json_noclobber(report_path, report)
    return report


def build_cache(args: argparse.Namespace) -> dict:
    """Preflight immutable final outputs before any expensive scene work."""

    output = Path(args.output).expanduser()
    report = output.with_suffix(output.suffix + ".json")
    existing = [path for path in (output, report) if path.exists()]
    if existing:
        raise FileExistsError(
            "immutable cache output/report already exists: "
            + ", ".join(str(path) for path in existing)
        )
    return _build_cache(args)


def main(argv: list[str] | None = None) -> None:
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
            CANONICAL_EXACT_MARGINAL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V2_CONTRACT_NAME,
            CANONICAL_FULL_OBSERVATION_V3_CONTRACT_NAME,
            CANONICAL_FACTORIZED_RADIO_CONTRACT_NAME,
        ],
        default=CANONICAL_OBSERVATION_CONTRACT_NAME,
        help=(
            "Versioned query-free lifting policy. canonical-exact-marginal-mpr-v1 "
            "binds one shared sparse exact-compositor authority; canonical-full-observation "
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
        "--capability-storage",
        choices=["dense", "channel_sharded"],
        default="dense",
        help=(
            "Keep the historical dense .pt default, or write a fail-closed "
            "JSON manifest plus row-major fp16 channel shards. The sharded "
            "mode is also valid for raw RADIO and requires a frozen "
            "responsibility sidecar."
        ),
    )
    parser.add_argument(
        "--capability-shard-channels",
        type=int,
        default=256,
        help="Outer feature-channel width for disk-backed MPR shards.",
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
        choices=[
            "center",
            "raster_gaussian_top1",
            "raster_adjoint",
            "raster_marginal_responsibility",
            "raster_exact_center_uncertainty",
        ],
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
    args = parser.parse_args(argv)
    try:
        validate_raster_reliability_policy(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(build_cache(args), indent=2))


if __name__ == "__main__":
    main()
