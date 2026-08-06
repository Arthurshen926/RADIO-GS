#!/usr/bin/env python3
"""
Extract and save RADIO features for all training images in a scene.

Usage:
    python radio_gs/scripts/extract_radio_features.py \
        --scene room_0 \
        --image_dir dataset/room_0/Sequence_1/rgb/ \
        --output_dir output/radio_features/room_0/ \
        --radio_repo /root/RADIO \
        --radio_version c-radio_v4-h \
        --batch_size 4 \
        --extract_adaptors
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.data.benchmark_paths import extract_feature_frame_index
from radio_gs.data.view_split import (
    load_excluded_image_stems,
    select_image_indices,
)
from radio_gs.utils.immutable_artifacts import (
    load_fixed_radio_checkpoint_payload,
    load_json_object,
    load_torch_payload,
    sha256_file as _stable_sha256_file,
    stable_descriptor_load,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRAME_ID_MODES = ("auto", "source_rank")
RESUME_CONTRACT_FILENAME = ".extract_resume_contract.json"
FRAME_COMMIT_DIRNAME = ".extract_frame_commits"
RESUME_CONTRACT_SCHEMA_VERSION = 2
FRAME_COMMIT_SCHEMA_VERSION = 1
OUTPUT_BUNDLE_SCHEMA_VERSION = 1
LEGACY_RESEAL_CONTRACT = "radio-feature-legacy-tensor-reseal-v1"
INCOMPLETE_RUNTIME_RESEAL_CONTRACT = (
    "radio-feature-incomplete-runtime-tensor-reseal-v1"
)
LEGACY_SOURCE_MANIFEST_FILENAME = "frame_manifest.legacy.json"


def _sha256_file(path: str | Path) -> str:
    return _stable_sha256_file(path)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _python_source_tree_fingerprint(root: str | Path) -> dict[str, object]:
    """Content-address every Python source below *root* in stable path order."""

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Python source tree is missing: {source_root}")
    paths = sorted(
        (path for path in source_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not paths:
        raise ValueError(f"Python source tree has no .py files: {source_root}")
    files = [
        {
            "relative_path": path.relative_to(source_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    return {
        "contract": "ordered-relative-python-source-tree-sha256-v1",
        "root": str(source_root),
        "file_count": len(files),
        "files": files,
        "tree_sha256": _canonical_json_sha256({"files": files}),
    }


def _runtime_fingerprint(device: torch.device) -> dict[str, object]:
    """Record numerical-runtime state that can affect resumed frame tensors."""

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return ""

    driver_path = Path("/proc/driver/nvidia/version")
    driver_record: dict[str, str] = {"version_text": "", "sha256": ""}
    if driver_path.is_file() and not driver_path.is_symlink():
        driver_text, driver_digest, _ = stable_descriptor_load(
            driver_path,
            lambda handle: handle.read().decode("utf-8", errors="replace"),
            label="NVIDIA driver version",
        )
        driver_record = {
            "version_text": driver_text.splitlines()[0],
            "sha256": driver_digest,
        }

    cuda_backend = torch.backends.cuda
    matmul_backend = cuda_backend.matmul
    result: dict[str, object] = {
        "contract": "radio-extraction-runtime-fingerprint-v2",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "numpy_version": str(np.__version__),
        "pillow_version": str(PIL.__version__),
        "timm_version": package_version("timm"),
        "einops_version": package_version("einops"),
        "nvidia_driver": driver_record,
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            getattr(matmul_backend, "allow_tf32", False)
        ),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
            getattr(
                matmul_backend,
                "allow_fp16_reduced_precision_reduction",
                False,
            )
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            getattr(
                matmul_backend,
                "allow_bf16_reduced_precision_reduction",
                False,
            )
        ),
        "cuda_sdpa_flash_enabled": bool(
            getattr(cuda_backend, "flash_sdp_enabled", lambda: False)()
        ),
        "cuda_sdpa_mem_efficient_enabled": bool(
            getattr(cuda_backend, "mem_efficient_sdp_enabled", lambda: False)()
        ),
        "cuda_sdpa_math_enabled": bool(
            getattr(cuda_backend, "math_sdp_enabled", lambda: False)()
        ),
        "cuda_sdpa_cudnn_enabled": bool(
            getattr(cuda_backend, "cudnn_sdp_enabled", lambda: False)()
        ),
        "cudnn_allow_tf32": bool(
            getattr(torch.backends.cudnn, "allow_tf32", False)
        ),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": str(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        ),
        "effective_device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["cuda_device"] = {
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
    else:
        result["cuda_device"] = None
    result["fingerprint_sha256"] = _canonical_json_sha256(result)
    return result


def _atomic_json_write(path: str | Path, payload: object) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(value: object, path: str | Path) -> None:
    """Save a torch artifact without ever exposing a partial target file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # A file object gives Torch a stable internal archive name ("archive")
        # instead of embedding the random temporary basename, so identical
        # tensors have reproducible serialized SHA256 values across retries.
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_record(
    output_root: Path,
    relative_path: str,
    value: torch.Tensor,
) -> dict[str, object]:
    path = output_root / relative_path
    return {
        "relative_path": relative_path,
        "sha256": _sha256_file(path),
        "dtype": _dtype_name(value),
        "shape": [int(dimension) for dimension in value.shape],
        "num_bytes": int(value.nelement() * value.element_size()),
    }


def _load_validated_tensor(
    path: Path,
    record: dict[str, object],
) -> torch.Tensor:
    try:
        value, _digest, _source = load_torch_payload(
            path,
            expected_sha256=str(record.get("sha256", "")),
            map_location="cpu",
            label="committed feature tensor",
        )
    except Exception as exc:
        raise ValueError(f"tensor file cannot be reopened: {path}") from exc
    if not torch.is_tensor(value):
        raise ValueError(f"committed artifact is not a tensor: {path}")
    if _dtype_name(value) != str(record.get("dtype", "")):
        raise ValueError(f"tensor dtype differs: {path}")
    if [int(dimension) for dimension in value.shape] != record.get("shape"):
        raise ValueError(f"tensor shape differs: {path}")
    expected_bytes = int(value.nelement() * value.element_size())
    if expected_bytes != int(record.get("num_bytes", -1)):
        raise ValueError(f"tensor logical byte size differs: {path}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"committed tensor contains non-finite values: {path}")
    return value


def _thermal_pause(
    device: torch.device,
    seconds_per_image: float,
) -> None:
    """Synchronize a CUDA frame before an execution-only cooling pause."""

    seconds = float(seconds_per_image)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("thermal pacing seconds must be finite and non-negative")
    if seconds == 0:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    time.sleep(seconds)


def _radio_model_source(
    radio_version: str,
    radio_checkpoint: str = "",
) -> dict[str, str]:
    """Resolve an optional content-addressed RADIO checkpoint for extraction."""

    raw_checkpoint = str(radio_checkpoint or "").strip()
    if not raw_checkpoint:
        return {
            "load_source": str(radio_version),
            "checkpoint": "",
            "checkpoint_sha256": "",
            "checkpoint_provenance": "unverified_torch_hub_version",
            "checkpoint_load_contract": "unverified_torch_hub_version",
        }
    checkpoint = Path(raw_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RADIO checkpoint is missing: {checkpoint}")
    return {
        "load_source": str(checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_provenance": "explicit_file_sha256",
        "checkpoint_load_contract": (
            "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
        ),
    }


# ---- image loading helpers ------------------------------------------------

def _collect_image_paths(image_dir: str) -> tuple[list[Path], str]:
    """Return image paths with numeric frame-order when indices are parseable."""
    paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    indexed: list[tuple[int, Path]] = []
    for path in paths:
        try:
            indexed.append((extract_feature_frame_index(path), path))
        except ValueError:
            indexed = []
            break

    if indexed:
        # Numeric suffixes are not unique in several NVOS captures (for
        # example multiple Horns filenames end in ``_427``).  A filename
        # tiebreaker makes source_rank stable across processes/filesystems.
        indexed.sort(key=lambda item: (item[0], item[1].name))
        return [path for _, path in indexed], "numeric_then_exact_filename"

    return sorted(paths, key=lambda path: path.name), "lexicographic_filename"


def _apply_subsampling(
    image_paths: list[Path],
    frame_stride: int,
    max_frames: int | None,
) -> list[Path]:
    sampled = image_paths[:: max(1, frame_stride)]
    if max_frames is not None:
        sampled = sampled[:max_frames]
    return sampled


def _saved_frame_indices(
    image_paths: list[Path],
    *,
    mode: str,
) -> list[int]:
    """Resolve output ids once and reject any would-be cache overwrite."""
    if mode not in FRAME_ID_MODES:
        raise ValueError(f"frame_id_mode must be one of {FRAME_ID_MODES}")
    frame_indices: list[int] = []
    by_index: dict[int, Path] = {}
    for source_rank, source_path in enumerate(image_paths):
        if mode == "source_rank":
            frame_idx = source_rank
        else:
            try:
                frame_idx = extract_feature_frame_index(source_path)
            except ValueError:
                frame_idx = source_rank
        if frame_idx in by_index:
            raise ValueError(
                f"Feature output collision at rgb_{frame_idx}.pt: "
                f"{by_index[frame_idx].name} and {source_path.name}. "
                "Use --frame-id-mode source_rank for a unique dense index."
            )
        by_index[frame_idx] = source_path
        frame_indices.append(frame_idx)
    return frame_indices


def _nearest_radio_resolution(h: int, w: int, patch_size: int = 16) -> tuple[int, int]:
    """Round h, w to nearest multiples of *patch_size*."""
    return (
        max(patch_size, round(h / patch_size) * patch_size),
        max(patch_size, round(w / patch_size) * patch_size),
    )


def _compute_scaled_radio_resolution(
    h: int,
    w: int,
    resolution_scale: float,
    patch_size: int = 16,
) -> tuple[int, int]:
    """Scale an image size and snap it to RADIO patch multiples."""
    if resolution_scale <= 0:
        raise ValueError("--resolution_scale must be positive")
    return _nearest_radio_resolution(
        h * resolution_scale,
        w * resolution_scale,
        patch_size=patch_size,
    )


def _load_and_preprocess(
    paths: list[Path],
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Load images, resize, normalize to [0, 1] and stack → (B, 3, H, W)."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0)
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _sliding_starts(length: int, tile_size: int, tile_overlap: int) -> list[int]:
    """Return start indices that cover *length* with overlapping tiles."""
    if tile_size <= 0:
        raise ValueError("--tile_size must be positive")
    if tile_overlap < 0:
        raise ValueError("--tile_overlap must be non-negative")
    if tile_overlap >= tile_size:
        raise ValueError("--tile_overlap must be smaller than --tile_size")
    if length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _stitch_sliding_window_features(
    output_shape: tuple[int, int, int, int] | torch.Size,
    tiles: list[tuple[int, int, torch.Tensor]],
) -> torch.Tensor:
    """Average overlapping feature tiles into one feature map.

    Tile coordinates are in feature-grid units, not input-pixel units.
    """
    if not tiles:
        raise ValueError("No sliding-window tiles to stitch")
    first = tiles[0][2]
    accum = first.new_zeros(tuple(output_shape), dtype=torch.float32)
    weight = first.new_zeros(tuple(output_shape), dtype=torch.float32)

    for top, left, feat in tiles:
        _, _, h, w = feat.shape
        accum[:, :, top : top + h, left : left + w] += feat.float()
        weight[:, :, top : top + h, left : left + w] += 1.0

    return accum / weight.clamp_min(1.0)


def _parse_adaptor_names(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names


def _adaptor_output_subdir(name: str) -> str:
    if name == "siglip2-g":
        return "siglip2"
    return name.replace("-", "_")


def _split_radio_output_pair(value) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(value):
        return None, value
    if isinstance(value, dict):
        summary = value.get("summary")
        features = value.get("features")
        if features is None:
            features = value.get("spatial")
        if features is None:
            raise ValueError("RADIO output dict must contain features/spatial")
        return summary, features
    if hasattr(value, "summary") and hasattr(value, "features"):
        return value.summary, value.features
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    raise TypeError(f"Unsupported RADIO output value type: {type(value)!r}")


def _spatial_to_feature_grid(
    spatial: torch.Tensor,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    if spatial.ndim == 4:
        return spatial
    if spatial.ndim != 3:
        raise ValueError(f"Expected RADIO spatial features as [B,N,D] or [B,D,H,W], got {tuple(spatial.shape)}")
    B, _, D = spatial.shape
    return spatial.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)


def _unpack_radio_output(
    output,
    patch_h: int,
    patch_w: int,
    adaptor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Convert RADIO output into summary, backbone grid, adaptor grids."""
    if isinstance(output, dict):
        if "backbone" not in output:
            raise ValueError("RADIO dict output is missing the 'backbone' entry")
        summary, spatial = _split_radio_output_pair(output["backbone"])
        adaptor_outputs = {
            name: value for name, value in output.items() if name != "backbone"
        }
    elif isinstance(output, tuple) and len(output) == 2:
        summary, spatial = _split_radio_output_pair(output)
        adaptor_outputs = {}
    else:
        summary, spatial, adaptor_outputs = output

    spatial_2d = _spatial_to_feature_grid(spatial, patch_h, patch_w)

    adaptor_2d: dict[str, torch.Tensor] = {}
    for name in adaptor_names or []:
        if not adaptor_outputs or name not in adaptor_outputs:
            continue
        ad_out = adaptor_outputs[name]
        _, ad_spatial = _split_radio_output_pair(ad_out)
        adaptor_2d[name] = _spatial_to_feature_grid(ad_spatial, patch_h, patch_w)
    return summary, spatial_2d, adaptor_2d


def _run_radio_batch(
    model,
    conditioner,
    imgs: torch.Tensor,
    amp: bool,
    patch_h: int,
    patch_w: int,
    adaptor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    imgs = conditioner(imgs)
    with torch.cuda.amp.autocast(enabled=amp):
        output = model(imgs)
    return _unpack_radio_output(output, patch_h, patch_w, adaptor_names=adaptor_names)


def _extract_sliding_window_single(
    model,
    conditioner,
    img: torch.Tensor,
    amp: bool,
    tile_size: int,
    tile_overlap: int,
    patch_size: int = 16,
    adaptor_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Extract and stitch features for a single preprocessed image tensor."""
    if img.shape[0] != 1:
        raise ValueError("Sliding-window extraction expects single-image batches")

    _, _, target_h, target_w = img.shape
    tile_size = _nearest_radio_resolution(tile_size, tile_size, patch_size)[0]
    tile_overlap = max(0, round(tile_overlap / patch_size) * patch_size)

    if target_h <= tile_size and target_w <= tile_size:
        return _run_radio_batch(
            model,
            conditioner,
            img,
            amp,
            target_h // patch_size,
            target_w // patch_size,
            adaptor_names=adaptor_names,
        )

    row_tile = min(tile_size, target_h)
    col_tile = min(tile_size, target_w)
    row_overlap = min(tile_overlap, max(0, row_tile - patch_size))
    col_overlap = min(tile_overlap, max(0, col_tile - patch_size))
    row_starts = _sliding_starts(target_h, row_tile, row_overlap)
    col_starts = _sliding_starts(target_w, col_tile, col_overlap)
    summaries: list[torch.Tensor] = []
    backbone_tiles: list[tuple[int, int, torch.Tensor]] = []
    adaptor_tiles: dict[str, list[tuple[int, int, torch.Tensor]]] = {}

    for top in row_starts:
        for left in col_starts:
            bottom = min(top + row_tile, target_h)
            right = min(left + col_tile, target_w)
            top = bottom - row_tile
            left = right - col_tile
            tile = img[:, :, top:bottom, left:right]
            patch_h = tile.shape[-2] // patch_size
            patch_w = tile.shape[-1] // patch_size
            summary, spatial_2d, adaptors = _run_radio_batch(
                model,
                conditioner,
                tile,
                amp,
                patch_h,
                patch_w,
                adaptor_names=adaptor_names,
            )
            summaries.append(summary)
            grid_top = top // patch_size
            grid_left = left // patch_size
            backbone_tiles.append((grid_top, grid_left, spatial_2d))
            for name, value in adaptors.items():
                adaptor_tiles.setdefault(name, []).append((grid_top, grid_left, value))

    full_shape = (
        1,
        backbone_tiles[0][2].shape[1],
        target_h // patch_size,
        target_w // patch_size,
    )
    spatial_full = _stitch_sliding_window_features(full_shape, backbone_tiles)
    adaptor_full = {
        name: _stitch_sliding_window_features(
            (1, tiles[0][2].shape[1], target_h // patch_size, target_w // patch_size),
            tiles,
        )
        for name, tiles in adaptor_tiles.items()
    }
    return torch.stack(summaries, dim=0).mean(dim=0), spatial_full, adaptor_full


# ---- model loading --------------------------------------------------------

def _load_radio_model(
    radio_repo: str,
    radio_version: str,
    adaptor_names: list[str] | None,
    device: torch.device,
    *,
    expected_checkpoint_sha256: str = "",
):
    """Load RADIO while preventing TorchHub from unpickling the checkpoint.

    The upstream local hub entrypoint unconditionally calls
    ``torch.load(..., weights_only=False)`` for explicit files.  Formal runs
    therefore deserialize the externally SHA-bound checkpoint first through
    our restricted allowlist, then temporarily intercept exactly that one hub
    load and return the already-validated payload.  Any other deserialization
    attempted while the hub entrypoint executes fails closed.
    """
    kwargs: dict = {"source": "local"}
    if adaptor_names:
        kwargs["adaptor_names"] = adaptor_names
    checkpoint = Path(radio_version).expanduser()
    if checkpoint.is_file():
        expected = str(expected_checkpoint_sha256 or "")
        payload, observed, source = load_fixed_radio_checkpoint_payload(
            checkpoint,
            expected_sha256=expected,
            map_location="cpu",
            label="formal RADIO extraction checkpoint",
        )
        if observed != expected:
            raise ValueError("formal RADIO checkpoint authority differs")
        original_torch_load = torch.load
        intercepted = 0

        def restricted_hub_load(candidate, *args, **load_kwargs):
            nonlocal intercepted
            if isinstance(candidate, (str, os.PathLike)) and Path(
                candidate
            ).expanduser().resolve() == source:
                intercepted += 1
                if intercepted != 1:
                    raise RuntimeError(
                        "RADIO hub attempted to deserialize its checkpoint more than once"
                    )
                return payload
            raise RuntimeError(
                "RADIO hub attempted an unapproved torch.load during formal loading"
            )

        torch.load = restricted_hub_load
        try:
            model = torch.hub.load(
                radio_repo,
                "radio_model",
                version=str(source),
                **kwargs,
            )
        finally:
            torch.load = original_torch_load
        if intercepted != 1:
            raise RuntimeError(
                "RADIO hub did not consume the restricted checkpoint payload"
            )
    else:
        if expected_checkpoint_sha256:
            raise ValueError(
                "a trusted RADIO checkpoint SHA-256 requires an explicit file"
            )
        model = torch.hub.load(
            radio_repo,
            "radio_model",
            version=radio_version,
            **kwargs,
        )
    model = model.to(device).eval()

    input_conditioner = model.make_preprocessor_external()

    return model, input_conditioner


# ---- PCA statistics -------------------------------------------------------

def _compute_pca_stats(
    all_features: list[torch.Tensor],
    n_components: int = 64,
) -> dict[str, torch.Tensor]:
    """Compute mean, std, and top-*n_components* PCA components from features.

    Args:
        all_features: list of (C, Hp, Wp) tensors (float32).
        n_components: number of PCA components to keep.

    Returns:
        dict with 'mean' (C,), 'std' (C,), 'components_64' (n_components, C).
    """
    # Stack all spatial tokens: (N_total, C)
    pixels = torch.cat([f.reshape(f.shape[0], -1).T for f in all_features], dim=0)
    mean = pixels.mean(dim=0)
    std = pixels.std(dim=0).clamp(min=1e-6)
    centered = pixels - mean
    # Economy SVD — only need top components
    k = min(n_components, centered.shape[0], centered.shape[1])
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:k]  # (k, C)
    return {"mean": mean, "std": std, "components_64": components}


def _feature_signature(
    backbone: torch.Tensor,
    summary: torch.Tensor,
    adaptors: dict[str, torch.Tensor],
    adaptor_names: list[str] | None,
    *,
    require_all_adaptors: bool,
) -> dict[str, object]:
    if backbone.ndim != 3 or backbone.dtype != torch.float16:
        raise ValueError(
            "per-frame RADIO backbone must be float16 [D,H,W], got "
            f"{backbone.dtype} {tuple(backbone.shape)}"
        )
    if summary.ndim != 1 or summary.dtype != torch.float32:
        raise ValueError(
            "per-frame RADIO summary must be float32 [D], got "
            f"{summary.dtype} {tuple(summary.shape)}"
        )
    if not torch.isfinite(backbone).all() or not torch.isfinite(summary).all():
        raise ValueError("per-frame RADIO backbone/summary contains non-finite values")
    adaptor_records: list[dict[str, object]] = []
    for name in adaptor_names or []:
        value = adaptors.get(name)
        if value is None:
            if require_all_adaptors:
                raise ValueError(
                    f"RADIO did not return requested adaptor output {name!r}"
                )
            adaptor_records.append(
                {
                    "name": name,
                    "subdir": _adaptor_output_subdir(name),
                    "dim": None,
                    "grid": None,
                    "dtype": "float16",
                }
            )
            continue
        if value.ndim != 3 or value.dtype != torch.float16:
            raise ValueError(
                f"per-frame RADIO adaptor {name!r} must be float16 [D,H,W], "
                f"got {value.dtype} {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"per-frame RADIO adaptor {name!r} contains non-finite values"
            )
        adaptor_records.append(
            {
                "name": name,
                "subdir": _adaptor_output_subdir(name),
                "dim": int(value.shape[0]),
                "grid": [int(value.shape[1]), int(value.shape[2])],
                "dtype": "float16",
            }
        )
    return {
        "backbone": {
            "subdir": "backbone",
            "dim": int(backbone.shape[0]),
            "grid": [int(backbone.shape[1]), int(backbone.shape[2])],
            "dtype": "float16",
        },
        "summary": {
            "subdir": "summary",
            "dim": int(summary.shape[-1]),
            "dtype": "float32",
        },
        "adaptors": adaptor_records,
    }


def _expected_tensor_relative_paths(
    stem: str,
    adaptor_names: list[str] | None,
) -> list[str]:
    paths = [f"backbone/{stem}.pt", f"summary/{stem}.pt"]
    paths.extend(
        f"{_adaptor_output_subdir(name)}/{stem}.pt"
        for name in (adaptor_names or [])
    )
    return paths


def _signature_from_committed_tensors(
    values: dict[str, torch.Tensor],
    stem: str,
    adaptor_names: list[str] | None,
) -> dict[str, object]:
    adaptors = {
        name: values[f"{_adaptor_output_subdir(name)}/{stem}.pt"]
        for name in (adaptor_names or [])
    }
    return _feature_signature(
        values[f"backbone/{stem}.pt"],
        values[f"summary/{stem}.pt"],
        adaptors,
        adaptor_names,
        require_all_adaptors=True,
    )


def _merge_feature_signature(
    current: dict[str, object] | None,
    observed: dict[str, object],
) -> dict[str, object]:
    if current is not None and current != observed:
        raise ValueError("RADIO feature dtype/shape contract differs across frames")
    return observed if current is None else current


def _resume_contract_payload(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model_source: dict[str, str],
    adaptor_names: list[str] | None,
    image_paths: list[Path],
    image_sort_mode: str,
    source_image_count: int,
    excluded_image_stems: tuple[str, ...],
    excluded_image_names: list[str],
    frame_records: list[dict[str, object]],
    target_h: int,
    target_w: int,
    pacing_seconds: float,
    radio_source_tree: dict[str, object],
    runtime_fingerprint: dict[str, object],
) -> dict[str, object]:
    radio_repo = Path(args.radio_repo).expanduser().resolve()
    hubconf = radio_repo / "hubconf.py"
    frames = []
    if len(image_paths) != len(frame_records):
        raise ValueError("resume contract frame identities are incomplete")
    for source_path, frame_record in zip(image_paths, frame_records):
        frames.append(
            {
                **frame_record,
                "source_path": str(source_path.resolve()),
            }
        )
    return {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "contract": "radio-feature-extraction-resume-v1",
        "extractor_sha256": _sha256_file(Path(__file__).resolve()),
        "scene": str(args.scene),
        "radio": {
            "version": str(args.radio_version),
            "repo": str(radio_repo),
            "repo_hubconf_sha256": _sha256_file(hubconf) if hubconf.is_file() else "",
            "checkpoint": model_source["checkpoint"],
            "checkpoint_sha256": model_source["checkpoint_sha256"],
            "checkpoint_provenance": model_source["checkpoint_provenance"],
            "checkpoint_load_contract": model_source[
                "checkpoint_load_contract"
            ],
            "requested_adaptors": list(adaptor_names or []),
            "python_source_tree": radio_source_tree,
        },
        "runtime": runtime_fingerprint,
        "input": {
            "image_dir": str(Path(args.image_dir).resolve()),
            "image_sort_mode": image_sort_mode,
            "source_image_count_before_exclusion": int(source_image_count),
            "excluded_image_stems": list(excluded_image_stems),
            "excluded_image_names": list(excluded_image_names),
            "frames": frames,
        },
        "configuration": {
            "batch_size": int(args.batch_size),
            "frame_stride": int(args.frame_stride),
            "max_frames": (
                int(args.max_frames) if args.max_frames is not None else None
            ),
            "frame_id_mode": str(getattr(args, "frame_id_mode", "auto")),
            "resolution_scale": float(args.resolution_scale),
            "radio_input_resolution_hw": [int(target_h), int(target_w)],
            "sliding_window": bool(args.sliding_window),
            "tile_size": int(args.tile_size) if args.sliding_window else None,
            "tile_overlap": int(args.tile_overlap) if args.sliding_window else None,
            "amp": bool(args.amp),
            "skip_pca_stats": bool(args.skip_pca_stats),
            "effective_device": str(device),
            "resume_partial": True,
            "radio_thermal_pacing_seconds_per_image": float(pacing_seconds),
        },
        "commit_protocol": {
            "tensor_commit": "same_directory_temp_then_os_replace_v1",
            "frame_commit": "validated_tensor_set_then_atomic_json_marker_v1",
            "committed_frame_validation": (
                "same_fd_sha256_weights_only_dtype_shape_finite_v2"
            ),
            "invalid_or_missing_frame_policy": "recompute_entire_frame_v1",
            "pacing_order": "frame_commit_then_cuda_synchronize_then_sleep_v1",
        },
    }


def _prepare_resume_contract(
    output_root: Path,
    payload: dict[str, object],
) -> tuple[Path, str]:
    contract_path = output_root / RESUME_CONTRACT_FILENAME
    if contract_path.exists():
        try:
            existing, _digest, _source = load_json_object(
                contract_path,
                label="RADIO extraction resume contract",
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"resume contract is unreadable; refusing loose resume: {contract_path}"
            ) from exc
        if existing != payload:
            raise ValueError(
                "resume contract differs from the requested extraction; "
                f"refusing to reuse {output_root}"
            )
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError(
                "partial output has no resume contract; refusing loose resume: "
                f"{output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(contract_path, payload)
    return contract_path, _canonical_json_sha256(payload)


def _frame_commit_path(output_root: Path, stem: str) -> Path:
    return output_root / FRAME_COMMIT_DIRNAME / f"{stem}.json"


def _validated_frame_snapshot(
    *,
    output_root: Path,
    frame_record: dict[str, object],
    adaptor_names: list[str] | None,
    resume_contract_sha256: str,
    expected_bundle_record: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, int, str]:
    stem = str(frame_record["saved_stem"])
    marker_path = _frame_commit_path(output_root, stem)
    if not marker_path.exists():
        return None, 0, "commit marker is missing"
    try:
        marker, marker_sha256, _source = load_json_object(
            marker_path,
            label="RADIO frame commit marker",
        )
    except (OSError, ValueError):
        return None, 0, "commit marker is unreadable"
    if not isinstance(marker, dict):
        return None, 0, "commit marker is not an object"
    if marker.get("schema_version") != FRAME_COMMIT_SCHEMA_VERSION:
        return None, 0, "commit marker schema differs"
    if marker.get("resume_contract_sha256") != resume_contract_sha256:
        return None, 0, "commit marker belongs to another resume contract"
    if marker.get("frame") != frame_record:
        return None, 0, "commit marker frame identity differs"
    tensor_records = marker.get("tensors")
    if not isinstance(tensor_records, list):
        return None, 0, "commit marker tensor set is invalid"
    expected_paths = _expected_tensor_relative_paths(stem, adaptor_names)
    actual_paths = [
        str(record.get("relative_path", ""))
        for record in tensor_records
        if isinstance(record, dict)
    ]
    if len(actual_paths) != len(tensor_records) or actual_paths != expected_paths:
        return None, 0, "commit marker tensor set differs"
    values: dict[str, torch.Tensor] = {}
    try:
        for record in tensor_records:
            relative_path = str(record["relative_path"])
            values[relative_path] = _load_validated_tensor(
                output_root / relative_path,
                record,
            )
        signature = _signature_from_committed_tensors(
            values,
            stem,
            adaptor_names,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, 0, str(exc)
    if marker.get("feature_signature") != signature:
        return None, 0, "commit marker feature signature differs"
    try:
        total_bytes = sum(int(record["num_bytes"]) for record in tensor_records)
    except (KeyError, TypeError, ValueError) as exc:
        return None, 0, f"commit marker logical byte count is invalid: {exc}"
    snapshot = {
        "frame": frame_record,
        "marker_relative_path": str(marker_path.relative_to(output_root)),
        "marker_sha256": marker_sha256,
        "feature_signature": signature,
        "tensors": tensor_records,
    }
    if expected_bundle_record is not None and snapshot != expected_bundle_record:
        return None, 0, "committed frame differs from the final output bundle"
    return snapshot, total_bytes, ""


def _validate_committed_frame(
    *,
    output_root: Path,
    frame_record: dict[str, object],
    adaptor_names: list[str] | None,
    resume_contract_sha256: str,
    expected_bundle_record: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, int, str]:
    snapshot, logical_bytes, reason = _validated_frame_snapshot(
        output_root=output_root,
        frame_record=frame_record,
        adaptor_names=adaptor_names,
        resume_contract_sha256=resume_contract_sha256,
        expected_bundle_record=expected_bundle_record,
    )
    if snapshot is None:
        return None, 0, reason
    return dict(snapshot["feature_signature"]), logical_bytes, ""


def _declared_output_bundle(
    manifest: dict[str, object],
    *,
    frame_records: list[dict[str, object]],
    resume_contract_sha256: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    bundle = manifest.get("output_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("final feature manifest has no output bundle")
    if (
        bundle.get("schema_version") != OUTPUT_BUNDLE_SCHEMA_VERSION
        or bundle.get("contract") != "radio-feature-output-bundle-v1"
        or bundle.get("resume_contract_sha256") != resume_contract_sha256
    ):
        raise ValueError("final feature output bundle contract differs")
    if manifest.get("output_bundle_sha256") != _canonical_json_sha256(bundle):
        raise ValueError("final feature output bundle SHA256 differs")
    records = bundle.get("frames")
    if not isinstance(records, list) or len(records) != len(frame_records):
        raise ValueError("final feature output bundle frame count differs")
    expected_by_stem: dict[str, dict[str, object]] = {}
    for frame_record, record in zip(frame_records, records):
        if not isinstance(record, dict) or record.get("frame") != frame_record:
            raise ValueError("final feature output bundle frame order differs")
        stem = str(frame_record["saved_stem"])
        if stem in expected_by_stem:
            raise ValueError("final feature output bundle repeats a frame")
        expected_by_stem[stem] = record
    return bundle, expected_by_stem


def _build_output_bundle(
    *,
    output_root: Path,
    frame_records: list[dict[str, object]],
    adaptor_names: list[str] | None,
    resume_contract_sha256: str,
) -> tuple[dict[str, object], dict[str, object], int]:
    snapshots: list[dict[str, object]] = []
    signature: dict[str, object] | None = None
    total_bytes = 0
    for frame_record in frame_records:
        snapshot, logical_bytes, reason = _validated_frame_snapshot(
            output_root=output_root,
            frame_record=frame_record,
            adaptor_names=adaptor_names,
            resume_contract_sha256=resume_contract_sha256,
        )
        if snapshot is None:
            raise ValueError(
                f"cannot finalize {frame_record['saved_stem']}: {reason}"
            )
        signature = _merge_feature_signature(
            signature,
            dict(snapshot["feature_signature"]),
        )
        total_bytes += logical_bytes
        snapshots.append(snapshot)
    if signature is None:
        raise ValueError("cannot finalize an empty feature output bundle")
    bundle = {
        "schema_version": OUTPUT_BUNDLE_SCHEMA_VERSION,
        "contract": "radio-feature-output-bundle-v1",
        "resume_contract_sha256": resume_contract_sha256,
        "frames": snapshots,
    }
    return bundle, signature, total_bytes


def _validate_resealed_legacy_output_bundle(
    root: Path,
    manifest: dict[str, object],
    manifest_sha256: str,
    *,
    verify_source_images: bool,
    expected_output_bundle_sha256: str | None,
) -> dict[str, object]:
    """Validate a content-addressed wrapper around immutable legacy tensors.

    This contract does not invent missing extraction-runtime provenance.  It
    proves only that the pre-existing manifest, source-image identities, and
    every declared tensor are exactly the files sealed into ``output_bundle``.
    """

    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("feature reseal execution contract is missing")
    formalization_contract = str(execution.get("formalization_contract", ""))
    if formalization_contract == LEGACY_RESEAL_CONTRACT:
        source_name_key = "legacy_source_manifest"
        source_sha_key = "legacy_source_manifest_sha256"
        source_bundle_sha_key = "legacy_source_manifest_sha256"
        source_name = str(execution.get(source_name_key, ""))
        if source_name != LEGACY_SOURCE_MANIFEST_FILENAME:
            raise ValueError("legacy feature reseal source path differs")
    elif formalization_contract == INCOMPLETE_RUNTIME_RESEAL_CONTRACT:
        source_name_key = "incomplete_runtime_source_manifest"
        source_sha_key = "incomplete_runtime_source_manifest_sha256"
        source_bundle_sha_key = "incomplete_runtime_source_manifest_sha256"
        source_name = str(execution.get(source_name_key, ""))
        expected_suffix = f".{execution.get(source_sha_key, '')}.json"
        if (
            Path(source_name).name != source_name
            or not source_name.startswith("frame_manifest.original.")
            or not source_name.endswith(expected_suffix)
        ):
            raise ValueError(
                "incomplete-runtime feature reseal source path is not content-addressed"
            )
    else:
        raise ValueError("feature reseal execution contract differs")
    legacy_path = root / source_name
    try:
        legacy, legacy_sha256, _source = load_json_object(
            legacy_path,
            expected_sha256=str(
                execution.get(source_sha_key, "")
            ),
            label="legacy RADIO feature manifest",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("legacy RADIO source manifest is unreadable") from exc
    if formalization_contract == LEGACY_RESEAL_CONTRACT:
        if any(
            key in legacy
            for key in ("execution", "output_bundle", "output_bundle_sha256")
        ):
            raise ValueError("legacy RADIO source manifest is not an unsealed source")
    else:
        source_execution = legacy.get("execution")
        if not isinstance(source_execution, dict):
            raise ValueError("incomplete-runtime source execution is missing")
        if (
            str(source_execution.get("resume_contract", ""))
            or str(source_execution.get("resume_contract_sha256", ""))
            or str(source_execution.get("resume_contract_file_sha256", ""))
            or legacy.get("output_bundle") is not None
            or str(legacy.get("output_bundle_sha256", ""))
        ):
            raise ValueError(
                "incomplete-runtime source is not an unbundled completed extraction"
            )
        if execution.get("original_extraction_execution") != source_execution:
            raise ValueError(
                "incomplete-runtime reseal does not preserve extraction execution"
            )
        if legacy_path.stat().st_mode & 0o222:
            raise ValueError("incomplete-runtime source manifest is not read-only")
    frame_records = legacy.get("frames")
    radio = legacy.get("radio")
    if not isinstance(frame_records, list) or not frame_records:
        raise ValueError("legacy RADIO source manifest has no frames")
    if not isinstance(radio, dict):
        raise ValueError("legacy RADIO source manifest has no RADIO declaration")
    if int(legacy.get("num_frames", -1)) != len(frame_records):
        raise ValueError("legacy RADIO source manifest frame count differs")
    adaptor_names_value = radio.get("requested_adaptors", [])
    if not isinstance(adaptor_names_value, list) or any(
        not isinstance(value, str) for value in adaptor_names_value
    ):
        raise ValueError("legacy RADIO adaptor declaration is invalid")
    adaptor_names = [str(value) for value in adaptor_names_value]

    bundle = manifest.get("output_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("resealed legacy manifest has no output bundle")
    if (
        bundle.get("schema_version") != OUTPUT_BUNDLE_SCHEMA_VERSION
        or bundle.get("contract") != "radio-feature-output-bundle-v1"
        or bundle.get("source_contract") != formalization_contract
        or bundle.get(source_bundle_sha_key) != legacy_sha256
    ):
        raise ValueError("resealed legacy output bundle contract differs")
    observed_bundle_sha256 = str(manifest.get("output_bundle_sha256", ""))
    if observed_bundle_sha256 != _canonical_json_sha256(bundle):
        raise ValueError("resealed legacy output bundle SHA256 differs")
    if (
        expected_output_bundle_sha256 is not None
        and observed_bundle_sha256 != expected_output_bundle_sha256
    ):
        raise ValueError("final feature output bundle differs from caller authority")
    records = bundle.get("frames")
    if not isinstance(records, list) or len(records) != len(frame_records):
        raise ValueError("resealed legacy output bundle frame count differs")

    expected_manifest = {
        **legacy,
        "execution": execution,
        "output_bundle": bundle,
        "output_bundle_sha256": observed_bundle_sha256,
    }
    if manifest != expected_manifest:
        raise ValueError(
            "resealed manifest changes fields outside the formal wrapper"
        )

    signature: dict[str, object] | None = None
    total_bytes = 0
    all_expected_paths: set[str] = set()
    for frame_record, snapshot in zip(frame_records, records):
        if not isinstance(frame_record, dict) or not isinstance(snapshot, dict):
            raise ValueError("resealed legacy frame record is invalid")
        if set(snapshot) != {"frame", "feature_signature", "tensors"}:
            raise ValueError("resealed legacy frame snapshot has unknown fields")
        if snapshot.get("frame") != frame_record:
            raise ValueError("resealed legacy output frame order differs")
        stem = str(frame_record.get("saved_stem", ""))
        expected_paths = _expected_tensor_relative_paths(stem, adaptor_names)
        tensor_records = snapshot.get("tensors")
        if not isinstance(tensor_records, list):
            raise ValueError("resealed legacy tensor set is invalid")
        actual_paths = [
            str(record.get("relative_path", ""))
            for record in tensor_records
            if isinstance(record, dict)
        ]
        if len(actual_paths) != len(tensor_records) or actual_paths != expected_paths:
            raise ValueError("resealed legacy tensor path set differs")
        if any(path in all_expected_paths for path in actual_paths):
            raise ValueError("resealed legacy tensor path is repeated")
        all_expected_paths.update(actual_paths)
        values: dict[str, torch.Tensor] = {}
        for record in tensor_records:
            if set(record) != {
                "relative_path",
                "sha256",
                "dtype",
                "shape",
                "num_bytes",
            }:
                raise ValueError("resealed legacy tensor record has unknown fields")
            relative_path = str(record["relative_path"])
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("resealed legacy tensor path is unsafe")
            values[relative_path] = _load_validated_tensor(
                root / relative,
                record,
            )
            total_bytes += int(record["num_bytes"])
        frame_signature = _signature_from_committed_tensors(
            values,
            stem,
            adaptor_names,
        )
        if snapshot.get("feature_signature") != frame_signature:
            raise ValueError("resealed legacy frame feature signature differs")
        signature = _merge_feature_signature(signature, frame_signature)

    features = legacy.get("features")
    if signature != features:
        raise ValueError("resealed legacy feature signature differs from source")
    feature_subdirs = {
        str(record.get("subdir", ""))
        for record in (
            [features.get("backbone", {}), features.get("summary", {})]
            + list(features.get("adaptors", []))
            if isinstance(features, dict)
            else []
        )
        if isinstance(record, dict) and str(record.get("subdir", ""))
    }
    observed_paths: set[str] = set()
    for subdir in feature_subdirs:
        directory = root / subdir
        if not directory.is_dir():
            raise ValueError(f"resealed legacy feature directory is missing: {directory}")
        observed_paths.update(
            path.relative_to(root).as_posix()
            for path in directory.iterdir()
            if path.is_file() and path.suffix == ".pt"
        )
    if observed_paths != all_expected_paths:
        raise ValueError("resealed legacy disk tensor set differs from its bundle")

    if verify_source_images:
        declared_image_dir = Path(
            str(legacy.get("image_dir", ""))
        ).expanduser().resolve()
        image_dir = Path(
            str(execution.get("resolved_source_image_dir", ""))
        ).expanduser().resolve()
        resolution_contract = str(
            execution.get("source_image_dir_resolution", "")
        )
        if image_dir == declared_image_dir:
            if resolution_contract != "declared_path":
                raise ValueError("legacy source-image resolution contract differs")
        elif resolution_contract != "explicit_override_all_frame_sha256_v1":
            raise ValueError("legacy source-image override is not content-bound")
        for frame_record in frame_records:
            source = image_dir / str(frame_record.get("source_file", ""))
            if not source.is_file():
                raise ValueError(f"legacy feature source image is missing: {source}")
            if _sha256_file(source) != str(frame_record.get("source_sha256", "")):
                raise ValueError(f"legacy feature source image SHA256 differs: {source}")
    return {
        "manifest_sha256": manifest_sha256,
        "output_bundle_sha256": observed_bundle_sha256,
        "legacy_source_manifest_sha256": legacy_sha256,
        "source_manifest_sha256": legacy_sha256,
        "formalization_contract": formalization_contract,
        "num_frames": len(frame_records),
        "logical_tensor_bytes": total_bytes,
        "feature_signature": signature,
    }


def _validate_final_output_bundle(
    output_root: str | Path,
    manifest: dict[str, object] | None = None,
    *,
    verify_source_images: bool = True,
    expected_output_bundle_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Independently reopen a completed extraction and verify every output."""

    root = Path(output_root).expanduser().resolve()
    manifest_path = root / "frame_manifest.json"
    if manifest is None:
        try:
            manifest, manifest_sha256, _source = load_json_object(
                manifest_path,
                expected_sha256=expected_manifest_sha256,
                label="final RADIO feature manifest",
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"final feature manifest is unreadable: {manifest_path}") from exc
    else:
        if expected_manifest_sha256 is not None:
            reopened, manifest_sha256, _source = load_json_object(
                manifest_path,
                expected_sha256=expected_manifest_sha256,
                label="final RADIO feature manifest",
            )
            if reopened != manifest:
                raise ValueError("provided feature manifest differs from disk")
        else:
            manifest_sha256 = _sha256_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("final feature manifest must contain an object")
    execution = manifest.get("execution")
    radio = manifest.get("radio")
    frame_records = manifest.get("frames")
    if not isinstance(execution, dict) or not isinstance(radio, dict):
        raise ValueError("final feature manifest provenance is incomplete")
    if execution.get("formalization_contract") in {
        LEGACY_RESEAL_CONTRACT,
        INCOMPLETE_RUNTIME_RESEAL_CONTRACT,
    }:
        return _validate_resealed_legacy_output_bundle(
            root,
            manifest,
            manifest_sha256,
            verify_source_images=verify_source_images,
            expected_output_bundle_sha256=expected_output_bundle_sha256,
        )
    if not isinstance(frame_records, list) or not frame_records:
        raise ValueError("final feature manifest has no frames")
    if int(manifest.get("num_frames", -1)) != len(frame_records):
        raise ValueError("final feature manifest frame count differs")
    contract_name = str(execution.get("resume_contract", ""))
    if contract_name != RESUME_CONTRACT_FILENAME:
        raise ValueError("final feature manifest is not strict-resume output")
    contract_path = root / contract_name
    try:
        resume_payload, resume_file_sha256, _source = load_json_object(
            contract_path,
            expected_sha256=str(
                execution.get("resume_contract_file_sha256", "")
            ),
            label="final RADIO feature resume contract",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("final feature resume contract is unreadable") from exc
    resume_contract_sha256 = _canonical_json_sha256(resume_payload)
    if (
        execution.get("resume_contract_sha256") != resume_contract_sha256
        or execution.get("resume_contract_file_sha256") != resume_file_sha256
    ):
        raise ValueError("final feature resume contract digest differs")
    resume_radio = resume_payload.get("radio")
    resume_input = resume_payload.get("input")
    if not isinstance(resume_radio, dict) or not isinstance(resume_input, dict):
        raise ValueError("final feature resume provenance is incomplete")
    if (
        radio != resume_radio
        or execution.get("runtime_fingerprint") != resume_payload.get("runtime")
        or execution.get("extractor_sha256")
        != resume_payload.get("extractor_sha256")
    ):
        raise ValueError("final feature source/runtime provenance differs")
    resume_frames = resume_input.get("frames")
    if not isinstance(resume_frames, list) or len(resume_frames) != len(frame_records):
        raise ValueError("final feature resume frame identities differ")
    for frame_record, resume_frame in zip(frame_records, resume_frames):
        if not isinstance(resume_frame, dict) or any(
            resume_frame.get(key) != value for key, value in frame_record.items()
        ):
            raise ValueError("final feature manifest is not bound to resume inputs")
    _, expected_by_stem = _declared_output_bundle(
        manifest,
        frame_records=frame_records,
        resume_contract_sha256=resume_contract_sha256,
    )
    observed_bundle_sha256 = str(manifest.get("output_bundle_sha256", ""))
    if (
        expected_output_bundle_sha256 is not None
        and observed_bundle_sha256 != expected_output_bundle_sha256
    ):
        raise ValueError("final feature output bundle differs from caller authority")
    adaptor_names = [str(value) for value in radio.get("requested_adaptors", [])]
    signature: dict[str, object] | None = None
    total_bytes = 0
    for frame_record in frame_records:
        if not isinstance(frame_record, dict):
            raise ValueError("final feature manifest frame identity is invalid")
        stem = str(frame_record.get("saved_stem", ""))
        snapshot, logical_bytes, reason = _validated_frame_snapshot(
            output_root=root,
            frame_record=frame_record,
            adaptor_names=adaptor_names,
            resume_contract_sha256=resume_contract_sha256,
            expected_bundle_record=expected_by_stem.get(stem),
        )
        if snapshot is None:
            raise ValueError(f"final feature frame {stem} is invalid: {reason}")
        signature = _merge_feature_signature(
            signature,
            dict(snapshot["feature_signature"]),
        )
        total_bytes += logical_bytes
    if signature != manifest.get("features"):
        raise ValueError("final feature manifest signature differs from its tensors")
    if verify_source_images:
        image_dir = Path(str(manifest.get("image_dir", ""))).expanduser().resolve()
        for frame_record in frame_records:
            source = image_dir / str(frame_record.get("source_file", ""))
            if not source.is_file():
                raise ValueError(f"final feature source image is missing: {source}")
            if _sha256_file(source) != str(frame_record.get("source_sha256", "")):
                raise ValueError(f"final feature source image SHA256 differs: {source}")
    return {
        "manifest_sha256": manifest_sha256,
        "output_bundle_sha256": observed_bundle_sha256,
        "resume_contract_sha256": resume_contract_sha256,
        "num_frames": len(frame_records),
        "logical_tensor_bytes": total_bytes,
        "feature_signature": signature,
    }


def _commit_frame(
    *,
    output_root: Path,
    frame_record: dict[str, object],
    backbone: torch.Tensor,
    summary: torch.Tensor,
    adaptors: dict[str, torch.Tensor],
    adaptor_names: list[str] | None,
    resume_contract_sha256: str,
) -> tuple[dict[str, object], int]:
    stem = str(frame_record["saved_stem"])
    relative_values: list[tuple[str, torch.Tensor]] = [
        (f"backbone/{stem}.pt", backbone),
        (f"summary/{stem}.pt", summary),
    ]
    for name in adaptor_names or []:
        if name not in adaptors:
            raise ValueError(f"RADIO did not return requested adaptor output {name!r}")
        relative_values.append(
            (f"{_adaptor_output_subdir(name)}/{stem}.pt", adaptors[name])
        )
    tensor_records: list[dict[str, object]] = []
    for relative_path, value in relative_values:
        _atomic_torch_save(value, output_root / relative_path)
        tensor_records.append(_tensor_record(output_root, relative_path, value))
    signature = _feature_signature(
        backbone,
        summary,
        adaptors,
        adaptor_names,
        require_all_adaptors=True,
    )
    # Reopen every tensor before advertising the frame as committed.  The same
    # validation runs again on resume; a missing or damaged member causes the
    # whole frame to be recomputed and atomically replaced.
    for record in tensor_records:
        _load_validated_tensor(
            output_root / str(record["relative_path"]),
            record,
        )
    marker = {
        "schema_version": FRAME_COMMIT_SCHEMA_VERSION,
        "resume_contract_sha256": resume_contract_sha256,
        "frame": frame_record,
        "feature_signature": signature,
        "tensors": tensor_records,
    }
    _atomic_json_write(_frame_commit_path(output_root, stem), marker)
    return signature, sum(int(record["num_bytes"]) for record in tensor_records)


# ---- main extraction loop -------------------------------------------------

@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    resume_partial = bool(getattr(args, "resume_partial", False))
    pacing_seconds = float(
        getattr(args, "radio_thermal_pacing_seconds_per_image", 0.0)
    )
    if not math.isfinite(pacing_seconds) or pacing_seconds < 0:
        raise ValueError(
            "radio-thermal-pacing-seconds-per-image must be finite and non-negative"
        )
    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if resume_partial and int(args.batch_size) != 1:
        raise ValueError("--resume-partial requires --batch_size 1")
    if resume_partial and not bool(args.skip_pca_stats):
        raise ValueError("--resume-partial requires --skip_pca_stats")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine adaptor setup
    adaptor_names: list[str] | None = None
    if args.extract_adaptors:
        adaptor_names = _parse_adaptor_names(args.adaptor_names)
        print(f"[RADIO] Extracting with adaptors: {adaptor_names}")

    # Load model
    model_source = _radio_model_source(
        args.radio_version,
        str(getattr(args, "radio_checkpoint", "")),
    )
    radio_source_tree = _python_source_tree_fingerprint(args.radio_repo)
    runtime_fingerprint = _runtime_fingerprint(device)
    extractor_sha256 = _sha256_file(Path(__file__).resolve())

    # Collect images
    image_paths, image_sort_mode = _collect_image_paths(args.image_dir)
    source_image_count = len(image_paths)
    excluded_image_stems = load_excluded_image_stems(
        args.exclude_image_stem,
        args.exclude_image_stems_file,
    )
    retained_indices, excluded_image_names = select_image_indices(
        image_paths,
        excluded_image_stems,
        min_remaining=1,
    )
    image_paths = [image_paths[index] for index in retained_indices]
    image_paths = _apply_subsampling(
        image_paths,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    frame_id_mode = getattr(args, "frame_id_mode", "auto")
    saved_frame_indices = _saved_frame_indices(
        image_paths,
        mode=frame_id_mode,
    )
    print(f"[RADIO] Found {len(image_paths)} images in {args.image_dir}")
    print(f"[RADIO] Image ordering: {image_sort_mode}")
    if excluded_image_names:
        print(
            "[RADIO] Excluded RGB feature views: "
            + ", ".join(excluded_image_names)
        )

    # Probe resolution from first image
    probe_img = Image.open(image_paths[0])
    orig_w, orig_h = probe_img.size
    target_h, target_w = _compute_scaled_radio_resolution(
        orig_h,
        orig_w,
        args.resolution_scale,
        patch_size=16,
    )
    patch_h, patch_w = target_h // 16, target_w // 16
    print(
        f"[RADIO] Input resolution: {orig_h}×{orig_w} "
        f"→ scale {args.resolution_scale:g} → {target_h}×{target_w}"
    )
    print(f"[RADIO] Feature grid: {patch_h}×{patch_w}")
    if args.sliding_window:
        if args.batch_size != 1:
            print("[RADIO] Sliding-window mode uses single-image batches; overriding batch_size=1")
            args.batch_size = 1
        print(
            f"[RADIO] Sliding-window extraction: tile={args.tile_size}px, "
            f"overlap={args.tile_overlap}px"
        )

    frame_manifest = [
        {
            "source_rank": source_rank,
            "frame_idx": int(saved_frame_indices[source_rank]),
            "source_file": source_path.name,
            "source_sha256": _sha256_file(source_path),
            "saved_stem": f"rgb_{saved_frame_indices[source_rank]}",
        }
        for source_rank, source_path in enumerate(image_paths)
    ]
    output_root = Path(args.output_dir)
    resume_contract_path: Path | None = None
    resume_contract_sha256 = ""
    if resume_partial:
        resume_payload = _resume_contract_payload(
            args=args,
            device=device,
            model_source=model_source,
            adaptor_names=adaptor_names,
            image_paths=image_paths,
            image_sort_mode=image_sort_mode,
            source_image_count=source_image_count,
            excluded_image_stems=excluded_image_stems,
            excluded_image_names=excluded_image_names,
            frame_records=frame_manifest,
            target_h=target_h,
            target_w=target_w,
            pacing_seconds=pacing_seconds,
            radio_source_tree=radio_source_tree,
            runtime_fingerprint=runtime_fingerprint,
        )
        resume_contract_path, resume_contract_sha256 = _prepare_resume_contract(
            output_root,
            resume_payload,
        )

    # Prepare output dirs
    subdirs = ["backbone", "summary"]
    if adaptor_names:
        subdirs += [_adaptor_output_subdir(name) for name in adaptor_names]
    for sd in subdirs:
        (output_root / sd).mkdir(parents=True, exist_ok=True)

    expected_bundle_by_stem: dict[str, dict[str, object]] = {}
    existing_manifest_path = output_root / "frame_manifest.json"
    if resume_partial and existing_manifest_path.exists():
        try:
            existing_manifest, _digest, _source = load_json_object(
                existing_manifest_path,
                label="existing final RADIO feature manifest",
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                "existing final feature manifest is unreadable; refusing loose repair"
            ) from exc
        if not isinstance(existing_manifest, dict):
            raise ValueError("existing final feature manifest is not an object")
        _, expected_bundle_by_stem = _declared_output_bundle(
            existing_manifest,
            frame_records=frame_manifest,
            resume_contract_sha256=resume_contract_sha256,
        )

    committed_frames: dict[int, tuple[dict[str, object], int]] = {}
    feature_signature: dict[str, object] | None = None
    if resume_partial:
        for source_rank, frame_record in enumerate(frame_manifest):
            signature, logical_bytes, reason = _validate_committed_frame(
                output_root=output_root,
                frame_record=frame_record,
                adaptor_names=adaptor_names,
                resume_contract_sha256=resume_contract_sha256,
                expected_bundle_record=expected_bundle_by_stem.get(
                    str(frame_record["saved_stem"])
                ),
            )
            if signature is not None:
                feature_signature = _merge_feature_signature(
                    feature_signature,
                    signature,
                )
                committed_frames[source_rank] = (signature, logical_bytes)
                continue
            stem = str(frame_record["saved_stem"])
            has_partial_artifact = _frame_commit_path(output_root, stem).exists() or any(
                (output_root / relative_path).exists()
                for relative_path in _expected_tensor_relative_paths(
                    stem,
                    adaptor_names,
                )
            )
            if has_partial_artifact:
                print(
                    f"[RADIO] Recomputing {stem}: committed-frame validation "
                    f"failed ({reason})"
                )
        if committed_frames:
            print(
                f"[RADIO] Resume validated {len(committed_frames)}/{len(image_paths)} "
                "committed frames"
            )

    model = None
    conditioner = None
    if len(committed_frames) != len(image_paths):
        print(f"[RADIO] Loading {args.radio_version} from {args.radio_repo} …")
        model, conditioner = _load_radio_model(
            args.radio_repo,
            model_source["load_source"],
            adaptor_names,
            device,
            expected_checkpoint_sha256=model_source["checkpoint_sha256"],
        )
    else:
        print("[RADIO] All frames passed committed-frame validation; no model load needed")

    pca_accumulator: list[torch.Tensor] = []
    total_bytes: int = 0
    t0 = time.time()

    # Process in batches
    n = len(image_paths)
    for start in tqdm(range(0, n, args.batch_size), desc="Extracting RADIO features"):
        if resume_partial and start in committed_frames:
            signature, logical_bytes = committed_frames[start]
            feature_signature = _merge_feature_signature(
                feature_signature,
                signature,
            )
            total_bytes += logical_bytes
            continue
        batch_paths = image_paths[start : start + args.batch_size]
        imgs = _load_and_preprocess(batch_paths, target_h, target_w, device)

        if args.sliding_window:
            summary, spatial_2d, adaptor_2d = _extract_sliding_window_single(
                model,
                conditioner,
                imgs,
                args.amp,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                adaptor_names=adaptor_names,
            )
        else:
            summary, spatial_2d, adaptor_2d = _run_radio_batch(
                model,
                conditioner,
                imgs,
                args.amp,
                patch_h,
                patch_w,
                adaptor_names=adaptor_names,
            )

        B, D, _, _ = spatial_2d.shape
        if B != len(batch_paths):
            raise ValueError(
                "RADIO output batch size differs from the input batch: "
                f"{B} != {len(batch_paths)}"
            )

        # Save per-frame
        for i in range(B):
            source_rank = start + i
            frame_record = frame_manifest[source_rank]
            stem = str(frame_record["saved_stem"])

            # Backbone: float16
            bb = spatial_2d[i].cpu().half()
            # Summary: float32
            sm = summary[i].cpu().float()
            frame_adaptors: dict[str, torch.Tensor] = {}
            if adaptor_names and adaptor_2d:
                for name in adaptor_names:
                    if name not in adaptor_2d:
                        continue
                    frame_adaptors[name] = adaptor_2d[name][i].cpu().half()

            if resume_partial:
                observed_signature, logical_bytes = _commit_frame(
                    output_root=output_root,
                    frame_record=frame_record,
                    backbone=bb,
                    summary=sm,
                    adaptors=frame_adaptors,
                    adaptor_names=adaptor_names,
                    resume_contract_sha256=resume_contract_sha256,
                )
            else:
                _atomic_torch_save(bb, output_root / "backbone" / f"{stem}.pt")
                _atomic_torch_save(sm, output_root / "summary" / f"{stem}.pt")
                logical_bytes = int(
                    bb.nelement() * bb.element_size()
                    + sm.nelement() * sm.element_size()
                )
                for name, ad_frame in frame_adaptors.items():
                    short_name = _adaptor_output_subdir(name)
                    _atomic_torch_save(
                        ad_frame,
                        output_root / short_name / f"{stem}.pt",
                    )
                    logical_bytes += int(
                        ad_frame.nelement() * ad_frame.element_size()
                    )
                observed_signature = _feature_signature(
                    bb,
                    sm,
                    frame_adaptors,
                    adaptor_names,
                    require_all_adaptors=False,
                )
            feature_signature = _merge_feature_signature(
                feature_signature,
                observed_signature,
            )
            total_bytes += logical_bytes

            # The commit marker (resume path) or complete atomic tensor set
            # (legacy path) is durable before synchronization and cooling.
            _thermal_pause(device, pacing_seconds)

            # Accumulate only when the optional PCA warm-start artifact is
            # requested.  Large oracle/teacher extractions otherwise spend
            # most of their runtime and RAM on an unused global SVD.
            if not args.skip_pca_stats:
                pca_accumulator.append(bb.float())

    if feature_signature is None:
        raise RuntimeError("RADIO extraction produced no feature signature")

    # Refuse to publish a terminal that could have crossed source, runtime,
    # checkpoint, or input-image generations while it was being produced.
    if _sha256_file(Path(__file__).resolve()) != extractor_sha256:
        raise RuntimeError("feature extractor source changed during extraction")
    if _python_source_tree_fingerprint(args.radio_repo) != radio_source_tree:
        raise RuntimeError("RADIO Python source tree changed during extraction")
    if _runtime_fingerprint(device) != runtime_fingerprint:
        raise RuntimeError("numerical runtime fingerprint changed during extraction")
    if _radio_model_source(
        args.radio_version,
        str(getattr(args, "radio_checkpoint", "")),
    ) != model_source:
        raise RuntimeError("RADIO checkpoint changed during extraction")
    for source_path, frame_record in zip(image_paths, frame_manifest):
        if _sha256_file(source_path) != str(frame_record["source_sha256"]):
            raise RuntimeError(
                f"source image changed during extraction: {source_path}"
            )

    # PCA statistics
    pca_path = output_root / "pca_stats.pt"
    if args.skip_pca_stats:
        pca_path = None
    else:
        print("[RADIO] Computing PCA statistics …")
        pca_stats = _compute_pca_stats(pca_accumulator, n_components=64)
        _atomic_torch_save(pca_stats, pca_path)

    output_bundle: dict[str, object] | None = None
    output_bundle_sha256 = ""
    if resume_partial:
        output_bundle, bundle_signature, bundle_bytes = _build_output_bundle(
            output_root=output_root,
            frame_records=frame_manifest,
            adaptor_names=adaptor_names,
            resume_contract_sha256=resume_contract_sha256,
        )
        if bundle_signature != feature_signature:
            raise RuntimeError("final output bundle feature signature differs")
        total_bytes = bundle_bytes
        output_bundle_sha256 = _canonical_json_sha256(output_bundle)

    manifest_path = output_root / "frame_manifest.json"
    manifest = {
        "scene": args.scene,
        "radio": {
            "version": args.radio_version,
            "repo": str(Path(args.radio_repo).expanduser().resolve()),
            "repo_hubconf_sha256": _sha256_file(
                Path(args.radio_repo).expanduser().resolve() / "hubconf.py"
            ),
            "checkpoint": model_source["checkpoint"],
            "checkpoint_sha256": model_source["checkpoint_sha256"],
            "checkpoint_provenance": model_source["checkpoint_provenance"],
            "checkpoint_load_contract": model_source[
                "checkpoint_load_contract"
            ],
            "requested_adaptors": list(adaptor_names or []),
            "python_source_tree": radio_source_tree,
        },
        "image_dir": str(Path(args.image_dir).resolve()),
        "image_sort_mode": image_sort_mode,
        "frame_id_mode": frame_id_mode,
        "batch_size": int(args.batch_size),
        "amp": bool(args.amp),
        "sliding_window": bool(args.sliding_window),
        "tile_size": int(args.tile_size) if args.sliding_window else None,
        "tile_overlap": int(args.tile_overlap) if args.sliding_window else None,
        "resolution_scale": float(args.resolution_scale),
        "radio_input_resolution_hw": [int(target_h), int(target_w)],
        "source_image_count_before_exclusion": source_image_count,
        "excluded_image_stems": list(excluded_image_stems),
        "excluded_image_names": excluded_image_names,
        "num_frames": len(frame_manifest),
        "features": feature_signature,
        "output_bundle": output_bundle,
        "output_bundle_sha256": output_bundle_sha256,
        "execution": {
            "atomic_tensor_commit": "same_directory_temp_then_os_replace_v1",
            "atomic_manifest_commit": "same_directory_temp_then_os_replace_v1",
            "resume_partial": resume_partial,
            "resume_contract": (
                RESUME_CONTRACT_FILENAME if resume_contract_path is not None else ""
            ),
            "resume_contract_sha256": resume_contract_sha256,
            "resume_contract_file_sha256": (
                _sha256_file(resume_contract_path)
                if resume_contract_path is not None
                else ""
            ),
            "committed_frame_validation": (
                "same_fd_sha256_weights_only_dtype_shape_finite_v2"
                if resume_partial
                else "disabled"
            ),
            "invalid_or_missing_frame_policy": (
                "recompute_entire_frame_v1" if resume_partial else "not_applicable"
            ),
            "radio_thermal_pacing_seconds_per_image": pacing_seconds,
            "pacing_order": "frame_commit_then_cuda_synchronize_then_sleep_v1",
            "extractor_sha256": extractor_sha256,
            "runtime_fingerprint": runtime_fingerprint,
        },
        "frames": frame_manifest,
    }
    _atomic_json_write(manifest_path, manifest)
    if resume_partial:
        _validate_final_output_bundle(output_root, manifest)

    # Summary
    elapsed = time.time() - t0
    disk_mb = total_bytes / (1024 * 1024)
    backbone_contract = feature_signature["backbone"]
    summary_contract = feature_signature["summary"]
    print("=" * 60)
    print(f"  Scene       : {args.scene}")
    print(f"  Frames      : {n}")
    print(
        f"  Backbone dim: {backbone_contract['dim']} × "
        f"{backbone_contract['grid'][0]}×{backbone_contract['grid'][1]}"
    )
    print(f"  Summary dim : {summary_contract['dim']}")
    print(f"  Disk usage  : {disk_mb:.1f} MB  (float16 spatial + float32 summary)")
    print(f"  PCA saved   : {pca_path if pca_path is not None else 'skipped'}")
    print(f"  Manifest    : {manifest_path}")
    print(f"  Time        : {elapsed:.1f}s  ({elapsed / n:.2f}s/frame)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RADIO features for all images in a scene."
    )
    parser.add_argument("--scene", type=str, default="room_0", help="Scene name")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory of RGB images (.png/.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save extracted features",
    )
    parser.add_argument(
        "--radio_repo",
        type=str,
        default="/root/RADIO",
        help="Local path to RADIO repo for torch.hub",
    )
    parser.add_argument(
        "--radio_version",
        type=str,
        default="c-radio_v4-h",
        help="RADIO model version string",
    )
    parser.add_argument(
        "--radio_checkpoint",
        type=str,
        default="",
        help=(
            "Optional explicit checkpoint loaded by torch.hub and recorded by "
            "SHA256. Required for promotion-grade extracted adaptor provenance."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Images per batch")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every Nth image from image_dir",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on the number of images to extract",
    )
    parser.add_argument(
        "--frame-id-mode",
        choices=FRAME_ID_MODES,
        default="auto",
        help=(
            "Output feature id policy. 'auto' preserves legacy numeric suffixes; "
            "'source_rank' assigns a unique dense id after exclusion/subsampling."
        ),
    )
    parser.add_argument(
        "--exclude-image-stem",
        action="append",
        default=[],
        help=(
            "Exact case-sensitive image basename stem to exclude before feature "
            "extraction; repeat for multiple held-out views"
        ),
    )
    parser.add_argument(
        "--exclude-image-stems-file",
        default="",
        help="Optional JSON/text file of exact image stems to exclude",
    )
    parser.add_argument(
        "--extract_adaptors",
        action="store_true",
        help="Also extract adaptor features listed by --adaptor_names",
    )
    parser.add_argument(
        "--adaptor_names",
        type=str,
        default="siglip2-g,sam3",
        help=(
            "Comma-separated RADIO adaptor names to extract when --extract_adaptors "
            "is set, e.g. siglip2-g,dino_v3,sam3"
        ),
    )
    parser.add_argument(
        "--resolution_scale",
        type=float,
        default=1.0,
        help="Scale input images before RADIO extraction (default: 1.0)",
    )
    parser.add_argument(
        "--sliding_window",
        action="store_true",
        help="Extract high-resolution features by stitching overlapping single-image tiles",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Sliding-window tile size in input pixels (default: 1024)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=128,
        help="Sliding-window tile overlap in input pixels (default: 128)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device (default: cuda)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (default: True)",
    )
    parser.add_argument(
        "--skip_pca_stats",
        action="store_true",
        help="Skip the optional global PCA warm-start artifact.",
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "Strictly resume atomically committed frames. This safe path is "
            "limited to --batch_size 1 with --skip_pca_stats."
        ),
    )
    parser.add_argument(
        "--radio-thermal-pacing-seconds-per-image",
        type=float,
        default=0.0,
        help=(
            "Execution-only cooling pause after each committed frame. CUDA is "
            "synchronized before a positive pause."
        ),
    )

    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
