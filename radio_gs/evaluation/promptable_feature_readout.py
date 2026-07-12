"""Protocol-safe prototype readout for promptable NVS feature fields.

This module produces continuous binary foreground scores and deliberately does
not evaluate them.  In particular, prediction generation never opens an
evaluation frame's ``ground_truth`` path.  The only mask-like inputs are the
declared prompt assets:

* NVOS: the positive and negative scribbles on the single reference frame;
* SPIn-NeRF: the binary mask on the single reference frame.

Scores are ``cos(feature, foreground) - cos(feature, background)``.  They are
written as float32 ``.npy`` arrays and bound to the input protocol with
``compute_protocol_hash`` so the separate evaluator can reject mismatches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from radio_gs.evaluation.promptable_segmentation import (
    compute_protocol_hash,
    load_ground_truth_mask,
    load_json_manifest,
    resize_mask_nearest,
    validate_manifest,
)
from radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint


FEATURE_LAYOUTS = ("auto", "chw", "hwc")
DEFAULT_FEATURE_PATTERN = "{scene_id}/{camera_name}.npy"
PREDICTION_MANIFEST_NAME = "prediction_manifest.json"


class FeatureReadoutError(ValueError):
    """Raised when prediction generation would be ambiguous or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(path: str | Path, base_dir: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else base_dir / value


def _load_torch_payload(path: Path) -> Any:
    """Load tensor-only project feature files without the general unpickler."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        # PyTorch before the weights_only argument was introduced.  Project
        # feature files are expected to contain tensors/basic containers only.
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


def _array_from_payload(payload: Any, *, path: Path) -> np.ndarray:
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().float().numpy()
    if isinstance(payload, np.ndarray):
        return np.asarray(payload)
    if isinstance(payload, Mapping):
        preferred = ("feature", "features", "embedding", "embeddings", "tensor")
        present = [key for key in preferred if key in payload]
        if len(present) == 1:
            return _array_from_payload(payload[present[0]], path=path)
        candidates = [
            value
            for value in payload.values()
            if isinstance(value, (torch.Tensor, np.ndarray))
        ]
        if not present and len(candidates) == 1:
            return _array_from_payload(candidates[0], path=path)
    if isinstance(payload, (list, tuple)) and len(payload) == 1:
        return _array_from_payload(payload[0], path=path)
    raise FeatureReadoutError(
        f"{path} must contain one feature tensor/array (or a standard feature key)"
    )


def _load_feature_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            preferred = [
                key
                for key in ("feature", "features", "embedding", "embeddings", "tensor")
                if key in archive.files
            ]
            if len(preferred) == 1:
                return np.asarray(archive[preferred[0]])
            if not preferred and len(archive.files) == 1:
                return np.asarray(archive[archive.files[0]])
            raise FeatureReadoutError(
                f"{path} must contain exactly one array or one standard feature key"
            )
    if suffix in {".pt", ".pth"}:
        return _array_from_payload(_load_torch_payload(path), path=path)
    raise FeatureReadoutError(
        f"Unsupported feature file {path}; expected .npy, .npz, .pt, or .pth"
    )


def _infer_layout(shape: Sequence[int], expected_channels: int | None) -> str:
    first, middle, last = (int(value) for value in shape)
    if expected_channels is not None:
        matches = []
        if first == expected_channels:
            matches.append("chw")
        if last == expected_channels:
            matches.append("hwc")
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FeatureReadoutError(
                f"Feature shape {tuple(shape)} has neither endpoint equal to the "
                f"expected {expected_channels} channels"
            )
        raise FeatureReadoutError(
            f"Feature shape {tuple(shape)} is ambiguous; set feature_layout explicitly"
        )

    # Dense embedding channels are usually clearly smaller or larger than both
    # spatial axes.  Fail closed for cube-like/ambiguous arrays instead of
    # silently transposing a feature map incorrectly.
    chw_clear = first < min(middle, last) or first > max(middle, last)
    hwc_clear = last < min(first, middle) or last > max(first, middle)
    if chw_clear and not hwc_clear:
        return "chw"
    if hwc_clear and not chw_clear:
        return "hwc"
    raise FeatureReadoutError(
        f"Cannot infer CHW versus HWC for feature shape {tuple(shape)}; "
        "set feature_layout explicitly"
    )


def load_feature_map(
    path: str | Path,
    *,
    layout: str = "auto",
    expected_channels: int | None = None,
) -> np.ndarray:
    """Load a project feature file and return a finite float32 ``[C,H,W]`` map."""

    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature map not found: {feature_path}")
    if layout not in FEATURE_LAYOUTS:
        raise FeatureReadoutError(f"feature_layout must be one of {FEATURE_LAYOUTS}")
    values = _load_feature_array(feature_path)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3:
        raise FeatureReadoutError(
            f"{feature_path} must have shape CHW/HWC (optionally batch size 1), "
            f"got {tuple(values.shape)}"
        )
    selected_layout = _infer_layout(values.shape, expected_channels) if layout == "auto" else layout
    if selected_layout == "hwc":
        values = np.moveaxis(values, -1, 0)
    values = np.ascontiguousarray(values, dtype=np.float32)
    if min(values.shape) <= 0:
        raise FeatureReadoutError(f"Feature map has an empty dimension: {feature_path}")
    if expected_channels is not None and int(values.shape[0]) != int(expected_channels):
        raise FeatureReadoutError(
            f"{feature_path} has {values.shape[0]} channels; expected {expected_channels}"
        )
    if not bool(np.isfinite(values).all()):
        raise FeatureReadoutError(f"Feature map contains NaN or infinity: {feature_path}")
    return values


class _FeatureProjector:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str,
        chunk_size: int,
    ) -> None:
        if int(chunk_size) <= 0:
            raise FeatureReadoutError("projection_chunk_size must be positive")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise FeatureReadoutError(f"CUDA adaptor device requested but unavailable: {device}")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.checkpoint_sha256 = _sha256(self.checkpoint)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.adaptor = load_radio_adaptor_from_checkpoint(
            self.checkpoint,
            "sam3",
            kind="feature_projection",
        )
        if self.adaptor.input_dim != 1280 or self.adaptor.output_dim != 1024:
            raise FeatureReadoutError(
                "RADIO sam3 feature_projection must map 1280d to 1024d; "
                f"checkpoint maps {self.adaptor.input_dim}d to {self.adaptor.output_dim}d"
            )
        self.adaptor.eval().requires_grad_(False).to(self.device)

    @property
    def input_dim(self) -> int:
        return int(self.adaptor.input_dim)

    @property
    def output_dim(self) -> int:
        return int(self.adaptor.output_dim)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        channels, height, width = features.shape
        if channels != self.input_dim:
            raise FeatureReadoutError(
                f"RADIO sam3 adaptor requires {self.input_dim} channels, got {channels}"
            )
        tokens = np.moveaxis(features, 0, -1).reshape(-1, channels)
        projected = np.empty((tokens.shape[0], self.output_dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], self.chunk_size):
                stop = min(tokens.shape[0], start + self.chunk_size)
                batch = torch.from_numpy(tokens[start:stop]).to(self.device)
                output = self.adaptor(batch).float().cpu().numpy()
                projected[start:stop] = output
        return np.moveaxis(projected.reshape(height, width, self.output_dim), -1, 0)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "radio_sam3_feature_projection_adaptor_embedding",
            "label": "RADIO SAM3 adaptor embedding",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "adaptor_name": "sam3",
            "adaptor_kind": "feature_projection",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "frozen": True,
            "official_sam_decoder": False,
            "clarification": (
                "This is the frozen RADIO sam3 feature_projection embedding; "
                "it is not an official SAM/SAM2/SAM3 mask decoder."
            ),
        }


def _normalized_pixels(features: np.ndarray) -> np.ndarray:
    flat = features.reshape(features.shape[0], -1)
    norms = np.linalg.norm(flat, axis=0, keepdims=True)
    return np.divide(flat, norms, out=np.zeros_like(flat), where=norms > 1e-12)


def _prototype(normalized: np.ndarray, mask: np.ndarray, *, role: str) -> np.ndarray:
    selected = normalized[:, mask.reshape(-1)]
    if selected.shape[1] == 0:
        raise FeatureReadoutError(f"Prompt {role} mask selects no feature pixels")
    prototype = selected.mean(axis=1)
    norm = float(np.linalg.norm(prototype))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise FeatureReadoutError(f"Prompt {role} prototype has zero/invalid norm")
    return prototype / norm


def _resize_prompt(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = mask.astype(bool, copy=False)
    if values.shape != shape:
        values = resize_mask_nearest(values, shape).astype(bool, copy=False)
    return values


def _prompt_prototypes(
    scene: Mapping[str, Any],
    prompt_features: np.ndarray,
    *,
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prompt = scene.get("prompt")
    if not isinstance(prompt, Mapping):
        raise FeatureReadoutError(f"Scene {scene['scene_id']} lacks a prompt object")
    prompt_type = str(prompt.get("type", ""))
    spatial_shape = (int(prompt_features.shape[1]), int(prompt_features.shape[2]))
    prompt_paths: dict[str, str] = {}
    if prompt_type == "positive_negative_scribbles":
        if not prompt.get("positive_path") or not prompt.get("negative_path"):
            raise FeatureReadoutError(
                f"NVOS scene {scene['scene_id']} requires positive_path and negative_path"
            )
        positive_path = _resolve_relative(prompt["positive_path"], base_dir)
        negative_path = _resolve_relative(prompt["negative_path"], base_dir)
        foreground = _resize_prompt(load_ground_truth_mask(positive_path), spatial_shape)
        background = _resize_prompt(load_ground_truth_mask(negative_path), spatial_shape)
        if bool(np.logical_and(foreground, background).any()):
            raise FeatureReadoutError(
                f"Scene {scene['scene_id']} positive/negative scribbles overlap after resizing"
            )
        prompt_paths = {
            "positive_scribble": str(positive_path),
            "negative_scribble": str(negative_path),
        }
    elif prompt_type == "reference_binary_mask":
        if not prompt.get("mask_path"):
            raise FeatureReadoutError(
                f"SPIn-NeRF scene {scene['scene_id']} requires prompt.mask_path"
            )
        mask_path = _resolve_relative(prompt["mask_path"], base_dir)
        foreground = _resize_prompt(load_ground_truth_mask(mask_path), spatial_shape)
        background = np.logical_not(foreground)
        prompt_paths = {"reference_binary_mask": str(mask_path)}
    else:
        raise FeatureReadoutError(
            f"Unsupported prompt type for scene {scene['scene_id']}: {prompt_type!r}"
        )

    normalized = _normalized_pixels(prompt_features)
    foreground_prototype = _prototype(normalized, foreground, role="foreground")
    background_prototype = _prototype(normalized, background, role="background")
    metadata = {
        "type": prompt_type,
        "paths": prompt_paths,
        "resampling": "nearest_to_prompt_feature_grid",
        "feature_grid_shape": list(spatial_shape),
        "foreground_pixels": int(foreground.sum()),
        "background_pixels": int(background.sum()),
        "asset_sha256": {
            role: _sha256(Path(path)) for role, path in prompt_paths.items()
        },
    }
    return foreground_prototype, background_prototype, metadata


def cosine_margin_scores(
    features: np.ndarray,
    foreground_prototype: np.ndarray,
    background_prototype: np.ndarray,
) -> np.ndarray:
    """Return ``cos(fg)-cos(bg)`` for a ``[C,H,W]`` feature map."""

    if features.ndim != 3:
        raise FeatureReadoutError("features must be [C,H,W]")
    channels, height, width = features.shape
    if foreground_prototype.shape != (channels,) or background_prototype.shape != (channels,):
        raise FeatureReadoutError("Feature/prototype channel dimensions differ")
    normalized = _normalized_pixels(features)
    scores = foreground_prototype @ normalized - background_prototype @ normalized
    scores = scores.reshape(height, width).astype(np.float32, copy=False)
    if not bool(np.isfinite(scores).all()):
        raise FeatureReadoutError("Cosine margin scores contain NaN or infinity")
    return scores


def _safe_component(value: str, *, role: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise FeatureReadoutError(f"Unsafe {role} path component: {value!r}")
    return value


def _resolve_feature_path(
    frame: Mapping[str, Any],
    *,
    scene_id: str,
    benchmark: str,
    manifest_base_dir: Path,
    feature_root: Path | None,
    feature_pattern: str,
) -> Path:
    inline = frame.get("feature_path")
    if inline:
        return _resolve_relative(str(inline), manifest_base_dir)
    if feature_root is None:
        raise FeatureReadoutError(
            f"{scene_id}/{frame['frame_id']} has no feature_path and no feature_root was given"
        )
    values = {
        "benchmark": benchmark,
        "scene": scene_id,
        "scene_id": scene_id,
        "frame_id": str(frame["frame_id"]),
        "camera_name": str(frame.get("camera_name") or frame["frame_id"]),
    }
    try:
        rendered = feature_pattern.format(**values)
    except (KeyError, IndexError, ValueError) as error:
        raise FeatureReadoutError(f"Invalid feature_pattern {feature_pattern!r}: {error}") from error
    path = Path(rendered).expanduser()
    return path if path.is_absolute() else feature_root / path


def _atomic_save_npy(path: Path, values: np.ndarray, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Prediction already exists (use overwrite): {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Prediction manifest already exists (use overwrite): {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_feature_readout_predictions(
    manifest: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    *,
    feature_root: str | Path | None = None,
    feature_pattern: str = DEFAULT_FEATURE_PATTERN,
    feature_layout: str = "auto",
    radio_sam3_adaptor_checkpoint: str | Path | None = None,
    adaptor_device: str = "cpu",
    projection_chunk_size: int = 8192,
    method_name: str = "GaussFM feature-field prototype readout",
    output_manifest_name: str = PREDICTION_MANIFEST_NAME,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate protocol-bound scores without opening evaluation ground truth."""

    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        raw_manifest = load_json_manifest(manifest_path)
        manifest_base_dir = manifest_path.parent
        manifest_source: str | None = str(manifest_path)
        manifest_file_sha256: str | None = _sha256(manifest_path)
    elif isinstance(manifest, Mapping):
        raw_manifest = copy.deepcopy(dict(manifest))
        manifest_base_dir = Path.cwd()
        manifest_source = None
        manifest_file_sha256 = None
    else:
        raise FeatureReadoutError("manifest must be a JSON path or mapping")

    # Evaluator validation is structural/hash-only and does not open any mask.
    normalized = validate_manifest(raw_manifest)
    threshold = normalized["protocol"]["threshold"]
    if threshold.get("mode") != "fixed" or float(threshold.get("value", float("nan"))) != 0.0:
        raise FeatureReadoutError(
            "Feature readout requires the frozen manifest threshold {mode: fixed, value: 0.0}"
        )
    if feature_layout not in FEATURE_LAYOUTS:
        raise FeatureReadoutError(f"feature_layout must be one of {FEATURE_LAYOUTS}")

    protocol_hash = compute_protocol_hash(raw_manifest)
    output_root = Path(output_dir).expanduser().resolve()
    resolved_feature_root = (
        Path(feature_root).expanduser().resolve() if feature_root is not None else None
    )
    projector = (
        _FeatureProjector(
            radio_sam3_adaptor_checkpoint,
            device=adaptor_device,
            chunk_size=projection_chunk_size,
        )
        if radio_sam3_adaptor_checkpoint is not None
        else None
    )
    benchmark = str(raw_manifest.get("benchmark") or normalized["protocol"]["benchmark"])
    predictions: MutableMapping[str, dict[str, str]] = {}
    prediction_sha256: MutableMapping[str, dict[str, str]] = {}
    scene_records: list[dict[str, Any]] = []

    for scene in normalized["scenes"]:
        scene_id = _safe_component(str(scene["scene_id"]), role="scene_id")
        prompt_ids = list(scene["prompt_frame_ids"])
        if len(prompt_ids) != 1:
            raise FeatureReadoutError(f"Scene {scene_id} must have exactly one prompt frame")
        prompt_id = str(prompt_ids[0])
        prompt = scene.get("prompt")
        if not isinstance(prompt, Mapping) or str(prompt.get("frame_id", "")) != prompt_id:
            raise FeatureReadoutError(
                f"Scene {scene_id} prompt.frame_id must match its declared prompt frame"
            )
        prompt_frame = scene["frames"][prompt_id]
        prompt_feature_path = _resolve_feature_path(
            prompt_frame,
            scene_id=scene_id,
            benchmark=benchmark,
            manifest_base_dir=manifest_base_dir,
            feature_root=resolved_feature_root,
            feature_pattern=feature_pattern,
        )
        prompt_layout = str(prompt_frame.get("feature_layout") or feature_layout)
        prompt_features = load_feature_map(
            prompt_feature_path,
            layout=prompt_layout,
            expected_channels=projector.input_dim if projector is not None else None,
        )
        if projector is not None:
            prompt_features = projector(prompt_features)
        foreground, background, prompt_metadata = _prompt_prototypes(
            scene,
            prompt_features,
            base_dir=manifest_base_dir,
        )

        scene_predictions: dict[str, str] = {}
        scene_prediction_hashes: dict[str, str] = {}
        output_frames: list[dict[str, Any]] = []
        for raw_frame_id in scene["evaluation_frame_ids"]:
            frame_id = _safe_component(str(raw_frame_id), role="frame_id")
            frame = scene["frames"][frame_id]
            feature_path = _resolve_feature_path(
                frame,
                scene_id=scene_id,
                benchmark=benchmark,
                manifest_base_dir=manifest_base_dir,
                feature_root=resolved_feature_root,
                feature_pattern=feature_pattern,
            )
            layout = str(frame.get("feature_layout") or feature_layout)
            expected_channels = projector.input_dim if projector is not None else int(prompt_features.shape[0])
            target_features = load_feature_map(
                feature_path,
                layout=layout,
                expected_channels=expected_channels,
            )
            if projector is not None:
                target_features = projector(target_features)
            scores = cosine_margin_scores(target_features, foreground, background)
            relative_path = Path("scores") / scene_id / f"{frame_id}.npy"
            destination = output_root / relative_path
            _atomic_save_npy(destination, scores, overwrite=overwrite)
            scene_predictions[frame_id] = relative_path.as_posix()
            scene_prediction_hashes[frame_id] = _sha256(destination)
            output_frames.append(
                {
                    "frame_id": frame_id,
                    "feature_path": str(feature_path),
                    "feature_sha256": _sha256(feature_path),
                    "score_path": relative_path.as_posix(),
                    "score_sha256": scene_prediction_hashes[frame_id],
                    "score_shape": list(scores.shape),
                    "score_dtype": "float32",
                }
            )
        predictions[scene_id] = scene_predictions
        prediction_sha256[scene_id] = scene_prediction_hashes
        scene_records.append(
            {
                "scene_id": scene_id,
                "prompt_frame_id": prompt_id,
                "prompt_feature_path": str(prompt_feature_path),
                "prompt_feature_sha256": _sha256(prompt_feature_path),
                "prompt": prompt_metadata,
                "embedding_dim": int(prompt_features.shape[0]),
                "outputs": output_frames,
            }
        )

    if projector is None:
        embedding_metadata: dict[str, Any] = {
            "type": "input_feature_field_embedding",
            "radio_sam3_adaptor_applied": False,
            "official_sam_decoder": False,
        }
    else:
        embedding_metadata = projector.metadata()
        embedding_metadata["radio_sam3_adaptor_applied"] = True

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "promptable_nvs_continuous_score_predictions",
        "protocol_hash": protocol_hash,
        "prediction_root": ".",
        "predictions": predictions,
        "prediction_sha256": prediction_sha256,
        "method": {
            "name": str(method_name),
            "readout": "reference_prototype_cosine_margin",
            "score_semantics": "cosine_similarity_foreground_minus_background",
            "prototype_reduction": (
                "mean_of_l2_normalized_prompt_pixel_embeddings_then_l2_normalize"
            ),
            "threshold": {"mode": "fixed", "value": 0.0, "source": "input_manifest"},
            "embedding_space": embedding_metadata,
        },
        "input": {
            "dataset_manifest": manifest_source,
            "dataset_manifest_sha256": manifest_file_sha256,
            "feature_root": str(resolved_feature_root) if resolved_feature_root else None,
            "feature_pattern": feature_pattern,
            "feature_layout": feature_layout,
        },
        "safety": {
            "evaluation_performed": False,
            "evaluation_ground_truth_opened": False,
            "evaluation_ground_truth_use": "none_prediction_generation_only",
            "allowed_mask_inputs": [
                "NVOS positive/negative reference scribbles",
                "SPIn-NeRF reference binary mask",
            ],
        },
        "scenes": scene_records,
    }
    manifest_path = output_root / output_manifest_name
    _atomic_write_json(manifest_path, result, overwrite=overwrite)
    result["prediction_manifest_path"] = str(manifest_path)
    return result


__all__ = [
    "DEFAULT_FEATURE_PATTERN",
    "FEATURE_LAYOUTS",
    "FeatureReadoutError",
    "PREDICTION_MANIFEST_NAME",
    "cosine_margin_scores",
    "generate_feature_readout_predictions",
    "load_feature_map",
]
