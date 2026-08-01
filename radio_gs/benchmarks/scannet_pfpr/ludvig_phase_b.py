"""Audited LUDVIG DINOv2 scene-feature and PCA phase for PFPR.

This module reproduces the feature/PCA portion of LUDVIG's
``predictors.dino.DINOv2Dataset``.  The released LUDVIG code builds a vendored
ViT-g/14 without register-token support and silently ignores the official
checkpoint's sole ``register_tokens`` tensor through ``strict=False``.  Here
that code-exact incompatibility is made explicit and fail-closed: precisely
that frozen tensor may be discarded and every other key or shape difference
is rejected.  Phase B is result-ineligible: it produces only hash-bound scene
tokens and a frozen transform for later query crops.  Uplift, scoring, and
evaluation remain separate fail-closed phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as torch_functional

from radio_gs.benchmarks.scannet_pfpr.ludvig_phase_a import (
    CAMERA_AXES,
    FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE,
    LUDVIG_AUDITED_COMMIT,
    OFFICIAL_DINO_CHECKPOINT_NAME,
    OFFICIAL_DINO_CHECKPOINT_SHA256,
    OFFICIAL_DINO_CHECKPOINT_SIZE,
    POSE_CONVENTION,
    SMOKE_VIEW_COUNT,
    UPSTREAM_AUDIT_FILES,
    LudvigPFPRPhaseAError,
    audit_checkpoint,
    audit_ludvig_upstream,
    sha256_file,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import canonical_json_sha256


PHASE_B_SCHEMA_VERSION = "ludvig_pfpr_dino_pca_v1"
PHASE_B_STATUS = "phase_b_dino_pca_complete_later_phases_not_implemented"
LUDVIG_VENDORED_MODEL_NAME = "vit_giant2_without_register_tokens"
LUDVIG_VENDORED_ARCH = "vit_giant2"
OFFICIAL_DINOV2_EMBED_DIM = 1536
OFFICIAL_DINOV2_REGISTER_TOKENS = 4
LUDVIG_VENDORED_REGISTER_TOKENS = 0
OFFICIAL_REGISTER_TOKENS_KEY = "register_tokens"
OFFICIAL_REGISTER_TOKENS_SHAPE = (1, 4, OFFICIAL_DINOV2_EMBED_DIM)
LUDVIG_PATCH_SIZE = 14
LUDVIG_SLIDING_CROP_SIZE = 840
LUDVIG_SLIDING_STRIDE = 200
LUDVIG_PCA_COMPONENTS = 40
LUDVIG_PCA_SUBSAMPLE = 500_000
LUDVIG_SEED = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_NVIDIA_DRIVER_VERSION = "535.288.01"
EXPECTED_DRIVER_LIBCUDA_SHA256 = (
    "deb92bbc5e2990ded3ca2fd13103d346d2a8dee4374d7f9f64d996e420c0ec9d"
)
DEFAULT_DRIVER_LIBRARY_DIR = Path("/root/baselines/LUDVIG/.driver535")

LUDVIG_VENDORED_DINOV2_SOURCE_FILES = (
    "dinov2/configs/ssl_default_config.yaml",
    "dinov2/configs/vitg14_pretrain.yaml",
    "dinov2/dino_utils.py",
    "dinov2/model.py",
    "dinov2/models/__init__.py",
    "dinov2/models/layers/__init__.py",
    "dinov2/models/layers/attention.py",
    "dinov2/models/layers/block.py",
    "dinov2/models/layers/dino_head.py",
    "dinov2/models/layers/drop_path.py",
    "dinov2/models/layers/layer_scale.py",
    "dinov2/models/layers/mlp.py",
    "dinov2/models/layers/patch_embed.py",
    "dinov2/models/layers/swiglu_ffn.py",
    "dinov2/models/vision_transformer.py",
    "dinov2/setup.py",
    "predictors/dino.py",
)
FROZEN_LUDVIG_VENDORED_DINOV2_SOURCE_TREE_SHA256 = (
    "a21e99cf81af9acaa7b8beff67405ca2ac6535b1a52995c5df06979422898777"
)


class LudvigPFPRPhaseBError(RuntimeError):
    """Raised before publishing a Phase-B artifact when a lock is invalid."""


@dataclass(frozen=True)
class PhaseBConfig:
    phase_a_dir: Path
    expected_phase_a_manifest_sha256: str
    dino_checkpoint: Path
    ludvig_upstream: Path
    source_adapter_ledger: Path
    dinov2_source: Path
    output_dir: Path
    driver_library_dir: Path = DEFAULT_DRIVER_LIBRARY_DIR
    device: str = "cuda:0"
    seed: int = LUDVIG_SEED
    view_count: int = SMOKE_VIEW_COUNT
    input_width: int = 640
    input_height: int = 480
    crop_size: int = LUDVIG_SLIDING_CROP_SIZE
    stride: int = LUDVIG_SLIDING_STRIDE
    patch_size: int = LUDVIG_PATCH_SIZE
    n_components: int = LUDVIG_PCA_COMPONENTS
    pca_subsample: int = LUDVIG_PCA_SUBSAMPLE
    eigval_weighting: bool = True
    expected_patches_per_view: int = 2
    expected_grid_height: int = 34
    expected_grid_width: int = 34
    expected_embedding_dim: int = OFFICIAL_DINOV2_EMBED_DIM
    expected_register_tokens: int = LUDVIG_VENDORED_REGISTER_TOKENS
    expected_checkpoint_size: int = OFFICIAL_DINO_CHECKPOINT_SIZE
    expected_checkpoint_sha256: str = OFFICIAL_DINO_CHECKPOINT_SHA256
    expected_ludvig_commit: str = LUDVIG_AUDITED_COMMIT
    expected_source_adapter_ledger_sha256: str = ""
    expected_dinov2_source_tree_sha256: str = (
        FROZEN_LUDVIG_VENDORED_DINOV2_SOURCE_TREE_SHA256
    )
    upstream_audit_files: tuple[str, ...] = UPSTREAM_AUDIT_FILES
    checkpoint_key: Optional[str] = None
    expected_driver_version: str = EXPECTED_NVIDIA_DRIVER_VERSION
    expected_driver_libcuda_sha256: str = EXPECTED_DRIVER_LIBCUDA_SHA256


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise LudvigPFPRPhaseBError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LudvigPFPRPhaseBError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LudvigPFPRPhaseBError(f"Invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise LudvigPFPRPhaseBError(f"{label} must be a JSON object")
    return payload


def _validate_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    expected_digest = _require_sha256(expected, f"expected {label} hash")
    if not path.is_file():
        raise LudvigPFPRPhaseBError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise LudvigPFPRPhaseBError(
            f"{label} SHA-256 mismatch: expected {expected_digest}, found {actual}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def audit_ludvig_vendored_dinov2_source(
    root: Path,
    *,
    expected_tree_sha256: str = FROZEN_LUDVIG_VENDORED_DINOV2_SOURCE_TREE_SHA256,
    files: Sequence[str] = LUDVIG_VENDORED_DINOV2_SOURCE_FILES,
) -> dict[str, Any]:
    source = root.resolve()
    if not source.is_dir():
        raise LudvigPFPRPhaseBError(f"Missing LUDVIG vendored source: {source}")
    records: list[dict[str, Any]] = []
    for relative in sorted(set(map(str, files))):
        path = source / relative
        if not path.is_file():
            raise LudvigPFPRPhaseBError(
                f"Missing audited LUDVIG vendored source file: {relative}"
            )
        records.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    tree_sha256 = canonical_json_sha256(records)
    expected = _require_sha256(expected_tree_sha256, "LUDVIG vendored DINOv2 source tree")
    if tree_sha256 != expected:
        raise LudvigPFPRPhaseBError(
            "LUDVIG vendored DINOv2 source tree changed: "
            f"expected {expected}, found {tree_sha256}"
        )
    return {
        "path": str(source),
        "model_entrypoint": LUDVIG_VENDORED_MODEL_NAME,
        "audited_files": records,
        "audited_files_sha256": tree_sha256,
    }


def audit_cuda_driver_binding(config: PhaseBConfig) -> dict[str, Any]:
    """Bind the child-only libcuda shim to the loaded kernel driver version."""

    root = config.driver_library_dir.resolve()
    library = root / "libcuda.so.1"
    binding = _validate_hash(
        library,
        config.expected_driver_libcuda_sha256,
        "kernel-compatible libcuda.so.1",
    )
    first_library_path = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0]
    if Path(first_library_path).resolve() != root:
        raise LudvigPFPRPhaseBError(
            "LD_LIBRARY_PATH must begin with the audited driver library directory"
        )
    version_path = Path("/proc/driver/nvidia/version")
    try:
        kernel_version_record = version_path.read_text(encoding="utf-8")
    except OSError as error:
        raise LudvigPFPRPhaseBError("Unable to read NVIDIA kernel version") from error
    if str(config.expected_driver_version) not in kernel_version_record:
        raise LudvigPFPRPhaseBError(
            "NVIDIA kernel driver version changed: expected "
            f"{config.expected_driver_version}"
        )
    binding.update(
        {
            "driver_library_dir": str(root),
            "resolved_library": str(library.resolve()),
            "kernel_driver_version": str(config.expected_driver_version),
            "process_scoped_only": True,
            "system_symlink_modified": False,
        }
    )
    return binding


def _phase_a_source_color_by_frame(
    manifest: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    inventory = manifest.get("source_inventory", {}).get("ordered_inventory", [])
    output: dict[int, Mapping[str, Any]] = {}
    for item in inventory:
        if not isinstance(item, Mapping):
            raise LudvigPFPRPhaseBError("Phase-A source inventory is invalid")
        frame_id = int(item.get("frame_id", -1))
        color = item.get("color")
        if frame_id in output or not isinstance(color, Mapping):
            raise LudvigPFPRPhaseBError("Phase-A source color inventory is invalid")
        output[frame_id] = color
    return output


def audit_phase_a_attempt(config: PhaseBConfig) -> dict[str, Any]:
    root = config.phase_a_dir.resolve()
    manifest_path = root / "run_manifest.json"
    manifest_binding = _validate_hash(
        manifest_path,
        config.expected_phase_a_manifest_sha256,
        "Phase-A run manifest",
    )
    manifest = _load_json(manifest_path, "Phase-A run manifest")
    if manifest.get("schema_version") != "ludvig_pfpr_phase_a_v1":
        raise LudvigPFPRPhaseBError("Phase-A schema changed")
    if manifest.get("status") != "phase_a_complete_phase_b_available_not_run":
        raise LudvigPFPRPhaseBError("Phase-A is not eligible to feed Phase B")
    for key in ("result_eligible", "official_ludvig_reproduction", "paper_metric_comparable"):
        if manifest.get(key) is not False:
            raise LudvigPFPRPhaseBError(f"Phase-A result flag changed: {key}")
    scene_id = str(manifest.get("scene_id", ""))
    if not scene_id:
        raise LudvigPFPRPhaseBError("Phase-A scene_id is empty")
    if int(config.view_count) != SMOKE_VIEW_COUNT and int(config.view_count) <= 0:
        raise LudvigPFPRPhaseBError("Phase-B test view_count must be positive")

    selection = manifest.get("view_selection", {})
    try:
        all_frame_ids = [int(value) for value in selection["ordered_frame_ids"]]
    except (KeyError, TypeError, ValueError) as error:
        raise LudvigPFPRPhaseBError("Phase-A ordered view IDs are invalid") from error
    if (
        len(all_frame_ids) != SMOKE_VIEW_COUNT
        or len(set(all_frame_ids)) != len(all_frame_ids)
        or selection.get("ordered_frame_ids_sha256")
        != canonical_json_sha256(all_frame_ids)
    ):
        raise LudvigPFPRPhaseBError("Phase-A ordered view lock changed")
    if int(config.view_count) <= 0 or int(config.view_count) > len(all_frame_ids):
        raise LudvigPFPRPhaseBError(
            f"Phase-B view_count must be within 1..{len(all_frame_ids)}"
        )
    frame_ids = all_frame_ids[: int(config.view_count)]

    ledger_expected = config.expected_source_adapter_ledger_sha256
    if not ledger_expected:
        ledger_expected = FROZEN_SOURCE_ADAPTER_LEDGER_SHA256_BY_SCENE.get(
            scene_id, ""
        )
    ledger_binding = _validate_hash(
        config.source_adapter_ledger.resolve(),
        ledger_expected,
        "source adapter ledger",
    )
    phase_a_ledger = manifest.get("source_adapter_ledger", {})
    if phase_a_ledger.get("sha256") != ledger_binding["sha256"]:
        raise LudvigPFPRPhaseBError("Phase-A source adapter ledger lock changed")
    if phase_a_ledger.get("coverage_prefix_sha256") != canonical_json_sha256(
        all_frame_ids
    ):
        raise LudvigPFPRPhaseBError("Phase-A ledger coverage lock changed")

    try:
        checkpoint = audit_checkpoint(
            config.dino_checkpoint,
            expected_size=config.expected_checkpoint_size,
            expected_sha256=config.expected_checkpoint_sha256,
        )
    except LudvigPFPRPhaseAError as error:
        raise LudvigPFPRPhaseBError(str(error)) from error
    phase_a_checkpoint = manifest.get("checkpoint", {})
    if (
        phase_a_checkpoint.get("sha256") != checkpoint["sha256"]
        or int(phase_a_checkpoint.get("bytes", -1)) != checkpoint["bytes"]
    ):
        raise LudvigPFPRPhaseBError("Phase-A checkpoint lock changed")

    try:
        upstream = audit_ludvig_upstream(
            config.ludvig_upstream,
            expected_commit=config.expected_ludvig_commit,
            audited_files=config.upstream_audit_files,
        )
    except LudvigPFPRPhaseAError as error:
        raise LudvigPFPRPhaseBError(str(error)) from error
    phase_a_upstream = manifest.get("ludvig_upstream", {})
    if (
        phase_a_upstream.get("commit") != upstream["commit"]
        or phase_a_upstream.get("source_sha256") != upstream["source_sha256"]
    ):
        raise LudvigPFPRPhaseBError("Phase-A LUDVIG source lock changed")

    dimensions = manifest.get("camera_intrinsics", {}).get("image_dimensions")
    if dimensions != [int(config.input_width), int(config.input_height)]:
        raise LudvigPFPRPhaseBError("Phase-A staged image dimensions changed")
    source_colors = _phase_a_source_color_by_frame(manifest)
    staged_all = manifest.get("colmap_staging", {}).get("staged_images", [])
    if not isinstance(staged_all, list) or len(staged_all) != len(all_frame_ids):
        raise LudvigPFPRPhaseBError("Phase-A staged image inventory changed")
    staged = staged_all[: int(config.view_count)]
    views: list[dict[str, Any]] = []
    for rank, (frame_id, item) in enumerate(zip(frame_ids, staged)):
        if not isinstance(item, Mapping):
            raise LudvigPFPRPhaseBError("Phase-A staged image record is invalid")
        name = str(item.get("staged_name", ""))
        if (
            int(item.get("rank", -1)) != rank
            or int(item.get("frame_id", -1)) != frame_id
            or not name
            or Path(name).name != name
        ):
            raise LudvigPFPRPhaseBError("Phase-A staged image ordering changed")
        image_path = root / "staging" / "colmap" / "images" / name
        if not image_path.is_file():
            raise LudvigPFPRPhaseBError(f"Missing Phase-A staged image: {image_path}")
        expected_color = source_colors.get(frame_id)
        if expected_color is None:
            raise LudvigPFPRPhaseBError("Phase-A source/staged frame domains disagree")
        image_hash = sha256_file(image_path)
        if image_hash != expected_color.get("sha256"):
            raise LudvigPFPRPhaseBError(
                f"Phase-A staged image mutated for frame {frame_id}"
            )
        with Image.open(image_path) as image:
            if image.size != (config.input_width, config.input_height):
                raise LudvigPFPRPhaseBError(
                    f"Phase-A staged image dimensions changed for frame {frame_id}"
                )
            if image.mode != "RGB":
                raise LudvigPFPRPhaseBError(
                    f"Phase-A staged image mode changed for frame {frame_id}"
                )
        views.append(
            {
                "rank": rank,
                "frame_id": frame_id,
                "staged_name": name,
                "path": str(image_path),
                "bytes": image_path.stat().st_size,
                "sha256": image_hash,
            }
        )
    return {
        "root": str(root),
        "manifest": manifest_binding,
        "scene_id": scene_id,
        "ordered_frame_ids": frame_ids,
        "ordered_frame_ids_sha256": canonical_json_sha256(frame_ids),
        "views": views,
        "views_sha256": canonical_json_sha256(views),
        "checkpoint": checkpoint,
        "source_adapter_ledger": ledger_binding,
        "ludvig_upstream": upstream,
    }


def _checkpoint_state_dict(
    payload: Any, checkpoint_key: Optional[str]
) -> tuple[Mapping[str, Any], Optional[str]]:
    selected_key = checkpoint_key
    value = payload
    if checkpoint_key is not None:
        if not isinstance(payload, Mapping) or checkpoint_key not in payload:
            raise LudvigPFPRPhaseBError(
                f"Checkpoint has no requested key {checkpoint_key!r}"
            )
        value = payload[checkpoint_key]
    elif isinstance(payload, Mapping):
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            value = payload["state_dict"]
            selected_key = "state_dict"
        elif "teacher" in payload and isinstance(payload["teacher"], Mapping):
            value = payload["teacher"]
            selected_key = "teacher"
    if not isinstance(value, Mapping) or not value:
        raise LudvigPFPRPhaseBError("Checkpoint state_dict is empty or invalid")
    return value, selected_key


def _strip_checkpoint_key(key: str) -> tuple[str, list[str]]:
    output = str(key)
    removed: list[str] = []
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "backbone."):
            if output.startswith(prefix):
                output = output[len(prefix) :]
                removed.append(prefix[:-1])
                changed = True
    return output, removed


def load_checkpoint_exact_ludvig_vendored(
    model: Any,
    checkpoint_path: Path,
    *,
    checkpoint_key: Optional[str] = None,
) -> dict[str, Any]:
    """Reproduce LUDVIG's sole incompatibility, rejecting every other one.

    Released LUDVIG loads the official reg4 checkpoint with ``strict=False``
    into a vendored model that has no register-token parameter.  We first
    observe that exact incompatibility, then remove only the frozen tensor and
    repeat the load with ``strict=True`` so no other mismatch can be hidden.
    """

    try:
        load_kwargs: dict[str, Any] = {
            "map_location": "cpu",
            "weights_only": True,
        }
        checkpoint_mmap = "mmap" in inspect.signature(torch.load).parameters
        if checkpoint_mmap:
            load_kwargs["mmap"] = True
        payload = torch.load(str(checkpoint_path), **load_kwargs)
    except Exception as error:
        raise LudvigPFPRPhaseBError(
            f"Unable to deserialize DINOv2 checkpoint: {checkpoint_path}"
        ) from error
    raw, selected_key = _checkpoint_state_dict(payload, checkpoint_key)
    stripped: dict[str, Any] = {}
    prefix_counts = {"module": 0, "backbone": 0}
    for raw_key, value in raw.items():
        key, removed = _strip_checkpoint_key(str(raw_key))
        if key in stripped:
            raise LudvigPFPRPhaseBError(
                f"Checkpoint key stripping collision at {key!r}"
            )
        stripped[key] = value
        for prefix in removed:
            prefix_counts[prefix] += 1
    try:
        incompatibility = model.load_state_dict(stripped, strict=False)
    except Exception as error:
        raise LudvigPFPRPhaseBError("DINOv2 checkpoint tensor load failed") from error
    missing = sorted(map(str, incompatibility.missing_keys))
    unexpected = sorted(map(str, incompatibility.unexpected_keys))
    if missing or unexpected != [OFFICIAL_REGISTER_TOKENS_KEY]:
        raise LudvigPFPRPhaseBError(
            "DINOv2 checkpoint differs from exact LUDVIG vendored contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    register_tokens = stripped.get(OFFICIAL_REGISTER_TOKENS_KEY)
    if not isinstance(register_tokens, torch.Tensor) or tuple(
        register_tokens.shape
    ) != OFFICIAL_REGISTER_TOKENS_SHAPE:
        raise LudvigPFPRPhaseBError(
            "Official register_tokens tensor shape changed: "
            f"expected {OFFICIAL_REGISTER_TOKENS_SHAPE}, found "
            f"{getattr(register_tokens, 'shape', None)}"
        )
    filtered = dict(stripped)
    del filtered[OFFICIAL_REGISTER_TOKENS_KEY]
    try:
        model.load_state_dict(filtered, strict=True)
    except Exception as error:
        raise LudvigPFPRPhaseBError(
            "DINOv2 strict=True verification failed after removing only "
            "register_tokens"
        ) from error
    return {
        "checkpoint_key": selected_key,
        "checkpoint_mmap": checkpoint_mmap,
        "raw_key_count": len(raw),
        "stripped_key_count": len(stripped),
        "loaded_key_count": len(filtered),
        "stripped_prefix_counts": prefix_counts,
        "missing_keys": [],
        "unexpected_keys": [OFFICIAL_REGISTER_TOKENS_KEY],
        "ignored_key_shapes": {
            OFFICIAL_REGISTER_TOKENS_KEY: list(register_tokens.shape)
        },
        "vendored_model_supports_register_tokens": False,
        "released_loader_strict": False,
        "verification_after_single_key_filter_strict": True,
        "interpretation": (
            "code_exact_LUDVIG_discards_official_register_tokens; "
            "true_reg4_is_a_corrected_variant_not_the_primary_reproduction"
        ),
    }


def build_ludvig_vendored_vitg14(source_root: Path) -> Any:
    """Build LUDVIG's vendored no-register ViT-g/14 without loading weights."""

    source = source_root.resolve()
    loaded = sys.modules.get("dinov2")
    if loaded is not None:
        module_file = Path(str(getattr(loaded, "__file__", ""))).resolve()
        try:
            module_file.relative_to(source)
        except ValueError as error:
            raise LudvigPFPRPhaseBError(
                f"A different dinov2 package is already imported: {module_file}"
            ) from error
    inserted = str(source) not in sys.path
    if inserted:
        sys.path.insert(0, str(source))
    try:
        models_module = importlib.import_module("dinov2.models")
        setup_module = importlib.import_module("dinov2.setup")
        config = setup_module.get_cfg_from_args(
            str(source / "dinov2" / "configs" / "vitg14_pretrain.yaml")
        )
        model, _ = models_module.build_model_from_cfg(config, only_teacher=True)
    except Exception as error:
        raise LudvigPFPRPhaseBError(
            f"Unable to build {LUDVIG_VENDORED_MODEL_NAME} from {source}"
        ) from error
    finally:
        if inserted and sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
    return model


def _model_patch_size(model: Any) -> int:
    value = getattr(model, "patch_size", None)
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise LudvigPFPRPhaseBError("DINOv2 patch size is not square")
        value = value[0]
    return int(value)


def audit_model_architecture(model: Any, config: PhaseBConfig) -> dict[str, Any]:
    patch_size = _model_patch_size(model)
    embed_dim = int(getattr(model, "embed_dim", -1))
    supports_register_tokens = hasattr(model, OFFICIAL_REGISTER_TOKENS_KEY)
    register_tokens = int(getattr(model, "num_register_tokens", 0))
    if patch_size != int(config.patch_size):
        raise LudvigPFPRPhaseBError("DINOv2 patch size changed")
    if embed_dim != int(config.expected_embedding_dim):
        raise LudvigPFPRPhaseBError("DINOv2 embedding dimension changed")
    if register_tokens != int(config.expected_register_tokens):
        raise LudvigPFPRPhaseBError("DINOv2 register-token count changed")
    if supports_register_tokens:
        raise LudvigPFPRPhaseBError(
            "Exact LUDVIG vendored model unexpectedly supports register tokens"
        )
    return {
        "architecture": LUDVIG_VENDORED_ARCH,
        "entrypoint": LUDVIG_VENDORED_MODEL_NAME,
        "patch_size": patch_size,
        "embedding_dim": embed_dim,
        "register_tokens": register_tokens,
        "supports_register_tokens": supports_register_tokens,
        "intermediate_layer": "last_of_n_last_4",
    }


class DinoPatchPredictor:
    def __init__(self, model: Any, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, patch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.device.type == "cuda":
                context = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
            else:
                context = torch.autocast(device_type="cpu", enabled=False)
            with context:
                outputs = self.model.get_intermediate_layers(
                    patch[None], n=4, return_class_token=True, reshape=True
                )
        if not outputs:
            raise LudvigPFPRPhaseBError("DINOv2 returned no intermediate layers")
        last = outputs[-1]
        feature = last[0] if isinstance(last, (tuple, list)) else last
        if feature.ndim == 4 and feature.shape[0] == 1:
            feature = feature[0]
        if feature.ndim != 3:
            raise LudvigPFPRPhaseBError(
                f"DINOv2 patch feature shape is invalid: {tuple(feature.shape)}"
            )
        return feature.float()


def _adjusted_stride(stride: int, crop_size: int, dimension: int) -> int:
    if dimension == crop_size:
        return int(stride)
    intervals = int(np.ceil((dimension - crop_size) / float(stride)))
    if intervals <= 0:
        raise LudvigPFPRPhaseBError("Sliding-window interval count is invalid")
    return int(np.ceil((dimension - crop_size) / float(intervals)))


def ludvig_sliding_plan(
    height: int,
    width: int,
    *,
    patch_size: int = LUDVIG_PATCH_SIZE,
    crop_size: int = LUDVIG_SLIDING_CROP_SIZE,
    stride: int = LUDVIG_SLIDING_STRIDE,
) -> dict[str, Any]:
    if min(height, width, patch_size, crop_size, stride) <= 0:
        raise LudvigPFPRPhaseBError("Sliding-window dimensions must be positive")
    aligned_height = (int(height) // int(patch_size)) * int(patch_size)
    aligned_width = (int(width) // int(patch_size)) * int(patch_size)
    effective_crop = min(int(crop_size), aligned_height, aligned_width)
    if effective_crop % int(patch_size) != 0:
        raise LudvigPFPRPhaseBError("Effective crop is not patch-aligned")
    stride_h = _adjusted_stride(stride, effective_crop, aligned_height)
    stride_w = _adjusted_stride(stride, effective_crop, aligned_width)
    indices: list[list[int]] = []
    for y_raw in range(0, aligned_height - effective_crop + stride_h, stride_h):
        for x_raw in range(0, aligned_width - effective_crop + stride_w, stride_w):
            y = min(y_raw, aligned_height - effective_crop)
            x = min(x_raw, aligned_width - effective_crop)
            indices.append([int(y), int(x)])
    grid = effective_crop // int(patch_size)
    return {
        "input_height": int(height),
        "input_width": int(width),
        "aligned_height": aligned_height,
        "aligned_width": aligned_width,
        "configured_crop_size": int(crop_size),
        "effective_crop_size": effective_crop,
        "configured_stride": int(stride),
        "effective_stride_height": stride_h,
        "effective_stride_width": stride_w,
        "indices_yx": indices,
        "patch_count": len(indices),
        "patch_size": int(patch_size),
        "token_grid_height": grid,
        "token_grid_width": grid,
        "tokens_per_view": len(indices) * grid * grid,
    }


def _image_tensor(path: Path, expected_size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        if image.size != expected_size or image.mode != "RGB":
            raise LudvigPFPRPhaseBError(f"Staged RGB contract changed: {path}")
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def extract_view_tokens(
    image_path: Path,
    predictor: DinoPatchPredictor,
    plan: Mapping[str, Any],
    *,
    expected_embedding_dim: int,
) -> np.ndarray:
    image = _image_tensor(
        image_path, (int(plan["input_width"]), int(plan["input_height"]))
    )
    aligned = torch_functional.interpolate(
        image[None].to(predictor.device),
        size=(int(plan["aligned_height"]), int(plan["aligned_width"])),
        mode="bilinear",
    ).squeeze(0)
    crop = int(plan["effective_crop_size"])
    grid_h = int(plan["token_grid_height"])
    grid_w = int(plan["token_grid_width"])
    patches: list[np.ndarray] = []
    for y, x in plan["indices_yx"]:
        patch = aligned[:, int(y) : int(y) + crop, int(x) : int(x) + crop]
        feature = predictor.predict(patch)
        if tuple(feature.shape) != (expected_embedding_dim, grid_h, grid_w):
            raise LudvigPFPRPhaseBError(
                "DINOv2 token shape changed: "
                f"expected {(expected_embedding_dim, grid_h, grid_w)}, "
                f"found {tuple(feature.shape)}"
            )
        patches.append(feature.permute(1, 2, 0).contiguous().cpu().numpy())
    return np.stack(patches, axis=0).astype(np.float32, copy=False)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_ludvig_scene_pca(
    raw_features: np.ndarray,
    *,
    n_components: int,
    pca_subsample: int,
    seed: int,
    statistics_device: torch.device,
) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_features, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise LudvigPFPRPhaseBError("Raw DINO feature matrix is invalid")
    if int(n_components) <= 0 or int(n_components) > min(raw.shape):
        raise LudvigPFPRPhaseBError("PCA component count is invalid")
    _seed_everything(seed, statistics_device)
    features = torch.from_numpy(np.ascontiguousarray(raw)).to(statistics_device)
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0)
    if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all():
        raise LudvigPFPRPhaseBError("Raw DINO mean/std is non-finite")
    if torch.any(feature_std <= 0):
        raise LudvigPFPRPhaseBError("Raw DINO feature std contains non-positive values")
    features = (features - feature_mean) / feature_std
    pca_on = features
    sampled_indices: Optional[np.ndarray] = None
    if len(features) > int(pca_subsample):
        sampled_indices = np.random.choice(
            range(len(features)), int(pca_subsample), replace=False
        )
        pca_on = features[sampled_indices]
    pca = PCA(n_components=int(n_components))
    pca.fit(pca_on.cpu().numpy())
    pca_mean = torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)).to(
        statistics_device
    )
    components = torch.from_numpy(
        np.asarray(pca.components_, dtype=np.float32)
    ).to(statistics_device)
    projected = (features - pca_mean) @ components.T
    return {
        "feature_mean": feature_mean.cpu().numpy().astype(np.float32),
        "feature_std": feature_std.cpu().numpy().astype(np.float32),
        "pca_mean": np.asarray(pca.mean_, dtype=np.float32),
        "pca_components": np.asarray(pca.components_, dtype=np.float32),
        "pca_singular_values": np.asarray(pca.singular_values_, dtype=np.float32),
        "projected": projected.cpu().numpy().astype(np.float32),
        "sampled_indices": (
            np.asarray(sampled_indices, dtype=np.int64)
            if sampled_indices is not None
            else np.empty((0,), dtype=np.int64)
        ),
    }


def apply_scene_pca_transform(
    raw_tokens: np.ndarray,
    transform: Mapping[str, np.ndarray],
    *,
    eigval_weighting: bool,
) -> np.ndarray:
    raw = np.asarray(raw_tokens, dtype=np.float32)
    if raw.ndim < 2:
        raise LudvigPFPRPhaseBError("Query tokens must have a feature dimension")
    feature_mean = np.asarray(transform["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(transform["feature_std"], dtype=np.float32)
    pca_mean = np.asarray(transform["pca_mean"], dtype=np.float32)
    components = np.asarray(transform["pca_components"], dtype=np.float32)
    singular = np.asarray(transform["pca_singular_values"], dtype=np.float32)
    if raw.shape[-1] != feature_mean.shape[0]:
        raise LudvigPFPRPhaseBError("Query DINO dimension differs from scene transform")
    flattened = raw.reshape(-1, raw.shape[-1])
    projected = ((flattened - feature_mean) / feature_std - pca_mean) @ components.T
    if eigval_weighting:
        projected = projected * singular
    return projected.reshape(*raw.shape[:-1], components.shape[0]).astype(np.float32)


def _save_array(root: Path, relative: str, value: np.ndarray) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    np.save(path, array, allow_pickle=False)
    actual_path = path if path.suffix == ".npy" else path.with_suffix(path.suffix + ".npy")
    return {
        "relative_path": str(actual_path.relative_to(root)),
        "bytes": actual_path.stat().st_size,
        "sha256": sha256_file(actual_path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _manifest_hash(path: Path) -> str:
    return sha256_file(path)


def run_phase_b(
    config: PhaseBConfig,
    *,
    model_factory: Optional[Callable[[], Any]] = None,
    argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Extract scene DINO tokens, fit PCA, and atomically publish Phase B."""

    output = config.output_dir.resolve()
    if output.exists():
        raise LudvigPFPRPhaseBError(f"Refusing to overwrite Phase-B output: {output}")
    if int(config.seed) != LUDVIG_SEED:
        raise LudvigPFPRPhaseBError(f"LUDVIG PFPR Phase B is frozen to seed {LUDVIG_SEED}")
    if int(config.view_count) == SMOKE_VIEW_COUNT:
        frozen_values = (
            config.input_width == 640,
            config.input_height == 480,
            config.crop_size == LUDVIG_SLIDING_CROP_SIZE,
            config.stride == LUDVIG_SLIDING_STRIDE,
            config.patch_size == LUDVIG_PATCH_SIZE,
            config.n_components == LUDVIG_PCA_COMPONENTS,
            config.eigval_weighting is True,
        )
        if not all(frozen_values):
            raise LudvigPFPRPhaseBError("Production Phase-B protocol constants changed")
    device = torch.device(config.device)
    driver_binding: Optional[dict[str, Any]] = None
    if device.type == "cuda":
        driver_binding = audit_cuda_driver_binding(config)
        if not torch.cuda.is_available():
            raise LudvigPFPRPhaseBError("Requested CUDA Phase B but CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats()
    phase_a = audit_phase_a_attempt(config)
    dino_source = audit_ludvig_vendored_dinov2_source(
        config.dinov2_source,
        expected_tree_sha256=config.expected_dinov2_source_tree_sha256,
    )
    plan = ludvig_sliding_plan(
        config.input_height,
        config.input_width,
        patch_size=config.patch_size,
        crop_size=config.crop_size,
        stride=config.stride,
    )
    if (
        int(plan["patch_count"]) != int(config.expected_patches_per_view)
        or int(plan["token_grid_height"]) != int(config.expected_grid_height)
        or int(plan["token_grid_width"]) != int(config.expected_grid_width)
    ):
        raise LudvigPFPRPhaseBError(f"LUDVIG sliding/token geometry changed: {plan}")

    _seed_everything(config.seed, device)
    model = model_factory() if model_factory is not None else build_ludvig_vendored_vitg14(
        config.dinov2_source
    )
    architecture = audit_model_architecture(model, config)
    checkpoint_load = load_checkpoint_exact_ludvig_vendored(
        model, config.dino_checkpoint, checkpoint_key=config.checkpoint_key
    )
    predictor = DinoPatchPredictor(model, device)
    raw_views: list[np.ndarray] = []
    for view in phase_a["views"]:
        raw_views.append(
            extract_view_tokens(
                Path(view["path"]),
                predictor,
                plan,
                expected_embedding_dim=config.expected_embedding_dim,
            )
        )
    raw_shapes = {tuple(value.shape) for value in raw_views}
    expected_raw_shape = (
        config.expected_patches_per_view,
        config.expected_grid_height,
        config.expected_grid_width,
        config.expected_embedding_dim,
    )
    if raw_shapes != {expected_raw_shape}:
        raise LudvigPFPRPhaseBError(f"Per-view raw token shapes changed: {raw_shapes}")
    raw_matrix = np.concatenate(
        [value.reshape(-1, config.expected_embedding_dim) for value in raw_views],
        axis=0,
    ).astype(np.float32, copy=False)
    pca = fit_ludvig_scene_pca(
        raw_matrix,
        n_components=config.n_components,
        pca_subsample=config.pca_subsample,
        seed=config.seed,
        statistics_device=device,
    )
    projected = pca.pop("projected")
    sampled_indices = pca.pop("sampled_indices")
    tokens_per_view = int(plan["tokens_per_view"])
    split_projected = [
        projected[index * tokens_per_view : (index + 1) * tokens_per_view].reshape(
            config.expected_patches_per_view,
            config.expected_grid_height,
            config.expected_grid_width,
            config.n_components,
        )
        for index in range(config.view_count)
    ]
    gpu_memory = None
    if device.type == "cuda":
        gpu_memory = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.phase_b_tmp_", dir=str(output.parent))
    )
    try:
        transform_bindings = {
            "feature_mean": _save_array(
                temporary, "transform/feature_mean.npy", pca["feature_mean"]
            ),
            "feature_std": _save_array(
                temporary, "transform/feature_std.npy", pca["feature_std"]
            ),
            "pca_mean": _save_array(
                temporary, "transform/pca_mean.npy", pca["pca_mean"]
            ),
            "pca_components": _save_array(
                temporary, "transform/pca_components.npy", pca["pca_components"]
            ),
            "pca_singular_values": _save_array(
                temporary,
                "transform/pca_singular_values.npy",
                pca["pca_singular_values"],
            ),
        }
        if sampled_indices.size:
            transform_bindings["pca_sampled_indices"] = _save_array(
                temporary,
                "transform/pca_sampled_indices.npy",
                sampled_indices,
            )
        view_records: list[dict[str, Any]] = []
        singular = pca["pca_singular_values"].reshape(1, 1, 1, -1)
        for view, tokens in zip(phase_a["views"], split_projected):
            stem = f"{int(view['rank']):06d}_{int(view['frame_id']):06d}"
            projected_binding = _save_array(
                temporary, f"views/projected/{stem}.npy", tokens
            )
            weighted = tokens * singular if config.eigval_weighting else tokens.copy()
            weighted_binding = _save_array(
                temporary, f"views/eigval_weighted/{stem}.npy", weighted
            )
            view_records.append(
                {
                    "rank": int(view["rank"]),
                    "frame_id": int(view["frame_id"]),
                    "source_staged_name": view["staged_name"],
                    "source_rgb_sha256": view["sha256"],
                    "raw_token_shape": list(expected_raw_shape),
                    "projected_tokens": projected_binding,
                    "eigval_weighted_tokens": weighted_binding,
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": PHASE_B_SCHEMA_VERSION,
            "status": PHASE_B_STATUS,
            "result_eligible": False,
            "official_ludvig_reproduction": False,
            "paper_metric_comparable": False,
            "scene_id": phase_a["scene_id"],
            "attempt_dir": str(output),
            "argv": list(argv or []),
            "gpu_work_started": device.type == "cuda",
            "device": str(device),
            "cuda_driver_binding": driver_binding,
            "cuda_peak_memory": gpu_memory,
            "seed": int(config.seed),
            "phase_a": phase_a,
            "ludvig_vendored_dinov2_source": dino_source,
            "checkpoint_exact_ludvig_load": checkpoint_load,
            "model": architecture,
            "image_preprocessing": {
                "loader": "LUDVIG_PILtoTorch_equivalent_float32_div255",
                "imagenet_mean": list(IMAGENET_MEAN),
                "imagenet_std": list(IMAGENET_STD),
                "aligned_resize_mode": "torch_bilinear_align_corners_false",
            },
            "sliding_window": plan,
            "pca": {
                "implementation": "sklearn.decomposition.PCA_defaults",
                "sklearn_version": sklearn_version,
                "n_components": int(config.n_components),
                "fit_scope": "all_ordered_scene_view_patch_tokens",
                "raw_feature_count": int(raw_matrix.shape[0]),
                "raw_feature_dimension": int(raw_matrix.shape[1]),
                "raw_standardization_std": "torch_std_unbiased_correction_1",
                "pca_subsample_limit": int(config.pca_subsample),
                "pca_fit_count": int(
                    sampled_indices.size if sampled_indices.size else raw_matrix.shape[0]
                ),
                "eigval_weighting": bool(config.eigval_weighting),
                "eigval_definition": "sklearn_PCA_singular_values_historical_LUDVIG_name",
                "transform_formula": (
                    "projected=((raw-feature_mean)/feature_std-pca_mean)@"
                    "pca_components.T; weighted=projected*pca_singular_values"
                ),
                "query_transform_rule": "reuse_these_scene_arrays_without_refit",
                "artifacts": transform_bindings,
            },
            "ordered_view_ids": phase_a["ordered_frame_ids"],
            "ordered_view_ids_sha256": phase_a["ordered_frame_ids_sha256"],
            "views": view_records,
            "views_sha256": canonical_json_sha256(view_records),
            "phase_status": {
                "phase_a_cpu_staging": "bound_complete",
                "phase_b_dino_scene_features_and_pca": "complete",
                "phase_c_inverse_render_uplift": "not_implemented_fail_closed",
                "phase_d_pfpr_crop_scoring": "not_implemented_fail_closed",
                "phase_e_pfpr_evaluation": "not_run_until_c_and_d_are_audited",
            },
            "fail_closed_reason": (
                "Phase B contains no uplifted Gaussian features, crop scores, or PFPR "
                "metrics; later phases remain unavailable."
            ),
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest_sha256 = _manifest_hash(manifest_path)
        (temporary / "run_manifest.sha256").write_text(
            manifest_sha256 + "\n", encoding="ascii"
        )
        if output.exists():
            raise LudvigPFPRPhaseBError(
                f"Refusing concurrent overwrite of Phase-B output: {output}"
            )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**manifest, "run_manifest_sha256": manifest_sha256}


def load_phase_b_transform(
    phase_b_dir: Path, *, expected_manifest_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = phase_b_dir.resolve()
    manifest_path = root / "run_manifest.json"
    _validate_hash(manifest_path, expected_manifest_sha256, "Phase-B run manifest")
    manifest = _load_json(manifest_path, "Phase-B run manifest")
    if manifest.get("schema_version") != PHASE_B_SCHEMA_VERSION:
        raise LudvigPFPRPhaseBError("Phase-B transform schema changed")
    if manifest.get("status") != PHASE_B_STATUS or manifest.get("result_eligible") is not False:
        raise LudvigPFPRPhaseBError("Phase-B transform manifest status changed")
    bindings = manifest.get("pca", {}).get("artifacts", {})
    required = (
        "feature_mean",
        "feature_std",
        "pca_mean",
        "pca_components",
        "pca_singular_values",
    )
    transform: dict[str, np.ndarray] = {}
    for name in required:
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise LudvigPFPRPhaseBError(f"Missing Phase-B transform artifact: {name}")
        relative = Path(str(binding.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise LudvigPFPRPhaseBError("Phase-B transform path escapes its attempt")
        path = root / relative
        _validate_hash(path, str(binding.get("sha256", "")), f"Phase-B {name}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise LudvigPFPRPhaseBError(f"Invalid Phase-B transform array: {name}") from error
        if list(array.shape) != binding.get("shape") or str(array.dtype) != binding.get(
            "dtype"
        ):
            raise LudvigPFPRPhaseBError(f"Phase-B transform shape/dtype changed: {name}")
        if not np.isfinite(array).all():
            raise LudvigPFPRPhaseBError(f"Phase-B transform is non-finite: {name}")
        transform[name] = array
    return transform, manifest
