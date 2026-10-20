"""Explicit inference-package support for evaluation, never training resume.

Legacy calls still use the existing checkpoint loader. A verified package
reader is scoped to one evaluation; repeated model/head reads reuse one CPU
snapshot. It is not a process-global cache and never changes file identity.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
import re
from types import SimpleNamespace

import torch

from radio_gs.utils.checkpoint_io import load_trusted_checkpoint
from radio_gs.utils.inference_checkpoint import load_inference_checkpoint


class InferencePackageReader:
    def __init__(self, path, *, expected_sha256, resume=False):
        if resume:
            raise ValueError("inference-only package cannot resume training")
        validate_inference_options(True, expected_sha256)
        self.path = os.path.abspath(os.fspath(path))
        self.expected_sha256 = expected_sha256
        self._state = None
        self.provenance = None
        self.record = None

    def bind_config(self, config):
        self.load(self.path)
        recorded = self.provenance.get("config_record", {}).get("config")
        if not isinstance(recorded, dict):
            raise ValueError("package lacks a provenance-bound configuration")
        exact = {"architecture", "latent_dim", "radio_feature_dim", "bottleneck_dim",
                 "codec_type", "dual_stream", "symmetric_decoder", "train_sh", "use_refiner",
                 "fine_dim", "coarse_dim", "feature_height", "feature_width", "image_height",
                 "image_width", "fx", "fy", "cx", "cy", "max_channels_per_chunk"}
        prefixes = ("hybrid_", "hash_", "codec_", "featsharp_", "refiner_", "point_summary_adapter_")
        mismatches = [k for k, v in recorded.items()
                      if (k in exact or k.startswith(prefixes))
                      and hasattr(config, k) and getattr(config, k) != v]
        if mismatches:
            raise ValueError("inference configuration mismatch: " + ", ".join(sorted(mismatches)))
        # Preserve caller data/protocol paths, but construction semantics come
        # from the exact configuration sealed inside the verified package.
        values = dict(vars(config))
        values.update({k: v for k, v in recorded.items()
                       if k in exact or k.startswith(prefixes)})
        return SimpleNamespace(**values)

    def load(self, path, *, resume=False):
        if resume:
            raise ValueError("inference-only package cannot resume training")
        if os.path.abspath(os.fspath(path)) != self.path:
            raise ValueError("inference reader is bound to a different checkpoint")
        if self._state is None:
            self._state, self.provenance, self.record = load_inference_checkpoint(
                self.path, expected_sha256=self.expected_sha256)
        return self._state


def validate_inference_options(inference_package, expected_sha256, *, checkpoint_path=None):
    if inference_package:
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("inference package requires an external lowercase SHA-256")
        if checkpoint_path is not None and ("{" in str(checkpoint_path) or "}" in str(checkpoint_path)):
            raise ValueError("inference-package mode requires one concrete checkpoint, not a scene template")
    elif expected_sha256:
        # Existing programmatic legacy SHA-bound loader remains available;
        # these command-line flags deliberately require an explicit format.
        raise ValueError("--expected_checkpoint_sha256 requires --inference_package")


def add_inference_checkpoint_arguments(parser):
    parser.add_argument("--inference_package", "--inference-package", action="store_true",
                        help="Explicit inference-only checkpoint; never usable with resume")
    parser.add_argument("--expected_checkpoint_sha256", "--checkpoint-sha256", default=None,
                        help="External SHA-256 of the inference package (not its source checkpoint)")


def inference_reader_from_args(args, parser):
    try:
        validate_inference_options(args.inference_package, args.expected_checkpoint_sha256,
                                   checkpoint_path=args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.inference_package:
        return None
    if not args.checkpoint:
        parser.error("--inference_package requires --checkpoint")
    reader = InferencePackageReader(args.checkpoint, expected_sha256=args.expected_checkpoint_sha256)
    reader.load(args.checkpoint)  # fail before text encoders, datasets or caches
    return reader


def load_evaluation_checkpoint(path, *, checkpoint_reader=None, expected_sha256=None, **kwargs):
    if checkpoint_reader is None:
        return load_trusted_checkpoint(path, expected_sha256=expected_sha256, **kwargs)
    if not isinstance(checkpoint_reader, InferencePackageReader):
        raise TypeError("checkpoint_reader must be an explicit InferencePackageReader")
    if expected_sha256 is not None and expected_sha256 != checkpoint_reader.expected_sha256:
        raise ValueError("conflicting package SHA-256")
    if set(kwargs) - {"map_location"}:
        raise TypeError("inference package only supports evaluation map_location")
    # Modules' existing load_state_dict performs device placement. Keep the
    # verified reusable source snapshot on CPU to avoid duplicated GPU states.
    return checkpoint_reader.load(path)


def evaluation_checkpoint_sha256(path, reader, legacy_hash):
    if reader is None:
        return legacy_hash(path)
    reader.load(path)
    return reader.record["sha256"]


def evaluation_package_identity(reader):
    if reader is None:
        return {}
    reader.load(reader.path)
    config = reader.provenance["config_record"]["config"]
    return {
        "inference_package_sha256": reader.record["sha256"],
        "inference_package_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "source_training_checkpoint_sha256": reader.provenance["source_checkpoint"]["sha256"],
    }


def initialize_package_geometry(model, state):
    """Allocate checkpoint-shaped geometry without opening the original PLY."""
    required = ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest")
    feature = "_latent" if hasattr(model, "_latent") else "_feature"
    missing = [k for k in (*required, feature) if k not in state]
    if missing:
        raise ValueError("inference package lacks geometry: " + ", ".join(missing))
    n = int(state["_xyz"].shape[0])
    for name in (*required, feature):
        if not torch.is_tensor(state[name]) or state[name].ndim < 2 or int(state[name].shape[0]) != n:
            raise ValueError("inconsistent geometry rows: " + name)
    for name, width in (("_xyz", 3), ("_rotation", 4), ("_opacity", 1)):
        if tuple(state[name].shape) != (n, width):
            raise ValueError("invalid geometry shape: " + name)
    if state["_scaling"].shape[1] not in (2, 3):
        raise ValueError("invalid geometry scaling")
    for name in required:
        if name not in model._buffers:
            raise ValueError("unsupported geometry buffer: " + name)
        model._buffers[name] = torch.empty_like(state[name], device="cpu")
    for name in (feature, "_sh_dc_param", "_sh_rest_param"):
        if name in state:
            setattr(model, name, torch.nn.Parameter(torch.empty_like(state[name], device="cpu")))
    rest = state["_features_rest"]
    model._sh_degree = int(math.sqrt(int(rest.shape[1]) + 1)) - 1 if rest.ndim == 3 else 0


def package_use_2dgs(config, state):
    scaling = state["_scaling"]
    if scaling.ndim != 2 or scaling.shape[1] not in (2, 3):
        raise ValueError("unsupported checkpoint scaling geometry")
    actual = int(scaling.shape[1]) == 2
    # Legacy resolve_use_2dgs prefers actual PLY scale width over the config
    # flag; the checkpoint geometry is the corresponding package authority.
    return actual
