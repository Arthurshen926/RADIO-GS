#!/usr/bin/env python3
"""Fail-closed CPU audit of the isolated exact-LUDVIG-vendored stack.

This verifier intentionally does not execute a forward pass or make a CUDA
device visible.  It binds the environment, upstream source, compiled sm86
extensions, and official DINOv2 checkpoint.  The vendored LUDVIG DINO model
does not implement register tokens, so the only accepted checkpoint/model
key difference is the frozen ``register_tokens`` tensor with shape
``[1, 4, 1536]``.  Every other missing, unexpected, or shape-mismatched key is
an error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Sequence


EXPECTED_UPSTREAM_COMMIT = "4461fc515439bb498a75d71738a1e73cf7a452ed"
EXPECTED_CHECKPOINT_NAME = "dinov2_vitg14_reg4_pretrain.pth"
EXPECTED_CHECKPOINT_SIZE = 4_546_140_349
EXPECTED_CHECKPOINT_SHA256 = (
    "746ecb8c6301c645c5c855be91687d274587d6e48fdaec4a729753160b34a283"
)
EXPECTED_REGISTER_KEY = "register_tokens"
EXPECTED_REGISTER_SHAPE = (1, 4, 1536)
EXPECTED_DIFF_RASTERIZER_TREE = "3df1b08faa4057c3e2bc4add4a5dae0b5e0a7386"
EXPECTED_SIMPLE_KNN_TREE = "bb973efb2459dc8df0d885ffdc233dfc5d2ba984"
EXPECTED_VERSIONS = {
    "python": "3.11.9",
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "xformers": "0.0.27.post2+cu118",
}
SOURCE_FILES = (
    "environment.yml",
    "dinov2/model.py",
    "dinov2/setup.py",
    "dinov2/dino_utils.py",
    "dinov2/models/__init__.py",
    "dinov2/models/vision_transformer.py",
    "dinov2/configs/vitg14_pretrain.yaml",
    "predictors/dino.py",
)


class AuditError(RuntimeError):
    """Raised when the isolated reproduction stack is not exactly bound."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(f"Unable to inspect git checkout {root}") from error
    return result.stdout.strip()


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise AuditError(f"Missing distribution {name!r}") from error


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--upstream", type=Path, default=Path("/root/baselines/LUDVIG")
    )
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/root/model_weights/ludvig/dinov2_vitg14_reg4_pretrain.pth"
        ),
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AuditError("Set CUDA_VISIBLE_DEVICES to the empty string for this audit")

    # Delay every torch-related import until the no-device contract is checked.
    import torch
    import torchvision
    import xformers
    from xformers import _cpp_lib as xformers_cpp

    if torch.cuda.is_initialized():
        raise AuditError("CUDA was initialized before the CPU audit began")

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "xformers": xformers.__version__,
    }
    for name, expected in EXPECTED_VERSIONS.items():
        require_equal(versions[name], expected, name)
    require_equal(versions["torch_cuda_build"], "11.8", "PyTorch CUDA build")
    if xformers_cpp._cpp_library_load_exception is not None:
        raise AuditError(
            f"xFormers extension did not load: {xformers_cpp._cpp_library_load_exception}"
        )
    xformers_metadata = xformers_cpp._build_metadata.metadata
    require_equal(
        xformers_metadata["version"]["torch"],
        "2.4.0+cu118",
        "xFormers torch build",
    )
    require_equal(
        xformers_metadata["version"]["cuda"], 1108, "xFormers CUDA build"
    )

    upstream = args.upstream.resolve()
    checkpoint = args.checkpoint.resolve()
    require_equal(
        git_output(upstream, "rev-parse", "HEAD"),
        EXPECTED_UPSTREAM_COMMIT,
        "LUDVIG commit",
    )
    tree_hashes = {
        "diff_gaussian_rasterization": git_output(
            upstream,
            "rev-parse",
            "HEAD:gaussiansplatting/submodules/diff-gaussian-rasterization",
        ),
        "simple_knn": git_output(
            upstream,
            "rev-parse",
            "HEAD:gaussiansplatting/submodules/simple-knn",
        ),
    }
    require_equal(
        tree_hashes["diff_gaussian_rasterization"],
        EXPECTED_DIFF_RASTERIZER_TREE,
        "diff rasterizer source tree",
    )
    require_equal(
        tree_hashes["simple_knn"],
        EXPECTED_SIMPLE_KNN_TREE,
        "simple-knn source tree",
    )
    source_hashes = {
        relative: sha256_file(upstream / relative) for relative in SOURCE_FILES
    }

    require_equal(checkpoint.name, EXPECTED_CHECKPOINT_NAME, "checkpoint basename")
    if not checkpoint.is_file():
        raise AuditError(f"Missing checkpoint: {checkpoint}")
    require_equal(checkpoint.stat().st_size, EXPECTED_CHECKPOINT_SIZE, "checkpoint size")
    require_equal(
        sha256_file(checkpoint), EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA-256"
    )

    # Import the compiled extensions, but do not call any CUDA kernel.
    import diff_gaussian_rasterization._C as rasterizer_c
    import sam2._C as sam2_c
    import simple_knn._C as simple_knn_c

    extension_modules = {
        "diff_gaussian_rasterization": Path(rasterizer_c.__file__).resolve(),
        "simple_knn": Path(simple_knn_c.__file__).resolve(),
        "sam2": Path(sam2_c.__file__).resolve(),
    }
    if not hasattr(rasterizer_c, "rasterize_gaussians"):
        raise AuditError("Rasterizer extension lacks rasterize_gaussians")
    if not any("weight" in name.lower() for name in dir(rasterizer_c)):
        raise AuditError("LUDVIG rasterizer extension lacks apply-weights support")
    if not hasattr(simple_knn_c, "distCUDA2"):
        raise AuditError("simple-knn extension lacks distCUDA2")
    cuobjdump = Path(sys.prefix) / "bin" / "cuobjdump"
    if not cuobjdump.is_file():
        raise AuditError(f"Missing environment-local cuobjdump: {cuobjdump}")
    extension_architectures: dict[str, list[str]] = {}
    for name, path in extension_modules.items():
        try:
            listing = subprocess.run(
                [str(cuobjdump), "--list-elf", str(path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise AuditError(f"Unable to inspect CUDA architecture for {name}") from error
        architectures = sorted(set(re.findall(r"\.sm_([0-9]+)\.cubin", listing)))
        require_equal(architectures, ["86"], f"{name} CUDA architectures")
        extension_architectures[name] = [f"sm_{value}" for value in architectures]

    # The vendored model's constructor uses Tensor.item() on a drop-path
    # linspace.  Force only that tiny linspace onto CPU while all parameters
    # are created on the meta device, avoiding a multi-gigabyte allocation.
    sys.path.insert(0, str(upstream))
    from dinov2.models import build_model_from_cfg
    from dinov2.setup import get_cfg_from_args

    config = get_cfg_from_args(str(upstream / "dinov2/configs/vitg14_pretrain.yaml"))
    original_linspace = torch.linspace

    def cpu_linspace(*linspace_args: Any, **linspace_kwargs: Any):
        linspace_kwargs["device"] = "cpu"
        return original_linspace(*linspace_args, **linspace_kwargs)

    torch.linspace = cpu_linspace
    try:
        with torch.device("meta"):
            model, _dimension = build_model_from_cfg(config, only_teacher=True)
    finally:
        torch.linspace = original_linspace

    model_state = model.state_dict()
    checkpoint_state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if "teacher" in checkpoint_state:
        checkpoint_state = checkpoint_state["teacher"]
    checkpoint_state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in checkpoint_state.items()
    }
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatch = sorted(
        key
        for key in model_keys & checkpoint_keys
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
    )
    register_shape = tuple(checkpoint_state[EXPECTED_REGISTER_KEY].shape)
    require_equal(missing, [], "DINO missing keys")
    require_equal(shape_mismatch, [], "DINO shape-mismatched keys")
    require_equal(
        unexpected, [EXPECTED_REGISTER_KEY], "DINO unexpected checkpoint keys"
    )
    require_equal(register_shape, EXPECTED_REGISTER_SHAPE, "register token shape")
    require_equal(
        hasattr(model, EXPECTED_REGISTER_KEY),
        False,
        "vendored model register-token support",
    )

    if torch.cuda.is_initialized():
        raise AuditError("CUDA was initialized during the CPU audit")
    return {
        "schema_version": "ludvig_pfpr_cpu_environment_audit_v1",
        "status": "pass_exact_ludvig_vendored_no_gpu_execution",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized": torch.cuda.is_initialized(),
        "environment_prefix": sys.prefix,
        "versions": versions,
        "distribution_versions": {
            name: distribution_version(name)
            for name in (
                "diff-gaussian-rasterization",
                "simple-knn",
                "segment-anything",
                "SAM-2",
            )
        },
        "xformers_build_metadata": xformers_metadata,
        "upstream": {
            "path": str(upstream),
            "commit": EXPECTED_UPSTREAM_COMMIT,
            "source_file_sha256": source_hashes,
            "extension_source_tree_git": tree_hashes,
            "working_tree_porcelain": git_output(
                upstream, "status", "--porcelain", "--untracked-files=normal"
            ).splitlines(),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": EXPECTED_CHECKPOINT_SHA256,
            "tensor_key_count": len(checkpoint_state),
            "register_tokens_shape": list(register_shape),
        },
        "exact_ludvig_vendored_weight_contract": {
            "model_state_key_count": len(model_state),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatch_keys": shape_mismatch,
            "vendored_model_supports_register_tokens": False,
            "interpretation": (
                "Exact LUDVIG silently discards the frozen register_tokens key; "
                "a true reg4 model is a corrected variant, not code-exact LUDVIG."
            ),
        },
        "compiled_extensions": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "target_architectures": extension_architectures[name],
                "kernel_executed": False,
            }
            for name, path in extension_modules.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
