"""Explicit, CPU-only inference envelopes for trusted RADIO-GS checkpoints.

Legacy training loaders intentionally cannot consume this envelope directly.
Use load_inference_checkpoint, then the existing model/codec/adapter APIs.
This API never resumes training and never opens a file for writing.
"""
from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import _RestrictedProjectCheckpointPickleModule

SCHEMA = "radiogs.inference_checkpoint.v1"
DROP_KEYS = frozenset({"optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"})


def fingerprint(info):
    return {k: int(getattr(info, "st_" + k)) for k in
            ("dev", "ino", "mode", "nlink", "size", "mtime_ns", "ctime_ns")}


def read_project_checkpoint_once(path, *, expected_sha256=None,
                                 expected_fingerprint=None, max_bytes=800_000_000):
    """Read one immutable in-memory snapshot with the existing project allowlist.

    Only trusted, explicitly selected project artifacts are admitted by the
    caller. Final symlinks, unstable descriptors, unknown pickle globals and
    digest/fingerprint mismatches fail closed. Source payload is read once;
    deserialization uses that same immutable byte buffer, not a second disk read.
    """
    p = Path(path)
    before = fingerprint(os.stat(p, follow_symlinks=False))
    if not stat.S_ISREG(before["mode"]) or before["size"] > max_bytes:
        raise ValueError("not a bounded regular checkpoint")
    if expected_fingerprint is not None and before != expected_fingerprint:
        raise ValueError("source fingerprint changed")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            if fingerprint(os.fstat(handle.fileno())) != before:
                raise ValueError("opened descriptor differs from selected source")
            total = 0
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ValueError("checkpoint exceeded read budget")
                digest.update(block)
                buffer.write(block)
            if total != before["size"] or fingerprint(os.fstat(handle.fileno())) != before:
                raise ValueError("checkpoint changed during read")
        if fingerprint(os.stat(p, follow_symlinks=False)) != before:
            raise ValueError("source path changed during read")
        sha = digest.hexdigest()
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
        buffer.seek(0)
        value = torch.load(buffer, map_location="cpu", weights_only=False,
                           pickle_module=_RestrictedProjectCheckpointPickleModule)
        if not isinstance(value, dict):
            raise ValueError("checkpoint must be a mapping")
        return value, {"path": str(p), "sha256": sha, "fingerprint": before,
                       "bytes_read": total}
    finally:
        buffer.close()


def make_inference_package(checkpoint, provenance):
    if "model_state_dict" not in checkpoint or not isinstance(checkpoint["model_state_dict"], dict):
        raise ValueError("missing project model state")
    return {"schema": SCHEMA, "resume_capable": False,
            "inference_state": {k: v for k, v in checkpoint.items() if k not in DROP_KEYS},
            "removed_training_keys": sorted(DROP_KEYS.intersection(checkpoint)),
            "provenance": provenance}


def load_inference_checkpoint(path, *, expected_sha256, resume=False, max_bytes=800_000_000):
    if resume:
        raise ValueError("inference-only package cannot resume training")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("an externally supplied package SHA-256 is required")
    package, record = read_project_checkpoint_once(
        path, expected_sha256=expected_sha256, max_bytes=max_bytes)
    if package.get("schema") != SCHEMA or package.get("resume_capable") is not False:
        raise ValueError("not an explicit inference-only package")
    state = package.get("inference_state")
    if not isinstance(state, dict) or "model_state_dict" not in state or DROP_KEYS.intersection(state):
        raise ValueError("invalid inference-only payload")
    return state, package["provenance"], record
