"""Bounded new-run writes; never retires historical or shared artifacts.

One lock accounts for the entire explicitly managed study, including its shared
blobs and atomic-write temporaries. This does not govern unrelated old writers.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch


def tree_bytes(root):
    total = 0
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            if (Path(base) / name).is_symlink():
                raise ValueError("managed output cannot contain symbolic links")
        total += sum((Path(base) / name).stat().st_size for name in files)
    return total


class LimitedFile:
    def __init__(self, handle, limit):
        self.handle, self.limit = handle, limit

    def write(self, data):
        if self.handle.tell() + len(data) > self.limit:
            raise ValueError("artifact byte budget exceeded")
        return self.handle.write(data)

    def __getattr__(self, key):
        return getattr(self.handle, key)


class BoundedExperiment:
    def __init__(self, study_root, run_name, policy, *, resume=False):
        self.study = Path(study_root).absolute()
        if not run_name or Path(run_name).name != run_name or run_name in {".", "..", "blobs"}:
            raise ValueError("run must have a single safe name")
        required = {"study_bytes", "run_bytes", "artifact_bytes", "reserve_bytes"}
        if set(policy) != required or any(type(v) is not int or v <= 0 for v in policy.values()):
            raise ValueError("explicit positive storage budgets are required")
        if not policy["artifact_bytes"] <= policy["run_bytes"] <= policy["study_bytes"]:
            raise ValueError("inconsistent storage budgets")
        self.policy = dict(policy)
        self.study.mkdir(parents=True, exist_ok=True)
        if self.study.is_symlink():
            raise ValueError("managed study must be a real directory")
        self.root = self.study / run_name
        with self.lock():
            registry = self.study / "storage_policy.json"
            if registry.exists():
                if json.loads(registry.read_text()) != policy:
                    raise ValueError("study storage budget changed")
            else:
                if any(p.name != ".budget.lock" for p in self.study.iterdir()):
                    raise ValueError("refuse to adopt unmanaged historical outputs")
                registry.write_text(json.dumps(policy, sort_keys=True) + "\n")
            if resume:
                if not self.root.is_dir() or self.root.is_symlink():
                    raise ValueError("resume requires an existing managed run")
            else:
                self.root.mkdir(exist_ok=False)

    @contextlib.contextmanager
    def lock(self):
        fd = os.open(self.study / ".budget.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _check(self, additional, *, run_additional=None):
        run_additional = additional if run_additional is None else run_additional
        if tree_bytes(self.study) + additional > self.policy["study_bytes"]:
            raise ValueError("study storage budget exceeded")
        if tree_bytes(self.root) + run_additional > self.policy["run_bytes"]:
            raise ValueError("run storage budget exceeded")
        if shutil.disk_usage(self.study).free < additional + self.policy["reserve_bytes"]:
            raise OSError("free-space reserve would be violated")

    def preflight(self, peak_bytes):
        with self.lock():
            self._check(peak_bytes)

    def _write(self, name, writer, *, limit, replace=False, nested=False):
        relative = Path(name)
        if (relative.is_absolute() or not relative.parts or
                any(part.startswith(".") for part in relative.parts) or
                (not nested and relative.name != name)):
            raise ValueError("artifact must have one ordinary filename")
        if replace and name not in {"latest.pt", "best.pt", "progress.json"}:
            raise ValueError("only rolling checkpoints/progress can be replaced")
        if limit > self.policy["artifact_bytes"] or limit < 1:
            raise ValueError("artifact reservation exceeds policy")
        target = self.root / name
        with self.lock():
            # Nested public prediction files remain under the SAME run/study
            # accounting. Never follow a directory link outside that boundary.
            parent = self.root
            for part in relative.parts[:-1]:
                parent = parent / part
                if parent.is_symlink():
                    raise ValueError("artifact directory cannot be a symbolic link")
                parent.mkdir(exist_ok=True)
            if target.is_symlink() or (target.exists() and not replace):
                raise FileExistsError(target)
            # Counts OLD + NEW, not only the eventual file-size delta.
            self._check(limit)
            fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    writer(LimitedFile(handle, limit))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return target

    def json(self, name, value, *, replace=False):
        data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        return self._write(name, lambda f: f.write(data), limit=len(data), replace=replace)

    def tensor(self, name, value, *, max_bytes, replace=False):
        return self._write(name, lambda f: torch.save(value, f), limit=max_bytes, replace=replace)

    def public_artifact(self, path, value, kind):
        """Bound existing public writers without duplicating their scorer logic."""
        import numpy as np
        relative = str(Path(path).absolute().relative_to(self.root))
        if kind == "json":
            data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
            writer, limit = lambda f: f.write(data), len(data)
        elif kind == "tensor":
            def size(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.numel() * obj.element_size()
                if isinstance(obj, dict):
                    return sum(size(v) for v in obj.values())
                if isinstance(obj, (tuple, list)):
                    return sum(size(v) for v in obj)
                return 0
            writer, limit = lambda f: torch.save(value, f), size(value) + 2 ** 20
        elif kind == "numpy":
            # Pass a wrapper (not a pathname), so numpy uses bounded write(),
            # including its chunked array path, instead of a raw fd tofile().
            writer, limit = lambda f: np.save(f, value, allow_pickle=False), value.nbytes + 4096
        else:
            raise ValueError("unknown public artifact type")
        return self._write(relative, writer, limit=limit, nested=True)

    def export(self, field, provenance, *, max_new_bytes):
        from .shared_storage import save_field_v2
        with self.lock():
            self._check(max_new_bytes)
            manifest = save_field_v2(field, self.root / "field", provenance,
                                     store_root=self.study / "blobs", max_new_bytes=max_new_bytes,
                                     reserve_bytes=self.policy["reserve_bytes"])
        return manifest

    def retire_checkpoints_after_verified_export(self, exported_state_identity):
        """Only own two rolling slots; caller must supply a cold-load receipt."""
        from .shared_storage import load_field_v2
        field, manifest = load_field_v2(self.root / "field", store_root=self.study / "blobs")
        if manifest["state_identity"] != exported_state_identity:
            raise ValueError("export identity mismatch")
        # Recheck best checkpoint against the independently reloaded export.
        best = torch.load(self.root / "best.pt", map_location="cpu", weights_only=True)
        state = field.state_dict()
        if set(state) != set(best["model"]) or any(not torch.equal(v, best["model"][k]) for k, v in state.items()):
            raise ValueError("export differs from selected best state")
        self.json("checkpoint_retirement.json", dict(state_identity=exported_state_identity,
                  removed_slots=["best.pt", "latest.pt"], inference_replay_verified=True,
                  exact_training_resume_retained=False))
        with self.lock():
            for name in ("best.pt", "latest.pt"):
                path = self.root / name
                if path.is_symlink():
                    raise ValueError("rolling checkpoint replaced by a symlink")
                path.unlink(missing_ok=True)
