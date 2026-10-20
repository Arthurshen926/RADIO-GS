"""Versioned, lossless tensor sharing for research fields.

A shared package is NOT standalone. Its reader must receive an explicit blob
store. No provenance path is ever opened. Frozen v1 packages remain supported
by storage.py; export_field_v1 creates a standalone deployment.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import torch

SCHEMA = "radio_gs.query_preserving_field.v2"
HEX = re.compile(r"[0-9a-f]{64}")
GEOMETRY = ("xyz", "normals", "quaternions", "scales", "opacities")
DTYPES = {str(getattr(torch, n)): getattr(torch, n) for n in (
    "float64", "float32", "float16", "bfloat16", "int64", "int32",
    "int16", "int8", "uint8", "bool", "complex64", "complex128")}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _safe_root(path, *, create=False):
    path = Path(os.path.abspath(path))
    # Do not resolve away symlinks before checking every component.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("storage path must not contain symlinks")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("storage root is missing or not a directory")
    return path


def _blob_path(root, digest):
    if not isinstance(digest, str) or not HEX.fullmatch(digest):
        raise ValueError("invalid content address")
    path = root / (digest + ".bin")
    if path.is_symlink():
        raise ValueError("blob symlink is forbidden")
    return path


def _bytes(tensor):
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("v2 currently supports dense nonquantized tensors only")
    value = tensor.detach().cpu().contiguous()
    if str(value.dtype) not in DTYPES:
        raise ValueError("unsupported lossless tensor dtype")
    return value, memoryview(value.reshape(-1).view(torch.uint8).numpy())


def _descriptor(tensor):
    value, data = _bytes(tensor)
    meta = dict(dtype=str(value.dtype), shape=list(value.shape), bytes=value.numel()*value.element_size())
    h = hashlib.sha256(_json(meta) + b"\0")
    h.update(data)
    return dict(meta, sha256=h.hexdigest()), value, data


def _verify_blob(root, descriptor):
    path = _blob_path(root, descriptor["sha256"])
    if not path.is_file() or path.stat().st_size != descriptor["bytes"]:
        raise ValueError("missing blob or blob size differs")
    meta = {k: descriptor[k] for k in ("dtype", "shape", "bytes")}
    h = hashlib.sha256(_json(meta) + b"\0")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        if os.fstat(handle.fileno()).st_size != descriptor["bytes"]:
            raise ValueError("blob size differs")
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    if h.hexdigest() != descriptor["sha256"]:
        raise ValueError("blob digest differs")
    return path



class VerifiedBlobStage:
    """One-process/store verification cache; never serialize or reuse across stages.

    Allocation is sampled cost, not identity. Cached tensors are checked in RAM
    on every reuse; callers must also check after scientific probes. This is for
    cooperative immutable storage, not a hostile concurrent filesystem sandbox.
    """
    def __init__(self, store_root, *, max_tensor_bytes=0):
        self.root = _safe_root(store_root)
        self.pid = os.getpid()
        self.root_identity = (self.root.stat().st_dev, self.root.stat().st_ino)
        self.max_tensor_bytes = max_tensor_bytes
        if type(max_tensor_bytes) is not int or max_tensor_bytes < 0:
            raise ValueError("invalid stage tensor budget")
        self.records = {}
        self.tensors = {}
        self.cached_bytes = 0
        self.read_bytes = 0
        self.closed = False

    def _check(self, root):
        root = _safe_root(root)
        if (self.closed or os.getpid() != self.pid or root != self.root or
                (root.stat().st_dev, root.stat().st_ino) != self.root_identity):
            raise ValueError("verification stage process/store identity differs")

    @staticmethod
    def _stamp(path):
        import stat
        s = path.lstat()
        if not stat.S_ISREG(s.st_mode):
            raise ValueError("stage blob must be a regular file")
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns,
                s.st_ctime_ns, s.st_nlink, s.st_mode)

    def _existing(self, root, entry):
        self._check(root)
        path = _blob_path(self.root, entry["sha256"])
        record = self.records.get(entry["sha256"])
        if record is not None:
            if record["descriptor"] != _json(entry) or self._stamp(path) != record["stamp"]:
                raise ValueError("cached blob identity changed")
            tensor = self.tensors.get(entry["sha256"])
            if tensor is not None and _descriptor(tensor)[0] != entry:
                raise ValueError("cached tensor changed")
        return path, record

    def _read(self, root, entry, *, retain):
        path, record = self._existing(root, entry)
        if record is not None and (not retain or entry["sha256"] in self.tensors):
            return path
        if retain and self.cached_bytes + entry["bytes"] > self.max_tensor_bytes:
            raise ValueError("stage tensor memory budget exceeded")
        before = self._stamp(path)
        if before[2] != entry["bytes"]:
            raise ValueError("stage blob size differs")
        meta = {k: entry[k] for k in ("dtype", "shape", "bytes")}
        digest = hashlib.sha256(_json(meta) + b"\0")
        data = bytearray(entry["bytes"]) if retain else None
        offset = 0
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != before[:3]:
                raise ValueError("blob replaced before open")
            while True:
                block = handle.read(min(8*1024*1024, entry["bytes"] - offset + 1))
                if not block:
                    break
                self.read_bytes += len(block)
                if offset + len(block) > entry["bytes"]:
                    raise ValueError("stage blob grew while reading")
                digest.update(block)
                if retain:
                    data[offset:offset+len(block)] = block
                offset += len(block)
        if (offset != entry["bytes"] or digest.hexdigest() != entry["sha256"]
                or self._stamp(path) != before):
            raise ValueError("stage blob digest/identity differs")
        self.records[entry["sha256"]] = dict(descriptor=_json(entry), stamp=before)
        if retain:
            tensor = (torch.frombuffer(data, dtype=DTYPES[entry["dtype"]])
                      if data else torch.empty(0, dtype=DTYPES[entry["dtype"]]))
            self.tensors[entry["sha256"]] = tensor.reshape(entry["shape"])
            self.cached_bytes += entry["bytes"]
        return path

    def verify(self, root, entry):
        return self._read(root, entry, retain=False)

    def load_state(self, root, entries):
        self._check(root)
        for entry in entries.values():
            self._read(root, entry, retain=True)
        return {key: self.tensors[entry["sha256"]] for key, entry in entries.items()}

    def assert_unchanged(self):
        self._check(self.root)
        for record in self.records.values():
            self._existing(self.root, json.loads(record["descriptor"]))

    def close(self):
        self.records.clear()
        self.tensors.clear()
        self.cached_bytes = 0
        self.closed = True


def _check_stage(stage, root):
    if type(stage) is not VerifiedBlobStage:
        raise ValueError("explicit VerifiedBlobStage required")
    stage._check(root)


@contextlib.contextmanager
def _lock(root):
    path = root / ".writer.lock"
    if path.is_symlink():
        raise ValueError("lock symlink is forbidden")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _publish_bytes(path, data):
    # Same-filesystem link provides atomic no-replace publication.
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


def estimate_storage(field, store_root):
    """Hashes in-memory state; does not read existing large blobs or write."""
    root = _safe_root(store_root)
    entries = {key: _descriptor(value)[0] for key, value in field.state_dict().items()}
    unique = {entry["sha256"]: entry for entry in entries.values()}
    missing = [entry for digest, entry in unique.items() if not _blob_path(root, digest).exists()]
    new = sum(entry["bytes"] for entry in missing)
    return dict(tensors=entries, logical_tensor_bytes=sum(e["bytes"] for e in entries.values()),
                unique_blob_bytes=sum(e["bytes"] for e in unique.values()),
                new_blob_bytes=new, temporary_peak_bytes=new,
                largest_temporary_blob_bytes=max((e["bytes"] for e in missing), default=0))


def save_field_v2(field, directory, provenance, *, store_root=None,
                  max_new_bytes=None, reserve_bytes=1024**3, verified_stage=None):
    """Save a new immutable variant, with exact-value deduplication.

    store_root=None creates a standalone v2 closure in directory/blobs.
    Explicit store_root creates a shared package requiring that root at load.
    Budget counts all missing blobs plus metadata; reserve is additional.
    Failed writes may leave unreferenced blobs, never deleted automatically.
    """
    if sys.byteorder != "little":
        raise ValueError("v2 raw tensor format requires little-endian host")
    target = Path(os.path.abspath(directory))
    _safe_root(target.parent, create=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    # Reserve the package name without overwriting an existing artifact.
    target.mkdir()
    root = _safe_root(target / "blobs" if store_root is None else store_root, create=True)
    if reserve_bytes < 0 or (max_new_bytes is not None and max_new_bytes < 0):
        raise ValueError("storage budgets must be nonnegative")
    if verified_stage is not None:
        _check_stage(verified_stage, root)
    with _lock(root):
        plan = estimate_storage(field, root)
        metadata_budget = 1024*1024 + len(plan["tensors"])*1024 + len(_json(provenance)) + len(_json(field.export_spec()))
        required = plan["new_blob_bytes"] + metadata_budget
        if max_new_bytes is not None and required > max_new_bytes:
            raise ValueError("new storage budget exceeded")
        if (shutil.disk_usage(root).free < required + reserve_bytes
                or shutil.disk_usage(target).free < metadata_budget + reserve_bytes):
            raise OSError("insufficient free storage including safety reserve")
        for key, tensor in field.state_dict().items():
            descriptor, value, data = _descriptor(tensor)
            if descriptor != plan["tensors"][key]:
                raise ValueError("field changed during save")
            path = _blob_path(root, descriptor["sha256"])
            if path.exists():
                if verified_stage is None:
                    _verify_blob(root, descriptor)
                else:
                    verified_stage.verify(root, descriptor)
            else:
                _publish_bytes(path, data)
        if verified_stage is not None:
            verified_stage.assert_unchanged()
        spec = field.export_spec()
        manifest = dict(schema=SCHEMA, byteorder="little", spec=spec,
            tensors=plan["tensors"], elements=len(field.xyz), provenance=provenance,
            blob_store="local" if store_root is None else "explicit",
            external_scene_dependencies=[], query_encoder_cost_excluded=True,
            row_contract=dict(kind="ordered_gaussian_rows_v1", count=len(field.xyz),
                geometry={k: plan["tensors"][k]["sha256"] for k in GEOMETRY},
                state_keys=sorted(plan["tensors"])),
            tensor_bytes=plan["logical_tensor_bytes"], unique_blob_bytes=plan["unique_blob_bytes"],
            new_blob_bytes=plan["new_blob_bytes"], temporary_peak_bytes=required,
            reserve_bytes=reserve_bytes)
        manifest["geometry_bytes"] = sum(plan["tensors"][k]["bytes"] for k in GEOMETRY)
        manifest["semantic_increment_bytes"] = manifest["tensor_bytes"] - manifest["geometry_bytes"]
        manifest["state_identity"] = _digest(dict(spec=spec, tensors=manifest["tensors"],
                                                row_contract=manifest["row_contract"]))
        manifest["manifest_sha256"] = _digest(manifest)
        _publish_bytes(target / "manifest.json", _json(manifest) + b"\n")
    return manifest


def _read_manifest(directory):
    root = _safe_root(directory)
    path = root / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 16*1024*1024:
        raise ValueError("invalid manifest path or size")
    manifest = json.loads(path.read_text())
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _digest(manifest):
        raise ValueError("manifest digest differs")
    manifest["manifest_sha256"] = claimed
    if manifest.get("schema") != SCHEMA or manifest.get("byteorder") != sys.byteorder:
        raise ValueError("unsupported field schema or byte order")
    if manifest.get("external_scene_dependencies") != []:
        raise ValueError("scene dependencies are forbidden")
    entries = manifest["tensors"]
    if not entries or any(not isinstance(k, str) for k in entries):
        raise ValueError("invalid state mapping")
    for entry in entries.values():
        if set(entry) != {"dtype", "shape", "bytes", "sha256"} or entry["dtype"] not in DTYPES:
            raise ValueError("invalid tensor descriptor")
        count = 1
        for n in entry["shape"]:
            if type(n) is not int or n < 0:
                raise ValueError("invalid tensor shape")
            count *= n
        if count * torch.empty((), dtype=DTYPES[entry["dtype"]]).element_size() != entry["bytes"]:
            raise ValueError("tensor byte count differs")
        _blob_path(root, entry["sha256"])
    unique = {entry["sha256"]: entry for entry in entries.values()}
    if (manifest["unique_blob_bytes"] != sum(e["bytes"] for e in unique.values())
            or manifest["tensor_bytes"] != sum(e["bytes"] for e in entries.values())):
        raise ValueError("complete storage accounting differs")
    row = manifest["row_contract"]
    if (row != dict(kind="ordered_gaussian_rows_v1", count=manifest["elements"],
                    geometry={k: entries[k]["sha256"] for k in GEOMETRY}, state_keys=sorted(entries))
            or any(entries[k]["shape"][0] != manifest["elements"] for k in GEOMETRY)):
        raise ValueError("Gaussian row contract differs")
    if manifest["state_identity"] != _digest(dict(spec=manifest["spec"], tensors=entries, row_contract=row)):
        raise ValueError("state identity differs")
    return root, manifest


def load_field_v2(directory, device="cpu", *, store_root=None, verified_stage=None):
    from .storage import field_from_state
    root, manifest = _read_manifest(directory)
    if manifest["blob_store"] == "local":
        if store_root is not None:
            raise ValueError("standalone package cannot use an external store")
        blobs = _safe_root(root / "blobs")
    elif manifest["blob_store"] == "explicit":
        if store_root is None:
            raise ValueError("shared v2 requires explicit store_root; provenance is never a fallback")
        blobs = _safe_root(store_root)
    else:
        raise ValueError("unknown blob store mode")
    if verified_stage is not None:
        _check_stage(verified_stage, blobs)
        state = verified_stage.load_state(blobs, manifest['tensors'])
    else:
        state = {}
        for key, entry in manifest["tensors"].items():
            path = _blob_path(blobs, entry["sha256"])
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as handle:
                if os.fstat(handle.fileno()).st_size != entry["bytes"]:
                    raise ValueError("blob size differs")
                data = bytearray(handle.read())
            meta = {k: entry[k] for k in ("dtype", "shape", "bytes")}
            digest = hashlib.sha256(_json(meta) + b"\0")
            digest.update(data)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("blob changed while loading")
            tensor = (torch.frombuffer(data, dtype=DTYPES[entry["dtype"]])
                      if data else torch.empty(0, dtype=DTYPES[entry["dtype"]]))
            state[key] = tensor.reshape(entry["shape"])
    field = field_from_state(state, manifest, "cpu", preserve_dtypes=True)
    restored = field.state_dict()
    if set(restored) != set(manifest["tensors"]) or any(
            _descriptor(value)[0] != manifest["tensors"][key] for key, value in restored.items()):
        raise ValueError("field constructor changed serialized tensor values or dtypes")
    return field.to(device).eval(), manifest


def export_field_v1(directory, destination, *, store_root=None):
    """Independent deployment; destination must not exist."""
    from .storage import save_field, field_from_state
    field, manifest = load_field_v2(directory, store_root=store_root)
    state = field.state_dict()
    legacy = field_from_state(state, manifest).state_dict()
    if any(_descriptor(value)[0] != manifest["tensors"][key] for key, value in legacy.items()):
        raise ValueError("v1 constructors cannot preserve this state; use a complete v2 closure")
    return save_field(field, destination, dict(parent_state_identity=manifest["state_identity"],
                                              original_provenance=manifest["provenance"]))


def report_gc_candidates(store_root, package_directories):
    """Report only. Caller must supply the COMPLETE reference inventory."""
    root = _safe_root(store_root)
    referenced = set()
    for directory in package_directories:
        _, manifest = _read_manifest(directory)
        if manifest["blob_store"] != "explicit":
            raise ValueError("GC inventory must contain shared packages")
        referenced.update(e["sha256"] for e in manifest["tensors"].values())
    return dict(inventory_complete_not_proven=True, deletion_performed=False,
                candidates=[p.name for p in root.glob("*.bin")
                            if HEX.fullmatch(p.stem) and p.stem not in referenced and not p.is_symlink()])
