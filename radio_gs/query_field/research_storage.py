"""Explicit storage policy for active research compilers; no scientific transforms."""
from functools import partial
from pathlib import Path
import os

from .storage import save_field, field_identity, deployment_bytes


def research_writer(*, store_root=None, legacy_v1=False):
    if legacy_v1:
        if store_root is not None:
            raise ValueError("legacy v1 and shared store are mutually exclusive")
        return save_field
    if store_root is None:
        raise ValueError("new research writes require explicit store_root; historical v1 requires legacy_v1=True")
    from .shared_storage import save_field_v2
    return partial(save_field_v2, store_root=store_root)


def export_receipt(manifest):
    if manifest["schema"].endswith(".v1"):
        return manifest["files"]["field.pt"]
    return dict(schema=manifest["schema"], state_identity=field_identity(manifest),
                bytes=deployment_bytes(manifest))


def storage_binding(*, store_root=None, legacy_v1=False):
    research_writer(store_root=store_root, legacy_v1=legacy_v1)
    return dict(storage_mode="explicit_legacy_v1" if legacy_v1 else "shared_v2",
                blob_store=None if legacy_v1 else os.path.abspath(store_root))


def verify_storage_binding(design, *, store_root=None, legacy_v1=False):
    expected=storage_binding(store_root=store_root,legacy_v1=legacy_v1)
    actual=design.get("storage")
    # Old designs are accepted only in the explicit legacy route.
    if actual is None and legacy_v1:
        return
    if actual != expected:
        raise ValueError("frozen design storage mode/store differs; use its explicit settings or a fresh root")


def validate_research_package(directory, *, store_root=None, legacy_v1=False):
    """Cheap resume check; the prediction loader still hashes every blob."""
    import json
    research_writer(store_root=store_root,legacy_v1=legacy_v1)
    path=Path(directory)
    manifest=json.loads((path/"manifest.json").read_text())
    expected="radio_gs.query_preserving_field.v1" if legacy_v1 else "radio_gs.query_preserving_field.v2"
    if manifest.get("schema") != expected:
        raise ValueError("existing field format differs from requested research storage mode")
    if not legacy_v1:
        from .shared_storage import _read_manifest,_safe_root,_blob_path
        _,manifest=_read_manifest(path)
        if manifest.get("blob_store") != "explicit":
            raise ValueError("new research package requires explicit shared store")
        root=_safe_root(store_root)
        for descriptor in manifest["tensors"].values():
            blob=_blob_path(root,descriptor["sha256"])
            if not blob.is_file() or blob.stat().st_size != descriptor["bytes"]:
                raise ValueError("existing field blob closure is missing or has wrong size")
    return manifest


def method_binding(directory, manifest=None):
    """Bind archive/tensor identity AND function semantics; v1 SHA meaning stays unchanged."""
    import json, hashlib
    from .storage import sha256
    path=Path(directory)/"manifest.json"
    if manifest is None:manifest=json.loads(path.read_text())
    spec_sha=hashlib.sha256(json.dumps(manifest["spec"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return dict(schema=manifest["schema"],manifest_sha256=sha256(path),
                field_identity=field_identity(manifest),spec_sha256=spec_sha)


def tensor_state_identity(field):
    import hashlib,json
    from .shared_storage import _descriptor
    descriptors={k:_descriptor(v)[0] for k,v in field.state_dict().items()}
    return hashlib.sha256(json.dumps(descriptors,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def runner_storage_args(root, *, store_root=None, legacy_v1=False):
    research_writer(store_root=store_root,legacy_v1=legacy_v1)
    return ["--root",str(root)] + (["--legacy-v1"] if legacy_v1 else ["--store-root",str(store_root)])

def prediction_binding(directory, design, *, inputs, query_files=()):
    """Small, explicit resume identity; no hashing dense prediction/native payloads."""
    import hashlib,json
    from .storage import sha256
    storage=design["storage"]
    validate_research_package(directory,store_root=storage["blob_store"],
        legacy_v1=storage["storage_mode"]=="explicit_legacy_v1")
    return dict(field=method_binding(directory),storage=storage,
        design_sha256=hashlib.sha256(json.dumps(design,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        inputs=inputs,query_files={str(p):sha256(p) for p in query_files})


def check_prediction_binding(report, expected):
    if report.get("prediction_binding") != expected:
        raise ValueError("cached prediction method/spec/query/config/storage identity differs")

