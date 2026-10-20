"""Explicit old archive identity -> immutable shared field migration binding.

A legacy archive SHA, a manifest-file SHA, and a state identity are separate
namespaces. No fallback occurs when a mapped replacement fails validation.
"""
import hashlib
import json
import os
from pathlib import Path
import re

from .storage import load_field,sha256,field_from_state
from .shared_storage import _read_manifest,_safe_root,_blob_path,_descriptor,load_field_v2,export_field_v1

SCHEMA="radio_gs.legacy_field_resolver.v1"
HEX=re.compile(r"[0-9a-f]{64}")


def _hash(value):
    if not isinstance(value,str) or not HEX.fullmatch(value):
        raise ValueError("invalid SHA identity")
    return value


def _old_payload(path,output_root):
    path=Path(path)
    if not path.is_absolute() and path.parts and path.parts[0]=="output":
        path=Path(output_root).joinpath(*path.parts[1:])
    path=Path(os.path.abspath(path))
    path=path if path.name=="field.pt" else path/"field.pt"
    if os.path.commonpath([str(path),os.path.abspath(output_root)])!=os.path.abspath(output_root):
        raise ValueError("legacy field path lies outside declared output root")
    return path


def create_mapping(bindings,destination,*,output_root):
    """Validate tiny or real migrated pairs before publishing a new mapping.

    This operation reads full states; callers must separately budget real I/O.
    Existing originals are retained. Existing mapping files are never replaced.
    """
    import torch
    entries=[]
    for source,target,store in bindings:
        old=_old_payload(source,output_root)
        manifest_path=old.parent/"manifest.json"
        original=json.loads(manifest_path.read_text())
        old_sha=_hash(original["files"]["field.pt"]["sha256"])
        if sha256(old)!=old_sha:raise ValueError("original archive SHA differs")
        field,new=load_field_v2(target,store_root=store)
        raw=torch.load(old,map_location="cpu",weights_only=True)
        if set(raw)!=set(field.state_dict()):raise ValueError("migration state keys differ")
        if any(_descriptor(raw[k])[0]!=_descriptor(field.state_dict()[k])[0] for k in raw):
            raise ValueError("migration changed original tensor dtype/value/shape")
        legacy=field_from_state(raw,original,preserve_dtypes=True)
        if legacy.export_spec()!=field.export_spec():raise ValueError("migration scientific spec differs")
        if new["provenance"].get("source_field_sha256")!=old_sha or new["provenance"].get("source_manifest")!=original:
            raise ValueError("migration source provenance differs")
        entries.append(dict(old_path=str(old),old_field_sha256=old_sha,
            old_manifest_sha256=sha256(manifest_path),target=str(_safe_root(target)),
            target_manifest_sha256=sha256(Path(target)/"manifest.json"),
            state_identity=new["state_identity"],store_root=str(_safe_root(store))))
    if len({e["old_path"] for e in entries})!=len(entries):raise ValueError("duplicate old field path")
    value=dict(schema=SCHEMA,output_root=os.path.abspath(output_root),entries=entries)
    with Path(destination).open("x") as handle:
        json.dump(value,handle,indent=2);handle.write("\n");handle.flush();os.fsync(handle.fileno())
    return value


class FieldResolver:
    def __init__(self,path,*,expected_mapping_sha256):
        path=Path(path)
        if path.is_symlink() or path.stat().st_size>1024*1024:raise ValueError("invalid resolver manifest")
        if sha256(path)!=_hash(expected_mapping_sha256):raise ValueError("resolver mapping digest differs")
        self.mapping=json.loads(path.read_text())
        if self.mapping.get("schema")!=SCHEMA:raise ValueError("unknown resolver schema")
        self.path=path;self.digest=expected_mapping_sha256
        self.output_root=Path(self.mapping["output_root"])
        if not self.output_root.is_absolute():raise ValueError("output root must be absolute")
        self.entries={}
        for e in self.mapping["entries"]:
            old=str(_old_payload(e["old_path"],self.output_root))
            if old!=e["old_path"] or old in self.entries:raise ValueError("duplicate/noncanonical old path")
            for k in ["old_field_sha256","old_manifest_sha256","target_manifest_sha256","state_identity"]:_hash(e[k])
            self.entries[old]=e

    def payload_path(self,path):
        return _old_payload(path,self.output_root)

    def inspect(self,path,expected_old_sha256):
        old=self.payload_path(path);expected=_hash(expected_old_sha256)
        _safe_root(old.parent)
        source_path=old.parent/"manifest.json"
        if source_path.is_symlink():raise ValueError("source manifest symlink forbidden")
        original=json.loads(source_path.read_text())
        if original.get("schema")!="radio_gs.query_preserving_field.v1" or original["files"]["field.pt"]["sha256"]!=expected:
            raise ValueError("expected original archive identity differs from retained source manifest")
        e=self.entries.get(str(old))
        if e is None:
            if not old.is_file() or old.is_symlink():raise ValueError("unmapped original v1 payload is missing")
            return dict(mode="unmigrated_v1",old_path=str(old),old_field_sha256=expected,
                        source_manifest=original,target=str(old.parent),store_root=None)
        if e["old_field_sha256"]!=expected or sha256(source_path)!=e["old_manifest_sha256"]:
            raise ValueError("mapping original path/SHA or source manifest differs")
        target=_safe_root(e["target"]);store=_safe_root(e["store_root"])
        _,m=_read_manifest(target)
        if sha256(target/"manifest.json")!=e["target_manifest_sha256"] or m["state_identity"]!=e["state_identity"]:
            raise ValueError("replacement manifest/state identity differs")
        if m.get("blob_store")!="explicit" or m["provenance"].get("source_field_sha256")!=expected or m["provenance"].get("source_manifest")!=original:
            raise ValueError("replacement source identity/store mode differs")
        for d in m["tensors"].values():
            blob=_blob_path(store,d["sha256"])
            if not blob.is_file() or blob.stat().st_size!=d["bytes"]:raise ValueError("replacement blob closure missing or wrong size")
        return dict(mode="mapped_v2",old_path=str(old),old_field_sha256=expected,
                    source_manifest=original,target=str(target),store_root=str(store),
                    target_manifest_sha256=e["target_manifest_sha256"],state_identity=e["state_identity"])

    def load(self,path,expected_old_sha256,device="cpu"):
        info=self.inspect(path,expected_old_sha256)
        if info["mode"]=="mapped_v2":
            field,_=load_field_v2(info["target"],device,store_root=info["store_root"])
        else:
            if sha256(info["old_path"])!=expected_old_sha256:raise ValueError("unmapped original archive SHA differs")
            field,_=load_field(info["target"],device)
        # Existing algorithms receive the real original v1 manifest, never a
        # fabricated field.pt receipt carrying a blob or state digest.
        return field,info["source_manifest"]

    def loader(self,expected_old_sha256):
        return lambda path,device="cpu":self.load(path,expected_old_sha256,device)

    def export_legacy(self,path,expected_old_sha256,destination,*,max_export_bytes=16*1024**3):
        info=self.inspect(path,expected_old_sha256)
        if info["mode"]!="mapped_v2":raise ValueError("explicit export requires mapped v2")
        import shutil
        _,manifest=_read_manifest(info['target'])
        estimate=manifest['tensor_bytes']+1024**2
        if estimate>max_export_bytes:raise ValueError('legacy export exceeds explicit size cap')
        parent=Path(destination).absolute().parent
        while not parent.exists():parent=parent.parent
        _safe_root(parent)
        if shutil.disk_usage(parent).free<estimate+1024**3:raise ValueError('legacy export free-space reserve not met')
        result=export_field_v1(info["target"],destination,store_root=info["store_root"])
        actual=sha256(Path(destination)/"field.pt")
        if actual!=expected_old_sha256:
            raise ValueError("standalone state exported but archive SHA differs; not exact legacy replay")
        return dict(original_archive_sha256=expected_old_sha256,exported_archive_sha256=actual,
                    manifest=result,path=str(destination),byte_exact_legacy_archive=True)
