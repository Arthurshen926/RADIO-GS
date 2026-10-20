"""Lossless, additive legacy capability compaction. Never replaces source files."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from radio_gs.interfaces.capability_cache import load_canonical_capability_bank, load_canonical_support_graph
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_rows(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("reader output shape or dtype differs")
    for start in range(0, len(left), 4096):
        if not torch.equal(left[start:start + 4096], right[start:start + 4096]):
            raise ValueError("reader output values differ")


def compact(source, destination, graph_paths=()):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("migration destination already exists")
    sidecar = source.with_suffix(source.suffix + ".json")
    source_before = fingerprint(source)
    sidecar_before = fingerprint(sidecar)
    legacy_sidecar = json.loads(sidecar.read_text())
    if legacy_sidecar.get("feature_storage") == "valid_rows_compact_v1":
        raise ValueError("source is already compact")
    field_hash = legacy_sidecar.get("field_checkpoint_sha256", "")
    if not field_hash:
        raise ValueError("source sidecar lacks field checkpoint authority")
    # Input hash identifies the exact old artifact. No metadata digest is
    # substituted for a fresh file digest in the migration receipt.
    source_hash = file_sha256(source)
    sidecar_hash = file_sha256(sidecar)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    required = {"schema_version", "xyz", "valid", "appearance_dino_v3",
                "boundary_sam3", "metadata"}
    if not isinstance(raw, dict) or set(raw) != required or raw["schema_version"] != 1:
        raise ValueError("unexpected top-level payload fields; refuse lossy migration")
    raw_metadata = copy.deepcopy(raw["metadata"])
    for key, value in raw_metadata.items():
        if key in legacy_sidecar and legacy_sidecar[key] != value:
            raise ValueError("embedded metadata and sidecar disagree: " + key)
    if raw["xyz"].dtype != torch.float32 or raw["valid"].dtype != torch.bool:
        raise ValueError("unsupported geometry dtype; no implicit conversion")
    for key, actual in {
        "num_gaussians": raw["xyz"].shape[0],
        "valid_gaussians": int(raw["valid"].sum()),
        "appearance_dim": raw["appearance_dino_v3"].shape[1],
        "boundary_dim": raw["boundary_sam3"].shape[1],
    }.items():
        if legacy_sidecar.get(key) != actual:
            raise ValueError("sidecar tensor dimensions disagree: " + key)
    old = load_canonical_capability_bank(
        source, expected_field_checkpoint_sha256=field_hash)
    graph_paths = [Path(p).resolve(strict=True) for p in graph_paths]
    graph_fingerprints = {str(p): fingerprint(p) for p in graph_paths}
    old_graphs = [load_canonical_support_graph(p, old) for p in graph_paths]
    if old.metadata.get("feature_storage") == "valid_rows_compact_v1":
        raise ValueError("source payload is already compact")
    if old.metadata.get("source") != legacy_sidecar.get("source"):
        raise ValueError("source sidecar/payload source differs")
    for key in ("field_checkpoint_sha256", "radio_checkpoint_sha256",
                "capability_signatures"):
        if old.metadata.get(key) != legacy_sidecar.get(key):
            raise ValueError("source sidecar/payload identity differs: " + key)

    # The deployed reader exposes only valid rows. Preserve those tensors
    # bit-for-bit, including dtype; never project, normalize, or quantize.
    features = old.valid_feature_banks()
    appearance = features["appearance"].contiguous().clone()
    boundary = features["boundary"].contiguous().clone()
    compare_rows(raw["xyz"], old.xyz)
    compare_rows(raw["valid"], old.valid)
    compare_rows(raw["appearance_dino_v3"][old.global_rows], appearance)
    compare_rows(raw["boundary_sam3"][old.global_rows], boundary)
    del raw
    metadata = copy.deepcopy(raw_metadata)
    metadata.update(
        feature_storage="valid_rows_compact_v1",
        feature_row_order="torch_where_valid_ascending",
        feature_row_count=int(old.valid.sum()))
    # Preserve the historical metadata contract, including absence of optional
    # authority. Injecting a new authority only into the bank invalidates old
    # support graphs that correctly bind to the original absent authority.
    # Existing authority is never removed; independently verified row identity
    # is recorded in this migration receipt for either legacy representation.
    verified_authority = PrimitiveRowAuthority.from_tensors(
        old.xyz, old.valid).to_dict()
    if fingerprint(source) != source_before or fingerprint(sidecar) != sidecar_before:
        raise RuntimeError("source changed during read; nothing published")
    # Exclusively create an independent artifact directory. The original
    # remains readable under the old hash and all existing frozen receipts.
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / source.name
    started = time.time()
    receipt = dict(schema="radio_gs.lossless_capability_compaction.v1",
                   status="in_progress", source=str(source),
                   source_sha256=source_hash, source_fingerprint=source_before,
                   source_sidecar_sha256=sidecar_hash,
                   legacy_sidecar=legacy_sidecar, destination=str(output),
                   original_retained=True, source_modified=False,
                   existing_support_graphs=graph_fingerprints,
                   verified_primitive_row_authority=verified_authority,
                   original_authority_present=raw_metadata.get(
                       "primitive_row_authority") is not None,
                   method="exact deployed-reader valid rows, no re-encoding")
    receipt_path = destination / "migration.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        torch.save(dict(schema_version=1, xyz=old.xyz.contiguous().clone(),
                        valid=old.valid.contiguous().clone(),
                        appearance_dino_v3=appearance,
                        boundary_sam3=boundary, metadata=metadata), output)
        new_sidecar = copy.deepcopy(legacy_sidecar)
        new_sidecar.update(metadata)
        new_sidecar.update(output=str(output), num_gaussians=len(old.xyz),
                           valid_gaussians=int(old.valid.sum()),
                           appearance_dim=appearance.shape[1],
                           boundary_dim=boundary.shape[1])
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(new_sidecar, indent=2) + "\n")
        new = load_canonical_capability_bank(
            output, expected_field_checkpoint_sha256=field_hash,
            require_row_authority=raw_metadata.get("primitive_row_authority") is not None)
        compare_rows(old.xyz, new.xyz)
        compare_rows(old.valid, new.valid)
        compare_rows(old.global_rows, new.global_rows)
        compare_rows(appearance, new.valid_feature_banks()["appearance"])
        compare_rows(boundary, new.valid_feature_banks()["boundary"])
        if old.signatures != new.signatures:
            raise ValueError("capability signatures changed")
        for graph_path, old_graph in zip(graph_paths, old_graphs):
            new_graph = load_canonical_support_graph(graph_path, new)
            for key in ("edge_index", "edge_weight", "raw_affinity", "local_sigma"):
                compare_rows(getattr(old_graph, key), getattr(new_graph, key))
        if any(fingerprint(p) != value for p, value in graph_fingerprints.items()):
            raise RuntimeError("support graph changed during migration")
        for key, value in raw_metadata.items():
            if key not in {"feature_storage", "feature_row_order",
                           "feature_row_count"}:
                if new.metadata.get(key) != value:
                    raise ValueError("metadata changed: " + key)
        if fingerprint(source) != source_before or fingerprint(sidecar) != sidecar_before:
            raise RuntimeError("source changed during migration")
        output_hash = file_sha256(output)
        if fingerprint(source) != source_before or fingerprint(sidecar) != sidecar_before:
            raise RuntimeError("source changed during final hash")
        if any(fingerprint(p) != value for p, value in graph_fingerprints.items()):
            raise RuntimeError("support graph changed during final hash")
        receipt.update(
            status="validated", destination_sha256=output_hash,
            source_bytes=source_before[2], destination_bytes=output.stat().st_size,
            projected_logical_reclaim_after_authorized_source_removal=(
                source_before[2] - output.stat().st_size),
            validation=dict(xyz_exact=True, valid_exact=True, global_rows_exact=True,
                            all_valid_appearance_values_exact=True,
                            all_valid_boundary_values_exact=True,
                            signatures_exact=True, original_metadata_retained=True,
                            shared_reader_load_passed=True,
                            existing_support_graphs_composed=len(graph_paths)),
            elapsed_seconds=time.time() - started,
            note="No original removed or references rebound. Downstream consumers "
                 "that directly load legacy dense arrays require separate review.")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=repr(exc))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--support-graph", type=Path, action="append", default=[])
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(compact(args.source, args.destination, args.support_graph), indent=2))


if __name__ == "__main__":
    main()

