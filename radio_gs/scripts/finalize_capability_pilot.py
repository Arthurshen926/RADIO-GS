"""Finalize an already validated additive pilot without rereading dense tensor data."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import zipfile
import torch
import torch._weights_only_unpickler as restricted
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank, load_canonical_support_graph
from radio_gs.scripts.migrate_capability_storage import fingerprint, file_sha256, compare_rows

STORAGE_KEYS = {"feature_storage", "feature_row_order", "feature_row_count"}
PAYLOAD_KEYS = {"schema_version", "xyz", "valid", "appearance_dino_v3", "boundary_sam3", "metadata"}

def metadata_only(path):
    """Read only bounded pickle metadata; storages are placeholders on meta."""
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if i.filename.endswith("/data.pkl")]
        if len(entries) != 1 or entries[0].file_size > 1024 * 1024:
            raise ValueError("missing, ambiguous or oversized tensor metadata")
        data = archive.read(entries[0])
    storages = {}
    def persistent(pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise ValueError("unsupported storage metadata")
        _, storage_type, key, location, count = pid
        dtype = storage_type.dtype
        if not isinstance(count, int) or not 0 <= count <= 10**12:
            raise ValueError("invalid storage extent")
        identity = (dtype, count)
        if key in storages and storages[key][0] != identity:
            raise ValueError("conflicting storage identity")
        if key not in storages:
            storage = torch.storage.TypedStorage(
                wrap_storage=torch.UntypedStorage(
                    count * torch._utils._element_size(dtype), device="meta"),
                dtype=dtype, _internal=True)
            storages[key] = (identity, storage)
        return storages[key][1]
    reader = restricted.Unpickler(io.BytesIO(data))
    reader.persistent_load = persistent
    payload = reader.load()
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or payload["schema_version"] != 1:
        raise ValueError("unexpected payload schema")
    if any(payload[k].device.type != "meta" for k in PAYLOAD_KEYS - {"metadata", "schema_version"}):
        raise ValueError("metadata inspection materialized a tensor")
    # JSON metadata only; reject hidden tensor/object metadata.
    return json.loads(json.dumps(payload["metadata"]))

def finalize(receipt_path, destination, graph_path):
    receipt_path = Path(receipt_path).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination already exists")
    previous = json.loads(receipt_path.read_text())
    required = {"xyz_exact", "valid_exact", "global_rows_exact",
                "all_valid_appearance_values_exact", "all_valid_boundary_values_exact",
                "signatures_exact", "original_metadata_retained", "shared_reader_load_passed"}
    if previous.get("schema") != "radio_gs.lossless_capability_compaction.v1" or previous.get("status") != "validated":
        raise ValueError("pilot is not validated")
    if not all(previous.get("validation", {}).get(k) is True for k in required):
        raise ValueError("incomplete pilot validation")
    source = Path(previous["source"]).resolve(strict=True)
    pilot = Path(previous["destination"]).resolve(strict=True)
    if pilot == source or pilot.parent != receipt_path.parent:
        raise ValueError("pilot and original identity invalid")
    sidecar = source.with_suffix(source.suffix + ".json")
    if fingerprint(source) != previous["source_fingerprint"]:
        raise ValueError("original source changed")
    if file_sha256(sidecar) != previous["source_sidecar_sha256"]:
        raise ValueError("original sidecar changed")
    before = {str(p): fingerprint(p) for p in [source, sidecar, pilot, receipt_path, Path(graph_path)]}
    if file_sha256(pilot) != previous["destination_sha256"]:
        raise ValueError("pilot full hash mismatch")
    original_metadata = metadata_only(source)
    if original_metadata.get("primitive_row_authority") != previous["legacy_sidecar"].get("primitive_row_authority"):
        raise ValueError("original embedded and sidecar authority disagree")
    raw = torch.load(pilot, map_location="cpu", weights_only=True)
    if set(raw) != PAYLOAD_KEYS or raw["schema_version"] != 1:
        raise ValueError("unexpected pilot payload")
    for key in (set(original_metadata) | set(raw["metadata"])) - STORAGE_KEYS - {"primitive_row_authority"}:
        if key not in original_metadata or key not in raw["metadata"] or original_metadata[key] != raw["metadata"][key]:
            raise ValueError("pilot metadata differs: " + key)
    metadata = copy.deepcopy(original_metadata)
    metadata.update(feature_storage="valid_rows_compact_v1",
                    feature_row_order="torch_where_valid_ascending",
                    feature_row_count=int(raw["valid"].sum()))
    old = load_canonical_capability_bank(pilot)
    # Compose the historical graph against the exact validated pilot tensors
    # with original metadata semantics. No dense source values are reread.
    historical_bank = replace(old, metadata=copy.deepcopy(metadata))
    old_graph = load_canonical_support_graph(graph_path, historical_bank)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / source.name
    receipt = dict(schema="radio_gs.capability_pilot_finalization.v1", status="in_progress",
                   original_source=str(source), original_source_sha256=previous["source_sha256"],
                   original_source_fingerprint=previous["source_fingerprint"],
                   original_source_hash_basis="inherited from validated first migration; fingerprint unchanged",
                   parent_receipt=str(receipt_path), parent_receipt_sha256=file_sha256(receipt_path),
                   verified_pilot=str(pilot), verified_pilot_sha256=previous["destination_sha256"],
                   parent_validation=previous["validation"], destination=str(output),
                   original_metadata=original_metadata, original_retained=True,
                   support_graph=str(Path(graph_path).resolve()))
    target_receipt = destination / "migration.json"
    target_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        final_payload = dict(raw, metadata=metadata)
        torch.save(final_payload, output)
        new_sidecar = copy.deepcopy(previous["legacy_sidecar"])
        new_sidecar.pop("primitive_row_authority", None)
        new_sidecar.update(metadata)
        new_sidecar["output"] = str(output)
        output.with_suffix(output.suffix + ".json").write_text(json.dumps(new_sidecar, indent=2) + "\n")
        checked = torch.load(output, map_location="cpu", weights_only=True)
        for key in PAYLOAD_KEYS - {"metadata", "schema_version"}:
            compare_rows(raw[key], checked[key])
        if checked["metadata"] != metadata:
            raise ValueError("final metadata differs")
        del checked
        new = load_canonical_capability_bank(output,
            expected_field_checkpoint_sha256=original_metadata["field_checkpoint_sha256"],
            require_row_authority=original_metadata.get("primitive_row_authority") is not None)
        for a, b in [(old.xyz, new.xyz), (old.valid, new.valid), (old.global_rows, new.global_rows),
                     (old.appearance, new.appearance), (old.boundary, new.boundary)]:
            compare_rows(a, b)
        if old.signatures != new.signatures:
            raise ValueError("signatures changed")
        new_graph = load_canonical_support_graph(graph_path, new)
        for key in ["edge_index", "edge_weight", "raw_affinity", "local_sigma"]:
            compare_rows(getattr(old_graph, key), getattr(new_graph, key))
        if any(fingerprint(p) != value for p, value in before.items()):
            raise ValueError("input changed during finalization")
        output_hash = file_sha256(output)
        if any(fingerprint(p) != value for p, value in before.items()):
            raise ValueError("input changed during final hash")
        receipt.update(status="validated", destination_sha256=output_hash,
                       destination_bytes=output.stat().st_size,
                       validation=dict(all_payload_tensors_exact=True, all_reader_tensors_exact=True,
                           original_metadata_semantics_restored=True, existing_support_graph_composes=True),
                       reclaimed_bytes=0)
        target_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=repr(exc))
        target_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        raise

