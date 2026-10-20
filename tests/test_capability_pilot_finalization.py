import copy
import json
import zipfile
import pytest
import torch
from test_capability_storage_migration import fixture
from radio_gs.scripts.migrate_capability_storage import compact, file_sha256
from radio_gs.scripts.finalize_capability_pilot import metadata_only, finalize
from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority

def setup(tmp_path, authority=False):
    source = tmp_path / "legacy.pt"
    payload = fixture(source)
    if authority:
        payload["metadata"]["primitive_row_authority"] = PrimitiveRowAuthority.from_tensors(payload["xyz"], payload["valid"]).to_dict()
        torch.save(payload, source)
        sidecar = source.with_suffix(".pt.json")
        data = json.loads(sidecar.read_text())
        data.update(payload["metadata"])
        sidecar.write_text(json.dumps(data))
    graph = tmp_path / "graph.pt"
    meta = {"capability_metadata": copy.deepcopy(payload["metadata"])}
    if authority:
        meta["primitive_row_authority"] = payload["metadata"]["primitive_row_authority"]
    torch.save(dict(schema_version=1, global_rows=torch.tensor([0, 3]), num_global_rows=5,
        edge_index=torch.tensor([[0, 1], [1, 0]]), edge_weight=torch.tensor([.25, .75]),
        raw_affinity=torch.tensor([.3, .8]), local_sigma=torch.ones(2), metadata=meta), graph)
    first = compact(source, tmp_path / "first")
    # Reproduce the first pilot's extra authority without altering source.
    pilot = torch.load(first["destination"], weights_only=True)
    pilot["metadata"]["primitive_row_authority"] = PrimitiveRowAuthority.from_tensors(payload["xyz"], payload["valid"]).to_dict()
    torch.save(pilot, first["destination"])
    first["destination_sha256"] = file_sha256(first["destination"])
    (tmp_path / "first/migration.json").write_text(json.dumps(first))
    return source, graph, payload

@pytest.mark.parametrize("authority", [False, True])
def test_chain_restores_original_semantics(tmp_path, authority):
    source, graph, payload = setup(tmp_path, authority)
    result = finalize(tmp_path / "first/migration.json", tmp_path / "final", graph)
    assert result["status"] == "validated"
    actual = torch.load(result["destination"], weights_only=True)
    assert actual["metadata"].get("primitive_row_authority") == payload["metadata"].get("primitive_row_authority")
    assert file_sha256(source) == result["original_source_sha256"]

def test_metadata_reader_never_reads_storage_records(tmp_path, monkeypatch):
    source = tmp_path / "legacy.pt"
    payload = fixture(source)
    original = zipfile.ZipFile.open
    opened = []
    def guarded(self, name, *args, **kwargs):
        filename = name.filename if hasattr(name, "filename") else name
        opened.append(filename)
        assert filename.endswith("/data.pkl")
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    assert metadata_only(source) == payload["metadata"]
    assert len(opened) == 1

@pytest.mark.parametrize("mutate", ["source", "pilot"])
def test_chain_rejects_changed_input(tmp_path, mutate):
    source, graph, payload = setup(tmp_path)
    path = source if mutate == "source" else tmp_path / "first/legacy.pt"
    with path.open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        finalize(tmp_path / "first/migration.json", tmp_path / "final", graph)
    assert not (tmp_path / "final").exists()

