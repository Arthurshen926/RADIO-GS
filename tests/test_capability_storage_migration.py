import json
from pathlib import Path
import pytest
import torch
from radio_gs.field import FeatureSpaceSignature
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank
from radio_gs.scripts.migrate_capability_storage import compact, file_sha256


def fixture(path):
    metadata = dict(source="canonical_radio_field_official_frozen_capability_views",
                    field_checkpoint_sha256="field-hash",
                    radio_checkpoint_sha256="radio-hash",
                    custom_adaptor_head=False, query_independent=True,
                    extra_provenance={"normalization": "already_applied", "frames": [3, 7]})
    metadata["capability_signatures"] = {
        name: FeatureSpaceSignature(
            radio_version="test", radio_checkpoint_sha256="radio-hash",
            raw_feature_dim=1280, adaptor_name=name, adaptor_sha256="radio-hash",
            adaptor_output_dim=dim, token_type="primitive",
            field_checkpoint_sha256="field-hash").to_dict()
        for name, dim in [("appearance", 4), ("boundary", 2)]}
    payload = dict(schema_version=1, xyz=torch.arange(15).reshape(5, 3).float(),
                   valid=torch.tensor([True, False, False, True, False]),
                   appearance_dino_v3=torch.arange(20).reshape(5, 4).half(),
                   boundary_sam3=torch.arange(10).reshape(5, 2).half(),
                   metadata=metadata)
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(
        dict(metadata, num_gaussians=5, valid_gaussians=2, appearance_dim=4,
             boundary_dim=2, output=str(path))))
    return payload


def test_exact_additive_reader_equivalence(tmp_path):
    source = tmp_path / "legacy.pt"
    payload = fixture(source)
    old_hash = file_sha256(source)
    result = compact(source, tmp_path / "migration")
    assert result["status"] == "validated"
    assert file_sha256(source) == old_hash
    assert result["original_retained"]
    new = load_canonical_capability_bank(result["destination"])
    assert "primitive_row_authority" not in new.metadata
    assert result["verified_primitive_row_authority"]["num_active_rows"] == 2
    assert new.features_are_compact
    assert new.global_rows.tolist() == [0, 3]
    assert torch.equal(new.appearance, payload["appearance_dino_v3"][[0, 3]])
    assert torch.equal(new.boundary, payload["boundary_sam3"][[0, 3]])
    assert new.metadata["extra_provenance"] == payload["metadata"]["extra_provenance"]
    assert result["legacy_sidecar"]["extra_provenance"] == new.metadata["extra_provenance"]


def test_existing_destination_is_never_overwritten(tmp_path):
    source = tmp_path / "legacy.pt"
    fixture(source)
    dest = tmp_path / "migration"
    dest.mkdir()
    marker = dest / "keep"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        compact(source, dest)
    assert marker.read_text() == "original"


def test_sidecar_payload_identity_disagreement_fails(tmp_path):
    source = tmp_path / "legacy.pt"
    fixture(source)
    sidecar = source.with_suffix(".pt.json")
    data = json.loads(sidecar.read_text())
    data["field_checkpoint_sha256"] = "wrong"
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagree"):
        compact(source, tmp_path / "migration")
    assert not (tmp_path / "migration").exists()


def test_unknown_payload_is_not_silently_discarded(tmp_path):
    source = tmp_path / "legacy.pt"
    payload = fixture(source)
    payload["extra_tensor"] = torch.ones(2)
    torch.save(payload, source)
    with pytest.raises(ValueError, match="unexpected top-level"):
        compact(source, tmp_path / "migration")
    assert not (tmp_path / "migration").exists()


def test_compact_source_rejected(tmp_path):
    source = tmp_path / "legacy.pt"
    fixture(source)
    first = compact(source, tmp_path / "first")
    with pytest.raises(ValueError, match="already compact"):
        compact(first["destination"], tmp_path / "second")


def test_large_legacy_memmap_branch_with_small_real_archive(tmp_path, monkeypatch):
    import radio_gs.interfaces.capability_cache as cache
    source = (tmp_path / "legacy.pt").resolve()
    fixture(source)
    original_stat = Path.stat
    original_loader = cache._load_memory_mapped_capability_payload
    entered = []
    class StatProxy:
        def __init__(self, raw):
            self.raw = raw
            self.st_size = 2 * 1024**3
        def __getattr__(self, name):
            return getattr(self.raw, name)
    def fake_stat(path, *args, **kwargs):
        raw = original_stat(path, *args, **kwargs)
        return StatProxy(raw) if path == source else raw
    def observed_loader(path):
        entered.append(str(path))
        return original_loader(path)
    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(cache, "_load_memory_mapped_capability_payload", observed_loader)
    result = compact(source, tmp_path / "migration")
    assert result["status"] == "validated"
    assert entered == [str(source)]
    new = load_canonical_capability_bank(result["destination"])
    assert new.global_rows.tolist() == [0, 3]


def test_bad_sidecar_dimensions_fail_before_output(tmp_path):
    source = tmp_path / "legacy.pt"
    fixture(source)
    sidecar = source.with_suffix(".pt.json")
    data = json.loads(sidecar.read_text())
    data["appearance_dim"] = 900
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tensor dimensions"):
        compact(source, tmp_path / "migration")
    assert not (tmp_path / "migration").exists()


@pytest.mark.parametrize("has_authority", [False, True])
def test_existing_support_graph_composes_with_migrated_bank(tmp_path, has_authority):
    import copy
    from radio_gs.interfaces.capability_cache import load_canonical_support_graph
    from radio_gs.interfaces.primitive_row_authority import PrimitiveRowAuthority
    source = tmp_path / "legacy.pt"
    payload = fixture(source)
    authority = PrimitiveRowAuthority.from_tensors(payload["xyz"], payload["valid"]).to_dict()
    if has_authority:
        payload["metadata"]["primitive_row_authority"] = authority
        torch.save(payload, source)
        sidecar = source.with_suffix(".pt.json")
        data = json.loads(sidecar.read_text())
        data["primitive_row_authority"] = authority
        sidecar.write_text(json.dumps(data))
    graph_metadata = {"capability_metadata": copy.deepcopy(payload["metadata"])}
    if has_authority:
        graph_metadata["primitive_row_authority"] = authority
    graph_path = tmp_path / "existing_graph.pt"
    torch.save(dict(schema_version=1, global_rows=torch.tensor([0, 3]),
                    num_global_rows=5, edge_index=torch.tensor([[0, 1], [1, 0]]),
                    edge_weight=torch.tensor([.25, .75]), raw_affinity=torch.tensor([.3, .8]),
                    local_sigma=torch.ones(2), metadata=graph_metadata), graph_path)
    old_bank = load_canonical_capability_bank(source, require_row_authority=has_authority)
    old_graph = load_canonical_support_graph(graph_path, old_bank, require_row_authority=has_authority)
    result = compact(source, tmp_path / "migration", [graph_path])
    assert result["validation"]["existing_support_graphs_composed"] == 1
    new_bank = load_canonical_capability_bank(result["destination"], require_row_authority=has_authority)
    new_graph = load_canonical_support_graph(graph_path, new_bank, require_row_authority=has_authority)
    assert new_bank.metadata.get("primitive_row_authority") == old_bank.metadata.get("primitive_row_authority")
    assert result["verified_primitive_row_authority"] == authority
    assert result["original_authority_present"] == has_authority
    assert torch.equal(old_graph.edge_index, new_graph.edge_index)
    assert torch.equal(old_graph.edge_weight, new_graph.edge_weight)
    assert torch.equal(old_graph.raw_affinity, new_graph.raw_affinity)
    assert torch.equal(old_graph.local_sigma, new_graph.local_sigma)



