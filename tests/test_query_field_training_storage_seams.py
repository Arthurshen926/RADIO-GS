"""Storage callback contracts only: synthetic CPU fixtures, no optimizer steps."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_query_preserving_field import example
from radio_gs.query_field.storage import save_field, load_field, sha256, deployment_bytes
from radio_gs.query_field.shared_storage import save_field_v2, load_field_v2


@pytest.fixture(params=["functional_training", "relation_training"])
def fitter(request):
    module = importlib.import_module("radio_gs.query_field." + request.param)
    kwargs = {} if request.param == "functional_training" else {"arm":"cross_view_anchor"}
    source_name = "FunctionalSource" if request.param == "functional_training" else "SourceRelations"
    return module, kwargs, source_name


def source_manifest(tmp_path):
    root = tmp_path/"observations"
    root.mkdir()
    (root/"observations.json").write_text(json.dumps({"fixture":"storage contract"}))
    return root, sha256(root/"observations.json")


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected default loader/writer or training operation")


def test_injected_parent_loader_reaches_source_before_training(tmp_path, monkeypatch, fitter):
    module, kwargs, source_name = fitter
    source, digest = source_manifest(tmp_path)
    field = example()
    manifest = save_field(field, tmp_path/"original", {"observations_sha256":digest})
    calls = []
    def loader(path, device):
        calls.append((path,device))
        return field, manifest
    class SourceReached(Exception):
        pass
    def source_sentinel(path, observed, *args):
        assert path == source and torch.equal(observed,field.observed)
        raise SourceReached
    monkeypatch.setattr(module, "load_field", forbidden)
    monkeypatch.setattr(module, source_name, source_sentinel)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(SourceReached):
        module.fit(tmp_path/"migrated",source,tmp_path/"out",device="cpu",
                   field_loader=loader,save_function=forbidden,**kwargs)
    assert calls == [(tmp_path/"migrated","cpu")]


def test_injected_parent_wrong_source_fails_before_compute(tmp_path, monkeypatch, fitter):
    module, kwargs, source_name = fitter
    source, _ = source_manifest(tmp_path)
    field = example()
    manifest = save_field(field,tmp_path/"original",{"observations_sha256":"0"*64})
    before = {key:value.clone() for key,value in field.state_dict().items()}
    def loader(path,device):
        assert path == tmp_path/"migrated" and device == "cpu"
        return field,manifest
    monkeypatch.setattr(module,"load_field",forbidden)
    monkeypatch.setattr(module,source_name,
                        lambda *args: SimpleNamespace(manifest={"world_units_per_metre":field.world_units_per_metre}))
    monkeypatch.setattr(torch.optim,"Adam",forbidden)
    monkeypatch.setattr(torch.optim,"AdamW",forbidden)
    with pytest.raises(ValueError,match="functional source and parent construction differ|parent and relation observations differ"):
        module.fit(tmp_path/"migrated",source,tmp_path/"out",device="cpu",
                   field_loader=loader,save_function=forbidden,**kwargs)
    for key,value in before.items():
        assert torch.equal(value,field.state_dict()[key]),key
    assert not (tmp_path/"out/field").exists()


def test_zero_step_storage_seams_keep_legacy_identity_and_full_cost(tmp_path, monkeypatch, fitter):
    module, kwargs, source_name = fitter
    source, digest = source_manifest(tmp_path)
    field = example()
    original = save_field(field,tmp_path/"original",{"observations_sha256":digest})
    parent = save_field_v2(field,tmp_path/"migrated",{"original_manifest":original},
                          store_root=tmp_path/"store",reserve_bytes=0)
    class NoTrainingSource:
        manifest = {"world_units_per_metre":field.world_units_per_metre,"raster_hw":[1,1]}
        examples = []
        views = []
        def __init__(self,*args): pass
        sample = pair = forbidden
    class NoStepOptimizer:
        def __init__(self,*args,**kwargs): pass
        step = zero_grad = forbidden
    monkeypatch.setattr(module,source_name,NoTrainingSource)
    monkeypatch.setattr(torch.optim,"Adam",NoStepOptimizer)
    monkeypatch.setattr(torch.optim,"AdamW",NoStepOptimizer)
    if source_name == "FunctionalSource":
        monkeypatch.setattr(module,"source_audit",lambda *args: {"fixture":"no training or audit sampling"})
    # The original entry remains a v1 reconstruction when no callbacks are supplied.
    legacy = module.fit(tmp_path/"original",source,tmp_path/"legacy",epochs=0,device="cpu",**kwargs)
    legacy_field,legacy_manifest = load_field(tmp_path/"legacy/field")
    calls = []
    def migrated_loader(path,device):
        loaded,storage_manifest = load_field_v2(path,device,store_root=tmp_path/"store")
        assert storage_manifest["state_identity"] == parent["state_identity"]
        calls.append("load")
        # The source resolver returns the ORIGINAL v1 construction identity.
        return loaded,original
    def shared_writer(value,path,provenance):
        assert provenance["parent_field_sha256"] == original["files"]["field.pt"]["sha256"]
        assert provenance["observations_sha256"] == digest
        calls.append("save")
        return save_field_v2(value,path,provenance,store_root=tmp_path/"store",reserve_bytes=0)
    monkeypatch.setattr(module,"load_field",forbidden)
    monkeypatch.setattr(module,"save_field",forbidden)
    shared = module.fit(tmp_path/"migrated",source,tmp_path/"shared",epochs=0,device="cpu",
                        field_loader=migrated_loader,save_function=shared_writer,**kwargs)
    shared_field,shared_manifest = load_field_v2(tmp_path/"shared/field",store_root=tmp_path/"store")
    assert calls == ["load","save"]
    assert legacy["steps"] == shared["steps"] == 0
    assert legacy["parent_sha256"] == shared["parent_sha256"] == original["files"]["field.pt"]["sha256"]
    assert legacy["parent_source_schema"] == shared["parent_source_schema"] == original["schema"]
    assert legacy["field_schema"] == "radio_gs.query_preserving_field.v1"
    assert legacy["export_sha256"] == legacy_manifest["files"]["field.pt"]["sha256"]
    assert legacy["export_state_identity"] is None
    assert shared["field_schema"] == "radio_gs.query_preserving_field.v2"
    assert shared["export_sha256"] == shared["export_state_identity"] == shared_manifest["state_identity"]
    assert shared["export_sha256"] != shared["parent_sha256"]
    assert shared["deployment_bytes"] == deployment_bytes(shared_manifest)
    assert shared["deployment_bytes"] >= shared_manifest["unique_blob_bytes"]
    for key,value in legacy_field.state_dict().items():
        assert value.dtype == shared_field.state_dict()[key].dtype
        assert torch.equal(value,shared_field.state_dict()[key]),key
    assert json.loads((tmp_path/"shared/training.json").read_text()) == shared
