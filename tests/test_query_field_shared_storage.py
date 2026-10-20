import builtins
from functools import partial
import json
from pathlib import Path
import shutil

import pytest
import torch

from test_query_preserving_field import example
from radio_gs.query_field import QueryPacket
from radio_gs.query_field.storage import save_field, load_field, deployment_bytes
from radio_gs.query_field.shared_storage import (
    save_field_v2, load_field_v2, export_field_v1, report_gc_candidates, _digest)
from radio_gs.query_field.runtime import QueryFieldRuntime


def save(field, destination, store=None, **kwargs):
    return save_field_v2(field, destination, {"source": "/forbidden/training"},
                         store_root=store, reserve_bytes=0, **kwargs)


def test_exact_roundtrip_dtypes_and_query(tmp_path):
    f = example()
    f.semantic.codes = f.semantic.codes.half()
    f.local.codes = f.local.codes.to(torch.bfloat16)
    m = save(f, tmp_path/"one")
    loaded, _ = load_field_v2(tmp_path/"one")
    for key, tensor in f.state_dict().items():
        assert tensor.dtype == loaded.state_dict()[key].dtype
        assert torch.equal(tensor, loaded.state_dict()[key])
    packet = QueryPacket("image", semantic=f.semantic.codes[0], local=f.local.codes[:2])
    a, b = f.query(packet), loaded.query(packet)
    assert torch.equal(a.selected, b.selected)
    assert torch.equal(a.membership_scores, b.membership_scores)
    assert deployment_bytes(m) >= m["unique_blob_bytes"]


def test_float64_constructor_dtypes_preserved(tmp_path):
    f = example().double()
    f.xyz[0, 0] = 1.0000000001
    save(f, tmp_path/"double")
    cold, _ = load_field_v2(tmp_path/"double")
    for key, tensor in f.state_dict().items():
        assert tensor.dtype == cold.state_dict()[key].dtype, key
        assert torch.equal(tensor, cold.state_dict()[key]), key


def test_shares_only_equal_values_and_versions_geometry(tmp_path):
    store = tmp_path/"store"
    f = example()
    a = save(f, tmp_path/"a", store)
    with torch.no_grad():
        f.group[0, 0] += 1
    b = save(f, tmp_path/"b", store)
    assert a["tensors"]["semantic.codes"] == b["tensors"]["semantic.codes"]
    assert a["tensors"]["group"] != b["tensors"]["group"]
    assert b["new_blob_bytes"] == f.group.numel()*f.group.element_size()
    f.semantic.codes[0, 0] += 1
    f.xyz[0, 0] += 1
    c = save(f, tmp_path/"c", store)
    assert c["tensors"]["semantic.codes"] != b["tensors"]["semantic.codes"]
    assert c["row_contract"] != b["row_contract"]
    assert len(list(store.glob("*.bin"))) < sum(len(m["tensors"]) for m in (a,b,c))


def test_explicit_store_and_legacy_reader_fail_closed(tmp_path):
    save(example(), tmp_path/"shared", tmp_path/"store")
    with pytest.raises(ValueError, match="explicit store_root"):
        load_field_v2(tmp_path/"shared")
    with pytest.raises(ValueError, match="unsupported field schema"):
        load_field(tmp_path/"shared")
    with pytest.raises(ValueError):
        load_field_v2(tmp_path/"shared", store_root=tmp_path/"absent")


def test_export_survives_removal_of_shared_sources_and_cold_query(tmp_path, monkeypatch):
    f = example()
    save(f, tmp_path/"shared", tmp_path/"store")
    export_field_v1(tmp_path/"shared", tmp_path/"deployment", store_root=tmp_path/"store")
    shutil.rmtree(tmp_path/"store")
    shutil.rmtree(tmp_path/"shared")
    loaded, _ = load_field(tmp_path/"deployment")
    packet = QueryPacket("text", semantic=f.semantic.codes[0])
    expected = f.query(packet)
    def denied(*args, **kwargs):
        raise AssertionError("query attempted file access")
    monkeypatch.setattr(builtins, "open", denied)
    monkeypatch.setattr(torch, "load", denied)
    assert torch.equal(loaded.query(packet).membership_scores, expected.membership_scores)


@pytest.mark.parametrize("damage", ["missing", "truncated", "same_size_wrong"])
def test_blob_damage_rejected(tmp_path, damage):
    m = save(example(), tmp_path/"one")
    entry = m["tensors"]["xyz"]
    path = tmp_path/"one/blobs"/(entry["sha256"]+".bin")
    path.chmod(0o600)
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(b"x")
    else:
        path.write_bytes(b"x"*entry["bytes"])
    with pytest.raises((ValueError, FileNotFoundError)):
        load_field_v2(tmp_path/"one")


def test_manifest_tampering_and_traversal_rejected(tmp_path):
    save(example(), tmp_path/"one")
    path = tmp_path/"one/manifest.json"
    path.chmod(0o600)
    m = json.loads(path.read_text())
    m["spec"]["threshold"] = .4
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="manifest digest"):
        load_field_v2(tmp_path/"one")
    m["tensors"]["xyz"]["sha256"] = "../escape"
    m.pop("manifest_sha256")
    m["manifest_sha256"] = _digest(m)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="content address"):
        load_field_v2(tmp_path/"one")


def test_symlink_blob_root_and_manifest_rejected(tmp_path):
    m = save(example(), tmp_path/"one")
    path = tmp_path/"one/blobs"/(m["tensors"]["xyz"]["sha256"]+".bin")
    outside = tmp_path/"outside"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load_field_v2(tmp_path/"one")
    (tmp_path/"alias").symlink_to(tmp_path/"one", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_field_v2(tmp_path/"alias")


def test_no_replace_and_budget_and_gc_are_safe(tmp_path):
    f = example()
    save(f, tmp_path/"one", tmp_path/"store")
    before = (tmp_path/"one/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save(f, tmp_path/"one", tmp_path/"store")
    assert before == (tmp_path/"one/manifest.json").read_bytes()
    with pytest.raises(ValueError, match="budget"):
        save(f, tmp_path/"overbudget", tmp_path/"store", max_new_bytes=0)
    report = report_gc_candidates(tmp_path/"store", [tmp_path/"one"])
    assert report["candidates"] == [] and report["deletion_performed"] is False


def test_scannet_real_prediction_path_matches_v1(tmp_path, monkeypatch):
    from radio_gs.query_field import public_scannet
    f = example()
    save_field(f, tmp_path/"v1", {})
    save(f, tmp_path/"v2", tmp_path/"store")
    queries = tmp_path/"queries.json"
    queries.write_text(json.dumps({"queries":[{"text":"a", "class_id":1},{"text":"b", "class_id":2}]}))
    class Encoder:
        def __init__(self, *args): pass
        def text(self, text, task):
            return QueryPacket("text", semantic=f.semantic.codes[0 if text=="a" else 1], task=task)
    monkeypatch.setattr(public_scannet, "NativeQueryEncoder", Encoder)
    a = public_scannet.predict(tmp_path/"v1", queries, tmp_path/"p1", "fake", device="cpu")
    b = public_scannet.predict(tmp_path/"v2", queries, tmp_path/"p2", "fake", device="cpu",
                               field_loader=partial(load_field_v2, store_root=tmp_path/"store"))
    aa = torch.load(tmp_path/"p1/predictions.pt", weights_only=True)
    bb = torch.load(tmp_path/"p2/predictions.pt", weights_only=True)
    for key in ["xyz", "observed", "native", "member"]:
        assert torch.equal(aa[key], bb[key])
    assert a["field_spec"] == b["field_spec"]
    assert b["field_schema"].endswith(".v2")
    assert b["complete_field_payload_bytes"] > 0
    loaded = QueryFieldRuntime.load_v2(tmp_path/"v2", store_root=tmp_path/"store")
    assert torch.equal(loaded.xyz, f.xyz)


def test_full_observation_evidence_and_offsets_roundtrip(tmp_path):
    from test_query_field_joint_observation import fixture, enable
    from radio_gs.query_field.contracts import Support
    f, state, _ = fixture()
    enable(f, state)
    first = save(f, tmp_path/"first", tmp_path/"store")
    second = save(f, tmp_path/"second", tmp_path/"store")
    assert second["new_blob_bytes"] == 0
    cold, _ = load_field_v2(tmp_path/"second", store_root=tmp_path/"store")
    for k, value in f.state_dict().items():
        assert value.dtype == cold.state_dict()[k].dtype
        assert torch.equal(value, cold.state_dict()[k]), k
    assert any(k.startswith("evidence.") for k in first["tensors"])
    packet = QueryPacket("point_2d", positive=Support(torch.tensor([0,1]), torch.tensor([.2,.8])))
    a, b = QueryFieldRuntime(f).query(packet), QueryFieldRuntime(cold).query(packet)
    assert torch.equal(a.membership_scores, b.membership_scores)
    assert torch.equal(a.selected, b.selected)


def test_optics_preparation_and_region_topology_exact(tmp_path):
    from test_query_field_extent_preparation import fixture
    from test_query_field_scope_topology import make
    from radio_gs.query_field.extent_factors import ExtentFactors
    from radio_gs.query_field.scope_topology import ScopeTopology
    f = example()
    optical = dict(pointer=torch.tensor([0,1,2]), index=torch.tensor([0,2]),
                   alpha=torch.tensor([.8,.8]), visibility=torch.ones(2), atom=torch.zeros(2,dtype=torch.long))
    f.extent_factors = ExtentFactors({f"views.{i}."+k:v for i in range(2) for k,v in optical.items()},12,f.opacities)
    f.extent_preparation = fixture(12)
    v = make(torch.tensor([[[True,False]],[[False,True]]]))
    f.scope_topology = ScopeTopology({f"views.{i}."+k:x for i in range(2) for k,x in v.state_dict().items()}, f.extent_preparation)
    save(f, tmp_path/"one")
    cold, _ = load_field_v2(tmp_path/"one")
    for key, tensor in f.state_dict().items():
        assert torch.equal(tensor, cold.state_dict()[key]), key
    assert cold.scope_topology.views[0].compose(torch.ones(2,dtype=torch.bool))[1].all()


def test_pipeline_training_entry_really_selects_v2_writer(tmp_path, monkeypatch):
    from radio_gs.query_field import pipeline_v2, train
    config = tmp_path/"config.json"
    config.write_text("{}")
    def fit(observations, output, config, device, *, save_function):
        return save_function(example(), Path(output)/"trained", {})
    monkeypatch.setattr(train, "fit", fit)
    pipeline_v2.main(["train", "--output", str(tmp_path/"run"), "--config", str(config),
                      "--store-root", str(tmp_path/"store"), "--reserve-bytes", "0", "--device", "cpu"])
    _, m = load_field_v2(tmp_path/"run/trained", store_root=tmp_path/"store")
    assert m["schema"].endswith(".v2")


def test_real_cpu_fit_v1_v2_and_standalone_export(tmp_path):
    from test_query_field_functional_training import small_field, source_fixture
    from radio_gs.query_field.train import fit
    from radio_gs.query_field.storage import sha256
    f = small_field()
    source = tmp_path/"source"
    source_fixture(source, f)
    for frame in [1, 2]:
        path = source/f"view_{frame:05d}.pt"
        view = torch.load(path, weights_only=True)
        view.update(intrinsic=torch.eye(3), world_to_camera=torch.eye(4), original_hw=(9,12))
        torch.save(view, path)
    initial = dict(geometry={k:getattr(f,k) for k in ["xyz","normals","quaternions","scales","opacities"]},
                   semantic=f.semantic.codes, local=f.local.codes, observed=f.observed)
    torch.save(initial, source/"initial.pt")
    manifest = json.loads((source/"observations.json").read_text())
    manifest["initial"] = {"sha256":sha256(source/"initial.pt")}
    manifest["raster_hw"] = [9,12]
    for entry in manifest["frames"]:
        entry["compiled"]["sha256"] = sha256(source/f"view_{entry['frame']:05d}.pt")
    (source/"observations.json").write_text(json.dumps(manifest))
    config = dict(seed=42, group_dim=2, threshold=.5, candidate_count=3,
                  candidate_radius_m=.1, learning_rate=.01, steps=2)
    old = fit(source, tmp_path/"old", config, "cpu", legacy_v1=True)
    new = fit(source, tmp_path/"new", config, "cpu",
              save_function=partial(save_field_v2, store_root=tmp_path/"store", reserve_bytes=0))
    assert old["history"] == new["history"]
    assert new["geometry_unchanged"] and new["native_coordinates_unchanged"]
    for stage in ["untrained", "trained"]:
        a, _ = load_field(tmp_path/"old"/stage)
        b, _ = load_field_v2(tmp_path/"new"/stage, store_root=tmp_path/"store")
        for key, tensor in a.state_dict().items():
            assert torch.equal(tensor, b.state_dict()[key]), key
    export_field_v1(tmp_path/"new/trained", tmp_path/"deployed", store_root=tmp_path/"store")
    c, _ = load_field(tmp_path/"deployed")
    for key, tensor in b.state_dict().items():
        assert torch.equal(tensor, c.state_dict()[key]), key


@pytest.mark.parametrize("kind", ["group", "semantic.codes", "codec", "evidence", "optics"])
def test_missing_required_typed_state_rejected_even_with_rebound_manifest(tmp_path, kind):
    from radio_gs.query_field.field import NativeCoordinates
    from radio_gs.query_field.shared_storage import _digest
    f = example()
    key = kind
    if kind == "codec":
        f.semantic = NativeCoordinates(f.semantic.codes, torch.randn(11,9), torch.randn(11))
        key = "semantic.weight"
    elif kind == "evidence":
        from test_query_field_joint_observation import fixture, enable
        f, state, _ = fixture()
        enable(f, state)
        key = next(k for k in f.state_dict() if k.startswith("evidence.") and "pointer" in k)
    elif kind == "optics":
        from radio_gs.query_field.extent_factors import ExtentFactors
        optical = dict(pointer=torch.tensor([0,1,2]), index=torch.tensor([0,2]),
                       alpha=torch.tensor([.8,.8]), visibility=torch.ones(2), atom=torch.zeros(2,dtype=torch.long))
        f.extent_factors = ExtentFactors({"views.0."+k:v for k,v in optical.items()},12,f.opacities)
        key = "extent_factors.views.0.pointer"
    m = save(f, tmp_path/"one")
    del m["tensors"][key]
    m["row_contract"]["state_keys"] = sorted(m["tensors"])
    m["tensor_bytes"] = sum(e["bytes"] for e in m["tensors"].values())
    m["unique_blob_bytes"] = sum(e["bytes"] for e in {e["sha256"]:e for e in m["tensors"].values()}.values())
    m["state_identity"] = _digest(dict(spec=m["spec"], tensors=m["tensors"], row_contract=m["row_contract"]))
    m.pop("manifest_sha256")
    m["manifest_sha256"] = _digest(m)
    path = tmp_path/"one/manifest.json"
    path.chmod(0o600)
    path.write_text(json.dumps(m))
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        load_field_v2(tmp_path/"one")


def test_float64_standalone_v1_refuses_lossy_constructor(tmp_path):
    f = example().double()
    f.xyz[0,0] = 1.0000000001
    save(f, tmp_path/"one")
    with pytest.raises(ValueError, match="v1 constructors"):
        export_field_v1(tmp_path/"one", tmp_path/"v1")
    assert not (tmp_path/"v1").exists()


def test_direct_fit_requires_explicit_storage_before_source_access(tmp_path, monkeypatch):
    from radio_gs.query_field.train import fit
    def unexpected_compute(*args, **kwargs):
        raise AssertionError("storage must be selected before training starts")
    monkeypatch.setattr(torch, "manual_seed", unexpected_compute)
    with pytest.raises(ValueError, match="explicit save_function"):
        fit(tmp_path/"absent", tmp_path/"default", {}, "cpu")
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit(tmp_path/"absent", tmp_path/"conflict", {}, "cpu",
            legacy_v1=True, save_function=save_field)
    assert not (tmp_path/"default").exists()
    assert not (tmp_path/"conflict").exists()


@pytest.mark.parametrize("program", ["original", "native_evidence_shared_members_v2"])
def test_uqis_cli_all_inputs_v1_v2_exact_and_complete_receipts(tmp_path, monkeypatch, program):
    import numpy as np
    from PIL import Image
    from radio_gs.query_field import public_uqis, geometry
    from radio_gs.query_field.storage import sha256
    f = example()
    old_manifest = save_field(f, tmp_path/"v1", {})
    manifest = save(f, tmp_path/"v2", tmp_path/"store")
    public = tmp_path/"public"
    (public/"queries").mkdir(parents=True)
    np.save(public/"mesh.npy", f.xyz.numpy())
    Image.new("RGB", (2,2), color=(20,40,60)).save(public/"crop.png")
    scene = "synthetic"
    domain = dict(scene_id=scene, mesh_xyz_path="mesh.npy",
                  mesh_xyz_sha256=sha256(public/"mesh.npy"))
    (public/"scenes.json").write_text(json.dumps({"scene_domains":[domain]}))
    query_inputs = dict(
        text=[dict(expression="object")],
        image=[dict(crop_rgb_path="crop.png", crop_rgb_sha256=sha256(public/"crop.png"))],
        point_2d=[dict(raster_size=[2,1], positive_pixel_uv=[0,0],
                       camera_intrinsics=torch.eye(3).tolist(),
                       camera_to_world=torch.eye(4).tolist())],
        point_3d=[dict(point_world_xyz=f.xyz[0].tolist()),
                  dict(point_world_xyz=[1000.,1000.,1000.])])
    query_ids = []
    for modality, inputs in query_inputs.items():
        queries = [dict(scene_id=scene, query_id=f"{modality}_{i}", modality=modality, **data)
                   for i,data in enumerate(inputs)]
        query_ids.extend(query["query_id"] for query in queries)
        (public/"queries"/f"{modality}.json").write_text(json.dumps({"queries":queries}))
    class Encoder:
        def __init__(self, *args): pass
        def text(self, expression):
            return QueryPacket("text", semantic=f.semantic.codes[0])
        def image(self, image):
            assert image.size == (2,2)
            return QueryPacket("image", semantic=f.semantic.codes[0], local=f.local.codes[:2])
    monkeypatch.setattr(public_uqis, "NativeQueryEncoder", Encoder)
    # CPU fixed ray fixture replaces only projection, retaining the real point adapter and member query.
    operator = torch.sparse_coo_tensor(torch.tensor([[0,0,1],[0,1,2]]),
                                      torch.tensor([.2,.8,1.]), (2,len(f.xyz))).coalesce()
    monkeypatch.setattr(geometry, "projection", lambda *args: (None,operator,None,None))
    common = ["--public",str(public),"--scene",scene,"--encoder","fixture",
              "--device","cpu","--query-program",program]
    a = public_uqis.main(common+["--field",str(tmp_path/"v1"),"--output",str(tmp_path/"p1")])
    b = public_uqis.main(common+["--field",str(tmp_path/"v2"),"--output",str(tmp_path/"p2"),
                                 "--store-root",str(tmp_path/"store")])
    assert a["field_sha256"] == old_manifest["files"]["field.pt"]["sha256"]
    assert b["field_sha256"] == manifest["state_identity"]
    assert b["field_schema"] == "radio_gs.query_preserving_field.v2"
    assert a["field_spec"] == b["field_spec"]
    assert b["complete_field_payload_bytes"] == deployment_bytes(manifest)
    assert b["complete_field_payload_bytes"] > manifest["unique_blob_bytes"]
    assert a["query_count"] == b["query_count"] == 5
    assert a["unsupported"] == b["unsupported"] == 1
    for qid in query_ids:
        for folder in ["predictions","selections"]:
            aa = np.load(tmp_path/"p1"/folder/f"{qid}.npy")
            bb = np.load(tmp_path/"p2"/folder/f"{qid}.npy")
            assert aa.dtype == bb.dtype and np.array_equal(aa,bb), (folder,qid)
        aa = torch.load(tmp_path/"p1/gaussians"/f"{qid}.pt", weights_only=True)
        bb = torch.load(tmp_path/"p2/gaussians"/f"{qid}.pt", weights_only=True)
        for key,value in aa.items():
            if value is None:
                assert bb[key] is None
            else:
                assert value.dtype == bb[key].dtype and torch.equal(value,bb[key]), (qid,key)
    assert [{k:v for k,v in row.items() if k != "seconds"} for row in a["queries"]] == [
        {k:v for k,v in row.items() if k != "seconds"} for row in b["queries"]]
    with pytest.raises(ValueError, match="explicit store_root"):
        public_uqis.main(common+["--field",str(tmp_path/"v2"),"--output",str(tmp_path/"missing")])
    assert list((tmp_path/"missing/predictions").glob("*.npy")) == []

