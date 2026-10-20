import json
import shutil

import pytest
import torch

from radio_gs.query_field.__main__ import main
from radio_gs.query_field.storage import sha256, load_field_versioned


def fixture(root):
    from test_query_field_functional_training import small_field, source_fixture
    root.mkdir()
    f = small_field()
    source = root/"observations"
    source_fixture(source, f)
    for frame in [1,2,3]:
        path = source/f"view_{frame:05d}.pt"
        view = torch.load(path, weights_only=True)
        view.update(intrinsic=torch.eye(3), world_to_camera=torch.eye(4), original_hw=(9,12), frame=frame)
        torch.save(view, path)
    initial = dict(geometry={k:getattr(f,k) for k in ["xyz","normals","quaternions","scales","opacities"]},
                   semantic=f.semantic.codes, local=f.local.codes, observed=f.observed)
    torch.save(initial, source/"initial.pt")
    manifest = json.loads((source/"observations.json").read_text())
    manifest.update(initial={"sha256":sha256(source/"initial.pt")}, raster_hw=[9,12], geometry_condition="synthetic CPU fixture")
    for entry in manifest["frames"]:
        entry["compiled"]["sha256"] = sha256(source/f"view_{entry['frame']:05d}.pt")
    (source/"observations.json").write_text(json.dumps(manifest))
    config = dict(training=dict(seed=42, group_dim=2, threshold=.5, candidate_count=3,
                  candidate_radius_m=.1, learning_rate=.01, steps=2), evaluation_stages=["trained"])
    config_path = root/"config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_fit_missing_store_rejected_before_source_or_training(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["fit","--output",str(tmp_path/"never"),"--config",str(tmp_path/"missing.json")])
    assert error.value.code == 2 and not (tmp_path/"never").exists()


def test_real_main_defaults_v2_explicit_legacy_and_evaluate(tmp_path, monkeypatch):
    from radio_gs.query_field import evaluate
    run = tmp_path/"new"
    config = fixture(run)
    legacy = tmp_path/"legacy"
    legacy.mkdir()
    shutil.copytree(run/"observations", legacy/"observations")
    store = tmp_path/"store"
    main(["fit","--config",str(config),"--output",str(run),"--device","cpu",
          "--store-root",str(store),"--reserve-bytes","0"])
    main(["fit","--config",str(config),"--output",str(legacy),"--device","cpu","--legacy-v1"])
    a, am = load_field_versioned(run/"models/trained",store_root=store)
    b, bm = load_field_versioned(legacy/"models/trained")
    assert am["schema"].endswith(".v2") and bm["schema"].endswith(".v1")
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    # Synthetic one-element-per-pixel renderer only; no GPU rendering claim.
    monkeypatch.setattr(evaluate, "selected_alpha",
                        lambda field, selected, camera, height, width, scale: selected.float().flatten())
    main(["evaluate","--config",str(config),"--output",str(run),"--device","cpu","--store-root",str(store)])
    main(["evaluate","--config",str(config),"--output",str(legacy),"--device","cpu"])
    new = json.loads((run/"development_trained.json").read_text())
    old = json.loads((legacy/"development_trained.json").read_text())
    assert new["rows"] == old["rows"] and new["summary"] == old["summary"]
    assert new["field_schema"].endswith(".v2")
    assert new["deployment_bytes"] >= am["unique_blob_bytes"]


def test_scannet_main_reads_v2_and_v1_without_injected_loader(tmp_path, monkeypatch):
    from test_query_preserving_field import example
    from radio_gs.query_field import QueryPacket, public_scannet
    from radio_gs.query_field.storage import save_field
    from radio_gs.query_field.shared_storage import save_field_v2
    f = example()
    save_field(f,tmp_path/"v1",{})
    save_field_v2(f,tmp_path/"v2",{},store_root=tmp_path/"store",reserve_bytes=0)
    queries = tmp_path/"queries.json"
    queries.write_text(json.dumps({"queries":[{"text":"a","class_id":1},{"text":"b","class_id":2}]}))
    class Encoder:
        def __init__(self,*args):pass
        def text(self,text,task):
            return QueryPacket("text",semantic=f.semantic.codes[0 if text=="a" else 1],task=task)
    monkeypatch.setattr(public_scannet,"NativeQueryEncoder",Encoder)
    common=["predict","--queries",str(queries),"--encoder","fake","--device","cpu"]
    public_scannet.main(common+["--field",str(tmp_path/"v1"),"--output",str(tmp_path/"p1")])
    public_scannet.main(common+["--field",str(tmp_path/"v2"),"--output",str(tmp_path/"p2"),
                               "--store-root",str(tmp_path/"store")])
    a=torch.load(tmp_path/"p1/predictions.pt",weights_only=True)
    b=torch.load(tmp_path/"p2/predictions.pt",weights_only=True)
    for key in ["xyz","observed","native","member"]:
        assert torch.equal(a[key],b[key])
    with pytest.raises(ValueError,match="explicit store_root"):
        public_scannet.main(common+["--field",str(tmp_path/"v2"),"--output",str(tmp_path/"missing_store")])
