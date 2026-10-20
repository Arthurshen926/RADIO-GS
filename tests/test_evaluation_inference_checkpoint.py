"""Small regression tests for explicit public inference-package integration."""
import argparse
import hashlib
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from radio_gs.utils import evaluation_checkpoint as ec
from radio_gs.utils.inference_checkpoint import make_inference_package
from radio_gs.scripts import eval_scannet_pointcloud_radio_gs as scan
from radio_gs.scripts import eval_lerf_grounding as grounding
from radio_gs.scripts import eval_lerf_direct_3d_selection as direct

@pytest.fixture
def package(tmp_path):
    path=tmp_path/"small.inference.pt"
    state={"model_state_dict":{"w":torch.tensor([1.,2.])},"codec_state_dict":{},"optimizer_state_dict":{"x":4},"epoch":9}
    prov={"config_record":{"config":{"architecture":"explicit","latent_dim":4,"fx":12.,"featsharp_strength":.3}},"source_checkpoint":{"sha256":"a"*64}}
    torch.save(make_inference_package(state,prov),path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return path,sha,state

def reader(package):
    path,sha,_=package
    return ec.InferencePackageReader(path,expected_sha256=sha)

def test_legacy_default_delegates_unchanged(monkeypatch):
    calls=[]
    value={"optimizer_state_dict":{"x":2}}
    def load(path,**kw):calls.append((path,kw));return value
    monkeypatch.setattr(ec,"load_trusted_checkpoint",load)
    assert ec.load_evaluation_checkpoint("old.pth",map_location="cpu") is value
    assert calls==[("old.pth",{"expected_sha256":None,"map_location":"cpu"})]

def test_legacy_does_not_unwrap_package(package):
    path,sha,state=package
    raw=ec.load_evaluation_checkpoint(path,map_location="cpu")
    assert "inference_state" in raw and "model_state_dict" not in raw
    assert "optimizer_state_dict" not in ec.load_evaluation_checkpoint(path,checkpoint_reader=reader(package))

def test_cached_read_once_and_path_binding(package,monkeypatch):
    r=reader(package);calls=[];real=ec.load_inference_checkpoint
    def load(*a,**k):calls.append(a[0]);return real(*a,**k)
    monkeypatch.setattr(ec,"load_inference_checkpoint",load)
    a=r.load(package[0]);b=r.load(package[0])
    assert a is b and len(calls)==1
    with pytest.raises(ValueError,match="different checkpoint"):r.load("other.pth")

@pytest.mark.parametrize("sha",[None,"","a"*63,"g"*64,"A"*64])
def test_bad_sha_rejected_before_io(sha):
    with pytest.raises(ValueError):ec.InferencePackageReader("/absent",expected_sha256=sha)

def test_wrong_sha_fail_closed(package):
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        ec.InferencePackageReader(package[0],expected_sha256="0"*64).load(package[0])

def test_resume_rejected_before_io():
    with pytest.raises(ValueError,match="resume"):
        ec.InferencePackageReader("/absent",expected_sha256="a"*64,resume=True)
    r=ec.InferencePackageReader("/absent",expected_sha256="a"*64)
    with pytest.raises(ValueError,match="resume"):r.load("/absent",resume=True)

@pytest.mark.parametrize("changes",[{"fx":14.},{"featsharp_strength":.7},{"latent_dim":7}])
def test_same_shape_config_changes_rejected(package,changes):
    r=reader(package)
    cfg={"architecture":"explicit","latent_dim":4,"fx":12.,"featsharp_strength":.3}
    cfg.update(changes)
    with pytest.raises(ValueError,match="configuration mismatch"):r.bind_config(SimpleNamespace(**cfg))

def test_snapshot_identity_not_rehashed_from_replaced_path(package):
    r=reader(package);r.load(package[0])
    original=ec.evaluation_package_identity(r)
    package[0].write_bytes(b"new bytes in this disposable fixture")
    assert ec.evaluation_checkpoint_sha256(package[0],r,lambda p:pytest.fail("rehash"))==package[1]
    assert ec.evaluation_package_identity(r)==original
    assert original["source_training_checkpoint_sha256"]=="a"*64

def test_cache_config_identity_changes(package):
    r=reader(package);r.load(package[0]);first=ec.evaluation_package_identity(r)
    r.provenance["config_record"]["config"]["fx"]=99.
    assert ec.evaluation_package_identity(r)["inference_package_config_sha256"]!=first["inference_package_config_sha256"]

def test_cli_requires_explicit_format_and_single_path():
    with pytest.raises(ValueError):ec.validate_inference_options(False,"a"*64)
    with pytest.raises(ValueError):ec.validate_inference_options(True,"a"*64,checkpoint_path="{scene}/best.pt")
    ec.validate_inference_options(False,None)

def test_cli_fails_before_external_work():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint");ec.add_inference_checkpoint_arguments(p)
    args=p.parse_args(["--checkpoint","/absent","--inference_package"])
    with pytest.raises(SystemExit):ec.inference_reader_from_args(args,p)

def test_geometry_row_mismatch_fails():
    from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
    m=ExplicitFeatureGaussian(latent_dim=4)
    state={"_xyz":torch.zeros(3,3),"_rotation":torch.zeros(3,4),"_scaling":torch.zeros(3,3),"_opacity":torch.zeros(3,1),"_features_dc":torch.zeros(3,1,3),"_features_rest":torch.zeros(3,0,3),"_feature":torch.zeros(2,4)}
    with pytest.raises(ValueError,match="rows"):ec.initialize_package_geometry(m,state)

def test_direct_metadata_helpers_reuse_reader(package,monkeypatch):
    r=reader(package)
    r.load(package[0])
    monkeypatch.setattr(ec,"load_trusted_checkpoint",lambda *a,**k:pytest.fail("unexpected legacy load"))
    assert direct._load_point_summary_adapter_valid_mask(str(package[0]),checkpoint_reader=r,expected_count=1,device=torch.device("cpu")) is None
    assert direct._load_point_summary_adapter_view_counts(str(package[0]),checkpoint_reader=r,expected_count=1,device=torch.device("cpu")) is None

@pytest.mark.parametrize("module",[scan,grounding,direct])
def test_public_cli_exposes_format_and_sha(module,monkeypatch,capsys):
    monkeypatch.setattr("sys.argv",[module.__name__,"--help"])
    with pytest.raises(SystemExit) as e:module.main()
    assert e.value.code==0
    out=capsys.readouterr().out
    assert "--inference_package" in out and "--expected_checkpoint_sha256" in out

def test_direct_cache_canonicalization_preserves_identity():
    identity={"inference_package_sha256":"b"*64,"inference_package_config_sha256":"c"*64}
    result=direct.canonical_score_cache_metadata(identity)
    assert all(result[k]==v for k,v in identity.items())
