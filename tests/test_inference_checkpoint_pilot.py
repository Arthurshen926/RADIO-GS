"""CPU-only contracts for explicit inference packages; no project weights."""
import hashlib
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from radio_gs.utils.inference_checkpoint import (
    DROP_KEYS, fingerprint, load_inference_checkpoint, make_inference_package,
    read_project_checkpoint_once,
)

def checkpoint():
    return {"model_state_dict":{"w":torch.tensor([1.,-0.])},
            "codec_state_dict":{"w":torch.arange(4)},
            "refiner_state_dict":{}, "depth_head_state_dict":{}, "seg_head_state_dict":{},
            "point_summary_adapter_state_dict":{}, "metrics":{"x":np.float32(0.25)},
            "optimizer_state_dict":{"state":torch.ones(3)},
            "scheduler_state_dict":{"last_epoch":8}, "scaler_state_dict":{},
            "epoch":8, "global_step":100}

def save(path, value):
    with open(path,"xb") as handle:torch.save(value,handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_only_training_keys_removed():
    old=checkpoint();new=make_inference_package(old,{"config":{"scene":"test"}})
    assert set(new["inference_state"])==set(old)-DROP_KEYS
    assert DROP_KEYS.issubset(old)
    assert new["inference_state"]["model_state_dict"] is old["model_state_dict"]
    assert new["resume_capable"] is False and "model_state_dict" not in new

def test_actual_public_loader_roundtrip(tmp_path):
    old=checkpoint();p=tmp_path/"new.pt";sha=save(p,make_inference_package(old,{"config":{}}))
    got,prov,receipt=load_inference_checkpoint(p,expected_sha256=sha)
    assert torch.equal(got["model_state_dict"]["w"].view(torch.uint8),old["model_state_dict"]["w"].view(torch.uint8))
    assert got["metrics"]["x"].tobytes()==old["metrics"]["x"].tobytes()
    assert set(got)==set(old)-DROP_KEYS
    assert receipt["sha256"]==sha
    assert receipt["bytes_read"]==p.stat().st_size
    assert prov=={"config":{}}

def test_resume_rejected_before_io():
    with pytest.raises(ValueError,match="cannot resume"):
        load_inference_checkpoint("/does-not-exist",expected_sha256="0"*64,resume=True)

def test_digest_and_fingerprint_fail_closed(tmp_path):
    p=tmp_path/"new.pt";sha=save(p,make_inference_package(checkpoint(),{}))
    with pytest.raises(ValueError,match="SHA-256"):
        load_inference_checkpoint(p,expected_sha256="0"*64)
    wrong=fingerprint(p.stat());wrong["size"]+=1
    with pytest.raises(ValueError,match="fingerprint"):
        read_project_checkpoint_once(p,expected_fingerprint=wrong)
    with pytest.raises(ValueError,match="bounded"):
        read_project_checkpoint_once(p,max_bytes=1)

def test_symlink_rejected(tmp_path):
    p=tmp_path/"original.pt";sha=save(p,make_inference_package(checkpoint(),{}))
    q=tmp_path/"alias.pt";q.symlink_to(p)
    with pytest.raises(ValueError,match="regular"):
        load_inference_checkpoint(q,expected_sha256=sha)

def test_restricted_pickle_no_unknown_global(tmp_path):
    p=tmp_path/"unsupported.pt";save(p,{"unsupported":complex(1,2)})
    with pytest.raises(pickle.UnpicklingError,match="forbidden"):
        read_project_checkpoint_once(p)

def test_legacy_trainer_fails_before_model_load(monkeypatch):
    import radio_gs.training.artifact_mixin as module
    class Spy:
        called=False
        def load_state_dict(self,*args,**kwargs):
            self.called=True
            raise AssertionError("must not touch model")
    spy=Spy();fake=SimpleNamespace(device="cpu",model=spy)
    monkeypatch.setattr(module,"load_trusted_checkpoint",lambda *a,**k:make_inference_package(checkpoint(),{}))
    with pytest.raises(KeyError,match="model_state_dict"):
        module.TrainingArtifactMixin.load_checkpoint(fake,"unused",resume=True)
    assert spy.called is False

def test_old_plain_dict_is_not_new_format(tmp_path):
    p=tmp_path/"old.pt";sha=save(p,checkpoint())
    with pytest.raises(ValueError,match="explicit inference-only"):
        load_inference_checkpoint(p,expected_sha256=sha)

def test_exclusive_write_preserves_existing(tmp_path):
    p=tmp_path/"existing";p.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        with open(p,"xb") as handle:handle.write(b"replace")
    assert p.read_bytes()==b"keep"
