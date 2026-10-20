import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.query_field.unified_five_benchmark import seal, verify_sources, TASKS
from radio_gs.query_field.storage import sha256


def test_nested_public_outputs_are_bounded_and_cannot_escape(tmp_path):
    policy = dict(study_bytes=8_000_000, run_bytes=5_000_000, artifact_bytes=2_000_000, reserve_bytes=1)
    store = BoundedExperiment(tmp_path / "study", "cohort", policy)
    array = np.arange(100,dtype=np.float32)
    path = store.root / "predictions/uqis/example.npy"
    store.public_artifact(path,array,"numpy")
    assert np.array_equal(np.load(path),array)
    tpath = store.root / "predictions/lerf/example.pt"
    store.public_artifact(tpath,dict(scores=torch.arange(20)),"tensor")
    assert torch.equal(torch.load(tpath,weights_only=True)["scores"],torch.arange(20))
    with pytest.raises(ValueError):
        store.public_artifact(tmp_path / "outside.npy",array,"numpy")
    (store.root / "alias").symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):
        store.public_artifact(store.root / "alias/bad.npy",array,"numpy")
    (store.root / "alias").unlink()
    with pytest.raises(ValueError):
        store.public_artifact(store.root / "oversize.npy",np.zeros(600_000,np.float32),"numpy")
    assert not (store.root / "oversize.npy").exists()


def test_score_barrier_requires_every_family_and_unchanged_payload(tmp_path):
    policy = dict(study_bytes=8_000_000, run_bytes=5_000_000, artifact_bytes=2_000_000, reserve_bytes=1)
    config = dict(study_root=str(tmp_path / "study"),storage=policy,architecture="unified_compact_v1")
    store = BoundedExperiment(config["study_root"],"cohort",policy)
    jobs = {f:[] for f in ("lerf","nvos","scannet","uqis")}
    for family in jobs:
        field = tmp_path / family; field.mkdir()
        (field / "manifest.json").write_text(json.dumps(dict(state_identity=family)))
        qfile = tmp_path / (family + "_queries.json");qfile.write_text('{"queries":[{}]}')
        jobs[family]=[dict(name="scene",field=str(field),queries=str(qfile))]
        folder = store.root / "predictions" / family / "scene"
        if family=="uqis":continue
        if family=="nvos":folder /= "scene"
        folder.mkdir(parents=True)
        (folder / "predictions.pt").write_bytes(b"sealed")
        (folder / "prediction.json").write_text(json.dumps(dict(field_sha256=family,
             predictions_sha256=sha256(folder / "predictions.pt"),queries=1 if family=="lerf" else [{}])))
    plan=dict(fields=[],jobs=jobs,source_hashes={},expected={},uqis_query_ids={m:[m] for m in ("text","image","point_2d","point_3d")})
    store.json("plan.json",plan)
    with pytest.raises(FileNotFoundError):seal(config)
    assert not (store.root / "prediction_barrier.json").exists()
    rows=[]
    for modality in plan["uqis_query_ids"]:
        row=dict(query_id=modality)
        for folder,suffix,key in [("predictions",".npy","prediction_sha256"),("selections",".npy","selection_sha256"),("gaussians",".pt","gaussian_sha256")]:
            path=store.root / "predictions/uqis" / folder / (modality+suffix)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"payload")
            row[key]=sha256(path)
        rows.append(row)
    receipt=store.root / "predictions/uqis/receipts/scene.json"
    receipt.parent.mkdir();receipt.write_text(json.dumps(dict(field_sha256="uqis",field_spec=dict(architecture=config["architecture"]),queries=rows)))
    assert seal(config)["tasks"]==list(TASKS)
    assert seal(config)["tasks"]==list(TASKS)
    (store.root / "predictions/lerf/scene/predictions.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="prediction changed"):seal(config)


def test_frozen_method_change_blocks_cohort(tmp_path):
    p=tmp_path / "method.py";p.write_text("first")
    plan=dict(source_hashes={str(p):sha256(p)})
    verify_sources(plan)
    p.write_text("second")
    with pytest.raises(ValueError,match="source changed"):verify_sources(plan)
