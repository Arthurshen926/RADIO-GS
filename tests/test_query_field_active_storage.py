import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import torch

from radio_gs.query_field.storage import save_field,load_field,load_field_versioned,sha256
from radio_gs.query_field.shared_storage import save_field_v2
from radio_gs.query_field.research_storage import storage_binding
from test_query_preserving_field import example


def assert_exact(a,b):
    assert a.export_spec()==b.export_spec()
    assert a.state_dict().keys()==b.state_dict().keys()
    for k,v in a.state_dict().items():
        assert v.dtype==b.state_dict()[k].dtype and torch.equal(v,b.state_dict()[k]),k


def region_inputs(tmp_path,shared):
    from test_query_field_soft_paired_evidence import soft_fixture
    from radio_gs.query_field.soft_paired_evidence import enable
    f=example();state=soft_fixture()
    for prefix in ['positive','negative','association']:
        p=state[prefix+'_pointer'];state[prefix+'_pointer']=torch.cat([p,p[-1].expand(8)])
    enable(f,state)
    source=tmp_path/'source';source.mkdir();frames=[];records=[]
    for vi,frame in enumerate([3,7]):
        ids=[2,0] if vi==0 else [1]
        path=source/f'view_{frame:05d}.pt'
        torch.save(dict(frame=frame,semantic_descriptors=torch.arange(27).reshape(3,9).float()+frame,scales_m=torch.ones(3)),path)
        frames.append(dict(frame=frame,split='train',compiled=dict(sha256=sha256(path))))
        records.append(dict(frame=frame,regions=ids))
    (source/'observations.json').write_text(json.dumps(dict(frames=frames,world_units_per_metre=f.world_units_per_metre,splits=dict(train_frames=[3,7],development_frames=[9]))))
    parent=tmp_path/'parent';provenance=dict(observations_sha256=sha256(source/'observations.json'))
    if shared:save_field_v2(f,parent/'field',provenance,store_root=tmp_path/'blobs',reserve_bytes=0)
    else:save_field(f,parent/'field',provenance)
    c=parent/'construction.json';c.write_text(json.dumps(dict(source_records=records)))
    return parent/'field',source,c


@pytest.mark.parametrize('shared_parent',[False,True])
def test_real_region_compiler_exact_shared_and_legacy(tmp_path,shared_parent):
    from radio_gs.query_field.compile_region_semantic import compile_field
    parent,source,c=region_inputs(tmp_path,shared_parent)
    store=tmp_path/'blobs'
    with pytest.raises(ValueError,match='store_root'):
        compile_field('/missing','/missing','/missing',tmp_path/'missing')
    assert not (tmp_path/'missing').exists()
    report=compile_field(parent,source,c,tmp_path/'v2',store_root=store)
    field,m=load_field_versioned(tmp_path/'v2/field',store_root=store)
    assert m['schema'].endswith('.v2') and report['deployment_bytes']>report['additional_tensor_bytes']
    assert field.evidence.region_semantic.shape==(3,9)
    if not shared_parent:
        compile_field(parent,source,c,tmp_path/'v1',legacy_v1=True)
        legacy,_=load_field(tmp_path/'v1/field');assert_exact(legacy,field)


def test_real_joint_compiler_keeps_all_four_exports_exact(tmp_path):
    import joblib
    from test_query_field_joint_observation import classifier
    from radio_gs.querying.direct_anchor_membership import FEATURE_NAMES
    from radio_gs.query_field.compile_joint_observation import compile_scene
    f=example();f.world_units_per_metre=1.
    parent=tmp_path/'parent';save_field(f,parent,{})
    obs=tmp_path/'obs';obs.mkdir();g=obs/'geometry.pt'
    torch.save(dict(source_vertex_ids=torch.arange(12)),g)
    (obs/'observations.json').write_text(json.dumps(dict(geometry=dict(path=str(g),sha256=sha256(g)))))
    source=tmp_path/'legacy';source.mkdir()
    rng=np.random.default_rng(12);frames=np.array([0,0,1,1,2])
    b=sparse.csr_matrix(rng.integers(0,2,(12,5)).astype(np.float32))
    v=sparse.csr_matrix(np.column_stack([np.asarray(b[:,frames==i].max(1).toarray()).ravel() for i in range(3)]))
    sparse.save_npz(source/'membership.npz',b);sparse.save_npz(source/'visibility.npz',v)
    np.save(source/'mesh_xyz.npy',f.xyz.numpy());np.save(source/'descriptors.npy',rng.normal(size=(5,7)).astype(np.float32))
    np.save(source/'region_offsets.npy',np.array([0,2,4,5]))
    manifest=source/'map.json';manifest.write_text(json.dumps(dict(query_independent=True,labels_used=False,query_manifest_read=False,source_frames=[3,7,11])))
    model,_=classifier();checkpoint=tmp_path/'forest.joblib'
    joblib.dump(dict(estimator=model,threshold=.35,feature_names=FEATURE_NAMES,fit=['training_room'],calibration=['calibration_room'],condition='toy independent rooms'),checkpoint)
    job=dict(protected_parent=str(parent),observations=str(obs),legacy_map=str(manifest),scene='test_scene')
    with pytest.raises(ValueError,match='store_root'):
        compile_scene({},tmp_path/'missing',{},None,[])
    assert not (tmp_path/'missing').exists()
    budgets=dict(small=[3,7],full=[3,7,11])
    compile_scene(job,tmp_path/'v1',budgets,checkpoint,[99],legacy_v1=True)
    report=compile_scene(job,tmp_path/'v2',budgets,checkpoint,[99],store_root=tmp_path/'store')
    assert len(report['records'])==4
    from scripts.research.run_query_field_joint_observation import verify_construction
    verify_construction(tmp_path/'v2',store_root=tmp_path/'store')
    with pytest.raises(ValueError,match='format'):
        verify_construction(tmp_path/'v2',legacy_v1=True)
    with pytest.raises(ValueError,match='root'):
        verify_construction(tmp_path/'v2',store_root=tmp_path/'missing-store')
    for row in report['records']:
        a,_=load_field(tmp_path/'v1'/row['arm']/'field')
        b,m=load_field_versioned(tmp_path/'v2'/row['arm']/'field',store_root=tmp_path/'store')
        assert_exact(a,b)
        assert row['field_schema'].endswith('.v2') and row['deployment_bytes']>0
        assert b.evidence.forest is not None if row['arm'].endswith('supervised') else b.evidence.forest is None


@pytest.mark.parametrize('name',['native_region','joint_observation'])
def test_real_runner_cli_fails_before_design_and_propagates_new_defaults(tmp_path,monkeypatch,name):
    import importlib
    runner=importlib.import_module('scripts.research.run_query_field_'+name)
    with pytest.raises(SystemExit) as e:runner.main(['prepare'])
    assert e.value.code==2
    captured=[]
    monkeypatch.setattr(runner,'prepare',lambda **kw:captured.append((runner.ROOT,kw)))
    runner.main(['prepare','--root',str(tmp_path/'new'),'--store-root',str(tmp_path/'store')])
    assert captured==[(tmp_path/'new',dict(store_root=tmp_path/'store',legacy_v1=False))]
    runner.main(['prepare','--root',str(tmp_path/'old'),'--legacy-v1'])
    assert captured[-1]==(tmp_path/'old',dict(store_root=None,legacy_v1=True))


@pytest.mark.parametrize('legacy',[False,True])
def test_joint_real_command_paths_propagate_storage(tmp_path,monkeypatch,legacy):
    from scripts.research import run_query_field_joint_observation as runner
    store=None if legacy else tmp_path/'store';options=dict(store_root=store,legacy_v1=legacy)
    monkeypatch.setattr(runner,'ROOT',tmp_path/'run')
    design=dict(storage=storage_binding(**options),fields=[dict(scene='s'+str(i)) for i in range(9)],arms={a:dict(path=a) for a in runner.ARMS})
    commands=[]
    monkeypatch.setattr(runner,'execute',lambda command,*args:commands.append(command))
    monkeypatch.setattr(runner,'lanes',lambda jobs,gpus,action:[action(j,0) for j in jobs])
    monkeypatch.setattr(runner,'receipt',lambda path:dict(path=str(path),sha256='fixture'))
    monkeypatch.setattr(runner,'write',lambda *args:None)
    monkeypatch.setattr(runner,'read',lambda path:dict(uqis=[dict(name='scene',field='field')],uqis_package='public',encoder='encoder'))
    monkeypatch.setattr(runner,'verify_predictions',lambda d,**kwargs:[])
    monkeypatch.setattr(runner,'verify_construction',lambda *args,**kwargs:None)
    runner.build(design,[0],**options)
    assert len(commands)==9
    for c in commands:
        assert c[c.index('--root')+1]==str(runner.ROOT)
        assert ('--legacy-v1' in c)==legacy
        assert ('--store-root' in c)==(not legacy)
    commands.clear();runner.predict(design,[0],**options)
    assert len(commands)==4
    assert all(('--store-root' in c)==(not legacy) for c in commands)


def test_native_real_subprocess_command_propagates_storage(tmp_path,monkeypatch):
    from scripts.research import run_query_field_native_region as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path/'run')
    monkeypatch.setattr(runner,'design',lambda **kw:dict(configuration=dict(path='config')))
    monkeypatch.setattr(runner,'read',lambda path:dict(jobs=[]))
    commands=[]
    class StopAfterCommands(Exception):pass
    def lanes(jobs,gpus,action):
        for j in jobs:action(j,0)
        raise StopAfterCommands()
    monkeypatch.setattr(runner,'lanes',lanes)
    monkeypatch.setattr(runner,'execute',lambda command,*args:commands.append(command))
    with pytest.raises(StopAfterCommands):runner.run([0],store_root=tmp_path/'store')
    assert len(commands)==4
    assert all(c[c.index('--store-root')+1]==str(tmp_path/'store') and c[c.index('--root')+1]==str(tmp_path/'run') for c in commands)

def test_cached_prediction_rejects_wrong_identity_or_schema(tmp_path):
    from scripts.research.run_query_field_joint_observation import verify_prediction_binding,PROGRAM
    m=save_field_v2(example(),tmp_path/'field',{},store_root=tmp_path/'store',reserve_bytes=0)
    job=dict(field=str(tmp_path/'field'),name='scene')
    report=dict(field_sha256=m['state_identity'],field_schema=m['schema'],scene='scene',query_program=PROGRAM)
    verify_prediction_binding(job,report,store_root=tmp_path/'store')
    for bad in [dict(report,field_sha256='wrong'),dict(report,field_schema='radio_gs.query_preserving_field.v1')]:
        with pytest.raises(ValueError,match='identity|schema'):
            verify_prediction_binding(job,bad,store_root=tmp_path/'store')


def test_uqis_missing_store_rejected_before_public_reads_or_encoding(tmp_path,monkeypatch):
    from radio_gs.query_field import public_uqis
    save_field_v2(example(),tmp_path/'field',{},store_root=tmp_path/'store',reserve_bytes=0)
    monkeypatch.setattr(public_uqis,'NativeQueryEncoder',lambda *a:pytest.fail('must not initialize encoder'))
    with pytest.raises(ValueError,match='store_root'):
        public_uqis.predict(tmp_path/'field','/missing','s',tmp_path/'prediction','/missing',device='cpu')
    assert not (tmp_path/'prediction').exists()

