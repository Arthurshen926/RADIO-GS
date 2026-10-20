"""CPU-only end-to-end storage policy for function-only compiler controls."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
import torch

from test_query_preserving_field import example
from test_query_field_mixture_evidence import data
from radio_gs.query_field.storage import save_field,load_field_versioned,sha256
from radio_gs.query_field.shared_storage import save_field_v2
from radio_gs.query_field.research_storage import (method_binding,prediction_binding,
    check_prediction_binding,storage_binding,tensor_state_identity)
from radio_gs.query_field.soft_paired_evidence import enable


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))


@pytest.mark.parametrize("parent_v2", [False,True])
def test_actual_mixture_compiler_shared_specs_and_exact_tensors(tmp_path,monkeypatch,parent_v2):
    from radio_gs.query_field import compile_mixture_evidence as compiler
    f=example();positive=np.zeros((12,2),np.float32);positive[:6,0]=1;positive[6:,1]=1
    negative=1-positive;enable(f,data(positive,negative,[0,0]))
    obs=tmp_path/'obs';dump(obs/'observations.json',dict(world_units_per_metre=f.world_units_per_metre))
    view=dict(frame=0,pixels=[torch.arange(6)],best=torch.arange(12),
        scales=torch.ones(2),columns=sparse.eye(12,format='csr',dtype=np.float32),
        positive=torch.from_numpy(positive.T.copy()),negative=torch.from_numpy(negative.T.copy()))
    monkeypatch.setattr(compiler,'SourceRelations',lambda *a:SimpleNamespace(manifest=json.loads((obs/'observations.json').read_text()),views=[view]))
    store=tmp_path/'store';parent=tmp_path/'parent'
    provenance=dict(observations_sha256=sha256(obs/'observations.json'))
    if parent_v2:save_field_v2(f,parent,provenance,store_root=store,reserve_bytes=0)
    else:save_field(f,parent,provenance)
    reports=[];loaded=[]
    for confidence in [False,True]:
        destination=tmp_path/str(confidence)
        reports.append(compiler.compile_field(parent,obs,destination,confidence_weighted=confidence,store_root=store))
        loaded.append(load_field_versioned(destination/'field',store_root=store))
    assert loaded[0][1]['tensors']==loaded[1][1]['tensors']
    assert loaded[1][1]['new_blob_bytes']==0
    assert loaded[0][1]['state_identity']!=loaded[1][1]['state_identity']
    assert loaded[0][0].evidence_kind!=loaded[1][0].evidence_kind
    for g,_ in loaded:
        for key,value in f.state_dict().items():
            assert value.dtype==g.state_dict()[key].dtype and torch.equal(value,g.state_dict()[key])
    assert all(r['exact_tensor_payload'] and max(x['coefficient_error'] for x in r['replay'])<1e-5 for r in reports)
    with pytest.raises(FileExistsError):compiler.compile_field(parent,obs,tmp_path/'False',store_root=store)
    if not parent_v2:
        legacy=compiler.compile_field(parent,obs,tmp_path/'legacy',legacy_v1=True)
        assert legacy['export_sha256']==sha256(parent/'field.pt')
        old,_=load_field_versioned(tmp_path/'legacy/field')
        assert old.evidence_kind==loaded[0][0].evidence_kind


@pytest.mark.parametrize('parent_v2',[False,True])
def test_actual_native_compiler_two_rules_share_state(tmp_path,parent_v2):
    from radio_gs.query_field.compile_native_modes import compile_scene
    f=example();parent=tmp_path/'parent'
    if parent_v2:save_field_v2(f,parent,{},store_root=tmp_path/'store',reserve_bytes=0)
    else:save_field(f,parent,{})
    source=tmp_path/'source';source.mkdir();obs=tmp_path/'obs';obs.mkdir()
    geometry=obs/'geometry.pt';torch.save(dict(source_vertex_ids=torch.arange(12)),geometry)
    dump(obs/'observations.json',dict(geometry=dict(path=str(geometry))))
    dump(source/'map.json',dict(query_independent=True,labels_used=False,query_manifest_read=False,source_frames=[0]))
    np.save(source/'mesh_xyz.npy',f.xyz.numpy())
    np.save(source/'region_offsets.npy',np.array([0,2]));np.save(source/'patch_offsets.npy',np.array([0,4]))
    b=np.zeros((12,2),np.float32);b[:6,0]=1;b[6:,1]=1
    sparse.save_npz(source/'membership.npz',sparse.csr_matrix(b))
    sparse.save_npz(source/'region_patch_weights.npz',sparse.csr_matrix(np.array([[1,1,0,0],[0,0,1,1]],np.float32)))
    np.save(source/'patch_features.npy',np.random.default_rng(4).normal(size=(4,6)).astype(np.float32))
    np.save(source/'descriptors.npy',np.random.default_rng(5).normal(size=(2,9)).astype(np.float32))
    job=dict(scene='tiny',legacy_map=str(source/'map.json'),observations=str(obs),budgets=dict(full=[0]))
    r=compile_scene(job,parent,tmp_path/'out',store_root=tmp_path/'store',device='cpu')
    a,am=load_field_versioned(tmp_path/'out/mean/field',store_root=tmp_path/'store')
    b,bm=load_field_versioned(tmp_path/'out/measure/field',store_root=tmp_path/'store')
    assert am['tensors']==bm['tensors'] and bm['new_blob_bytes']==0
    assert am['state_identity']!=bm['state_identity']
    assert a.native_mode_rule=='mean' and b.native_mode_rule=='measure'
    assert r['records'][0]['tensor_state_identity']==r['records'][1]['tensor_state_identity']
    for key,value in f.state_dict().items():assert torch.equal(value,a.state_dict()[key])
    from radio_gs.query_field.runtime import QueryFieldRuntime
    from radio_gs.query_field.contracts import QueryPacket
    packet=QueryPacket('image',semantic=torch.randn(9),local=torch.randn(3,6))
    for field in [a,b]:
        reference=copy.deepcopy(f);reference.native_modes=copy.deepcopy(field.native_modes);reference.native_mode_rule=field.native_mode_rule
        assert torch.equal(QueryFieldRuntime(reference).query(packet).membership_scores,QueryFieldRuntime(field).query(packet).membership_scores)


@pytest.mark.parametrize("module",["mixture_functions","confidence_mixture_functions","native_modes"])
def test_real_runner_cli_missing_store_before_prepare(tmp_path,monkeypatch,module):
    runner=importlib.import_module('scripts.research.run_query_field_'+module)
    called=[];monkeypatch.setattr(runner,'prepare',lambda:called.append(1))
    with pytest.raises(ValueError,match='store_root'):runner.main(['prepare','--root',str(tmp_path/'new')])
    assert not called and not (tmp_path/'new').exists()
    runner.main(['prepare','--root',str(tmp_path/'new'),'--store-root',str(tmp_path/'store')])
    assert called==[1] and runner.STORE_ROOT==str(tmp_path/'store') and not runner.LEGACY_V1
    with pytest.raises(ValueError,match='mutually'):runner.main(['prepare','--legacy-v1','--store-root',str(tmp_path/'store')])


def test_resume_binds_method_queries_store_and_code(tmp_path):
    f=example();store=tmp_path/'store';save_field_v2(f,tmp_path/'field',{},store_root=store,reserve_bytes=0)
    query=tmp_path/'queries.json';dump(query,dict(query='one'))
    design=dict(storage=storage_binding(store_root=store),code=[dict(sha256='abc')])
    binding=prediction_binding(tmp_path/'field',design,inputs=dict(arm='mean'),query_files=[query])
    check_prediction_binding(dict(prediction_binding=binding),binding)
    for replacement in [dict(arm='measure'),dict(arm='mean',query='changed')]:
        with pytest.raises(ValueError):check_prediction_binding(dict(prediction_binding=binding),prediction_binding(tmp_path/'field',design,inputs=replacement,query_files=[query]))
    dump(query,dict(query='two'))
    with pytest.raises(ValueError):check_prediction_binding(dict(prediction_binding=binding),prediction_binding(tmp_path/'field',design,inputs=dict(arm='mean'),query_files=[query]))
    changed=copy.deepcopy(design);changed['storage']['blob_store']=str(tmp_path/'missing')
    with pytest.raises((ValueError,FileNotFoundError)):prediction_binding(tmp_path/'field',changed,inputs={})
    manifest=tmp_path/'field/manifest.json';m=json.loads(manifest.read_text());m['spec']['threshold']+=.01;dump(manifest,m)
    with pytest.raises(ValueError):prediction_binding(tmp_path/'field',design,inputs={})


@pytest.mark.parametrize("module",["public_lerf","public_nvos"])
def test_public_reader_missing_store_before_output_or_encoder(tmp_path,module):
    reader=importlib.import_module('radio_gs.query_field.'+module)
    save_field_v2(example(),tmp_path/'field',{},store_root=tmp_path/'store',reserve_bytes=0)
    with pytest.raises(ValueError,match='store_root'):
        reader.predict({},dict(name='tiny',field=str(tmp_path/'field')),tmp_path/'pred',device='cpu')
    assert not (tmp_path/'pred').exists()


@pytest.mark.parametrize("legacy",[False,True])
def test_native_prediction_child_argv(tmp_path,monkeypatch,legacy):
    from scripts.research import run_query_field_native_modes as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path/'run');monkeypatch.setattr(runner,'STORE_ROOT',None if legacy else str(tmp_path/'store'))
    monkeypatch.setattr(runner,'LEGACY_V1',legacy)
    seen=[];monkeypatch.setattr(runner,'execute',lambda argv,*a:seen.append(argv))
    monkeypatch.setattr(runner,'lanes',lambda tasks,gpus,action:action(0,0))
    monkeypatch.setattr(runner,'verify_scene_cache',lambda *a:None)
    monkeypatch.setattr(runner,'verify',lambda *a:[])
    runner.predict(dict(fields=[dict(scene='tiny')]),[0])
    assert seen[0][seen[0].index('--root')+1]==str(tmp_path/'run')
    assert ('--legacy-v1' in seen[0])==legacy
    assert ('--store-root' in seen[0])!=legacy


def test_common_scene_only_lerf_cache_and_score_identity(tmp_path,monkeypatch):
    from scripts.research import run_query_field_relations as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path/'run')
    f=example();store=tmp_path/'store';field=tmp_path/'field';save_field_v2(f,field,{},store_root=store,reserve_bytes=0)
    query=tmp_path/'queries.json';dump(query,dict(queries=[]))
    job=dict(scene='tiny',field=str(field))
    effective=dict(name='tiny',field=str(field),queries=str(query))
    config=runner.ROOT/'arm/configs/lerf.json';dump(config,dict(jobs=[effective]))
    cohort=tmp_path/'cohort.json';dump(cohort,{})
    d=dict(storage=storage_binding(store_root=store),arms=dict(arm=dict(path=str(cohort))))
    binding=runner.current_prediction_binding(d,'arm','lerf',job)
    complete=tmp_path/'prediction.json';dump(complete,dict(field_binding=binding['field']))
    with pytest.raises(FileNotFoundError):runner.verify_prediction_cache(complete,d,'arm','lerf',job)
    dump(Path(str(complete)+'.binding.json'),dict(prediction_binding=binding))
    runner.verify_prediction_cache(complete,d,'arm','lerf',job)
    mpath=field/'manifest.json';m=json.loads(mpath.read_text());m['provenance']['changed']=True;dump(mpath,m)
    with pytest.raises(ValueError):runner.verify_prediction_cache(complete,d,'arm','lerf',job)
    dump(runner.ROOT/'design.json',d);dump(runner.ROOT/'predictions_complete.json',dict(records=[]))
    evaluation=tmp_path/'evaluation.json';dump(evaluation,dict(protocol='a'))
    score_job=dict(evaluation=str(evaluation))
    old=runner.current_score_binding(d,'arm','scannet',score_job,{})
    dump(evaluation,dict(protocol='b'))
    assert old!=runner.current_score_binding(d,'arm','scannet',score_job,{})


def test_partial_compile_resume_missing_closure_rejected(tmp_path,monkeypatch):
    from scripts.research import run_query_field_mixture_functions as runner
    f=example();store=tmp_path/'store';parent=tmp_path/'parent';out=tmp_path/'out'
    save_field(f,parent,{})
    m=save_field_v2(f,out/'field',{},store_root=store,reserve_bytes=0)
    report=dict(storage=storage_binding(store_root=store),export_binding=method_binding(out/'field'),
        parent_binding=method_binding(parent),exact_tensor_payload=True)
    dump(out/'construction.json',report)
    monkeypatch.setattr(runner,'STORE_ROOT',str(store));monkeypatch.setattr(runner,'LEGACY_V1',False)
    job=dict(parent=str(parent),parent_binding=method_binding(parent))
    runner.validate_compile(job,out)
    blob=store/(next(iter(m['tensors'].values()))['sha256']+'.bin');blob.rename(blob.with_suffix('.held'))
    with pytest.raises(ValueError,match='closure'):runner.validate_compile(job,out)


def test_float64_function_state_preserved(tmp_path,monkeypatch):
    # Same actual compiler fixture as above, with exact float64 source state.
    from radio_gs.query_field import compile_mixture_evidence as compiler
    f=example();p=np.zeros((12,1),np.float32);p[:6]=1;enable(f,data(p,1-p,[0]));f=f.double()
    obs=tmp_path/'obs';dump(obs/'observations.json',dict(world_units_per_metre=f.world_units_per_metre))
    view=dict(frame=0,pixels=[torch.arange(6)],best=torch.arange(12),scales=torch.ones(1),
        columns=sparse.eye(12,format='csr',dtype=np.float32),positive=torch.from_numpy(p.T.copy()),negative=torch.from_numpy((1-p).T.copy()))
    monkeypatch.setattr(compiler,'SourceRelations',lambda *a:SimpleNamespace(manifest=json.loads((obs/'observations.json').read_text()),views=[view]))
    store=tmp_path/'store';parent=tmp_path/'parent'
    save_field_v2(f,parent,dict(observations_sha256=sha256(obs/'observations.json')),store_root=store,reserve_bytes=0)
    compiler.compile_field(parent,obs,tmp_path/'out',store_root=store)
    g,_=load_field_versioned(tmp_path/'out/field',store_root=store)
    for key,value in f.state_dict().items():assert value.dtype==g.state_dict()[key].dtype and torch.equal(value,g.state_dict()[key])


def test_public_lerf_actual_v1_v2_cpu_predictions_equal(tmp_path,monkeypatch):
    from radio_gs.query_field import public_lerf as reader
    from radio_gs.query_field.contracts import QueryPacket
    f=example();save_field(f,tmp_path/'v1',{});save_field_v2(f,tmp_path/'v2',{},store_root=tmp_path/'store',reserve_bytes=0)
    class Encoder:
        def __init__(self,*a):pass
        def text(self,text):return QueryPacket('text',semantic=f.semantic.codes[0 if text=='object' else 1])
    monkeypatch.setattr(reader,'NativeQueryEncoder',Encoder)
    queries=tmp_path/'queries.json';dump(queries,dict(queries=[dict(modality='text',text='object')]))
    cfg=dict(shared_encoder='tiny',generic_negative_texts=['background'],native_temperature=10)
    for mode in ['v1','v2']:
        job=dict(name='tiny',field=str(tmp_path/mode),queries=str(queries))
        reader.predict(cfg,job,tmp_path/('pred_'+mode),device='cpu',store_root=None if mode=='v1' else tmp_path/'store')
    a=torch.load(tmp_path/'pred_v1/tiny/predictions.pt',weights_only=True)
    b=torch.load(tmp_path/'pred_v2/tiny/predictions.pt',weights_only=True)
    for k in a['scores']:assert torch.equal(a['scores'][k],b['scores'][k])
    assert torch.equal(a['selected'],b['selected'])


@pytest.mark.parametrize("module",["mixture_functions","confidence_mixture_functions","native_modes"])
def test_actual_compile_cli_threads_storage_policy(tmp_path,monkeypatch,module):
    runner=importlib.import_module('scripts.research.run_query_field_'+module)
    compiler=importlib.import_module('radio_gs.query_field.compile_'+('native_modes' if module=='native_modes' else 'mixture_evidence'))
    job=dict(parent='parent',observations='obs',key='tiny',scene='tiny',parent_manifest={})
    monkeypatch.setattr(runner,'design' if module=='native_modes' else 'frozen_design',lambda:dict(fields=[job]))
    monkeypatch.setattr(runner,'check',lambda *a:None)
    monkeypatch.setattr(runner,'validate_compile',lambda *a:None)
    seen=[]
    monkeypatch.setattr(compiler,'compile_scene' if module=='native_modes' else 'compile_field',lambda *a,**kw:seen.append(kw))
    runner.main(['compile','--index','0','--root',str(tmp_path/'run'),'--store-root',str(tmp_path/'store')])
    assert seen[0]['store_root']==str(tmp_path/'store') and not seen[0]['legacy_v1']


def test_completed_score_requires_matching_binding(tmp_path,monkeypatch):
    from scripts.research import run_query_field_relations as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path/'run');monkeypatch.setattr(runner,'ARMS',('arm',))
    evaluation=tmp_path/'evaluation-config.json';dump(evaluation,dict(protocol='one'))
    job=dict(scene='tiny',evaluation=str(evaluation));cohort=dict(lerf=[],nvos=[],scannet=[job],uqis=[],uqis_package='unused')
    cohort_path=tmp_path/'cohort.json';dump(cohort_path,cohort)
    d=dict(storage=storage_binding(store_root=tmp_path/'store'),arms=dict(arm=dict(path=str(cohort_path))))
    dump(runner.ROOT/'design.json',d)
    dump(runner.ROOT/'predictions_complete.json',dict(design=runner.receipt(runner.ROOT/'design.json')))
    monkeypatch.setattr(runner,'verify_predictions',lambda *a:[])
    monkeypatch.setattr(runner,'lanes',lambda tasks,gpus,action:action(tasks[0],0))
    monkeypatch.setattr(runner,'execute',lambda *a:pytest.fail('must not rerun completed result'))
    completed=runner.ROOT/'arm/predictions/scannet/tiny/evaluation.json';dump(completed,dict(metric=1))
    with pytest.raises(FileNotFoundError):runner.score(d,[0])
    binding=runner.current_score_binding(d,'arm','scannet',job,cohort)
    dump(Path(str(completed)+'.binding.json'),dict(score_binding=binding,result=runner.receipt(completed)))
    runner.score(d,[0])
    dump(evaluation,dict(protocol='two'))
    with pytest.raises(ValueError,match='cached score'):runner.score(d,[0])


