import json
from pathlib import Path
import pytest
import torch

from test_query_preserving_field import example
from radio_gs.query_field.storage import save_field,sha256
from radio_gs.query_field.shared_storage import save_field_v2
from radio_gs.query_field.legacy_field_resolver import create_mapping,FieldResolver
from scripts.research import run_query_field_protected_v2 as runner


def pair(root,name,*,double=False):
    field=example().double() if double else example()
    source=root/'original'/name
    old=save_field(field,source,dict(observations_sha256='a'*64))
    target=root/'v2'/name
    save_field_v2(field,target,dict(source_field_sha256=old['files']['field.pt']['sha256'],source_manifest=old),store_root=root/'store',reserve_bytes=0)
    return field,source,target,old


def resolver_fixture(tmp_path,double=False):
    f,source,target,old=pair(tmp_path,'a',double=double)
    mapping=tmp_path/'mapping.json'
    create_mapping([(source,target,tmp_path/'store')],mapping,output_root=tmp_path)
    return f,source,target,old,FieldResolver(mapping,expected_mapping_sha256=sha256(mapping))


@pytest.mark.parametrize('double',[False,True])
def test_resolver_survives_retired_payload_with_exact_dtype_spec(tmp_path,double):
    f,source,target,old,resolver=resolver_fixture(tmp_path,double)
    (source/'field.pt').rename(source/'field.pt.test-held')
    actual,m=resolver.load(source,old['files']['field.pt']['sha256'])
    assert m==old and actual.export_spec()==f.export_spec()
    for k,v in f.state_dict().items():
        assert v.dtype==actual.state_dict()[k].dtype and torch.equal(v,actual.state_dict()[k]),k
    with pytest.raises(ValueError,match='identity'):
        resolver.load(source,json.loads((target/'manifest.json').read_text())['state_identity'])
    (tmp_path/'store').rename(tmp_path/'held-store')
    with pytest.raises(ValueError,match='root'):
        resolver.load(source,old['files']['field.pt']['sha256'])


def test_mapping_digest_and_original_manifest_fail_closed(tmp_path):
    f,source,target,old,resolver=resolver_fixture(tmp_path)
    with pytest.raises(ValueError,match='mapping digest'):
        FieldResolver(resolver.path,expected_mapping_sha256='f'*64)
    m=json.loads((source/'manifest.json').read_text());m['provenance']['changed']=True
    (source/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='source manifest'):
        resolver.inspect(source,old['files']['field.pt']['sha256'])


def test_registration_rejects_changed_native_or_source_binding(tmp_path):
    f,source,target,old=pair(tmp_path,'a')
    f.semantic.codes[0,0]+=1
    save_field_v2(f,tmp_path/'changed',dict(source_field_sha256=old['files']['field.pt']['sha256'],source_manifest=old),store_root=tmp_path/'store',reserve_bytes=0)
    with pytest.raises(ValueError,match='tensor'):
        create_mapping([(source,tmp_path/'changed',tmp_path/'store')],tmp_path/'bad.json',output_root=tmp_path)
    assert not (tmp_path/'bad.json').exists()


def chain(tmp_path):
    source=tmp_path/'frozen';bindings=[];originals={}
    observations=tmp_path/'observations';observations.mkdir();(observations/'observations.json').write_text('{}')
    for arm in runner.ARMS:
        field=example();directory=source/'fits'/arm/'scene'
        old=save_field(field,directory/'field',dict(observations_sha256=sha256(observations/'observations.json')))
        (directory/'training.json').write_text(json.dumps(dict(export_sha256=old['files']['field.pt']['sha256'],trace=[],protected_buffers_unchanged=True,protected_member_parameters_unchanged=True,protected_state_unchanged=True)))
        originals[arm]=(field,directory,old)
        if arm!='functional_native':
            target=tmp_path/'v2'/arm
            save_field_v2(field,target,dict(source_field_sha256=old['files']['field.pt']['sha256'],source_manifest=old),store_root=tmp_path/'store',reserve_bytes=0)
            bindings.append((directory/'field',target,tmp_path/'store'))
    guard=tmp_path/'guard.json';guard.write_text('{}');record=dict(path=str(guard),sha256=sha256(guard))
    trace=tmp_path/'previous_training.json';trace.write_text(json.dumps(dict(trace=[])))
    parent=originals['functional_native'][1]/'field'
    old=originals['functional_native'][2]
    design=dict(protocol=record,previous_design=record,previous_summary=record,configs=[record],arms={'x':record},
                code=[dict(path=str(runner.PROJECT/'radio_gs/query_field/contracts.py'),sha256='b'*64)],
                fields=[dict(key='scene',parent=str(parent),cross_view_training=dict(path=str(trace),sha256=sha256(trace)),parent_manifest=dict(path=str(parent/'manifest.json'),sha256=sha256(parent/'manifest.json')),
                parent_field=dict(path=str(parent/'field.pt'),sha256=old['files']['field.pt']['sha256']),
                cross_view_field=dict(path=str(parent/'field.pt'),sha256=old['files']['field.pt']['sha256']),
                observations=str(tmp_path/'observations'))],
                functional=dict(epochs=2,seed=20261009,learning_rate=.0001,pairs=64),
                relation=dict(epochs=2,seed=20261008,learning_rate=.001,parameter_policy='group_only'))
    reports=[dict(path=str(source/'fits'/arm/'scene/training.json'),sha256=sha256(source/'fits'/arm/'scene/training.json')) for arm in runner.ARMS]
    exports=[dict(path=str(source/'fits'/arm/'scene/field/manifest.json'),sha256=sha256(source/'fits'/arm/'scene/field/manifest.json')) for arm in runner.ARMS]
    (source/'construction_complete.json').write_text(json.dumps(dict(arms=list(runner.ARMS),records=[dict(key='scene',reports=reports,exports=exports)])))
    (source/'design.json').write_text(json.dumps(design))
    mapping=tmp_path/'mapping.json';create_mapping(bindings,mapping,output_root=tmp_path)
    run=tmp_path/'run'
    runner.prepare(source,sha256(source/'design.json'),mapping,sha256(mapping),run,tmp_path/'store')
    return source,run,originals


def test_versioned_resume_mapping_and_unmigrated_parent_no_training(tmp_path):
    source,run,originals=chain(tmp_path)
    for arm in ['decoder_rollback','protected_training']:
        p=originals[arm][1]/'field/field.pt';p.rename(p.with_suffix('.held'))
    before=sha256(source/'design.json')
    report=runner.main(['build','--root',str(run),'--store-root',str(tmp_path/'store'),'--keys','scene'])
    assert len(report['records'])==3 and not report['training_launched']
    assert [x['resolution']['mode'] for x in report['records']]==['mapped_v2','mapped_v2','unmigrated_v1']
    assert runner.build(run,store_root=tmp_path/'store')==report
    assert sha256(source/'design.json')==before
    with pytest.raises(ValueError,match='run/store'):
        runner.build(run,store_root=tmp_path/'wrong-store')
    p=originals['protected_training'][1]/'training.json';p.rename(p.with_suffix('.held'))
    with pytest.raises(ValueError,match='frozen|partial'):
        runner.build(run,store_root=tmp_path/'store')


def test_functional_fit_loads_migrated_parent_before_operation(tmp_path,monkeypatch):
    from radio_gs.query_field import functional_training
    source,run,originals=chain(tmp_path)
    field,directory,old=originals['protected_training']
    (directory/'field/field.pt').rename(directory/'field/field.pt.held')
    captured=[]
    def fake(parent,observations,output,**kwargs):
        actual,manifest=kwargs['field_loader'](parent,'cpu')
        assert manifest==old
        for k,v in field.state_dict().items():assert torch.equal(v,actual.state_dict()[k]) and v.dtype==actual.state_dict()[k].dtype
        captured.append(kwargs)
        return dict(test_only=True)
    monkeypatch.setattr(functional_training,'fit',fake)
    assert runner.fit(run,0,'functional_native',store_root=tmp_path/'store')==dict(test_only=True)
    assert captured[0]['epochs']==2 and captured[0]['seed']==20261009 and callable(captured[0]['save_function'])
    assert not (run/'fits').exists()
    # A new protected result never silently replaces the frozen parent binding.
    (run/'fits/protected_training/scene').mkdir(parents=True)
    assert runner.fit(run,0,'functional_native',store_root=tmp_path/'store')==dict(test_only=True)



def test_explicit_export_proves_archive_sha(tmp_path):
    f,source,target,old,resolver=resolver_fixture(tmp_path)
    with pytest.raises(ValueError,match='size cap'):
        resolver.export_legacy(source,old['files']['field.pt']['sha256'],tmp_path/'too-big',max_export_bytes=1)
    assert not (tmp_path/'too-big').exists()
    result=resolver.export_legacy(source,old['files']['field.pt']['sha256'],tmp_path/'deploy')
    assert result['byte_exact_legacy_archive']
    assert sha256(tmp_path/'deploy/field.pt')==old['files']['field.pt']['sha256']


def test_new_fit_partial_resume_is_rejected(tmp_path):
    source,run,_=chain(tmp_path)
    (run/'fits/functional_native/scene/field').mkdir(parents=True)
    with pytest.raises(ValueError,match='partial'):
        runner.verify_new_fit(run,0,'functional_native',store_root=tmp_path/'store')

def test_overlay_rejects_symlink_and_frozen_source_store(tmp_path):
    source,run,_=chain(tmp_path)
    overlay=json.loads((run/'storage_overlay.json').read_text())
    alias=tmp_path/'alias';alias.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        runner.prepare(source,overlay['source_design_sha256'],overlay['mapping'],overlay['mapping_sha256'],alias/'new',tmp_path/'store')
    with pytest.raises(ValueError,match='inside frozen'):
        runner.prepare(source,overlay['source_design_sha256'],overlay['mapping'],overlay['mapping_sha256'],tmp_path/'new',source/'new-store')
    run.rename(tmp_path/'real-run');run.symlink_to(tmp_path/'real-run',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        runner.build(run,store_root=tmp_path/'store')


def test_overlay_rejects_changed_construction_receipt(tmp_path):
    source,run,_=chain(tmp_path)
    p=source/'fits/protected_training/scene/training.json'
    data=json.loads(p.read_text());data['trace']=['changed'];p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='frozen'):
        runner.build(run,store_root=tmp_path/'store')


def test_fit_rejects_changed_observation_before_operation(tmp_path,monkeypatch):
    from radio_gs.query_field import functional_training
    source,run,_=chain(tmp_path)
    (tmp_path/'observations/observations.json').write_text('{"changed":true}')
    monkeypatch.setattr(functional_training,'fit',lambda *a,**k:pytest.fail('must reject before operation'))
    with pytest.raises(ValueError,match='observations'):
        runner.fit(run,0,'functional_native',store_root=tmp_path/'store')

def test_verify_new_protected_fit_rejects_wrong_arm_and_parent(tmp_path):
    source,run,originals=chain(tmp_path)
    f,_,old=originals['functional_native']
    parent_sha=old['files']['field.pt']['sha256']
    obs_sha=sha256(tmp_path/'observations/observations.json')
    directory=run/'fits/protected_training/scene'
    m=save_field_v2(f,directory/'field',dict(parent_field_sha256=parent_sha,observations_sha256=obs_sha,arm='cross_view_anchor',parameter_policy='group_only'),store_root=tmp_path/'store',reserve_bytes=0)
    report=dict(field_schema=m['schema'],export_state_identity=m['state_identity'],export_sha256=m['state_identity'],
                parent_sha256=parent_sha,observations_sha256=obs_sha,arm='cross_view_anchor',
                epochs=2,seed=20261008,learning_rate=.001,parameter_policy='group_only',
                protected_buffers_unchanged=True,protected_member_parameters_unchanged=True)
    p=directory/'training.json';p.write_text(json.dumps(report))
    assert runner.verify_new_fit(run,0,'protected_training',store_root=tmp_path/'store')==report
    for changes in [dict(arm='local_anchor_control'),dict(parent_sha256='f'*64),dict(protected_buffers_unchanged=False)]:
        p.write_text(json.dumps(dict(report,**changes)))
        with pytest.raises(ValueError):
            runner.verify_new_fit(run,0,'protected_training',store_root=tmp_path/'store')


