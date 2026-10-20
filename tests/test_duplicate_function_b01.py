"""Small CPU checks for B01 one-read publication and source-denied group cold."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest,torch
from test_query_preserving_field import example
from test_duplicate_function_pilot import fixture_plan
from radio_gs.query_field.shared_storage import (VerifiedBlobStage,save_field_v2,load_field_v2,_descriptor)
from radio_gs.query_field.storage import sha256
from scripts.audits import migrate_duplicate_function_b01 as batch

def plan_for(tmp_path):
 p=fixture_plan(tmp_path,historical=True);g=p['groups'][0];g['group_id']='G49'
 second=copy.deepcopy(g);second['group_id']='G48'
 old=Path(p['output_root'])/'old';shutil.copytree(old/'G01',old/'G48')
 for s in second['source_members']:
  s['path']=s['path'].replace('old/G01/','old/G48/')
  s.update(batch.base.fp(Path(p['output_root'])/s['path']))
 p['groups']=[g,second];p['new_root']=str(Path(p['output_root'])/'b01')
 for g in p['groups']:
  g['archive_bytes']=g['one_archive_bytes']
  g['tensor_bytes']=json.loads((Path(p['output_root'])/g['source_members'][0]['path']).with_name('manifest.json').read_text())['tensor_bytes']
  g['consumer_bindings']=[{'arm':str(i),'scientific_config_sha256':'toy','historical_prediction':'absent'} for i in range(3)]
 p.update(max_new_logical_bytes=32*1024**2,max_new_allocated_bytes=32*1024**2,metadata_reserve_bytes=1024**2,
  new_logical_tensor_upper_bytes=sum(g['tensor_bytes'] for g in p['groups']),metadata_bindings=[])
 return p

@pytest.mark.parametrize('change_existing_blob',[False,True])
def test_two_groups_single_source_read_and_independent_group_cold(tmp_path,change_existing_blob):
 p=plan_for(tmp_path)
 if change_existing_blob:
  first=p['groups'][0]['source_members'][0];source=Path(p['output_root'])/first['path']
  raw=torch.load(source,map_location='cpu',weights_only=True);manifest=json.loads(source.with_name('manifest.json').read_text())
  field=batch.field_from_state(raw,manifest,preserve_dtypes=True)
  existing=save_field_v2(field,tmp_path/'preexisting',{},store_root=p['store_root'])
 for g in p['groups']:
  batch.migrate(p,g)
  result=json.loads((Path(p['new_root'])/g['group_id']/'stage_migrate_result.json').read_text())
  assert result['source_read_bytes']==g['archive_bytes']
  assert result['blob_read_bytes']<=g['tensor_bytes']
 config=tmp_path/'plan.json';config.write_text(json.dumps(p))
 for g in p['groups']:
  gr=Path(p['new_root'])/g['group_id']
  child=subprocess.run([sys.executable,'-m','scripts.audits.migrate_duplicate_function_b01','--plan',str(config),
   '--plan-sha256',sha256(config),'cold','--group',g['group_id'],'--migration-sha256',sha256(gr/'stage_migrate_result.json')],capture_output=True,text=True,timeout=60)
  assert child.returncode==0,child.stderr
  c=json.loads((gr/'stage_cold_result.json').read_text())
  assert c['blob_read_bytes']<=g['tensor_bytes'] and len(c['accepted'])==3
  assert not c['forbidden_runtime_attempts'] and c['source_open_negative_tests']==14
 if change_existing_blob:
  entry=next(iter(existing['tensors'].values()));blob=Path(p['store_root'])/(entry['sha256']+'.bin')
  os.utime(blob,ns=(blob.stat().st_atime_ns,blob.stat().st_mtime_ns+1))
  with pytest.raises(ValueError,match='cold blob identity'):batch.finalize(p)
  assert not (Path(p['new_root'])/'consumer_overlay.json').exists()
  return
 batch.finalize(p)
 assert json.loads((Path(p['new_root'])/'final_summary.json').read_text())['variants']==6
 for g in p['groups']:batch.base.checked_sources(p,g)

def test_stage_hits_do_not_reread_and_original_types_exact(tmp_path):
 store=tmp_path/'store';store.mkdir();f=example().double()
 cache=VerifiedBlobStage(store);a=save_field_v2(f,tmp_path/'a',{},store_root=store,verified_stage=cache)
 f.threshold=0.6;save_field_v2(f,tmp_path/'b',{},store_root=store,verified_stage=cache)
 read_once=cache.read_bytes;assert read_once==a['unique_blob_bytes']
 f.threshold=0.7;save_field_v2(f,tmp_path/'c',{},store_root=store,verified_stage=cache)
 assert cache.read_bytes==read_once
 cold=VerifiedBlobStage(store,max_tensor_bytes=a['tensor_bytes'])
 x,_=load_field_v2(tmp_path/'a',store_root=store,verified_stage=cold);first=cold.read_bytes
 y,_=load_field_v2(tmp_path/'b',store_root=store,verified_stage=cold)
 assert cold.read_bytes==first and first==a['unique_blob_bytes']
 assert all(_descriptor(v)[0]==a['tensors'][k] for k,v in x.state_dict().items())
 assert x.threshold!=y.threshold
 x.semantic.codes.data.add_(1)
 with pytest.raises(ValueError,match='tensor changed'):cold.assert_unchanged()

@pytest.mark.parametrize('kind',['inode','mtime','corrupt','wrong_store','other_process','closed'])
def test_cache_fail_closed(tmp_path,kind):
 store=tmp_path/'store';store.mkdir();m=save_field_v2(example(),tmp_path/'field',{},store_root=store)
 entry=next(iter(m['tensors'].values()));path=store/(entry['sha256']+'.bin')
 stage=VerifiedBlobStage(store);stage.verify(store,entry)
 if kind=='inode':
  replacement=store/'replacement';replacement.write_bytes(path.read_bytes());os.replace(replacement,path)
 elif kind=='mtime':os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1))
 elif kind=='corrupt':
  path.chmod(0o644)
  with path.open('r+b') as h:h.write(b'corrupt')
 elif kind=='other_process':stage.pid=-1
 elif kind=='closed':stage.close()
 elif kind=='wrong_store':
  other=tmp_path/'other';other.mkdir()
  with pytest.raises(ValueError,match='identity'):stage.verify(other,entry)
  return
 with pytest.raises(ValueError,match='changed|identity'):stage.verify(store,entry)

def test_cold_memory_budget_before_read(tmp_path):
 store=tmp_path/'store';store.mkdir();save_field_v2(example(),tmp_path/'field',{},store_root=store)
 stage=VerifiedBlobStage(store,max_tensor_bytes=1)
 with pytest.raises(ValueError,match='memory budget'):load_field_v2(tmp_path/'field',store_root=store,verified_stage=stage)
 assert stage.read_bytes==0

@pytest.mark.parametrize('where',['constructor','probe'])
def test_constructor_or_probe_state_mutation_stops_publication(tmp_path,monkeypatch,where):
 p=plan_for(tmp_path)
 if where=='constructor':
  original=batch.field_from_state
  def bad(*args,**kwargs):
   f=original(*args,**kwargs);f.xyz.data.add_(1);return f
  monkeypatch.setattr(batch,'field_from_state',bad)
 else:
  def bad(field,*args,**kwargs):
   field.group.data.add_(1);return {}
  monkeypatch.setattr(batch.base,'probe_results',bad)
 with pytest.raises(ValueError,match='state|dtype|tensor|query'):batch.migrate(p,p['groups'][0])
 assert not list((Path(p['new_root'])/'G49').glob('variant_*'))


def test_disjoint_group_budget_reserves_metadata_once():
 p=dict(metadata_reserve_bytes=1000,max_new_logical_bytes=4000,max_new_allocated_bytes=4750)
 first=dict(new_blobs={},logical=120,allocated=120)
 batch.reserve_write(p,first,1000)
 second=dict(new_blobs={'only_first':{'stat':{'bytes':1000,'allocated':1250}}},logical=1400,allocated=1650)
 batch.reserve_write(p,second,2000)
 with pytest.raises(ValueError,match='logical'):batch.reserve_write(p,second,2001)

def test_lerf_config_copy_changes_only_bound_fields(tmp_path):
 p=plan_for(tmp_path);old=tmp_path/'old_config.json'
 cfg=dict(schema='test',generic_negative_texts=['object'],jobs=[dict(name='teatime',field='old',queries='q'),dict(name='untouched',field='keep')])
 old.write_text(json.dumps(cfg));original=old.read_bytes()
 p['consumer_configs']=[dict(path=str(old),sha256=sha256(old),filename='soft.json',
  overrides=[dict(job_index=0,job_name='teatime',old_field='old',group_id='G49',variant_index=0)])]
 entries=[dict(group_id='G49',index=0,target='new-package',state_identity='new-state',target_manifest_sha256='new-manifest')]
 output=batch.consumer_configs(p,entries)
 new=json.loads(Path(output[0]['new_path']).read_text())
 assert old.read_bytes()==original and new['jobs'][1]==cfg['jobs'][1]
 new['jobs'][0]['field']='old';assert new==cfg
 argv=output[0]['commands'][0]['argv']
 assert argv[-2:]==['--store-root',p['store_root']]
 with pytest.raises(FileExistsError):batch.consumer_configs(p,entries)

def test_real_fork_cannot_reuse_parent_cache(tmp_path):
 store=tmp_path/'store';store.mkdir();m=save_field_v2(example(),tmp_path/'field',{},store_root=store)
 stage=VerifiedBlobStage(store);entry=next(iter(m['tensors'].values()));stage.verify(store,entry)
 pid=os.fork()
 if pid==0:
  try:stage.verify(store,entry)
  except ValueError:os._exit(0)
  os._exit(9)
 assert os.waitpid(pid,0)[1]==0

def test_source_change_stops_before_archive_read(tmp_path,monkeypatch):
 p=plan_for(tmp_path);source=Path(p['output_root'])/p['groups'][0]['source_members'][0]['path']
 with source.open('ab') as h:h.write(b'x')
 monkeypatch.setattr(batch,'read_archive_once',lambda *a,**k:pytest.fail('source must not load'))
 with pytest.raises(ValueError,match='fingerprint'):batch.migrate(p,p['groups'][0])
