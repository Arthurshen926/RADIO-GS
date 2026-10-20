"""Tiny CPU rehearsal of the exact non-destructive duplicate pilot script."""
import copy,importlib,json,shutil,subprocess,sys
from pathlib import Path
import pytest,torch
from test_query_preserving_field import example
from test_query_field_mixture_evidence import data
from test_query_field_native_modes import state as mode_state
from radio_gs.query_field.storage import save_field,sha256
from radio_gs.query_field.soft_paired_evidence import enable
from radio_gs.query_field.native_modes import NativeModes
from scripts.audits import pilot_duplicate_function_storage as pilot

def fixture_plan(tmp_path,historical=False):
 output=tmp_path/'output';output.mkdir();store=output/'store';store.mkdir()
 f=example();enable(f,data([[1,1],[0,1]]+[[0,0]]*10,[[0,0],[1,0]]+[[0,0]]*10,[0,0]))
 native=example();native.native_modes=NativeModes(mode_state()[0],12)
 groups=[]
 for gid,field,rules in [('G01',f,['soft_partial_view_evidence_v1','mixture_partial_view_evidence_v1','confidence_mixture_partial_view_evidence_v1']),('G14',native,['measure','mean'])]:
  members=[]
  for i,rule in enumerate(rules):
   if gid=='G01':field.evidence_kind=rule
   else:field.native_mode_rule=rule
   directory=output/'old'/gid/str(i);m=save_field(field,directory,{})
   if i:
    shutil.copyfile(output/'old'/gid/'0/field.pt',directory/'field.pt')
    m['files']['field.pt']['sha256']=sha256(directory/'field.pt');(directory/'manifest.json').write_text(json.dumps(m))
   if historical:
    m['spec'].pop('semantic_measure_rule')
    if gid=='G01':m['spec'].pop('native_mode_rule')
    (directory/'manifest.json').write_text(json.dumps(m))
   entry=dict(pilot.fp(directory/'field.pt'),path=str((directory/'field.pt').relative_to(output)),manifest_sha256=sha256(directory/'manifest.json'))
   members.append(entry)
  groups.append(dict(group_id=gid,raw_payload_sha256=m['files']['field.pt']['sha256'],one_archive_bytes=m['files']['field.pt']['bytes'],source_members=members))
 return dict(output_root=str(output),new_root=str(output/'pilot'),store_root=str(store),minimum_free_bytes=0,minimum_ram_bytes=0,
  max_total_new_bytes=32*1024**2,max_application_io_bytes=128*1024**2,reserve_bytes=0,groups=groups,consumers=[],
  forbidden_source_roots=[str(output/'training')],code=[])

@pytest.mark.parametrize('historical',[False,True])
def test_real_two_groups_and_independent_denied_cold(tmp_path,historical):
 plan=fixture_plan(tmp_path,historical)
 if historical:plan['groups'][1]['artifact_directory']='G14_attempt2'
 for group in plan['groups']:pilot.migrate(plan,group)
 config=tmp_path/'plan.json';config.write_text(json.dumps(plan))
 for group in plan['groups']:
  records=[json.loads(p.read_text()) for p in pilot.group_root(plan,group['group_id']).glob('publish_*.json')]
  assert len({r['state_identity'] for r in records})==len(records)
  assert len({pilot.digest(r['tensor_descriptors']) for r in records})==1
  for i in range(len(group['source_members'])):
   result=subprocess.run([sys.executable,'-m','scripts.audits.pilot_duplicate_function_storage','--plan',str(config),
    '--plan-sha256',sha256(config),'cold','--key',group['group_id']+':'+str(i),'--migration-sha256',sha256(pilot.group_root(plan,group['group_id'])/'migration_complete.json')],capture_output=True,text=True,timeout=45)
   assert result.returncode==0,result.stderr
   receipt=json.loads(result.stdout.strip().splitlines()[-1]);assert receipt['source_open_negative_tests']==12 and not receipt['forbidden_runtime_attempts']
 pilot.finalize(plan)
 assert (Path(plan['new_root'])/'consumer_overlay.json').is_file()
 for group in plan['groups']:pilot.checked_sources(plan,group)
 with pytest.raises(FileExistsError):pilot.migrate(plan,plan['groups'][0])

def test_changed_source_or_manifest_stops_before_load(tmp_path,monkeypatch):
 plan=fixture_plan(tmp_path);group=plan['groups'][0];source=Path(plan['output_root'])/group['source_members'][0]['path']
 with source.open('ab') as h:h.write(b'changed')
 monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('must not read changed state'))
 with pytest.raises(ValueError,match='fingerprint'):pilot.migrate(plan,group)

def test_cap_stops_without_published_package(tmp_path):
 plan=fixture_plan(tmp_path);plan['max_total_new_bytes']=1
 with pytest.raises(ValueError,match='cap|gate'):pilot.migrate(plan,plan['groups'][0])
 assert not list(Path(plan['store_root']).glob('*.bin'))
@pytest.mark.parametrize('effective',[
 {'semantic_measure_rule':'measure','native_mode_rule':'none'},
 {'semantic_measure_rule':'none','native_mode_rule':'none','unknown':'none'},
 {'semantic_measure_rule':'none'},
])
def test_only_two_missing_disabled_defaults_allowed(effective):
 with pytest.raises(ValueError,match='unapproved'):pilot.spec_contract({},effective)

def test_existing_values_cannot_change_and_historical_identity_retained():
 old={'native_mode_rule':'measure','threshold':0.5}
 effective=dict(old,semantic_measure_rule='none')
 contract=pilot.spec_contract(old,effective)
 assert contract['historical_spec']==old and contract['historical_spec_sha256']==pilot.digest(old)
 assert contract['effective_spec_sha256']!=contract['historical_spec_sha256']
 assert contract['added_defaults']=={'semantic_measure_rule':'none'}
 with pytest.raises(ValueError,match='unapproved'):
  pilot.spec_contract(old,dict(effective,native_mode_rule='none'))
 with pytest.raises(ValueError,match='unapproved'):
  pilot.spec_contract(old,dict(effective,threshold=0.6))

@pytest.mark.parametrize('old,new',[(8,8.0),(1,True)])
def test_spec_existing_value_json_type_cannot_change(old,new):
 with pytest.raises(ValueError,match='unapproved'):
  pilot.spec_contract({'candidate_count':old},dict(pilot.SPEC_DEFAULTS,candidate_count=new))

def test_prior_failed_read_cost_enforced_before_payload(tmp_path,monkeypatch):
 plan=fixture_plan(tmp_path);plan['prior_attempt_application_read_bytes']=plan['max_application_io_bytes']
 monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('must not load over budget'))
 with pytest.raises(ValueError,match='IO budget'):pilot.migrate(plan,plan['groups'][0])

def usage_fixture(tmp_path):
 root=tmp_path/'pilot';group=root/'G01';group.mkdir(parents=True)
 blob=tmp_path/'blob.bin';blob.write_bytes(b'hello')
 recorded=pilot.fp(blob);recorded['allocated']=0
 (group/'intent_0.json').write_text(json.dumps({'missing_blobs':[{'path':str(blob),'descriptor':{'bytes':5}}]}))
 (group/'publish_0.json').write_text(json.dumps({'new_blobs':[{'path':str(blob),'stat':recorded}]}))
 return {'new_root':str(root),'max_total_new_bytes':1024**2},blob,recorded

def test_only_allocation_change_uses_fresh_cost(tmp_path):
 plan,blob,recorded=usage_fixture(tmp_path)
 usage=pilot.current_new_usage(plan)
 assert usage['new_blobs'][str(blob)]['stat']['allocated']==pilot.fp(blob)['allocated']>0
 assert usage['allocation_observations'][str(blob)]['published_allocated']==0
 assert json.loads((Path(plan['new_root'])/'G01/publish_0.json').read_text())['new_blobs'][0]['stat']==recorded
 plan['max_total_new_bytes']=usage['allocated']-1
 with pytest.raises(ValueError,match='cap'):pilot.check_cap(plan,usage)

@pytest.mark.parametrize('key',['ino','mtime_ns','ctime_ns','nlink','mode','dev','bytes'])
def test_non_allocation_fingerprint_change_refused(tmp_path,key):
 plan,blob,recorded=usage_fixture(tmp_path)
 path=Path(plan['new_root'])/'G01/publish_0.json';entry=json.loads(path.read_text())
 entry['new_blobs'][0]['stat'][key]='changed'
 path.write_text(json.dumps(entry))
 with pytest.raises(ValueError,match='fingerprint|changed'):pilot.current_new_usage(plan)

def test_failed_group_preserved_and_explicit_resume_directory(tmp_path):
 plan=fixture_plan(tmp_path);group=plan['groups'][0]
 abandoned=Path(plan['new_root'])/'G01';abandoned.mkdir(parents=True);(abandoned/'failed.json').write_text('retained')
 group['artifact_directory']='G01_attempt2';pilot.migrate(plan,group)
 assert (abandoned/'failed.json').read_text()=='retained'
 assert (pilot.group_root(plan,'G01')/'migration_complete.json').is_file()
 with pytest.raises(FileExistsError):pilot.migrate(plan,group)
 group['artifact_directory']='../outside'
 with pytest.raises(ValueError,match='directory'):pilot.group_root(plan,'G01')

