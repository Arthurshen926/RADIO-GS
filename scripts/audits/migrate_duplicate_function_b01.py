"""B01-only, non-destructive duplicate migration; bounded stage-local validation."""
import argparse,datetime,gc,hashlib,io,json,os,resource,sys,time
from pathlib import Path
import torch
from radio_gs.query_field.storage import field_from_state,sha256
from radio_gs.query_field.shared_storage import (VerifiedBlobStage,save_field_v2,load_field_v2,
 _descriptor,_read_manifest,_safe_root,_blob_path)
from scripts.audits import pilot_duplicate_function_storage as base

def read(p):return base.read(p)
def publish(p,v):return base.publish(p,v)
def descriptors(state):return base.state_descriptors(state)
def require_state(raw,expected):
 if descriptors(raw)!=expected:raise ValueError('immutable raw tensor state changed')
def exact_field(raw,manifest,expected,*,preserve):
 field=field_from_state(raw,manifest,preserve_dtypes=preserve)
 base.assert_state(raw,field);require_state(raw,expected)
 if descriptors(field.state_dict())!=expected:raise ValueError('constructor changed tensor state')
 return field
def fixture_for(field):
 return {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in base.probes(field).items()}
def checked_probes(field,fixture,raw,expected):
 require_state(raw,expected)
 result=base.probe_results(field,fixture,'B01')
 require_state(raw,expected)
 if descriptors(field.state_dict())!=expected:raise ValueError('query mutated field state')
 return result
def usage(plan):
 result=base.current_new_usage(plan)
 if sum(x['bytes'] for x in result['metadata'].values())>plan['metadata_reserve_bytes']:
  raise ValueError('metadata reserve exceeded')
 if result['logical']>plan['max_new_logical_bytes'] or result['allocated']>plan['max_new_allocated_bytes']:
  raise ValueError('cumulative data allocation/logical cap exceeded')
 return result
def reserve_write(plan,before,new):
 if sum(x['stat']['bytes'] for x in before['new_blobs'].values())+new+plan['metadata_reserve_bytes']>plan['max_new_logical_bytes']:raise ValueError('logical write reservation exceeded')
 if sum(x['stat']['allocated'] for x in before['new_blobs'].values())+(new*5+3)//4+plan['metadata_reserve_bytes']>plan['max_new_allocated_bytes']:raise ValueError('allocated write reservation exceeded')
def io_account(plan):
 root=Path(plan['new_root']);spent=0;details=[]
 if root.exists():
  for p in root.glob('*/stage_*_intent.json'):
   intent=read(p);terminal=p.with_name(p.name.replace('_intent.json','_result.json'))
   if terminal.exists():
    result=read(terminal)
    # Failed partial operations retain their full reservation until audited.
    amount=result['application_bytes'] if result['status']=='passed' else intent['reserved_application_bytes']
   else:amount=intent['reserved_application_bytes']
   spent+=amount;details.append(dict(path=str(p),charged_bytes=amount,terminal=terminal.exists()))
 return spent,details
def preflight(plan,*,source_checks=True):
 if {g['group_id'] for g in plan['groups']}!={'G49','G48'}:raise ValueError('only B01 groups admitted')
 _safe_root(plan['store_root']);base.safe(plan['new_root'],plan['output_root'])
 resources=base.gate(plan)
 if source_checks:
  for g in plan['groups']:base.checked_sources(plan,g)
  for item in plan['metadata_bindings']:
   p=Path(item['path'])
   if p.is_symlink() or p.stat().st_size!=item['bytes'] or sha256(p)!=item['sha256']:
    raise ValueError('frozen consumer/hash-ledger input changed')
  for path in plan.get('absent_metadata_paths',[]):
   if Path(path).exists():raise ValueError('historically absent consumer result appeared')
 for item in plan['code']:
  if sha256(item['path'])!=item['sha256']:raise ValueError('frozen runtime code changed')
 if plan['new_logical_tensor_upper_bytes']!=sum(g['tensor_bytes'] for g in plan['groups']):
  raise ValueError('tensor budget identity differs')
 resources['visible_process_check']=(base.no_refs([s for g in plan['groups'] for s in g['source_members']]) if source_checks else {'cold_source_metadata_not_opened':True})
 return resources
def start_stage(plan,group,stage,reserved):
 resources=preflight(plan,source_checks=stage!='cold');root=Path(plan['new_root'])/group['group_id']
 if stage=='migrate':root.mkdir(parents=True,exist_ok=False)
 elif not root.is_dir():raise ValueError('publication absent')
 spent,charges=io_account(plan)
 if spent+reserved>plan['max_application_io_bytes']:raise ValueError('cumulative application IO gate')
 path=root/('stage_'+stage+'_intent.json')
 publish(path,dict(stage=stage,group=group['group_id'],started_utc=base.now(),
  reserved_application_bytes=reserved,previous_application_charges=spent,
  resources=resources,plan_identity=base.digest(plan)))
 return root,resources
def read_archive_once(path,expected_sha,expected_bytes,counter):
 before=base.fp(path)
 if before['bytes']!=expected_bytes:raise ValueError('archive byte count changed')
 buffer=io.BytesIO();h=hashlib.sha256()
 try:
  with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as handle:
   st=os.fstat(handle.fileno())
   if (st.st_dev,st.st_ino,st.st_size)!=(before['dev'],before['ino'],before['bytes']):raise ValueError('archive replaced before open')
   while True:
    block=handle.read(8*1024**2)
    if not block:break
    counter['source_read_bytes']+=len(block)
    if buffer.tell()+len(block)>expected_bytes:raise ValueError('source buffer limit exceeded')
    h.update(block);buffer.write(block)
  if buffer.tell()!=expected_bytes or h.hexdigest()!=expected_sha or base.fp(path)!=before:
   raise ValueError('source archive hash/fingerprint differs')
  buffer.seek(0)
  raw=torch.load(buffer,map_location='cpu',weights_only=True)
  if base.fp(path)!=before:raise ValueError('source changed during load')
  return raw
 finally:buffer.close()
def finish_stage(root,stage,status,counter,started,**extra):
 result=dict(status=status,stage=stage,ended_utc=base.now(),seconds=time.monotonic()-started,
  **counter,application_bytes=sum(counter.values()),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  torch_threads=torch.get_num_threads(),**extra)
 publish(root/('stage_'+stage+'_result.json'),result)
 return result
def migrate(plan,group):
 root,resources=start_stage(plan,group,'migrate',group['archive_bytes']+2*group['tensor_bytes'])
 started=time.monotonic();counter=dict(source_read_bytes=0,blob_read_bytes=0,blob_write_bytes=0)
 cache=VerifiedBlobStage(plan['store_root']);records=[]
 try:
  sources=base.checked_sources(plan,group)
  raw=read_archive_once(sources[0][2],group['raw_payload_sha256'],group['archive_bytes'],counter)
  expected=descriptors(raw)
  if sum(x['bytes'] for x in expected.values())!=group['tensor_bytes']:raise ValueError('raw state budget differs')
  fixture=None
  for i,(src,original,payload) in enumerate(sources):
   legacy=exact_field(raw,original,expected,preserve=False);contract=base.spec_contract(original['spec'],legacy.export_spec())
   if fixture is None:
    fixture=fixture_for(legacy);torch.save(fixture,root/'probe_fixture.pt')
   baseline=checked_probes(legacy,fixture,raw,expected);del legacy;gc.collect()
   field=exact_field(raw,original,expected,preserve=True)
   if base.digest(field.export_spec())!=contract['effective_spec_sha256']:raise ValueError('effective scientific spec differs')
   store=Path(plan['store_root']);missing={x['sha256']:x for x in expected.values() if not _blob_path(store,x['sha256']).exists()}
   before=usage(plan);new=sum(x['bytes'] for x in missing.values())
   reserve_write(plan,before,new)
   target=root/('variant_'+str(i))
   publish(root/('intent_'+str(i)+'.json'),dict(target=str(target),missing_blobs=[dict(path=str(_blob_path(store,h)),descriptor=x) for h,x in missing.items()]))
   provenance=dict(original['provenance'],source_manifest=original,source_field_sha256=group['raw_payload_sha256'],
    source_manifest_sha256=src['manifest_sha256'],spec_migration=contract,migration_kind='exact_function_variant_storage_only',
    original_method_binding=group['consumer_bindings'][i])
   manifest=save_field_v2(field,target,provenance,store_root=store,verified_stage=cache,
    max_new_bytes=plan['max_new_logical_bytes']-before['logical'],reserve_bytes=plan['reserve_bytes'])
   counter['blob_read_bytes']=cache.read_bytes;counter['blob_write_bytes']+=new
   require_state(raw,expected)
   if manifest['tensors']!=expected or base.digest(manifest['spec'])!=contract['effective_spec_sha256']:
    raise ValueError('published field identity differs')
   record=dict(group_id=group['group_id'],index=i,source_path=str(payload),source_sha256=group['raw_payload_sha256'],
    source_manifest_sha256=src['manifest_sha256'],target=str(target),store_root=str(store),
    target_manifest_sha256=sha256(target/'manifest.json'),state_identity=manifest['state_identity'],
    original_spec=original['spec'],spec_migration=contract,tensor_descriptors=expected,method_binding=group['consumer_bindings'][i],
    probe_fixture_sha256=sha256(root/'probe_fixture.pt'),expected_probes=baseline,
    new_blobs=[dict(path=str(_blob_path(store,h)),stat=base.fp(_blob_path(store,h))) for h in missing])
   publish(root/('publish_'+str(i)+'.json'),record);records.append(record);usage(plan)
   cache.assert_unchanged();del field;gc.collect()
  if len({r['state_identity'] for r in records})!=3:raise ValueError('three scientific functions lost distinct identity')
  for g in plan['groups']:base.checked_sources(plan,g)
  cache.assert_unchanged()
  result=finish_stage(root,'migrate','passed',counter,started,records={str(i):sha256(root/('publish_'+str(i)+'.json')) for i in range(3)},resources=resources)
  print(json.dumps(result))
 except BaseException as error:
  counter['blob_read_bytes']=cache.read_bytes
  finish_stage(root,'migrate','failed',counter,started,error=repr(error),cost_note='Failed stage charged full reserved budget pending separate audit; partial blobs retained')
  raise
 finally:cache.close()
def install_guard(plan):
 blocked=[Path(os.path.realpath(x)) for x in plan['forbidden_source_roots']]
 blocked += [Path(os.path.realpath(Path(plan['output_root'])/Path(s['path']).parent)) for g in plan['groups'] for s in g['source_members']]
 attempts=[]
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.path.realpath(os.fsdecode(args[0])))
   if any(p==r or r in p.parents for r in blocked):
    attempts.append(str(p));raise PermissionError('old source forbidden')
 sys.addaudithook(guard)
 for p in blocked:
  for leaf in ['field.pt','manifest.json']:
   try:
    with (p/leaf).open('rb'):pass
   except PermissionError:pass
   else:raise ValueError('source guard negative control failed')
 n=len(attempts);attempts.clear()
 return attempts,n
def cold(plan,group,migration_sha):
 root,resources=start_stage(plan,group,'cold',group['tensor_bytes'])
 started=time.monotonic();counter=dict(source_read_bytes=0,blob_read_bytes=0,blob_write_bytes=0)
 cache=VerifiedBlobStage(plan['store_root'],max_tensor_bytes=group['tensor_bytes'])
 try:
  mp=root/'stage_migrate_result.json'
  if sha256(mp)!=migration_sha:raise ValueError('migration result changed')
  migration=read(mp)
  if migration['status']!='passed':raise ValueError('publication incomplete')
  # Preload only sealed small receipt/manifest data, never raw source state.
  records=[]
  for i,source in enumerate(group['source_members']):
   p=root/('publish_'+str(i)+'.json')
   if sha256(p)!=migration['records'][str(i)]:raise ValueError('variant receipt changed')
   r=read(p)
   if (r['target']!=str(root/('variant_'+str(i))) or r['store_root']!=plan['store_root']
       or r['source_path']!=str(Path(plan['output_root'])/source['path'])
       or r['source_sha256']!=group['raw_payload_sha256'] or r['source_manifest_sha256']!=source['manifest_sha256']
       or r['method_binding']!=group['consumer_bindings'][i]):raise ValueError('variant path/source/method binding differs')
   records.append(r)
  attempts,negative_tests=install_guard(plan)
  fixture=torch.load(root/'probe_fixture.pt',map_location='cpu',weights_only=True);accepted=[]
  for i,r in enumerate(records):
   if sha256(root/'probe_fixture.pt')!=r['probe_fixture_sha256']:raise ValueError('probe fixture changed')
   if sha256(Path(r['target'])/'manifest.json')!=r['target_manifest_sha256']:raise ValueError('new manifest changed')
   field,manifest=load_field_v2(r['target'],store_root=plan['store_root'],verified_stage=cache)
   counter['blob_read_bytes']=cache.read_bytes
   cache.assert_unchanged()
   contract=base.spec_contract(r['original_spec'],field.export_spec())
   if (manifest['state_identity']!=r['state_identity'] or manifest['tensors']!=r['tensor_descriptors']
       or contract!=r['spec_migration'] or manifest['provenance']['spec_migration']!=contract
       or manifest['provenance']['original_method_binding']!=r['method_binding']):raise ValueError('cold scientific/method identity differs')
   # The cache owns these tensors; every constructor and probe must leave them exact.
   raw=cache.load_state(plan['store_root'],manifest['tensors'])
   require_state(raw,r['tensor_descriptors'])
   actual=checked_probes(field,fixture,raw,r['tensor_descriptors'])
   cache.assert_unchanged()
   if actual!=r['expected_probes'] or attempts:raise ValueError('cold query differs or source access attempted')
   accepted.append(dict(index=i,manifest_sha256=r['target_manifest_sha256'],state_identity=r['state_identity'],
    method_binding=r['method_binding'],probe_count=len(actual),exact_tensor_dtype_spec=True,exact_query_outputs=True,
    publish_receipt_sha256=sha256(root/('publish_'+str(i)+'.json'))))
   del raw,field;gc.collect()
  counter['blob_read_bytes']=cache.read_bytes
  if counter['blob_read_bytes']>group['tensor_bytes']:raise ValueError('cold unique-blob read budget exceeded')
  result=finish_stage(root,'cold','passed',counter,started,accepted=accepted,migration_sha256=migration_sha,
   validated_blob_fingerprints={h:list(v['stamp']) for h,v in cache.records.items()},
   source_open_negative_tests=negative_tests,forbidden_runtime_attempts=attempts,process_boundary='Independent cold process, group-local immutable tensor cache; all three variants independently constructed and probed',resources=resources)
  print(json.dumps(result))
 except BaseException as error:
  counter['blob_read_bytes']=cache.read_bytes
  finish_stage(root,'cold','failed',counter,started,error=repr(error))
  raise
 finally:cache.close()
def consumer_configs(plan,entries):
 root=Path(plan['new_root']);index={(x['group_id'],x['index']):x for x in entries};result=[]
 for binding in plan.get('consumer_configs',[]):
  original=Path(binding['path'])
  if sha256(original)!=binding['sha256']:raise ValueError('original LERF config changed')
  old=read(original);new=json.loads(json.dumps(old));commands=[]
  for change in binding['overrides']:
   j=change['job_index'];entry=index[(change['group_id'],change['variant_index'])]
   if (old['jobs'][j].get('name')!=change['job_name']
       or old['jobs'][j]['field']!=change['old_field']):raise ValueError('consumer job binding differs')
   new['jobs'][j]['field']=entry['target']
   commands.append(dict(execution_cwd=plan.get('execution_cwd','/root/RADIO-GS-query-field'),job=change['job_name'],state_identity=entry['state_identity'],
    manifest_sha256=entry['target_manifest_sha256'],store_root=plan['store_root']))
  path=root/'consumer_configs'/binding['filename']
  if Path(binding['filename']).name!=binding['filename']:raise ValueError('invalid consumer config name')
  publish(path,new)
  for command in commands:
   command['argv']=['-m','radio_gs.query_field.public_lerf','predict','--config',str(path),
    '--job',command['job'],'--output',str(root/'future_replay'/binding['filename'].removesuffix('.json')/command['job']),
    '--store-root',plan['store_root']]
  result.append(dict(original_path=str(original),original_sha256=binding['sha256'],new_path=str(path),
   new_sha256=sha256(path),commands=commands,validation='Only specified field paths changed; no benchmark executed or historical result adopted'))
 return result
def finalize(plan):
 preflight(plan);root=Path(plan['new_root']);entries=[];union={};before=usage(plan)
 for g in plan['groups']:
  gr=root/g['group_id'];migration=read(gr/'stage_migrate_result.json');cold=read(gr/'stage_cold_result.json')
  if migration['status']!='passed' or cold['status']!='passed' or cold['migration_sha256']!=sha256(gr/'stage_migrate_result.json'):raise ValueError('stage not accepted')
  for i,source in enumerate(g['source_members']):
   p=gr/('publish_'+str(i)+'.json');r=read(p);accepted=cold['accepted'][i]
   if sha256(p)!=migration['records'][str(i)] or accepted['publish_receipt_sha256']!=sha256(p):raise ValueError('publication receipt changed')
   _,m=_read_manifest(r['target'])
   if sha256(Path(r['target'])/'manifest.json')!=r['target_manifest_sha256'] or m['state_identity']!=accepted['state_identity']:raise ValueError('target/cold binding changed')
   for entry in m['tensors'].values():
    p=_blob_path(Path(plan['store_root']),entry['sha256'])
    if not p.is_file() or p.stat().st_size!=entry['bytes']:raise ValueError('closure missing')
    if list(VerifiedBlobStage._stamp(p))!=cold['validated_blob_fingerprints'].get(entry['sha256']):raise ValueError('cold blob identity changed before finalize')
    union[entry['sha256']]=entry
   entries.append({k:r[k] for k in ['group_id','index','source_path','source_sha256','source_manifest_sha256','target','target_manifest_sha256','state_identity','store_root','method_binding','spec_migration']})
 spent,charges=io_account(plan)
 if spent>plan['max_application_io_bytes']:raise ValueError('total application IO exceeded')
 configs=consumer_configs(plan,entries)
 overlay=dict(consumer_configs=configs,schema='radio_gs.function_variant_consumer_overlay.v2',plan_identity=base.digest(plan),entries=entries,
  code=plan['code'],metadata_bindings=plan['metadata_bindings'],retirement_authorized=False,
  boundary='Explicit versioned package/store mapping, not legacy path interception or a new benchmark result; mixture B01 historical prediction/score absent')
 publish(root/'consumer_overlay.json',overlay)
 cost=usage(plan)
 summary=dict(status='B01_non_destructive_validated',source_originals_retained=True,groups=2,variants=6,freed_now_bytes=0,
  old_payload_bytes=sum(s['bytes'] for g in plan['groups'] for s in g['source_members']),
  full_blob_union_bytes=sum(e['bytes'] for e in union.values()),new_unique_blob_bytes=sum(v['stat']['bytes'] for v in cost['new_blobs'].values()),
  new_logical_bytes=cost['logical'],new_allocated_bytes=cost['allocated'],cost_observed_utc=cost['observed_utc'],
  application_bytes_including_failed_reservations=spent,application_charges=charges,observed_df=base.df(root),
  overlay_sha256=sha256(root/'consumer_overlay.json'),cost_note='Application bytes and sampled allocation, not backend NFS measurements; summary itself excluded')
 publish(root/'final_summary.json',summary);print(json.dumps(summary))
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
 p.add_argument('stage',choices=['migrate','cold','finalize']);p.add_argument('--group');p.add_argument('--migration-sha256');args=p.parse_args()
 if sha256(args.plan)!=args.plan_sha256:raise ValueError('plan digest differs')
 plan=read(args.plan);torch.set_num_threads(2);torch.set_num_interop_threads(2)
 if args.stage=='finalize':finalize(plan)
 else:
  group=next(g for g in plan['groups'] if g['group_id']==args.group)
  if args.stage=='migrate':migrate(plan,group)
  else:cold(plan,group,args.migration_sha256)
if __name__=='__main__':main()
