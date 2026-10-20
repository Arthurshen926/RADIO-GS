"""Bounded non-destructive G01/G14 shared-v2 migration and source-denied CPU replay."""
import argparse,datetime,gc,hashlib,json,os,resource,stat,sys,time
from pathlib import Path
import torch
from radio_gs.query_field.storage import sha256,field_from_state,load_field_versioned
from radio_gs.query_field.shared_storage import save_field_v2,_descriptor,_read_manifest,_blob_path,_safe_root
from radio_gs.query_field.contracts import QueryPacket,Support
from radio_gs.query_field.runtime import QueryFieldRuntime

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
SPEC_DEFAULTS={'semantic_measure_rule':'none','native_mode_rule':'none'}
SPEC_RULE='historical_disabled_rules_explicit_none.v1'
def spec_contract(historical,effective):
 expected=dict(historical)
 added={k:v for k,v in SPEC_DEFAULTS.items() if k not in historical}
 expected.update(added)
 if digest(effective)!=digest(expected):raise ValueError('unapproved effective spec difference')
 return dict(rule=SPEC_RULE,historical_spec=dict(historical),historical_spec_sha256=digest(historical),
  effective_spec=dict(effective),effective_spec_sha256=digest(effective),added_defaults=added,
  identity_note='Historical dictionary values retained; exact original JSON bytes bound by source_manifest_sha256')
def publish(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:json.dump(v,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
def fp(p):
 s=Path(p).lstat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('regular file required')
 return dict(dev=s.st_dev,ino=s.st_ino,bytes=s.st_size,allocated=s.st_blocks*512,
             mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,nlink=s.st_nlink,mode=oct(stat.S_IMODE(s.st_mode)))
def safe(p,root):
 p=Path(p).absolute();root=Path(root).absolute()
 if p!=root and root not in p.parents:raise ValueError('outside explicit root')
 for part in [p]+list(p.parents):
  if part.is_symlink():raise ValueError('symlink forbidden')
 return p
def df(root):
 s=os.statvfs(root);return dict(available=s.f_bavail*s.f_frsize,total=s.f_blocks*s.f_frsize)
def gate(plan):
 root=Path(plan['output_root'])
 if df(root)['available']<plan['minimum_free_bytes']:raise ValueError('free-space gate')
 available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 if available<plan['minimum_ram_bytes']:raise ValueError('RAM gate')
 return dict(df=df(root),ram_available=available,torch_threads=torch.get_num_threads())
def group_root(plan,group_id):
 group=next(g for g in plan['groups'] if g['group_id']==group_id)
 name=group.get('artifact_directory',group_id)
 if name not in {group_id,group_id+'_attempt2'}:raise ValueError('unapproved group directory')
 return Path(plan['new_root'])/name
def content_fp(value):return {k:v for k,v in value.items() if k!='allocated'}
def check_input_budget(plan):
 tensor_total=0;representative_tensors=0
 for group in plan['groups']:
  sources=checked_sources(plan,group)
  tensor_total+=sum(m['tensor_bytes'] for _,m,_ in sources)
  representative_tensors+=sources[0][1]['tensor_bytes']
 source_hash=sum(s['bytes'] for g in plan['groups'] for s in g['source_members'])
 source_load=sum(g['one_archive_bytes'] for g in plan['groups'])
 # Conservative: count ALL writer verification reads AND all possible new writes.
 bound=source_hash+source_load+2*tensor_total+representative_tensors+64*1024**2+plan.get('prior_attempt_application_read_bytes',0)
 if bound>plan['max_application_io_bytes']:raise ValueError('precompute IO budget exceeded')
 if bound!=plan.get('application_io_upper_bound_bytes',bound):raise ValueError('frozen IO bound changed')
 for entry in plan.get('metadata_bindings',[]):
  path=Path(entry['path'])
  if path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:raise ValueError('consumer input receipt changed')
 return bound
def state_descriptors(state):return {k:_descriptor(v)[0] for k,v in state.items()}
def assert_state(state,field):
 current=field.state_dict()
 if set(current)!=set(state):raise ValueError('state keys differ')
 for key,value in state.items():
  other=current[key]
  if value.dtype!=other.dtype or value.shape!=other.shape or not torch.equal(value,other):
   raise ValueError('state dtype/value differs: '+key)
def checked_sources(plan,group,hash_payload=False):
 root=Path(plan['output_root']);answer=[]
 for source in group['source_members']:
  p=safe(root/source['path'],root);actual=fp(p)
  if actual!={k:source[k] for k in actual}:raise ValueError('source fingerprint changed: '+str(p))
  mp=p.with_name('manifest.json');safe(mp,root)
  if sha256(mp)!=source['manifest_sha256']:raise ValueError('source manifest changed')
  m=read(mp)
  if m['schema']!='radio_gs.query_preserving_field.v1' or m['files']['field.pt']!={'sha256':group['raw_payload_sha256'],'bytes':group['one_archive_bytes']}:
   raise ValueError('source archive receipt differs')
  if hash_payload and sha256(p)!=group['raw_payload_sha256']:raise ValueError('fresh archive hash differs')
  if fp(p)!=actual:raise ValueError('source changed during read')
  answer.append((source,m,p))
 return answer
def no_refs(rows):
 ids={(x['dev'],x['ino']) for x in rows};found=[];blocked=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   for fd in (proc/'fd').iterdir():
    try:
     s=fd.stat()
     if (s.st_dev,s.st_ino) in ids:found.append([int(proc.name),fd.name])
    except FileNotFoundError:pass
   for line in (proc/'maps').read_text().splitlines():
    f=line.split(None,5)
    if len(f)>4:
     a,b=f[3].split(':')
     if (os.makedev(int(a,16),int(b,16)),int(f[4])) in ids:found.append([int(proc.name),'mmap'])
  except (FileNotFoundError,ProcessLookupError):pass
  except PermissionError:blocked.append(int(proc.name))
 if found or blocked:raise ValueError('visible source references/inaccessible processes: '+str((found,blocked)))
 return dict(visible_source_refs=0,inaccessible_pids=0,other_hosts_not_visible=True)

def probes(field):
 ids=torch.where(field.observed)[0][:3]
 if len(ids)<2:raise ValueError('insufficient fixed probe support')
 semantic=field.semantic.decode(ids[:1])[0].detach().cpu()
 local=field.local.decode(ids).detach().cpu()
 return dict(semantic=semantic,local=local,positive=ids[:1].cpu(),negative=ids[1:2].cpu(),scale=float(field.default_scale))
@torch.no_grad()
def probe_results(field,fixture,group_id):
 runtime=QueryFieldRuntime(field);scale=fixture['scale']
 packets=dict(text_union=QueryPacket('text',task='union',semantic=fixture['semantic']),
  signed_point=QueryPacket('point_2d',positive=Support(fixture['positive'],torch.ones(1)),
    negative=Support(fixture['negative'],torch.ones(1)),scale=scale))
 if group_id=='G14':
  # Fixed synthetic native/resized-shaped packets, NOT a real image encoding.
  packets['native_input_proxy']=QueryPacket('image',semantic=fixture['semantic'],local=fixture['local'],
     local_weights=torch.arange(1,len(fixture['local'])+1,dtype=torch.float32))
  packets['resized_input_proxy']=QueryPacket('image',semantic=fixture['semantic'],local=fixture['local'][:2])
 results={}
 for name,packet in packets.items():
  answer=runtime.query(packet)
  results[name]=dict(scale=float(answer.scale),selected_count=int(answer.selected.sum()),
   tensors={k:None if getattr(answer,k) is None else _descriptor(getattr(answer,k))[0]
    for k in ['semantic_scores','membership_scores','selected','candidate_indices','candidate_weights']})
 return results

def metadata_files(root):
 answer={};total=0
 for p in Path(root).rglob('*'):
  if p.is_symlink():raise ValueError('pilot symlink')
  if not p.is_file():continue
  s=fp(p);total+=s['bytes']
  if total>16*1024**2:raise ValueError('pilot metadata/fixture budget exceeded')
  answer[str(p)]=s
 return answer
def current_new_usage(plan):
 root=Path(plan['new_root']);new={};allocation_observations={}
 if root.exists():
  for record in root.glob('*/intent_*.json'):
   for item in read(record)['missing_blobs']:
    p=Path(item['path'])
    if p.exists():
     s=fp(p)
     if s['bytes']!=item['descriptor']['bytes']:raise ValueError('partial blob size differs')
     new[str(p)]=dict(path=str(p),stat=s)
  for record in root.glob('*/publish_*.json'):
   for item in read(record)['new_blobs']:
    if item['path'] in new and content_fp(new[item['path']]['stat'])!=content_fp(item['stat']):raise ValueError('completed blob fingerprint changed')
    current=fp(item['path'])
    if content_fp(current)!=content_fp(item['stat']):raise ValueError('published blob changed')
    if current['allocated']!=item['stat']['allocated']:
     allocation_observations[item['path']]=dict(published_allocated=item['stat']['allocated'],current_allocated=current['allocated'])
    new[item['path']]=dict(path=item['path'],stat=current)
 for path,item in new.items():
  latest=fp(path)
  if content_fp(latest)!=content_fp(item['stat']):raise ValueError('published blob changed')
  item['stat']=latest
 metadata=metadata_files(root) if root.exists() else {}
 return dict(new_blobs=new,metadata=metadata,observed_utc=now(),allocation_observations=allocation_observations,
  logical=sum(x['stat']['bytes'] for x in new.values())+sum(x['bytes'] for x in metadata.values())+plan.get('prior_attempt_artifact_logical_bytes',0),
  allocated=sum(x['stat']['allocated'] for x in new.values())+sum(x['allocated'] for x in metadata.values())+plan.get('prior_attempt_artifact_allocated_bytes',0))
def check_cap(plan,usage):
 if max(usage['logical'],usage['allocated'])>plan['max_total_new_bytes']:raise ValueError('pilot cumulative new-disk cap exceeded')
def migrate(plan,group):
 root=Path(plan['new_root']);destination=group_root(plan,group['group_id']);destination.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();resources=gate(plan);check_input_budget(plan)
 sources=checked_sources(plan,group,hash_payload=True);process=no_refs([s for s,_,_ in sources])
 # Only one raw archive load per group; all original archives were freshly hashed.
 raw=torch.load(sources[0][2],map_location='cpu',weights_only=True)
 checked_sources(plan,group)
 expected=state_descriptors(raw);reference_fixture=None;records=[];io_read=sum(s['bytes'] for s,_,_ in sources)+sources[0][0]['bytes'];io_write=0
 for index,(source,original,payload) in enumerate(sources):
  # Historical loader semantics must themselves preserve this real raw state.
  legacy=field_from_state(raw,original,preserve_dtypes=False);assert_state(raw,legacy)
  contract=spec_contract(original['spec'],legacy.export_spec())
  if reference_fixture is None:
   reference_fixture=probes(legacy)
   torch.save(reference_fixture,destination/'probe_fixture.pt')
  baseline=probe_results(legacy,reference_fixture,group['group_id'])
  del legacy;gc.collect()
  field=field_from_state(raw,original,preserve_dtypes=True);assert_state(raw,field)
  if field.export_spec()!=contract['effective_spec']:raise ValueError('lossless effective spec differs')
  target=destination/('variant_'+str(index));store=Path(plan['store_root'])
  absent={d['sha256']:d for d in expected.values() if not (store/(d['sha256']+'.bin')).exists()}
  before_usage=current_new_usage(plan);check_cap(plan,before_usage)
  # 25% allocation margin plus 16 MiB metadata budget, in addition to observed use.
  missing=sum(d['bytes'] for d in absent.values())
  if before_usage['allocated']+int(missing*1.25)+16*1024**2>plan['max_total_new_bytes']:raise ValueError('conservative cumulative allocation gate')
  provenance=dict(original['provenance'],source_manifest=original,source_field_sha256=group['raw_payload_sha256'],
      source_manifest_sha256=source['manifest_sha256'],spec_migration=contract,migration_kind='exact_function_variant_storage_only')
  publish(destination/('intent_'+str(index)+'.json'),dict(target=str(target),missing_blobs=[
    dict(path=str(store/(h+'.bin')),descriptor=d) for h,d in absent.items()]))
  manifest=save_field_v2(field,target,provenance,store_root=store,
      max_new_bytes=plan['max_total_new_bytes']-before_usage['logical'],reserve_bytes=plan['reserve_bytes'])
  if manifest['tensors']!=expected or manifest['spec']!=contract['effective_spec']:raise ValueError('published scientific state differs')
  new_blobs=[dict(path=str(store/(h+'.bin')),stat=fp(store/(h+'.bin'))) for h in absent]
  io_write+=sum(d['bytes'] for d in absent.values())
  # Writer verifies one blob for each already-existing tensor key (including intra-state aliases).
  io_read+=sum(d['bytes'] for d in expected.values())-sum(d['bytes'] for d in absent.values())
  record=dict(index=index,group_id=group['group_id'],old_path=str(payload),old_manifest=str(payload.with_name('manifest.json')),
   old_manifest_sha256=source['manifest_sha256'],old_field_sha256=group['raw_payload_sha256'],target=str(target),
   target_manifest_sha256=sha256(target/'manifest.json'),state_identity=manifest['state_identity'],
   store_root=plan['store_root'],original_spec=original['spec'],spec_migration=contract,tensor_descriptors=expected,
   probe_fixture=str(destination/'probe_fixture.pt'),probe_fixture_sha256=sha256(destination/'probe_fixture.pt'),
   expected_probes=baseline,new_blobs=new_blobs,raw_legacy_and_lossless_dtype_value_exact=True)
  publish(destination/('publish_'+str(index)+'.json'),record);records.append(record)
  check_cap(plan,current_new_usage(plan));del field;gc.collect()
 if len({r['state_identity'] for r in records})!=len(records):raise ValueError('distinct functions must have distinct state identities')
 checked_sources(plan,group)
 report=dict(group=group['group_id'],status='published_pending_cold',variants=len(records),
  source_raw_loads=1,source_hash_bytes=sum(s['bytes'] for s,_,_ in sources),source_load_bytes=sources[0][0]['bytes'],
  application_read_bytes=io_read,application_write_bytes=io_write,seconds=time.monotonic()-started,
  resources=resources,process=process,new_usage={k:v for k,v in current_new_usage(plan).items() if k not in ['new_blobs','metadata']},
  record_sha256={str(i):sha256(destination/('publish_'+str(i)+'.json')) for i in range(len(records))},
  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 publish(destination/'migration_complete.json',report);print(json.dumps(report))

def cold(plan,key,migration_sha256):
 group_id,index=key.split(':')
 group=next(g for g in plan['groups'] if g['group_id']==group_id)
 if index not in [str(i) for i in range(len(group['source_members']))]:raise ValueError('unknown variant')
 migration_path=group_root(plan,group_id)/'migration_complete.json'
 if sha256(migration_path)!=migration_sha256:raise ValueError('migration receipt changed')
 record_path=group_root(plan,group_id)/('publish_'+index+'.json')
 migration=read(migration_path);record_sha=sha256(record_path)
 if migration['record_sha256'][index]!=record_sha:raise ValueError('publish receipt changed')
 record=read(record_path);source=group['source_members'][int(index)]
 expected_target=group_root(plan,group_id)/('variant_'+index)
 if record['target']!=str(expected_target) or record['store_root']!=plan['store_root']:raise ValueError('unapproved package/store redirect')
 if (record['old_path']!=str(Path(plan['output_root'])/source['path']) or record['old_field_sha256']!=group['raw_payload_sha256']
     or record['old_manifest_sha256']!=source['manifest_sha256']):raise ValueError('old identity changed')
 safe(record['target'],plan['new_root']);_safe_root(record['store_root'])
 blocked=[Path(plan['output_root'])/Path(s['path']).parent for g in plan['groups'] for s in g['source_members']]
 blocked+=[Path(x) for x in plan['forbidden_source_roots']]
 blocked=[Path(os.path.realpath(p)) for p in blocked]
 attempts=[]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.path.realpath(os.fsdecode(args[0])))
  if any(p==b or b in p.parents for b in blocked):
   attempts.append(str(p));raise PermissionError('old package/source access forbidden')
 sys.addaudithook(guard)
 for path in blocked:
  for leaf in ['field.pt','manifest.json']:
   try:
    with (path/leaf).open('rb'):pass
   except PermissionError:pass
   else:raise ValueError('active open-denial negative test failed')
 negative_tests=len(attempts);attempts.clear()
 fixture=torch.load(record['probe_fixture'],map_location='cpu',weights_only=True)
 if sha256(record['probe_fixture'])!=record['probe_fixture_sha256']:raise ValueError('fixed probe changed')
 before=fp(Path(record['target'])/'manifest.json')
 if sha256(Path(record['target'])/'manifest.json')!=record['target_manifest_sha256']:raise ValueError('target manifest changed')
 field,manifest=load_field_versioned(record['target'],store_root=record['store_root'])
 contract=spec_contract(record['original_spec'],manifest['spec'])
 if contract!=record['spec_migration'] or manifest['provenance']['spec_migration']!=contract:raise ValueError('spec migration identity differs')
 if manifest['state_identity']!=record['state_identity'] or field.export_spec()!=contract['effective_spec'] or manifest['tensors']!=record['tensor_descriptors']:
  raise ValueError('cold scientific identity differs')
 actual=probe_results(field,fixture,group_id)
 if actual!=record['expected_probes'] or attempts:raise ValueError('cold query mismatch or forbidden source attempted')
 if fp(Path(record['target'])/'manifest.json')!=before:raise ValueError('target changed')
 result=dict(status='validated',key=key,publish_record_sha256=record_sha,migration_sha256=migration_sha256,
  target_manifest_sha256=record['target_manifest_sha256'],state_identity=record['state_identity'],source_open_negative_tests=negative_tests,forbidden_runtime_attempts=attempts,
  exact_state_dtype_spec=True,exact_cpu_probe_outputs=True,probe_count=len(actual),
  application_blob_read_bytes=manifest['tensor_bytes'],torch_threads=torch.get_num_threads(),
  source_free_scope='All five old package directories including manifests and all declared source roots denied',
  probe_scope='fixed synthetic packets; no real encoder/GPU/benchmark execution')
 publish(group_root(plan,group_id)/('cold_'+index+'.json'),result);print(json.dumps(result))

def finalize(plan):
 check_input_budget(plan)
 root=Path(plan['new_root']);entries=[];union={};new={};read_bytes=plan.get('prior_attempt_application_read_bytes',0);write_bytes=0
 for group in plan['groups']:
  checked_sources(plan,group)
  migration=read(group_root(plan,group['group_id'])/'migration_complete.json')
  read_bytes+=migration['application_read_bytes'];write_bytes+=migration['application_write_bytes']
  for index,source in enumerate(group['source_members']):
   record=read(group_root(plan,group['group_id'])/('publish_'+str(index)+'.json'))
   cold_record=read(group_root(plan,group['group_id'])/('cold_'+str(index)+'.json'))
   expected_key=group['group_id']+':'+str(index)
   if (cold_record['status']!='validated' or cold_record['key']!=expected_key
       or cold_record['publish_record_sha256']!=sha256(group_root(plan,group['group_id'])/('publish_'+str(index)+'.json'))
       or cold_record['migration_sha256']!=sha256(group_root(plan,group['group_id'])/'migration_complete.json')
       or cold_record['state_identity']!=record['state_identity']
       or cold_record['target_manifest_sha256']!=record['target_manifest_sha256']):raise ValueError('cold acceptance identity differs')
   read_bytes+=cold_record['application_blob_read_bytes']
   _,manifest=_read_manifest(record['target'])
   if sha256(Path(record['target'])/'manifest.json')!=record['target_manifest_sha256']:raise ValueError('target changed')
   union.update({d['sha256']:d for d in manifest['tensors'].values()})
   new.update({x['path']:x for x in record['new_blobs']})
   entries.append({k:record[k] for k in ['group_id','index','old_path','old_manifest','old_manifest_sha256','old_field_sha256','target','target_manifest_sha256','state_identity','store_root','original_spec','spec_migration']})
 for h,d in union.items():
  p=Path(plan['store_root'])/(h+'.bin')
  if not p.is_file() or p.stat().st_size!=d['bytes']:raise ValueError('closure missing')
 overlay=dict(schema='radio_gs.function_variant_consumer_overlay.v1',plan_identity=digest(plan),code=plan.get('code',[]),entries=entries,consumers=plan['consumers'],
  metadata_bindings=plan.get('metadata_bindings',[]),inherited_acceptance=plan.get('inherited_acceptance',{}),
  interpretation='Explicit old path + archive SHA + manifest SHA to new package; not automatic old field.pt interception',
  history='Original design/config/cache untouched; mixture ScanNet never completed, no new benchmark result asserted')
 publish(root/'consumer_overlay.json',overlay)
 usage=current_new_usage(plan);check_cap(plan,usage)
 if read_bytes+write_bytes>plan['max_application_io_bytes']:raise ValueError('application IO budget exceeded')
 result=dict(status='validated_non_destructive_pilot',source_files=5,groups=2,source_originals_retained=True,
  old_logical_bytes=sum(s['bytes'] for g in plan['groups'] for s in g['source_members']),
  old_allocated_bytes=sum(s['allocated'] for g in plan['groups'] for s in g['source_members']),
  complete_union_blob_bytes=sum(d['bytes'] for d in union.values()),
  complete_union_allocated_bytes=sum(fp(Path(plan['store_root'])/(h+'.bin'))['allocated'] for h in union),
  new_unique_blob_bytes=sum(x['stat']['bytes'] for x in new.values()),
  new_unique_blob_allocated_bytes=sum(fp(x['path'])['allocated'] for x in new.values()),
  pilot_artifact_logical_bytes=sum(x['bytes'] for x in usage['metadata'].values()),
  pilot_artifact_allocated_bytes=sum(x['allocated'] for x in usage['metadata'].values()),
  total_new_logical_bytes=usage['logical'],total_new_allocated_bytes=usage['allocated'],
  application_read_bytes=read_bytes,application_write_bytes=write_bytes,
  prior_attempt_application_read_bytes=plan.get('prior_attempt_application_read_bytes',0),
  accounting='Operation-size application byte estimates, not backend NFS measurements; small metadata IO separately bounded16MiB',
  allocation_observation_utc=usage['observed_utc'],allocation_changes=usage['allocation_observations'],
  observed_df=df(plan['output_root']),freed_now_bytes=0,overlay_sha256=sha256(root/'consumer_overlay.json'),
  retirement='Not authorized; all original payloads retained')
 result['artifact_accounting_excludes_this_summary']=True
 publish(root/'final_summary.json',result)
 final_usage=current_new_usage(plan);check_cap(plan,final_usage)
 result.update(final_all_artifacts_logical_bytes=sum(x['bytes'] for x in final_usage['metadata'].values()),
  final_all_artifacts_allocated_bytes=sum(x['allocated'] for x in final_usage['metadata'].values()),
  final_total_new_logical_bytes=final_usage['logical'],final_total_new_allocated_bytes=final_usage['allocated'])
 print(json.dumps(result))
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
 p.add_argument('stage',choices=['migrate','cold','finalize']);p.add_argument('--group');p.add_argument('--key');p.add_argument('--migration-sha256');args=p.parse_args()
 if sha256(args.plan)!=args.plan_sha256:raise ValueError('plan identity differs')
 plan=read(args.plan)
 for item in plan['code']:
  if sha256(item['path'])!=item['sha256']:raise ValueError('frozen source code changed')
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 _safe_root(plan['store_root']);safe(plan['new_root'],plan['output_root'])
 if args.stage=='migrate':
  group=next(g for g in plan['groups'] if g['group_id']==args.group);migrate(plan,group)
 elif args.stage=='cold':cold(plan,args.key,args.migration_sha256)
 else:finalize(plan)
if __name__=='__main__':main()
