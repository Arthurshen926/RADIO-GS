"""Retire only the eleven explicitly approved archived field.pt payloads."""
import os,sys,json,hashlib,datetime
from pathlib import Path
sys.path.insert(0,'/root/RADIO-GS-query-field')
from scripts.audits import migrate_duplicate_function_b01 as b
REQUEST=Path('/root/RADIO-GS-query-field/docs/experiments/2026-10-13-duplicate-exact11-retirement-request.json')
REQUEST_SHA='ea6b8ae9847e04c16ab04302ef04476b6e2c49c272bfcfb0360d14ae160a08e0'
OUT=Path('/mnt/pool/sqy/results/RADIO-GS/output')
DEST=OUT/'storage_retirements_20261013/exact11_shared_fields'
STORE=OUT/'storage_v2_pilot_20261012/figurines_18views/blobs'
PILOT=OUT/'storage_v2_pilots_20261013/duplicate_functions_G01_G14_attempt2'
BATCH=OUT/'storage_v2_batches_20261013/duplicate_functions_B01'
def expected_paths():
 answer=set()
 for study,arm in [('soft_paired_functions_v1','soft_paired_function'),('mixture_functions_v1','mixture_function'),('confidence_mixture_functions_v1','confidence_mixture_function')]:
  for scene in ['scannet_scene0062_00','lerf_teatime','lerf_waldo_kitchen']:
   answer.add('query_field_20261009/'+study+'/fits/'+arm+'/'+scene+'/field/field.pt')
 for rule in ['mean','measure']:answer.add('query_field_20261009/native_mode_functions_v1/fits/scene0535_00/'+rule+'/field/field.pt')
 return answer
def metadata(paths):
 result={};total=0
 for path in sorted(set(map(str,paths))):
  p=Path(path);b.base.safe(p,Path('/'))
  st=b.base.fp(p);total+=st['bytes']
  if st['bytes']>2*1024**2 or total>24*1024**2:raise ValueError('metadata bound exceeded')
  result[path]=dict(bytes=st['bytes'],sha256=b.sha256(p))
 return result
def assert_metadata(records):
 for path,r in records.items():
  if b.base.fp(path)['bytes']!=r['bytes'] or b.sha256(path)!=r['sha256']:raise ValueError('retained metadata changed: '+path)
def closure(request):
 expected={};stamps={}
 for name,digest in [('G01','cbdab43f7b2f15f0163f9511e6678462e9a8cb48f7ff49090a07e5b51b4ea0f9'),('G14_attempt2','fe5e07542f648f354b5c7744354cd0da2d2628679ab9b82298c3ed87a0f5b380')]:
  folder=PILOT/name;mp=folder/'migration_complete.json'
  if b.sha256(mp)!=digest:raise ValueError('accepted pilot migration receipt changed')
  migration=b.read(mp)
  for index,sha in migration['record_sha256'].items():
   p=folder/('publish_'+index+'.json')
   if b.sha256(p)!=sha:raise ValueError('accepted pilot publication receipt changed')
   for row in b.read(p)['new_blobs']:expected[row['path']]=row['stat']
 for gid,digest in [('G49','6b785794aa192c9a09d1e1ccd231e7dfccb3984a29c1e76964dcfa5ea972d977'),('G48','117265edb5071c6eb234910e725362e97ecb337cc0e54658c0cab33f10e7a40f')]:
  cp=BATCH/gid/'stage_cold_result.json'
  if b.sha256(cp)!=digest:raise ValueError('accepted B01 cold receipt changed')
  cold=b.read(cp)
  if cold['status']!='passed':raise ValueError('cold acceptance missing')
  for h,st in cold['validated_blob_fingerprints'].items():stamps[str(STORE/(h+'.bin'))]=st
 found={}
 for row in request['entries']:
  target=Path(row['new_package']);b.base.safe(target,OUT)
  if b.sha256(target/'manifest.json')!=row['new_manifest_sha256']:raise ValueError('new manifest changed')
  _,m=b._read_manifest(target)
  if m['state_identity']!=row['state_identity']:raise ValueError('state identity differs')
  for desc in m['tensors'].values():
   p=b._blob_path(STORE,desc['sha256']);st=b.base.fp(p)
   if st['bytes']!=desc['bytes']:raise ValueError('blob size differs')
   if str(p) in stamps:
    if list(b.VerifiedBlobStage._stamp(p))!=stamps[str(p)]:raise ValueError('cold blob fingerprint changed')
   elif str(p) in expected:
    if b.base.content_fp(st)!=b.base.content_fp(expected[str(p)]):raise ValueError('pilot blob fingerprint changed')
   else:raise ValueError('blob lacks accepted identity receipt')
   found[str(p)]=st
 return found
def main():
 if b.sha256(REQUEST)!=REQUEST_SHA:raise ValueError('exact request changed')
 request=b.read(REQUEST);rows=request['entries']
 for receipt in request['source_summary_receipts']:
  if b.sha256(receipt['path'])!=receipt['sha256']:raise ValueError('accepted summary changed')
 if len(rows)!=11 or {r['output_relative_path'] for r in rows}!=expected_paths():raise ValueError('scope differs')
 files=[]
 for row in rows:
  p=OUT/row['output_relative_path'];b.base.safe(p,OUT)
  if str(p)!=row['absolute_path'] or b.base.fp(p)!=row['stat'] or row['stat']['nlink']!=1:raise ValueError('source fingerprint/path changed')
  mp=p.with_name('manifest.json')
  if b.sha256(mp)!=row['manifest_sha256']:raise ValueError('old manifest changed')
  m=b.read(mp)
  if m['files']['field.pt']!={'bytes':row['stat']['bytes'],'sha256':row['archive_sha256']}:raise ValueError('old archive receipt changed')
  files += [x for x in p.parent.iterdir() if x!=p and x.is_file()]
 for root in [PILOT,BATCH]:
  files += [x for x in root.rglob('*') if x.is_file()]
 for planpath in [
 '/root/RADIO-GS-query-field/docs/experiments/2026-10-13-duplicate-function-pilot-plan-v7.json',
 '/root/RADIO-GS-query-field/docs/experiments/2026-10-13-duplicate-function-B01-plan-v2.json']:
  plan=b.read(planpath);files.append(Path(planpath))
  for entry in plan['metadata_bindings']:
   p=Path(entry['path'])
   if b.sha256(p)!=entry['sha256']:raise ValueError('historical consumer record changed')
   files.append(p)
 files.append(REQUEST)
 kept=metadata(files);blobs=closure(request)
 process=b.base.no_refs([r['stat'] for r in rows])
 b.base.safe(DEST,OUT);DEST.mkdir(parents=True,exist_ok=False)
 b.publish(DEST/'retained_metadata.json',kept);b.publish(DEST/'closure_fingerprints.json',blobs)
 before=b.base.df(OUT)
 b.publish(DEST/'preflight.json',dict(time=b.base.now(),request_sha256=REQUEST_SHA,source_rows=rows,df=before,
  process_check=process,retained_metadata_count=len(kept),closure_blob_count=len(blobs),
  verification='Fresh exact source stat/manifest against full-SHA acceptance receipts; accepted blob identities and sizes rechecked, no redundant large payload hash',
  originals_to_remove=11,only_old_payloads=True))
 removed=[];error=None
 with (DEST/'journal.jsonl').open('x') as journal:
  def log(value):journal.write(json.dumps(value)+'\n');journal.flush();os.fsync(journal.fileno())
  try:
   assert_metadata(kept);closure(request)
   for i,row in enumerate(rows):
    p=Path(row['absolute_path']);b.base.safe(p,OUT)
    if b.base.fp(p)!=row['stat']:raise ValueError('source changed immediately before deletion')
    b.base.no_refs([row['stat']])
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
     st=os.stat(p.name,dir_fd=fd,follow_symlinks=False)
     if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_nlink)!=(row['stat']['dev'],row['stat']['ino'],row['stat']['bytes'],row['stat']['mtime_ns'],row['stat']['ctime_ns'],1):raise ValueError('bound parent entry changed')
     log(dict(event='unlink_intent',index=i,time=b.base.now(),path=str(p),archive_sha256=row['archive_sha256'],stat=row['stat']))
     os.unlink(p.name,dir_fd=fd);os.fsync(fd)
    finally:os.close(fd)
    if os.path.lexists(p):raise ValueError('payload still exists')
    removed.append(row);log(dict(event='unlinked',index=i,path=str(p),time=b.base.now(),bytes=row['stat']['bytes'],allocated=row['stat']['allocated']))
   assert_metadata(kept);closure(request)
  except BaseException as exc:
   error=repr(exc);log(dict(event='failure',time=b.base.now(),error=error,removed_count=len(removed)))
 summary=dict(status='deleted_exact11' if error is None else 'partial_failure',error=error,finished_utc=b.base.now(),
  request_sha256=REQUEST_SHA,deleted_count=len(removed),deleted_logical_bytes=sum(x['stat']['bytes'] for x in removed),
  deleted_allocated_bytes=sum(x['stat']['allocated'] for x in removed),all_11_absent=all(not os.path.lexists(x['absolute_path']) for x in rows),
  retained_metadata_sha256=b.sha256(DEST/'retained_metadata.json'),closure_fingerprints_sha256=b.sha256(DEST/'closure_fingerprints.json'),
  journal_sha256=b.sha256(DEST/'journal.jsonl'),df_before=before,df_after=b.base.df(OUT),
  scope='Only exact 11 approved old field.pt. No manifests, configs, shared blobs, exports or other candidates removed.',
  df_note='Shared filesystem samples, not exclusive attribution; st_blocks and logical bytes reported separately.')
 b.publish(DEST/'summary.json',summary);print(json.dumps(summary))
 if error:raise RuntimeError(error)
if __name__=='__main__':main()
