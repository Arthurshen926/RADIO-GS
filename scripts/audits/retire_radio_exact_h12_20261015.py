#!/usr/bin/env python3
"""Only approved H12 exact leaves. Audit/preflight and closed inode-group unlink."""
import argparse,collections,datetime,gzip,hashlib,importlib.util,json,os,stat,sys,time
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized execution forbidden')
REPO=Path('/root/RADIO-GS-query-field');DOC=REPO/'docs/experiments';BASE=Path('/mnt/pool/sqy/results/RADIO-GS/output')
PLAN=DOC/'2026-10-15-radio-h12-retirement-proposal-0352'
RUN=DOC/'2026-10-15-radio-h12-retirement-execution-0459'
MANIFEST=PLAN/'frozen_leaves.json.gz';SHA='e7b9ace9e0bce68838a3fd327907db2ca5b10f44b6498814867e7883f1c06cae'
spec=importlib.util.spec_from_file_location('guard',REPO/'scripts/audits/retire_radio_exact762_20261015_v4.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
G.root_for=lambda p:str(BASE/Path(p).relative_to(BASE).parts[0])
F=G.FIELDS;FIXED=('dev','ino','mode','size','mtime_ns')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
def zsave(p,d):
 with Path(p).open('xb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(d,separators=(',',':')).encode())
def zread(p):return json.load(gzip.open(p,'rt'))
def load():
 raw=MANIFEST.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
 xs=json.loads(gzip.decompress(raw))['leaves'];assert len(xs)==18544 and len({x['path'] for x in xs})==18544
 groups=collections.defaultdict(list)
 for x in xs:
  p=Path(x['path']);assert p.is_absolute() and '..' not in p.parts and p.is_relative_to(BASE) and len(p.relative_to(BASE).parts)>1
  assert stat.S_ISREG(x['mode']) and x['nlink'] in (1,2)
  groups[(x['dev'],x['ino'])].append(x)
 assert len(groups)==18370
 for k,xs2 in groups.items():assert len(xs2)==xs2[0]['nlink'] and all(all(x[k2]==xs2[0][k2] for k2 in F) for x in xs2)
 return xs,groups
def check_map(saved,hash_records=True,targets=True):
 bad=[];count=0
 for p,s in saved.items():
  try:
   if s.get('missing'):
    if os.path.lexists(p):bad.append({'path':p,'reason':'known missing became present'})
    continue
   n=G.rec(p)
   if any(n[k]!=s[k] for k in F):bad.append({'path':p,'reason':'identity changed'})
   if 'link' in s and n.get('link')!=s['link']:bad.append({'path':p,'reason':'symlink changed'})
   if hash_records and 'sha256' in s:
    count+=1
    if G.sha(p)!=s['sha256']:bad.append({'path':p,'reason':'record SHA changed'})
   if targets and 'target' in s:
    q=G.rec(p,True)
    if any(q[k]!=s['target'][k] for k in F):bad.append({'path':p,'reason':'protected resolved target changed'})
   if targets and 'realpath' in s and os.path.realpath(p)!=s['realpath']:bad.append({'path':p,'reason':'resolution changed'})
   if targets and s.get('target_missing') and os.path.exists(p):bad.append({'path':p,'reason':'known broken target changed'})
  except OSError as e:bad.append({'path':p,'reason':str(e)})
 return {'entries':len(saved),'hashed_records':count,'errors':bad}
def protected():
 p=DOC/'2026-10-15-radio-762-retirement-execution-0340/protected_preflight.json.gz'
 assert G.sha(p)=='c737ebdf76d2117ed762e2ee282f1b640163985f7febc9a404ac580adef9b0d0'
 return zread(p)
def preflight():
 xs,groups=load();assert not (RUN/'preflight.json').exists()
 prot=protected();pc=check_map(prot);assert not pc['errors'],pc['errors'][:20]
 paths={os.path.realpath(p) for p,x in prot.items() if not x.get('missing')};inos={(x['dev'],x['ino']) for x in prot.values() if not x.get('missing')}
 for x in prot.values():
  if x.get('target'):inos.add((x['target']['dev'],x['target']['ino']))
 holds={}
 for key,group in groups.items():
  for x in group:
   p=x['path']
   try:
    n=G.rec(p)
    if not G.match(n,x):holds[str(key)]='candidate identity changed'
    elif os.path.realpath(p) in paths or key in inos:holds[str(key)]='current protected input'
   except OSError as e:holds[str(key)]=str(e)
 print('CURRENT_PROTECTED_AND_CANDIDATES_CHECKED',len(prot),len(xs),'held_groups',len(holds),flush=True)
 rp=PLAN/'retained_files.json.gz';assert G.sha(rp)=='f40d2809e7cc3af1c8678bed10dffef90b4b8107785634648f6d0bae5ef32088'
 ret=zread(rp);rc=check_map(ret,False,False)
 blocked_roots={G.root_for(e['path']) for e in rc['errors']}
 records=json.loads((PLAN/'retained_result_evidence.json').read_text())['fresh_records']
 ec=check_map(records);blocked_roots|={G.root_for(e['path']) for e in ec['errors']}
 # Preserve all file symlink targets; two approved old directory aliases are intentionally retired below.
 candidate_paths={x['path'] for x in xs};syms={};allowed_dir_loss={}
 for p,s in ret.items():
  if not stat.S_ISLNK(s['mode']):continue
  target=os.path.realpath(p);n=G.rec(p);n['realpath']=target
  try:
   q=G.rec(p,True)
   if stat.S_ISDIR(q['mode']):
    count=sum(x.startswith(target.rstrip('/')+'/') for x in candidate_paths)
    if count:allowed_dir_loss[p]={'realpath':target,'approved_child_leaves':count}
    n['target_directory_dev_ino']=[q['dev'],q['ino']]
   else:
    n['target']=q
    if (q['dev'],q['ino']) in groups:holds[str((q['dev'],q['ino']))]='retained file symlink target'
  except FileNotFoundError:n['target_missing']=True
  syms[p]=n
 expected=json.loads((PLAN/'human_scope_and_symlink_check.json').read_text())['symlinks']['directory_aliases_containing_candidates']
 assert {p:(x['realpath'],x['approved_child_leaves']) for p,x in allowed_dir_loss.items()}=={x['path']:(x['real'],x['candidate_descendants']) for x in expected},'directory alias loss changed'
 for key,group in groups.items():
  if any(G.root_for(x['path']) in blocked_roots for x in group):holds[str(key)]='retained record changed in root'
 scan=G.proc_scan(xs)
 result={'manifest_sha256':SHA,'created_utc':utc(),'leaf_count':len(xs),'inode_count':len(groups),'held_groups':holds,'blocked_roots':sorted(blocked_roots),'protected_check':pc,'retained_check':rc,'evidence_check':ec,'eden_scan':scan,'df_before':G.df(),'approved_old_directory_alias_loss':allowed_dir_loss,'scope':'Only selected v10/T0 current runtime/chosen cold-build; visible process namespaces not all NFS clients; no 181 batch'}
 zsave(RUN/'retained_symlink_preflight.json.gz',syms);save(RUN/'preflight.json',result);print(json.dumps({k:v for k,v in result.items() if k!='eden_scan'}),flush=True);print('EDEN_SCAN',json.dumps(scan),flush=True)
def fixed_ok(s,x):return all(getattr(s,'st_'+k)==x[k] for k in FIXED) and s.st_blocks*512==x['allocated']
def execute():
 xs,groups=load();pre=json.loads((RUN/'preflight.json').read_text());pakka=json.loads((RUN/'pakka_writer_check.json').read_text())
 assert pakka['manifest_sha256']==SHA and pakka['candidate_count']==18544 and pakka['inode_count']==18370
 assert pakka['ready_for_execution'] and not pakka['matches'] and not pakka['permission_errors'] and time.time()<pakka['valid_until_epoch']
 pc=check_map(protected());assert not pc['errors'],pc['errors'][:20]
 ec=check_map(json.loads((PLAN/'retained_result_evidence.json').read_text())['fresh_records']);assert not ec['errors'],ec['errors'][:20]
 scan=G.proc_scan(xs);save(RUN/'eden_immediate_process_scan.json',scan);assert not scan['errors']
 blocked=set(pre['blocked_roots'])|set(pakka.get('blocked_roots',[]))|{r for m in scan['matches'] for r in m['roots']}
 assert not any(m['candidate_inode'] and not m['roots'] for m in scan['matches']),'unmapped candidate inode in process'
 assert not (RUN/'leaf_receipts.jsonl').exists() and not (RUN/'unlink_intents.jsonl').exists(),'never blindly rerun'
 started=time.monotonic();deleted=skipped=reclaimed=closed=0
 with (RUN/'leaf_receipts.jsonl').open('x') as rf,(RUN/'unlink_intents.jsonl').open('x') as inf:
  def emit(f,d):f.write(json.dumps(d,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
  for gi,(key,group) in enumerate(sorted(groups.items())):
   if time.monotonic()-started>480 or time.time()+3>=pakka['valid_until_epoch']:
    save(RUN/'bounded_stop.json',{'utc':utc(),'deleted':deleted,'skipped':skipped,'reason':'bound or writer snapshot expired','remaining_groups':len(groups)-gi});break
   why=pre['held_groups'].get(str(key))
   if any(G.root_for(x['path']) in blocked for x in group):why='visible active root or retained evidence changed'
   if why:
    for x in group:emit(rf,{'utc':utc(),'path':x['path'],'status':'SKIPPED','reason':why});skipped+=1
    continue
   handles=[];leaf_fd=None
   try:
    for x in group:
     fd,name=G.secure_parent(x['path']);handles.append((fd,name,x))
     q=os.stat(name,dir_fd=fd,follow_symlinks=False)
     if not fixed_ok(q,x) or q.st_nlink!=x['nlink'] or q.st_ctime_ns!=x['ctime_ns']:why='group pre-unlink identity changed'
    if why:
     for x in group:emit(rf,{'utc':utc(),'path':x['path'],'status':'SKIPPED','reason':why});skipped+=1
     continue
    leaf_fd=os.open(handles[0][1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=handles[0][0]);s=os.fstat(leaf_fd)
    assert fixed_ok(s,group[0]) and s.st_nlink==len(group) and s.st_ctime_ns==group[0]['ctime_ns']
    expected_ctime=s.st_ctime_ns
    for j,(fd,name,x) in enumerate(handles):
     expected_nlink=len(group)-j
     emit(inf,{'utc':utc(),'path':x['path'],'status':'UNLINK_INTENT','dev':key[0],'ino':key[1],'expected_nlink':expected_nlink,'expected_ctime_ns':expected_ctime,'original_identity':x})
     q=os.stat(name,dir_fd=fd,follow_symlinks=False);s=os.fstat(leaf_fd)
     if not (fixed_ok(q,x) and fixed_ok(s,x) and q.st_nlink==s.st_nlink==expected_nlink and q.st_ctime_ns==s.st_ctime_ns==expected_ctime):
      raise RuntimeError('final group identity race; stop and reconcile '+x['path'])
     os.unlink(name,dir_fd=fd);after=os.fstat(leaf_fd)
     if not fixed_ok(after,x) or after.st_nlink!=expected_nlink-1:
      emit(rf,{'utc':utc(),'path':x['path'],'status':'UNLINKED_POST_INODE_ANOMALY','dev':key[0],'ino':key[1]});raise RuntimeError('unexpected inode mutation after unlink')
     expected_ctime=after.st_ctime_ns
     emit(rf,{'utc':utc(),'path':x['path'],'status':'UNLINKED','dev':key[0],'ino':key[1],'remaining_nlink':after.st_nlink,'original_allocated':x['allocated'],'allocated_counted_on_this_unlink':x['allocated'] if after.st_nlink==0 else 0})
     deleted+=1
    closed+=1;reclaimed+=group[0]['allocated']
   except BaseException as e:
    save(RUN/'execution_error.json',{'utc':utc(),'group':list(key),'paths':[x['path'] for x in group],'error':repr(e),'deleted_before_error':deleted});raise
   finally:
    if leaf_fd is not None:os.close(leaf_fd)
    for fd,name,x in handles:os.close(fd)
   if (gi+1)%1000==0:print(json.dumps({'inode_groups':gi+1,'deleted':deleted,'skipped':skipped,'closed_allocated':reclaimed}),flush=True)
 save(RUN/'execution_result.json',{'completed_utc':utc(),'deleted':deleted,'skipped':skipped,'closed_inode_groups':closed,'closed_last_link_allocated_bytes':reclaimed,'seconds':time.monotonic()-started,'df_after_unlink':G.df()});print('EXECUTED',deleted,skipped,closed,reclaimed,flush=True)
def post():
 xs,groups=load();rows=[json.loads(l) for l in (RUN/'leaf_receipts.jsonl').read_text().splitlines()];assert len({r['path'] for r in rows})==len(rows)
 by={r['path']:r for r in rows};assert set(by)<={x['path'] for x in xs};errors=[];deleted=skipped=0;closed=allocated=0
 for x in xs:
  r=by.get(x['path']);exists=os.path.lexists(x['path'])
  if r and r['status']=='UNLINKED' and not exists:deleted+=1
  elif r and r['status']=='SKIPPED' and exists:skipped+=1
  else:errors.append({'path':x['path'],'receipt':r,'exists':exists})
 for key,group in groups.items():
  if all(by.get(x['path'],{}).get('status')=='UNLINKED' and not os.path.lexists(x['path']) for x in group):closed+=1;allocated+=group[0]['allocated']
 pc=check_map(protected());rc=check_map(zread(PLAN/'retained_files.json.gz'),False,False);ec=check_map(json.loads((PLAN/'retained_result_evidence.json').read_text())['fresh_records'])
 syms=zread(RUN/'retained_symlink_preflight.json.gz');sc=check_map(syms,False,True)
 for p,x in syms.items():
  if x.get('target_directory_dev_ino'):
   try:
    s=os.stat(p)
    if [s.st_dev,s.st_ino]!=x['target_directory_dev_ino']:sc['errors'].append({'path':p,'reason':'retained directory symlink target changed'})
   except OSError as e:sc['errors'].append({'path':p,'reason':str(e)})
 pre=json.loads((RUN/'preflight.json').read_text());after=G.df();status='VERIFIED' if not errors and not any(x['errors'] for x in [pc,rc,ec,sc]) else 'POSTCHECK_ISSUE'
 out={'status':status,'completed_utc':utc(),'manifest_sha256':SHA,'approved_leaves':18544,'verified_deleted':deleted,'verified_skipped':skipped,'closed_inode_groups':closed,'closed_last_link_allocated_bytes':allocated,'leaf_errors':errors,'protected_check':pc,'retained_check':rc,'record_check':ec,'retained_symlink_check':sc,'approved_old_directory_alias_loss':pre['approved_old_directory_alias_loss'],'df_before':pre['df_before'],'df_after':after,'df_available_delta':after['available']-pre['df_before']['available'],'limits':'df includes concurrent G4 deletion and all shared-pool writers. Only selected current runtime/chosen coldbuild and visible Eden/pakka process namespaces guarded. Historical exact weights/cache/replay retired as approved; known174 SAM loss not restored.','sam_pilot':'PAUSED_NOT_STARTED','other181_deleted':False}
 save(RUN/'completion.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','execute','post']);a=p.parse_args();{'preflight':preflight,'execute':execute,'post':post}[a.phase]()
