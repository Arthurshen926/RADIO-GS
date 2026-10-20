#!/usr/bin/env python3
"""Only approved 181 single-link historical leaves; no directory/symlink deletion."""
import argparse,datetime,gzip,hashlib,json,os,stat,sys,time,types
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized execution forbidden')
REPO=Path('/root/RADIO-GS-query-field');DOC=REPO/'docs/experiments';BASE=Path('/mnt/pool/sqy/results/RADIO-GS/output')
PLAN=DOC/'2026-10-15-radio-best-core-retirement-proposal-0403';EVIDENCE=DOC/'2026-10-15-radio-best-core-retirement-proposal-0358/evidence_and_record_guard.json'
RUN=DOC/'2026-10-15-radio-181-retirement-execution-0607'
MANIFEST=PLAN/'frozen_181_leaves.json.gz';SHA='5afded247f65ac2259c5a80aaee2f2422a6eb0485487ec47ad66b6dcdc9890d6'
hp=REPO/'scripts/audits/retire_radio_exact_h12_20261015_v2.py';hr=hp.read_bytes()
assert hashlib.sha256(hr).hexdigest()=='0d45c4b4bc697b0df916c142abb90e9df6b58e7cd210052785e852dba89f4e80'
H=types.ModuleType('h12_guard');exec(compile(hr,str(hp),'exec'),H.__dict__);G=H.G;F=G.FIELDS

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def exists(p):
 try:os.lstat(p);return True
 except FileNotFoundError:return False
G.os.path.lexists=exists

def root_for(p):
 q=Path(p).relative_to(BASE).parts
 return str(BASE/q[0]/q[1]) if q[0]=='radio_gs' else str(Path(p).parent)
G.root_for=root_for

def save(p,d):return H.save(p,d)
def zsave(p,d):return H.zsave(p,d)
def zread(p):return H.zread(p)
def load():
 r=MANIFEST.read_bytes();assert hashlib.sha256(r).hexdigest()==SHA
 xs=json.loads(gzip.decompress(r))['leaves'];assert len(xs)==181 and len({x['path'] for x in xs})==len({(x['dev'],x['ino']) for x in xs})==181
 for x in xs:
  p=Path(x['path']);assert p.is_absolute() and '..' not in p.parts and str(p).startswith(str(BASE)+'/')
  assert stat.S_ISREG(x['mode']) and x['nlink']==1 and p.suffix=='.pth'
 return xs

def records():
 assert G.sha(EVIDENCE)=='10f0cbd4a64881b49ea98d67fdcf549626675e66c59fe467e7b2572777774d93'
 d=json.loads(EVIDENCE.read_text());assert len(d['records'])==686
 assert set(x['path'] for x in load())<=set(d['leaf_evidence'])
 return d['records']

def held348():
 p=DOC/'2026-10-15-radio-h12-retirement-proposal-0352/frozen_leaves.json.gz'
 assert G.sha(p)=='e7b9ace9e0bce68838a3fd327907db2ca5b10f44b6498814867e7883f1c06cae'
 xs=[x for x in zread(p)['leaves'] if x['nlink']==2];assert len(xs)==348
 return {x['path']:x for x in xs}

def checks(saved,targets=True):
 bad=[];hashed=0
 for p,s in saved.items():
  try:
   if s.get('missing'):
    if exists(p):bad.append({'path':p,'reason':'known missing became present'})
    continue
   n=G.rec(p)
   if any(n[k]!=s[k] for k in F):bad.append({'path':p,'reason':'identity changed'})
   if 'link' in s and n.get('link')!=s['link']:bad.append({'path':p,'reason':'symlink changed'})
   if 'sha256' in s:
    hashed+=1
    if G.sha(p)!=s['sha256']:bad.append({'path':p,'reason':'SHA changed'})
   if targets and 'target' in s:
    q=G.rec(p,True)
    if any(q[k]!=s['target'][k] for k in F):bad.append({'path':p,'reason':'resolved target changed'})
   if targets and 'realpath' in s and os.path.realpath(p)!=s['realpath']:bad.append({'path':p,'reason':'resolution changed'})
   if targets and s.get('target_missing'):
    try:os.stat(p)
    except FileNotFoundError:pass
    else:bad.append({'path':p,'reason':'known broken target became present'})
  except OSError as e:bad.append({'path':p,'reason':repr(e)})
 return {'entries':len(saved),'hashed_records':hashed,'errors':bad}

def protected_intersections(xs,prot):
 paths={x['path'] for x in xs};inos={(x['dev'],x['ino']):x['path'] for x in xs};hits={}
 for p,s in prot.items():
  if s.get('missing'):continue
  real=os.path.realpath(p);t=s.get('target',s)
  if real in paths:hits[real]='current protected path or symlink target'
  key=(t['dev'],t['ino'])
  if key in inos:hits[inos[key]]='current protected inode'
  if stat.S_ISDIR(t['mode']) or s.get('target_directory_dev_ino'):
   for x in xs:
    if x['path'].startswith(real.rstrip('/')+'/'):hits[x['path']]='current protected directory subtree'
 return hits

def retained_snapshot(xs):
 saved={};errors=[];leaves={x['path'] for x in xs}
 for root in sorted({root_for(x['path']) for x in xs}):
  for parent,dirs,files in os.walk(root,followlinks=False,onerror=errors.append):
   for name in dirs+files:
    p=str(Path(parent)/name)
    if p in leaves:continue
    try:
     n=G.rec(p)
     if stat.S_ISDIR(n['mode']):continue
     if stat.S_ISREG(n['mode']) and Path(p).suffix.lower() in G.TEXT and n['size']<=2097152:n['sha256']=G.sha(p)
     saved[p]=n
    except OSError as e:errors.append(e)
 assert not errors,[repr(e) for e in errors[:10]]
 for p,s in saved.items():
  if stat.S_ISLNK(s['mode']):
   s['realpath']=os.path.realpath(p)
   try:
    t=G.rec(p,True)
    if stat.S_ISDIR(t['mode']):s['target_directory_dev_ino']=[t['dev'],t['ino']]
    else:s['target']=t
   except FileNotFoundError:s['target_missing']=True
 return saved

def retention_check(saved):
 out=checks(saved)
 for p,s in saved.items():
  if s.get('target_directory_dev_ino'):
   try:
    q=os.stat(p)
    if [q.st_dev,q.st_ino]!=s['target_directory_dev_ino']:out['errors'].append({'path':p,'reason':'retained directory alias changed'})
   except OSError as e:out['errors'].append({'path':p,'reason':repr(e)})
 return out

def all_checks(ret):
 return {'protected':checks(H.protected()),'records':checks(records()),'held348':checks(held348()),'retained':retention_check(ret)}

def preflight():
 xs=load();assert not RUN.exists();RUN.mkdir()
 p=H.protected();pc=checks(p);ec=checks(records());hc=checks(held348())
 assert not any(x['errors'] for x in [pc,ec,hc]),{'protected':pc,'records':ec,'held':hc}
 skip=protected_intersections(xs,p)
 for x in xs:
  try:
   if not G.match(G.rec(x['path']),x):skip[x['path']]='source identity changed'
  except OSError as e:skip[x['path']]=repr(e)
 ret=retained_snapshot(xs)
 # Any retained symlink alias to a candidate, including directory subtree, is held.
 skip.update(protected_intersections(xs,ret))
 rc=retention_check(ret);assert not rc['errors'],rc['errors'][:10]
 scan=G.proc_scan(xs);assert not scan['errors'],scan['errors'][:10]
 zsave(RUN/'retained_preflight.json.gz',ret)
 out={'utc':utc(),'manifest_sha256':SHA,'leaf_count':181,'roots':sorted({root_for(x['path']) for x in xs}),'skip':skip,'checks':{'protected':pc,'records':ec,'held348':hc,'retained':rc},'eden_scan':scan,'df_before':G.df(),'scope':'selected current runtime and chosen coldbuild closure; visible Eden/pakka namespaces only; no SAM execution or 348 held unlink'}
 save(RUN/'preflight.json',out);print(json.dumps({k:v for k,v in out.items() if k!='roots'}),flush=True)

def execute():
 xs=load();pre=json.loads((RUN/'preflight.json').read_text());pakka=json.loads((RUN/'pakka_writer_check.json').read_text())
 assert pakka['manifest_sha256']==SHA and pakka['candidate_count']==181 and pakka['inode_count']==181
 assert pakka['ready_for_execution'] and not pakka['matches'] and not pakka['permission_errors'] and time.time()<pakka['valid_until_epoch']
 ret=zread(RUN/'retained_preflight.json.gz');fresh_ret=retained_snapshot(xs);assert set(fresh_ret)==set(ret),'retained traversal changed';save(RUN/'retained_traversal_recheck.json',{'utc':utc(),'entries':len(fresh_ret),'onerror_checked':True,'same_path_set':True});ck=all_checks(ret);assert not any(x['errors'] for x in ck.values()),ck
 skip=dict(pre['skip']);skip.update(protected_intersections(xs,H.protected()));skip.update(protected_intersections(xs,ret))
 scan=G.proc_scan(xs);save(RUN/'eden_immediate_process_scan.json',scan);assert not scan['errors'] and not scan['vanished'],'Eden immediate scan incomplete; refresh before continuation'
 blocked=set(pakka.get('blocked_roots',[]))|{r for m in scan['matches'] for r in m['roots']}
 assert not any(m['candidate_inode'] and not m['roots'] for m in scan['matches']),'unmapped candidate inode in process'
 for x in xs:
  if root_for(x['path']) in blocked:skip[x['path']]='visible active experiment root'
 assert not exists(RUN/'leaf_receipts.jsonl') and not exists(RUN/'unlink_intents.jsonl'),'never blindly rerun'
 started=time.monotonic();deleted=skipped=allocated=0
 with (RUN/'leaf_receipts.jsonl').open('x') as rf,(RUN/'unlink_intents.jsonl').open('x') as inf:
  def emit(f,d):f.write(json.dumps(d,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
  for i,x in enumerate(xs):
   if time.monotonic()-started>480 or time.time()+3>=pakka['valid_until_epoch']:raise RuntimeError('bounded stop; reconcile before continuation')
   if x['path'] in skip:emit(rf,{'utc':utc(),'path':x['path'],'status':'SKIPPED','reason':skip[x['path']]});skipped+=1;continue
   fd=None
   try:
    fd,name=G.secure_parent(x['path']);s=os.stat(name,dir_fd=fd,follow_symlinks=False)
    def ident(q):return {**{k:getattr(q,'st_'+k) for k in F},'allocated':q.st_blocks*512}
    if not G.match(ident(s),x):emit(rf,{'utc':utc(),'path':x['path'],'status':'SKIPPED','reason':'source identity changed at final no-follow check'});skipped+=1;continue
    emit(inf,{'utc':utc(),'path':x['path'],'status':'UNLINK_INTENT','original_identity':x})
    s=os.stat(name,dir_fd=fd,follow_symlinks=False)
    if not G.match(ident(s),x):emit(rf,{'utc':utc(),'path':x['path'],'status':'SKIPPED','reason':'post-intent source identity changed'});skipped+=1;continue
    if time.time()>=pakka['valid_until_epoch'] or time.monotonic()-started>480:raise RuntimeError('writer window expired; reconcile before continuation')
    os.unlink(name,dir_fd=fd)
    emit(rf,{'utc':utc(),'path':x['path'],'status':'UNLINKED','dev':x['dev'],'ino':x['ino'],'original_allocated':x['allocated']});deleted+=1;allocated+=x['allocated']
   except BaseException as e:
    save(RUN/'execution_error.json',{'utc':utc(),'path':x['path'],'error':repr(e),'deleted_before_error':deleted});raise
   finally:
    if fd is not None:os.close(fd)
   if (i+1)%50==0:print('PROGRESS',i+1,deleted,skipped,flush=True)
 save(RUN/'execution_result.json',{'utc':utc(),'deleted':deleted,'skipped':skipped,'closed_last_link_allocated_bytes':allocated,'seconds':time.monotonic()-started,'df_after_unlink':G.df()});print('EXECUTED',deleted,skipped,allocated,flush=True)

def post():
 xs=load();rows=[json.loads(l) for l in (RUN/'leaf_receipts.jsonl').read_text().splitlines()];by={r['path']:r for r in rows};assert len(by)==len(rows) and set(by)<={x['path'] for x in xs}
 errors=[];deleted=skipped=allocated=0
 for x in xs:
  r=by.get(x['path']);ex=exists(x['path'])
  if r and r['status']=='UNLINKED' and not ex:deleted+=1;allocated+=x['allocated']
  elif r and r['status']=='SKIPPED' and ex:skipped+=1
  else:errors.append({'path':x['path'],'receipt':r,'exists':ex})
 ck=all_checks(zread(RUN/'retained_preflight.json.gz'));pre=json.loads((RUN/'preflight.json').read_text());after=G.df()
 out={'status':'VERIFIED' if not errors and not any(v['errors'] for v in ck.values()) else 'POSTCHECK_ISSUE','completed_utc':utc(),'manifest_sha256':SHA,'approved_leaves':181,'verified_deleted':deleted,'verified_skipped':skipped,'closed_last_link_allocated_bytes':allocated,'leaf_errors':errors,'checks':ck,'df_before':pre['df_before'],'df_after':after,'df_available_delta':after['available']-pre['df_before']['available'],'limits':'Shared-pool df can include concurrent G4 and other writers. Current selected runtime/chosen coldbuild plus visible Eden/pakka namespaces guarded, not other NFS clients or future writers. Historical weights/resume and old build validators retired as approved; no bitwise retrain guarantee.','sam_pilot':'PAUSED_NOT_STARTED','held348_deleted':False}
 save(RUN/'completion.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','execute','post']);a=p.parse_args();{'preflight':preflight,'execute':execute,'post':post}[a.phase]()
