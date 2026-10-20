#!/usr/bin/env python3
"""Exact approved 404-leaf retirement with closed inode group transitions."""
import argparse,collections,datetime,gzip,hashlib,json,os,stat,sys,time,types
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized execution forbidden')
REPO=Path('/root/RADIO-GS-query-field');BASE=Path('/mnt/pool/sqy/results/RADIO-GS/output')
PLAN=REPO/'docs/experiments/2026-10-15-tmm-retention-review-1932'
RUN=REPO/'docs/experiments/2026-10-15-radio-404-retirement-execution-1946'
MAN=PLAN/'frozen_priority_404_leaves.json.gz'
SHA='315a31c0da67969232cd7d32798b80f10e2926c1224ca82b6d0bf2483f755b96'
support=REPO/'scripts/audits/retire_radio_exact762_20261015_v4.py'
raw=support.read_bytes();assert hashlib.sha256(raw).hexdigest()=='55dc7f26d75cb13aa5591ebe17fb37f93cc227c206774d9c839133fa8fc70bfd'
G=types.ModuleType('guard');exec(compile(raw,str(support),'exec'),G.__dict__)
G.root_for=lambda p:str(BASE/Path(p).relative_to(BASE).parts[0])
F=G.FIELDS+('allocated',);FIXED=('dev','ino','mode','size','mtime_ns','allocated')
utc=G.utc;save=G.write;zsave=G.zipwrite;sha=G.sha;rec=G.rec
def zr(p):return json.load(gzip.open(p,'rt'))
def exists(p):
 try:os.lstat(p);return True
 except FileNotFoundError:return False
def eq(a,b,fields=F):return all(a[k]==b[k] for k in fields)
def load():
 assert sha(MAN)==SHA
 xs=zr(MAN)['leaves'];assert len(xs)==len({x['path'] for x in xs})==404
 groups=collections.defaultdict(list)
 allowed={'depth_opacity_pipeline_20260918','lerf_complete_v2_20260918','system_complete_storage','system_native_field'}
 for x in xs:
  p=Path(x['path']);assert p.is_absolute() and '..' not in p.parts and str(p).startswith(str(BASE)+'/')
  assert p.relative_to(BASE).parts[0] in allowed and stat.S_ISREG(x['mode']) and x['nlink'] in (1,2)
  groups[(x['dev'],x['ino'])].append(x)
 assert len(groups)==230 and sum(len(g)==2 for g in groups.values())==174
 for group in groups.values():assert len(group)==group[0]['nlink'] and all(eq(g,group[0]) for g in group)
 return xs,groups
def source_bindings():
 p=PLAN/'priority404_summary.json';assert sha(p)=='65da7635ec6fd0587995b80a3d5dcf80950a1fe248b123c5fe7ccf86105aae37'
 b=json.loads(p.read_text())['verified_report_files']
 p=PLAN/'source_json_bindings.json';assert sha(p)=='b4d3ba2e89f29e580835eb0326e0737171884571ba7603346e06a115af5c4cd9'
 b.update(json.loads(p.read_text()))
 old=zr(REPO/'docs/experiments/2026-10-15-radio-762-retirement-execution-0340/protected_preflight.json.gz')
 for p,r in old.items():
  if r.get('sha256'):
   if p in b:assert b[p]['sha256']==r['sha256'],('old/current binding conflict',p)
   b[p]=r
 changed=str(REPO/'scripts/research/run_tmm_response_study.py')
 assert b[changed]['sha256']=='5ff67cd26acc732276be3adf1e667c8cdfbe4e26f6f6a5d5331ac0ef1d9ae7c6'
 b[changed]=dict(b[changed],sha256='402cbcc561cdc38ca611fb3dced9621f9a0a0ae1afa902835d7084743e683d28',prior_sha256=b[changed]['sha256'],review='Fresh full input/score caller review: scoring response_loader/arms parameterization; same system_v10 jobs, SYSTEM frozen providers and adapter input paths. Original failed preflight preserved.')
 return b
def check_bindings(b):
 return [dict(path=p,reason='binding SHA mismatch or absent') for p,r in b.items() if not Path(p).is_file() or sha(p)!=r['sha256']]
def protected(xs):
 p=PLAN/'protected_paths.json.gz';assert sha(p)=='2e6bd4e930379e23c1b30b9ff27e9184b621f4602b37ee9999dbce02743dc17d'
 paths=zr(p);paths['/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt']='actual model path'
 missing=set(zr(PLAN/'excluded_changed_missing.json.gz')['protected_missing'])
 cpaths={x['path'] for x in xs};inos={(x['dev'],x['ino']) for x in xs};snap={};hits=[];errors=[]
 for path in paths:
  try:
   n=rec(path,True);n['realpath']=os.path.realpath(path);snap[path]=n
   if n['realpath'] in cpaths or (n['dev'],n['ino']) in inos:hits.append(path)
   if stat.S_ISDIR(n['mode']) and any(x.startswith(n['realpath'].rstrip('/')+'/') for x in cpaths):hits.append(path)
  except FileNotFoundError:
   snap[path]={'missing':True}
   if path not in missing:errors.append(dict(path=path,reason='new missing current input'))
  except OSError as e:errors.append(dict(path=path,reason=repr(e)))
 return snap,hits,errors
def retained(xs):
 p=PLAN/'priority404_retained_snapshot.json.gz';assert sha(p)=='4e0883a32c412ab0d49ae16c4b15b9b035c0117ce8c2edd8615549964531cd1b'
 old={r['path']:r for r in zr(p)};assert len(old)==3598
 leaves={x['path'] for x in xs};paths=set();errors=[]
 for root in sorted({G.root_for(x['path']) for x in xs}):
  def walkerr(e):errors.append(dict(reason=repr(e)))
  for parent,dirs,files in os.walk(root,followlinks=False,onerror=walkerr):
   for name in dirs+files:
    p=str(Path(parent)/name)
    if p in leaves:continue
    n=rec(p)
    if stat.S_ISDIR(n['mode']):continue
    paths.add(p)
    if p not in old or not eq(n,old[p]):errors.append(dict(path=p,reason='retained identity changed'))
    if stat.S_ISLNK(n['mode']):
     rp=os.path.realpath(p)
     if n['link']!=old[p]['target'] or rp!=old[p]['realpath']:errors.append(dict(path=p,reason='retained symlink changed'))
     if any(c==rp or c.startswith(rp.rstrip('/')+'/') for c in leaves):errors.append(dict(path=p,reason='retained symlink reaches candidate'))
 if paths!=set(old):errors.append(dict(reason='retained path set changed',missing=sorted(set(old)-paths),added=sorted(paths-set(old))))
 return dict(count=len(paths),symlinks=sum(stat.S_ISLNK(r['mode']) for r in old.values()),errors=errors)
def scan(xs):
 roots=sorted({G.root_for(x['path']) for x in xs});inos={(x['dev'],x['ino']) for x in xs}
 d=dict(started_utc=utc(),scope='visible Eden /proc only; resolved closed fd/vanished pid races are separately recorded',pids=0,fds=0,maps_lines=0,matches=[],errors=[],vanished=0,resolved_races=[])
 def state(pid):
  try:
   t=Path('/proc/'+str(pid)+'/stat').read_text();v=t[t.rfind(')')+2:].split()
   return v[0],v[19]
  except FileNotFoundError:return None,None
 def err(pid,kind,p,e):
  st,_=state(pid)
  if isinstance(e,FileNotFoundError) and (st is None or (kind=='fd' and not exists(p)) or st=='Z'):
   d['resolved_races'].append(dict(pid=pid,kind=kind,path=str(p),state=st,reason='pid gone/zombie or exact enumerated fd closed'))
  else:d['errors'].append(dict(pid=pid,kind=kind,path=str(p),error=repr(e),state=st))
 def hit(pid,kind,value,p=None):
  text=G.canonical(value)
  matches=[r for r in roots if r in text or (kind=='cmdline' and ('output/'+Path(r).name in text or Path(r).name in text))]
  im=False
  if p:
   try:
    q=os.stat(p);im=(q.st_dev,q.st_ino) in inos
   except OSError as e:err(pid,kind,p,e)
  if matches or im:d['matches'].append(dict(pid=pid,kind=kind,roots=matches,candidate_inode=im,path=value[:1200]))
 pids=[int(x) for x in os.listdir('/proc') if x.isdigit() and int(x)!=os.getpid()]
 for pid in pids:
  d['pids']+=1;p=Path('/proc')/str(pid)
  for kind in ['cmdline','cwd','maps','fd']:
   try:
    if kind=='cmdline':hit(pid,kind,(p/kind).read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    elif kind=='cwd':hit(pid,kind,os.readlink(p/kind),str(p/kind))
    elif kind=='maps':
     for line in (p/kind).read_text(errors='replace').splitlines():
      d['maps_lines']+=1
      cols=line.split(None,5)
      if len(cols)==6 and '/' in cols[-1]:hit(pid,kind,cols[-1])
    else:
     for name in os.listdir(p/kind):
      f=p/kind/name
      try:
       target=os.readlink(f);d['fds']+=1
       hit(pid,kind,target,str(f) if target.startswith('/') else None)
      except OSError as e:err(pid,kind,f,e)
   except OSError as e:err(pid,kind,p/kind,e)
 d['completed_utc']=utc();return d
def source_changes(xs,groups):
 changes={}
 for k,group in groups.items():
  for x in group:
   try:
    if not eq(rec(x['path']),x):changes[str(k)]='candidate identity changed'
   except OSError as e:changes[str(k)]=repr(e)
 return changes
def preflight():
 xs,groups=load();RUN.mkdir(exist_ok=False)
 b=source_bindings();be=check_bindings(b);prot,hits,pe=protected(xs);ret=retained(xs);sc=scan(xs);changed=source_changes(xs,groups)
 zsave(RUN/'protected_preflight.json.gz',prot);save(RUN/'source_bindings.json',b)
 out=dict(utc=utc(),manifest_sha256=SHA,candidate_count=404,inodes=230,changed_groups=changed,source_errors=be,protected_errors=pe,protected_hits=hits,protected_count=len(prot),retained=ret,eden_scan=sc,df_before=G.df())
 save(RUN/'preflight.json',out)
 print(json.dumps({k:v for k,v in out.items() if k!='eden_scan'}));print('EDEN',json.dumps(sc),flush=True)
def strict_protected(saved):
 errors=[]
 for p,x in saved.items():
  try:
   if x.get('missing'):
    if exists(p):errors.append(dict(path=p,reason='missing protected path appeared'))
   else:
    n=rec(p,True)
    # Directory mtime/size may change from current experiments; its inode binding must not.
    fields=('dev','ino','mode') if stat.S_ISDIR(x['mode']) else F
    if not eq(n,x,fields) or os.path.realpath(p)!=x['realpath']:errors.append(dict(path=p,reason='protected identity changed'))
  except OSError as e:errors.append(dict(path=p,reason=repr(e)))
 return errors
def execute():
 xs,groups=load();pre=json.loads((RUN/'preflight.json').read_text());pak=json.loads((RUN/'pakka_writer_check.json').read_text())
 assert not pre['source_errors']+pre['protected_errors']+pre['protected_hits']+pre['retained']['errors']
 assert pak['manifest_sha256']==SHA and pak['candidate_count']==404 and pak['ready_for_execution']
 assert not pak['matches'] and not pak['errors'] and time.time()<pak['valid_until_epoch']
 b=json.loads((RUN/'source_bindings.json').read_text());assert not check_bindings(b),'current source bindings changed'
 ps=zr(RUN/'protected_preflight.json.gz');assert not strict_protected(ps),'protected inputs changed'
 ret=retained(xs);assert not ret['errors'],ret
 sc=scan(xs);save(RUN/'eden_immediate_process_scan.json',sc);assert not sc['errors'] and not sc['vanished'],'Eden scan incomplete'
 blocked={r for m in sc['matches'] for r in m['roots']}
 assert not any(m['candidate_inode'] and not m['roots'] for m in sc['matches']),'unmapped candidate inode open'
 holds=dict(pre['changed_groups']);holds.update(source_changes(xs,groups))
 for k,group in groups.items():
  if any(G.root_for(x['path']) in blocked for x in group):holds[str(k)]='visible active root'
 assert not (RUN/'leaf_receipts.jsonl').exists() and not (RUN/'unlink_intents.jsonl').exists(),'reconcile before any continuation'
 start=time.monotonic();deleted=skipped=freed=0
 with (RUN/'leaf_receipts.jsonl').open('x') as rf,(RUN/'unlink_intents.jsonl').open('x') as inf:
  def emit(f,d):f.write(json.dumps(d,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
  for k,group in groups.items():
   if time.monotonic()-start>480 or time.time()+5>=pak['valid_until_epoch']:raise RuntimeError('bounded stop; reconcile receipts')
   if str(k) in holds:
    for x in group:
     identity=rec(x['path']) if exists(x['path']) else {'missing':True}
     emit(rf,dict(utc=utc(),path=x['path'],status='SKIPPED',reason=holds[str(k)],skipped_identity=identity));skipped+=1
    continue
   expected=dict(group[0])
   for i,x in enumerate(group):
    fd=None
    try:
     fd,name=G.secure_parent(x['path'])
     def at():
      s=os.stat(name,dir_fd=fd,follow_symlinks=False)
      return dict([(f,getattr(s,'st_'+f)) for f in G.FIELDS]+[('allocated',s.st_blocks*512)])
     n=at()
     if not eq(n,expected):raise RuntimeError('final group identity changed')
     emit(inf,dict(utc=utc(),path=x['path'],status='UNLINK_INTENT',identity=n,group_index=i,group_size=len(group)))
     if not eq(at(),expected):raise RuntimeError('post-intent identity changed')
     if time.time()>=pak['valid_until_epoch']:raise RuntimeError('writer window expired')
     os.unlink(name,dir_fd=fd)
     try:os.stat(name,dir_fd=fd,follow_symlinks=False)
     except FileNotFoundError:pass
     else:raise RuntimeError('unlink did not establish absence')
     emit(rf,dict(utc=utc(),path=x['path'],status='UNLINKED',dev=x['dev'],ino=x['ino'],group_index=i,group_size=len(group),allocated_last_link=x['allocated'] if i==len(group)-1 else 0))
     deleted+=1
     if i<len(group)-1:
      next_identity=rec(group[i+1]['path'])
      assert eq(next_identity,expected,FIXED) and next_identity['nlink']==expected['nlink']-1 and next_identity['ctime_ns']>=expected['ctime_ns'],'unexpected hardlink transition'
      emit(rf,dict(utc=utc(),path=group[i+1]['path'],status='OWN_UNLINK_IDENTITY_TRANSITION',before=expected,after=next_identity))
      expected=next_identity
     else:freed+=x['allocated']
    except BaseException as e:
     save(RUN/'execution_error.json',dict(utc=utc(),path=x['path'],error=repr(e),deleted_before_error=deleted));raise
    finally:
     if fd is not None:os.close(fd)
   if deleted and deleted%50==0:print('PROGRESS',deleted,skipped,flush=True)
 save(RUN/'execution_result.json',dict(utc=utc(),deleted=deleted,skipped=skipped,closed_inode_allocated_bytes=freed,seconds=time.monotonic()-start,df_after_unlink=G.df()))
 print('EXECUTED',deleted,skipped,freed,flush=True)
def post():
 xs,groups=load();receipts=[json.loads(l) for l in (RUN/'leaf_receipts.jsonl').read_text().splitlines()]
 terminal=[r for r in receipts if r['status'] in ('UNLINKED','SKIPPED')]
 by={r['path']:r for r in terminal};assert len(by)==len(terminal)
 errors=[];gone=skip=freed=0
 for x in xs:
  r=by.get(x['path']);e=exists(x['path'])
  if r and r['status']=='UNLINKED' and not e:gone+=1
  elif r and r['status']=='SKIPPED' and e and not r['skipped_identity'].get('missing') and eq(rec(x['path']),r['skipped_identity']):skip+=1
  else:errors.append(dict(path=x['path'],receipt=r,exists=e))
 for group in groups.values():
  if all(by.get(x['path'],{}).get('status')=='UNLINKED' and not exists(x['path']) for x in group):freed+=group[0]['allocated']
 b=json.loads((RUN/'source_bindings.json').read_text());be=check_bindings(b);pe=strict_protected(zr(RUN/'protected_preflight.json.gz'));ret=retained(xs)
 pre=json.loads((RUN/'preflight.json').read_text());df=G.df()
 out=dict(status='VERIFIED' if not errors+be+pe+ret['errors'] else 'POSTCHECK_ISSUE',completed_utc=utc(),manifest_sha256=SHA,approved_count=404,verified_deleted=gone,verified_skipped=skip,unique_closed_inode_allocated_bytes=freed,GiB=freed/2**30,leaf_errors=errors,source_binding_errors=be,protected_errors=pe,retained=ret,df_before=pre['df_before'],df_after=df,df_available_delta=df['available']-pre['df_before']['available'],limits='Shared NFS df includes all writers; allocated is closed inode attribution, not guaranteed df causality. Selected current TMM/runtime and chosen cold-build, visible Eden/pakka only; not all other clients or future builders. Metadata checks/no-follow unlink are not an atomic inode-conditional unlink or a global write lock; residual concurrent replacement risk is bounded by observed idle roots, not eliminated.',sam_pilot='PAUSED',secondary899_deleted=False)
 save(RUN/'completion.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','execute','post']);a=p.parse_args()
 {'preflight':preflight,'execute':execute,'post':post}[a.phase]()
