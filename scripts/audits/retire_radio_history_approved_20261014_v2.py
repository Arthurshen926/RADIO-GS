"""Closed-manifest retirement. Preflight never unlinks. Exact owner approval required."""
import os,sys,json,gzip,hashlib,stat,time,pathlib,collections,traceback
if sys.flags.optimize:
 raise RuntimeError('Optimized Python disables identity gates; execution forbidden')
BASE=pathlib.Path('/mnt/pool/sqy/results/RADIO-GS/output')
PLAN=pathlib.Path('docs/experiments/2026-10-14-radio-history-payload-plan-1547')
OUT=pathlib.Path('docs/experiments/2026-10-14-radio-history-retirement-execution-1600')
SHA='a4da97c62e608413c97e2456d4e3e70c923b5941efa6763171c3cde7b26cfd93'
AUTH='Sentinel_0c1e3e3f96f481918b2af79f0cbdddf5'
FIELDS=('dev','ino','size','allocated','nlink','mode','mtime_ns','ctime_ns')
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sig(s):return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,allocated=s.st_blocks*512,nlink=s.st_nlink,mode=s.st_mode,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def savegz(name,value):
 with (OUT/name).open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(json.dumps(value,separators=(',',':')).encode())
  raw.flush();os.fsync(raw.fileno())
def df():
 s=os.statvfs(BASE);return dict(block_size=s.f_frsize,total=s.f_blocks*s.f_frsize,free=s.f_bfree*s.f_frsize,available=s.f_bavail*s.f_frsize)
def load():
 b=(PLAN/'final_retirement_leaves.json.gz').read_bytes();assert hashlib.sha256(b).hexdigest()==SHA
 rows=json.loads(gzip.decompress(b));summary=json.loads((PLAN/'final_retirement_summary.json').read_text())
 assert len(rows)==24389 and len({r['path'] for r in rows})==24389
 assert summary['payload_manifest_sha256']==SHA
 roots=set(summary['roots']);protected={pathlib.Path(r['root']).name for r in summary['protected_candidate_roots']}
 groups=collections.defaultdict(list)
 for r in rows:
  p=pathlib.Path(r['path']);rel=p.relative_to(BASE)
  assert '..' not in rel.parts and len(rel.parts)>1 and rel.parts[0]==r['root'] and r['root'] in roots and r['root'] not in protected
  assert p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.npy','.npz','.f16','.bin','.ply'} and r['size']>=1024**2 and stat.S_ISREG(r['mode'])
  groups[(r['dev'],r['ino'])].append(r)
 for rs in groups.values():assert len(rs)==rs[0]['nlink']
 assert len(groups)==23886 and sum(rs[0]['allocated'] for rs in groups.values())==1544622349312
 return rows,summary,groups
def open_abs_dir(p):
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for c in pathlib.Path(p).parts[1:]:
   nxt=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  return fd
 except BaseException:os.close(fd);raise
def fresh(r):
 fd=open_abs_dir(pathlib.Path(r['path']).parent)
 try:return sig(os.stat(pathlib.Path(r['path']).name,dir_fd=fd,follow_symlinks=False))
 finally:os.close(fd)
def proc_scan(rows):
 roots=sorted({r['root'] for r in rows});inode_roots=collections.defaultdict(set)
 for r in rows:inode_roots[(r['dev'],r['ino'])].add(r['root'])
 hits=[];issues=[];related=[];pids=0;fds=0;blocked=set()
 for e in os.scandir('/proc'):
  if not e.name.isdigit() or int(e.name)==os.getpid():continue
  pids+=1;p=pathlib.Path(e.path)
  try:cmd=(p/'cmdline').read_bytes()[:1048576].replace(b'\0',b' ').decode(errors='replace')
  except (OSError,PermissionError) as ex:issues.append(dict(pid=e.name,kind='cmdline',error=str(ex)));continue
  if any(x in cmd.lower() for x in ('radio','train','fit_','python')):related.append(dict(pid=e.name,cmdline=cmd[:2000]))
  sources=[('cmdline',cmd)]
  for name in ('cwd','exe'):
   try:sources.append((name,os.readlink(p/name)))
   except OSError:pass
  try:
   with (p/'maps').open(errors='replace') as f:sources.append(('maps',f.read(4*1024**2)))
  except OSError as ex:issues.append(dict(pid=e.name,kind='maps',error=str(ex)))
  try:
   for fe in os.scandir(p/'fd'):
    fds+=1
    if fds>200000:raise RuntimeError('FD scan count bound')
    try:
     target=os.readlink(fe.path);s=os.stat(fe.path);rr=inode_roots.get((s.st_dev,s.st_ino),set())
     if rr:blocked.update(rr);hits.append(dict(pid=e.name,kind='fd_inode',fd=fe.name,target=target,roots=sorted(rr)))
     sources.append(('fd',target))
    except (FileNotFoundError,ProcessLookupError):continue
    except PermissionError as ex:issues.append(dict(pid=e.name,kind='fd',error=str(ex)))
  except (FileNotFoundError,ProcessLookupError):pass
  except PermissionError as ex:issues.append(dict(pid=e.name,kind='fd_dir',error=str(ex)))
  for kind,value in sources:
   rr=[r for r in roots if '/'+r+'/' in value or value.endswith('/'+r) or ('output/'+r) in value]
   if rr:blocked.update(rr);hits.append(dict(pid=e.name,kind=kind,roots=rr,text=value[:1000]))
 return dict(utc=now(),scope='Eden container visible /proc only; not all NFS clients',pids=pids,fds=fds,hits=hits,blocked_roots=sorted(blocked),visibility_issues=issues,related_processes=related)
def preserve_snapshot(rows,summary):
 leaves={r['path'] for r in rows};kept=[];errors=[]
 for root in sorted(summary['roots']):
  def walk_error(ex):raise ex
  for parent,ds,fs in os.walk(BASE/root,followlinks=False,onerror=walk_error):
   for n in ds+fs:
    p=pathlib.Path(parent)/n
    if str(p) in leaves:continue
    try:
     s=os.lstat(p)
     if not stat.S_ISDIR(s.st_mode):kept.append(dict(path=str(p),**sig(s)))
    except OSError as ex:errors.append(dict(path=str(p),error=str(ex)))
 if errors:raise RuntimeError('retained snapshot errors: '+repr(errors[:5]))
 return kept
def protected_snapshot(summary):
 paths={r['root'] for r in summary['protected_candidate_roots']}
 paths.update(str(BASE/r) for r in summary['current_declared_roots'])
 old=json.loads((BASE/'storage_cleanup_20261014/radiogs_v10_json_dependencies_20261014.json').read_text())
 q=[pathlib.Path(x) for x in old['json_files_read']]+[pathlib.Path(old['seed'])];done=set();contracts=[];missing=[]
 def strings(v):
  if isinstance(v,str):yield v
  elif isinstance(v,dict):
   for x in v.values():yield from strings(x)
  elif isinstance(v,list):
   for x in v:yield from strings(x)
 while q:
  p=q.pop();ps=str(p)
  if ps in done or ps.startswith('/tmp/') or '/storage_' in ps:continue
  done.add(ps)
  assert len(done)<2000 and p.stat().st_size<2*1024**2
  b=p.read_bytes();obj=json.loads(b);contracts.append(dict(path=ps,sha256=hashlib.sha256(b).hexdigest()))
  for v in strings(obj):
   if not v or len(v)>4096 or '\n' in v:continue
   t=None
   if v.startswith(('/mnt/pool/sqy/results/RADIO-GS/output/','/root/RADIO-GS/output/','/root/RADIO-GS-query-field/output/')):t=BASE/v.split('/output/',1)[1]
   elif v.startswith('output/'):t=BASE/v[7:]
   elif not v.startswith('/') and '/' in v:
    a=p.parent/v
    if str(a).startswith(str(BASE)+'/') and a.exists():t=a
   if t is None:continue
   assert '..' not in t.parts
   if t.relative_to(BASE).parts[0] in summary['roots']:raise RuntimeError('fresh current consumer intersects retirement: '+str(t))
   if t.exists():paths.add(str(t))
   else:missing.append(dict(owner=ps,path=str(t)))
   if t.suffix=='.json' and not t.relative_to(BASE).parts[0].startswith('storage_') and t.is_file():q.append(t)
 snapshots=[]
 for p in sorted(paths):
  try:snapshots.append(dict(path=p,**sig(os.lstat(p))))
  except OSError as ex:missing.append(dict(path=p,error=str(ex)))
 return dict(paths=snapshots,contracts=contracts,missing=missing)
def preflight(rows,summary,groups):
 OUT.mkdir(exist_ok=False);changed=[]
 for r in rows:
  try:
   s=fresh(r)
   if any(s[k]!=r[k] for k in FIELDS):changed.append(dict(path=r['path'],expected=r,actual=s))
  except OSError as ex:changed.append(dict(path=r['path'],error=str(ex)))
 process=proc_scan(rows);kept=preserve_snapshot(rows,summary);protected=protected_snapshot(summary)
 savegz('retained_preflight.json.gz',kept);save('protected_preflight.json',protected)
 out=dict(status='PREFLIGHT_COMPLETE_NOT_YET_DELETED',utc=now(),manifest_sha256=SHA,approval_message_id=AUTH,rows=len(rows),unique_inodes=len(groups),expected_reclaim_bytes=sum(rs[0]['allocated'] for rs in groups.values()),changed=changed,process=process,retained_files=len(kept),protected_paths=len(protected['paths']),current_contracts=len(protected['contracts']),current_missing=protected['missing'],df_before=df())
 save('preflight.json',out);print(json.dumps(out))
def safety(rows,summary,groups):
 pre=json.loads((OUT/'preflight.json').read_text())
 kept=json.load(gzip.open(OUT/'retained_preflight.json.gz','rt'))
 old={r['path']:r for r in kept};current=preserve_snapshot(rows,summary)
 if len(old)!=len(current) or {r['path'] for r in current}!=set(old):raise RuntimeError('retained path set changed or initial traversal omitted entries')
 for r in current:
  if any(r[k]!=old[r['path']][k] for k in FIELDS):raise RuntimeError('retained identity changed: '+r['path'])
 leafpaths={r['path'] for r in rows};inode_roots=collections.defaultdict(set)
 for r in rows:inode_roots[(r['dev'],r['ino'])].add(r['root'])
 protected=json.loads((OUT/'protected_preflight.json').read_text());target_rows=[];blocked=set();legacy=[];broken=[]
 for r in protected['paths']+kept:
  iscurrent=r in protected['paths']
  if not iscurrent and not stat.S_ISLNK(r['mode']):continue
  p=pathlib.Path(r['path'])
  try:
   resolved=str(p.resolve(strict=True));ts=os.stat(p);rr=inode_roots.get((ts.st_dev,ts.st_ino),set())
   if stat.S_ISDIR(ts.st_mode):rr=rr|{x['root'] for x in rows if x['path'].startswith(resolved.rstrip('/')+'/')}
  except (OSError,RuntimeError) as ex:
   if iscurrent:raise RuntimeError('current target broken: '+str(p)) from ex
   broken.append(dict(path=str(p),error=str(ex)));continue
  record_suffix=p.suffix.lower() in {'.json','.csv','.tsv','.md','.txt','.html','.pdf','.png','.jpg','.jpeg','.webp','.svg','.mp4','.gif','.yaml','.yml'}
  if rr and (iscurrent or record_suffix):blocked.update(rr)
  if rr and not(iscurrent or record_suffix):legacy.append(dict(path=str(p),resolved=resolved,retiring_roots=sorted(rr),meaning='historical alias may become unusable with approved old branch retirement'))
  if iscurrent or record_suffix:target_rows.append(dict(path=str(p),resolved=resolved,target=sig(ts)))
 index=json.loads((PLAN/'record_and_current_input_review.json').read_text());evidence=[]
 for root in summary['roots']:
  for rec in index['record_reviews'][root]['records']:
   p=pathlib.Path(rec['path']);b=p.read_bytes()
   if hashlib.sha256(b).hexdigest()!=rec['sha256']:raise RuntimeError('preserved result SHA changed: '+str(p))
   evidence.append(dict(path=str(p),sha256=rec['sha256'],bytes=len(b)))
 result=dict(status='SAFETY_GATES_COMPLETE',utc=now(),manifest_sha256=SHA,retained_entries_checked=len(current),protected_targets=target_rows,blocked_roots=sorted(blocked),historical_aliases_affected=legacy,preexisting_broken_historical_symlinks=broken,result_records_hashed=evidence,script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 save('safety_preflight.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('protected_targets','historical_aliases_affected','preexisting_broken_historical_symlinks','result_records_hashed')}));print(json.dumps(dict(historical_aliases_affected=len(legacy),preexisting_broken_historical_symlinks=len(broken),result_records_hashed=len(evidence))))
def execute(rows,summary,groups):
 pre=json.loads((OUT/'preflight.json').read_text());external=json.loads((OUT/'pakka_writer_preflight.json').read_text())
 safe=json.loads((OUT/'safety_preflight.json').read_text())
 assert safe['status']=='SAFETY_GATES_COMPLETE' and safe['manifest_sha256']==SHA
 if scan_errors:=pre['process']['visibility_issues']:raise RuntimeError('unreviewed Eden process visibility errors: '+repr(scan_errors))
 assert pre['manifest_sha256']==SHA and external['manifest_sha256']==SHA and external['status']=='REVIEWED_FOR_EXECUTION'
 assert not pre['current_missing']
 assert time.time()-os.stat(OUT/'pakka_writer_preflight.json').st_mtime<900
 scan=proc_scan(rows);save('eden_process_immediate.json',scan)
 if scan['visibility_issues']:raise RuntimeError('immediate process visibility incomplete')
 blocked=set(safe['blocked_roots'])|set(scan['blocked_roots'])|set(external['blocked_roots'])|{pathlib.Path(x['path']).relative_to(BASE).parts[0] for x in pre['changed']}
 prot=json.loads((OUT/'protected_preflight.json').read_text())
 for x in prot['paths']:
  s=os.lstat(x['path']);assert (s.st_dev,s.st_ino)==(x['dev'],x['ino'])
 for x in prot['contracts']:assert hashlib.sha256(pathlib.Path(x['path']).read_bytes()).hexdigest()==x['sha256']
 for r in json.load(gzip.open(OUT/'retained_preflight.json.gz','rt')):
  s=sig(os.lstat(r['path']))
  if any(s[k]!=r[k] for k in FIELDS):raise RuntimeError('pre-delete retained identity changed: '+r['path'])
 for r in safe['result_records_hashed']:
  if hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('pre-delete evidence SHA changed')
 for r in safe['protected_targets']:
  s=sig(os.stat(r['path']))
  if any(s[k]!=r['target'][k] for k in ('dev','ino','mode','size','mtime_ns')):raise RuntimeError('pre-delete current/record target changed: '+r['path'])
 save('execution_started.json',dict(utc=now(),manifest_sha256=SHA,approval_message_id=AUTH,blocked_roots=sorted(blocked),accepted_loss=summary['accepted_loss_if_approved'],df=df()))
 deleted=0;skipped=0;released=0;stopped=None
 with (OUT/'leaf_receipts.jsonl').open('x',buffering=1) as log:
  def emit(x):log.write(json.dumps(x,separators=(',',':'))+'\n')
  try:
   for key,rs in groups.items():
    reasons=[]
    if any(r['root'] in blocked for r in rs):reasons.append('blocked_root')
    for r in rs:
     try:
      s=fresh(r)
      if any(s[k]!=r[k] for k in FIELDS):reasons.append('identity_changed:'+r['path'])
     except OSError as ex:reasons.append('stat_error:'+r['path']+':'+str(ex))
    if reasons:
     for r in rs:emit(dict(path=r['path'],status='SKIPPED',reasons=reasons,utc=now()));skipped+=1
     continue
    for j,r in enumerate(rs):
     fd=open_abs_dir(pathlib.Path(r['path']).parent)
     try:
      name=pathlib.Path(r['path']).name;s=sig(os.stat(name,dir_fd=fd,follow_symlinks=False))
      assert all(s[k]==r[k] for k in ('dev','ino','size','allocated','mode','mtime_ns')) and s['nlink']==r['nlink']-j
      if j==0:assert s['ctime_ns']==r['ctime_ns']
      os.unlink(name,dir_fd=fd)
      try:os.stat(name,dir_fd=fd,follow_symlinks=False);raise RuntimeError('leaf still exists after unlink')
      except FileNotFoundError:pass
      deleted+=1;emit(dict(path=r['path'],status='UNLINKED',original=r,utc=now()))
      if deleted%64==0:log.flush();os.fsync(log.fileno())
     finally:os.close(fd)
    released+=rs[0]['allocated']
  except BaseException as ex:stopped=repr(ex);emit(dict(status='STOPPED_UNCERTAIN_RECONCILIATION_REQUIRED',error=stopped,traceback=traceback.format_exc(),utc=now()))
  finally:log.flush();os.fsync(log.fileno())
 result=dict(status='EXECUTION_FINISHED' if stopped is None else 'STOPPED',utc=now(),manifest_sha256=SHA,deleted=deleted,skipped=skipped,expected_lastlink_reclaimed_bytes=released,error=stopped,df_after=df())
 save('execution_summary.json',result);print(json.dumps(result))
 if stopped:raise SystemExit(2)
def verify(rows,summary,groups):
 receipts=[json.loads(x) for x in (OUT/'leaf_receipts.jsonl').read_text().splitlines()];by={x['path']:x for x in receipts if 'path' in x};remaining=[];anomalies=[]
 for r in rows:
  exists=os.path.lexists(r['path']);status=by.get(r['path'],{}).get('status')
  if exists:remaining.append(dict(path=r['path'],receipt_status=status))
  if (status=='UNLINKED' and exists) or (status=='SKIPPED' and not exists) or status is None:anomalies.append(dict(path=r['path'],exists=exists,receipt_status=status))
 kept=json.load(gzip.open(OUT/'retained_preflight.json.gz','rt'));keep_errors=[]
 for r in kept:
  try:
   s=sig(os.lstat(r['path']))
   if any(s[k]!=r[k] for k in FIELDS):keep_errors.append(dict(path=r['path'],before=r,after=s))
  except OSError as ex:keep_errors.append(dict(path=r['path'],error=str(ex)))
 protected=json.loads((OUT/'protected_preflight.json').read_text());problems=[]
 for r in protected['paths']:
  try:
   s=sig(os.lstat(r['path']));ks=('dev','ino','mode') if stat.S_ISDIR(r['mode']) else FIELDS
   if any(s[k]!=r[k] for k in ks):problems.append(dict(path=r['path'],before=r,after=s))
  except OSError as ex:problems.append(dict(path=r['path'],error=str(ex)))
 for r in protected['contracts']:
  try:
   if hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:problems.append(dict(path=r['path'],error='current_contract_sha_changed'))
  except OSError as ex:problems.append(dict(path=r['path'],error=str(ex)))
 safe=json.loads((OUT/'safety_preflight.json').read_text())
 for r in safe['protected_targets']:
  try:
   s=sig(os.stat(r['path']))
   if any(s[k]!=r['target'][k] for k in ('dev','ino','mode','size','mtime_ns')):problems.append(dict(path=r['path'],error='protected resolved target changed'))
  except OSError as ex:problems.append(dict(path=r['path'],error=str(ex)))
 for r in safe['result_records_hashed']:
  if hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:keep_errors.append(dict(path=r['path'],error='record_sha_changed'))
 result=dict(status='POST_VERIFIED' if not(anomalies or keep_errors or problems) else 'POSTCHECK_REVIEW_NEEDED',utc=now(),manifest_sha256=SHA,approved_leaves=len(rows),unlinked=sum(x.get('status')=='UNLINKED' for x in by.values()),remaining_count=len(remaining),remaining=remaining,leaf_anomalies=anomalies,retained_checked=len(kept),retained_errors=keep_errors,protected_checked=len(protected['paths']),protected_errors=problems,df_settled=df(),loss_statement=summary['accepted_loss_if_approved'])
 save('post_verification.json',result);print(json.dumps(result))
if __name__=='__main__':
 rows,summary,groups=load();mode=sys.argv[1]
 {'preflight':preflight,'safety':safety,'execute':execute,'verify':verify}[mode](rows,summary,groups)
