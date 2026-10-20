"""Reconcile approved original scope before an isolated continuation. No blind retries."""
import importlib.util,pathlib,json,gzip,hashlib,os,stat,sys,time,collections,shutil
SPEC=importlib.util.spec_from_file_location('approved_retirement','scripts/audits/retire_radio_history_approved_20261014_v4.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)
OLD=A.OUT;R=OLD/'continuation01';FINAL=OLD/'final_combined'
STABLE=('dev','ino','size','allocated','mode','mtime_ns')
PROOFS=['preflight.json','retained_preflight.json.gz','protected_preflight.json','safety_preflight.json','directory_symlink_preflight.json']
def no_old_executor():
 hits=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if 'retire_radio_history_approved_20261014_v4.py execute' in cmd:hits.append(dict(pid=p.name,cmdline=cmd,start_ticks=(p/'stat').read_text().split()[21]))
  except (FileNotFoundError,ProcessLookupError):pass
  except PermissionError as ex:raise RuntimeError('process visibility incomplete') from ex
 if hits:raise RuntimeError('Original executor still active; do not continue: '+repr(hits))
def parse_receipts(path):
 raw=path.read_bytes();rows=[];invalid=[]
 for i,line in enumerate(raw.splitlines()):
  try:rows.append(json.loads(line))
  except ValueError:invalid.append(dict(line=i+1,bytes=len(line),sha256=hashlib.sha256(line).hexdigest()))
 return rows,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),invalid_lines=invalid)
def reconcile(original,summary,groups):
 cached_dirs();no_old_executor();terminal=json.loads((OLD/'original_job_terminal.json').read_text())
 if not terminal.get('terminal'):raise RuntimeError('Original job not terminal')
 rec,proof=parse_receipts(OLD/'leaf_receipts.jsonl');by={x['path']:x for x in rec if 'path' in x}
 if len(by)!=sum('path' in x for x in rec):raise RuntimeError('duplicate original receipts')
 absent=[];unknown=[];hold=[];pending=[];remaining_groups=0
 for key,rs in groups.items():
  survivors=[];missing=[];why=[]
  for r in rs:
   status=by.get(r['path'],{}).get('status')
   try:s=A.fresh(r)
   except FileNotFoundError:
    missing.append(r)
    (absent if status=='UNLINKED' else unknown).append(r['path'])
    continue
   except OSError as ex:why.append(dict(path=r['path'],reason=repr(ex)));continue
   if status=='UNLINKED':why.append(dict(path=r['path'],reason='previously unlinked path reappeared'));continue
   if status=='SKIPPED':why.append(dict(path=r['path'],reason='prior skip remains held'));continue
   if any(s[k]!=r[k] for k in STABLE):why.append(dict(path=r['path'],reason='original stable identity changed',actual=s));continue
   survivors.append((r,s))
  if why:
   hold.append(dict(inode=key,paths=[r['path'] for r in rs],reasons=why));continue
  if not survivors:continue
  if any(s['nlink']!=len(survivors) for r,s in survivors):hold.append(dict(inode=key,paths=[r['path'] for r in rs],reasons=['remaining hardlink closure mismatch']));continue
  if not missing and any(s['ctime_ns']!=r['ctime_ns'] for r,s in survivors):hold.append(dict(inode=key,paths=[r['path'] for r in rs],reasons=['unexplained ctime change']));continue
  remaining_groups+=1
  for r,s in survivors:pending.append(dict(r,**s))
 no_old_executor();R.mkdir(exist_ok=False);A.OUT=R
 for name in PROOFS:shutil.copyfile(OLD/name,R/name)
 A.savegz('remaining_manifest.json.gz',pending)
 m=(R/'remaining_manifest.json.gz').read_bytes()
 out=dict(status='RECONCILED_FIXED_ORIGINAL_APPROVAL_SCOPE',utc=A.now(),original_manifest_sha256=A.SHA,original_job_terminal=terminal,original_receipt=proof,confirmed_absent_with_unlink_receipt=len(absent),absent_without_unlink_receipt=unknown,remaining_leaves=len(pending),remaining_inodes=remaining_groups,remaining_manifest_sha256=hashlib.sha256(m).hexdigest(),held_groups=hold,original_approved_leaves=len(original),continuation_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
 A.save('reconciliation.json',out);print(json.dumps(out))
def cached_dirs():
 raw=A.open_abs_dir;cache=collections.OrderedDict()
 def cached(p):
  key=str(p)
  if key in cache:
   fd=cache.pop(key);s=os.stat(p,follow_symlinks=False);t=os.fstat(fd)
   if not stat.S_ISDIR(s.st_mode) or (s.st_dev,s.st_ino)!=(t.st_dev,t.st_ino):os.close(fd);raise RuntimeError('cached parent replaced')
   cache[key]=fd;return os.dup(fd)
  fd=raw(p);cache[key]=fd
  if len(cache)>256:os.close(cache.popitem(last=False)[1])
  return os.dup(fd)
 A.open_abs_dir=cached
def resume(original,summary,groups):
 no_old_executor();A.OUT=R;r=json.loads((R/'reconciliation.json').read_text())
 if r['original_manifest_sha256']!=A.SHA:raise RuntimeError('original authority mismatch')
 blob=(R/'remaining_manifest.json.gz').read_bytes()
 if hashlib.sha256(blob).hexdigest()!=r['remaining_manifest_sha256']:raise RuntimeError('remaining manifest mismatch')
 pending=json.loads(gzip.decompress(blob));orig={x['path']:x for x in original};g=collections.defaultdict(list)
 if len({x['path'] for x in pending})!=len(pending):raise RuntimeError('duplicate remaining leaves')
 for x in pending:
  if x['path'] not in orig or any(x[k]!=orig[x['path']][k] for k in STABLE):raise RuntimeError('remaining scope expanded')
  g[(x['dev'],x['ino'])].append(x)
 for rs in g.values():
  if len(rs)!=rs[0]['nlink']:raise RuntimeError('remaining hardlink closure mismatch')
 A.save('continuation_authority.json',dict(utc=A.now(),original_manifest_sha256=A.SHA,remaining_manifest_sha256=r['remaining_manifest_sha256'],source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),note='Same approved original leaves only. Cached parent dirfds retain no-follow validated directory identities.'))
 cached_dirs();A.execute(pending,summary,g)
def verify_all(original,summary,groups):
 no_old_executor()
 FINAL.mkdir(exist_ok=False);A.OUT=FINAL
 for name in PROOFS:shutil.copyfile(OLD/name,FINAL/name)
 sources=[];combined=[]
 for path in [OLD/'leaf_receipts.jsonl',R/'leaf_receipts.jsonl']:
  if path.exists():rs,p=parse_receipts(path);sources.append(p);combined.extend(rs)
 with (FINAL/'leaf_receipts.jsonl').open('x') as f:
  for x in combined:f.write(json.dumps(x,separators=(',',':'))+'\n')
 A.save('receipt_sources.json',dict(source_logs=sources,original_manifest_sha256=A.SHA,derived_view_only=True,no_missing_success_records_synthesized=True))
 A.verify(original,summary,groups)
if __name__=='__main__':
 original,summary,groups=A.load()
 {'reconcile':reconcile,'resume':resume,'verify_all':verify_all}[sys.argv[1]](original,summary,groups)
