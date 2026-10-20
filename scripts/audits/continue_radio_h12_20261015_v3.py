#!/usr/bin/env python3
"""Resume only unprocessed exact approved H12 leaves after terminal, reconciled prior batches."""
import argparse,collections,datetime,gzip,hashlib,json,os,shutil,stat,sys,types
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized execution forbidden')
REPO=Path('/root/RADIO-GS-query-field');D=REPO/'docs/experiments'
ORIG=D/'2026-10-15-radio-h12-retirement-execution-0459'
SRC=REPO/'scripts/audits/retire_radio_exact_h12_20261015_v2.py'
raw=SRC.read_bytes();assert hashlib.sha256(raw).hexdigest()=='0d45c4b4bc697b0df916c142abb90e9df6b58e7cd210052785e852dba89f4e80'
M=types.ModuleType('h12_guard');exec(compile(raw,str(SRC),'exec'),M.__dict__);original_load=M.load
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def exists_strict(path):
 try:os.lstat(path);return True
 except FileNotFoundError:return False
def prior_dirs(stage):return [ORIG]+[ORIG/('continuation_'+str(i)) for i in range(1,stage)]
def reconcile(stage):
 xs,groups=original_load();index={x['path']:x for x in xs};rows=[];seen=set();provenance=[]
 for d in prior_dirs(stage):
  p=d/'leaf_receipts.jsonl';b=p.read_bytes();assert b.endswith(b'\n') or not b
  provenance.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest()})
  for line in b.splitlines():
   r=json.loads(line);p0=r['path'];assert p0 in index and p0 not in seen;seen.add(p0);rows.append(r)
   assert r['status'] in ('UNLINKED','SKIPPED'),('uncertain receipt',r)
   if r['status']=='UNLINKED':assert not exists_strict(p0),('deleted path exists',p0)
   else:assert exists_strict(p0),('skipped path missing',p0)
  assert (d/'execution_result.json').exists() or (d/'execution_error.json').exists(),('prior execution not terminal on disk',str(d))
 remaining=[x for x in xs if x['path'] not in seen]
 for x in remaining:
  assert exists_strict(x['path']),('unreceipted missing leaf: reconcile manually',x['path'])
 remaining_groups=collections.defaultdict(list)
 for x in remaining:remaining_groups[(x['dev'],x['ino'])].append(x)
 for key,g in remaining_groups.items():assert len(g)==g[0]['nlink'],'do not resume partial hardlink group'
 return xs,remaining,remaining_groups,rows,provenance
def idle_guard():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/'cmdline').read_bytes().decode(errors='replace')
  except OSError:continue
  assert not (('retire_radio_exact_h12_20261015_v2.py' in cmd or 'continue_radio_h12_' in cmd) and '\0execute' in cmd),('prior executor active',p.name)
def setup(stage):
 idle_guard();xs,remaining,groups,rows,prov=reconcile(stage);r=ORIG/('continuation_'+str(stage));assert not r.exists()
 pc=M.check_map(M.protected());assert not pc['errors'],pc['errors'][:10]
 rc=M.check_map(M.zread(M.PLAN/'retained_files.json.gz'),False,False);assert not rc['errors'],rc['errors'][:10]
 ec=M.check_map(json.loads((M.PLAN/'retained_result_evidence.json').read_text())['fresh_records']);assert not ec['errors'],ec['errors'][:10]
 syms=M.zread(ORIG/'retained_symlink_preflight.json.gz');sc=M.check_map(syms,False,True);assert not sc['errors'],sc['errors'][:10]
 for p,s in syms.items():
  if s.get('target_directory_dev_ino'):
   q=os.stat(p);assert [q.st_dev,q.st_ino]==s['target_directory_dev_ino']
 pre=json.loads((ORIG/'preflight.json').read_text());remaining_keys={str(k) for k in groups};holds={k:v for k,v in pre['held_groups'].items() if k in remaining_keys}
 prot=M.protected();protected_paths={os.path.realpath(p) for p,s in prot.items() if not s.get('missing')};protected_inodes={(s['dev'],s['ino']) for s in prot.values() if not s.get('missing')};protected_dirs=[]
 for p,s in prot.items():
  if s.get('target'):
   protected_inodes.add((s['target']['dev'],s['target']['ino']))
   if stat.S_ISDIR(s['target']['mode']):protected_dirs.append(os.path.realpath(p).rstrip('/')+'/')
 for p,s in syms.items():
  if s.get('target'):protected_inodes.add((s['target']['dev'],s['target']['ino']))
 for k,g in groups.items():
  if len(g)>1:holds[str(k)]='multi-hardlink group held: strict original ctime identity, no transition exemption'
  for x in g:
   real=os.path.realpath(x['path'])
   if k in protected_inodes or real in protected_paths or any(real.startswith(d) for d in protected_dirs):holds[str(k)]='current protected input or retained symlink target'
   elif not M.G.match(M.G.rec(x['path']),x):holds[str(k)]='remaining candidate identity changed'
 pre.update({'created_utc':utc(),'stage':stage,'leaf_count':len(remaining),'inode_count':len(groups),'held_groups':holds,'blocked_roots':pre.get('blocked_roots',[]),'protected_check':pc,'retained_check':rc,'evidence_check':ec,'prior_receipt_provenance':prov,'prior_processed_count':len(rows),'df_before':M.G.df()})
 r.mkdir();M.save(r/'preflight.json',pre);M.zsave(r/'remaining_frozen_subset.json.gz',{'original_manifest_sha256':M.SHA,'leaves':remaining,'prior_receipts':prov});(r/'retained_symlink_preflight.json.gz').write_bytes((ORIG/'retained_symlink_preflight.json.gz').read_bytes())
 print(json.dumps({'stage':stage,'remaining_leaves':len(remaining),'remaining_single_leaves':sum(x['nlink']==1 for x in remaining),'prior_statuses':dict(collections.Counter(x['status'] for x in rows)),'held_groups':len(holds),'fresh_checks':'passed','directory':str(r)}),flush=True)
def execute(stage):
 idle_guard();xs,remaining,groups,rows,prov=reconcile(stage);r=ORIG/('continuation_'+str(stage));pre=json.loads((r/'preflight.json').read_text());assert pre['prior_receipt_provenance']==prov
 frozen=M.zread(r/'remaining_frozen_subset.json.gz');assert frozen['original_manifest_sha256']==M.SHA and frozen['leaves']==remaining
 M.RUN=r;M.load=lambda:(remaining,groups);M.execute()
def aggregate(stage):
 idle_guard();xs,remaining,groups,rows,prov=reconcile(stage+1);assert not remaining,('still unprocessed',len(remaining))
 r=ORIG/'final_aggregate';assert not r.exists();r.mkdir()
 M.save(r/'preflight.json',json.loads((ORIG/'preflight.json').read_text()))
 (r/'retained_symlink_preflight.json.gz').write_bytes((ORIG/'retained_symlink_preflight.json.gz').read_bytes())
 with (r/'leaf_receipts.jsonl').open('x') as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 M.save(r/'batch_provenance.json',{'original_manifest_sha256':M.SHA,'receipt_files':prov,'stages':stage+1})
 M.RUN=r;M.load=original_load
 original_lexists=M.os.path.lexists;M.os.path.lexists=exists_strict
 try:M.post()
 finally:M.os.path.lexists=original_lexists
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['setup','execute','aggregate']);p.add_argument('stage',type=int);a=p.parse_args();assert 1<=a.stage<=5;{'setup':setup,'execute':execute,'aggregate':aggregate}[a.phase](a.stage)
