#!/usr/bin/env python3
"""Read-only exact H12 proposal; writes audit files only, never deletes payloads."""
import collections, datetime, gzip, hashlib, json, os, stat, time
from pathlib import Path
BASE=Path('/mnt/pool/sqy/results/RADIO-GS/output')
DOC=Path('/root/RADIO-GS-query-field/docs/experiments')
OLD=DOC/'2026-10-14-radio-h449-payload-plan-1635'
OUT=DOC/'2026-10-15-radio-h12-retirement-proposal-0352'
ROOTS=['depth_opacity_pipeline_20260918','lerf_complete_v2_20260918','nvos_ropefix_pipeline_20260917','scannet_uqis_9_v0_1','paired_source_resolution','scannet_pointcloud_eval','samclip_ablation_eval','local_element_fusion','source_referent_context','native_region_ownership','query_consistency','region_identity_transport']
FIELDS=('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')
TEXT={'.json','.jsonl','.md','.txt','.log','.yaml','.yml','.csv','.tsv','.py','.sh','.toml'}
IMAGE={'.png','.jpg','.jpeg','.webp','.ppm','.gif','.mp4','.avi','.mov'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):
 s=os.lstat(p);v={k:getattr(s,'st_'+k) for k in FIELDS};v['allocated']=s.st_blocks*512
 if stat.S_ISLNK(s.st_mode):v['link']=os.readlink(p)
 return v
def zread(p):return json.load(gzip.open(p,'rt'))
def same(a,b):return all(a[k]==b[k] for k in FIELDS) and a['allocated']==b['allocated']
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def zw(p,v):
 with p.open('xb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(json.dumps(v,separators=(',',':')).encode())
t=time.monotonic()
closure=DOC/'2026-10-15-selected-closure-readonly.json'
assert sha(closure)=='bd55915776649ad61abc0ce95c50d696e52480c14524eeee6283085ac88d643d'
c=json.loads(closure.read_text());assert not c['unknown_required_edges']
cs=json.loads((DOC/'2026-10-15-selected-closure-summary.json').read_text())
holds=set(cs['required_output_roots'])|{'canonical_fields','storage_v2_pilots_20261013','storage_v2_batches_20261013'}
protected=zread(DOC/'2026-10-15-radio-762-retirement-execution-0340/protected_preflight.json.gz')
protected_paths=set();protected_inodes=set();protected_dirs=[];issues=[]
for p,x in protected.items():
 if x.get('missing'):continue
 try:
  n=rec(p)
  if not all(n[k]==x[k] for k in FIELDS):issues.append({'path':p,'reason':'protected identity changed'})
  if 'sha256' in x and sha(p)!=x['sha256']:issues.append({'path':p,'reason':'protected source SHA changed'})
  protected_paths.add(os.path.realpath(p));protected_inodes.add((n['dev'],n['ino']))
  if 'target' in x:
   s=os.stat(p);q={k:getattr(s,'st_'+k) for k in FIELDS}
   if any(q[k]!=x['target'][k] for k in FIELDS):issues.append({'path':p,'reason':'protected target changed'})
   protected_inodes.add((s.st_dev,s.st_ino))
   if stat.S_ISDIR(s.st_mode):protected_dirs.append(os.path.realpath(p).rstrip('/')+'/')
  if 'realpath' in x and os.path.realpath(p)!=x['realpath']:issues.append({'path':p,'reason':'protected resolution changed'})
  if x.get('target_missing') and os.path.exists(p):issues.append({'path':p,'reason':'known missing target changed'})
 except OSError as e:issues.append({'path':p,'reason':str(e)})
assert not issues,issues[:20]
old=zread(OLD/'content_reviewed_leaves.json.gz')
reviews=json.loads((OLD/'content_review.json').read_text())['reviews']
approved=zread(DOC/'2026-10-15-radio-three-group-retirement-proposal-0303/frozen_762_leaves.json.gz')
deleted_paths={x['path'] for x in approved['leaves']}
record_guard={};bad_roots={};exclusions=[];eligible={}
for root in ROOTS:
 if root in holds:bad_roots[root]='selected runtime/cold-build required root';continue
 r=reviews[root];assert r['status']=='PRESERVED_QUANTITATIVE_OR_TEXT_RESULT_VERIFIED'
 for ev in r['records']:
  p=ev['path']
  if not Path(p).is_file() or sha(p)!=ev['sha256']:bad_roots[root]='quantitative/decision record changed or missing'
  else:record_guard[p]={'sha256':ev['sha256'],**rec(p)}
for x in old:
 if x['root'] not in ROOTS:continue
 p=x['path'];reason=None
 if x['root'] in bad_roots:reason=bad_roots[x['root']]
 elif p in deleted_paths:reason='previous approved/deleted leaf'
 elif os.path.realpath(p) in protected_paths or (x['dev'],x['ino']) in protected_inodes or any(os.path.realpath(p).startswith(d) for d in protected_dirs):reason='selected runtime/cold-build path/inode/subtree'
 else:
  try:
   n=rec(p)
   if not stat.S_ISREG(n['mode']) or not same(n,x):reason='changed/missing/nonregular since H audit'
   else:eligible[p]={'path':p,'root':x['root'],**n}
  except OSError as e:reason=str(e)
 if reason:exclusions.append({'path':p,'reason':reason})
all_entries={};symlink_targets=set();scan_errors=[]
def onerror(e):scan_errors.append(str(e))
for root in ROOTS:
 for parent,dirs,files in os.walk(BASE/root,followlinks=False,onerror=onerror):
  for name in dirs+files:
   p=str(Path(parent)/name)
   try:
    n=rec(p)
    if stat.S_ISDIR(n['mode']):continue
    all_entries[p]=n
    if stat.S_ISLNK(n['mode']):
     target=os.path.realpath(p);symlink_targets.add(target)
     if os.path.isfile(p):
      s=os.stat(p);protected_inodes.add((s.st_dev,s.st_ino))
   except OSError as e:scan_errors.append({'path':p,'error':str(e)})
assert not scan_errors,scan_errors[:10]
groups=collections.defaultdict(list)
for x in eligible.values():groups[(x['dev'],x['ino'])].append(x)
leaves=[]
for key,group in groups.items():
 reason=None
 if key in protected_inodes or any(x['path'] in symlink_targets for x in group):reason='retained symlink target/current protected inode'
 elif len(group)!=group[0]['nlink']:reason='hardlink names outside selected eligible set'
 if reason:exclusions.extend({'path':x['path'],'reason':reason} for x in group)
 else:leaves.extend(group)
chosen={x['path'] for x in leaves}
retained={p:n for p,n in all_entries.items() if p not in chosen}
assert all(p in retained for p in record_guard)
for x in leaves:assert same(rec(x['path']),x),('changed on second pass',x['path'])
seen=set();byroot={r:{'leaves':0,'joint_exclusive_allocated_bytes':0,'retained_files':0,'retained_text':0,'retained_images':0} for r in ROOTS}
suffix=collections.defaultdict(lambda:{'leaves':0,'path_allocated_bytes':0})
for x in leaves:
 root=x['root'];byroot[root]['leaves']+=1;k=(x['dev'],x['ino'])
 if k not in seen:byroot[root]['joint_exclusive_allocated_bytes']+=x['allocated'];seen.add(k)
 s=Path(x['path']).suffix.lower();suffix[s]['leaves']+=1;suffix[s]['path_allocated_bytes']+=x['allocated']
for p,n in retained.items():
 root=Path(p).relative_to(BASE).parts[0];byroot[root]['retained_files']+=1
 if Path(p).suffix.lower() in TEXT:byroot[root]['retained_text']+=1
 if Path(p).suffix.lower() in IMAGE:byroot[root]['retained_images']+=1
OUT.mkdir(exist_ok=False)
payload={'status':'PROPOSAL_ONLY_NOT_AUTHORIZED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Only exact H12 historical leaves; no deletion','leaves':sorted(leaves,key=lambda x:x['path'])}
zw(OUT/'frozen_leaves.json.gz',payload);zw(OUT/'retained_files.json.gz',retained);write(OUT/'retained_result_evidence.json',{'evidence':{r:reviews[r] for r in ROOTS},'fresh_records':record_guard});write(OUT/'excluded_leaves.json',exclusions)
summary={'status':'FROZEN_PROPOSAL_REQUIRES_NEW_PERMANENT_DELETE_APPROVAL','root_count':len(ROOTS),'leaf_count':len(leaves),'unique_inode_count':len(seen),'joint_exclusive_allocated_bytes':sum(v['joint_exclusive_allocated_bytes'] for v in byroot.values()),'roots':byroot,'suffix':dict(suffix),'exclusion_count':len(exclusions),'exclusion_reasons':dict(collections.Counter(x['reason'] for x in exclusions)),'current_protection_count':len(protected),'current_closure_sha256':sha(closure),'protected_changes':issues,'retained_file_count':len(retained),'retained_result_records_sha_verified':len(record_guard),'retained_symlinks':sum(stat.S_ISLNK(x['mode']) for x in retained.values()),'prior762_overlap':len(chosen&deleted_paths),'payload_bytes_read':0,'seconds':time.monotonic()-t,'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir()},'loss':'Exact historical teacher/field/prediction states and old branch replay may be irrecoverable. Retain all code/config/logs/quantitative conclusions/images/small records. Current selected runtime and chosen cold-build protected; unknown future/cross-project consumers not exhaustively established.','execution_gates':'Fresh source identity, all hardlink group identity, retained records and both visible process namespaces required before any approved unlink. No authorization granted by this proposal.','sam_pilot':'PAUSED_NOT_STARTED'}
write(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
