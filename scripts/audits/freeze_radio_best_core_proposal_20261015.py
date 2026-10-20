#!/usr/bin/env python3
"""Read-only next best/core candidate freeze; no deletion and no tensor reads."""
import collections,datetime,gzip,hashlib,json,os,stat,time
from pathlib import Path
BASE=Path('/mnt/pool/sqy/results/RADIO-GS/output')
DOC=Path('/root/RADIO-GS-query-field/docs/experiments')
OUT=DOC/'2026-10-15-radio-best-core-retirement-proposal-0358'
FIELDS=('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')
def rec(p):
 s=os.lstat(p);d={k:getattr(s,'st_'+k) for k in FIELDS};d['allocated']=s.st_blocks*512;return d
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def zw(p,v):
 with p.open('xb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(v,separators=(',',':')).encode())
input_path=DOC/'2026-10-15-radio-next-best-core-input-0353.json';data=json.loads(input_path.read_text())
closure=DOC/'2026-10-15-selected-closure-readonly.json'
assert sha(closure)=='bd55915776649ad61abc0ce95c50d696e52480c14524eeee6283085ac88d643d'
c=json.loads(closure.read_text());assert not c['unknown_required_edges']
paths=set();inos=set();dirs=[]
for x in c['entries']:
 paths.add(x['path']);paths.add(x.get('realpath',x['path']))
 if x.get('exists'):
  inos.add((x['dev'],x['ino']))
  if not x.get('is_file',True):dirs.append(x['realpath'].rstrip('/')+'/')
# Ensure the actual selected source/config files have not changed since full closure validation.
pre=json.load(gzip.open(DOC/'2026-10-15-radio-762-retirement-execution-0340/protected_preflight.json.gz','rt'))
for p in c['source_jsons']:
 assert p in pre and pre[p].get('sha256')==sha(p),('selected source/config changed',p)
deleted=json.load(gzip.open(DOC/'2026-10-15-radio-three-group-retirement-proposal-0303/frozen_762_leaves.json.gz','rt'))
deleted_paths={x['path'] for x in deleted['leaves']}
leaves=[];excluded=[];records={};evidence={}
def admissible(p):
 if p in deleted_paths:return None,'previous762'
 try:
  s=rec(p)
  if not stat.S_ISREG(s['mode']):return None,'nonregular'
  if s['nlink']!=1:return None,'not unique hardlink'
  real=os.path.realpath(p)
  if real in paths or (s['dev'],s['ino']) in inos or any(real.startswith(d) for d in dirs):return None,'selected runtime/coldbuild dependency'
  return s,None
 except OSError as e:return None,str(e)
def save_record(p):
 p=Path(p);assert p.is_file() and p.stat().st_size<=2097152
 records[str(p)]={'sha256':sha(p),**rec(p)}
for x in data['best']:
 p=x['absolute_path'];s,reason=admissible(p)
 if reason:excluded.append({'path':p,'reason':reason});continue
 root=Path(x['run_root']);report=root/'reports/experiment_report.json';config=root/'reports/resolved_config.json'
 if not report.is_file() or not config.is_file():excluded.append({'path':p,'reason':'canonical completed experiment report/config unavailable; alternate schema held'});continue
 if report.stat().st_size>2097152 or config.stat().st_size>2097152:excluded.append({'path':p,'reason':'record exceeds bounded read'});continue
 try:d=json.loads(report.read_text());cfg=json.loads(config.read_text())
 except Exception as e:excluded.append({'path':p,'reason':'record parse issue: '+str(e)});continue
 if d.get('status')!='completed' or not d.get('final'):excluded.append({'path':p,'reason':'record not final completed','status':d.get('status')});continue
 if not d.get('last_val_metrics') and d.get('best_selection_value') is None:excluded.append({'path':p,'reason':'no retained quantitative outcome'});continue
 save_record(report);save_record(config)
 for q in (root/'reports').iterdir():
  if q.is_file() and q.suffix.lower() in {'.json','.jsonl','.csv','.txt','.md','.yaml','.yml'} and q.stat().st_size<=2097152:save_record(q)
 leaves.append({'path':p,'group':'best_'+x['old_role'],'research_family':x['research_family'],**s})
 evidence[p]={'report':str(report),'config':str(config),'status':d['status'],'epoch':d.get('epoch'),'best_metric':d.get('best_metric_name'),'best_selection_value':d.get('best_selection_value'),'record_policy':'Keep all reports/configuration/logs/figures; retired historical intermediate/variant checkpoint, no training resume guarantee.'}
print('BEST_DONE',len(leaves),len(excluded),flush=True)
for x in data['core']:
 p=x['absolute_path'];s,reason=admissible(p)
 if reason:excluded.append({'path':p,'reason':reason});continue
 q=Path(p);side=q.with_suffix('.json')
 if x['group']=='A_rejected_four':
  candidates=[side];reason_record='rejected pre-sharded support-tolerance branch; distinct version, not byte duplicate'
 else:
  candidates=[side,q.parent/'method_v1_gate.json'];reason_record='historical factorized/siglip/region predecessor; keep final generic field, retire original construction-lineage rehash and resume'
 found=[z for z in candidates if z.is_file() and z.stat().st_size<=2097152]
 if not found:excluded.append({'path':p,'reason':'expected stage sidecar/gate absent; bounded evidence incomplete'});continue
 for z in found:save_record(z)
 leaves.append({'path':p,'group':'core_'+x['group'],**s});evidence[p]={'preserved_records':[str(z) for z in found],'retired_capability':reason_record}
for x in leaves:
 n=rec(x['path']);assert all(n[k]==x[k] for k in FIELDS) and n['allocated']==x['allocated']
OUT.mkdir(exist_ok=False);zw(OUT/'frozen_leaves.json.gz',{'status':'PROPOSAL_ONLY_NOT_AUTHORIZED','leaves':leaves});write(OUT/'evidence_and_record_guard.json',{'leaf_evidence':evidence,'records':records});write(OUT/'excluded.json',excluded)
group=collections.defaultdict(lambda:{'leaves':0,'exclusive_allocated_bytes':0})
for x in leaves:group[x['group']]['leaves']+=1;group[x['group']]['exclusive_allocated_bytes']+=x['allocated']
summary={'status':'FROZEN_PROPOSAL_REQUIRES_NEW_APPROVAL','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'leaves':len(leaves),'exclusive_allocated_bytes':sum(x['allocated'] for x in leaves),'groups':dict(group),'excluded':len(excluded),'exclusion_reasons':dict(collections.Counter(x['reason'] for x in excluded)),'record_sha_count':len(records),'fresh_source_config_sha_count':len(c['source_jsons']),'current_closure_path':str(closure),'current_closure_sha256':sha(closure),'previous762_overlap':len({x['path'] for x in leaves}&deleted_paths),'held_families':data['held_families'],'source':{'path':str(input_path),'sha256':sha(input_path)},'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir()},'loss':'Original historical weights and resume state lost; cannot promise bitwise retrain. Keep final fields, explicitly selected endpoints, code/config/reports/metrics/logs/figures. Old hashes/provenance stay recorded; old validators may report retired input rather than complete revalidation.','bounds':'Selected current runtime/chosen coldbuild closure only, not all cross-project or future consumers. Final execution needs fresh dependency identities, record checks and Eden/pakka visible process/FD review. This audit authorizes no deletion.','sam_pilot':'PAUSED_NOT_STARTED'}
write(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
