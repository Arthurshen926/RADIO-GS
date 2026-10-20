#!/usr/bin/env python3
"""Exact approved 762-leaf retirement. Audit-only preflight; explicit execute/post phases."""
import argparse, datetime, gzip, hashlib, json, os, stat, time
from pathlib import Path
BASE=Path('/mnt/pool/sqy/results/RADIO-GS/output')
REPO=Path('/root/RADIO-GS-query-field')
DOCS=REPO/'docs/experiments'
PLAN=DOCS/'2026-10-15-radio-three-group-retirement-proposal-0303'
MANIFEST=PLAN/'frozen_762_leaves.json.gz'
APPROVED_SHA='cfe6c785e406cae304f601b8bf974484453f05c3d4f6d7024896997be9ff0dc6'
RUN=DOCS/'2026-10-15-radio-762-retirement-execution-0335'
FIELDS=('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')
TEXT={'.json','.jsonl','.md','.txt','.log','.yaml','.yml','.csv','.tsv','.toml','.sh','.py'}
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,data):
 with Path(path).open('x') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
def zipwrite(path,data):
 with Path(path).open('xb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(data,separators=(',',':')).encode())
def rec(p,follow=False):
 s=os.stat(p) if follow else os.lstat(p)
 d={k:getattr(s,'st_'+k) for k in FIELDS};d['allocated']=s.st_blocks*512
 if not follow and stat.S_ISLNK(s.st_mode):d['link']=os.readlink(p)
 return d
def match(actual,expected):
 return all(actual[k]==expected[k] for k in FIELDS) and actual['allocated']==expected['allocated']
def load():
 assert sha(MANIFEST)==APPROVED_SHA,'approved manifest SHA mismatch'
 d=json.load(gzip.open(MANIFEST,'rt'));xs=d['leaves'];assert len(xs)==762
 assert len({x['path'] for x in xs})==len({(x['dev'],x['ino']) for x in xs})==762
 for x in xs:
  p=Path(x['path']);assert p.is_absolute() and '..' not in p.parts and str(p).startswith(str(BASE)+'/')
  assert stat.S_ISREG(x['mode']) and x['nlink']==1
 return xs,json.loads((PLAN/'proposal_summary.json').read_text())
def root_for(p):
 rel=Path(p).relative_to(BASE).parts
 return str(BASE/rel[0]) if rel[0]=='unified_query' else str(BASE/rel[0]/rel[1]) if rel[0]=='radio_gs' else (_ for _ in ()).throw(ValueError(p))
def canonical(s):
 for old in ('/root/RADIO-GS-query-field/output','/root/RADIO-GS/output'):
  s=s.replace(old,str(BASE))
 return s
def proc_scan(xs):
 roots=sorted({root_for(x['path']) for x in xs});inos={(x['dev'],x['ino']) for x in xs}
 result={'started_utc':utc(),'scope':'visible Eden /proc namespace only','pids':0,'fds':0,'maps_lines':0,'matches':[],'errors':[],'vanished':0}
 def hit(pid,kind,s,path=None):
  c=canonical(s);rs=[r for r in roots if c==r or r+'/' in c or r+'\\x00' in c or (kind=='cmdline' and r in c)]
  inode=False
  if path:
   try:
    st=os.stat(path);inode=(st.st_dev,st.st_ino) in inos
   except FileNotFoundError:pass
   except PermissionError as e:result['errors'].append({'pid':pid,'kind':kind,'error':str(e)})
  if rs or inode:result['matches'].append({'pid':pid,'kind':kind,'roots':rs,'candidate_inode':inode,'relevant_path':s[:1200]})
 for de in os.scandir('/proc'):
  if not de.name.isdigit() or int(de.name)==os.getpid():continue
  pid=int(de.name);result['pids']+=1;p=Path(de.path)
  for kind in ('cmdline','cwd','maps','fd'):
   try:
    if kind=='cmdline':hit(pid,kind,(p/kind).read_bytes().decode(errors='replace'))
    elif kind=='cwd':hit(pid,kind,os.readlink(p/kind),str(p/kind))
    elif kind=='maps':
     for line in (p/kind).read_text(errors='replace').splitlines():
      result['maps_lines']+=1
      if '/' in line:hit(pid,kind,line.split(None,5)[-1])
    else:
     for fd in os.scandir(p/kind):
      try:hit(pid,kind,os.readlink(fd.path),fd.path);result['fds']+=1
      except FileNotFoundError:result['vanished']+=1
   except FileNotFoundError:result['vanished']+=1
   except (PermissionError,OSError) as e:result['errors'].append({'pid':pid,'kind':kind,'error':str(e)})
 result['completed_utc']=utc();return result
def protected_snapshot(xs,summary):
 candidates={(x['dev'],x['ino']) for x in xs};paths={x['path'] for x in xs}
 closure_path=DOCS/'2026-10-15-selected-closure-readonly.json'
 assert sha(closure_path)=='bd55915776649ad61abc0ce95c50d696e52480c14524eeee6283085ac88d643d'
 c=json.loads(closure_path.read_text())
 cutoff=datetime.datetime.fromisoformat(c['created_utc'].replace('Z','+00:00')).timestamp()*1e9
 protected={};issues=[];intersections=set()
 for entry in c['entries']:
  p=entry['path']
  if p in protected:continue
  try:
   value=rec(p);value['realpath']=os.path.realpath(p)
   st=rec(p,True);value['target']=st
   if value['realpath'] in paths or (st['dev'],st['ino']) in candidates:
    intersections.update(x['path'] for x in xs if x['path']==value['realpath'] or (x['dev'],x['ino'])==(st['dev'],st['ino']))
   if entry.get('exists') and (st['dev'],st['ino'])!=(entry.get('dev'),entry.get('ino')):issues.append({'path':p,'reason':'protected inode changed since dependency audit'})
   protected[p]=value
  except FileNotFoundError:
   protected[p]={'missing':True}
   if entry.get('exists'):issues.append({'path':p,'reason':'newly missing protected dependency'})
 for p in c['source_jsons']:
  try:
   st=rec(p,True)
   if max(st['mtime_ns'],st['ctime_ns'])>cutoff:issues.append({'path':p,'reason':'source JSON newer than closure audit'})
   if p not in protected:protected[p]=rec(p)
   protected[p]['sha256']=sha(p)
  except FileNotFoundError:issues.append({'path':p,'reason':'source JSON missing'})
 for x in summary['replacement_guard23']:
  p=x['package_path'];now=rec(p);prior=x['package_fingerprint']
  assert all(now[k]==prior[k] for k in FIELDS),(p,'replacement identity changed')
  assert now['allocated']==x['package_settled_allocated_bytes']
  protected[p]=now
  for key,hashkey in [('config_path','config_sha256')]:
   q=x[key];st=rec(q);assert sha(q)==x[hashkey],(q,'replacement config SHA')
   st['sha256']=x[hashkey];protected[q]=st
  q=x['receipt'];st=rec(q);st['sha256']=sha(q);protected[q]=st
 return protected,issues,sorted(intersections)
def retained_snapshot(xs):
 leaves={x['path'] for x in xs};saved={};errors=[]
 for root in sorted({root_for(x['path']) for x in xs}):
  for parent,dirs,files in os.walk(root,followlinks=False):
   for name in dirs+files:
    p=str(Path(parent)/name)
    if p in leaves:continue
    try:
     s=rec(p)
     if stat.S_ISDIR(s['mode']):continue
     if stat.S_ISREG(s['mode']) and Path(p).suffix.lower() in TEXT and s['size']<=2097152:s['sha256']=sha(p)
     saved[p]=s
    except OSError as e:errors.append({'path':p,'error':str(e)})
 return saved,errors
def df():
 s=os.statvfs(BASE);return {'utc':utc(),'available':s.f_bavail*s.f_frsize,'free':s.f_bfree*s.f_frsize,'capacity':s.f_blocks*s.f_frsize}
def preflight():
 xs,summary=load();RUN.mkdir(exist_ok=False);changed=[]
 for x in xs:
  try:
   if not match(rec(x['path']),x):changed.append({'path':x['path'],'reason':'identity mismatch'})
  except OSError as e:changed.append({'path':x['path'],'reason':str(e)})
 protected,issues,intersections=protected_snapshot(xs,summary)
 retained,errors=retained_snapshot(xs)
 scan=proc_scan(xs)
 result={'created_utc':utc(),'manifest_sha256':APPROVED_SHA,'manifest_count':len(xs),'changed_candidates':changed,'dependency_issues':issues,'protected_intersections':intersections,'retained_errors':errors,'eden_process_scan':scan,'df_before':df(),'protected_count':len(protected),'retained_count':len(retained)}
 zipwrite(RUN/'protected_preflight.json.gz',protected);zipwrite(RUN/'retained_preflight.json.gz',retained)
 write(RUN/'preflight.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='eden_process_scan'}));print('EDEN_SCAN',json.dumps(scan));return result
def secure_parent(path):
 p=Path(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for piece in p.parent.parts[1:]:
   n=os.open(piece,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return fd,p.name
 except BaseException:os.close(fd);raise
def execute():
 xs,summary=load();pre=json.loads((RUN/'preflight.json').read_text())
 assert not pre['dependency_issues'] and not pre['retained_errors'],'preflight dependency/record issue'
 pakka=json.loads((RUN/'pakka_writer_check.json').read_text())
 assert pakka['ready_for_execution'] and not pakka['matches'] and not pakka['permission_errors']
 assert time.time()<pakka['valid_until_epoch'],'pakka writer snapshot expired'
 scan=proc_scan(xs);write(RUN/'eden_immediate_process_scan.json',scan);assert not scan['errors'],'Eden visibility errors'
 blocked_roots={r for m in scan['matches'] for r in m['roots']}|set(pakka.get('blocked_roots',[]))
 assert not any(m['candidate_inode'] and not m['roots'] for m in scan['matches']),'unmapped open candidate inode'
 skip={x['path']:x['reason'] for x in pre['changed_candidates']}
 skip.update({p:'current protected dependency' for p in pre['protected_intersections']})
 for x in xs:
  if root_for(x['path']) in blocked_roots:skip[x['path']]='active visible process root'
 # Fresh source JSON SHA and replacement identity recheck immediately before any unlink.
 protected=json.load(gzip.open(RUN/'protected_preflight.json.gz','rt'))
 for p,s in protected.items():
  if s.get('missing'):continue
  now=rec(p)
  assert all(now[k]==s[k] for k in FIELDS),(p,'protected input changed after preflight')
  if 'sha256' in s:assert sha(p)==s['sha256'],(p,'protected source changed after preflight')
 receipt=RUN/'leaf_receipts.jsonl';intent=RUN/'unlink_intents.jsonl'
 assert not receipt.exists() and not intent.exists(),'never blindly rerun execution'
 t=time.monotonic();deleted=0;skipped=0;allocated=0
 with receipt.open('x') as rf,intent.open('x') as inf:
  def emit(f,d):f.write(json.dumps(d,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
  for i,x in enumerate(xs):
   p=x['path']
   if time.monotonic()-t>300 or time.time()>=pakka['valid_until_epoch']:raise RuntimeError('bounded execution stopped; reconcile receipts before continuation')
   if p in skip:emit(rf,{'utc':utc(),'path':p,'status':'SKIPPED','reason':skip[p]});skipped+=1;continue
   fd=None;leaf_fd=None
   try:
    fd,name=secure_parent(p)
    leaf_fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    s=os.fstat(leaf_fd);now={k:getattr(s,'st_'+k) for k in FIELDS};now['allocated']=s.st_blocks*512
    if not match(now,x):
     emit(rf,{'utc':utc(),'path':p,'status':'SKIPPED','reason':'final no-follow identity changed','observed':now});skipped+=1;continue
    q=os.stat(name,dir_fd=fd,follow_symlinks=False)
    if (q.st_dev,q.st_ino,q.st_ctime_ns)!=(s.st_dev,s.st_ino,s.st_ctime_ns):
     emit(rf,{'utc':utc(),'path':p,'status':'SKIPPED','reason':'last identity race'});skipped+=1;continue
    emit(inf,{'utc':utc(),'path':p,'status':'UNLINK_INTENT','identity':x})
    os.unlink(name,dir_fd=fd)
    emit(rf,{'utc':utc(),'path':p,'status':'UNLINKED','dev':s.st_dev,'ino':s.st_ino,'allocated':s.st_blocks*512})
    deleted+=1;allocated+=s.st_blocks*512
   except (FileNotFoundError,PermissionError,OSError) as e:
    emit(rf,{'utc':utc(),'path':p,'status':'ERROR_STOP','error':repr(e)});raise
   finally:
    if leaf_fd is not None:os.close(leaf_fd)
    if fd is not None:os.close(fd)
   if (i+1)%100==0:print(json.dumps({'processed':i+1,'deleted':deleted,'skipped':skipped}),flush=True)
 write(RUN/'execution_result.json',{'utc':utc(),'deleted':deleted,'skipped':skipped,'closed_last_link_allocated':allocated,'seconds':time.monotonic()-t,'df_after_unlink':df()})
 print('EXECUTED',deleted,skipped,allocated,flush=True)
def post():
 xs,summary=load();receipts=[json.loads(l) for l in (RUN/'leaf_receipts.jsonl').read_text().splitlines()];assert len({r['path'] for r in receipts})==len(receipts)
 by={r['path']:r for r in receipts};errors=[];deleted=skipped=allocated=0
 for x in xs:
  r=by.get(x['path']);exists=os.path.lexists(x['path'])
  if r and r['status']=='UNLINKED' and not exists:deleted+=1;allocated+=x['allocated']
  elif r and r['status']=='SKIPPED' and exists:skipped+=1
  else:errors.append({'path':x['path'],'receipt':r,'exists':exists})
 checks={}
 for name in ['protected','retained']:
  saved=json.load(gzip.open(RUN/(name+'_preflight.json.gz'),'rt'));bad=[]
  for p,s in saved.items():
   if s.get('missing'):
    if os.path.exists(p):bad.append({'path':p,'reason':'previously missing became present'})
    continue
   try:
    n=rec(p)
    if not all(n[k]==s[k] for k in FIELDS):bad.append({'path':p,'reason':'identity changed'})
    if 'sha256' in s and sha(p)!=s['sha256']:bad.append({'path':p,'reason':'record SHA changed'})
   except OSError as e:bad.append({'path':p,'reason':str(e)})
  checks[name]={'count':len(saved),'errors':bad}
 pre=json.loads((RUN/'preflight.json').read_text());after=df()
 result={'status':'VERIFIED' if not errors and not any(v['errors'] for v in checks.values()) else 'POSTCHECK_ISSUE','completed_utc':utc(),'manifest_sha256':APPROVED_SHA,'approved':762,'verified_deleted':deleted,'verified_skipped':skipped,'closed_last_link_allocated_bytes':allocated,'leaf_errors':errors,'preservation':checks,'df_before':pre['df_before'],'df_after':after,'df_available_delta':after['available']-pre['df_before']['available'],'limits':'Shared-pool df includes unrelated writers; process visibility limited to Eden/pakka containers. Historical state retired; known174 SAM inputs not restored.','sam_pilot':'PAUSED_NOT_STARTED'}
 write(RUN/'completion.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','execute','post']);a=p.parse_args()
 {'preflight':preflight,'execute':execute,'post':post}[a.phase]()
