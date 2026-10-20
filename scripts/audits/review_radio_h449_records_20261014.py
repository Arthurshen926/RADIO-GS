"""Content-review H449 preserved result records; proposal only, no payload reads."""
import json,pathlib,gzip,hashlib,re,math,collections,os,time
D=pathlib.Path('docs/experiments/2026-10-14-radio-h449-payload-plan-1635')
S=json.loads((D/'summary.json').read_text());leaves=json.load(gzip.open(D/'closed_payload_candidates.json.gz','rt'))
roots={r['root'] for r in leaves};blocked={'scannet_cold_completion','compiled_native_observations','adapter_observation_control'}
reviews={};readbytes=0;MAX=64*1024**2
numre=re.compile(r'iou|accuracy|precision|recall|cosine|psnr|rmse|(^|/)ap$|mean_ap|score|loss|f1',re.I)
confre=re.compile(r'weight|threshold|config|hyperparam|learning_rate|loss_type|loss_mode',re.I)
for root in sorted(roots):
 if root in blocked:reviews[root]={'status':'HOLD_CURRENT_RUNTIME_INPUT_IDENTIFIED','records':[]};continue
 paths=sorted(S['roots'][root]['explicit_result_records'],key=lambda x:(0 if pathlib.Path(x['path']).name in ('summary.json','comparison.json','results.json','final_comparison.json','summary.md','report.md') else 1,len(pathlib.Path(x['path']).parts),x['bytes']))
 found=[]
 for x in paths[:24]:
  p=pathlib.Path(x['path']);size=p.stat().st_size
  if size>2*1024**2 or readbytes+size>MAX:continue
  b=p.read_bytes();readbytes+=len(b);nums=[];dec=[]
  if p.suffix=='.json':
   try:obj=json.loads(b)
   except ValueError:continue
   def walk(v,k=''):
    if isinstance(v,dict):
     for a,z in v.items():walk(z,k+'/'+str(a))
    elif isinstance(v,list):
     for i,z in enumerate(v[:10000]):walk(z,k+'/'+str(i))
    elif isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and numre.search(k) and not confre.search(k):nums.append({'key':k,'value':v})
    elif isinstance(v,(str,bool)) and re.search('decision|conclusion|promot|default_changed|development_only|reject|recommend',k,re.I):dec.append({'key':k,'value':v})
   walk(obj)
  else:
   t=b.decode('utf8',errors='replace')
   m=re.search(r'IoU|mIoU|accuracy|cosine|loss|conclusion|decision|结论|结果|拒绝|失败|retire',t,re.I)
   if len(t)>64 and m and re.search(r'\d',t):dec=[{'key':'text_result_excerpt','value':t[max(0,m.start()-200):m.start()+1600]}]
  # Explicit decision flags alone can be config, so require quantitative or result text.
  if nums or any(z['key']=='text_result_excerpt' for z in dec):
   found.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'quantitative_excerpt':nums[:16],'decision_excerpt':dec[:12]})
   if len(found)>=2:break
 reviews[root]={'status':'PRESERVED_QUANTITATIVE_OR_TEXT_RESULT_VERIFIED' if found else 'HOLD_RESULT_CONTENT_NOT_ESTABLISHED','records':found}
verified={n for n,x in reviews.items() if x['status']=='PRESERVED_QUANTITATIVE_OR_TEXT_RESULT_VERIFIED'}
g=collections.defaultdict(list)
for r in leaves:
 if r['root'] in verified:g[(r['dev'],r['ino'])].append(r)
outrows=[]
for rs in g.values():
 if len(rs)==rs[0]['nlink']:outrows.extend(rs)
seen=set();by=collections.defaultdict(lambda:{'leaves':0,'joint_inode_attribution_bytes':0})
for r in outrows:
 s=os.stat(r['path'],follow_symlinks=False)
 if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)!=(r['dev'],r['ino'],r['size'],r['mtime_ns'],r['ctime_ns'],r['nlink']):raise RuntimeError('candidate identity changed')
 by[r['root']]['leaves']+=1;k=(r['dev'],r['ino'])
 if k not in seen:seen.add(k);by[r['root']]['joint_inode_attribution_bytes']+=r['allocated']
with (D/'content_review.json').open('x') as f:json.dump({'status':'RESULT_CONTENT_REVIEWED_CURRENT_T0_C0_DEPENDENCY_PASS_PENDING','reviews':reviews,'small_record_bytes_read':readbytes,'current_roots_already_held':sorted(blocked)},f,indent=2)
with (D/'content_reviewed_leaves.json.gz').open('xb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(json.dumps(outrows,separators=(',',':')).encode())
total=sum(x['joint_inode_attribution_bytes'] for x in by.values())
with (D/'content_reviewed_summary.json').open('x') as f:json.dump({'status':'NOT_YET_READY_FOR_APPROVAL_CURRENT_INPUT_REVIEW_PENDING','root_count':len(by),'leaf_count':len(outrows),'unique_inodes':len(seen),'joint_exclusive_allocated_bytes':total,'roots':dict(by),'small_record_bytes_read':readbytes,'held_content_roots':[n for n,x in reviews.items() if x['status']=='HOLD_RESULT_CONTENT_NOT_ESTABLISHED'],'current_roots_held':sorted(blocked)},f,indent=2)
print(json.dumps({'root_count':len(by),'leaves':len(outrows),'unique_inodes':len(seen),'exclusive_bytes':total,'record_bytes_read':readbytes,'held_content_roots':[n for n,x in reviews.items() if x['status']=='HOLD_RESULT_CONTENT_NOT_ESTABLISHED']}))
print('TOP20',json.dumps(sorted([{'root':n,**x} for n,x in by.items()],key=lambda r:-r['joint_inode_attribution_bytes'])[:20]))
