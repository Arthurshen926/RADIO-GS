"""Durable serial GPU5 evaluation queue for asynchronously rebuilt four scenes."""
import json,os,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19');marker=root/'official_evaluation_queue.json';assert not marker.exists();scenes=['figurines','ramen','teatime','waldo_kitchen'];record=dict(complete=False,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),stages=[],states={s:'waiting_for_rebuild' for s in scenes});assert record['gpu_uuid']=='GPU-733bbc56-b192-f82f-b17b-e2e0a0fc06ca'
def save():record['updated_unix']=time.time();marker.write_text(json.dumps(record,indent=2)+'\n')
def run(scene,kind,entry,args):
 cmd=[sys.executable,entry,*args];row=dict(scene=scene,kind=kind,command=cmd,status='running');record['stages'].append(row);save()
 with (root/scene/(kind+'_queue.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 row.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode);save();return r.returncode==0
save()
while any(state=='waiting_for_rebuild' for state in record['states'].values()):
 progressed=False
 for scene in scenes:
  if record['states'][scene]!='waiting_for_rebuild':continue
  p=root/scene/'execution.json'
  if not p.exists():continue
  try:build=json.loads(p.read_text())
  except json.JSONDecodeError:continue
  if not build.get('complete'):continue
  record['states'][scene]='evaluating';save();progressed=True
  good=run(scene,'official_evaluation','scripts/audits/evaluate_full_field_rebuild_next19.py',['--scene',scene,'--device','cuda:0','--with-structure-graph'])
  if good:good=run(scene,'lineage_audit','scripts/audits/audit_full_field_rebuild_lineage_next19.py',['--scene',scene])
  record['states'][scene]='complete' if good else 'failed';save()
 if not progressed:time.sleep(30)
if all(s=='complete' for s in record['states'].values()):
 cmd=[sys.executable,'scripts/audits/evaluate_full_field_rebuild_next19.py','--summarize'];r=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'));assert r.returncode==0;record['complete']=True;record['summary_sha256']=sha256_file(root/'evaluation_summary.json');record['lineage_sha256']={s:sha256_file(root/s/'lineage_audit.json') for s in scenes};save()
else:record['needs_recovery']=True;save()
print(json.dumps(record,indent=2),flush=True)
