"""Authorized idle-GPU preprepare, with durable queue pause/resume receipt."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-733bbc56-b192-f82f-b17b-e2e0a0fc06ca';queuepid=2440367;cmdline=Path(f'/proc/{queuepid}/cmdline').read_bytes();assert b'queue_full_field_rebuild_evaluation_next19.py' in cmdline;top=Path('output/full_field_rebuild_next19');q=json.loads((top/'official_evaluation_queue.json').read_text());
for check in top.glob('*/execution.json'):
 assert not json.loads(check.read_text()).get('complete',False),'Prioritize ready evaluation over preprepare'
assert all(s=='waiting_for_rebuild' for s in q['states'].values());root=top/'ramen';root.mkdir(exist_ok=True);marker=root/'execution.json';assert not marker.exists();control=top/'ramen_preprepare_gpu5.json';assert not control.exists();state=dict(queue_pid=queuepid,queue_paused=False,complete=False,started=time.time());control.write_text(json.dumps(state,indent=2)+'\n');os.kill(queuepid,signal.SIGSTOP);state['queue_paused']=True;control.write_text(json.dumps(state,indent=2)+'\n')
try:
 gp=Path('output/depth_opacity_repair_20260918/ramen/geometry_report.json');entry='scripts/audits/prepare_full_field_rebuild_next19.py';command=[sys.executable,entry,'--scene','ramen'];row=dict(name='prepare',command=command,status='running',implementation_sha256=sha256_file(entry));record=dict(scene='ramen',complete=False,geometry_report_sha256=sha256_file(gp),full_source_asset_and_field_rebuild=True,stages=[row]);marker.write_text(json.dumps(record,indent=2)+'\n')
 with (root/'prepare.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 row.update(returncode=r.returncode,status='complete' if r.returncode==0 else 'failed')
 if r.returncode==0:row['outputs']={str(p):sha256_file(p) for p in [root/'base/source_plan.json',root/'base/source_sam_execution.json']}
 marker.write_text(json.dumps(record,indent=2)+'\n');state['complete']=r.returncode==0;state['returncode']=r.returncode;assert r.returncode==0
finally:
 os.kill(queuepid,signal.SIGCONT);state['queue_paused']=False;state['finished']=time.time();control.write_text(json.dumps(state,indent=2)+'\n')
print(state)
