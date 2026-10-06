"""Authorized GPU5 rebuild scheduling; resumes only the waiting evaluation queue."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-733bbc56-b192-f82f-b17b-e2e0a0fc06ca'
queue_pid=2440367
assert b'queue_full_field_rebuild_evaluation_next19.py' in Path(f'/proc/{queue_pid}/cmdline').read_bytes()
root=Path('output/full_field_rebuild_next19'); queue=json.loads((root/'official_evaluation_queue.json').read_text())
assert not queue['stages'] and all(v=='waiting_for_rebuild' for v in queue['states'].values())
receipt=root/'waldo_full_rebuild_gpu5.json';assert not receipt.exists()
entry='scripts/audits/run_full_field_rebuild_next19.py';cmd=[sys.executable,entry,'--scene','waldo_kitchen']
r=dict(complete=False,pid=os.getpid(),queue_pid=queue_pid,queue_paused=False,started=time.time(),command=cmd,implementation_sha256=sha256_file(entry))
def save():receipt.write_text(json.dumps(r,indent=2)+'\n')
save();os.kill(queue_pid,signal.SIGSTOP);r['queue_paused']=True;save()
try:
 result=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 r.update(returncode=result.returncode,complete=result.returncode==0)
finally:
 os.kill(queue_pid,signal.SIGCONT);r.update(queue_paused=False,finished=time.time());save()
assert r['complete'],r
