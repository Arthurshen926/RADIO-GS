"""Serial GPU1 closure of already-running teatime and full waldo rebuild."""
import json,os,subprocess,sys,time
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/full_field_rebuild_next19');receipt=root/'teatime_waldo_queue.json';assert not receipt.exists()
r=dict(started=time.time(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),initial_teatime_pid=2434500,stages=[],complete=False)
def save():receipt.write_text(json.dumps(r,indent=2)+'\n')
save()
while Path('/proc/2434500/cmdline').exists() and b'run_full_field_rebuild_next19.py' in Path('/proc/2434500/cmdline').read_bytes():time.sleep(10)
assert json.loads((root/'teatime/execution.json').read_text())['complete'],'teatime initial runner failed; inspect original attempts'
commands=[('teatime_evaluation','scripts/audits/evaluate_full_field_rebuild_next19.py',['--scene','teatime','--device','cuda:0','--with-structure-graph']),('waldo_rebuild','scripts/audits/run_full_field_rebuild_next19.py',['--scene','waldo_kitchen']),('waldo_evaluation','scripts/audits/evaluate_full_field_rebuild_next19.py',['--scene','waldo_kitchen','--device','cuda:0','--with-structure-graph'])]
for name,entry,args in commands:
 cmd=[sys.executable,entry,*args];row=dict(name=name,command=cmd,status='running',implementation_sha256=sha256_file(entry),start=time.time());r['stages'].append(row);save()
 with (root/(name+'_queue.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4'))
 row.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode,seconds=time.time()-row['start']);save();assert p.returncode==0,name+' failed'
r['complete']=True;r['seconds']=time.time()-r['started'];save();print('TEATIME AND WALDO COMPLETE')
