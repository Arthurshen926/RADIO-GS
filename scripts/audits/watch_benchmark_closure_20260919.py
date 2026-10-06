"""Persist worker liveness and summarize when both authorized queues stop."""
import json,os
from pathlib import Path
import subprocess,sys,time

root=Path('output/benchmark_closure_20260919')
while True:
    rows=[]
    for scene in ['teatime','waldo_kitchen']:
        path=root/scene/'execution.json'
        if not path.exists():rows.append(dict(scene=scene,alive=False,missing_receipt=True));continue
        d=json.loads(path.read_text());rows.append(dict(scene=scene,pid=d['pid'],alive=Path(f"/proc/{d['pid']}").exists(),state=d['state'],complete=d['complete'],failures=d['failures']))
    temp=root/'liveness.tmp';temp.write_text(json.dumps(dict(watcher_pid=os.getpid(),unix_time=time.time(),workers=rows),indent=2)+'\n');temp.replace(root/'liveness.json')
    if not any(r['alive'] for r in rows):
        subprocess.run([sys.executable,'scripts/audits/summarize_benchmark_closure_20260919.py','--output',str(root/'completed_queue_summary.json')],check=True)
        break
    time.sleep(30)
