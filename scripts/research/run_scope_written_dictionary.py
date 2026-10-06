import argparse,json,os,queue,subprocess,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def main():
 p=argparse.ArgumentParser();p.add_argument('--local',action='store_true');p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args();python='/root/miniconda3/envs/cybersim_agent/bin/python';base=Path('output/scope_written_'+('local' if a.local else 'matched')+'_next38');base.mkdir(parents=True,exist_ok=True)
 keys=['lerf_'+s for s in ['figurines','ramen','teatime','waldo_kitchen']]
 if not a.local:keys+=['scannet_scene'+s+'_00' for s in ['0000','0062','0070','0097','0140','0347','0400','0590']]
 work=queue.Queue();results=[];lock=threading.Lock()
 for key in keys:work.put(key)
 def run(gpu):
  while True:
   try:key=work.get_nowait()
   except queue.Empty:return
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
   commands=[[python,'scripts/research/build_scope_written_dictionary.py','--key',key]+(['--local'] if a.local else []),[python,'scripts/research/calibrate_scope_dictionary.py','--key',key,'--base',str(base)],[python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline','--calibrated']]
   with (base/(key+'.log')).open('w') as log:
    for command in commands:
     rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
     if rc:break
   with lock:
    row=dict(key=key,exit_code=rc);results.append(row);(base/'execution.json').write_text(json.dumps(results,indent=2)+'\n');print(row,flush=True)
 with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:list(pool.map(run,a.gpus.split(',')))


if __name__=='__main__':main()
