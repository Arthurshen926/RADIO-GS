"""Fixed-input composition controls over existing reconstructed scopes."""
import argparse,json,os,queue,shutil,subprocess,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['matched','local','native'],required=True);p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args();python='/root/miniconda3/envs/cybersim_agent/bin/python'
 keys=['lerf_'+s for s in ['figurines','ramen','teatime','waldo_kitchen']]
 if a.mode!='local':keys+=['scannet_scene'+s+'_00' for s in ['0000','0062','0070','0097','0140','0347','0400','0590']]+['nvos_'+s for s in ['fern','flower','fortress','horns_center','leaves','orchids','trex']]
 base=Path('output/scope_composition_'+a.mode+'_next38');base.mkdir(parents=True,exist_ok=True)
 for key in keys:
  source=Path('output/local_rendered_scopes_next37' if a.mode=='local' else 'output/rendered_scopes_next37')/key
  data=torch.load(source/'field.pt',weights_only=False);data['scope_readout']='composition';data['scope_modalities']=['text','image'];data['scope_after_prompt']=False
  data['composition_fixed_cores']=a.mode=='native'
  if a.mode=='native' and not key.startswith('nvos_'):
   data['scope_native_dictionary']=True;data['native_modalities']=['text']
  out=base/key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt');shutil.copyfile(source/'construction.json',out/'construction.json')
  if a.mode=='native' and not key.startswith('nvos_'):shutil.copyfile(Path('output/scope_query_modes_next34')/key/'calibration.pt',out/'calibration.pt')
 work=queue.Queue();results=[];lock=threading.Lock()
 for key in keys:work.put(key)
 def run(gpu):
  while True:
   try:key=work.get_nowait()
   except queue.Empty:return
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
   if key.startswith('nvos_'):
    commands=[[python,'scripts/research/evaluate_nvos_scope_field.py','--scene',s,'--base',str(base)] for s in [key[5:]]+(['horns_left'] if key=='nvos_horns_center' else [])]
   else:
    commands=[[python,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline']+(['--calibrated'] if a.mode=='native' else [])]
    if key.startswith('lerf_'):commands.append([python,'scripts/research/evaluate_scope_reference_images.py','--key',key,'--base',str(base)])
   started=time.time()
   with (base/key/'execution.log').open('w') as log:
    for command in commands:
     rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
     if rc:break
   with lock:
    row=dict(key=key,exit_code=rc,seconds=time.time()-started);results.append(row);(base/'execution.json').write_text(json.dumps(results,indent=2)+'\n');print(row,flush=True)
 with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:list(pool.map(run,a.gpus.split(',')))


if __name__=='__main__':main()
