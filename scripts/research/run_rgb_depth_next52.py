import os,json,subprocess,argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
parser=argparse.ArgumentParser();parser.add_argument('--arm',default='fit');parser.add_argument('--gpus',default='0,1,2,3');parser.add_argument('--scenes',default='figurines,ramen,teatime,waldo_kitchen');parser.add_argument('--ray-distribution',action='store_true');parser.add_argument('--recover-calibration',action='store_true');args=parser.parse_args();arm=args.arm;gpus=args.gpus.split(',')
def run(job):
    gpu,scene=job;key='lerf_'+scene;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH='.')
    root=Path('output/ray_selection_next52' if args.ray_distribution else 'output/depth_selection_next52_recovered' if args.recover_calibration else 'output/depth_selection_next52');root.mkdir(parents=True,exist_ok=True)
    commands=[[PYTHON,'scripts/research/reconstruct_rgb_depth_next52.py','--scene',scene,'--steps','1500']+(['--recover-calibration'] if args.recover_calibration else [])+(['--ray-distribution'] if args.ray_distribution else []),[PYTHON,'scripts/research/decode_depth_selection_next52.py','--key',key,'--arm',arm,'--geometry-root','output/carrier_ray_next52' if args.ray_distribution else 'output/carrier_depth_next52_recovered' if args.recover_calibration else 'output/carrier_depth_next52','--output-root',str(root)],[PYTHON,'scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(root/arm),'--readout','anchored','--pipeline','--calibrated','--element-results',str(root/arm/key/'typed_readout.pt')]];rows=[]
    with (root/(scene+'_'+arm+'.log')).open('w') as log:
        for c in commands:
            rc=subprocess.run(c,env=env,stdout=log,stderr=subprocess.STDOUT).returncode;rows.append(dict(scene=scene,command=c,exit_code=rc));print(scene,c[1],rc,flush=True)
            if rc:break
    return rows
rows=[];scenes=args.scenes.split(',')
for start in range(0,len(scenes),len(gpus)):
    with ThreadPoolExecutor(len(gpus)) as pool:rows+=sum(pool.map(run,[(gpus[i],s) for i,s in enumerate(scenes[start:start+len(gpus)])]),[])
(Path('output/ray_selection_next52' if args.ray_distribution else 'output/depth_selection_next52_recovered' if args.recover_calibration else 'output/depth_selection_next52')/('execution_'+arm+'_'+'_'.join(scenes)+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
