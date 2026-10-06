"""Full fixed-cohort controls for partial evidence and conditional transfer."""
import argparse,json,os,queue,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args()
    jobs=queue.Queue()
    for path in sorted(Path('output/odds_kernel_extent_next40').glob('*/field.pt')):jobs.put(path.parent.name)
    def run(gpu):
        rows=[]
        while True:
            try:key=jobs.get_nowait()
            except queue.Empty:break
            for mode in ['reselected','conditional']:
                base=Path('output/partial_extent_next41')/mode;out=base/key;out.mkdir(parents=True,exist_ok=True)
                fit=(['scripts/research/reselect_partial_extent.py','--key',key,'--output',str(base)] if mode=='reselected' else
                     ['scripts/research/fit_source_kernel_extent.py','--key',key,'--output',str(base),'--baseline','pipeline','--readout','odds','--negative-policy','disjoint_regions','--interpolation','conditional_evidence','--save-training'])
                with (out/'execution.log').open('w') as log:
                    for command in [fit,['scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline']]:
                        rc=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python']+command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT).returncode
                        if rc:break
                row=dict(key=key,mode=mode,exit_code=rc);rows.append(row);print(row,flush=True)
        return rows
    with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:rows=sum(pool.map(run,a.gpus.split(',')),[])
    Path('output/partial_extent_next41/execution.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
