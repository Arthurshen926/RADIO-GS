"""Run independent source-field reconstructions and the frozen text scorer."""
import argparse,json,os,queue,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--keys',required=True);p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--base',default='output/pipeline_kernel_extent_next40');p.add_argument('--evaluate-only',action='store_true');p.add_argument('--carrier',choices=['elements','scopes'],default='elements');p.add_argument('--evidence-policy',choices=['observations','consistent_views'],default='observations');p.add_argument('--readout',choices=['probability','odds'],default='odds');p.add_argument('--prior-domain',choices=['known','visible'],default='known');a=p.parse_args()
    base=Path(a.base);base.mkdir(parents=True,exist_ok=True);jobs=queue.Queue()
    for key in a.keys.split(','):jobs.put(key)
    def run(gpu):
        results=[]
        while True:
            try:key=jobs.get_nowait()
            except queue.Empty:break
            out=base/key;out.mkdir(parents=True,exist_ok=True);commands=[]
            if not a.evaluate_only:
                commands.append(['scripts/research/fit_source_kernel_extent.py','--key',key,'--baseline','pipeline','--save-training','--carrier',a.carrier,'--evidence-policy',a.evidence_policy,'--readout',a.readout,'--prior-domain',a.prior_domain,'--output',str(base)])
            commands.append(['scripts/research/evaluate_scope_dictionary.py','--key',key,'--base',str(base),'--readout','anchored','--pipeline'])
            with (out/'execution.log').open('w') as log:
                for command in commands:
                    rc=subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python']+command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT).returncode
                    if rc:break
            result=dict(key=key,exit_code=rc);results.append(result);print(result,flush=True)
        return results
    with ThreadPoolExecutor(len(a.gpus.split(','))) as pool:results=sum(pool.map(run,a.gpus.split(',')),[])
    (base/('execution_'+'_'.join(a.gpus.split(','))+'.json')).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
