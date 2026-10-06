"""Run explicit category Scene files through the public entry and frozen evaluator."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--output',required=True);p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args()
    root=Path(a.output);gpus=a.gpus.split(',');jobs=sorted(Path(a.configs).glob('*/*.scene.json'))
    def worker(i):
        for path in jobs[i::len(gpus)]:
            mode=path.parent.name;name=path.name.removesuffix('.scene.json');out=root/mode/name;out.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[i])
            with (out/'run.log').open('w') as log:
                subprocess.run([PYTHON,'-m','radio_gs.pipeline','--scene',str(path),'--queries',f'configs/consolidated/scannet_native/{name}.queries.json','--output',str(out/'query'),'--access','source-assisted','--gpu',gpus[i]],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/category/typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(mode,name,'complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(worker,range(len(gpus))))
    rows={}
    for path in jobs:
        mode=path.parent.name;name=path.name.removesuffix('.scene.json')
        rows.setdefault(mode,{})[name]=json.loads((root/mode/name/'evaluation/report.json').read_text())['point_metrics']
    macro={mode:{arm:{split:sum(r[arm][split]['miou'] for r in scenes.values())/len(scenes) for split in ['19','15','10']} for arm in ['map','source']} for mode,scenes in rows.items()}
    (root/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,completed=len(jobs),configs=a.configs,
        independent_test=False,official_sota_equivalence=False,public_entry=True),indent=2)+'\n');print(macro)
if __name__=='__main__':main()
