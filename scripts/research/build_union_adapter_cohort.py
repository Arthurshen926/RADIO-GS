"""Full declared-source cohort; six workers, no source-view subsampling."""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/category_union_adapter')
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
GPUS=['0','1','2','3','4','5']
def worker(i):
    for path in SCENES[i::len(GPUS)]:
        out=ROOT/path.name.removesuffix('.scene.json');out.mkdir(parents=True,exist_ok=True)
        with (out/'run.log').open('w') as log:
            subprocess.run([PYTHON,'-m','radio_gs.pipeline.category_union_adapter','--scene',str(path),'--output',str(out)],
                env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i]),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(path.name,'complete',flush=True)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(len(GPUS)) as pool:list(pool.map(worker,range(len(GPUS))))
    scenes={p.name.removesuffix('.scene.json'):json.loads((ROOT/p.name.removesuffix('.scene.json')/'encoding.json').read_text()) for p in SCENES}
    summary={'scenes':scenes,**{k:sum(v[k] for v in scenes.values()) for k in ['source_views','new_backbone_image_encodings','new_region_encodings','new_sam_calls']}}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='scenes'})
