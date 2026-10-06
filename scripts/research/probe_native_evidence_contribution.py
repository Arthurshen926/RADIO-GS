"""Complete-cohort source identity binding and native prior information controls."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess

PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/native_evidence_contribution')
CONFIG=Path('configs/pipeline/scannet_native_complete')


def scene_run(name,gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    from torch.nn import functional as F
    from radio_gs.models.native_query_encoders import load_siglip_text
    from radio_gs.pipeline.native_field import score_scene_field
    from radio_gs.querying.cold_category import categorical_prior
    from radio_gs.querying.scannet_runtime import refine_from_declared_sources
    torch.set_num_threads(4)
    scene=json.loads((CONFIG/f'{name}.scene.json').read_text())
    queries=json.loads(Path(f'configs/consolidated/scannet_native/{name}.queries.json').read_text())['queries']
    texts=[q['text'] for q in queries]
    with torch.inference_mode():
        encoder,tokenizer=load_siglip_text(scene['text_encoder'],'cuda')
        tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
        encoding=F.normalize(F.normalize(encoder(**tokens).pooler_output.float(),dim=-1).half().float(),dim=-1)
        del encoder
        field=torch.load(scene['native_field'],map_location='cpu',weights_only=False)
        observed=field['supported'].reshape(-1).cuda()
        native=categorical_prior(score_scene_field(scene,field,encoding,'cuda').cuda(),100.)
        for arm in ['spatial_identity','uniform_prior']:
            settings=dict(scene)
            prior=native
            if arm=='spatial_identity':settings['source_identity_assignment']='spatial_proposal'
            else:prior=torch.full_like(native,-math.log(len(texts)-1))
            values,stats=refine_from_declared_sources(prior,observed,settings,texts)
            out=ROOT/name/arm;out.mkdir(parents=True,exist_ok=True)
            torch.save(dict(source_evidence_used=True,queries=queries,texts=texts,category_logits=values.cpu(),scores=values.cpu(),native_logits=prior.cpu(),semantic_geometry=scene['geometry'],semantic_observed=observed.cpu(),source_stats=stats,arm=arm),out/'typed_readout.pt')
            subprocess.run([PYTHON,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'typed_readout.pt'),'--evaluation',f'configs/consolidated/scannet_native/{name}.evaluation.json','--output',str(out/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--gpu',default='0');a=p.parse_args()
    if a.scene:scene_run(a.scene,a.gpu);return
    names=sorted(p.name.removesuffix('.scene.json') for p in CONFIG.glob('*.scene.json'))
    ROOT.mkdir(parents=True,exist_ok=True)
    def worker(gpu):
        for name in names[gpu::6]:
            out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
            with (out/'run.log').open('w') as log:
                subprocess.run([PYTHON,__file__,'--scene',name,'--gpu',str(gpu)],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
            print(name,'complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
    rows={}
    for name in names:
        rows[name]={'v2':json.loads(Path('output/system_v2/scannet',name,'evaluation/report.json').read_text())['point_metrics']['source']}
        for arm in ['spatial_identity','uniform_prior']:
            rows[name][arm]=json.loads((ROOT/name/arm/'evaluation/report.json').read_text())['point_metrics']['source']
    macro={a:{s:sum(r[a][s]['miou'] for r in rows.values())/len(rows) for s in ['19','15','10']} for a in ['v2','spatial_identity','uniform_prior']}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,all_source_frames=1158,independent_test=False,prior_ablation='uniform 1/19 with identical support and0.1 regularization; native identity teacher remains'),indent=2)+'\n')
    print(json.dumps(macro,indent=2),flush=True)

if __name__=='__main__':main()
