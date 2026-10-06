"""Information x consumed-objective controls on the next three fixed rooms."""
import argparse,concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
from evaluate_independent_scope_reliability import evaluate_jobs

ROOT=Path('output/element_retrieval_reliability')
SCENES=['scene0575_00','scene0591_02','scene0599_02']
PY='/root/miniconda3/envs/cybersim_agent/bin/python'
MODES=['siglip_vote','adapter_vote','base_quality','native_quality','base_retrieval','native_retrieval','native_retrieval_shared_matched_mass',
    'native_mlp_quality','native_mlp_retrieval','native_mlp_retrieval_shared_matched_mass']

def encode(i):
    name=SCENES[i];out=ROOT/'region_identity'/name;out.mkdir(parents=True,exist_ok=True)
    scene=json.loads((ROOT/'sources'/name/'provider_scene.json').read_text())
    scene['source_evidence']=json.loads((ROOT/'evidence'/name/'views_0.json').read_text())
    path=out/'scene.json';path.write_text(json.dumps(scene,indent=2)+'\n')
    with (out/'encode.log').open('w') as log:
        subprocess.run([PY,'-m','radio_gs.pipeline.category_union_adapter','--scene',str(path),'--output',str(out)],
            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(i+3)),stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'region identity encoded',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','encode','evaluate'],default='all');args=p.parse_args()
    if args.stage in ['all','encode']:
        with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(encode,range(3)))
    if args.stage=='encode':return
    models=[ROOT/f'{n}.pt' for n in ['base_quality','native_quality','base_retrieval','native_retrieval','native_mlp_quality','native_mlp_retrieval']]
    (ROOT/'frozen_models.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in models},indent=2)+'\n')
    jobs=[]
    for name in SCENES:
        views=json.loads((ROOT/'evidence'/name/'views_0.json').read_text())['views']
        for mode in MODES:
            path=ROOT/'maps'/name/('compiled/scene.json' if mode=='siglip_vote' else 'adapter/mask_adapter/scene.json')
            scene=json.loads(path.read_text());scene['source_streaming']=True;scene['native_observation_reliability']='entropy_information'
            scene.pop('observation_weighting',None);scene['source_evidence']=dict(views=views,raster_hw=[96,128])
            if mode not in ['siglip_vote','adapter_vote']:
                model=mode.removesuffix('_shared_matched_mass')
                scene['source_reliability_checkpoint']=str(ROOT/f'{model}.pt')
                scene['source_reliability_mode']='shared_matched_mass' if mode.endswith('_shared_matched_mass') else 'typed'
                scene['source_region_identity']=str(ROOT/'region_identity'/name/'mask_adapter')
                scene['map_contract']['supervised_source_reliability']=True
            dest=ROOT/'configs'/mode/f'{name}.scene.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(scene,indent=2)+'\n')
            jobs.append((mode,name,dest))
    evaluate_jobs(ROOT,SCENES,jobs)

if __name__=='__main__':main()
