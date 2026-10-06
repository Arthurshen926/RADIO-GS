"""Individual scope identity and one-hop geometry, fixed six-arm comparison."""
import argparse,concurrent.futures,json,os,shutil,subprocess
from pathlib import Path
from evaluate_independent_scope_reliability import evaluate_jobs

ROOT=Path('output/region_identity_transport')
SCENES=['scene0670_01','scene0678_00','scene0685_02']
MODES=['v6','emitter_instance','native_instance','verified_instance','verified_union','linked_verified']
PY='/root/miniconda3/envs/cybersim_agent/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');a=p.parse_args()
    root=ROOT/'development' if a.development else ROOT;root.mkdir(parents=True,exist_ok=True)
    baseline={};queries={};evaluations=None
    if a.development:
        recipe=json.loads(Path('configs/pipeline/system_v6.json').read_text());evaluations={}
        for job in recipe['jobs']:
            if job['benchmark']!='scannet':continue
            name=job['name'];baseline[name]=json.loads(Path(job['scene']).read_text());queries[name]=job['queries'];evaluations[name]=job['evaluation']
        shutil.copyfile(ROOT/'scannetv2-labels.combined.tsv',root/'scannetv2-labels.combined.tsv')
    else:
        for name in SCENES:
            scene=json.loads((ROOT/'maps'/name/'compiled/scene.json').read_text())
            scene['source_evidence']=dict(views=json.loads((ROOT/'evidence'/name/'views_0.json').read_text())['views'],raster_hw=[96,128])
            scene['local_element_control']='average_prior';scene['local_auxiliary_scene']=str(ROOT/'maps'/name/'adapter/mask_adapter/scene.json')
            scene['native_observation_reliability']='entropy_information';scene.pop('observation_weighting',None)
            baseline[name]=scene;queries[name]=str(ROOT/f'{name}.queries.json')
    split=json.loads((ROOT/'split.json').read_text())
    split.update(fit=[],training_labels_used=False,fit_policy='no fitting; all mechanisms fixed without scene semantic labels',validation=list(baseline),
        validation_policy='original ten opened development rooms; no per-scene routing' if a.development else 'next three official-validation room prefixes; first use for this mechanism, not a sealed final test')
    (root/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    jobs=[];build=[]
    for name,scene in baseline.items():
        destination=root/f'{name}.queries.json'
        if Path(queries[name])!=destination:shutil.copyfile(queries[name],destination)
        for mode in MODES:
            spec=json.loads(json.dumps(scene))
            if mode!='v6':
                spec['source_region_observations']=str(root/'region_observations'/name);spec['source_region_mode']=mode
            dest=root/'configs'/mode/f'{name}.scene.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(spec,indent=2)+'\n')
            jobs.append((mode,name,dest))
            if mode=='v6':build.append((name,dest))
    def worker(i):
        for name,scene in build[i::6]:
            out=root/'region_observations'/name;out.mkdir(parents=True,exist_ok=True)
            with (out/'build.log').open('w') as log:
                subprocess.run([PY,'-m','radio_gs.pipeline.region_observation_identity','--scene',str(scene),'--output',str(out)],
                    env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
            print(name,'region identity and geometry links complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
    evaluate_jobs(root,list(baseline),jobs,evaluation_specs=evaluations)
    path=root/'summary.json';summary=json.loads(path.read_text());summary['new_scene_for_this_mechanism']=not a.development
    summary['ground_truth_training']=False;summary['previously_opened_development_rooms']=a.development
    path.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
