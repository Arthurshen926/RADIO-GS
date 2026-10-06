"""Independent reference-image identity, within the existing common recipe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

PY='/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/region_appearance_study'))
    p.add_argument('--gpus',default='4,5');p.add_argument('--appearance-maps',type=Path)
    p.add_argument('--fusion',action='store_true');a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(Path('configs/pipeline/system_v7.json').read_text());gpus=a.gpus.split(',')
    jobs=[j for j in recipe['jobs'] if j['benchmark']=='uqis']
    def build(lane):
        for job in jobs[lane::len(gpus)]:
            output=root/'maps'/job['name'];output.mkdir(parents=True,exist_ok=True)
            if not (output/'build.json').exists():
                if a.appearance_maps:
                    source=a.appearance_maps/job['name'];spec=json.loads((source/'map.json').read_text())
                    for name in spec['map_files']:os.link(source/name,output/name)
                    (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
                    report=json.loads((source/'build.json').read_text());report.update(feature_build_reused=True)
                    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
                else:
                    with (output/'compile.log').open('w') as log:
                        subprocess.run([PY,'-u','-m','radio_gs.pipeline.region_appearance','--scene',job['scene'],
                            '--mapping','output/scannet_uqis_9_v0_3_1_candidate/mapping/scenes.json','--output',str(output)],
                            env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=gpus[lane],OPENBLAS_NUM_THREADS='1'),
                            stdout=log,stderr=subprocess.STDOUT,check=True)
            if a.fusion:
                spec=json.loads((output/'map.json').read_text());spec['image_identity_fusion']='equal_rank'
                (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
            job['scene']=str(output/'map.json');print('built',job['name'],flush=True)
    with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(build,range(len(gpus))))
    recipe.update(default=False,name='Independent native local appearance for image reference identity',
        validation={'status':'fixed full UQIS cohort; original public providers unchanged'},
        comparison='same legal photos and masks; additional DINOv2-base teacher; native image space replaced, text and point rules unchanged')
    if a.fusion:
        recipe.update(name='Dual native image identity with original semantic field and physical scope',
            comparison='same SigLIP and DINO maps; equal candidate rank evidence for identity; original semantic projection and scope retained')
    path=root/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    with (root/'run.log').open('w') as log:
        subprocess.run([PY,'-u','-m','radio_gs.pipeline.cohort','--recipe',str(path),'--output',str(root/'system'),
            '--gpus',a.gpus,'--benchmarks','uqis'],env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Independent appearance comparison completed on all four query modalities.',flush=True)


if __name__=='__main__':main()
