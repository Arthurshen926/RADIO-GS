"""Evaluate one declared learned membership condition in the complete system."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/learned_instance_additions'))
    p.add_argument('--decoder',choices=['region','element','element_local','element_anchor','element_joint','element_joint_oof','element_query'],required=True)
    p.add_argument('--instance-first',action='store_true',help='Run the changed instance reader before unchanged public tasks')
    p.add_argument('--benchmarks',nargs='+',help='Explicit execution subset; saved recipe still declares the whole system')
    p.add_argument('--gpus',default='0,5');a=p.parse_args()
    recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
    for job in recipe['jobs']:
        if job['benchmark']!='uqis':continue
        source=Path(job['scene']).parent;spec=json.loads((source/'map.json').read_text())
        out=a.root/'benchmark_maps'/a.decoder/job['name'];out.mkdir(parents=True,exist_ok=True)
        for name in spec['map_files']:
            if not (out/name).exists():os.link(source/name,out/name)
        checkpoint={'region':'conditional_membership.joblib','element':'conditional_elements.joblib',
                    'element_local':'conditional_local_elements.joblib','element_anchor':'conditional_anchor_elements.joblib',
                    'element_joint':'conditional_joint_elements.joblib',
                    'element_joint_oof':'conditional_joint_oof_elements.joblib',
                    'element_query':'conditional_query_elements.joblib'}[a.decoder]
        local=a.decoder in ['element_local','element_anchor','element_joint','element_joint_oof','element_query']
        if local:
            extra=['element_appearance.npy','element_appearance_observation.npy']
            for name in extra:
                if not (out/name).exists():os.link(a.root/'benchmark_local_appearance'/job['name']/name,out/name)
            spec['map_files']+=extra
            spec['map_bytes']=sum((out/name).stat().st_size for name in spec['map_files'])
        spec['conditional_membership']=dict(checkpoint=str((a.root/checkpoint).resolve()),
            decoder='element' if a.decoder.startswith('element') else 'region',local_appearance=local,
            condition='additional independent 3D instance training on rendered RGB; original query maps and native identity retained')
        (out/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
        job['scene']=str(out/'map.json')
    if a.instance_first:recipe['jobs'].sort(key=lambda job:job['benchmark']!='uqis')
    recipe.update(name='Conditional '+a.decoder+' supervision, whole-instance reader',default=False,
        validation=dict(scope='full five-task development evaluation; source-assisted task readers remain v8',
            additional_supervision='independent official-training 3D whole-instance annotations',
            training_design=str(a.root/'design.json'),public_reader_method_changed=False))
    path=a.root/(a.decoder+'_recipe.json');path.write_text(json.dumps(recipe,indent=2)+'\n')
    subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','-u','-m','radio_gs.pipeline.cohort',
        '--recipe',str(path),'--output',str(a.root/(a.decoder+'_system')),'--gpus',a.gpus]+
        (['--benchmarks']+a.benchmarks if a.benchmarks else []),
        env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1'),check=True)


if __name__=='__main__':main()
