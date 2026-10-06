"""Freeze a boundary relation condition and execute the common benchmark recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=['fixed','learned'],required=True)
    parser.add_argument('--gpus',default='0,4,5')
    parser.add_argument('--benchmarks',nargs='+')
    args=parser.parse_args()
    root=Path('output/surface_membership_study')
    frozen=json.loads((root/'cut_readout.json').read_text())
    recipe=json.loads(Path('output/learned_instance_additions/element_query_recipe.json').read_text())
    for job in recipe['jobs']:
        if job['benchmark']!='uqis':continue
        source=Path(job['scene']).parent
        spec=json.loads((source/'map.json').read_text())
        destination=root/'maps'/args.condition/job['name']
        destination.mkdir(parents=True,exist_ok=True)
        for name in spec['map_files']:
            if not (destination/name).exists():os.link(source/name,destination/name)
        graph='surface_affinity.npz'
        if not (destination/graph).exists():
            os.link(root/'benchmark_graphs'/job['name']/(args.condition+'.npz'),destination/graph)
        spec['map_files']=[*spec['map_files'],graph]
        spec['map_bytes']=sum((destination/name).stat().st_size for name in spec['map_files'])
        spec['surface_consistency']=dict(graph=graph,strength=frozen['strengths'][args.condition],
            solver=frozen['solver'],score_rule='map_conditional',condition=args.condition,readout_calibration=str(root/'cut_readout.json'),
            query_contract='whole-instance; same surface graph for all four inputs',
            query_source_rgb=False,geometry_optimized=False,native_coordinates_optimized=False)
        (destination/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
        job['scene']=str(destination/'map.json')
    recipe.update(name='Shared whole-instance surface relation '+args.condition,
        default=False,validation=dict(scope='five-task development; UQIS shared instance set solver',
            source_assisted_public_readers_changed=False,comparison_parameters_frozen_before_predictions=True,
            additional_supervision='24 independent ScanNet training rooms, 3D instance labels; rendered observations',
            graph_condition=args.condition,calibration=str(root/'cut_readout.json')))
    recipe['jobs'].sort(key=lambda job:job['benchmark']!='uqis')
    path=root/(args.condition+'_recipe.json')
    path.write_text(json.dumps(recipe,indent=2)+'\n')
    subprocess.run(['/root/miniconda3/envs/cybersim_agent/bin/python','-u','-m','radio_gs.pipeline.cohort',
        '--recipe',str(path),'--output',str(root/(args.condition+'_system')),'--gpus',args.gpus]+
        (['--benchmarks']+args.benchmarks if args.benchmarks else []),
        env=dict(os.environ,PYTHONPATH='.',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True)


if __name__=='__main__':main()
