"""Execute the chosen consolidated recipe; generate predictions before evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='configs/recipes/current_research.json')
    parser.add_argument('--benchmark',choices=['lerf','nvos'],default='lerf')
    parser.add_argument('--profile', choices=['map', 'source', 'regional', 'calibration', 'identity', 'identity_response', 'reference'])
    parser.add_argument('--scene', default='all')
    parser.add_argument('--scene-input',help='Explicit same-map reference scene, outside benchmark presets')
    parser.add_argument('--query-input',help='Explicit image/2D/3D reference-instance query')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Print exact commands without running models or evaluation')
    args = parser.parse_args()
    recipe = json.loads(Path(args.recipe).read_text())
    if args.scene_input:
        if not args.query_input:parser.error('--scene-input requires --query-input')
        if args.evaluate:parser.error('Custom image/point queries have no frozen benchmark evaluator; evaluate under an explicit task protocol')
        scene=json.loads(Path(args.scene_input).read_text());query=json.loads(Path(args.query_input).read_text())
        profile_name='reference';names=[scene['scene']];out=Path(args.output)/profile_name/scene['scene']/query['id']
        commands=[[recipe['python'],'scripts/research/run_reference_query.py','--scene',args.scene_input,'--query',args.query_input,'--gpu',args.gpu,'--output',str(out)]]
    elif args.benchmark=='nvos':
        benchmark=recipe['benchmarks']['nvos'];profile_name=args.profile or benchmark['default_profile']
        if profile_name!='reference':parser.error('NVOS uses the reference-instance response contract; text profiles do not apply')
        names=list(benchmark['scenes']) if args.scene=='all' else [args.scene];commands=[]
        for name in names:
            scene=benchmark['scenes'][name];out=Path(args.output)/profile_name/name
            commands.append([recipe['python'],benchmark['generation_entry'],'--scene',scene['scene'],'--query',scene['query'],'--gpu',args.gpu,'--output',str(out)])
            if args.evaluate:
                commands.append([recipe['python'],benchmark['evaluation_entry'],'--scene',scene['scene'],'--evaluation',scene['evaluation'],'--field',str(out/'typed_readout.pt'),'--typed-readout','--output',str(out/'evaluation')])
    else:
        profile_name = args.profile or recipe['default_profile']
        profile = recipe['profiles'][profile_name]
        names = list(recipe['scenes']) if args.scene == 'all' else [args.scene]
        commands = []
        for name in names:
            scene = recipe['scenes'][name]
            out = Path(args.output)/profile_name/name
            commands.append([recipe['python'], recipe['generation_entry'],
                             '--scene', scene['scene'], '--queries', scene['queries'],
                             '--mode', profile['mode'], '--checkpoint', recipe['checkpoint'],
                             '--gpu', args.gpu, '--output', str(out), *profile['flags']])
            if args.evaluate:
                commands.append([recipe['python'], recipe['evaluation_entry'],
                                 '--scene', scene['scene'], '--readout', str(out/profile['readout'].format(scene=name)),
                                 '--output', str(out/'evaluation'), *profile['evaluation_flags']])
    plan = dict(benchmark='explicit_reference' if args.scene_input else args.benchmark, recipe=args.recipe, profile=profile_name, scenes=names,
                gpu=args.gpu, commands=commands, existing_predictions_reused=False)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    root = Path(args.output)/profile_name
    root.mkdir(parents=True, exist_ok=True)
    (root/'execution_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    env = dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4')
    for command in commands:
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
