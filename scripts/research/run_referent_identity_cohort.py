"""Fixed target-role/landmark comparison within the common system recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from radio_gs.models.referent_language import parse_referents


PYTHON = '/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['target', 'relations', 'native_relations', 'relation_support'], required=True)
    p.add_argument('--prepared-queries', type=Path)
    p.add_argument('--appearance-maps', type=Path)
    p.add_argument('--gpus', default='4,5')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); root = a.output; root.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(Path('configs/pipeline/system_v7.json').read_text())
    jobs = [job for job in recipe['jobs'] if job['benchmark'] == 'uqis']
    query_path = a.prepared_queries or root / 'queries.json'
    if not query_path.exists():
        public = Path(jobs[0]['queries'])
        queries = [q for modality in ['text', 'image', 'point_2d', 'point_3d']
            for q in json.loads((public / 'queries' / (modality + '.json')).read_text())['queries']]
        texts = [q for q in queries if q['modality'] == 'text']
        plans, report = parse_referents([q['expression'] for q in texts])
        for q, plan in zip(texts, plans): q['referent_plan'] = plan
        for q in queries:
            if q['modality'] == 'image': q['crop_rgb_path'] = str((public / q['crop_rgb_path']).resolve())
        query_path.write_text(json.dumps(dict(queries=queries, language_preprocessing=report), indent=2) + '\n')
    if a.prepare_only: return
    for job in jobs:
        source = a.appearance_maps / job['name'] if a.appearance_maps else Path(job['scene']).parent
        destination = root / 'maps' / job['name']; destination.mkdir(parents=True, exist_ok=True)
        spec = json.loads((source / 'map.json').read_text())
        for name in spec['map_files']:
            if not (destination / name).exists(): os.link(source / name, destination / name)
        spec['text_identity'] = a.mode
        if a.appearance_maps: spec['image_identity_fusion'] = 'equal_rank'
        (destination / 'map.json').write_text(json.dumps(spec, indent=2) + '\n')
        job.update(scene=str(destination / 'map.json'), queries=str(query_path))
    recipe.update(default=False, name='Native target roles and uncertain geometric landmarks: ' + a.mode,
        comparison='same map, original semantic field and physical scope; identity roles changed only',
        query_preprocessing=str(query_path),
        validation=dict(status='fixed full opened-development UQIS cohort; public providers unchanged'))
    path = root / 'recipe.json'; path.write_text(json.dumps(recipe, indent=2) + '\n')
    with (root / 'run.log').open('w') as log:
        subprocess.run([PYTHON, '-u', '-m', 'radio_gs.pipeline.cohort', '--recipe', str(path),
            '--output', str(root / 'system'), '--gpus', a.gpus, '--benchmarks', 'uqis'],
            env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1'),
            stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__': main()
