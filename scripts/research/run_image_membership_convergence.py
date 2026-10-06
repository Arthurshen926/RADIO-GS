"""Converge the previously supported image component, without failed text heads."""
import json
import os
from pathlib import Path
import subprocess

from scripts.research.fit_query_conditioned_membership import ROOT, write

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    folder = ROOT / 'image_convergence';folder.mkdir(exist_ok=True)
    declaration = dict(primary='native image identity plus v9 complete-carrier member decoder; native v9 text; actual v9 points',
        selection_basis='previously opened UQIS development factorial: image gain5.0218pp with positive scene interval; uncertain text gain excluded by modality, never per-query routing',
        gain_is_new_this_round=False, new_training_models_included=False,
        benchmark_jobs=31, promotion='image gains at least2pp over v9 with positive scene interval; text/point/native outputs exact; joint success nondecreasing',
        public_replay='same inputs and readers, numerical differences reported explicitly; no new public-task improvement claimed',
        independent_final_test=False, same_map_all_queries=True, source_access_at_query=False)
    write(folder / 'design.json', declaration)
    recipe = json.loads(Path('configs/pipeline/system_v9.json').read_text())
    recipe.update(name='RADIO-GS shared observation architecture v10 candidate', default=False,
        role='native text scope; local image identity plus inferred-anchor membership; actual positive-anchor membership for points',
        declaration='configs/recipes/query_conditioned_membership.json',
        validation=dict(status='component convergence candidate', independent_test=False, summary=str(folder / 'system/summary.json')))
    for job in recipe['jobs']:
        if job['benchmark'] != 'uqis':continue
        parent = Path(job['scene']).parent;out = folder / 'maps' / job['name'];out.mkdir(parents=True, exist_ok=True)
        for file in parent.iterdir():
            if file.is_file() and file.name != 'map.json' and not (out / file.name).exists():os.link(file, out / file.name)
        spec = json.loads((parent / 'map.json').read_text())
        spec.update(parent_map=str(parent / 'map.json'), save_identity_evidence=True, inferred_anchor_membership=['image'])
        write(out / 'map.json', spec);job['scene'] = str(out / 'map.json')
        job['selection_contract'] = 'original text physical scope; images use estimated-anchor complete membership; real points retain original positive-anchor membership'
    write(folder / 'recipe.json', recipe)
    command = [PY, '-u', '-m', 'radio_gs.pipeline.cohort', '--recipe', str(folder / 'recipe.json'),
        '--output', str(folder / 'system'), '--gpus', '1,2,3,4,5']
    with (folder / 'system.log').open('w') as log:
        subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    print('complete common31 component convergence', flush=True)


if __name__ == '__main__':main()
