"""Gate on independent rooms, then run shared five-benchmark entry points."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from scripts.research.fit_real_direct_membership import ROOT, write

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    fits = {c: json.loads((ROOT / c / 'fit.json').read_text()) for c in ['rendered', 'real']}
    real = fits['real']['validation_macro'][0];rendered = fits['rendered']['validation_macro'][0]
    native = fits['real']['native_validation_macro'][0]
    gate = dict(real=real, rendered=rendered, native=native, passed=real >= max(rendered, native) + .02,
        selection='declared independent real-room development gate; no benchmark metrics consulted')
    write(ROOT / 'development_gate.json', gate)
    if not gate['passed']:
        print('development gate failed', gate, flush=True);return
    for condition in ['rendered', 'real']:
        recipe = json.loads(Path('configs/pipeline/system_v8.json').read_text())
        recipe.update(name='Direct geometric anchor membership: ' + condition, default=False,
            declaration='configs/recipes/real_direct_membership.json',
            comparison='same field and teachers; one full-domain actual-anchor decoder; matched training observation control',
            validation=dict(status='research candidate, no automatic promotion', independent_test=False,
                summary=str(ROOT / condition / 'system/summary.json'), comparison=str(ROOT / 'comparison.json')))
        for job in recipe['jobs']:
            if job['benchmark'] != 'uqis':continue
            parent = Path(job['scene']).parent;folder = ROOT / condition / 'maps' / job['name']
            folder.mkdir(parents=True, exist_ok=True)
            for file in parent.iterdir():
                if file.is_file() and file.name != 'map.json' and not (folder / file.name).exists():
                    os.link(file, folder / file.name)
            spec = json.loads((parent / 'map.json').read_text())
            spec.update(parent_map=str(parent / 'map.json'), save_identity_evidence=True,
                direct_anchor_membership=dict(checkpoint=str((ROOT / condition / 'model.joblib').resolve()),
                    supervision='official3D instances in disjoint22 rooms, actual positive-anchor identity',
                    training_observations=condition, inference_domain='all map elements',
                    applies_to=['point_2d', 'point_3d']))
            write(folder / 'map.json', spec);job['scene'] = str(folder / 'map.json')
            job['selection_contract'] = ('independent boolean sets; original native identity/physical scope for text and image; '
                'actual-anchor complete-carrier classifier for points, threshold calibrated only on independent rooms')
        recipe['jobs'].sort(key=lambda j: j['benchmark'] != 'uqis')
        write(ROOT / (condition + '_recipe.json'), recipe)
    def run(condition):
        gpus = '2,4,5' if condition == 'real' else '0,1'
        command = [PY, '-u', '-m', 'radio_gs.pipeline.cohort', '--recipe', str(ROOT / (condition + '_recipe.json')),
            '--output', str(ROOT / condition / 'system'), '--gpus', gpus]
        if condition == 'rendered':command.extend(['--benchmarks', 'uqis'])
        with (ROOT / (condition + '_system.log')).open('w') as log:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('completed', condition, flush=True)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, ['rendered', 'real']))


if __name__ == '__main__':
    main()
