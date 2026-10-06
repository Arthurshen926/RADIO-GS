"""Run one shared candidate and a supervision-only control after development."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from scripts.research.fit_query_conditioned_membership import ROOT, write

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    fit = json.loads((ROOT / 'fit.json').read_text())
    exact = json.loads((ROOT / 'exact_development.json').read_text())
    primary = 'query_evidence' if exact['development_gates']['query_evidence'] else 'query_only'
    if not any(exact['development_gates'].values()):
        print('Development gate failed; no new benchmark execution or promotion.', flush=True)
        return
    write(ROOT / 'benchmark_design.json', dict(primary=primary, control='anchor_only',
        primary_jobs=31, control_jobs=9, all_queries_retained=True,
        baseline='v9, plus previous native-identity estimated-anchor bridge as declared research comparator',
        promotion='both text/image setIoU gains at least2pp with positive paired-scene interval; actual points and native arrays exact; all-four success nondecreasing',
        transfer='training uses natural text references; image region choice retains existing SigLIP/DINO ranks; evidence channels use native SigLIP cosine, no image benchmark adaptation',
        public_comparison='fresh fixed-protocol replay, no independent SOTA assertion',
        threshold_selection='exact complete-range cut on independent calibration only', independent_final_test=False))
    for name in [primary, 'anchor_only']:
        recipe = json.loads(Path('configs/pipeline/system_v9.json').read_text())
        recipe.update(name='Uncertain query-conditioned membership: ' + name, default=False,
            declaration='configs/recipes/query_conditioned_membership.json',
            validation=dict(status='research candidate', summary=str(ROOT / name / 'system/summary.json'), independent_test=False))
        for job in recipe['jobs']:
            if job['benchmark'] != 'uqis':continue
            parent = Path(job['scene']).parent;folder = ROOT / name / 'maps' / job['name'];folder.mkdir(parents=True, exist_ok=True)
            for file in parent.iterdir():
                if file.is_file() and file.name != 'map.json' and not (folder / file.name).exists():os.link(file, folder / file.name)
            spec = json.loads((parent / 'map.json').read_text())
            spec.update(parent_map=str(parent / 'map.json'), save_identity_evidence=True,
                query_conditioned_membership=dict(checkpoint=str((ROOT / (name + '_exact.joblib')).resolve()), applies_to=['text', 'image'],
                    supervision=fit['design']['supervision'], training_identity='actual referred instance; estimated anchor may be wrong'))
            write(folder / 'map.json', spec);job['scene'] = str(folder / 'map.json')
            job['selection_contract'] = 'native query-conditioned member decision; estimated anchors never forced positive; original actual point contract'
        write(ROOT / (name + '_recipe.json'), recipe)
    def run(name):
        # Current external GPU users are retained; these lanes use the larger
        # free-memory devices and no process is interrupted.
        devices = '3,4,5' if name == primary else '0'
        command = [PY, '-u', '-m', 'radio_gs.pipeline.cohort', '--recipe', str(ROOT / (name + '_recipe.json')),
            '--output', str(ROOT / name / 'system'), '--gpus', devices]
        if name == 'anchor_only':command.extend(['--benchmarks', 'uqis'])
        with (ROOT / (name + '_system.log')).open('w') as log:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('complete', name, flush=True)
    with ThreadPoolExecutor(2) as pool:list(pool.map(run, [primary, 'anchor_only']))


if __name__ == '__main__':main()
