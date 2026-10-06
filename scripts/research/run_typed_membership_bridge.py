"""Execute a complete common recipe and both identity/scope attribution controls."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from scripts.research.fit_typed_referent_binding import ROOT, SOURCE, write

PY = '/root/miniconda3/envs/cybersim_agent/bin/python'


def main():
    bridge = json.loads((ROOT / 'membership_bridge/summary.json').read_text())
    if not bridge['development_gate']['passed']:
        print('independent development gate failed', bridge['development_gate'], flush=True);return
    declaration = dict(primary='semantic_bridge', controls=['semantic_family', 'native_bridge'],
        baseline='system_v9: native identity plus previous physical scope',
        primary_jobs=31, control_jobs=18, modalities=['text', 'image', 'point_2d', 'point_3d'],
        new_reader='text lexical identity; text/image inferred-anchor full-carrier membership; original actual points',
        all_queries_retained=True, thresholds_changed=False, source_or_query_permissions_changed=False,
        promotion='both text/image setIoU each improve at least2pp, paired scene interval lower bound positive; '
            'native and point predictions/sets exact; joint4query success nondecreasing; public scores retained',
        official_sota_comparison=False, independent_final_test=False)
    write(ROOT / 'benchmark_design.json', declaration)
    for condition in ['semantic_bridge', 'semantic_family', 'native_bridge']:
        recipe = json.loads(Path('configs/pipeline/system_v9.json').read_text())
        recipe.update(name='Typed identity and inferred-anchor membership: ' + condition, default=False,
            declaration='configs/recipes/typed_referent_binding.json',
            validation=dict(status='research candidate', independent_test=False, summary=str(ROOT / condition / 'system/summary.json')))
        for job in recipe['jobs']:
            if job['benchmark'] != 'uqis':continue
            parent = Path(job['scene']).parent;out = ROOT / condition / 'maps' / job['name'];out.mkdir(parents=True, exist_ok=True)
            for file in parent.iterdir():
                if file.is_file() and file.name != 'map.json' and not (out / file.name).exists():os.link(file, out / file.name)
            spec = json.loads((parent / 'map.json').read_text());spec.update(parent_map=str(parent / 'map.json'), save_identity_evidence=True)
            if condition.startswith('semantic'):
                spec['lexical_identity'] = dict(checkpoint=str((ROOT / 'semantic/model.pt').resolve()),
                    role_checkpoint=str((SOURCE / 'roles/model.pt').resolve()),
                    training_condition='disjoint22 rooms, NR3D/ScanEnts3D/official3D classes; frozen native teacher coordinates')
            if condition.endswith('bridge'):spec['inferred_anchor_membership'] = ['text', 'image']
            write(out / 'map.json', spec);job['scene'] = str(out / 'map.json')
            job['selection_contract'] = 'explicit complete-carrier membership from provided or inferred anchor; no evaluator information at query' if condition.endswith('bridge') else job['selection_contract']
        recipe['jobs'].sort(key=lambda job: job['benchmark'] != 'uqis')
        write(ROOT / (condition + '_recipe.json'), recipe)
    def run(condition):
        gpus = {'semantic_bridge': '0,1,4,5', 'semantic_family': '2', 'native_bridge': '3'}[condition]
        command = [PY, '-u', '-m', 'radio_gs.pipeline.cohort', '--recipe', str(ROOT / (condition + '_recipe.json')),
            '--output', str(ROOT / condition / 'system'), '--gpus', gpus]
        if condition != 'semantic_bridge':command.extend(['--benchmarks', 'uqis'])
        with (ROOT / (condition + '_system.log')).open('w') as log:
            subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('complete', condition, flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(run, ['semantic_bridge', 'semantic_family', 'native_bridge']))


if __name__ == '__main__':main()
