"""Run matched native-capacity mechanism controls and existing scorers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

PYTHON = '/root/miniconda3/envs/cybersim_agent/bin/python'


def run(job):
    gpu, key, mode = job
    base = Path('output/region_native_next39') / mode
    out = base / key
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', PYTHONPATH='.')
    commands = []
    if not (out/'fit.json').exists():
        commands.append([PYTHON, 'scripts/research/fit_region_native_field.py', '--key', key, '--mode', mode])
    commands.extend([
        [PYTHON, 'scripts/research/calibrate_scope_dictionary.py', '--key', key, '--base', str(base)],
        [PYTHON, 'scripts/research/evaluate_scope_dictionary.py', '--key', key, '--base', str(base), '--readout', 'anchored', '--pipeline', '--calibrated'],
        [PYTHON, 'scripts/research/evaluate_scope_reference_images.py', '--key', key, '--base', str(base)],
    ])
    with (out/'execution.log').open('w') as log:
        for command in commands:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                return dict(key=key, mode=mode, exit_code=result.returncode, failed_command=command)
    return dict(key=key, mode=mode, exit_code=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', default='0:lerf_figurines:normalized_pool,1:lerf_figurines:unit_mean,2:lerf_ramen:normalized_pool,3:lerf_ramen:unit_mean')
    a = p.parse_args()
    jobs = [job.split(':') for job in a.jobs.split(',')]
    with ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    path = Path('output/region_native_next39/execution_'+'_'.join(j[0] for j in jobs)+'.json')
    path.write_text(json.dumps(results, indent=2)+'\n')
    print(results, flush=True)
