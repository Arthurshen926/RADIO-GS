"""Continue an authorized ScanNet cohort after its already-running pilot.

One worker per assigned GPU. Failed stages stop that worker and preserve all
receipts; no automatic retry, training reduction, or result-based selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot-pid', type=int, required=True)
    p.add_argument('--scenes', nargs='+', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    allowed = {'GPU-888bd14c-aa37-e1f4-dd2d-54385991d363',
               'GPU-20e3b4ca-01ad-a922-b8ea-c9d66e30c02e'}
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in allowed:
        raise ValueError('An explicitly assigned GPU 0/1 UUID is required')
    if a.receipt.exists():
        raise FileExistsError(a.receipt)
    status = dict(pid=os.getpid(), assigned_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  scenes=a.scenes, complete=False, state='waiting_for_pilot', stages=[])
    def save():
        temp = a.receipt.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2)+'\n')
        temp.replace(a.receipt)
    save()
    while Path(f'/proc/{a.pilot_pid}').exists():
        time.sleep(15)
    try:
        for i, scene in enumerate(a.scenes):
            root = Path('output/scannet_frozen_field_20260918')/scene
            if i:
                status['state'] = 'building_'+scene
                save()
                with (root/'queue_build.log').open('x') as log:
                    subprocess.run([sys.executable, 'scripts/audits/run_scannet_frozen_mesh_scene.py',
                        '--scene', scene, '--device', 'cuda:0', '--resume'],
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            if not json.loads((root/'execution.json').read_text()).get('complete'):
                raise RuntimeError(f'{scene}: source pipeline incomplete; inspect stage receipts')
            status['state'] = 'evaluating_'+scene
            save()
            out = root/'official_protocol_evaluation'
            with (root/'queue_evaluation.log').open('x') as log:
                subprocess.run([sys.executable, 'scripts/audits/evaluate_scannet_frozen_field.py',
                    '--scene', scene, '--device', 'cuda:0', '--output', str(out)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            status['stages'].append(dict(scene=scene, report=str(out/'report.json')))
            save()
        status.update(complete=True, state='complete')
    except Exception as error:
        status.update(state='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
