"""Rebuild scene maps after SAM3 rotary-phase corruption; no old SAM reuse."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser()
p.add_argument('--scenes', nargs='+', required=True)
p.add_argument('--device', required=True)
a=p.parse_args()
base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
queue=base/'nvos_ropefix_pipeline_20260917'
queue.mkdir(exist_ok=True)
for scene in a.scenes:
    marker=queue/scene/'execution.json'
    if marker.exists():
        print('waiting for previously dispatched scene',scene,flush=True)
        while True:
            try:receipt=json.loads(marker.read_text())
            except json.JSONDecodeError:time.sleep(5);continue
            if any(s['status']=='failed' for s in receipt['stages'].values()):
                raise RuntimeError(scene+' existing execution failed; inspect before resuming')
            if receipt.get('finished'):break
            time.sleep(5)
        for name,stage in receipt['stages'].items():
            assert stage['status']=='complete'
            for file,digest in stage['outputs'].items():assert sha256_file(file)==digest
            for file,digest in stage['implementations'].items():
                assert sha256_file(queue/scene/'sources'/name/Path(file).name)==digest
        contract=json.loads((queue/scene/'source_sam/manifest.json').read_text())['generation_contract']
        assert contract['complex_rope_buffers_preserved'] and contract['schema_version']>=3
        print('verified previously dispatched scene',scene,flush=True)
        continue
    prior_dir = ('nvos_local_scale_repair_20260917' if scene=='fern' else
                 'nvos_scale_repair_20260917' if scene=='fortress' else 'nvos_local_scale2_repair_20260917')
    command=[sys.executable,'scripts/audits/run_nvos_bound_pipeline.py','--scene',scene,
        '--device',a.device,'--output-root',str(queue/scene),'--scale-prior',str(base/prior_dir/scene/'source_scale_prior.json'),
        '--allow-empty-text-supervision','--query-root',str(base/'nvos_reference_sam_queries_ropefix_20260917'),
        '--signed-policy','adaptive']
    if scene=='leaves':command+=['--voxel-metres','.0075','--minimum-voxel-views','1']
    print('starting',scene,flush=True)
    with (queue/(scene+'_queue.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
    if result.returncode:raise RuntimeError(scene+' failed; inspect queue log and stage receipt')
    print('completed',scene,flush=True)
