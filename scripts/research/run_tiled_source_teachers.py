"""Six GPU shards covering all four declared source cohorts, without sampling."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = '/root/miniconda3/envs/sam3/bin/python'
ROOT = Path('output/tiled_source_evidence')
RECIPE = json.loads(Path('configs/recipes/current_research.json').read_text())
JOBS = [(0, 'figurines', 0, 2), (4, 'figurines', 1, 2), (1, 'ramen', 0, 1),
        (2, 'teatime', 0, 1), (3, 'waldo_kitchen', 0, 2), (5, 'waldo_kitchen', 1, 2)]


def worker(job):
    gpu, scene, shard, shards = job
    item = RECIPE['scenes'][scene]
    out = ROOT / scene / 'tiles'
    out.mkdir(parents=True, exist_ok=True)
    command = [PY, 'scripts/research/detect_tiled_sources.py', '--scene', item['scene'],
               '--queries', item['queries'], '--output', str(out), '--shard', str(shard), '--shards', str(shards)]
    (out / f'command_{shard}.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / f'run_{shard}.log').open('w') as log:
        subprocess.run(command, env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu)),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(scene, shard, 'teacher complete', flush=True)


with ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(worker, JOBS))
