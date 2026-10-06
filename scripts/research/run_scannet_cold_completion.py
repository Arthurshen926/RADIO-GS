"""Complete the two previously missing source-assisted scenes on six GPUs."""
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('output/scannet_cold_completion')
CONFIG = Path('configs/pipeline/scannet_cold_completion')
NATIVE = '/root/miniconda3/envs/cybersim_agent/bin/python'
SAM = '/root/miniconda3/envs/sam3/bin/python'
SCENES = ['scene0200_00', 'scene0645_00']


def worker(gpu):
    name, shard, shards = (SCENES[0], 0, 1) if gpu == 0 else (SCENES[1], gpu-1, 5)
    out = ROOT/name
    out.mkdir(parents=True, exist_ok=True)
    args = ['--scene', f'configs/consolidated/scannet_native/{name}.scene.json', '--queries', f'configs/consolidated/scannet_native/{name}.queries.json', '--shard', str(shard), '--shards', str(shards)]
    env = dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu))
    with (out/f'worker_{shard}.log').open('w') as log:
        subprocess.run([SAM, 'scripts/research/detect_consolidated_sources.py', *args, '--checkpoint', 'checkpoints/sam3_modelscope/sam3.pt', '--output', str(out/'detections'), '--height', '96', '--width', '128'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([NATIVE, 'scripts/research/build_category_source_evidence.py', *args, '--output', str(out)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, shard, 'source evidence finished', flush=True)


def finish(name, gpu):
    out = ROOT/name
    scene = json.loads(Path(f'configs/consolidated/scannet_native/{name}.scene.json').read_text())
    views = sorted([v for p in out.glob('views_*.json') for v in json.loads(p.read_text())['views']], key=lambda v: v['frame'])
    scene['native_category_prior'] = dict(kind='mutually_exclusive_native_log_odds', temperature=100., vocabulary='complete declared ScanNet category query list', historical_scene_parameters=False)
    scene['calibration'] = dict(kind='native_categorical_prior_not_empirically_calibrated')
    scene['source_evidence'] = dict(provider='SAM3 literal text with native foreground-context evidence', raster_hw=[96, 128], views=views)
    scene['category_evidence_frames'] = [v['frame'] for v in views]
    scene['source_query_batch_size'] = 4
    config = CONFIG/f'{name}.scene.json'
    config.write_text(json.dumps(scene, indent=2)+'\n')
    with (out/'evaluation.log').open('w') as log:
        subprocess.run([NATIVE, '-m', 'radio_gs.pipeline', '--scene', str(config), '--queries', f'configs/consolidated/scannet_native/{name}.queries.json', '--access', 'source-assisted', '--output', str(out/'query'), '--gpu', str(gpu)], env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([NATIVE, 'scripts/research/evaluate_scannet_runtime.py', '--readout', str(out/'query/category/typed_readout.pt'), '--evaluation', f'configs/consolidated/scannet_native/{name}.evaluation.json', '--output', str(out/'evaluation'), '--point-only'], env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, 'evaluated', flush=True)


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(worker, range(6)))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(finish, SCENES, [0, 1]))
    rows = {name: json.loads((ROOT/name/'evaluation/report.json').read_text()) for name in SCENES}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows, source_policy='all declared source frames for these two scenes; old eight-scene evidence has a different budget', new_scene_calibration_parameters=False), indent=2)+'\n')
