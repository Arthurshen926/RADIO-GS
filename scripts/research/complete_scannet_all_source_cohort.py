"""Build and evaluate one cold-category recipe on all ten declared source sets.

Shards partition work, never select views. Existing cold-completion evidence
for scene0200 and scene0645 is reused; every other scene is freshly generated.
"""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import queue
import subprocess
from run_scannet_cold_completion import ROOT, CONFIG, NATIVE, SAM, finish

SCENES = sorted(p.name.removesuffix('.scene.json') for p in Path('configs/consolidated/scannet_native').glob('*.scene.json'))


def main():
    tasks = queue.Queue()
    for name in SCENES:
        if name in ('scene0200_00', 'scene0645_00'):
            continue
        scene = json.loads(Path(f'configs/consolidated/scannet_native/{name}.scene.json').read_text())
        shards = math.ceil(len(scene['source_frames'])/80)
        for shard in range(shards):
            tasks.put((name, shard, shards))

    def worker(gpu):
        while True:
            try:
                name, shard, shards = tasks.get_nowait()
            except queue.Empty:
                return
            out = ROOT/name
            out.mkdir(parents=True, exist_ok=True)
            args = ['--scene', f'configs/consolidated/scannet_native/{name}.scene.json', '--queries', f'configs/consolidated/scannet_native/{name}.queries.json', '--shard', str(shard), '--shards', str(shards)]
            env = dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu))
            with (out/f'worker_{shard}.log').open('w') as log:
                subprocess.run([SAM, 'scripts/research/detect_consolidated_sources.py', *args, '--checkpoint', 'checkpoints/sam3_modelscope/sam3.pt', '--output', str(out/'detections'), '--height', '96', '--width', '128'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([NATIVE, 'scripts/research/build_category_source_evidence.py', *args, '--output', str(out)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(name, shard, 'complete', flush=True)

    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(worker, range(6)))

    def evaluate(gpu):
        for name in SCENES[gpu::6]:
            finish(name, gpu)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(evaluate, range(6)))
    reports = {name: json.loads((ROOT/name/'evaluation/report.json').read_text()) for name in SCENES}
    old_names = [n for n in SCENES if n not in ('scene0200_00', 'scene0645_00')]
    macro = lambda names, arm: {s: sum(reports[n]['point_metrics'][arm][s]['miou'] for n in names)/len(names) for s in ['19', '15', '10']}
    summary = dict(scenes=reports, full10={arm: macro(SCENES, arm) for arm in ['map', 'source']}, matched8={arm: macro(old_names, arm) for arm in ['map', 'source']}, source_policy='all 1158 declared source frames; same SAM3, native identity and cold prior on all ten scenes', historical_scene_parameters=False, independent_test=False, official_protocol_equivalence_verified=False)
    (ROOT/'full10_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    recipe = json.loads(Path('configs/pipeline/system_cold_category.json').read_text())
    recipe['name'] = 'RADIO-GS complete-source cold-category system'
    recipe['role'] = 'uniform all-source ScanNet10 alternative; LERF and NVOS retained unchanged'
    recipe['jobs'] = [j for j in recipe['jobs'] if j['benchmark'] != 'scannet']
    for name in SCENES:
        recipe['jobs'].append(dict(benchmark='scannet', name=name, scene=str(CONFIG/f'{name}.scene.json'), queries=f'configs/consolidated/scannet_native/{name}.queries.json', evaluation=f'configs/consolidated/scannet_native/{name}.evaluation.json', input_track='provided_unlabelled_mesh', outputs=['point_miou_19_15_10']))
    Path('configs/pipeline/system_complete_sources.json').write_text(json.dumps(recipe, indent=2)+'\n')
    print(json.dumps({key: summary[key] for key in ['full10', 'matched8']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
