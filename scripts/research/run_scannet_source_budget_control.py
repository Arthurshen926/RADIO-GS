"""Same fresh detector/identity tensors, old176 vs full886 source evidence.

Scene calibration, native maps, class prior and evaluator are identical. The
source subset comes from the frozen old input declaration, never label scores.
"""
import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import subprocess
from run_scannet_cold_completion import ROOT, CONFIG, NATIVE

OUT = Path('output/scannet_source_budget_control')
CONFIG_OUT = Path('configs/pipeline/scannet_source_budget_control')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1,2,3,4,5')
    gpus = parser.parse_args().gpus.split(',')
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.mkdir(parents=True, exist_ok=True)
    old = sorted(Path('configs/pipeline/scannet_cold_category').glob('*.scene.json'))
    def worker(index):
        gpu = gpus[index]
        for p in old[index::len(gpus)]:
            name = p.name.removesuffix('.scene.json')
            previous = json.loads(p.read_text())
            scene = json.loads((CONFIG/p.name).read_text())
            frames = set(previous['category_evidence_frames'])
            scene['source_evidence']['views'] = [v for v in scene['source_evidence']['views'] if v['frame'] in frames]
            scene['category_evidence_frames'] = sorted(frames)
            config = CONFIG_OUT/p.name
            config.write_text(json.dumps(scene, indent=2)+'\n')
            out = OUT/name
            out.mkdir(parents=True, exist_ok=True)
            with (out/'run.log').open('w') as log:
                subprocess.run([NATIVE, '-m', 'radio_gs.pipeline', '--scene', str(config), '--queries', f'configs/consolidated/scannet_native/{name}.queries.json', '--access', 'source-assisted', '--output', str(out/'query'), '--gpu', str(gpu)], env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([NATIVE, 'scripts/research/evaluate_scannet_runtime.py', '--readout', str(out/'query/category/typed_readout.pt'), '--evaluation', f'configs/consolidated/scannet_native/{name}.evaluation.json', '--output', str(out/'evaluation'), '--point-only'], env=dict(os.environ, PYTHONPATH='.'), stdout=log, stderr=subprocess.STDOUT, check=True)
            print(name, 'matched-source evaluated', flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))
    rows = {}
    for p in old:
        name = p.name.removesuffix('.scene.json')
        rows[name] = {arm: json.loads((root/name/'evaluation/report.json').read_text())['point_metrics']['source'] for arm, root in [('matched', OUT), ('all', ROOT)]}
    macro = {arm: {s: sum(r[arm][s]['miou'] for r in rows.values())/len(rows) for s in ['19', '15', '10']} for arm in ['matched', 'all']}
    (OUT/'summary.json').write_text(json.dumps(dict(scenes=rows, macro=macro, source_counts=dict(matched=176, all=886), comparison='same fresh native identity and detector tensors; only source frame set changes', independent_test=False), indent=2)+'\n')
    print(json.dumps(macro, indent=2), flush=True)


if __name__ == '__main__':
    main()
