"""Controlled natural-vs-gray crop input, complete native map and source readout.

The bbox, teacher, mask set, cameras, source fusion and evaluator stay fixed.
This changes teacher input, not a same-evidence algorithmic contribution.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

PYTHON = '/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT = Path('output/native_crop_context')
CONFIG = Path('configs/pipeline/native_crop_context')
BASE = Path('configs/pipeline/scannet_native_complete')


def rebuild(name):
    import torch
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.pipeline.native_reconstruction import NativeFieldAccumulator
    from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
    torch.set_num_threads(4)
    scene = json.loads((BASE / f'{name}.scene.json').read_text())
    out = ROOT / name
    binding = scene['geometry']
    geometry = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    accumulator = NativeFieldAccumulator(geometry, 1536)
    cameras = {c['frame_index']: c for c in json.loads(Path(scene['source_cameras']).read_text())['records']}
    records = []
    started = time.monotonic()
    for frame in scene['source_frames']:
        row = torch.load(out / 'regions' / f'frame_{frame:05d}.pt', map_location='cpu', weights_only=False)
        sam = torch.load(row['sam_path'], map_location='cpu', weights_only=False)
        masks = unpack_masks(sam['packed_masks'], sam['mask_shape'][1])[row['native_proposal_indices'].numpy()].copy()
        c = cameras[frame]
        h, w = row['native_mask_hw']
        record = accumulator.add_view(masks, row['descriptors'], c['intrinsic'], c['world_to_camera'],
            height=round(h * 256 / w), width=256, image_height=c['height'], image_width=c['width'])
        records.append(dict(frame=frame, **record))
        print(name, frame, 'written', flush=True)
    field = accumulator.field(binding, 'siglip')
    original = torch.load(scene['native_field'], map_location='cpu', weights_only=False)
    support_difference = int((field['supported'] != original['supported']).sum())
    torch.save(field, out / 'native_field.pt')
    scene['native_field'] = str(out / 'native_field.pt')
    scene['native_region_directories'] = [str(out / 'regions')]
    scene['map_contract']['native_crop_policy'] = 'natural_same_bbox'
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / f'{name}.scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    (out / 'build.json').write_text(json.dumps(dict(records=records, support_difference=support_difference,
        seconds=time.monotonic()-started, geometry_changed=False, labels_read=False, source_evidence_changed=False), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild')
    args = parser.parse_args()
    if args.rebuild:
        rebuild(args.rebuild)
        return
    scenes = sorted(BASE.glob('*.scene.json'))
    def worker(gpu):
        env = dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(gpu))
        for path in scenes[gpu::6]:
            name = path.name.removesuffix('.scene.json')
            out = ROOT / name
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'run.log').open('w') as log:
                def run(*argv):
                    subprocess.run([PYTHON, *argv], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                run('-m', 'radio_gs.pipeline.region_teacher', '--scene', str(path), '--output', str(out/'regions'), '--crop-policy', 'natural')
                run(__file__, '--rebuild', name)
                run('-m', 'radio_gs.pipeline', '--scene', str(CONFIG/path.name), '--queries', f'configs/consolidated/scannet_native/{name}.queries.json', '--access', 'source-assisted', '--output', str(out/'query'), '--gpu', str(gpu))
                run('scripts/research/evaluate_scannet_runtime.py', '--readout', str(out/'query/category/typed_readout.pt'), '--evaluation', f'configs/consolidated/scannet_native/{name}.evaluation.json', '--output', str(out/'evaluation'), '--point-only')
            print(name, 'complete', flush=True)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(worker, range(6)))
    rows = {p.name.removesuffix('.scene.json'): json.loads((ROOT/p.name.removesuffix('.scene.json')/'evaluation/report.json').read_text())['point_metrics'] for p in scenes}
    macro = {a: {s: sum(r[a][s]['miou'] for r in rows.values()) / len(rows) for s in ['19', '15', '10']} for a in ['map', 'source']}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows, macro=macro, independent_test=False, changed_teacher_input=True), indent=2)+'\n')
    print(macro, flush=True)

if __name__ == '__main__':
    main()
