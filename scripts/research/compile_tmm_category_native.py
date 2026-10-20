"""Original Mask-Adapter responses on all 29 independent-room region carriers.

Only 19-column query responses are retained. High-dimensional image features
and dense point predictions remain transient. No semantic labels are read.
"""
import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.models.pretrained_mask_adapter import PretrainedMaskAdapter
from radio_gs.pipeline.category_union_adapter import CHECKPOINT
from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scannet_constants import NYU40_ID_TO_NAME
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_study_adapters import read
from scripts.research.compile_tmm_image_supervision import masks_root
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/category_response_supervision.json')


def stored_native_cosines(descriptors, query):
    """Reproduce the production descriptor/query storage boundary before cosine."""
    descriptor = F.normalize(descriptors.half().float(), dim=-1)
    query = F.normalize(query.half().float(), dim=-1)
    return descriptor @ query.T


@torch.inference_mode()
def compile_room(scene, split, config, teacher, query, run):
    start = time.monotonic(); folder = Path(config['source'])/'maps'/scene
    spec = read(folder/'map.json'); frames = spec['source_frames']
    offsets = np.load(folder/'region_offsets.npy')
    masks_folder = masks_root(config, scene, frames)
    sensor_path = Path(config['raw_source'])/'raw'/scene/(scene+'.sens')
    sensor = SensorIndex(sensor_path)
    paths = [folder/'map.json', folder/'region_offsets.npy', sensor_path]
    inputs = {str(p): sha(p) for p in paths}
    scores = np.empty((int(offsets[-1]), len(query)), np.float32)
    records = []
    torch.cuda.reset_peak_memory_stats()
    for index, frame in enumerate(frames):
        path = masks_folder/f'{frame:06d}.pt'; inputs[str(path)] = sha(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        masks = unpack_masks(payload['packed_masks'], payload['mask_shape'][1]).copy()
        if len(masks) != offsets[index+1]-offsets[index]:
            raise ValueError('original native-region order/count does not match SAM carrier')
        _, _, rgb = sensor.read(frame)
        rgb_array = np.array(rgb, copy=True)
        if len(masks):
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()/255
            encoded = teacher.encode(rgb_tensor, torch.from_numpy(masks))['mask_adapter']
            values = stored_native_cosines(encoded, query).cpu().numpy()
            scores[offsets[index]:offsets[index+1]] = values
            del encoded, rgb_tensor, values
        records.append(dict(frame=frame, regions=len(masks),
            rgb_sha256=hashlib.sha256(rgb_array.tobytes()).hexdigest()))
        if (index+1) % 10 == 0 or index+1 == len(frames):
            print('native', split, scene, index+1, '/', len(frames), 'seconds', round(time.monotonic()-start, 1), flush=True)
    if not np.isfinite(scores).all() or np.any(abs(scores) > 1.00001):
        raise ValueError('invalid native query responses')
    path = run.root/(scene+'.npy')
    run.public_artifact(path, scores, 'numpy')
    report = dict(scene=scene, split=split, frames=len(frames), regions=len(scores), class_ids=config['class_ids'],
        response_file=path.name, response_sha256=sha(path), input_hashes=inputs, records=records,
        seconds=time.monotonic()-start, cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        all_original_views=True, original_masks_unchanged=True, native_quantization_reproduced=True,
        labels_read=False, benchmark_answers_read=False, high_dimensional_features_saved=False,
        category_source_evidence_ready=False)
    run.json(scene+'.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='category_native_responses_v1')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--scenes', nargs='+', help='disjoint room shards; no room sampling inside a room')
    args = p.parse_args(); torch.set_num_threads(4)
    config = read(CONFIG); design = read(Path(config['source'])/'design.json')
    splits = {k: design[k] for k in ('fit', 'calibration', 'validation')}
    rooms = [(scene, split) for split, scenes in splits.items() for scene in scenes]
    if args.scenes and not set(args.scenes).issubset({s for s, _ in rooms}):
        raise ValueError('undeclared room shard')
    sources = [str(CONFIG), __file__, 'radio_gs/models/pretrained_mask_adapter.py',
        'radio_gs/models/native_query_encoders.py', 'radio_gs/scripts/build_sam3_automatic_mask_cache.py',
        'radio_gs/benchmarks/uqis/sensor.py', 'scripts/research/compile_tmm_image_supervision.py']
    manifest = dict(config=config, splits=splits, source_hashes={p: sha(p) for p in sources},
        checkpoint=CHECKPOINT, checkpoint_sha256=sha(CHECKPOINT),
        design_sha256=sha(Path(config['source'])/'design.json'), labels_read=False,
        target_manifest_sha256=sha(STUDY/config['target_run']/'manifest.json'))
    run = BoundedExperiment(STUDY, args.run, POLICY, resume=args.resume)
    total_regions = sum(read(Path(config['source'])/'maps'/s/'map.json')['regions'] for s, _ in rooms)
    run.preflight(total_regions*19*4+3*2**20 if not args.resume else 2**20)
    if args.resume:
        if read(run.root/'manifest.json') != manifest:
            raise ValueError('resume changed teacher, protocol or source')
    else:
        run.json('manifest.json', manifest)
        run.json('source_snapshot.json', {p: dict(text=Path(p).read_text(), sha256=sha(p)) for p in sources})
    pending = [(s, split) for s, split in rooms if (not args.scenes or s in args.scenes)
               and not (run.root/(s+'.json')).exists()]
    if pending:
        teacher = PretrainedMaskAdapter(CHECKPOINT)
        query = teacher.text([NYU40_ID_TO_NAME[c] for c in config['class_ids']])
        for scene, split in pending:
            compile_room(scene, split, config, teacher, query, run)
            gc.collect(); torch.cuda.empty_cache()
        del teacher, query
    completed = []
    for scene, _ in rooms:
        path = run.root/(scene+'.json')
        if path.exists():
            report = read(path)
            if sha(run.root/report['response_file']) != report['response_sha256']:
                raise ValueError('native cache changed after room completion')
            completed.append(report)
    if len(completed) == len(rooms) and not (run.root/'summary.json').exists():
        run.json('summary.json', dict(complete=True, rooms=len(completed),
            frames=sum(r['frames'] for r in completed), regions=sum(r['regions'] for r in completed),
            source_category_stage_complete=False, utility_training_ready=False,
            labels_read=False, high_dimensional_features_saved=False, dense_point_predictions_saved=False,
            report_hashes={r['scene']+'.json': sha(run.root/(r['scene']+'.json')) for r in completed},
            study_bytes=tree_bytes(STUDY)))
    print('completed native rooms', len(completed), '/', len(rooms), flush=True)


if __name__ == '__main__':
    main()
