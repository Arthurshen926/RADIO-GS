"""Compile full-teacher complete-menu utility supervision on separate rooms.

SAM3 masks stream from a separate environment; SigLIP foreground/context scores
and original dual-native fusion run here. Dense responses exist only per room.
The registered training carrier differs from benchmark Gaussian rendering.
"""
import argparse
import gc
import hashlib
import io
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.pipeline.registered_observation_map import visible_vertices
from radio_gs.models.native_query_encoders import encode_category_text
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.querying.category_response_supervision import native_menu, confidence_ratio, complete_candidates
from radio_gs.querying.response_replacement import RegionResponseEvidence, FEATURE_NAMES
from radio_gs.data.scannet_source_region_semantics import load_scannet_raw_to_nyu40, official_vertex_nyu40_labels
from radio_gs.scannet_constants import NYU40_ID_TO_NAME, OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS, compute_split_metrics
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.querying.response_study_adapters import read
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from scripts.research.prepare_tmm_cross_task_targets import MAPPING, RAW, view_folds

CONFIG = Path('configs/research/category_response_supervision.json')


def receive(stream):
    header = stream.read(8)
    if len(header) != 8:
        raise RuntimeError('SAM3 worker closed before all declared source views')
    size = struct.unpack('<Q', header)[0]
    if size > 32*2**20:
        raise ValueError('source-view payload exceeds bounded transport')
    parts = []; left = size
    while left:
        data = stream.read(left)
        if not data:
            raise RuntimeError('incomplete source-view payload')
        parts.append(data); left -= len(data)
    return torch.load(io.BytesIO(b''.join(parts)), map_location='cpu', weights_only=False)


def menu_signature(probabilities, supported, membership, visibility, offsets, scores, evidence_views):
    columns = np.concatenate([np.arange(offsets[f], offsets[f+1]) for f in evidence_views])
    frames = np.repeat(np.arange(len(evidence_views)), np.diff(offsets)[evidence_views])
    encoder = RegionResponseEvidence(membership[:, columns], visibility[:, evidence_views], frames)
    chosen = probabilities.argmax(2)
    full, permuted = [], []
    for c in range(probabilities.shape[2]):
        selected = (chosen == c)&supported[None]
        x, info = encoder.encode(probabilities[:, :, c], selected, scores[columns, c])
        xp, _ = encoder.encode(probabilities[:, :, c], selected, scores[columns, c], permuted=True)
        full.append(x); permuted.append(xp)
    # One whole-menu signature; no labels select present classes or class winners.
    return np.mean(full, 0), np.mean(permuted, 0), info


@torch.inference_mode()
def compile_room(scene, split, config, runtime, query, stream, native_run, run):
    start = time.monotonic(); folder = Path(config['source'])/'maps'/scene
    spec = read(folder/'map.json'); frames = spec['source_frames']; classes = config['class_ids']
    xyz = np.load(folder/'mesh_xyz.npy'); n, c = len(xyz), len(classes)
    offsets = np.load(folder/'region_offsets.npy')
    membership = sparse.load_npz(folder/'membership.npz').tocsr()
    visibility = sparse.load_npz(folder/'visibility.npz').tocsc()
    transfer = sparse.load_npz(folder/'transfer.npz')
    descriptor = torch.from_numpy(np.load(folder/'descriptors.npy').astype(np.float32)).cuda()
    siglip_scores = (F.normalize(descriptor, dim=-1) @ query.T).cpu().numpy(); del descriptor
    native_receipt = read(native_run/(scene+'.json'))
    if sha(native_run/(scene+'.npy')) != native_receipt['response_sha256']:
        raise ValueError('Mask-Adapter class-response cache changed')
    auxiliary_scores = np.load(native_run/(scene+'.npy'))
    if auxiliary_scores.shape != siglip_scores.shape:
        raise ValueError('dual teachers differ in original region/menu order')
    query_scores = np.stack([siglip_scores, auxiliary_scores], -1)
    folds = view_folds(frames); assignment = np.empty(len(frames), int)
    for fold in folds:
        assignment[fold['evidence_view_indices']] = fold['fold']
    # One set of sufficient statistics per chronological block. No mask bank.
    numerators = np.zeros((3, n, c), np.float32)
    masses = np.zeros_like(numerators); visibilities = np.zeros_like(numerators)
    sensor_path = Path(config['raw_source'])/'raw'/scene/(scene+'.sens'); sensor = SensorIndex(sensor_path)
    records = []
    for index, frame in enumerate(frames):
        payload = receive(stream)
        if payload['scene'] != scene or payload['frame'] != frame:
            raise ValueError('SAM stream does not follow declared source views')
        rows = payload['rows']; pose, depth, rgb = sensor.read(frame)
        rgb_array = np.array(rgb, copy=True); rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()/255
        for qi, row in enumerate(rows):
            row['margin'] = torch.full((len(row['scores']),), -1.)
            for j, mask in enumerate(row['masks']):
                if not (mask > .5).any():
                    continue
                full_mask = F.interpolate((mask > .5)[None, None].float(), rgb_tensor.shape[1:], mode='nearest-exact')[0, 0].bool()
                triple, _ = same_window(rgb_tensor, full_mask)
                feature = F.normalize(runtime.encode(triple).float(), dim=-1)
                row['margin'][j] = (feature[1] @ query[qi]-feature[2] @ query[qi]).cpu()
        target = torch.stack([row['masks'].float().amax(0) if len(row['scores']) else torch.zeros(96, 128) for row in rows], 0)
        binary = (target[:, None] > .5).float()
        positive = (-F.max_pool2d(-binary, 3, 1, 1)[:, 0] > .5).flatten(1).T
        negative = (F.max_pool2d(binary, 3, 1, 1)[:, 0] < .5).flatten(1).T
        ids, u, v = visible_vertices(xyz, pose, depth, sensor, (128, 96), spec['configuration']['depth_tolerance_m'])
        pixels = v*128+u; counts = np.bincount(pixels, minlength=96*128)
        detector = np.array([float(row['scores'].max()) if len(row['scores']) else 0. for row in rows], np.float32)
        known = (positive|negative).numpy() & (detector[None] > 0)
        known_counts = (known*(counts[:, None] > 0)).sum(0)
        weight = known[pixels]*detector[None]/np.maximum(counts[pixels, None], 1)
        block = assignment[index]
        numerators[block, ids] += target.flatten(1).T.numpy()[pixels]*weight
        masses[block, ids] += weight
        visibilities[block, ids] += (detector[None] > 0)/np.maximum(counts[pixels, None], 1)
        records.append(dict(frame=frame, visible_vertices=len(ids),
            rgb_sha256=hashlib.sha256(rgb_array.tobytes()).hexdigest(),
            detection_sha256=hashlib.sha256(b''.join(row['masks'].numpy().tobytes()+row['scores'].numpy().tobytes() for row in rows)).hexdigest(),
            identity=[dict(score=row['scores'].tolist(), margin=row['margin'].tolist(), known_pixels=int(known_counts[i])) for i, row in enumerate(rows)]))
        if (index+1) % 10 == 0 or index+1 == len(frames):
            print('source', split, scene, index+1, '/', len(frames), 'seconds', round(time.monotonic()-start, 1), flush=True)
    episodes = []
    # Full-map deployment-style example + three truly disjoint input-view folds.
    # Full-map evidence is reused. Fold geometry is still the official full mesh.
    arrangements = [dict(fold='full', candidate_view_indices=list(range(len(frames))), evidence_view_indices=list(range(len(frames))))] + folds
    for arrangement in arrangements:
        keep = arrangement['candidate_view_indices']; evidence = arrangement['evidence_view_indices']
        regions = np.concatenate([np.arange(offsets[f], offsets[f+1]) for f in keep])
        siglip, supported = native_menu(transfer, siglip_scores, regions)
        auxiliary, _ = native_menu(transfer, auxiliary_scores, regions)
        blocks = sorted(set(assignment[keep].tolist()))
        ratio = confidence_ratio(records, keep, c)
        probability = complete_candidates(siglip, auxiliary, numerators[blocks].sum(0),
            masses[blocks].sum(0), visibilities[blocks].sum(0), ratio)
        probability[:, ~supported] = 0
        x, xp, info = menu_signature(probability, supported, membership, visibility, offsets, query_scores, evidence)
        prediction_hash = hashlib.sha256(probability.tobytes()).hexdigest()
        # Only now open the independent target labels; none enter candidates or features.
        labels, _ = official_vertex_nyu40_labels(scene_id=scene, vertex_count=n,
            segmentation=read(RAW/scene/(scene+'_vh_clean_2.0.010000.segs.json')),
            aggregation=read(RAW/scene/(scene+'.aggregation.json')), raw_to_nyu40=load_scannet_raw_to_nyu40(MAPPING))
        metrics = {}
        for subset, ids in SPLITS.items():
            columns = [classes.index(i) for i in ids]
            predictions = np.asarray(ids)[probability[:, :, columns].argmax(2)]
            predictions[:, ~supported] = 0
            metrics[subset] = [compute_split_metrics(pred, labels, ids)['miou'] for pred in predictions]
        episodes.append(dict(scene=scene, query_index=len(episodes), fold=arrangement['fold'],
            features=x.tolist(), permuted_features=xp.tolist(), ious=metrics['19'], menu_utilities=metrics,
            prediction_sha256=prediction_hash, evidence=info,
            candidate_views=len(keep), evidence_views=len(evidence),
            observation_views_disjoint=not bool(set(keep)&set(evidence)), geometry_held_out=False,
            complete_menu=True, no_classwise_selection=True))
        del probability
    input_paths = [folder/n for n in ('map.json', 'mesh_xyz.npy', 'membership.npz', 'visibility.npz', 'transfer.npz', 'descriptors.npy', 'region_offsets.npy')]
    input_paths += [sensor_path, native_run/(scene+'.json'), native_run/(scene+'.npy'), MAPPING,
        RAW/scene/(scene+'_vh_clean_2.0.010000.segs.json'), RAW/scene/(scene+'.aggregation.json')]
    report = dict(scene=scene, split=split, rows=episodes, queries=len(episodes), records=records,
        input_hashes={str(p): sha(p) for p in input_paths}, seconds=time.monotonic()-start,
        inherited_full_mesh_geometry=True, carrier='registered mesh vertices with depth-tested normalized pixel contributions',
        benchmark_gaussian_renderer_equivalence=False, all_original_views=True,
        teachers=['SigLIP2 giant', 'Mask-Adapter ConvNeXt CLIP', 'literal-category SAM3'],
        complete_candidates=['dual_native_source', 'dual_native', 'source_readout'],
        new_dense_prediction_bytes=0, new_mask_bytes=0, new_high_dimensional_feature_bytes=0,
        benchmark_labels_used=False, official_independent_labels_used_for_utilities=True)
    run.json(scene+'.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='category_utility_supervision_v1'); p.add_argument('--resume', action='store_true')
    p.add_argument('--native-run', default='category_native_responses_v1'); p.add_argument('--sam-gpu', default='2')
    p.add_argument('--scenes', nargs='+')
    args = p.parse_args(); torch.set_num_threads(4)
    config = read(CONFIG); design = read(Path(config['source'])/'design.json')
    splits = {k: design[k] for k in ('fit', 'calibration', 'validation')}
    rooms = [(s, split) for split, scenes in splits.items() for s in scenes]
    if args.scenes and not set(args.scenes).issubset({s for s, _ in rooms}):
        raise ValueError('undeclared supervision room')
    sources = [str(CONFIG), __file__, 'scripts/research/stream_tmm_category_detections.py',
        'radio_gs/querying/category_response_supervision.py', 'radio_gs/querying/response_replacement.py',
        'radio_gs/models/native_query_encoders.py', 'radio_gs/querying/query_region_evidence.py',
        'radio_gs/pipeline/registered_observation_map.py', 'radio_gs/scripts/build_sam3_foundation_cache.py']
    native_run = STUDY/args.native_run
    manifest = dict(config=config, splits=splits, feature_names=FEATURE_NAMES,
        source_hashes={p: sha(p) for p in sources},
        native_manifest_sha256=sha(native_run/'manifest.json'),
        sam_checkpoint='/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt',
        sam_checkpoint_sha256=sha('/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt'),
        calibration_gap_retained=read(STUDY/'cross_task_targets_v1/summary.json')['missing_calibration_classes'],
        stage='complete-menu supervision, not benchmark performance', independent_final_test=False)
    run = BoundedExperiment(STUDY, args.run, POLICY, resume=args.resume); run.preflight(10*2**20)
    if args.resume:
        if read(run.root/'manifest.json') != manifest:
            raise ValueError('supervision protocol changed on resume')
    else:
        run.json('manifest.json', manifest)
        run.json('source_snapshot.json', {p: dict(text=Path(p).read_text(), sha256=sha(p)) for p in sources})
    pending = [(s, split) for s, split in rooms if (not args.scenes or s in args.scenes) and not (run.root/(s+'.json')).exists()]
    if pending:
        if any(not (native_run/(s+'.json')).exists() for s, _ in pending):
            raise ValueError('original Mask-Adapter responses must be complete before source compilation')
        spec = read(Path(config['source'])/'maps'/pending[0][0]/'map.json')
        query = encode_category_text({'text_encoder': spec['native_encoder']}, [NYU40_ID_TO_NAME[c] for c in config['class_ids']])
        runtime = NativeSiglip2Runtime.load(Path(spec['native_encoder']), device=torch.device('cuda'))
        if not (run.root/'teacher_bundle.json').exists():
            run.json('teacher_bundle.json', runtime.bundle)
        command = ['/root/miniconda3/envs/sam3/bin/python', 'scripts/research/stream_tmm_category_detections.py',
                   '--config', str(CONFIG), '--scenes', *[s for s, _ in pending]]
        worker = subprocess.Popen(command, stdout=subprocess.PIPE,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.sam_gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'))
        try:
            for scene, split in pending:
                compile_room(scene, split, config, runtime, query, worker.stdout, native_run, run)
                gc.collect(); torch.cuda.empty_cache()
            if worker.wait() != 0:
                raise RuntimeError('SAM3 worker failed')
        finally:
            if worker.poll() is None:
                worker.terminate(); worker.wait()
            worker.stdout.close()
    completed = [read(run.root/(s+'.json')) for s, _ in rooms if (run.root/(s+'.json')).exists()]
    if len(completed) == len(rooms) and not (run.root/'summary.json').exists():
        run.json('summary.json', dict(complete=True, rooms=len(completed),
            episodes=sum(r['queries'] for r in completed), room_counts={s: len(v) for s, v in splits.items()},
            all_original_views=True, all_applicable_category_teachers=True, native_and_source_supervision_ready=True,
            report_hashes={r['scene']+'.json': sha(run.root/(r['scene']+'.json')) for r in completed},
            dense_prediction_bytes=0, new_mask_bytes=0, high_dimensional_feature_bytes=0,
            benchmark_gaussian_renderer_equivalence=False, independent_final_test=False,
            calibration_gap_retained=manifest['calibration_gap_retained'], study_bytes=tree_bytes(STUDY)))
    print('complete category utility rooms', len(completed), '/', len(rooms), flush=True)


if __name__ == '__main__':
    main()
