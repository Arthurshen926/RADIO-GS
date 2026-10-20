"""Real, held-out-view image episodes on the existing 29 independent rooms.

Uses both ORIGINAL image teachers and original SAM masks. Streams per-view DINO
features; keeps no dense features, predictions, crops or duplicate scene maps.
Annotations are used only for independent query construction and IoU targets.
"""
import argparse
from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.spatial import cKDTree
import torch

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.benchmarks.uqis.construct import ConstructionConfig, projected_observation, registered_crop, choose_prompt
from radio_gs.models.native_region_appearance import NativeRegionAppearance, CHECKPOINT, REPOSITORY
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.querying.local_appearance_match import match_regions
from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership, view_balanced_transfer
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.response_replacement import RegionResponseEvidence, propose_regions, query_ranks
from radio_gs.querying.response_supervision import coverage_reference_views
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from scripts.research.replay_tmm_response import STUDY, POLICY, sha

CONFIG = Path('configs/research/response_replacement_joint.json')


def read(path):
    return json.loads(Path(path).read_text())


def masks_root(config, scene, frames):
    candidates = [Path(config['source'])/'maps'/scene/'source_masks',
                  Path(config['raw_source'])/'maps'/scene/'source_masks',
                  Path('output/referent_grounding_20261005/real_observation_control/maps')/scene/'source_masks']
    for root in candidates:
        if root.is_dir() and all((root/f'{f:06d}.pt').is_file() for f in frames):
            return root
    raise FileNotFoundError('original SAM masks missing for '+scene)


def construct_queries(sensor, refs, xyz, labels, queries, visibility, config, scene):
    construction = ConstructionConfig(); tree = cKDTree(xyz); best = {}
    targets = sorted({int(q['raw_object_id'])+1 for q in queries
                      if q['listener_correct'] and q['mentions_target_class']})
    supported = np.asarray(visibility.getnnz(axis=1)) > 0
    eligible = []
    for instance in targets:
        target = labels == instance
        views = np.count_nonzero(np.asarray(visibility[target].sum(0)))
        if target.any() and supported[target].mean() >= config['minimum_target_supported_fraction'] and views >= config['minimum_target_mapping_views']:
            eligible.append(instance)
    yy, xx = np.mgrid[:sensor.height, :sensor.width]
    kd, kc = sensor.k_depth, sensor.k_color
    u = kc[0, 0]*(xx-kd[0, 2])/kd[0, 0]+kc[0, 2]
    v = kc[1, 1]*(yy-kd[1, 2])/kd[1, 1]+kc[1, 2]
    visible = (u >= 0)&(v >= 0)&(u < sensor.color_width)&(v < sensor.color_height)
    margin = construction.minimum_rgb_border_px
    interior = (u >= margin)&(v >= margin)&(u < sensor.color_width-margin)&(v < sensor.color_height-margin)
    for frame in refs:
        _, _, rgb, world, distance, _, ids = projected_observation(sensor, frame, tree, labels)
        for instance in eligible:
            extent = (ids == instance)&(distance <= construction.extent_correspondence_m)&visible
            if extent.sum() < construction.minimum_visible_depth_pixels:
                continue
            reliable = extent&(distance <= construction.correspondence_m)&interior
            if choose_prompt(extent, reliable, world, construction) is None:
                continue
            crop, quality = registered_crop(rgb, extent, kd, kc, construction)
            sharp = quality['foreground_laplacian_energy']/(quality['foreground_laplacian_energy']+.0025)
            area = min(1., float(extent.sum())/(.10*extent.size))
            score = float((.1+.9*sharp)*np.sqrt(area)*(.1+.9*quality['usable_exposure_fraction'])*np.sqrt(.1+.9*quality['foreground_fraction']))
            if instance not in best or score > best[instance][0]['quality_score']:
                meta = dict(instance=instance, frame=frame, quality_score=score, quality=quality,
                    crop_rgb_sha256=hashlib.sha256(np.asarray(crop).tobytes()).hexdigest())
                best[instance] = (meta, crop)
    keys = sorted(best, key=lambda i: hashlib.sha256(f'{scene}/{i}/joint-response'.encode()).hexdigest())[:config['image_queries_per_scene']]
    if not keys:
        raise ValueError('no supported independent image queries in '+scene)
    return [best[k] for k in keys], dict(requested_targets=len(targets), mapping_supported_targets=len(eligible), visible_targets=len(best))


@torch.inference_mode()
def compile_room(scene, split, config, native, appearance):
    started = time.monotonic(); folder = Path(config['source'])/'maps'/scene
    spec = read(folder/'map.json'); original_frames = spec['source_frames']
    sensor_path = Path(config['raw_source'])/'raw'/scene/(scene+'.sens')
    sensor = SensorIndex(sensor_path)
    labels_path = Path(config['source'])/'labels'/f'{scene}.npy'
    language_path = Path(config['source'])/'language'/f'{scene}.json'
    labels = np.load(labels_path); language = read(language_path)
    original_visibility = sparse.load_npz(folder/'visibility.npz')
    targets = sorted({int(q['raw_object_id'])+1 for q in language if q['listener_correct'] and q['mentions_target_class']})
    refs, keep, excluded = coverage_reference_views(sensor, original_frames, original_visibility, labels, targets, config)
    masks = masks_root(config, scene, original_frames)
    paths = [folder/n for n in ('map.json', 'membership.npz', 'visibility.npz', 'region_offsets.npy', 'mesh_xyz.npy', 'descriptors.npy')]
    paths += [sensor_path, Path(config['source'])/'labels'/f'{scene}.npy', Path(config['source'])/'language'/f'{scene}.json']
    inputs = {str(p): sha(p) for p in paths}
    old_offsets = np.load(folder/'region_offsets.npy')
    columns = np.concatenate([np.arange(old_offsets[f], old_offsets[f+1]) for f in keep])
    offsets = np.r_[0, np.cumsum(np.diff(old_offsets)[keep])]
    membership = sparse.load_npz(folder/'membership.npz')[:, columns].tocsr()
    visibility = original_visibility[:, keep].tocsr()
    frames = np.repeat(np.arange(len(keep)), np.diff(offsets))
    transfer = view_balanced_transfer(membership, offsets)
    xyz = np.load(folder/'mesh_xyz.npy'); labels = np.load(paths[-2])
    queries, construction = construct_queries(sensor, refs, xyz, labels, read(paths[-1]), visibility, config, scene)
    descriptors = np.load(folder/'descriptors.npy')[columns].astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    vectors = []; query_patches = []; query_weights = []
    for meta, crop in queries:
        resized = crop.resize((384, 384), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float()[None]/255
        vectors.append(native.encode(pixels)[0].cpu().numpy())
        local, valid = appearance.patches(crop)
        query_patches.append(local.cpu().numpy()); query_weights.append(valid.cpu().numpy())
    native_scores = descriptors @ np.stack(vectors).T
    local_scores = np.zeros_like(native_scores); width = spec['configuration']['source_rgb_width']
    for index, f in enumerate(keep):
        frame = original_frames[f]; path = masks/f'{frame:06d}.pt'; inputs[str(path)] = sha(path)
        _, _, rgb = sensor.read(frame)
        rgb = rgb.resize((width, round(rgb.height*width/rgb.width)), Image.Resampling.LANCZOS)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        region_masks = unpack_masks(payload['packed_masks'], width)
        if len(region_masks) != offsets[index+1]-offsets[index]:
            raise ValueError('original SAM region order/count mismatch')
        if not len(region_masks):
            continue
        local, _ = appearance.patches(rgb)
        # The original local map persists fp16 patch descriptors, then matches
        # in fp32. Reproduce that quantization without persisting a new map.
        local = local.half().cpu().numpy()
        weights = sparse.csr_matrix(appearance.patch_weights(region_masks).cpu().numpy())
        for q0 in range(0, len(queries), 4):
            values = match_regions(local, weights, np.array([0, len(local)]), np.array([0, len(region_masks)]),
                query_patches[q0:q0+4], query_weights[q0:q0+4], device=appearance.device)['chamfer']
            local_scores[offsets[index]:offsets[index+1], q0:q0+4] = values
        if (index+1) % 20 == 0:
            print('teacher observations', split, scene, index+1, '/', len(keep), 'queries', len(queries), flush=True)
    reader = DirectAnchorMembership(membership, visibility, frames, xyz, descriptors, transfer, config['provider_checkpoint'])
    reader.saved['estimator'].n_jobs = 4
    scope = InstanceObservationScope(membership, visibility, frames)
    evidence = RegionResponseEvidence(membership, visibility, frames)
    supported = transfer.getnnz(axis=1) > 0
    cache = OrderedDict(); rows = []
    for i, (meta, _) in enumerate(queries):
        scores = np.stack([native_scores[:, i], local_scores[:, i]], 1)
        joint = query_ranks(scores, evidence.available).mean(1).astype(np.float32)
        seeds = propose_regions(scores, evidence.available, frames, budget=config['candidate_budget'])
        semantic = np.clip((np.asarray(transfer @ native_scores[:, i])+1)*.5, 0, 1).astype(np.float32)
        semantic[~supported] = 0
        decoded = []; providers = []; proposal_seeds = []
        for seed in seeds:
            if seed not in cache:
                cache[seed] = reader.infer_region(seed)
                if len(cache) > 64:
                    cache.popitem(last=False)
            cache.move_to_end(seed); decoded.append(cache[seed]); providers.append('direct'); proposal_seeds.append(seed)
            _, rank, sets, _ = scope.infer(joint, semantic, seed=seed)
            decoded.append((rank.astype(np.float32), sets['identity_scope'], dict(anchor_vertex=None)))
            providers.append('scope'); proposal_seeds.append(seed)
        probability = np.stack([r[0] for r in decoded]); selected = np.stack([r[1] for r in decoded])
        excluded_views = sorted(set(frames[seeds].tolist()))
        x, info = evidence.encode(probability, selected, scores, exclude_views=excluded_views)
        control, _ = evidence.encode(probability, selected, scores, exclude_views=excluded_views, permuted=True)
        truth = labels == meta['instance']
        ious = ((selected & truth).sum(1)/np.maximum((selected | truth).sum(1), 1)).tolist()
        rows.append(dict(scene=scene, modality='image', query_index=i, reference=meta, seeds=proposal_seeds,
            providers=providers, anchors=[r[2]['anchor_vertex'] for r in decoded], features=x.tolist(),
            permuted_features=control.tolist(), ious=ious, evidence=info,
            prediction_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
            selection_sha256=hashlib.sha256(selected.tobytes()).hexdigest()))
    report = dict(scene=scene, split=split, modality='image', queries=len(rows), rows=rows, input_hashes=inputs,
        construction=construction, reference_frames=refs, excluded_sensor_frames=excluded.tolist(),
        retained_source_frames=np.asarray(original_frames)[keep].tolist(), original_source_views=len(original_frames),
        teacher_score_sha256=hashlib.sha256(np.stack([native_scores, local_scores]).tobytes()).hexdigest(),
        original_teachers_and_masks=True, reference_views_excluded_from_all_providers=True,
        benchmark_labels_used=False, dense_predictions_retained=False, seconds=time.monotonic()-started)
    print('image room complete', split, scene, len(rows), 'seconds', round(report['seconds'], 1), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='response_replacement_images_v2'); p.add_argument('--resume', action='store_true')
    p.add_argument('--scenes', nargs='*'); p.add_argument('--device', default='cuda')
    args = p.parse_args(); torch.set_num_threads(4); config = read(CONFIG)
    design = read(Path(config['source'])/'design.json'); splits = {k: design[k] for k in ('fit', 'calibration', 'validation')}
    rooms = sum(splits.values(), [])
    benchmark = {j['name'] for j in read('configs/pipeline/system_v10.json')['jobs']}
    if len(set(rooms)) != len(rooms) or set(rooms)&benchmark:
        raise ValueError('independent supervision scenes overlap')
    sources = [str(CONFIG), __file__, 'radio_gs/querying/response_supervision.py', 'radio_gs/querying/response_replacement.py',
        'radio_gs/querying/local_appearance_match.py', 'radio_gs/models/native_region_appearance.py',
        'radio_gs/benchmarks/uqis/construct.py', 'radio_gs/benchmarks/uqis/sensor.py',
        'radio_gs/querying/direct_anchor_membership.py', 'radio_gs/querying/instance_observation_scope.py',
        'radio_gs/querying/anchor_observation_evidence.py', 'radio_gs/querying/observation_readout.py',
        'radio_gs/scripts/build_multiscale_sam_mask_aligned_crop_summary_teacher.py',
        'radio_gs/scripts/build_sam3_automatic_mask_cache.py']
    manifest = dict(config=config, splits=splits, source_hashes={s: sha(s) for s in sources},
        input_hashes={config['provider_checkpoint']: sha(config['provider_checkpoint']), CHECKPOINT: sha(CHECKPOINT)},
        no_benchmark_feedback=True, final_test=False)
    run = BoundedExperiment(STUDY, args.run, POLICY, resume=args.resume); run.preflight(24*2**20)
    if args.resume:
        if read(run.root/'manifest.json') != manifest:
            raise ValueError('resume implementation/configuration changed')
    else:
        run.json('manifest.json', manifest); run.json('source_snapshot.json', {s: Path(s).read_text() for s in sources})
    pending = [(split, scene) for split, scenes in splits.items() for scene in scenes
               if not (run.root/(scene+'.json')).exists() and (not args.scenes or scene in args.scenes)]
    if pending:
        model_path = read(Path(config['source'])/'maps'/pending[0][1]/'map.json')['native_encoder']
        native = NativeSiglip2Runtime.load(Path(model_path), device=torch.device(args.device))
        appearance = NativeRegionAppearance(device=args.device)
        if not (run.root/'teacher_seal.json').exists():
            run.json('teacher_seal.json', dict(siglip=native.bundle, dino_checkpoint=CHECKPOINT, dino_repository=REPOSITORY))
        for split, scene in pending:
            run.json(scene+'.json', compile_room(scene, split, config, native, appearance)); gc.collect(); torch.cuda.empty_cache()
    complete = all((run.root/(s+'.json')).exists() for s in rooms)
    progress = dict(complete=complete, scenes={s: read(run.root/(s+'.json'))['queries'] for s in rooms if (run.root/(s+'.json')).exists()},
                    storage_bytes=tree_bytes(run.root), dense_artifacts_saved=False)
    run.json('progress.json', progress, replace=True)
    if complete and not (run.root/'summary.json').exists():
        run.json('summary.json', progress)
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == '__main__':
    main()
