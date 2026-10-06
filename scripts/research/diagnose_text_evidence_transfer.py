"""Trace fixed source evidence to both LERF error domains, after inference.

Teacher witnesses are not true identity labels. The partitions describe where
evidence is missing or mixed, not causal fractions or an oracle performance bound.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.querying.typed_evidence import visible_operator
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES, load_binary_mask
from scripts.research.decode_expanded_extent_next72 import sample_operator_rows


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def describe(gt, prediction, evidence, valid):
    gt = torch.as_tensor(gt, device='cuda', dtype=torch.bool)
    prediction = torch.as_tensor(prediction, device='cuda', dtype=torch.bool)
    tp = int((gt & prediction).sum())
    fn, fp = gt & ~prediction, ~gt & prediction
    remaining = fn.clone()
    parts = {}
    conditions = [
        ('unsupported_geometry', ~valid),
        ('no_raw_positive_witness', evidence[..., 0] <= 1e-8),
        ('positive_removed_by_erosion', evidence[..., 1] <= 1e-8),
        ('positive_witness_minority', evidence[..., 1] <= .5),
        ('no_source_view_majority_positive', evidence[..., 3] <= .5),
        ('positive_supported_but_not_recalled', torch.ones_like(gt)),
    ]
    for name, condition in conditions:
        part = remaining & condition
        parts[name] = int(part.sum())
        remaining &= ~part
    return dict(tp=tp, fn=int(fn.sum()), fp=int(fp.sum()), positives=int(gt.sum()),
                predicted=int(prediction.sum()), fn_partition=parts,
                fp_with_positive_witness_majority=int((fp & (evidence[..., 1] > .5)).sum()),
                fp_with_source_view_majority_positive=int((fp & (evidence[..., 3] > .5)).sum()),
                fp_without_known_source_support=int((fp & (evidence[..., 2] <= 1e-8)).sum()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(f'configs/consolidated/{args.scene}.scene.json').read_text())
    inputs = json.loads(Path(f'output/conditioned_identity/full_source/{args.scene}/identity_evidence/inputs.json').read_text())
    native = load(inputs['response'])
    baseline = Path('output/consolidated_response_control') / args.scene / 'fusion/native_protected'
    field = load(baseline / 'typed_readout.pt')
    binding = native['geometry']
    geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    cameras = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    observed = native['observed'].cuda()
    prior, response = native['semantic_probability'].cuda(), field['scores'].cuda()
    raw_mass, positive_mass, known_mass, majority_witnesses = [torch.zeros_like(prior) for _ in range(4)]
    height, width = 192, 256
    for index, frame in enumerate(scene['source_frames']):
        records = load(Path(inputs['detections']) / f'identity_{frame}.pt')['instances']
        cam = cameras.cameras[frame]
        k = cam.intrinsic.cuda().clone()
        k[0] *= width / cam.width
        k[1] *= height / cam.height
        renderer = SimpleNamespace(scaled_intrinsics=lambda w, h: k,
                                   near_plane=.001 * cameras.scale, far_plane=100 * cameras.scale)
        hit = rasterize_single_view_contributions(geo, renderer, cam.world_to_camera.cuda(),
                                                  height=height, width=width, include_depths=False)
        hit['base_weights'] = hit['weights']
        op, _, valid = visible_operator(hit, observed, height * width, False)
        target = torch.stack([F.interpolate(r['masks'].float().amax(0)[None, None],
                              size=(height, width), mode='bilinear', align_corners=False)[0, 0]
                              if len(r['scores']) else torch.zeros(height, width) for r in records], -1).cuda()
        binary = (target.permute(2, 0, 1)[:, None] > .5).float()
        raw = binary.flatten(1).T.bool() & valid
        pos = (-F.max_pool2d(-binary, 3, 1, 1) > .5).flatten(1).T & valid
        neg = (F.max_pool2d(binary, 3, 1, 1) < .5).flatten(1).T & valid
        for qi, record in enumerate(records):
            unknown = record.get('unknown_masks')
            if unknown is not None and len(unknown):
                mask = F.interpolate(unknown.float().amax(0)[None, None].cuda(), size=(height, width),
                                     mode='bilinear', align_corners=False)[0, 0].flatten() > .5
                neg[:, qi] &= ~mask
        phase = index % 4
        pixels = torch.arange(height * width, device='cuda').reshape(height, width)[phase // 2::2, phase % 2::2].flatten()
        op = sample_operator_rows(op, pixels)
        raw, pos, neg = raw[pixels], pos[pixels], neg[pixels]
        neg &= torch.sparse.mm(op, prior) <= .5
        confidence = torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in records], device='cuda')
        view_positive = torch.sparse.mm(op.T, raw.float())
        view_mass = torch.sparse.sum(op, dim=0).to_dense()[:, None]
        majority_witnesses += ((view_positive / view_mass.clamp_min(1e-8) > .5) & (confidence[None] > 0)).float()
        raw_mass += view_positive * confidence
        positive_mass += torch.sparse.mm(op.T, pos.float() * confidence)
        known_mass += torch.sparse.mm(op.T, (pos | neg).float() * confidence)
    source_values = torch.stack([raw_mass > 1e-8, positive_mass > 1e-8, known_mass > 1e-8,
                                 majority_witnesses > 0], -1).float()
    # No target labels or target photographs were used to form these statistics.
    targets = load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label') / args.scene,
                               frames=SCENE_GT_FRAMES[args.scene])
    evaluation = json.loads((baseline / 'evaluation/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
    selected_records = {(r['frame'], r['query']): r for r in evaluation['selected']['objects']}
    rows = []
    for frame, objects in targets.items():
        height, width = objects[0].mask.shape
        cam = cameras.cameras[int(frame.split('_')[-1])]
        k = cam.intrinsic.cuda().clone()
        k[0] *= width / cam.width
        k[1] *= height / cam.height
        for obj in objects:
            qi = native['texts'].index(obj.query)
            colors = torch.cat([response[:, qi:qi + 1], observed[:, None].float(), source_values[:, qi]], 1)
            colors *= observed[:, None]
            image, alpha, _ = rasterization(means=geo.xyz, quats=geo.quaternions, scales=geo.scales,
                                           opacities=geo.opacities.flatten(), colors=colors,
                                           viewmats=cam.world_to_camera.cuda()[None], Ks=k[None],
                                           width=width, height=height, near_plane=.001 * cameras.scale,
                                           far_plane=100 * cameras.scale, packed=False)
            mass = image[0, ..., 1]
            valid = (mass / alpha[0, ..., 0].clamp_min(1e-8) > .5) & (alpha[0, ..., 0] > 1e-8)
            values = image[0] / mass[..., None].clamp_min(1e-8)
            semantic = describe(obj.mask, valid & (values[..., 0] > .5), values[..., 2:], valid)
            record = selected_records[(frame, obj.query)]
            selected = describe(load_binary_mask(Path(record['gt_path']), threshold=10),
                                load_binary_mask(Path(record['pred_path']), threshold=10, grayscale=True),
                                values[..., 2:], valid)
            rows.append(dict(frame=frame, query=obj.query, semantic=semantic, selected=selected))
        print(args.scene, frame, flush=True)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['query']].append(r)
    queries = []
    for query, records in grouped.items():
        metrics = {}
        for head in ['semantic', 'selected']:
            totals = {k: sum(r[head][k] for r in records) for k in records[0][head] if k != 'fn_partition'}
            totals['fn_partition'] = {k: sum(r[head]['fn_partition'][k] for r in records) for k in records[0][head]['fn_partition']}
            totals['iou'] = totals['tp'] / max(1, totals['tp'] + totals['fn'] + totals['fp'])
            totals['precision'] = totals['tp'] / max(1, totals['predicted'])
            totals['recall'] = totals['tp'] / max(1, totals['positives'])
            metrics[head] = totals
        queries.append(dict(query=query, **metrics))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'diagnosis.json').write_text(json.dumps(dict(scene=args.scene, source_frames=len(scene['source_frames']),
        queries=queries, views=rows, positive_witness_is_not_identity_truth=True,
        diagnostic_not_causal_proof=True, selected_support_uses_full_scene_operator=True,
        per_query_iou_is_pixel_pooled_not_official_view_mean=True, target_rgb_used=False,
        target_labels_only_for_error_reporting=True, predictions_changed=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
