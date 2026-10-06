"""All-source RGB matching and a matched-pixel scope intervention.

Registered poses choose neighboring images; LoFTR uses only their RGB. The
Gaussian carrier is used afterwards for a separate projection diagnostic,
never to decide image matches or erase unmatched source supervision.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from kornia.feature import LoFTR

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.querying.rgb_scope_correspondence import camera_neighbor_pairs, epipolar_residual, raster_indices, matched_negative_conflicts
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(args.inputs).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    frames = scene['source_frames']
    gp = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    response = load(inputs['response'])
    geo = FrozenGaussianGeometry.load(response['geometry']['path'], expected_sha256=response['geometry']['sha256'], device='cuda')
    if 'source_rgb' in scene:
        rgb = {int(frame): path for frame, path in scene['source_rgb'].items()}
    else:
        rgb_rows = json.loads(Path(scene['source_rgb_authority']).read_text())['images']
        rgb = {int(row['image_id'].split('_')[-1]): row['path'] for row in rgb_rows}
    poses = torch.stack([gp.cameras[frame].world_to_camera for frame in frames])
    pairs = camera_neighbor_pairs(poses)
    image_hw, raster_hw = (480, 640), (192, 256)
    h, w = raster_hw
    radius = scene.get('boundary_radius', 1)
    views, intrinsics, centroids = [], [], []
    images = []
    for frame in frames:
        cam = gp.cameras[frame]
        k = cam.intrinsic.clone()
        k[0] *= image_hw[1] / cam.width
        k[1] *= image_hw[0] / cam.height
        intrinsics.append(k)
        gray = Image.open(rgb[frame]).convert('L').resize((image_hw[1], image_hw[0]), Image.Resampling.BILINEAR)
        images.append(torch.from_numpy(np.asarray(gray).copy()).float()[None, None] / 255)
        data = load(Path(inputs['detections']) / f'identity_{frame}.pt')
        target = torch.stack([F.interpolate(row['masks'].float().amax(0)[None, None], size=raster_hw, mode='bilinear', align_corners=False)[0, 0] if len(row['scores']) else torch.zeros(raster_hw) for row in data['instances']])
        binary = (target[:, None] > .5).float()
        positive = (-F.max_pool2d(-binary, 2 * radius + 1, 1, radius) > .5).flatten(1).T
        negative = (F.max_pool2d(binary, 2 * radius + 1, 1, radius) < .5).flatten(1).T
        for qi, row in enumerate(data['instances']):
            unknown = row.get('unknown_masks')
            if unknown is not None and len(unknown):
                negative[:, qi] &= ~(F.interpolate(unknown.float().amax(0)[None, None], size=raster_hw, mode='bilinear', align_corners=False)[0, 0].flatten() > .5)
        active = torch.tensor([bool(len(row['scores'])) for row in data['instances']])
        views.append(dict(positive=positive, negative=negative, active=active, unknown=torch.zeros_like(negative), matched=torch.zeros(h * w, dtype=torch.bool)))
        rk = cam.intrinsic.cuda().clone()
        rk[0] *= w / cam.width
        rk[1] *= h / cam.height
        renderer = SimpleNamespace(scaled_intrinsics=lambda width, height: rk, near_plane=.001 * gp.scale, far_plane=100 * gp.scale)
        hit = rasterize_single_view_contributions(geo, renderer, cam.world_to_camera.cuda(), height=h, width=w, include_depths=False)
        pid, gid, weight = hit['pixel_ids'].long().cuda(), hit['gaussian_ids'].long().cuda(), hit['weights'].float().cuda()
        xyz = torch.zeros(h * w, 3, device='cuda')
        mass = torch.zeros(h * w, device='cuda')
        xyz.index_add_(0, pid, weight[:, None] * geo.xyz[gid])
        mass.index_add_(0, pid, weight)
        centroids.append((xyz / mass[:, None].clamp_min(1e-8)).cpu())
    checkpoint = '/root/.cache/torch/hub/checkpoints/loftr_outdoor.ckpt'
    matcher = LoFTR(pretrained=None).cuda().eval()
    matcher.load_state_dict(load(checkpoint)['state_dict'])
    fine_radius = matcher.config['fine_window_size'] / 2
    diagnostics, all_matches, all_reprojection = [], [], []
    for pi, (a, b) in enumerate(pairs):
        result = matcher(dict(image0=images[a].cuda(), image1=images[b].cuda()))
        x0, x1 = result['keypoints0'].cpu(), result['keypoints1'].cpu()
        error = epipolar_residual(x0, x1, intrinsics[a], intrinsics[b], poses[a], poses[b])
        keep = error <= fine_radius
        xy0, xy1 = x0[keep], x1[keep]
        ids0, ids1 = raster_indices(xy0, image_hw, raster_hw), raster_indices(xy1, image_hw, raster_hw)
        v0, v1 = views[a], views[b]
        unknown0, unknown1 = matched_negative_conflicts(v0['positive'], v0['negative'], v0['active'], ids0, v1['positive'], v1['negative'], v1['active'], ids1)
        v0['unknown'] |= unknown0
        v1['unknown'] |= unknown1
        v0['matched'][ids0] = True
        v1['matched'][ids1] = True
        residuals = []
        for src, dst, ids, matched in [(a, b, ids0, xy1), (b, a, ids1, xy0)]:
            xyz = centroids[src][ids]
            camera_xyz = xyz @ poses[dst][:3, :3].T + poses[dst][:3, 3]
            projected = camera_xyz @ intrinsics[dst].T
            uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
            residuals.append(((uv - matched) * torch.tensor([w / image_hw[1], h / image_hw[0]])).norm(dim=1))
        residual = torch.cat(residuals)
        all_reprojection.append(residual)
        row = dict(frames=[frames[a], frames[b]], raw_matches=len(x0), admitted_matches=int(keep.sum()), epipolar_median_pixels=float(error.median()) if len(error) else None, map_reprojection_median_raster_pixels=float(residual.median()) if len(residual) else None, scope_conflicts=[int(unknown0.sum()), int(unknown1.sum())])
        diagnostics.append(row)
        all_matches.append(dict(frames=row['frames'], xy0=xy0, xy1=xy1, confidence=result['confidence'].cpu()[keep], epipolar_residual=error[keep]))
        print(scene['scene'], pi + 1, len(pairs), row, flush=True)
    destination = output / 'evidence'
    destination.mkdir(exist_ok=True)
    totals = torch.zeros(len(response['texts']), dtype=torch.long)
    observations = []
    for frame, view in zip(frames, views):
        evidence = load(Path(inputs['detections']) / f'identity_{frame}.pt')
        for qi, row in enumerate(evidence['instances']):
            correction = view['unknown'][:, qi].reshape(h, w)
            if correction.any():
                old = row.get('unknown_masks')
                unknown = correction.float()
                if old is not None and len(old):
                    old = F.interpolate(old.float().amax(0)[None, None], size=raster_hw, mode='bilinear', align_corners=False)[0, 0]
                    unknown = torch.maximum(unknown, old)
                row['unknown_masks'] = unknown[None].half()
            totals[qi] += correction.sum()
        torch.save(evidence, destination / f'identity_{frame}.pt')
        observations.append(dict(frame=frame, matched_pixels=int(view['matched'].sum()), negative_conflict_pixels=int(view['unknown'].sum()), active_queries=int(view['active'].sum())))
    amended = dict(inputs, detections=str(destination), evidence_contract='native admitted source scopes; RGB-matched positive versus negative conflicts become unknown only at matched negative pixels')
    (destination / 'inputs.json').write_text(json.dumps(amended, indent=2) + '\n')
    torch.save(all_matches, output / 'matches.pt')
    residual = torch.cat(all_reprojection)
    summary = dict(scene=scene['scene'], source_frames=frames, source_images=len(frames), pairs=len(pairs), checkpoint=checkpoint, image_hw=image_hw, scope_raster_hw=raster_hw, epipolar_tolerance_from_fine_window=fine_radius, raw_matches=sum(x['raw_matches'] for x in diagnostics), admitted_matches=sum(x['admitted_matches'] for x in diagnostics), matched_pixels=sum(x['matched_pixels'] for x in observations), scope_negative_conflict_pixels=int(totals.sum()), conflicts_by_query=dict(zip(response['texts'], totals.tolist())), gaussian_reprojection_median_raster_pixels=float(residual.median()), gaussian_reprojection_p90_raster_pixels=float(residual.quantile(.9)), source_view_records=observations, pair_records=diagnostics, seconds=time.monotonic() - start, target_rgb_used=False, target_labels_used=False, positives_changed=False, unmatched_pixels_changed=False, correspondence_is_verified_truth=False, native_response_unchanged=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
