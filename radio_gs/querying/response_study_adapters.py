"""Frozen-provider adapters for a declared offline response-refinement study.

Reads retained v10 frontend responses/evidence explicitly. This is a matched
post-refinement experiment, not a cold end-to-end system replay. No evaluator
labels, source RGB for UQIS, extra teachers or fitted thresholds are consumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from .response_permissions import admit_response_observation, exclusive_category_exclusion
from .typed_evidence import visible_operator, view_evidence

SYSTEM = Path('/mnt/pool/sqy/results/RADIO-GS/output/query_conditioned_membership_20261006/image_convergence/system')


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def readout_path(job):
    return SYSTEM / job['benchmark'] / job['name'] / 'query' / (
        'category/typed_readout.pt' if job['benchmark'] == 'scannet' else 'typed_readout.pt')


def source_observations(job, data, prior):
    family = job['benchmark']
    root = SYSTEM / family / job['name'] / 'query'
    scene = read(job['scene'])
    observed = data['semantic_observed'].cuda()
    if family == 'lerf':
        from scripts.research.fit_consolidated_response import iter_source_response_views
        inputs = read(root/'text/identity_evidence/inputs.json')
        source_scene = read(inputs['scene'])
        binding = data['semantic_geometry']
        geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
        gp = MogeGeometryPrior(source_scene['camera_report'], source_scene['camera_report_sha256'])
        for op, target, pos, neg, _native_positive, confidence in iter_source_response_views(
                inputs, source_scene, data, geo, gp, observed, prior):
            # Identity-rejected masks were already removed from neg by the
            # frozen adapter; no native-positive veto is applied in either arm.
            yield admit_response_observation(op, target, pos, neg, confidence[None])
    elif family == 'nvos':
        query = read(job['queries'])
        binding = data['semantic_geometry']
        geo = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
        gp = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
        h, w = scene['source_fit_hw']
        for frame in scene['source_frames']:
            record = load(root/'reference/0/evidence'/f'identity_{frame}.pt')['instances'][0]
            if not len(record['scores']):
                continue
            cam = gp.cameras[frame]
            k = cam.intrinsic.cuda().clone(); k[0] *= w/cam.width; k[1] *= h/cam.height
            renderer = SimpleNamespace(scaled_intrinsics=lambda width,height:k,
                near_plane=.001*gp.scale, far_plane=100*gp.scale)
            hit = rasterize_single_view_contributions(geo, renderer, cam.world_to_camera.cuda(),
                height=h, width=w, include_depths=False)
            hit['base_weights'] = hit['weights']
            op, _, visible = visible_operator(hit, observed, h*w, False)
            target = F.interpolate(record['masks'].float().amax(0)[None,None].cuda(),
                size=(h,w), mode='bilinear', align_corners=False).flatten()[:,None]
            e = view_evidence(op, op, visible, target, torch.ones(1,device='cuda',dtype=torch.bool),
                frame, h, w, torch.ones(1,device='cuda'), boundary_radius=scene['boundary_radius'])
            unknown = record.get('unknown_masks')
            if unknown is not None and len(unknown):
                mask = F.interpolate(unknown.float().amax(0)[None,None].cuda(),size=(h,w),
                    mode='bilinear',align_corners=False).flatten()[:,None] > .5
                e.negative &= ~mask
            # This is a supplied reference-mask estimate, not negative scribble
            # GT. Tracked partial regions in other frames cannot exclude.
            exclusion = e.negative if frame == query['reference_frame'] else torch.zeros_like(e.negative)
            yield admit_response_observation(op, target, e.positive, e.negative,
                e.confidence, exclusion=exclusion)
    elif family == 'scannet':
        spec = scene['source_evidence']; h, w = spec['raster_hw']
        records = [load(v['identity']) for v in spec['views']]
        ids = [records[0]['queries'].index(t) for t in data['texts']]
        square = torch.zeros(len(ids)); count = torch.zeros_like(square)
        for record in records:
            for j, qi in enumerate(ids):
                row = record['instances'][qi]
                square[j] += row['foreground_margin'].square().sum(); count[j] += len(row['scores'])
        scale = (square/count.clamp_min(1)).sqrt().clamp_min(1e-6)
        for view, record in zip(spec['views'], records):
            if [record['queries'].index(t) for t in data['texts']] != ids:
                raise ValueError('source query order changed')
            rows = [record['instances'][qi] for qi in ids]
            op, _, visible = visible_operator(load(view['contributions']), observed, h*w, False)
            target = torch.stack([r['masks'].float().amax(0) if len(r['scores']) else torch.zeros(h,w)
                for r in rows], -1).reshape(h*w,-1).cuda()
            confidence = torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max())
                if len(r['scores']) else 0. for j,r in enumerate(rows)], device='cuda')
            e = view_evidence(op, op, visible, target, confidence>0, view['frame'],h,w,
                confidence, boundary_radius=record.get('boundary_radius',1))
            # Complete 19-class menu, never a vocabulary chunk. A visible
            # foreground of a different exclusive class supplies counterevidence.
            exclusive = exclusive_category_exclusion(e.positive)
            rival_confidence = (e.positive*confidence[None]).amax(1,keepdim=True)
            weight = torch.where(e.positive, confidence[None], torch.maximum(confidence[None],rival_confidence))
            # No detection is not background. The complement control deliberately
            # lifts this permission where a query or rival has nonzero confidence.
            binary = (target.T.reshape(-1,1,h,w)>.5).float()
            radius = record.get('boundary_radius',1)
            neg = (F.max_pool2d(binary,2*radius+1,1,radius)<.5).flatten(1).T & visible
            for qi,row in enumerate(rows):
                unknown = row.get('unknown_masks')
                if unknown is not None and len(unknown):
                    neg[:,qi] &= ~(unknown.float().amax(0).flatten().cuda()>.5)
            yield admit_response_observation(op,target,e.positive,neg,weight,exclusion=exclusive)
    else:
        raise ValueError('source-assisted adapter family not declared')


def uqis_inputs(job):
    root = Path(job['scene']).parent
    report = read(SYSTEM/'uqis'/job['name']/'query/registered/runs'/f"{job['name']}.json")
    membership = sparse.load_npz(root/'membership.npz').tocsr()
    visibility = sparse.load_npz(root/'visibility.npz').tocsc()
    frames = np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(root/'region_offsets.npy')))
    return membership, visibility, frames, report['query_details']


def mean_view_membership(membership, vertices, regions):
    # scipy float32 sparse mean can yield 1.0000001 for eleven unit entries.
    # Sum in float64, divide once, then cast the bounded result for the refiner.
    total = np.asarray(membership[vertices][:,regions].sum(1,dtype=np.float64)).ravel()
    return (total/len(regions)).astype(np.float32)


def uqis_observations(membership, visibility, frames, detail):
    """Use frozen query identity, retaining nested-mask dependence within a view."""
    modality = detail['modality']
    if modality.startswith('point') and not detail['anchor_has_feature']:
        return
    if modality.startswith('point'):
        regions = membership[detail['anchor_vertex']].indices
    elif modality == 'image':
        anchor = detail['inferred_anchor_membership']['anchor_vertex']
        regions = membership[anchor].indices  # estimated identity, never a hard click
    else:
        regions = np.asarray(detail['instance_identity']['matched_regions'],dtype=np.int64)
    for frame in np.unique(frames[regions]):
        local = regions[frames[regions] == frame]
        vertices = visibility[:,frame].nonzero()[0]
        if not len(vertices):
            continue
        # One mean observation per view, not one independent vote per nested mask.
        target = mean_view_membership(membership,vertices,local)
        target = torch.from_numpy(target[:,None]).cuda()
        ids = torch.from_numpy(vertices.astype(np.int64)).cuda()
        operator = torch.sparse_coo_tensor(torch.stack([torch.arange(len(ids),device='cuda'),ids]),
            torch.ones(len(ids),device='cuda'),(len(ids),membership.shape[0])).coalesce()
        yield admit_response_observation(operator,target,target>.5,target<.5,
            torch.ones((),device='cuda'))
