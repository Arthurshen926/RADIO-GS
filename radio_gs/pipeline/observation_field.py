"""Compile query-independent region-to-element observation weights once.

Preserves native region atoms instead of averaging feature directions. Runtime
only needs the dictionary and sparse transfer, not source masks or cameras.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--retain-visibility', action='store_true',
                   help='Retain sparse mask-independent compositing mass; also renders empty-region views')
    a = p.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    scene = json.loads(Path(a.scene).read_text())
    geometry = FrozenGaussianGeometry.load(scene['geometry']['path'], expected_sha256=scene['geometry']['sha256'], device='cuda')
    cameras = {c['frame_index']: c for c in json.loads(Path(scene['source_cameras']).read_text())['records']}
    count_elements = len(geometry.xyz)
    mass = torch.zeros(count_elements, 1, device='cuda')
    elements, regions, weights, descriptors = [], [], [], []
    offset = 0
    region_offsets = [0]
    from radio_gs import evidence_core
    from radio_gs.evidence_core.provenance import check_region_row, file_sha256
    provenance, visibility_rows, visibility_columns, visibility_weights, raster_hw = [], [], [], [], []
    if a.retain_visibility and not evidence_core.enabled():
        raise ValueError('--retain-visibility requires the explicit shared_v1 profile')
    for view_index, frame in enumerate(scene['source_frames']):
        region_path = Path(scene['native_region_directories'][0])/f'frame_{frame:05d}.pt'
        row = torch.load(region_path, map_location='cpu', weights_only=False)
        proposal_indices = check_region_row(row, frame)
        provenance.append(dict(frame=frame, proposal_indices=proposal_indices,
                               cache=dict(path=str(region_path), sha256=file_sha256(region_path))))
        sam = torch.load(row['sam_path'], map_location='cpu', weights_only=False)
        masks = unpack_masks(sam['packed_masks'], sam['mask_shape'][1])[row['native_proposal_indices'].numpy()].copy()
        if not len(masks) and not a.retain_visibility:
            region_offsets.append(offset)
            continue
        image_h, image_w = row['native_mask_hw']
        h, w = round(image_h*256/image_w), 256
        m = (F.interpolate(torch.as_tensor(masks).float()[:,None], size=(h,w), mode='nearest-exact')[:,0].flatten(1).cuda()
             if len(masks) else torch.empty(0, h*w, device='cuda'))
        c = cameras[frame]
        k = torch.as_tensor(c['intrinsic'], device='cuda', dtype=torch.float32).clone()
        k[0] *= w/c['width']; k[1] *= h/c['height']
        renderer = SimpleNamespace(scaled_intrinsics=lambda width,height:k, near_plane=.001, far_plane=100.)
        hit = rasterize_single_view_contributions(geometry, renderer, torch.as_tensor(c['world_to_camera'], device='cuda', dtype=torch.float32), height=h, width=w, include_depths=False)
        if evidence_core.enabled():
            from radio_gs.evidence_core.gaussian_observations import compile_gaussian_view
            observation = compile_gaussian_view(hit, m, count_elements, retain_visibility=a.retain_visibility)
            transfer, view_mass = observation.region_contributions, observation.supported_mass
            if a.retain_visibility:
                visible = torch.where(observation.geometric_mass > 0)[0]
                visibility_rows.append(visible.cpu())
                visibility_columns.append(torch.full((len(visible),), view_index, dtype=torch.long))
                visibility_weights.append(observation.geometric_mass[visible].cpu())
                raster_hw.append([h, w])
        else:
            count = m.sum(0)
            pixel, element = hit['pixel_ids'].long(), hit['gaussian_ids'].long()
            weight = hit['weights'].float()*(count[pixel]>0)
            adjoint = torch.sparse_coo_tensor(torch.stack([element,pixel]), weight, (count_elements,h*w)).coalesce()
            transfer = torch.sparse.mm(adjoint, (m/count.clamp_min(1)).T)
            view_mass = torch.sparse.mm(adjoint, torch.ones(h*w,1,device='cuda'))
        e, r = torch.where(transfer>0)
        elements.append(e.cpu()); regions.append(r.cpu()+offset); weights.append(transfer[e,r].cpu())
        mass += view_mass
        descriptors.append(row['descriptors'].half())
        offset += len(masks)
        region_offsets.append(offset)
        print(scene['scene'], frame, 'compiled', flush=True)
    e = torch.cat(elements); r = torch.cat(regions)
    v = torch.cat(weights)/mass[:,0].cpu()[e]
    transfer = torch.sparse_coo_tensor(torch.stack([e,r]),v,(count_elements,offset)).coalesce().to_sparse_csr()
    output = Path(a.output); output.mkdir(parents=True,exist_ok=True)
    payload = dict(descriptors=torch.cat(descriptors), crow=transfer.crow_indices().int(),
        columns=transfer.col_indices().int(), weights=transfer.values(), shape=[count_elements,offset],
        supported=mass[:,0].cpu()>0, geometry=scene['geometry'], native_space='siglip',
        query_independent=True, teacher_input_unchanged=True,
        region_offsets=torch.tensor(region_offsets), source_frames=scene['source_frames'],
        regions_by_view=provenance)
    if a.retain_visibility:
        visibility = torch.sparse_coo_tensor(torch.stack([torch.cat(visibility_rows), torch.cat(visibility_columns)]),
            torch.cat(visibility_weights), (count_elements, len(scene['source_frames']))).coalesce().to_sparse_csr()
        payload['geometric_visibility'] = dict(crow=visibility.crow_indices().int(), columns=visibility.col_indices().int(),
            weights=visibility.values(), shape=list(visibility.shape), kind='unconditioned_compositing_mass',
            raster_hw=raster_hw, near_plane=.001, far_plane=100.,
            cameras=dict(path=scene['source_cameras'], sha256=file_sha256(scene['source_cameras'])))
    torch.save(payload, output/'observation_field.pt')
    scene['native_category_prior']['kind']='compiled_region_response_mixture'
    scene['native_observation_field']=str(output/'observation_field.pt')
    scene.pop('native_field', None)
    scene.pop('native_region_directories', None)
    scene.pop('source_cameras', None)
    scene['map_contract']['representation']='native_region_dictionary_with_sparse_element_weights'
    (output/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(dict(elements=count_elements,regions=offset,nonzeros=transfer._nnz(),
        bytes=(output/'observation_field.pt').stat().st_size,seconds=time.monotonic()-started,
        source_frames=len(scene['source_frames']),query_labels_read=False),indent=2)+'\n')

if __name__=='__main__':
    main()
