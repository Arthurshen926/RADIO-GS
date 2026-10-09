"""Offline compilation of Gaussian-supported native measure functions."""
from pathlib import Path
import json

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .native_modes import NativeModes
from .storage import load_field,save_field,sha256


def compile_scene(job,parent,out):
    root=Path(job['legacy_map']).parent;spec=json.loads((root/'map.json').read_text())
    if not spec['query_independent'] or spec['labels_used'] or spec['query_manifest_read']:
        raise ValueError('source modes are not query independent')
    field,manifest=load_field(parent);before={k:v.detach().clone() for k,v in field.state_dict().items()}
    source=json.loads((Path(job['observations'])/'observations.json').read_text())
    geometry=torch.load(source['geometry']['path'],map_location='cpu',weights_only=True)
    ids=geometry['source_vertex_ids'].numpy();xyz=np.load(root/'mesh_xyz.npy')
    if not np.array_equal(field.xyz.numpy(),xyz[ids]):raise ValueError('mode carrier mapping differs')
    offsets=np.load(root/'region_offsets.npy');patch_offsets=np.load(root/'patch_offsets.npy')
    chosen=[i for i,f in enumerate(spec['source_frames']) if f in job['budgets']['full']]
    regions=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in chosen])
    patches=np.concatenate([np.arange(patch_offsets[i],patch_offsets[i+1]) for i in chosen])
    b=sparse.load_npz(root/'membership.npz')[ids][:,regions].tocsc()
    weights=sparse.load_npz(root/'region_patch_weights.npz')[regions][:,patches].tocsr()
    valid=(b.getnnz(0)>0)&(np.asarray(weights.sum(1)).ravel()>0)
    mode_ids=np.flatnonzero(valid);weights=weights[valid];weights.eliminate_zeros()
    # Remove unbound atoms and source raster/frame organization from the export.
    used=np.unique(weights.indices);weights=weights[:,used].tocsr()
    raw=np.asarray(np.load(root/'patch_features.npy',mmap_mode='r')[patches[used]],np.float32)
    atoms=F.normalize(torch.from_numpy(raw),dim=1)
    weights=(sparse.diags(1/np.asarray(weights.sum(1)).ravel())@weights).tocsr();weights.sort_indices()
    anchors=[];source_anchor_differences=0
    original=sparse.load_npz(root/'membership.npz').tocsc()
    for i in mode_ids:
        rows=b[:,i].indices;points=field.xyz.numpy()[rows];median=np.median(points,axis=0)
        anchor=rows[np.argmin(((points-median)**2).sum(1))];anchors.append(anchor)
        old_rows=original[:,regions[i]].indices;old_points=xyz[old_rows]
        old_anchor=old_rows[np.argmin(((old_points-np.median(old_points,axis=0))**2).sum(1))]
        source_anchor_differences+=int(ids[anchor]!=old_anchor)
    state=dict(anchors=torch.tensor(anchors,dtype=torch.long),
        semantic=torch.from_numpy(np.asarray(np.load(root/'descriptors.npy')[regions[valid]],np.float32)),
        atoms=atoms,pointer=torch.from_numpy(weights.indptr.astype(np.int64)),
        index=torch.from_numpy(weights.indices.astype(np.int64)),mass=torch.from_numpy(weights.data.astype(np.float32)),
        mean=torch.from_numpy(np.asarray(weights@atoms.numpy(),np.float32)),relation_columns=torch.from_numpy(mode_ids))
    modes=NativeModes(state,len(ids)).cuda()
    # Fixed source pseudoqueries, never benchmark crops, prompts or labels.
    from radio_gs.querying.local_appearance_match import match_regions
    replay=[];probes=[];masses=[]
    for i in np.linspace(0,len(raw)-1,8).round().astype(int):
        q=raw[np.unique(np.r_[i,min(i+1,len(raw)-1),min(i+7,len(raw)-1)])]
        mass=np.arange(1,len(q)+1,dtype=np.float32);probes.append(q);masses.append(mass)
    reference_weights=sparse.load_npz(root/'region_patch_weights.npz')[regions[valid]][:,patches].tocsr()
    groups=np.repeat(np.arange(len(chosen)),[offsets[i+1]-offsets[i] for i in chosen])
    reference_offsets=np.r_[0,np.cumsum(np.bincount(groups[valid],minlength=len(chosen)))]
    reference_patches=np.r_[0,np.cumsum([patch_offsets[i+1]-patch_offsets[i] for i in chosen])]
    reference_raw=np.asarray(np.load(root/'patch_features.npy',mmap_mode='r')[patches],np.float32)
    expected=match_regions(reference_raw,reference_weights,reference_patches,reference_offsets,probes,masses)['chamfer']
    for i,(q,m) in enumerate(zip(probes,masses)):
        actual=modes.local_scores(torch.from_numpy(q).cuda(),torch.from_numpy(m).cuda()).cpu().numpy()
        error=float(np.abs(actual-expected[:,i]).max());replay.append(dict(query=i,modes=len(mode_ids),maximum_error=error))
        if error>2e-6:raise ValueError('native measure function replay failed')
    modes=modes.cpu();field.native_modes=modes
    reports=[]
    for rule in ['mean','measure']:
        field.native_mode_rule=rule
        if any(not torch.equal(field.state_dict()[k],v) for k,v in before.items()):raise ValueError('native compiler changed parent')
        saved=save_field(field,Path(out)/rule/'field',dict(parent_manifest=manifest['files'],
            source_views=job['budgets']['full'],extra_instance_supervision=True,
            native_mode_function=rule,uncompressed_measure_reference=True,source_layout_retained=False,
            supported_modes=len(mode_ids),supported_atoms=len(raw),training_steps=0,queries_used=False,labels_used=False))
        reports.append(dict(rule=rule,tensor_bytes=saved['tensor_bytes'],field=saved['files']))
    result=dict(scene=job['scene'],modes=len(mode_ids),atoms=len(raw),removed_unbound_modes=int((~valid).sum()),
        removed_unbound_atoms=len(patches)-len(raw),source_anchor_differences=source_anchor_differences,
        source_replay=replay,parent_unchanged=True,records=reports,source_frames=job['budgets']['full'],
        source_receipts=[dict(path=str(root/p),sha256=sha256(root/p)) for p in
            ['map.json','membership.npz','region_patch_weights.npz','patch_features.npy','descriptors.npy']])
    (Path(out)/'construction.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
