"""Compile native identity of the elements actually carried by each region.

The emitting view is excluded in full. Original descriptors, transfer, masks and
geometry are retained. Other-view coverage controls corroboration, not removal
of surface members. No query or benchmark annotations are inputs.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.querying.local_observation_membership import torch_sparse


@torch.inference_mode()
def compile_identity(transfer, membership, descriptors, offsets, device='cuda'):
    start=time.monotonic();transfer=transfer.astype(np.float32).tocsr()
    descriptor=F.normalize(torch.as_tensor(descriptors,dtype=torch.float32,device=device),dim=-1)
    total=torch.sparse.mm(torch_sparse(transfer,device),descriptor)
    mass=np.asarray(transfer.sum(1)).reshape(-1);columns=transfer.tocsc();members=membership.tocsc()
    prototypes=[];fractions=[]
    for view,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        lo,hi=int(lo),int(hi);matrix=columns[:,lo:hi].tocsr();active=np.flatnonzero(matrix.getnnz(axis=1))
        if not len(active):
            prototypes.append(np.zeros((hi-lo,descriptor.shape[1]),np.float16))
            fractions.append(np.zeros(hi-lo,np.float32));continue
        local=matrix[active];covered=members[active,lo:hi].tocsr()
        own=torch.sparse.mm(torch_sparse(local,device),descriptor[lo:hi])
        external=F.normalize(total[torch.from_numpy(active).to(device)]-own,dim=-1)
        available=(mass[active]-np.asarray(local.sum(1)).reshape(-1))>1e-6
        external*=torch.from_numpy(available).to(device)[:,None]
        pooled=torch.sparse.mm(torch_sparse(covered.T,device),external)
        prototypes.append(F.normalize(pooled,dim=-1).half().cpu().numpy())
        area=np.asarray(covered.sum(0)).reshape(-1)
        seen=np.asarray(covered.T@available.astype(np.float32)).reshape(-1)
        fractions.append(np.divide(seen,area,out=np.zeros_like(seen),where=area>0))
        if view%25==0:print('member identity',view+1,'/',len(offsets)-1,flush=True)
        del own,external,pooled
    return np.concatenate(prototypes),np.concatenate(fractions),dict(
        rule='native identity pooled from actual members after excluding the entire emitting view',
        seconds=time.monotonic()-start,regions=transfer.shape[1],elements=transfer.shape[0],
        original_geometry_membership_transfer_retained=True,original_native_descriptors_retained=True,
        query_read=False,labels_read=False,source_rgb_read=False,new_teacher_calls=0,
        missing_other_view_policy='original native identity retained',identity_is_not_verified_truth=True)


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def registered(scene, output):
    source=scene.parent;spec=json.loads(scene.read_text());output.mkdir(parents=True,exist_ok=True)
    descriptors,fraction,report=compile_identity(sparse.load_npz(source/'transfer.npz'),
        sparse.load_npz(source/'membership.npz'),np.load(source/'descriptors.npy').astype(np.float32),np.load(source/'region_offsets.npy'))
    for name in spec['map_files']:os.link(source/name,output/name)
    np.save(output/'member_identity.npy',descriptors);np.save(output/'member_identity_coverage.npy',fraction)
    spec.update(parent_map=str(scene),member_identity='member_identity.npy',member_identity_coverage='member_identity_coverage.npy',
        member_identity_build=report,map_files=spec['map_files']+['member_identity.npy','member_identity_coverage.npy'])
    spec['map_bytes']=sum((output/f).stat().st_size for f in spec['map_files'])
    write(output/'map.json',spec);write(output/'build.json',report)


def category(scene, output, offsets):
    spec=json.loads(scene.read_text());output.mkdir(parents=True,exist_ok=True)
    field=torch.load(spec['native_observation_field'],map_location='cpu',weights_only=False)
    transfer=sparse.csr_matrix((field['weights'].float().numpy(),field['columns'].numpy(),field['crow'].numpy()),shape=field['shape'])
    descriptors,fraction,report=compile_identity(transfer,transfer,field['descriptors'].float().numpy(),offsets)
    field.update(member_identity_descriptors=torch.from_numpy(descriptors),member_identity_coverage=torch.from_numpy(fraction),
        member_identity_build=report)
    torch.save(field,output/'observation_field.pt');spec['native_observation_field']=str(output/'observation_field.pt')
    if spec.get('local_auxiliary_scene'):
        category(Path(spec['local_auxiliary_scene']),output/'auxiliary',offsets)
        spec['local_auxiliary_scene']=str(output/'auxiliary/scene.json')
    write(output/'scene.json',spec);write(output/'build.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--offsets',type=Path);a=p.parse_args()
    torch.set_num_threads(4);spec=json.loads(a.scene.read_text())
    if spec.get('schema')=='registered_native_observation_map_v1':registered(a.scene,a.output)
    else:category(a.scene,a.output,np.load(a.offsets))


if __name__=='__main__':main()
