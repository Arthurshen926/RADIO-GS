"""Resolve overlapping observation membership using other-view native evidence.

This query-independent operator changes element/region weights, not descriptors.
Each source view keeps exactly its original mass at each element. Missing other
views retain the original mixture. Competing whole/part regions can both survive;
this is soft local responsibility, not an object partition or ground truth.
"""
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F


def torch_sparse(matrix, device):
    value=matrix.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(value.indptr).to(device),
        torch.from_numpy(value.indices).to(device),torch.from_numpy(value.data).to(device),
        size=value.shape,device=device)


@torch.inference_mode()
def local_membership(transfer, descriptors, offsets, device='cuda', temperature=100.):
    """Return mass-preserving transfer and relative per-element memberships.

    For region r in view t and element v, compare its native descriptor to the
    original fused feature with *all* of t's contributions subtracted. Among
    regions in t that really cover v, rho=exp(T*(agreement-max agreement)).
    The best supported interpretation keeps rho=1; ties preserve alternatives.
    There is one pass, no graph iteration, labels, query vocabulary or fitting.
    """
    start=time.monotonic();transfer=transfer.astype(np.float32).tocsr()
    descriptor=F.normalize(torch.as_tensor(descriptors,dtype=torch.float32,device=device),dim=-1)
    total=torch.sparse.mm(torch_sparse(transfer,device),descriptor)
    total_mass=np.asarray(transfer.sum(1)).reshape(-1)
    columns=transfer.tocsc();weighted=[];relative=[];changed=0;unwitnessed=0;entries=0
    largest_mass_error=0.
    for view,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
        matrix=columns[:,int(lo):int(hi)].tocsr()
        active=np.flatnonzero(matrix.getnnz(axis=1))
        if not len(active):
            weighted.append(matrix);relative.append(matrix.copy());continue
        local=matrix[active];own_mass=np.asarray(local.sum(1)).reshape(-1)
        own=torch.sparse.mm(torch_sparse(local,device),descriptor[int(lo):int(hi)])
        external=total[torch.from_numpy(active).to(device)]-own
        available=(total_mass[active]-own_mass)>1e-6
        external=F.normalize(external,dim=-1)
        rows=np.repeat(np.arange(len(active)),np.diff(local.indptr))
        scores=np.empty(local.nnz,np.float32)
        for first in range(0,len(rows),32768):
            last=first+32768
            rr=torch.from_numpy(rows[first:last]).to(device)
            cc=torch.from_numpy(local.indices[first:last]+int(lo)).to(device)
            scores[first:last]=(external[rr]*descriptor[cc]).sum(1).cpu().numpy()
        maximum=np.full(len(active),-np.inf,np.float32)
        np.maximum.at(maximum,rows,scores)
        rho=np.exp(temperature*np.minimum(scores-maximum[rows],0)).astype(np.float32)
        rho[~available[rows]]=1.
        adjusted=local.data*rho
        mass=np.bincount(rows,weights=adjusted,minlength=len(active)).astype(np.float32)
        adjusted*=own_mass[rows]/np.maximum(mass[rows],np.finfo(np.float32).tiny)
        out=matrix.copy();out.data=adjusted
        rel=matrix.copy();rel.data=rho
        weighted.append(out);relative.append(rel)
        changed+=int((rho<.5).sum());unwitnessed+=int((~available[rows]).sum());entries+=len(rho)
        largest_mass_error=max(largest_mass_error,float(np.max(np.abs(np.asarray(out.sum(1)).reshape(-1)[active]-own_mass))))
        del own,external
        if view%25==0:print('local membership',view+1,'/',len(offsets)-1,flush=True)
    result=sparse.hstack(weighted,format='csr');reliability=sparse.hstack(relative,format='csr')
    report=dict(rule='leave-whole-view-out native agreement; relative competing membership; preserve element/view mass',
        temperature=temperature,passes=1,regions=transfer.shape[1],elements=transfer.shape[0],
        entries=entries,membership_below_half=changed,no_other_view_entries=unwitnessed,
        largest_view_mass_error=largest_mass_error,native_coordinates_changed=False,
        source_rgb_read=False,labels_read=False,query_read=False,
        seconds=time.monotonic()-start,agreement_is_not_verified_identity=True)
    return result,reliability,report
