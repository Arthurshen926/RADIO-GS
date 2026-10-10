"""One-pass conditional exclusion over field-owned joint coverage factors.

This is a conservative discrete set decoder, not v10's per-query fitted logits.
Only selected members touching a currently violated known negative factor may
be removed. Already-covered positive factors remain covered. Unknown factors
never create a removal request. This operator cannot add missing members.
"""
import numpy as np
import torch
from scipy import sparse
from numba import njit

CUTOFF = 10.5/255
# Frozen acceptance tolerance of the compiled-opacity function, not tuned on GT.
ALPHA_MARGIN = 1e-4


@njit
def _initial_weights(pointer,index,alpha,selected):
    weights=np.zeros(len(index),np.float64)
    for r in range(len(pointer)-1):
        transmission=1.
        for k in range(pointer[r],pointer[r+1]):
            if not selected[index[k]]:continue
            next_t=transmission*(1-alpha[k])
            if next_t<1e-4:break
            weights[k]=alpha[k]*transmission;transmission=next_t
    return weights


@njit
def _prune(pointer,rows,values,order,selected,positive,negative,coverage,cutoff,safe):
    result=selected.copy();current=coverage.copy()
    required=np.minimum(current,safe)
    protected=positive & (current>cutoff)
    removed=0
    for g in order:
        if not result[g]:continue
        start,stop=pointer[g],pointer[g+1]
        request=False;allowed=True
        for k in range(start,stop):
            r=rows[k];a=values[k]
            if a>0 and negative[r] and current[r]>cutoff:request=True
            if protected[r] and current[r]-a<required[r]-1e-12:allowed=False;break
        if request and allowed:
            result[g]=False;removed+=1
            for k in range(start,stop):current[rows[k]]-=values[k]
    return result,current,removed


class JointExclusionDecoder:
    """Query-independent CPU graph derived from one field; no sidecar access."""
    def __init__(self,field):
        if field.coverage_factors is None or field.scope_constraints is None:
            raise ValueError('joint exclusion requires compiled field coverage')
        self.field=field;self.count=len(field.xyz)
        pointer=[np.array([0],np.int64)];index=[];alpha=[];offset=0;self.slices=[]
        for v in field.coverage_factors.views:
            ptr=v.pointer.cpu().numpy();pointer.append(ptr[1:]+offset);offset+=len(v.index)
            index.append(v.index.cpu().numpy());alpha.append(v.alpha.cpu().numpy().astype(np.float64).clip(max=.999))
            self.slices.append(len(v.atom))
        self.pointer=np.concatenate(pointer);self.index=np.concatenate(index);self.alpha=np.concatenate(alpha)
        self.optical=-np.log1p(-self.alpha)
        self.matrix=sparse.csr_matrix((self.optical,self.index,self.pointer),shape=(len(self.pointer)-1,self.count))
        self.columns=self.matrix.tocsc()

    @torch.no_grad()
    def labels(self,accepted):
        field=self.field
        if accepted.dtype!=torch.bool or accepted.shape!=field.scope_permissions.accepted_views.shape:
            raise ValueError('invalid accepted region domain')
        permitted=field.scope_permissions(accepted);positive=[];negative=[]
        for c,f in zip(field.scope_constraints.views,field.coverage_factors.views):
            a=accepted[c.regions]
            if bool(a.any()):
                p=c.positive[:,a].any(1);n=c.negative[:,a].all(1)
                e=permitted[c.candidates]
                if bool(e.any()):n=n|c.interior[:,e].any(1)
            else:p=torch.zeros(len(c.positive),dtype=torch.bool,device=c.positive.device);n=p.clone()
            if bool((p&n).any()):raise ValueError('contradictory joint labels')
            positive.append(p[f.atom].cpu().numpy());negative.append(n[f.atom].cpu().numpy())
        return np.concatenate(positive),np.concatenate(negative)

    def decode(self,initial,accepted,*,joint=True):
        if initial.shape!=(self.count,) or not initial.is_floating_point() or not torch.isfinite(initial).all() or bool(((initial<0)|(initial>1)).any()):
            raise ValueError('invalid initial member scores')
        score=initial.detach().cpu().numpy();selected=score>.5
        positive,negative=self.labels(accepted)
        order=np.argsort(score,kind='stable').astype(np.int64)
        if joint:
            matrix=self.matrix;columns=self.columns
            cutoff=-np.log1p(-CUTOFF);safe=-np.log1p(-(CUTOFF+ALPHA_MARGIN))
        else:
            weights=_initial_weights(self.pointer,self.index,self.alpha,selected)
            matrix=sparse.csr_matrix((weights,self.index,self.pointer),shape=self.matrix.shape)
            columns=matrix.tocsc();cutoff=CUTOFF;safe=CUTOFF+ALPHA_MARGIN
        coverage=np.asarray(matrix@selected.astype(np.float64)).ravel()
        result,current,removed=_prune(columns.indptr,columns.indices,columns.data,order,selected,positive,negative,coverage,cutoff,safe)
        assert np.all(current[positive&(coverage>cutoff)]>cutoff)
        assert not np.any(result&~selected)
        # Same actual binary coverage can be verified in the nonlinear graph.
        before=np.asarray(self.matrix@selected.astype(np.float64)).ravel()
        after=np.asarray(self.matrix@result.astype(np.float64)).ravel();tau=-np.log1p(-CUTOFF)
        losses=int(np.sum(positive&(before>tau)&(after<=tau)))
        if losses:raise ValueError('joint positive coverage lost')
        output=initial.clone();deleted=torch.from_numpy(selected&~result).to(initial.device);output[deleted]=0.
        return output,dict(joint_coverage=joint,passes=1,removed=removed,initial_selected=int(selected.sum()),
            sampled_positive=int(positive.sum()),sampled_negative=int(negative.sum()),
            positive_covered_before=int(np.sum(positive&(before>tau))),positive_coverage_losses=losses,
            negative_covered_before=int(np.sum(negative&(before>tau))),negative_covered_after=int(np.sum(negative&(after>tau))),
            query_parameter_fitting=False,training=False,source_access=False,addition_supported=False,
            protection_is_sampled_not_full_image=True)


@torch.no_grad()
def query_text_joint_exclusion(field,query,negatives,*,decoder=None,joint=True):
    from .scope_confidence import query_text_scope_confident
    result=query_text_scope_confident(field,query,negatives)
    decoder=decoder if decoder is not None else JointExclusionDecoder(field)
    if decoder.field is not field:raise ValueError('decoder bound to another field')
    member,details=decoder.decode(result.membership_score,result.accepted_regions,joint=joint)
    result.membership_score=member;result.selected=member>.5
    result.diagnostics=dict(result.diagnostics,joint_exclusion=details)
    return result
