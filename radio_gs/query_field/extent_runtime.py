"""Public text-to-actual-set execution from one field and current legal vectors.

No research artifacts, stored query scopes or source readers are dependencies.
The atomic rule is the measured research baseline, not a global default change.
Other-view completion is an explicit experimental option. Other modalities must
provide their own validated typed scope adapters; they are not coerced to text.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .atomic_scope import assemble_atomic_scope
from .scope_topology import assemble_scope
from .scope_completion import complete_other_view_positive
from .native_text_device import NativeTextDevice
from .extent_decision_linked import LinkedExtentDecision, decisions, CUTOFF


def warm_extent_kernel(e):
    # Explicit cutoff must match solve's Numba signature. Warming the omitted
    # default overload would cause first-query compilation/cache file access.
    decisions(np.array([0,1],dtype=e.pointer.dtype),np.array([0],dtype=e.index.dtype),
        np.array([.8],dtype=e.alpha.dtype),np.array([0,1],dtype=e.colptr.dtype),
        np.array([0],dtype=e.colrows.dtype),np.array([0],dtype=e.colposition.dtype),
        np.array([0],dtype=e.rowview.dtype),np.array([True]),np.array([1.]),np.array([True]),
        np.array([.5]),np.array([1.]),np.array([0.]),4,CUTOFF)


@dataclass(frozen=True)
class ExtentTextResult:
    native_probability: torch.Tensor
    membership_scores: torch.Tensor
    selected: torch.Tensor
    diagnostics: dict


class ExtentTextRuntime:
    def __init__(self, field, *, scope_device='cuda'):
        required=('scope_topology','extent_preparation','extent_factors','semantic_measure')
        if any(getattr(field,k,None) is None for k in required):
            raise ValueError('text extent needs topology, observation lift, set optics and native measure in one field')
        if field.xyz.device.type!='cpu':
            raise ValueError('load the field on CPU; scope_device controls the transient compute mirror')
        self.field=field;self.device=torch.device(scope_device)
        self.native=NativeTextDevice(field,self.device)
        self.topology=field.scope_topology.to(self.device)
        self.preparation=field.extent_preparation.to(self.device)
        self.engine=LinkedExtentDecision(field.extent_factors,len(field.xyz))
        # Query-independent one-ray warmup loads/compiles the numerical kernel.
        # No actual query, source input or stored answer is used to initialize it.
        warm_extent_kernel(self.engine)

    @torch.no_grad()
    def query(self,native_query,generic_negatives,*,scope_rule='atomic'):
        if scope_rule not in {'atomic','other_view_positive'}:
            raise ValueError('unknown text extent scope rule')
        dim=self.field.semantic.native_dim
        if (native_query.shape not in {(dim,),(1,dim)} or generic_negatives.ndim!=2
                or generic_negatives.shape[1]!=dim or not len(generic_negatives)):
            raise ValueError('current native text vectors have incompatible dimensions')
        for v in (native_query,generic_negatives):
            if not v.is_floating_point() or not torch.isfinite(v).all() or bool((v.norm(dim=-1)<=0).any()):
                raise ValueError('current text vectors must be finite and nonzero')
        prior,_=self.native(native_query,generic_negatives);prior=prior.to(self.device)
        native=[];atomic=[]
        for view,pv in zip(self.topology.views,self.preparation.views):
            accepted,c,p=view.native_admission(native_query,generic_negatives)
            atomic.append(assemble_atomic_scope(view,pv,prior,p,c)[0])
            if scope_rule=='other_view_positive':native.append(assemble_scope(view,pv,prior,accepted,c))
        completion=None
        if scope_rule=='other_view_positive':
            scopes,completion=complete_other_view_positive(self.preparation,atomic,native)
        else:scopes=atomic
        initial,cost,info=self.preparation(prior,scopes)
        cpu=[{k:v.cpu() for k,v in s.items()} for s in scopes]
        confidence=np.zeros(len(cpu))
        for vi in info['active_views']:confidence[vi]=float(cpu[vi]['confidence'])
        value=initial.cpu()
        if info['active_views']:
            selected,report=self.engine.solve((value>.5).numpy(),torch.cat([s['target'] for s in cpu]).numpy(),
                torch.cat([s['positive'] for s in cpu]).numpy(),torch.cat([s['negative'] for s in cpu]).numpy(),
                confidence,cost.cpu().numpy(),max_passes=4)
            value=torch.from_numpy(selected).float();info.update(report)
        available=self.field.observed|self.field.semantic_measure.supported
        if bool((value>.5)[~available].any()):raise RuntimeError('selected an unavailable field member')
        info.update(scope_rule=scope_rule,completion=completion,source_access=False,
            query_parameter_fitting=False,shared_actual_set=True,default_promoted=False,
            typed_scope='text hypothesis union; other modality adapters not implied')
        return ExtentTextResult(prior.cpu(),value,value>.5,info)
