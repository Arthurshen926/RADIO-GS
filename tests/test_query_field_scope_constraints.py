import itertools
import numpy as np
import torch
from scipy import sparse
from radio_gs.query_field.scope_constraints import ScopeConstraints,compile_view
from radio_gs.query_field.scope_permissions import ScopePermissions,compile_view as graph_view
from radio_gs.query_field.scope_exclusion import union_exclusion
from radio_gs.query_field.observations import erode,partial_mask_target
from test_query_field_scope_exclusion import fixture


def test_all_subsets_preserve_union_before_lifting_including_overlaps():
    masks,scales=fixture();ids=torch.arange(len(masks));g=ScopePermissions(graph_view(masks,scales,ids,0))
    visible=torch.ones_like(masks[0]);visible[:,0]=False
    p,n=zip(*(partial_mask_target(masks,i,scales) for i in range(len(masks))))
    p=torch.stack(p)&visible;n=torch.stack(n)&visible;e=erode(masks)&visible
    rng=np.random.RandomState(11);transfer=rng.rand(3,masks[0].numel()).astype(np.float32);transfer/=transfer.sum(1,keepdims=True)
    state=compile_view(sparse.csr_matrix(transfer),p.flatten(1),n.flatten(1),e.flatten(1),ids,ids)
    module=ScopeConstraints({'views.0.'+k:v for k,v in state.items()},3,g)
    for bits in itertools.product([False,True],repeat=len(ids)):
        a=torch.tensor(bits)
        old=n[a].all(0) if a.any() else torch.zeros_like(visible)
        neg,_=union_exclusion(masks,scales,a,visible,old)
        pos=p[a].any(0) if a.any() else torch.zeros_like(visible)
        expected=torch.from_numpy(transfer@torch.stack([pos,neg],-1).reshape(-1,2).float().numpy())
        actual=module(a,g);torch.testing.assert_close(torch.stack([actual[0][:,0],actual[1][:,0]],1),expected,atol=2e-6,rtol=0)
        baseline=module(a,g,expanded=False)
        assert torch.equal(actual[0],baseline[0]) and bool((actual[1]>=baseline[1]-1e-7).all())


def test_overlap_is_counted_once_not_sum_of_region_marginals():
    # Two permitted regions share the same atom; the lift must stay at one.
    state=dict(pointer=torch.tensor([0,1]),index=torch.tensor([0]),value=torch.tensor([1.]),
        positive=torch.tensor([[False]]),negative=torch.tensor([[False]]),interior=torch.tensor([[True,True]]),
        regions=torch.tensor([0]),candidates=torch.tensor([0,1]))
    from radio_gs.query_field.scope_constraints import ConstraintView
    result=ConstraintView(state,1)(torch.tensor([True]),torch.tensor([True,True]))
    assert result.tolist()==[[0.,1.]]
