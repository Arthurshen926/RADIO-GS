import unittest
import torch
from radio_gs.query_field.coverage_factors import CoverageView


def factor(pointer,index,alpha):
    return CoverageView(dict(pointer=torch.tensor(pointer),index=torch.tensor(index,dtype=torch.long),alpha=torch.tensor(alpha),atom=torch.zeros(len(pointer)-1,dtype=torch.long)),2,1)


class CoverageFactorTest(unittest.TestCase):
    def test_overlap_is_not_disjoint_coverage(self):
        overlap=factor([0,2],[0,1],[.8,.8]); distinct=factor([0,1,2],[0,1],[.8,.8])
        member=torch.ones(2)
        self.assertAlmostEqual(float(overlap(member).sum()),.96,places=6)
        self.assertAlmostEqual(float(distinct(member).sum()),1.6,places=6)
        # Same per-Gaussian marginal mass, different joint response/gradient.
        x=torch.full((2,),.5,requires_grad=True)
        a=torch.autograd.grad(overlap(x).sum(),x)[0]
        b=torch.autograd.grad(distinct(x).sum(),x)[0]
        self.assertTrue(torch.all(a<b))

    def test_occluded_member_survives_foreground_removal(self):
        f=factor([0,2],[0,1],[.999,.999])
        self.assertAlmostEqual(float(f(torch.ones(2))),.999,places=6)
        self.assertAlmostEqual(float(f(torch.tensor([0.,1.]))),.999,places=6)
        self.assertEqual(float(f(torch.zeros(2))),0.)

    def test_member_scaled_cutoff_and_empty_factor(self):
        f=factor([0,0,1],[0],[.01])
        self.assertTrue(torch.equal(f(torch.tensor([.1,1.])),torch.zeros(2)))
        self.assertAlmostEqual(float(f(torch.tensor([.5,1.]))[1]),.005,places=7)

    def test_binding_and_member_validation(self):
        f=factor([0,1],[0],[.5])
        with self.assertRaises(ValueError):f(torch.tensor([float('nan'),0.]))
        with self.assertRaises(ValueError):f(torch.tensor([True,False]))
        with self.assertRaises(ValueError):factor([0,1],[2],[.5])

if __name__=='__main__':unittest.main()


def test_field_roundtrip_keeps_parent_and_joint_function(tmp_path):
    from scipy import sparse
    from test_query_preserving_field import example
    from test_query_field_mixture_evidence import data
    from test_query_field_scope_exclusion import fixture
    from radio_gs.query_field.confidence_mixture_evidence import enable
    from radio_gs.query_field.scope_permissions import ScopePermissions,compile_view as permission_view
    from radio_gs.query_field.scope_constraints import ScopeConstraints,compile_view
    from radio_gs.query_field.coverage_factors import CoverageFactors
    from radio_gs.query_field.observations import partial_mask_target,erode
    from radio_gs.query_field.storage import save_field,load_field
    field=example();n=len(field.xyz)
    enable(field,data([[.8,.2]]*n,[[.2,.8]]*n,[0,0]))
    masks,scales=fixture();ids=torch.tensor([0,1]);candidates=torch.arange(len(masks))
    field.scope_permissions=ScopePermissions(permission_view(masks,scales,ids,0))
    p,neg=zip(*(partial_mask_target(masks,i,scales) for i in ids))
    state=compile_view(sparse.csr_matrix(torch.ones(n,masks[0].numel()).numpy()/masks[0].numel()),torch.stack(p).flatten(1),torch.stack(neg).flatten(1),erode(masks).flatten(1),ids,candidates)
    field.scope_constraints=ScopeConstraints({'views.0.'+k:v for k,v in state.items()},n,field.scope_permissions)
    old={k:v.clone() for k,v in field.state_dict().items()}
    state=dict(pointer=torch.tensor([0,2]),index=torch.tensor([0,1]),alpha=torch.tensor([.9,.8]),atom=torch.tensor([0]))
    field.coverage_factors=CoverageFactors({'views.0.'+k:v for k,v in state.items()},n,field.scope_constraints)
    save_field(field,tmp_path/'field',{});restored,_=load_field(tmp_path/'field')
    for k,v in old.items():assert torch.equal(v,restored.state_dict()[k])
    member=torch.full((n,),.5,requires_grad=True)
    a=field.coverage_factors.views[0](member);b=restored.coverage_factors.views[0](member)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a.sum(),member)[0],torch.autograd.grad(b.sum(),member)[0])
