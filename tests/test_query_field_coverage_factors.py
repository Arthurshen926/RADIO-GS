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
        with self.assertRaises(ValueError):factor([0,1],[2],[.5])

if __name__=='__main__':unittest.main()
