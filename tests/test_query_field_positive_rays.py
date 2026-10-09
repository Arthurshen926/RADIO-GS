import torch
from radio_gs.query_field.contracts import Support,QueryPacket
from radio_gs.query_field.positive_rays import mixture_logits,query
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.storage import save_field
from test_query_preserving_field import example


def test_full_ray_mixture_equals_independent_member_energies_and_duplicate_measure():
    f=example();p=Support(torch.tensor([0,1,2]),torch.tensor([.1,.3,.6]),torch.tensor([0,0,1]))
    expected=torch.logsumexp(torch.stack([f.member_logits(Support(torch.tensor([0,1]),torch.tensor([.1,.3])),1.),f.member_logits(Support(torch.tensor([2]),torch.tensor([.6])),1.)],1)+torch.tensor([.4,.6]).log(),1)
    actual=mixture_logits(f,p,1.);torch.testing.assert_close(actual,expected)
    duplicate=Support(torch.tensor([0,1,2,0,1,2]),torch.tensor([.05,.15,.3,.05,.15,.3]),torch.tensor([0,0,1,2,2,3]))
    torch.testing.assert_close(mixture_logits(f,duplicate,1.),actual)
    torch.testing.assert_close(mixture_logits(f,p,1.,chunk_size=1),actual)


def test_positive_ray_runtime_cold_and_single_support_identity(tmp_path):
    f=example();packet=QueryPacket('point_2d',positive=Support(torch.tensor([0,1]),torch.tensor([.5,.5]),torch.tensor([0,1])),negative=Support(torch.tensor([2]),torch.ones(1)),scale=1.)
    actual=query(f,packet);save_field(f,tmp_path/'field',{})
    torch.testing.assert_close(actual.membership_scores,QueryFieldRuntime.load(tmp_path/'field').positive_rays(packet).membership_scores)
    one=QueryPacket('point_2d',positive=Support(torch.tensor([0]),torch.ones(1)))
    assert torch.equal(query(f,one).membership_scores,f.query(one).membership_scores)
    assert actual.diagnostics['positive_rays']==2 and not actual.diagnostics['ray_subsampling']
