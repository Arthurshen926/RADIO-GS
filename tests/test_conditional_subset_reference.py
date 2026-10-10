import torch
from scripts.research.conditional_subset_reference import silhouette_loss,solve


def test_unknown_pixels_do_not_enter_silhouette_loss_or_gradient():
    p=torch.tensor([True,False,False]);n=torch.tensor([False,True,False])
    a=torch.tensor([.2,.02,.8],requires_grad=True);b=torch.tensor([.2,.02,.01])
    loss=silhouette_loss(a,p,n);torch.testing.assert_close(loss,silhouette_loss(b,p,n))
    loss.backward();assert a.grad[2]==0 and a.grad[0]<0 and a.grad[1]>0


def test_empty_source_reference_does_not_fit_or_change_initializer():
    initial=torch.tensor([.2,.8]);result,details=solve(None,initial,[],1.)
    assert torch.equal(result,initial) and details['steps']==0


def test_fixed_measure_solver_replays_original_when_initializers_match(monkeypatch):
    from types import SimpleNamespace
    import scripts.research.conditional_subset_reference as old
    import scripts.research.fixed_initializer_subset_reference as new
    alpha=lambda field,member,*args: member
    monkeypatch.setattr(old,'relaxed_subset_alpha',alpha);monkeypatch.setattr(new,'relaxed_subset_alpha',alpha)
    field=SimpleNamespace(observed=torch.ones(2,dtype=torch.bool))
    views=[dict(positive=torch.tensor([True,False]),negative=torch.tensor([False,True]),
        footprint=torch.tensor([1.,3.]),camera=None,height=1,width=2)]
    initial=torch.tensor([.3,.7])
    a,da=old.solve(field,initial,views,1.,steps=4)
    b,db=new.solve_fixed(field,initial,views,1.,normalizer=initial,steps=4)
    assert torch.equal(a,b) and da['final_objective']==db['final_objective']


def test_changing_initializer_does_not_renormalize_fixed_energy(monkeypatch):
    from types import SimpleNamespace
    import scripts.research.conditional_subset_reference as old
    import scripts.research.fixed_initializer_subset_reference as new
    alpha=lambda field,member,*args: member
    monkeypatch.setattr(old,'relaxed_subset_alpha',alpha);monkeypatch.setattr(new,'relaxed_subset_alpha',alpha)
    field=SimpleNamespace(observed=torch.ones(2,dtype=torch.bool));initial=torch.tensor([.2,.8]);reference=torch.tensor([.8,.2])
    views=[dict(positive=torch.tensor([True,False]),negative=torch.tensor([False,True]),
        footprint=torch.tensor([1.,3.]),camera=None,height=1,width=2)]
    _,a=old.solve(field,initial,views,1.,steps=1)
    _,b=new.solve_fixed(field,initial,views,1.,normalizer=reference,steps=1)
    assert abs(a['final_footprint_penalty']-.1)<1e-6
    assert abs(b['final_footprint_penalty']-.1*2.6/1.4)<1e-6
    assert a['final_source_loss']==b['final_source_loss']
