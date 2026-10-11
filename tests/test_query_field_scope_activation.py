import torch
from radio_gs.query_field.scope_activation import admitted_negative
from test_query_field_extent_preparation import fixture,scopes


def test_materializing_negative_permission_preserves_the_original_function():
    prep=fixture();prior=torch.tensor([.1,.3,.4,.7])
    for zero_confidence,no_positive in [(False,False),(True,False),(False,True),(True,True)]:
        original=scopes()
        if zero_confidence:original[0]['confidence'].zero_()
        if no_positive:original[1]['positive'].zero_()
        bounded=[dict(s,negative=admitted_negative(s)) for s in original]
        a,c,i=prep(prior,original);b,d,j=prep(prior,bounded)
        assert torch.equal(a,b) and torch.equal(c,d) and i==j


def test_new_positive_or_confidence_does_not_authorize_an_inactive_origins_negative():
    from scripts.research.run_query_field_signed_scope import hybrid_scope
    n=torch.tensor([False,True]);p=torch.tensor([True,False])
    positive=dict(target=p.float(),positive=p)
    for origin in [dict(positive=p,negative=n,confidence=torch.tensor(0.)),
                   dict(positive=torch.zeros_like(p),negative=n,confidence=torch.tensor(.9))]:
        raw,_=hybrid_scope(positive,origin['negative'],torch.tensor(.8))
        permitted,_=hybrid_scope(positive,admitted_negative(origin),torch.tensor(.8))
        assert raw['negative'].any() and not permitted['negative'].any()
        assert torch.equal(raw['positive'],permitted['positive'])
