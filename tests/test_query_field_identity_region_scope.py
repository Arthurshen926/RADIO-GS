import torch
from radio_gs.query_field.identity_region_scope import IdentityRegionScopeDecoder
from test_query_field_region_carrier import carrier


def test_instance_quality_cannot_override_native_negative_identity():
    c = carrier(); xyz = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0], [3., 0, 0]])
    old = c.decoder(xyz); new = IdentityRegionScopeDecoder(c, xyz)
    # The exact same high-quality candidate passes old instance admission.
    assert old.scope(torch.tensor([.9, .4]))[1]['applied']
    value, detail = new.scope(torch.tensor([.9, .4]))
    assert value is None and detail['seed'] == 0 and detail['native_rejected_pairs'] == 1
    # When every candidate has native support, preserve the original function.
    previous, _ = old.scope(torch.tensor([.9, .6])); current, detail = new.scope(torch.tensor([.9, .6]))
    assert detail['admitted'] == [1]
    for a, b in zip(previous, current): assert torch.equal(a, b)


def test_identity_boundary_uses_existing_strict_native_positive_rule():
    c = carrier(); d = IdentityRegionScopeDecoder(c, torch.arange(12).reshape(4, 3).float())
    assert not d.scope(torch.tensor([.9, .5]))[1]['applied']
    assert d.scope(torch.tensor([.9, .50001]))[1]['applied']
    assert d.scope(torch.tensor([.5, .5]))[0] is None


def test_public_identity_scope_keeps_seed_when_all_extensions_rejected(monkeypatch):
    from types import SimpleNamespace
    from test_query_field_functional_training import small_field
    from radio_gs.query_field.identity_region_scope import query_text_identity_scope
    import radio_gs.query_field.scope_confidence as module
    f = small_field(4); f.region_carrier = carrier()
    f.semantic_measure = SimpleNamespace(supported=torch.ones(4, dtype=torch.bool))
    prior = torch.tensor([.8, .2, .1, .9]); prob = torch.tensor([.9, .4])
    monkeypatch.setattr(module, 'native_text_evidence', lambda *a: (prior, prob))
    monkeypatch.setattr(module, 'native_scope_confidence', lambda *a: (prob > .5, torch.ones(2)))
    class Fixed:
        field = f
        def decode(self, *a, **k): return prior.clone(), {}
    result = query_text_identity_scope(f, None, None, exclusion_decoder=Fixed())
    assert result.selected.tolist() == [True, True, False, True]
    assert result.accepted_regions.tolist() == [True, False]
