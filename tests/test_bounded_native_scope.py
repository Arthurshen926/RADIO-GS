import pytest
import torch
from radio_gs.querying.bounded_native_scope import refine_native_scope_logits


def test_disabled_is_exact_identity():
    native = torch.tensor([[5., 1., 0.]])
    assert refine_native_scope_logits(native, -native) is native


def test_adversarial_scope_cannot_flip_protected_native_margin():
    native = torch.tensor([[5., 1., 0.], [0., 4., 1.]])
    scope = torch.tensor([[-100., 100., 100.], [100., -100., 100.]])
    result = refine_native_scope_logits(native, scope, max_correction=1.)
    assert torch.equal(result.argmax(-1), native.argmax(-1))
    assert (result-native).abs().max() <= 1.


@pytest.mark.parametrize('bound', [-1., float('nan'), float('inf')])
def test_invalid_bound_rejected(bound):
    native = torch.ones(2, 3)
    with pytest.raises(ValueError):
        refine_native_scope_logits(native, native, max_correction=bound)


def test_incompatible_or_nonfinite_evidence_rejected():
    native = torch.ones(2, 3)
    for scope in (torch.ones(2, 1), native.double(), native * float('nan')):
        with pytest.raises(ValueError):
            refine_native_scope_logits(native, scope, max_correction=1.)
