from copy import deepcopy

import pytest
import torch

from radio_gs.querying.source_accepted_category import source_accepted_category_logits


def receipts(accepted=True):
    scope = dict(schema='bounded_fallback_scope.v1', source_only=True,
                 benchmark_data_used=False, field_sha256='a'*64,
                 source_acceptance=accepted, score_order=['retained', 1.],
                 selected_strength=1., validation_iou=[.5, .6 if accepted else .4])
    admission = dict(schema='counterfactual_extent_benefit_admission.v1',
                     source_only=True, benchmark_data_used=False,
                     parent_field_sha256='a'*64, source_routed_validation_passed=True,
                     selected=dict(validation_routed_iou=.6, validation_routed_parent_iou=.5))
    return scope, admission


def test_source_rejection_bypasses_scope_and_preserves_native_exactly():
    native = torch.tensor([[1., 2.]])
    scope, admission = receipts(False)
    def forbidden():
        raise AssertionError('Rejected scope must never execute')
    actual, decision = source_accepted_category_logits(
        native, forbidden, scope_receipt=scope, admission_receipt=admission)
    assert actual is native
    assert decision['reason'] == 'scope_source_rejected'


def test_accepted_scope_is_bounded_and_preserves_large_native_margins():
    native = torch.tensor([[5., 0.], [0., .2]])
    scope, admission = receipts()
    actual, decision = source_accepted_category_logits(
        native, lambda: torch.tensor([[-100., 100.], [100., -100.]]),
        scope_receipt=scope, admission_receipt=admission)
    assert decision['use_scope']
    assert actual[0].argmax() == native[0].argmax()
    assert (actual-native).abs().max() <= 1.


@pytest.mark.parametrize('corruption', ['ancestry', 'missing', 'string', 'contradiction', 'target', 'nan'])
def test_invalid_receipts_cannot_enable_scope(corruption):
    scope, admission = deepcopy(receipts())
    if corruption == 'ancestry': admission['parent_field_sha256'] = 'b'*64
    if corruption == 'missing': scope.pop('source_acceptance')
    if corruption == 'string': scope['source_acceptance'] = 'false'
    if corruption == 'contradiction': scope['validation_iou'][1] = .1
    if corruption == 'target': admission['benchmark_data_used'] = True
    if corruption == 'nan': scope['validation_iou'][1] = float('nan')
    with pytest.raises(ValueError):
        source_accepted_category_logits(torch.ones(1, 2), lambda: torch.ones(1, 2),
                                        scope_receipt=scope, admission_receipt=admission)


def test_admission_rejection_also_bypasses_scope():
    scope, admission = receipts()
    admission['source_routed_validation_passed'] = False
    admission['selected']['validation_routed_iou'] = .1
    native = torch.ones(1, 2)
    actual, decision = source_accepted_category_logits(native, lambda: 1/0,
        scope_receipt=scope, admission_receipt=admission)
    assert actual is native and decision['reason'] == 'admission_source_rejected'


def test_original_admission_numerical_tolerance_is_preserved():
    scope, admission = receipts()
    admission['selected']['validation_routed_iou'] = .5 - 5e-7
    _, decision = source_accepted_category_logits(torch.ones(1, 2),
        lambda: torch.zeros(1, 2), scope_receipt=scope, admission_receipt=admission)
    assert decision['use_scope']
