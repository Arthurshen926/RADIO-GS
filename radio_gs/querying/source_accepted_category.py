"""Source-validated opt-in category correction with lazy scope evaluation."""
import math
from typing import Callable, Mapping

import torch

from radio_gs.querying.bounded_native_scope import refine_native_scope_logits


def category_scope_decision(scope: Mapping, admission: Mapping) -> dict:
    """Check both source stages and their ancestry; never use target metrics.

    These source checks are development checks, not independent validation or
    a guarantee of native-harm protection on unseen queries.
    """
    for receipt in (scope, admission):
        if receipt.get('source_only') is not True or receipt.get('benchmark_data_used') is not False:
            raise ValueError('Source-only receipt required')
    if scope.get('schema') != 'bounded_fallback_scope.v1':
        raise ValueError('Unsupported scope receipt')
    if admission.get('schema') != 'counterfactual_extent_benefit_admission.v1':
        raise ValueError('Unsupported admission receipt')
    digest = scope.get('field_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or admission.get('parent_field_sha256') != digest:
        raise ValueError('Admission does not descend from the validated scope field')
    flags = (scope.get('source_acceptance'), admission.get('source_routed_validation_passed'))
    if any(type(flag) is not bool for flag in flags):
        raise ValueError('Explicit boolean source acceptance required')
    order, values = scope['score_order'], scope['validation_iou']
    if len(order) != len(values) or order[0] != 'retained':
        raise ValueError('Invalid scope validation comparison')
    index = order.index(scope['selected_strength'])
    if index == 0 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError('Invalid scope validation score')
    if flags[0] != (values[index] >= values[0]):
        raise ValueError('Scope acceptance contradicts its source validation')
    selected = admission['selected']
    actual_admission = selected['validation_routed_iou'] >= selected['validation_routed_parent_iou'] - 1e-6
    if any(not math.isfinite(float(selected[k])) for k in
           ('validation_routed_iou', 'validation_routed_parent_iou')) or flags[1] != actual_admission:
        raise ValueError('Admission acceptance contradicts its source validation')
    accepted = all(flags)
    return dict(use_scope=accepted,
                readout='bounded_scope' if accepted else 'calibrated_native',
                reason='source_stages_passed' if accepted else
                       'scope_source_rejected' if not flags[0] else 'admission_source_rejected')


def source_accepted_category_logits(native: torch.Tensor, scope: Callable[[], torch.Tensor],
                                    *, scope_receipt: Mapping, admission_receipt: Mapping,
                                    max_correction: float = 1.0):
    decision = category_scope_decision(scope_receipt, admission_receipt)
    if not native.is_floating_point() or not torch.isfinite(native).all():
        raise ValueError('Finite floating native logits required')
    if not math.isfinite(float(max_correction)) or max_correction < 0:
        raise ValueError('Invalid correction bound')
    if not decision['use_scope'] or max_correction == 0:
        return native, decision
    return refine_native_scope_logits(native, scope(), max_correction=max_correction), decision
