"""Experimental biased gradient for the existing probability-clamp forward.

Inference is unchanged. Outside the probability interval, only a backward
direction that moves raw values back toward that interval is passed through.
This is deliberately not the mathematical derivative of the clamp.
"""
import torch
from torch.nn import functional as F


class _DirectionalClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw):
        if not torch.isfinite(raw).all():
            raise ValueError('Finite scope raw probabilities required')
        ctx.save_for_backward(raw)
        return raw.clamp(1e-6, 1-1e-6)

    @staticmethod
    def backward(ctx, gradient):
        raw, = ctx.saved_tensors
        inside = (raw >= 1e-6) & (raw <= 1-1e-6)
        toward = ((raw < 1e-6) & (gradient < 0)) | ((raw > 1-1e-6) & (gradient > 0))
        return torch.where(inside | toward, gradient, 0)


def directional_probability_clamp(raw):
    return _DirectionalClamp.apply(raw)


def recovered_source_concept_probability(model, query):
    """Exact model forward; replace gradients only on admitted concept routes.

    This deliberately supports only the retained bounded-fallback architecture.
    Route/admission are derived independently, checked against real model output,
    and left hard. No gradient is passed through their decisions.
    """
    head = model.grounded_transfer
    selection = head.identity_selection
    if (head.hidden_dim or selection.get('completion_mode') != 'native_bounded_extent'
            or selection.get('calibrate_accepted_concepts', False)
            or selection.get('combine_candidates') or 'trusted_concept_start' not in selection
            or model.extent_benefit_admission is None or model.query_scope_correction is not None
            or model.bounded_extent is not None or 'text' not in model.native_query_thresholds):
        raise ValueError('Only retained linear concept/bounded-fallback architecture supported')
    actual = model.logits(query, 'text').sigmoid() * model.observed[:, None]
    with torch.no_grad():
        index, concept_accepted, _ = head.route(query)
        column = index + selection['trusted_concept_start']
    features = head.features(model.latent)
    raw = F.linear(features, head.decoder.weight[column], head.decoder.bias[column])
    if head.residual is not None:
        hidden = head.residual[1](head.residual[0](features))
        raw = raw + F.linear(hidden, head.residual[2].weight[column], head.residual[2].bias[column])
    if not torch.isfinite(actual).all() or not torch.isfinite(raw).all():
        raise ValueError('Finite actual and reconstructed source probabilities required')
    with torch.no_grad():
        native = model.native_cosine(query)
        threshold = model.native_query_thresholds['text'](query)
        proposed = torch.logit(raw.clamp(1e-6, 1-1e-6))
        admitted = model.extent_benefit_admission(native, threshold, proposed, model.observed,
                                                 concept_accepted, query=query, head=head)
        active = concept_accepted & admitted
        expected = raw.clamp(1e-6, 1-1e-6) * model.observed[:, None]
        error = float((actual[:, active] - expected[:, active]).abs().max()) if active.any() else 0.
        if error > 2e-6:
            raise RuntimeError('Source concept route differs from the real inference path')
    surrogate = directional_probability_clamp(raw) * model.observed[:, None]
    replacement = actual.detach() + (surrogate - surrogate.detach())
    result = torch.where(active[None], replacement, actual)
    if not torch.equal(result.detach(), actual.detach()):
        raise RuntimeError('Gradient recovery changed actual forward probabilities')
    return result, dict(queries=len(query), replaced_concepts=int(active.sum()),
                        hard_or_rejected_queries=int((~active).sum()),
                        concept_probability_replay_max_error=error,
                        actual_forward_bit_exact=True)
