"""Opt-in local source-scope correction over existing text element scores.

The caller must bind the scope cache to the field/geometry and supply the
existing concept route. This preserves unknown/conflicting regions exactly,
but does NOT certify semantic correctness of confident teacher evidence.
"""
import torch


@torch.no_grad()
def refine_source_scope_consensus(scores, observed, mean, variance, count,
                                  concept_indices, accepted, *, modality,
                                  enabled=False):
    """Apply fixed bounded evidence only to admitted text routes.

Other modalities and the default disabled path return the original scores.
No field, query embedding, geometry, or input tensor is mutated.
"""
    if not enabled or modality != 'text':
        return scores
    if scores.ndim != 2 or observed.shape != (len(scores),):
        raise ValueError('scores and observed domain must align')
    if mean.ndim != 2 or mean.shape != variance.shape or mean.shape != count.shape or len(mean) != len(scores):
        raise ValueError('scope cache must align with the complete field')
    if concept_indices.shape != (scores.shape[1],) or accepted.shape != concept_indices.shape:
        raise ValueError('one concept route and admission per query required')
    if concept_indices.dtype not in (torch.int32, torch.int64) or count.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError('integer concept indices and view counts required')
    if observed.dtype != torch.bool or accepted.dtype != torch.bool:
        raise ValueError('observed and accepted must be Boolean masks')
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('finite probability scores required')
    if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any() or (count < 0).any():
        raise ValueError('finite nonnegative scope evidence required')
    if (mean < 0).any() or (mean > 1 + 1e-6).any():
        raise ValueError('scope means must be probabilities')
    if not accepted.any():
        return scores
    chosen = concept_indices[accepted]
    if (chosen < 0).any() or (chosen >= mean.shape[1]).any():
        raise ValueError('admitted concept index is outside the cache')
    result = scores.clone()
    for qi in torch.where(accepted)[0].tolist():
        ci = int(concept_indices[qi])
        mu = mean[:, ci].to(scores)
        var = variance[:, ci].to(scores)
        votes = count[:, ci].to(scores.device)
        reliable = observed & (votes >= 2) & (var <= .04) & ((mu >= .75) | (mu <= .25))
        delta = torch.logit(mu.clamp(.1, .9)).clamp(-2.197224577, 2.197224577)
        corrected = torch.sigmoid(torch.logit(scores[:, qi].clamp(1e-6, 1-1e-6)) + delta)
        result[:, qi] = torch.where(reliable, corrected, scores[:, qi])
    return result
