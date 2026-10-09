"""Read-only operator ablations; no predictions, target labels, fitting or tuning."""
import torch


def association_variants(relation, support, scale):
    weights = support.weights.to(relation.scales); weights = weights / weights.sum()
    values = {k: torch.sparse.mm(relation.matrix(k, support.indices).transpose(0, 1),
        weights[:, None]).flatten() for k in ['association', 'positive', 'negative']}
    p, n = values['positive'], values['negative']
    ratio = relation.scales / scale; compatible = (ratio >= .5) & (ratio <= 2)
    def normalized(a):
        total = a.new_zeros(relation.view_count).scatter_add_(0, relation.views, a)
        return a / total[relation.views].clamp_min(1e-12)
    hard = values['association']
    original = normalized(hard * ((hard >= .5) & compatible))
    soft_global = normalized(p * ((p >= .5) & compatible))
    admitted = (p > n) & compatible
    known = p + n
    preference = torch.where(admitted, p / known.clamp_min(1e-12), 0.)
    confidence = p.new_zeros(relation.view_count).scatter_reduce_(0, relation.views,
        torch.where(admitted, known, 0.), reduce='amax', include_self=True).clamp_max(1.)
    conditional_global = normalized(preference) * confidence[relation.views]
    confidence_global = normalized(torch.where(admitted, p, 0.)) * confidence[relation.views]
    mixture = relation.coefficients(support, scale)
    return dict(maximum_hard=float(hard.max()), maximum_soft_positive=float(p.max()),
        maximum_conditional_preference=float((p / known.clamp_min(1e-12)).max()),
        variants={k: dict(coordinates=int((a > 0).sum()),
            views=int(torch.unique(relation.views[a > 0]).numel()), effective_view_mass=float(a.sum()))
            for k, a in [('hard_global_majority', original), ('soft_global_majority', soft_global),
                         ('soft_global_conditional', conditional_global),
                         ('soft_global_confidence', confidence_global), ('component_mixture', mixture)]},
        interpretation='Association-only function contrasts; no claim of isolated benchmark improvement')
