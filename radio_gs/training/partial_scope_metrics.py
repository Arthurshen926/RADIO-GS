"""Query-view equal partial-label risk; NaN denotes absent support internally."""
import math
import torch


def ratio(num, den):
    return torch.where(den > 0, num / den.clamp_min(1), torch.full_like(num.float(), float('nan')))


def odds_probability(logits, delta):
    p = logits.sigmoid()
    return (logits + (torch.logit((p + delta).clamp(1e-6, 1-1e-6))
                      - torch.logit(p.clamp(1e-6, 1-1e-6)))).sigmoid()


def extent_measures(probability, example, parent=None):
    v = example['visible'][:, None].bool().expand_as(probability)
    k = example['known'].bool() & v
    p = example['target'].bool() & k
    n = k & ~p
    u = v & ~k
    selected = (probability > .5) & v
    tp = (selected & p).sum(0).float()
    pc, nc, uc = p.sum(0), n.sum(0), u.sum(0)
    pe = ratio(((probability-1).square()*p).sum(0), pc)
    ne = ratio((probability.square()*n).sum(0), nc)
    classes = (pc > 0).long() + (nc > 0).long()
    result = dict(known_iou=ratio(tp, ((selected & k) | p).sum(0)),
                  positive_recall=ratio(tp, pc), negative_selection=ratio((selected & n).sum(0).float(), nc),
                  unknown_selection=ratio((selected & u).sum(0).float(), uc),
                  known_brier=ratio(torch.nan_to_num(pe)+torch.nan_to_num(ne), classes),
                  positive_brier=pe, negative_brier=ne,
                  closed_mask_iou=ratio(tp, (selected | p).sum(0)),
                  positive_count=pc, negative_count=nc, unknown_count=uc,
                  visible_count=v.sum(0), class_count=classes)
    # Exact historical objective, including invisible target and old empty-union convention.
    target = example['target'].bool()
    result['legacy_visible_iou'] = (selected & target).sum(0).float() / (selected | target).sum(0).clamp_min(1)
    if parent is not None:
        result['unknown_absolute_change'] = ratio(((probability-parent).abs()*u).sum(0), uc)
        result['unknown_selection_change'] = ratio((((parent > .5) != selected) & u).sum(0).float(), uc)
    return result


def aggregate(rows):
    """Every row is one query-view; export nulls and explicit metric denominators."""
    keys = rows[0].keys() if rows else []
    summary = {'query_views': len(rows), 'valid_counts': {}}
    for key in keys:
        vals = [r[key] for r in rows if math.isfinite(r[key])]
        summary[key] = sum(vals)/len(vals) if vals else None
        summary['valid_counts'][key] = len(vals)
    summary['support'] = {name: sum(r.get(name, 0) for r in rows) for name in
                          ('positive_count','negative_count','unknown_count','visible_count')}
    summary['episode_classes'] = {str(c): sum(r['class_count'] == c for r in rows) for c in (0,1,2)}
    return summary


def episode_rows(metrics, view, columns):
    return [dict(view=view, query=int(col), **{k: float(v[j]) for k,v in metrics.items()})
            for j,col in enumerate(columns)]


def select_ridge(choices, rule):
    if rule == 'legacy_visible_iou':
        return max(choices, key=lambda r:r['legacy_visible_iou'])
    if rule != 'partial_known_brier':
        raise ValueError(rule)
    valid = [r for r in choices if r['known_brier'] is not None and math.isfinite(r['known_brier'])]
    if not valid:
        raise ValueError('No valid held-source supervision; cannot select ridge')
    minimum = min(r['known_brier'] for r in valid)
    return max((r for r in valid if r['known_brier'] <= minimum + 1e-6), key=lambda r:r['ridge'])
