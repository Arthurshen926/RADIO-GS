"""Partial-label retrieval diagnostics: unknown predictions are not false positives."""
import torch


def summarize_partial_episodes(rows):
    """Keep undefined ratios explicit and expose a fixed-denominator IoU score."""
    if not rows:
        raise ValueError('no evaluation episodes')
    result = {}
    counts = {}
    for key in rows[0]:
        if key == 'unknown_is_negative':
            continue
        values = [row[key] for row in rows if row[key] is not None]
        counts[key] = len(values)
        result[key] = sum(values) / len(values) if values else None
    result['defined_episode_counts'] = counts
    result['episode_count'] = len(rows)
    result['known_iou_all_episodes_zero_for_undefined'] = sum(
        row['known_iou'] if row['known_iou'] is not None else 0. for row in rows
    ) / len(rows)
    return result


def partial_retrieval_metrics(probability, target, known, supported=None, threshold=.5):
    if probability.shape != target.shape or known.shape != target.shape:
        raise ValueError('metric axes differ')
    if not torch.isfinite(probability).all():
        raise ValueError('nonfinite retrieval response')
    if supported is None:
        supported = torch.ones_like(known)
    if supported.shape != known.shape:
        raise ValueError('support axes differ')
    supported = supported.bool()
    known = known.bool() & supported
    if not torch.isfinite(target[known]).all():
        raise ValueError('known targets must be finite')
    truth = target.bool()
    pred = probability > threshold
    tp = int((pred & truth & known).sum())
    fp = int((pred & ~truth & known).sum())
    fn = int((~pred & truth & known).sum())
    negatives = int((~truth & known).sum())
    positives = tp + fn
    unknown = ~known & supported.bool()
    total_mass = float(probability[supported].sum())
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    return {
        'known_iou': ratio(tp, tp + fp + fn),
        'known_precision': ratio(tp, tp + fp),
        'known_positive_recall': ratio(tp, positives),
        'known_negative_false_positive_rate': ratio(fp, negatives),
        'known_fraction': float(known.float().mean()),
        'supported_fraction': float(supported.float().mean()),
        'unknown_positive_pixel_fraction': float((pred & unknown).float().mean()),
        'unknown_response_mass_fraction': ratio(float(probability[unknown].sum()), total_mass),
        'known_positive_count': positives, 'known_negative_count': negatives,
        'unknown_is_negative': False,
    }


def coverage_aware_retrieval_metrics(probability, target, known, supported, threshold=.5):
    """Score all explicit labels; unsupported responses are operational misses.

    Unlike conditional metrics, positive labels outside map coverage remain in
    the denominator. Unknown and conflicting labels remain excluded.
    """
    if supported.shape != probability.shape or known.shape != probability.shape:
        raise ValueError('coverage metric axes differ')
    supported = supported.bool()
    result = partial_retrieval_metrics(
        torch.where(supported, probability, torch.zeros_like(probability)),
        target, known, threshold=threshold,
    )
    positive = known.bool() & target.bool()
    count = int(positive.sum())
    covered = int((positive & supported).sum())
    result['label_positive_support_recall'] = covered / count if count else None
    result['unsupported_known_positive_count'] = count - covered
    result['supported_fraction'] = float(supported.float().mean())
    return result


def binary_ranking_metrics(probability, target):
    """Threshold-free full-label ranking diagnostics with grouped score ties."""
    if probability.shape != target.shape or not torch.isfinite(probability).all():
        raise ValueError('invalid ranking axes or response')
    score=probability.flatten();truth=target.bool().flatten()
    positives=int(truth.sum());negatives=len(truth)-positives
    if not len(score):raise ValueError('empty ranking domain')
    order=torch.argsort(score,descending=True,stable=True)
    values=score[order];labels=truth[order].double()
    ends=torch.cat([torch.where(values[:-1]!=values[1:])[0],torch.tensor([len(values)-1],device=score.device)])
    tp=labels.cumsum(0)[ends];seen=(ends+1).double();fp=seen-tp
    dtp=torch.diff(torch.cat([tp.new_zeros(1),tp]));dfp=torch.diff(torch.cat([fp.new_zeros(1),fp]))
    ap=float((dtp*(tp/seen)).sum()/positives) if positives else None
    auc=float((dfp*(tp-.5*dtp)).sum()/(positives*negatives)) if positives and negatives else None
    return {'average_precision':ap,'auroc':auc,'positive_fraction':positives/len(truth),
            'positive_mean_response':float(score[truth].mean()) if positives else None,
            'negative_mean_response':float(score[~truth].mean()) if negatives else None}
