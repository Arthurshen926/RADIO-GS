"""Post-evaluation candidate capacity and decision regret, never a policy.

Every menu oracle picks ONE candidate for all reported outputs. Diagnostics
cannot choose separate winners for individual submenus or alter predictions.
"""
import numpy as np


def menu_capacity(candidate_metrics, selected, *, protect_all=False):
    values=np.asarray(candidate_metrics,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise ValueError('each intact category candidate needs all three output utilities')
    if not 0<=selected<len(values):raise ValueError('selected candidate outside fixed pool')
    eligible=np.all(values>=values[0]-1e-12,axis=1) if protect_all else np.ones(len(values),bool)
    utility=values.mean(1)
    best=int(np.argmax(np.where(eligible,utility,-np.inf)))
    return dict(oracle_candidate=best,oracle_metrics=values[best].tolist(),
        baseline_metrics=values[0].tolist(),selected_metrics=values[selected].tolist(),
        available_mean_gain=float(utility[best]-utility[0]),
        decision_mean_regret=float(utility[best]-utility[selected]),
        selected_mean_gain=float(utility[selected]-utility[0]),
        all_outputs_protected=protect_all,diagnostic_only=True)


def instance_capacity(current, historical, candidate_hash, actual_iou):
    if (current['query_id']!=historical['query_id'] or current['candidates']!=historical['candidates']
            or current['candidate_selection_sha256']!=candidate_hash):
        raise ValueError('historical utilities require identical complete candidate sets and order')
    values=np.asarray(historical['candidate_iou'],dtype=float)
    winner=current['arms']['full']['candidate']
    if abs(values[winner]-actual_iou)>1e-12:
        raise ValueError('selected candidate utility differs from fresh target scoring')
    return dict(query_id=current['query_id'],scene=historical['scene'],modality=historical['modality'],
        candidate=winner,baseline_iou=float(values[0]),selected_iou=float(actual_iou),
        oracle_iou=float(values.max()),fixed_provider_iou=float(values[1]),
        available_gain=float(values.max()-values[0]),decision_regret=float(values.max()-actual_iou),
        same_choice_as_blind=winner==current['arms']['query_blind']['candidate'],
        same_choice_as_permuted=winner==current['arms']['permuted']['candidate'])
