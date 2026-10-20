"""Attribute a sealed decision, without fitting or changing its guard."""
from .utility_diagnosis import gain_audit


def decision_attribution(rows, margin):
    guarded = gain_audit(rows, margin)
    zero = gain_audit(rows, 0.)
    result = dict(guarded=guarded, unrestricted=zero)
    g, z = guarded['scene_macro'], zero['scene_macro']
    result['decomposition'] = dict(available_gain=g['available_gain'],
        ranking_regret=z['regret'], retention_cost=z['actual_gain']-g['actual_gain'],
        actual_gain=g['actual_gain'])
    # Retention cost is signed: a negative cost is damage PREVENTED by the guard.
    d = result['decomposition']
    if abs(d['available_gain']-d['ranking_regret']-d['retention_cost']-d['actual_gain']) > 1e-10:
        raise ValueError('complete-answer decision decomposition failed')
    return result
