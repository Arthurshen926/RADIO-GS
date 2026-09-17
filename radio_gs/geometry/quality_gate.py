"""Require rendered source-view quality before accepting a scale candidate."""
import math


def joint_scale_quality_gate(baseline, candidate, *, correspondence_accepted):
    directions={'coverage':1,'absrel':-1,'free_space_fraction':-1,'depth_agreement_fraction':1,'agreement_coverage':1}
    failures=[]
    for metric,direction in directions.items():
        before=float(baseline[metric]);after=float(candidate[metric])
        if not math.isfinite(before) or not math.isfinite(after) or direction*(after-before)<-1e-6:
            failures.append(metric)
    if not correspondence_accepted:failures.append('correspondence_validation')
    return {'accepted':not failures,'failed_checks':failures,'policy':'no degradation of rendered heldout-source coverage, depth error, free-space consistency and depth agreement; correspondence improvement also required','reference_is_ground_truth':False}
