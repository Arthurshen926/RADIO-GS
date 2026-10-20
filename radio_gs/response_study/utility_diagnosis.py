"""Descriptive utility and input-domain audits; never modify a decision."""
import numpy as np
from radio_gs.querying.response_replacement import choose_response


def range_audit(fit, target, names, *, tolerance=1e-6):
    fit=np.asarray(fit,dtype=float);target=np.asarray(target,dtype=float)
    if (fit.ndim!=2 or target.ndim!=2 or fit.shape[1]!=target.shape[1]
            or fit.shape[1]!=len(names) or not len(fit) or not len(target)
            or not np.isfinite(fit).all() or not np.isfinite(target).all()):
        raise ValueError('aligned finite nonempty fit and target signatures required')
    low,high=fit.min(0),fit.max(0)
    outside=(target<low-tolerance)|(target>high+tolerance)
    rows=[dict(feature=name,fit_min=float(low[i]),fit_max=float(high[i]),
        target_min=float(target[:,i].min()),target_max=float(target[:,i].max()),
        outside_fit_count=int(outside[:,i].sum()),outside_fit_fraction=float(outside[:,i].mean()),
        constant_in_fit=bool(high[i]-low[i]<=tolerance)) for i,name in enumerate(names)]
    return dict(fit_candidates=len(fit),target_candidates=len(target),features=rows,
        target_candidates_outside_any_fit_range=int(outside.any(1).sum()),
        numerical_tolerance=tolerance,range_exceedance_is_not_causal_attribution=True)


def gain_audit(rows, margin):
    if not rows:raise ValueError('nonempty fixed-policy episodes required')
    records=[]
    for row in rows:
        gains=np.asarray(row['gains'],dtype=float);utility=np.asarray(row['ious'],dtype=float)
        if gains.shape!=utility.shape or not np.isfinite(utility).all():
            raise ValueError('gains and complete-candidate utilities must align')
        chosen=choose_response(gains,margin,evidence_available=row['evidence_available'])
        top=int(np.argmax(gains));best=int(np.argmax(utility))
        records.append(dict(scene=row['scene'],query_index=row['query_index'],candidate=chosen,
            baseline=float(utility[0]),selected=float(utility[chosen]),oracle=float(utility[best]),
            actual_gain=float(utility[chosen]-utility[0]),predicted_gain=float(gains[chosen]),
            top_candidate=top,top_predicted_gain=float(gains[top]),
            top_actual_gain=float(utility[top]-utility[0]),
            available_gain=float(utility[best]-utility[0]),regret=float(utility[best]-utility[chosen])))
    rooms=sorted({r['scene'] for r in records})
    keys=('baseline','selected','oracle','actual_gain','available_gain','regret')
    macro={k:float(np.mean([np.mean([r[k] for r in records if r['scene']==s]) for s in rooms])) for k in keys}
    accepted=[r for r in records if r['candidate']!=0]
    top=max(records,key=lambda r:r['top_predicted_gain'])
    return dict(queries=len(records),rooms=len(rooms),margin=margin,scene_macro=macro,
        changed=len(accepted),improved=sum(r['actual_gain']>1e-12 for r in accepted),
        harmed=sum(r['actual_gain']<-1e-12 for r in accepted),
        accepted_query_mean_predicted_gain=float(np.mean([r['predicted_gain'] for r in accepted])) if accepted else None,
        accepted_query_mean_actual_gain=float(np.mean([r['actual_gain'] for r in accepted])) if accepted else None,
        largest_predicted_replacement=top,records=records)
