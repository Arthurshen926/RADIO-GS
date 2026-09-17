"""Source-track scale estimation with disjoint fit/admission/audit point IDs."""
import torch


def relative_error_stats(values):
    x=torch.as_tensor(values,dtype=torch.float64)
    if not len(x):return {'count':0,'median':None,'p90':None,'within_5pct':None}
    return dict(count=len(x),median=float(x.median()),p90=float(x.quantile(.9)),within_5pct=float((x<.05).double().mean()))


def fit_independent_view_scale(predicted,reference,point_ids,valid,*,minimum_fit=32,minimum_gate=16,maximum_median=.05,maximum_p90=.15):
    p=torch.as_tensor(predicted,dtype=torch.float64).flatten();r=torch.as_tensor(reference,dtype=torch.float64).flatten();ids=torch.as_tensor(point_ids).flatten();v=torch.as_tensor(valid,dtype=torch.bool).flatten()
    if not p.shape==r.shape==ids.shape==v.shape:raise ValueError('unaligned source scale samples')
    if ids.is_floating_point() or minimum_fit<3 or minimum_gate<1 or not 0<maximum_median<=maximum_p90:raise ValueError('invalid scale policy or point IDs')
    legal=v&torch.isfinite(p)&torch.isfinite(r)&(p>0)&(r>0)&(ids>=0)
    fit=legal&(ids%5!=0);gate=legal&(ids%10==0);audit=legal&(ids%10==5)
    if int(fit.sum())<minimum_fit:
        return dict(accepted=False,scale=None,reason='insufficient_fit_tracks',fit_points=int(fit.sum()),gate_points=int(gate.sum()),audit_points=int(audit.sum()))
    scale=float((r[fit]/p[fit]).median());errors=(scale*p-r).abs()/r.clamp_min(1e-12)
    gs=relative_error_stats(errors[gate]);audit_stats=relative_error_stats(errors[audit])
    reason='insufficient_admission_tracks' if gs['count']<minimum_gate else 'heldout_depth_shape_error' if gs['median']>maximum_median or gs['p90']>maximum_p90 else None
    return dict(accepted=reason is None,scale=scale,reason=reason,fit_points=int(fit.sum()),gate_points=int(gate.sum()),audit_points=int(audit.sum()),fit=relative_error_stats(errors[fit]),gate=gs,audit=audit_stats)
