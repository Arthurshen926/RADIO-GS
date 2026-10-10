"""Initialization intervention with unchanged source objective and footprint measure."""
import time
import torch
from radio_gs.query_field.geometry import relaxed_subset_alpha
from scripts.research.conditional_subset_reference import silhouette_loss

def solve_fixed(field,initial,views,world_scale,*,normalizer,footprint=True,steps=240):
    active=[v for v in views if bool(v['positive'].any())]
    if not active:return initial.clone(),dict(active_views=0,steps=0,status='no_source_positive_keep_initializer')
    start=time.monotonic();cost=sum(v['footprint'] for v in active)
    cost=cost/torch.dot(cost,normalizer).clamp_min(1e-8)
    with torch.enable_grad():
        parameter=torch.nn.Parameter(torch.logit(initial.clamp(.001,.999)))
        optimizer=torch.optim.Adam([parameter],lr=.05)
        for step in range(steps):
            v=active[step%len(active)];tau=.5*(.1/.5)**(step/max(steps-1,1))
            member=torch.where(field.observed,(parameter/tau).sigmoid(),(initial>.5).float())
            alpha=relaxed_subset_alpha(field,member,v['camera'],v['height'],v['width'],world_scale)
            loss=silhouette_loss(alpha,v['positive'],v['negative'])
            penalty=.1*torch.dot(cost,parameter.sigmoid()) if footprint else loss.new_zeros(())
            objective=loss+penalty
            if not bool(torch.isfinite(objective)):raise ValueError('nonfinite joint objective')
            optimizer.zero_grad();objective.backward();optimizer.step()
        result=torch.where(field.observed,parameter.sigmoid().detach(),initial)
    return result,dict(active_views=len(active),steps=steps,footprint=footprint,final_objective=float(objective.detach()),
        final_source_loss=float(loss.detach()),final_footprint_penalty=float(penalty.detach()),seconds=time.monotonic()-start,
        transient_query_fit=True,field_updated=False,global_optimum_claim=False)
