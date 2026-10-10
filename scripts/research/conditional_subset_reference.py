"""Privileged joint-subset capacity reference, never a standard field decoder.

Uses fixed partial source labels and transient per-query parameters. No geometry
or persistent field state is optimized. This is not a global attainable bound.
"""
import math,time
import torch
from torch.nn import functional as F
from radio_gs.query_field.geometry import relaxed_subset_alpha


def silhouette_loss(alpha,positive,negative):
    cutoff=alpha.new_tensor(10.5/255)
    decision=torch.sigmoid(torch.logit(alpha.clamp(1e-6,1-1e-6))-torch.logit(cutoff))
    target=positive.to(alpha);known=positive|negative
    bce=F.binary_cross_entropy(decision.clamp(1e-5,1-1e-5),target,reduction='none')
    balanced=.5*((bce*positive).sum()/positive.sum().clamp_min(1)+(bce*negative).sum()/negative.sum().clamp_min(1))
    overlap=1-(2*(decision*target).sum()+1)/((decision*known).sum()+target.sum()+1)
    return balanced+overlap


@torch.no_grad()
def unoccluded_footprint(field,camera,height,width,world_scale):
    from gsplat.cuda._wrapper import fully_fused_projection
    k=camera.intrinsic.to(field.xyz).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
    radii,_,_,conics,_=fully_fused_projection(field.xyz,None,field.quaternions,field.scales,
        camera.world_to_camera.to(field.xyz)[None],k[None],width,height,near_plane=.001*world_scale,far_plane=100*world_scale)
    seen=radii[0]>0;cost=torch.zeros(len(field.xyz),device=field.xyz.device);c=conics[0,seen]
    cost[seen]=2*math.pi/(c[:,0]*c[:,2]-c[:,1].square()).clamp_min(1e-12).sqrt()*field.opacities.flatten()[seen]
    return cost*field.observed


def solve(field,initial,views,world_scale,*,footprint=True,steps=240):
    active=[v for v in views if bool(v['positive'].any())]
    if not active:return initial.clone(),dict(active_views=0,steps=0,status='no_source_positive_keep_initializer')
    start=time.monotonic();cost=sum(v['footprint'] for v in active)
    cost=cost/torch.dot(cost,initial).clamp_min(1e-8)
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
