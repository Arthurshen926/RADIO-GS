"""Associate a reference prompt with stored multiview scopes before completion."""
import cv2
import numpy as np
import torch
from gsplat import rasterization
from radio_gs.models.grounded_semantic_transfer import greedy_native_support_union


@torch.no_grad()
def complete_reference_scopes(scores,observed,geo,camera,prior,width,height,positive,sam,field):
    membership=field.scope_graph.matrix.to_dense()
    k=camera.intrinsic.to(scores.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
    mask=sam.cpu().numpy().astype(np.uint8);_,labels=cv2.connectedComponents(mask,connectivity=8)
    chosen=np.unique(labels[positive.cpu().numpy()&mask.astype(bool)]);chosen=chosen[chosen!=0]
    target=torch.from_numpy(np.isin(labels,chosen)).to(scores.device).flatten()
    projected=[];visible=None
    for start in range(0,membership.shape[1],16):
        color=torch.cat([membership[:,start:start+16]*observed[:,None],observed.float()[:,None]],1)
        rendered,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=color,viewmats=camera.world_to_camera.to(scores.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
        mass=rendered[0,...,-1];visible=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
        projected.append(((rendered[0,...,:-1]/mass[...,None].clamp_min(1e-8)>.5)&visible[...,None]).reshape(height*width,-1))
    masks=torch.cat(projected,1)
    target=target&visible.flatten()
    union,quality,selected=greedy_native_support_union(masks,target[:,None],preserve_precision=True)
    picked=selected[0]
    if not picked or quality[0]<.5:
        return scores,dict(selected_scopes=picked,reference_union_iou=float(quality[0]),accepted=False)
    selected_union=membership[:,picked].amax(1)
    intersection=(union[:,0]&target).sum().float()
    precision=intersection/union[:,0].sum().clamp_min(1)
    # Reference compatibility supplies independent extent evidence. Missing
    # reference coverage remains unknown and never suppresses the parent.
    positive=selected_union>.5
    confidence=selected_union*precision
    candidate=1-(1-scores[:,0])*(1-confidence)
    output=scores.clone();output[:,0]=torch.where(positive&observed,candidate,scores[:,0])
    return output,dict(selected_scopes=picked,reference_union_iou=float(quality[0]),reference_precision=float(precision),accepted=True,new_selected_elements=int(((output[:,0]>=.5)&(scores[:,0]<.5)).sum()),outside_reference_is_negative=False)
