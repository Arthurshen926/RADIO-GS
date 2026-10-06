"""Reference-seeded global extent selection on a locally scaled surface graph."""
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from gsplat import rasterization
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order, connected_components


def seeded_extent_cut(scores, observed, geo, camera, prior, width, height, positive, sam, field, strength=1., geometry='local', evidence='learned'):
    ids = torch.where(observed)[0]
    xyz = geo.xyz[ids].cpu().numpy()
    distance, neighbors = cKDTree(xyz).query(xyz, k=17, workers=4)
    distance, neighbors = distance[:, 1:], neighbors[:, 1:]
    local_scale = distance[:, 3].clip(1e-8)
    scale = np.sqrt(local_scale[:, None] * local_scale[neighbors])
    spatial = np.exp(-.5*(distance/scale)**2) * (distance <= 3*scale)
    if geometry=='retained':
        radius=3*float(np.median(distance[:,0][distance[:,0]>0]))
        spatial=(distance<=radius).astype(float)
    structure = F.normalize(field.latent.detach()[ids, field.native_semantic_dim:], dim=1)
    neighbor_tensor = torch.from_numpy(neighbors).to(ids.device)
    affinity = np.empty_like(spatial)
    for start in range(0, len(ids), 8192):
        stop = start+8192
        cosine = (structure[start:stop,None]*structure[neighbor_tensor[start:stop]]).sum(-1)
        affinity[start:stop] = torch.exp(-(1-cosine).clamp_min(0)/.5).cpu().numpy()
    weight = spatial*affinity
    if evidence=='scopes':
        membership=field.scope_graph.matrix.to_dense()[ids]
        supported=membership.norm(dim=1)>0
        membership=F.normalize(membership,dim=1)
        for start in range(0,len(ids),2048):
            stop=start+2048
            peers=neighbor_tensor[start:stop]
            same=(membership[start:stop,None]*membership[peers]).sum(-1).clamp(0,1)
            known=supported[start:stop,None]&supported[peers]
            weight[start:stop] *= torch.where(known,same.square(),torch.ones_like(same)).cpu().numpy()
    row = np.repeat(np.arange(len(ids)), 16)
    col = neighbors.ravel()
    graph = coo_matrix((weight.ravel(),(row,col)),shape=(len(ids),len(ids))).tocsr()
    graph = (graph+graph.T)*.5
    degree = np.asarray(graph.sum(1)).ravel().clip(1e-8)
    graph = graph.tocoo()
    graph.data /= np.sqrt(degree[graph.row]*degree[graph.col])
    graph = graph.tocsr()

    mask = sam.cpu().numpy().astype(np.uint8)
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    selected_labels = np.unique(labels[positive.cpu().numpy() & mask.astype(bool)])
    selected_labels = selected_labels[selected_labels != 0]
    core = cv2.erode(np.isin(labels, selected_labels).astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)
    k = camera.intrinsic.to(ids.device).clone()
    k[0] *= width/camera.width
    k[1] *= height/camera.height
    with torch.enable_grad():
        value = scores.detach().clone().requires_grad_(True)
        image,alpha,_ = rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=torch.cat([value,observed.float()[:,None]],1),viewmats=camera.world_to_camera.to(ids.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
        mass = image[0,...,1]
        valid = (mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
        response = torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0)
        total = torch.autograd.grad(response[valid].sum(),value,retain_graph=True)[0].flatten()
        prompt_mass = torch.autograd.grad(response[positive.to(ids.device)&valid].sum(),value,retain_graph=True)[0].flatten()
        core_mass = torch.autograd.grad(response[torch.from_numpy(core).to(ids.device)&valid].sum(),value)[0].flatten()
    seeds = (((prompt_mass/total.clamp_min(1e-8)>=.1)|(core_mass/total.clamp_min(1e-8)>=.9))&(total>1e-8))[ids].cpu().numpy()
    if not seeds.any():
        return scores, dict(no_supported_reference_seeds=True)
    p = scores[ids,0].cpu().numpy().clip(1e-6,1-1e-6)
    foreground,background = -np.log(p),-np.log1p(-p)
    weighted_degree = strength*np.asarray(graph.sum(1)).ravel()
    background[seeds] = foreground[seeds]+2*weighted_degree[seeds]+1
    offset = np.minimum(foreground,background)
    foreground,background = foreground-offset,background-offset
    count = len(ids)
    edges = graph.tocoo()
    rows = np.concatenate([edges.row,np.full(count,count),np.arange(count)])
    columns = np.concatenate([edges.col,np.arange(count),np.full(count,count+1)])
    capacities = np.rint(np.concatenate([strength*edges.data,background,foreground])*4096).astype(np.int64)
    network = coo_matrix((capacities,(rows,columns)),shape=(count+2,count+2)).tocsr()
    network.eliminate_zeros()
    flow = maximum_flow(network,count,count+1)
    residual = network-flow.flow
    residual.data = (residual.data>0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual,count,directed=True,return_predecessors=False)
    selected = np.zeros(count,dtype=bool)
    selected[reachable[reachable<count]] = True
    # The query asks for the reference-connected target, not all similar objects.
    edges = graph.tocoo()
    keep_edge = selected[edges.row]&selected[edges.col]
    components = coo_matrix((np.ones(int(keep_edge.sum())),(edges.row[keep_edge],edges.col[keep_edge])),shape=graph.shape).tocsr()
    _,component = connected_components(components,directed=False)
    selected &= np.isin(component,np.unique(component[seeds]))
    margin = np.log(p)-np.log1p(-p)+strength*(graph@(2*selected.astype(float)-1))
    probability = 1/(1+np.exp(-margin.clip(-30,30)))
    probability = np.where(selected,np.maximum(probability,.500001),0)
    probability[seeds] = np.maximum(probability[seeds],.55)
    result = scores.clone()
    result[ids,0] = torch.from_numpy(probability).to(result)
    from scripts.audits.predict_nvos_positive_ensemble_next32 import refine_positive
    result,repair = refine_positive(result,observed,geo,camera,prior,width,height,positive,sam,'prompt_only')
    return result,dict(strength=strength,seed_elements=int(seeds.sum()),selected_elements=int(selected.sum()),added_below_original_threshold=int((selected&(p<.5)).sum()),removed_original_positive=int((~selected&(p>=.5)).sum()),graph_edges=int(graph.nnz),geometry_graph=geometry,structure_evidence=evidence,flow_integer_scale=4096,flow_value=int(flow.flow_value),sam_outside_negative=False,positive_repair=repair)
