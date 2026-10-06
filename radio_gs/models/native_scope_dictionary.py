"""Sparse native semantic distributions, separated by source-region granularity."""
import torch
from torch.nn import functional as F


def attach_scope_text_readout(model, dictionary):
    """Use native multi-granularity evidence in the declared native modalities.

    Scope heads retain their original coordinates. Image anchoring uses its
    own existing residual decoder; prompt readout is unchanged.
    """
    original=model.native_cosine
    def native_cosine(query, rows=None, *, modality='text'):
        if modality not in dictionary.modalities:return original(query,rows,modality=modality)
        if modality=='text':anchor=(model.latent[:,:64],model.native_decoder.weight)
        else:
            def anchor(q,start,stop):
                return original(q,torch.arange(start,stop,device=q.device),modality=modality)
        value=dictionary.scores(query,pool='anchored',anchor=anchor)
        if modality!='text':
            value=torch.where(dictionary.valid.any(1)[:,None],value,original(query,modality=modality))
        return value if rows is None else value[rows]
    model.native_cosine=native_cosine
    return model


def pixel_scope_owners(masks):
    """Choose smallest, median and largest covering masks independently per pixel.

    Area order separates granularity without certifying containment or matching
    physical object identity across views.
    """
    order=masks.sum(1).argsort(stable=True)
    sorted_masks=masks[order].bool()
    ranks=sorted_masks.long().cumsum(0)
    count=sorted_masks.sum(0)
    owners=[]
    for rank in (torch.ones_like(count),(count+1)//2,count):
        match=sorted_masks&(ranks==rank[None])
        owners.append(order[match.long().argmax(0)])
    return torch.stack(owners),count>0


def containment_scope_owners(masks):
    """Follow observed mask containment rather than unrelated area ranks.

    The chain is local to a source image; it makes no cross-view identity claim.
    A missing ancestor repeats the available scope instead of inventing one.
    """
    fine,covered=pixel_scope_owners(masks)
    area=masks.sum(1).float().clamp_min(1)
    intersection=masks.float()@masks.float().T
    contains=(intersection/area[:,None]>=.9)&(area[None]>=area[:,None])
    coarse=area[None].expand_as(intersection).masked_fill(~contains,-1).argmax(1)
    chain=contains&contains[:,coarse].T
    target=(area.log()+area[coarse].log())/2
    middle=(area.log()[None]-target[:,None]).abs().masked_fill(~chain,float('inf')).argmin(1)
    smallest=fine[0]
    return torch.stack([smallest,middle[smallest],coarse[smallest]]),covered


@torch.no_grad()
def spherical_dictionary(features, size=512, iterations=15):
    features = F.normalize(features.float(), dim=-1)
    size = min(size, len(features))
    # Farthest-first covers infrequent region descriptions as well as common ones.
    chosen = [0]
    similarity = features @ features[0]
    for _ in range(1, size):
        index = int(similarity.argmin())
        chosen.append(index)
        similarity = torch.maximum(similarity, features @ features[index])
    centers = features[chosen].clone()
    for _ in range(iterations):
        labels = (features @ centers.T).argmax(1)
        sums = torch.zeros_like(centers).index_add_(0, labels, features)
        occupied = sums.norm(dim=1) > 0
        centers[occupied] = F.normalize(sums[occupied], dim=-1)
    return centers, (features @ centers.T).argmax(1)


class NativeScopeDictionary:
    """A query-independent field with native-space codewords and sparse weights.

    Each row contains three scale distributions. Query matching never consults
    source images, proposal masks, query lists, or the evaluation annotations.
    """
    def __init__(self, payload, device='cuda'):
        self.codebook = F.normalize(payload['codebook'].to(device).float(), dim=1)
        self.indices = payload['indices'].to(device).long()
        self.weights = payload['weights'].to(device).float()
        self.weights /= self.weights.sum(-1, keepdim=True).clamp_min(1e-8)
        self.valid = self.weights.sum(-1) > 0
        self.observed = payload['observed'].to(device)
        self.temperature = payload.get('temperature', 20.)
        self.local_scope_routing = payload.get('local_scope_routing', False)
        self.query_code_routing = payload.get('query_code_routing', False)
        self.modalities = payload.get('native_modalities', ['text'])
        self.scale_prior = payload.get('scale_prior')
        if self.scale_prior is not None:
            self.scale_prior = self.scale_prior.to(device).float()

    def scores(self, query, pool='mixture', anchor=None):
        query = F.normalize(query.float().to(self.codebook.device), dim=1)
        similarity = self.codebook @ query.T
        if pool in ('routed','cosine_routed','anchored'):
            standardized=(similarity-similarity.mean(0))/similarity.std(0).clamp_min(1e-6)
            prior=(standardized*2).softmax(0).T@self.scale_prior
        gram=self.codebook@self.codebook.T if pool.startswith('cosine') or pool=='anchored' else None
        if pool=='anchored':
            if not callable(anchor):
                anchor_features,basis=anchor
                code_to_anchor=self.codebook@basis
        chunks=[]
        for start in range(0, len(self.indices), 8192):
            ids=self.indices[start:start+8192]
            weight=self.weights[start:start+8192]
            values=similarity[ids]
            if pool.startswith('cosine') or pool=='anchored':
                inner=gram[ids[...,None],ids[...,None,:]]
                if pool=='anchored':
                    if callable(anchor):
                        stop=start+len(ids)
                        native_score=anchor(query,start,stop)
                        code_agreement=anchor(self.codebook,start,stop).gather(1,ids.reshape(len(ids),-1)).reshape_as(weight)
                    else:
                        native=F.normalize(anchor_features[start:start+8192],dim=1)
                        native_score=native@(query@basis).T
                        code_agreement=(native[:,None,None]*code_to_anchor[ids]).sum(-1)
                if pool=='anchored' and self.query_code_routing:
                    posterior=(values*self.temperature+weight.clamp_min(1e-12).log()[...,None]).softmax(2)
                    norm=(inner[...,None]*posterior[:,:,:,None,:]*posterior[:,:,None,:,:]).sum((2,3)).clamp_min(1e-10).sqrt()
                    matched=(values*posterior).sum(2)/norm
                    agreement=(code_agreement[...,None]*posterior).sum(2)/norm
                    disagreement=(1-agreement.square()).clamp_min(0).sqrt()
                    strength=(.25/disagreement.clamp_min(1e-8)).clamp_max(1)
                    residual=matched-agreement*native_score[:,None]
                    scores=(native_score[:,None]+strength*residual)/(1+(strength*disagreement).square()).sqrt()
                    available=prior.T[None]*self.valid[start:start+8192,...,None]
                    if self.local_scope_routing:
                        available=available*((scores-scores.max(1,keepdim=True).values)*self.temperature).exp()
                    available=available/available.sum(1,keepdim=True).clamp_min(1e-8)
                    chunks.append((scores*available).sum(1).masked_fill(~self.valid[start:start+8192].any(1)[:,None],-1))
                    continue
                norm=(inner*weight[...,None]*weight[...,None,:]).sum((-1,-2)).clamp_min(1e-10).sqrt()
                scores=(values*weight[...,None]).sum(2)/norm[...,None]
                if pool=='anchored':
                    agreement=(code_agreement*weight).sum(-1)/norm
                    disagreement=(1-agreement.square()).clamp_min(0).sqrt()
                    strength=(.25/disagreement.clamp_min(1e-8)).clamp_max(1)
                    residual=scores-agreement[...,None]*native_score[:,None]
                    scores=(native_score[:,None]+strength[...,None]*residual)/(1+(strength*disagreement).square()).sqrt()[...,None]
            elif pool=='mean':
                scores=(values*weight[...,None]).sum(2)
            else:
                scores=torch.logsumexp(values*self.temperature+weight.clamp_min(1e-12).log()[...,None],dim=2)/self.temperature
            scores=scores.masked_fill(~self.valid[start:start+8192,...,None],-1)
            # Different scopes remain distinguishable until the query is present.
            if pool in ('routed','cosine_routed','anchored'):
                available=prior.T[None]*self.valid[start:start+8192,...,None]
                if self.local_scope_routing:
                    local=(scores-scores.max(1,keepdim=True).values)*self.temperature
                    available=available*local.exp()
                available=available/available.sum(1,keepdim=True).clamp_min(1e-8)
                routed=((scores*available).sum(1) if pool in ('cosine_routed','anchored') else
                    torch.logsumexp(scores*self.temperature+available.clamp_min(1e-12).log(),dim=1)/self.temperature)
                chunks.append(routed.masked_fill(~self.valid[start:start+8192].any(1)[:,None],-1))
            else:
                chunks.append(scores.max(1).values)
        return torch.cat(chunks)
