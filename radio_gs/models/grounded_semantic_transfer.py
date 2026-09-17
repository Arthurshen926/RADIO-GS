"""Read source-grounded extent from an unchanged native/structural element field."""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def greedy_native_support_union(membership,support,*,preserve_precision=False):
    """Add candidates only when union IoU with native support strictly improves.

    Inputs must already exclude unobserved elements. This is an inference
    objective against the native field, not a guarantee of ground-truth IoU.
    """
    if membership.ndim!=2 or support.ndim!=2 or len(membership)!=len(support):
        raise ValueError('aligned element/candidate and element/query arrays required')
    masks=membership.bool();weights=masks.float();output=torch.zeros_like(support,dtype=torch.bool)
    selected=[];quality=[]
    for qi in range(support.shape[1]):
        target=support[:,qi].bool();union=output[:,qi];chosen=[];current=0.
        if target.any():
            for _ in range(membership.shape[1]):
                remaining=~union
                extra=(remaining&target).float()@weights
                area=remaining.float()@weights
                hit=(union&target).sum()+extra
                scores=hit/(union.sum()+area+target.sum()-hit).clamp_min(1)
                if preserve_precision and chosen:
                    # Compare counts by cross multiplication, not a tunable
                    # confidence threshold or rounded division.
                    retained=hit*union.sum()>=(union&target).sum()*(union.sum()+area)
                    scores=scores.masked_fill(~retained,-1)
                best,index=scores.max(0)
                if float(best)<=current:break
                current=float(best);chosen.append(int(index));union|=masks[:,index]
        selected.append(chosen);quality.append(current)
    return output,support.new_tensor(quality,dtype=torch.float32),selected


@torch.no_grad()
def rank_preserving_extent_threshold(cosine,target,observed):
    """Fit a native-score cutoff to a structural extent without changing ranks.

    Optimize element IoU over realizable strict-threshold sets, never split tied
    scores. Inputs cover the whole field regardless of requested output rows.
    """
    if cosine.shape!=target.shape or observed.shape!=(len(cosine),) or not observed.any():
        raise ValueError('aligned full-field scores, target and observed support required')
    if not torch.isfinite(cosine).all():raise ValueError('finite native scores required')
    values,order=cosine[observed].sort(dim=0,descending=True)
    positive=target[observed].bool().gather(0,order)
    if not positive.any(0).all():raise ValueError('nonempty target for each query required')
    hit=positive.cumsum(0);size=torch.arange(1,len(values)+1,device=values.device)[:,None]
    iou=hit/(size+positive.sum(0)[None]-hit)
    valid=torch.cat([values[:-1]>values[1:],torch.ones_like(values[-1:],dtype=torch.bool)])
    best=iou.masked_fill(~valid,-1).argmax(0);column=torch.arange(values.shape[1],device=values.device)
    next_value=values[(best+1).clamp_max(len(values)-1),column]
    return torch.where(best==len(values)-1,torch.nextafter(next_value,torch.full_like(next_value,-torch.inf)),next_value)


class GroundedSemanticTransfer(nn.Module):
    def __init__(self, latent_dim, native_dim, query_dim, num_groups, hidden_dim=0, routing=None, residual_dim=0, num_proposals=0, proposal_ranker=False, identity_selection=None):
        super().__init__()
        self.native_dim=native_dim
        self.hidden_dim=hidden_dim
        extent_count=num_proposals or num_groups
        self.decoder=(nn.Sequential(nn.Linear(latent_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,extent_count))
                      if hidden_dim else nn.Linear(latent_dim,extent_count))
        self.register_buffer('prototypes',torch.zeros(num_groups,query_dim))
        self.register_buffer('acceptance',torch.ones(num_groups))
        self.routing = routing
        self.num_proposals=num_proposals
        self.proposal_ranker=proposal_ranker
        self.identity_selection=identity_selection
        if identity_selection is not None:
            if not num_proposals or proposal_ranker:raise ValueError('native identity selection requires unranked proposals')
            if identity_selection.get('metric','mean_cosine') not in ('mean_cosine','support_iou'):
                raise ValueError('unknown native identity selection metric')
            if not 0<=identity_selection.get('minimum_support_iou',0.)<=1:
                raise ValueError('minimum support IoU must lie in [0,1]')
            if identity_selection.get('combine_candidates') and (identity_selection.get('combine_candidates')!='greedy_union' or identity_selection.get('metric')!='support_iou' or identity_selection.get('restrict_concepts',False)):
                raise ValueError('candidate union requires unrestricted native support IoU')
            start=identity_selection.get('trusted_concept_start')
            if start is not None and (start<0 or start+num_groups!=num_proposals or identity_selection.get('metric')!='support_iou'):
                raise ValueError('trusted concepts must be the final ordered overlap candidates')
            self.register_buffer('native_extent_means',torch.zeros(num_proposals,query_dim))
        if proposal_ranker:
            if not num_proposals:raise ValueError('proposal ranker requires proposals')
            self.register_buffer('proposal_quality',torch.zeros(num_groups,num_proposals))
            self.register_buffer('proposal_rank_weights',torch.tensor([1.,0.]))
        if num_proposals:
            self.register_buffer('proposal_embeddings',torch.zeros(num_proposals,query_dim))
            self.register_buffer('proposal_groups',torch.zeros(num_proposals,num_groups,dtype=torch.bool))
        if residual_dim and hidden_dim:
            raise ValueError('residual probability and logit MLP are mutually exclusive')
        self.residual = None
        if residual_dim:
            self.residual = nn.Sequential(nn.Linear(latent_dim,residual_dim),nn.ReLU(),nn.Linear(residual_dim,extent_count))
            nn.init.zeros_(self.residual[2].weight)
            nn.init.zeros_(self.residual[2].bias)
        if routing is not None:
            self.register_buffer('routing_center', torch.zeros(query_dim))
            self.register_buffer('routing_competitors', torch.zeros(routing.get('num_competitors', 0), query_dim))
            if routing.get('scope_gate',False):
                self.register_buffer('routing_scope_images',torch.zeros(num_groups,query_dim))
                self.register_buffer('routing_scope_weights',torch.zeros(2))
                self.register_buffer('routing_scope_bias',torch.tensor(0.))

    def features(self,latent):
        return torch.cat([F.normalize(latent[:,:self.native_dim],dim=-1),F.normalize(latent[:,self.native_dim:],dim=-1)],dim=-1)

    def route(self, query):
        q, prototypes = F.normalize(query,dim=-1), F.normalize(self.prototypes,dim=-1)
        if self.routing is not None:
            q = F.normalize(q-self.routing_center, dim=-1)
            prototypes = F.normalize(prototypes-self.routing_center, dim=-1)
        similarities=q@prototypes.T
        score,index=similarities.max(1)
        accepted=score>=self.acceptance[index]
        if self.routing is not None and len(self.routing_competitors):
            competitors=F.normalize(F.normalize(self.routing_competitors,dim=-1)-self.routing_center,dim=-1)
            accepted &= score > (q@competitors.T).amax(1)
        if self.routing is not None and self.routing.get('scope_gate',False):
            # Scope admission is learned from co-observed source masks, not from
            # the self-similarity of two wordings of a concept name.
            native_q=F.normalize(query,dim=-1)
            text=(native_q*F.normalize(self.prototypes[index],dim=-1)).sum(-1)
            visual=(native_q*self.routing_scope_images[index]).sum(-1)
            evidence=torch.stack([text,visual],-1)@self.routing_scope_weights+self.routing_scope_bias
            accepted &= evidence>=0
        return index, accepted, score

    def support_overlap(self,latent,native_support,observed):
        """Global element IoU; query/row batches never redefine the candidate scope."""
        if native_support is None or observed is None:
            raise ValueError('full native support and observed elements required')
        if native_support.shape[0]!=len(latent) or observed.shape!=(len(latent),):
            raise ValueError('support must cover the full element field')
        positive=native_support.bool()&observed[:,None]
        intersection=latent.new_zeros((positive.shape[1],self.num_proposals))
        mass=latent.new_zeros(self.num_proposals)
        with torch.no_grad():
            for start in range(0,len(latent),4096):
                end=start+4096;features=self.features(latent[start:end])
                extent=self.decoder(features)
                if self.hidden_dim:extent=extent.sigmoid()
                if self.residual is not None:extent=extent+self.residual(features)
                membership=((extent>.5)&observed[start:end,None]).float()
                intersection+=positive[start:end].float().T@membership
                mass+=membership.sum(0)
        union=positive.sum(0)[:,None]+mass[None]-intersection
        return intersection/union.clamp_min(1)

    def forward(self,query,latent,rows=None,native_threshold=None,native_support=None,observed=None):
        index,accepted,_=self.route(query)
        concept_index,concept_accepted=index,accepted
        if self.identity_selection is not None:
            overlap=self.identity_selection.get('metric')=='support_iou'
            similarities=(self.support_overlap(latent,native_support,observed) if overlap
                          else F.normalize(query,dim=-1)@self.native_extent_means.T)
            if not overlap:
                similarities=similarities.masked_fill(~self.native_extent_means.any(1)[None],-torch.inf)
            if self.identity_selection.get('restrict_concepts',False):
                eligible=self.proposal_groups[:,index].T
                similarities=similarities.masked_fill(~eligible,-torch.inf)
                accepted=accepted&eligible.any(1)
            else:accepted=torch.ones(len(query),dtype=torch.bool,device=query.device)
            score,index=similarities.max(1)
            accepted=accepted&torch.isfinite(score)
            if overlap:accepted=accepted&(score>self.identity_selection.get('minimum_support_iou',0.))
            if not overlap and self.identity_selection.get('require_native_support',True):
                if native_threshold is None:raise ValueError('native decision threshold required')
                accepted=accepted&(score>native_threshold)
            start=self.identity_selection.get('trusted_concept_start')
            if start is not None:
                index=torch.where(concept_accepted,concept_index+start,index)
                accepted=accepted|concept_accepted
        elif self.num_proposals:
            eligible=self.proposal_groups[:,index].T
            similarities=F.normalize(query,dim=-1)@F.normalize(self.proposal_embeddings,dim=-1).T
            if self.proposal_ranker:
                similarities=self.proposal_rank_weights[0]*similarities+self.proposal_rank_weights[1]*self.proposal_quality[index]
            index=similarities.masked_fill(~eligible,-torch.inf).argmax(1)
            accepted=accepted&eligible.any(1)
        selected=latent if rows is None else latent[rows]
        if self.hidden_dim:
            hidden=self.decoder[1](self.decoder[0](self.features(selected)))
            logits=F.linear(hidden,self.decoder[2].weight[index],self.decoder[2].bias[index])
        else:
            features=self.features(selected)
            probability=features@self.decoder.weight[index].T+self.decoder.bias[index]
            if self.residual is not None:
                hidden=self.residual[1](self.residual[0](features))
                probability=probability+F.linear(hidden,self.residual[2].weight[index],self.residual[2].bias[index])
            logits=torch.logit(probability.clamp(1e-6,1-1e-6))
        if self.identity_selection is not None and self.identity_selection.get('combine_candidates')=='greedy_union':
            fallback=~concept_accepted
            if fallback.any():
                with torch.no_grad():
                    memberships=[]
                    for z in latent.split(4096):
                        features=self.features(z);probability=self.decoder(features)
                        if self.hidden_dim:probability=probability.sigmoid()
                        if self.residual is not None:probability=probability+self.residual(features)
                        memberships.append(probability>.5)
                    membership=torch.cat(memberships)&observed[:,None]
                    union,quality,_=greedy_native_support_union(membership,native_support[:,fallback]&observed[:,None],preserve_precision=self.identity_selection.get('union_preserve_precision',False))
                    if rows is not None:union=union[rows]
                    logits[:,fallback]=torch.logit(union.float().clamp(1e-6,1-1e-6))
                    accepted[fallback]=quality>self.identity_selection.get('minimum_support_iou',0.)
        return logits,accepted
