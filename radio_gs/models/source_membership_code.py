"""Source semantic prototypes read compressed memberships from one element field.

The global decoder/prototypes are deployment state, not another per-element map.
Top-1 prototype selection is a positive-query development control; absence and
previously ungrounded concept handling are not established by this module.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SourceMembershipCode(nn.Module):
    def __init__(self, structural_dim, num_groups, query_dim):
        super().__init__()
        if min(structural_dim,num_groups,query_dim)<1:raise ValueError('nonempty membership code required')
        self.decoder=nn.Linear(structural_dim,num_groups,bias=False)
        self.register_buffer('text_prototypes',torch.zeros(num_groups,query_dim))
        self.register_buffer('image_prototypes',torch.zeros(num_groups,query_dim))

    def forward(self,query,modality,structure,observed,registration=None,rows=None):
        selected=structure if rows is None else structure[rows]
        if modality in ('text','image'):
            prototypes=getattr(self,modality+'_prototypes')
            groups=(F.normalize(query,dim=-1)@F.normalize(prototypes,dim=-1).T).argmax(1)
            probability=selected@self.decoder.weight[groups].T
        elif modality=='prompt':
            if registration is None:raise ValueError('registered signed prompt required')
            seeds=registration.seed_rows(query);result=[]
            for positive,negative in seeds:
                if positive==negative or not observed[positive] or not observed[negative]:
                    result.append(torch.zeros(len(selected),device=structure.device));continue
                evidence=self.decoder(structure[torch.stack([positive,negative])]).clamp(0,1)
                admitted=(evidence[0]>.5)&(evidence[1]<.5)
                if not admitted.any():result.append(torch.zeros(len(selected),device=structure.device));continue
                group=(evidence[0]-evidence[1]).masked_fill(~admitted,-torch.inf).argmax()
                result.append(selected@self.decoder.weight[group])
            probability=torch.stack(result,1)
        else:raise ValueError('unsupported query modality')
        return torch.logit(probability.clamp(1e-6,1-1e-6))
