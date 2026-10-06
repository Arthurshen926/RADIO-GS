"""Source-supervised scope ranking with unchanged native query coordinates."""
import torch
from torch import nn
from torch.nn import functional as F


class ScopeQuality(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
  self.net=nn.Sequential(nn.Linear(len(mean),32),nn.SiLU(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  self.scale=nn.Parameter(torch.tensor(20.))
 def forward(self,x):return self.scale.clamp(1,40)*(x[...,0]-x[...,0].max(-1,keepdim=True).values)+self.net((x-self.mean)/self.std).squeeze(-1)


@torch.no_grad()
def scope_features(model,graph,query,modality):
 indices=graph['indices'].to(query.device);weights=graph['weights'].to(query.device);support=model.observed_for(modality)
 matrix=torch.sparse_coo_tensor(indices,weights*support[indices[0]],graph['shape']).coalesce();transpose=matrix.transpose(0,1).coalesce();mass=torch.sparse.sum(matrix,dim=0).to_dense().clamp_min(1e-8)
 response=model.retrieve(query,modality)['scores'];native=model.native_cosine(query,modality=modality)
 pooled=torch.sparse.mm(transpose,response)/mass[:,None];compact=torch.sparse.mm(transpose,native*support[:,None])/mass[:,None]
 positive=(response>.5).float();precision=torch.sparse.mm(transpose,positive)/mass[:,None];recall=torch.sparse.mm(transpose,positive)/positive.sum(0).clamp_min(1)[None]
 cosine=F.normalize(graph['prototypes'].float().to(query.device),dim=1)@F.normalize(query.float(),dim=1).T
 quality=graph['quality'].to(query.device)[:,None].expand_as(cosine);area=(mass/len(support)).clamp_min(1e-8).log()[:,None].expand_as(cosine)
 return torch.stack([cosine,compact,pooled,precision,recall,quality,area,cosine-compact],-1).transpose(0,1)
