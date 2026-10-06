"""Query-conditioned diffusion over independently validated scope hypotheses."""
import torch
from torch.nn import functional as F


class ScopeHypergraph:
 def __init__(self,payload,device='cuda',semantic_transport=False,fixed_cores=False,compatible_scopes=False):
  self.semantic_transport=semantic_transport
  self.fixed_cores=fixed_cores
  self.compatible_scopes=compatible_scopes
  indices=payload['indices'].to(device);weights=payload['weights'].to(device)
  row_mass=torch.zeros(payload['shape'][0],device=device).scatter_add_(0,indices[0],weights)
  self.membership=torch.sparse_coo_tensor(indices,weights/row_mass[indices[0]].clamp_min(1e-8),payload['shape']).coalesce()
  self.transpose=self.membership.transpose(0,1).coalesce();self.mass=torch.sparse.sum(self.membership,dim=0).to_dense().clamp_min(1e-8)
  self.prototypes=payload['prototypes'].to(device).float();self.quality=payload['quality'].to(device)

 @torch.no_grad()
 def refine(self,scores,query,model,modality):
  u=scores;support=model.observed_for(modality)[:,None]
  mass=(torch.sparse.mm(self.transpose,support.float())[:,0].clamp_min(1e-8) if self.compatible_scopes else self.mass)
  mu=torch.sparse.mm(self.transpose,u)/mass[:,None]
  variance=(torch.sparse.mm(self.transpose,u.square())/mass[:,None]-mu.square()).clamp_min(0)
  # Scope regularization is weak for mixed positive/negative hypotheses. Native
  # teacher agreement controls permission, not a replacement class decision.
  homogeneous=(1-4*variance).clamp(0,1).square()
  cosine=F.normalize(self.prototypes,dim=1)@F.normalize(query.float(),dim=1).T
  if modality in model.native_query_thresholds:
   threshold=model.native_query_thresholds[modality](query)
   scale=model.readout_log_scale[modality].clamp(-4,6).exp()
   teacher=torch.sigmoid(scale*(cosine-threshold))
  else:teacher=model.cosine_logits(cosine,modality).sigmoid()
  agreement=(1-(teacher-mu).abs()).square()
  permission=self.quality[:,None]*homogeneous*agreement
  if self.compatible_scopes:
   positive=torch.sparse.mm(self.transpose,((u>.5)&support).float())>0
   negative=torch.sparse.mm(self.transpose,((u<.5)&support).float())>0
   positive_core=torch.sparse.mm(self.transpose,((u>=.75)&support).float())>0
   negative_core=torch.sparse.mm(self.transpose,((u<=.25)&support).float())>0
   # A small positive part is not background merely because its enclosing
   # scope has a large negative majority. Use another compatible hypothesis.
   permission=permission.masked_fill((positive&negative_core)|(negative&positive_core),0)
  residual=None
  if self.semantic_transport:
   native=model.native_cosine(query,modality=modality)
   compact=torch.sparse.mm(self.transpose,native*support if self.compatible_scopes else native)/mass[:,None]
   scale=model.readout_log_scale[modality].clamp(-4,6).exp()
   residual=(scale*(cosine-compact)).clamp(-1,1)
  degree=torch.sparse.mm(self.membership,permission)
  fidelity=1+8*(2*u-1).square()
  value=u.clone()
  fixed=(u<=.25)|(u>=.75)
  for _ in range(24):
   center=torch.sparse.mm(self.transpose,value*support if self.compatible_scopes else value)/mass[:,None]
   if residual is not None:center=torch.sigmoid(torch.logit(center.clamp(1e-6,1-1e-6))+residual)
   value=(fidelity*u+torch.sparse.mm(self.membership,permission*center))/(fidelity+degree)
   if self.fixed_cores:value=torch.where(fixed,u,value)
  return torch.where(degree>0,value,u)


def attach_scope_hypergraph(model,payload,query_task=None):
 if payload.get('scope_native_dictionary',False):
  from radio_gs.models.native_scope_dictionary import NativeScopeDictionary,attach_scope_text_readout
  attach_scope_text_readout(model,NativeScopeDictionary(payload))
 if payload.get('scope_readout')=='composition':
  from radio_gs.querying.scope_composition import ScopeComposition
  graph=ScopeComposition(payload['scope_hypergraph'],fixed_cores=payload.get('composition_fixed_cores',False))
 else:
  graph=ScopeHypergraph(payload['scope_hypergraph'],semantic_transport=payload.get('scope_semantic_transport',False),fixed_cores=payload.get('scope_fixed_cores',False),compatible_scopes=payload.get('scope_compatible',False))
 original=model.logits
 modalities=payload.get('scope_modalities',('text','image'))
 if payload.get('scope_task_policy')=='typed':
  # A category-union image query and a reference-grounded prompt do not
  # assert the same extent. Prompt inference subsequently supplies its
  # authorized spatial constraints; a standalone image crop does not.
  modalities=('text','image') if query_task=='reference_prompt' else ('text',)
 def logits(query,modality,rows=None):
  if modality not in modalities:return original(query,modality,rows)
  base=original(query,modality);probability=base.sigmoid()*model.observed_for(modality)[:,None]
  refined=graph.refine(probability,query,model,modality)
  # Add an odds correction rather than replacing the calibrated base logits.
  # An unchanged scope response must leave even saturated native logits intact.
  delta=torch.logit(refined.clamp(1e-30,1-1e-7))-torch.logit(probability.clamp(1e-30,1-1e-7))
  result=base+delta
  return result if rows is None else result[rows]
 model.logits=logits
 model.scope_graph=graph
 return model
