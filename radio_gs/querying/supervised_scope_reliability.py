"""Scope reliability from geometry statistics and optional native identity.

The shared head has no scene or class-ID parameters. Optional identity inputs
are frozen region/query compatibility, not validation labels. Quality-only heads
estimate mask precision/recall; retrieval-trained heads output evidence weights.
"""
import math
import torch

FEATURE_NAMES=['detector','native_identity','log_instances','log_area','boundary_fraction',
    'mean_mask_inside','prior_inside','prior_outside','prior_mass_covered','prior_peak_inside']
NATIVE_FEATURE_NAMES=['region_query_probability','region_competitor_probability',
    'region_semantic_entropy','region_query_log_margin']

def native_identity_features(record,ids,device):
    """Full declared category menu is retained even for batched query execution."""
    p=record['class_probability'].float().to(device)[ids]
    own=p[torch.arange(len(ids),device=device),torch.tensor(ids,device=device)]
    other=p.clone();other[torch.arange(len(ids),device=device),torch.tensor(ids,device=device)]=0
    peak=other.amax(1)
    entropy=-(p*p.clamp_min(1e-8).log()).sum(1)/math.log(p.shape[1])
    margin=(own.clamp_min(1e-8).log()-peak.clamp_min(1e-8).log())/10
    return torch.stack([own,peak,entropy,margin],1)

def scope_features(evidence,probability,detector,confidence,instance_count,projected_prior):
    visible=evidence.visible.float();binary=(probability>.5).float()*visible
    area=binary.sum(0);outside=(1-binary)*visible;outside_area=outside.sum(0)
    mean=lambda x,mass,den:(x*mass).sum(0)/den.clamp_min(1)
    covered=(projected_prior*binary).sum(0)/((projected_prior*visible).sum(0).clamp_min(1e-8))
    known=(evidence.positive|evidence.negative).float()
    return torch.stack([detector,confidence/detector.clamp_min(1e-8),instance_count.log1p(),
        (area/visible.sum(0).clamp_min(1)).clamp_min(1e-6).log(),
        ((visible-known).sum(0)/visible.sum(0).clamp_min(1)),
        mean(probability,binary,area),mean(projected_prior,binary,area),
        mean(projected_prior,outside,outside_area),covered,(projected_prior*binary).amax(0)],dim=1)

class ScopeReliability:
    def __init__(self,path,device):
        self.state=torch.load(path,map_location=device,weights_only=False)
        self.network=None
        if self.state.get('hidden_dimension',0):
            self.network=make_reliability_head(len(self.state['mean']),self.state['hidden_dimension']).to(device).eval()
            self.network.load_state_dict(self.state['network'])
    def __call__(self,features):
        s=self.state
        x=(features-s['mean'])/s['std']
        return (self.network(x) if self.network is not None else x@s['weight'].T+s['bias']).sigmoid()

def make_reliability_head(dimension,hidden=0):
    if hidden:
        return torch.nn.Sequential(torch.nn.Linear(dimension,hidden),torch.nn.SiLU(),torch.nn.Linear(hidden,2))
    return torch.nn.Linear(dimension,2)
