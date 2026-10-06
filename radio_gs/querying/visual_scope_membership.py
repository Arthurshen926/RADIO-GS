"""Versioned visual region membership and conditional geometric transport.

Identity consumes only frozen image/query evidence. Source labels are absent
from this runtime. M is the Stage1 corrected duplicate/view-normalized support.
"""
import torch
from torch import nn
from torch.nn import functional as F


class VisualMembershipHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(8))
        self.register_buffer('scale', torch.ones(8))
        self.net = nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net((features-self.mean)/self.scale).squeeze(-1)


class VisualMembershipRuntime:
    def __init__(self, bank, head, arm='B2', device='cpu'):
        self.head, self.arm = head.to(device), arm
        self.m = torch.sparse_coo_tensor(bank['indices'].long().to(device),bank['weights'].float().to(device),bank['shape']).coalesce()
        self.mt = self.m.T.coalesce()
        # Accumulate fixed sparse transport in float64, round once to float32.
        # CUDA float32 sparse atomics drift even on repeated identical queries.
        self.forward_op = self.m.double().to_sparse_csr()
        self.transpose_op = self.mt.double().to_sparse_csr()
        self.mass = torch.sparse.mm(self.transpose_op,torch.ones(self.m.shape[0],1,device=device,dtype=torch.float64)).flatten().clamp_min(1e-8)
        self.views = bank['views'].to(device)
        self.allowed = ~torch.isin(self.views,torch.tensor(bank['held_views'],device=device))
        self.native = {k:F.normalize(bank[k].float().to(device),dim=-1) for k in ['C','F','B','tight']}
        self.disjoint = bank['disjoint'].to(device).bool() & (self.views[:,None]==self.views[None,:])
        self.quality, self.area = bank['quality'].float().to(device),bank['area'].float().to(device)

    def features(self, q):
        q=F.normalize(q.float(),dim=-1)
        c,t = self.native['C']@q.T,self.native['tight']@q.T
        f,b = self.native['F']@q.T,self.native['B']@q.T
        zero=torch.zeros_like(c)
        return torch.stack([c,t,self.quality[:,None].expand_as(c),self.area[:,None].expand_as(c),
                            f if self.arm=='B2' else zero,b if self.arm=='B2' else zero,
                            f-b if self.arm=='B2' else zero,c-b if self.arm=='B2' else zero],-1)

    def membership_logits(self,q):
        return self.head(self.features(q))

    def transport(self, membership_logits, parent, exclude_view=None):
        allowed=self.allowed if exclude_view is None else self.allowed & (self.views!=exclude_view)
        r=membership_logits.sigmoid()
        positive=(r>=.8)&allowed[:,None]
        competitor=(self.disjoint.float()@positive.float())>0
        negative=(r<=.2)&competitor&allowed[:,None]
        prior=torch.sparse.mm(self.transpose_op,parent.sigmoid().double())/self.mass[:,None]
        difference=(membership_logits-torch.logit(prior.float().clamp(1e-6,1-1e-6))).clamp(-2,2)
        coefficient=torch.where(positive,difference.clamp_min(0),torch.where(negative,difference.clamp_max(0),0))*2*(r-.5).abs()
        return torch.sparse.mm(self.forward_op,coefficient.double()).float(),dict(probability=r,positive=positive,negative=negative,coefficient=coefficient,allowed=allowed)

    def delta(self,q,parent,exclude_view=None):
        return self.transport(self.membership_logits(q),parent,exclude_view)[0]

    def logits(self,q,parent,modality='text',exclude_view=None):
        if modality!='text':return parent
        return parent+self.delta(q,parent,exclude_view)


def load_runtime(addon,device):
    from radio_gs.querying.query_region_evidence import load
    head=VisualMembershipHead();saved=load(addon['checkpoint'])
    assert saved['schema']=='visual_membership.next42.v2'
    head.load_state_dict(saved['state_dict'])
    return VisualMembershipRuntime(load(addon['bank']),head,addon['arm'],device)


def balanced_query_risks(error,target,valid,weight=None):
    """Return one risk per query-view; absent supervision is NaN, never zero.

    Soft region targets use y>=.5 as the positive stratum. Each present stratum
    is weighted equally within a query, then supervised queries are averaged.
    """
    weight=torch.ones_like(error) if weight is None else weight
    risks=[];exists=[]
    for positive in [True,False]:
        domain=valid & ((target>=.5)==positive)
        w=domain*weight;mass=w.sum(0)
        risks.append((error*w).sum(0)/mass.clamp_min(1e-12));exists.append(mass>0)
    count=exists[0].long()+exists[1].long()
    value=(risks[0]+risks[1])/count.clamp_min(1)
    return torch.where(count>0,value,torch.full_like(value,float('nan')))


def supervised_mean(risks):
    valid=torch.isfinite(risks)
    return risks[valid].mean() if valid.any() else None


def membership_loss(logits,supervision,brier=False):
    target=supervision['target']
    error=(logits.sigmoid()-target).square() if brier else F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    return supervised_mean(balanced_query_risks(error,target,supervision['valid'],supervision['known_fraction']))


def known_loss(pred,episode):
    return supervised_mean(balanced_query_risks((pred-episode['target']).square(),episode['target'],episode['known']))


def region_supervision(projected,episode,allowed):
    # Peak normalization gives pixel-equivalent area without undoing shape.
    a=projected/projected.amax(0).clamp_min(1e-12)
    a=a*episode['visible'][:,None]
    p=episode['known']&episode['target'].bool()
    n=episode['known']&~episode['target'].bool()
    pm,nm=a.T@p.float(),a.T@n.float()
    total=a.sum(0)[:,None].expand_as(pm)
    um=(total-pm-nm).clamp_min(0)
    fraction=(pm+nm)/total.clamp_min(1e-12)
    target=pm/(pm+nm).clamp_min(1e-12)
    valid=(total>=20)&(fraction>=.5)&allowed[:,None]
    return dict(target=target,valid=valid,known_fraction=fraction,effective_pixels=total,p_mass=pm,n_mass=nm,u_mass=um)


def render_probabilities(op,element_logits):
    return torch.sparse.mm(op,element_logits.sigmoid())
