"""Opt-in next42 source-native evidence; no benchmark vocabulary or query state.

The bank is extra deployment state, not a compact-field replacement. A signed
carrier requires source-native positive identity or a co-view disjoint competitor.
Source text anchors are evidence vectors, never runtime phrase lookup keys.
"""
from pathlib import Path
import hashlib
import json
import re
import torch
from torch import nn
from torch.nn import functional as F


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def phrase_family(text):
    return ' '.join(w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in {'a','an','the'})


def held_phrase(text):
    return int(hashlib.sha256(phrase_family(text).encode()).hexdigest()[:8], 16) % 5 == 0


def same_window(rgb, mask):
    """Full / foreground / complement: identical 1.5x window and 384 resize."""
    y, x = torch.where(mask)
    if not len(x):
        raise ValueError('empty source mask')
    h, w = mask.shape
    cx, cy = (int(x.min()) + int(x.max()) + 1) / 2, (int(y.min()) + int(y.max()) + 1) / 2
    bw, bh = (int(x.max())-int(x.min())+1)*1.5, (int(y.max())-int(y.min())+1)*1.5
    import math
    box = [max(0, math.floor(cx-bw/2)), max(0, math.floor(cy-bh/2)),
           min(w, math.ceil(cx+bw/2)), min(h, math.ceil(cy+bh/2))]
    x0,y0,x1,y1 = box
    crop, m = rgb[:,y0:y1,x0:x1], mask[y0:y1,x0:x1][None]
    triple = torch.stack([crop, torch.where(m,crop,.5), torch.where(m,.5,crop)])
    return F.interpolate(triple, (384,384), mode='bicubic', align_corners=False, antialias=True).clamp(0,1), box


def normalize_membership(indices, values, views, n):
    """Use each view's strongest support, then average over ALL visible views.

    Repeated proposals cannot amplify confidence. Uncovered views remain in the
    supplied visible-mass denominator during bank construction.
    """
    indices, values = indices.long(), values.float()
    pair = indices[0] * (int(views.max())+1) + views[indices[1]]
    _, inv = torch.unique(pair, return_inverse=True)
    mass = torch.zeros(int(inv.max())+1, device=values.device).scatter_add_(0,inv,values)
    maximum = torch.zeros_like(mass).scatter_reduce_(0,inv,values,reduce='amax',include_self=True)
    return values / mass[inv].clamp_min(1e-12) * maximum[inv]


class RelationHead(nn.Module):
    """Same 12 -> 16 -> 1 architecture in both arms; zero initial correction."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12,16), nn.Tanh(), nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return 2 * torch.tanh(self.net(features).squeeze(-1))


class RegionEvidence:
    def __init__(self, bank, head, arm='B2', device='cpu'):
        self.bank, self.head, self.arm = bank, head.to(device), arm
        self.device = torch.device(device)
        self.m = torch.sparse_coo_tensor(bank['indices'].long().to(device),bank['weights'].float().to(device),bank['shape']).coalesce()
        self.mt = self.m.T.coalesce()
        self.mass = torch.sparse.sum(self.m,dim=0).to_dense().clamp_min(1e-8)
        self.views = bank['views'].to(device)
        self.allowed = ~torch.isin(self.views,torch.tensor(bank['held_views'],device=device))
        self.native = {k:F.normalize(bank[k].float().to(device),dim=-1) for k in ['C','F','B','tight','text']}
        self.disjoint = bank['disjoint'].to(device).bool()
        self.quality = bank['quality'].float().to(device)
        self.area = bank['area'].float().to(device)

    def carriers(self, q, exclude_view=None):
        # Every query is computed separately: batch composition cannot change it.
        raw = self.native['tight'] @ q.T
        text = self.native['text'] @ q.T
        allowed = self.allowed.clone()
        if exclude_view is not None:
            allowed &= self.views != exclude_view
        positive = (text >= .75) & (raw >= .12) & allowed[:,None]
        # Disjoint AND independent positive competitor AND lower native response.
        relation = self.disjoint[:,:,None] & positive[None,:,:]
        relation &= raw[:,None,:] < raw[None,:,:] - .03
        negative = relation.any(1) & ~positive & allowed[:,None]
        return positive, negative, raw, text, allowed

    def features(self, q, parent, exclude_view=None):
        pos,neg,tight,text,allowed = self.carriers(q,exclude_view)
        c = self.native['C'] @ q.T
        f = self.native['F'] @ q.T
        b = self.native['B'] @ q.T
        region_parent = torch.sparse.mm(self.mt,parent.sigmoid())/self.mass[:,None]
        # Independent-view consistency uses image evidence, not source labels.
        independent = (self.views[:,None] != self.views[None,:]) & allowed[None,:]
        similarity = (self.native['tight'] @ self.native['tight'].T).clamp_min(0) * independent
        similarity = similarity / similarity.sum(1,keepdim=True).clamp_min(1e-8)
        mean = similarity @ tight
        dispersion = (similarity @ tight.square()-mean.square()).clamp_min(0).sqrt()
        zero = torch.zeros_like(c)
        components = [c,tight,text,region_parent,mean,dispersion,
                      self.quality[:,None].expand_as(c),self.area[:,None].expand_as(c),
                      f if self.arm=='B2' else zero,b if self.arm=='B2' else zero,
                      f-b if self.arm=='B2' else zero,c-b if self.arm=='B2' else zero]
        return torch.stack(components,-1),pos,neg

    def delta(self, q, parent, exclude_view=None):
        q = F.normalize(q.float(),dim=-1)
        features,pos,neg = self.features(q,parent,exclude_view)
        raw = self.head(features)
        # A zero-centred signed head can increase P and decrease qualified N.
        # Other regions receive exactly zero, regardless of the learned bias.
        coefficient = torch.where(pos,raw.clamp_min(0),torch.where(neg,raw.clamp_max(0),0))
        return torch.sparse.mm(self.m,coefficient)

    def logits(self, q, parent, modality='text', exclude_view=None):
        if modality != 'text':
            return parent
        return parent + self.delta(q,parent,exclude_view)


def load_runtime(addon, device):
    if addon.get('kind') == 'element_posterior':
        from radio_gs.querying.element_posterior_transport import load_runtime as element_runtime
        return element_runtime(addon, device)
    if addon.get('kind') == 'visual_membership':
        from radio_gs.querying.visual_scope_membership import load_runtime as visual_runtime
        return visual_runtime(addon, device)
    bank = load(addon['bank'])
    head = RelationHead()
    head.load_state_dict(load(addon['checkpoint'])['state_dict'])
    return RegionEvidence(bank,head,addon['arm'],device)


def corrected_scores(runtime, q, parent, categorical=False, base_logits=None):
    """Apply once AFTER the final next38 calibration/category composition.

    Exact zero fallback avoids sigmoid(logit(p)) roundoff and saturated clipping.
    """
    if categorical:
        return runtime.logits(q,parent)
    from radio_gs.querying.element_posterior_transport import ElementPosteriorRuntime
    if isinstance(runtime, ElementPosteriorRuntime):
        base = torch.logit(parent) if base_logits is None else base_logits
        updated = runtime.logits(q, base)
        return torch.where(updated == base, parent, updated.sigmoid())
    base = torch.logit(parent) if base_logits is None else base_logits
    # Features consume bounded parent probability, not infinite logits directly.
    delta = runtime.delta(q,base)
    return torch.where(delta == 0,parent,(base+delta).sigmoid())


def parent_model(key, device='cuda'):
    """Reproduce evaluate_scope_dictionary --pipeline --calibrated, exactly."""
    from radio_gs.models.element_retrieval_field import ElementRetrievalField
    from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
    from radio_gs.querying.source_accepted_category import source_accepted_category_logits
    from radio_gs.querying.category_competition import correct_category_logits
    directory=Path('output/taskwise_scope_composition_next38')/key
    payload=load(directory/'field.pt');parent=load(payload['base_field'])
    model=ElementRetrievalField(**parent['architecture']).to(device)
    model.load_state_dict(parent['state_dict']);model.requires_grad_(False).eval()
    if 'scope_state' in payload:model.grounded_transfer.load_state_dict(payload['scope_state'])
    attach_scope_hypergraph(model,payload)
    cal=load(directory/'calibration.pt');original=model.native_query_thresholds['text'].forward
    def threshold(q):
        delta=F.normalize(q,dim=-1)@cal['weight'].to(device)+cal['bias'].to(device)
        return original(q)+delta.clamp(-cal['max_delta'],cal['max_delta'])
    model.native_query_thresholds['text'].forward=threshold
    def logits(q):
        if not key.startswith('scannet_'):return model.logits(q,'text')
        root=Path(payload['root'])
        native=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))
        value=source_accepted_category_logits(native,lambda:model.logits(q,'text'),scope_receipt=json.loads((root/'bounded_fallback_withheld/report.json').read_text()),admission_receipt=json.loads((Path(payload['base_field']).parent/'report.json').read_text()),max_correction=1.)[0]
        if 'category_competition' in payload:value=correct_category_logits(value,q,payload['category_competition'])
        return value
    return model,logits,payload
