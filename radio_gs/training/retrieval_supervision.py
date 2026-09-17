"""Known-support retrieval losses and optional native teacher heads."""
import torch
from torch import nn
from torch.nn import functional as F


def teacher_element_features(latent, observed, unit_norm=False):
    """Use the query's element directions when constructing teacher renders."""
    if latent.ndim != 2 or observed.ndim != 1 or len(latent) != len(observed):
        raise ValueError('teacher feature axes differ')
    features=F.normalize(latent,dim=-1) if unit_norm else latent
    return features*observed[:,None]

def membership_loss(probability, target, known):
    if probability.shape!=target.shape or known.shape!=target.shape:
        raise ValueError("membership axes differ")
    if not known.any():return probability.sum()*0
    p=probability[known];y=target[known]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError("known membership labels must be finite probabilities")
    bce=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y)
    dice=1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    return bce+dice

def boundary_loss(probability,target,known,height,width):
    p=probability.reshape(height,width);y=target.reshape(height,width);k=known.reshape(height,width)
    terms=[]
    for x,z,valid in [(p[1:]-p[:-1],y[1:]-y[:-1],k[1:]&k[:-1]),
                      (p[:,1:]-p[:,:-1],y[:,1:]-y[:,:-1],k[:,1:]&k[:,:-1])]:
        if valid.any():terms.append(F.smooth_l1_loss(x[valid],z[valid]))
    return sum(terms)/len(terms) if terms else probability.sum()*0

class OptionalTeacherHeads(nn.Module):
    """Teacher-specific training projections; never part of query inference."""
    def __init__(self,latent_dim,dimensions,weights):
        super().__init__()
        if not set(weights)<={"radio","dino","sam","siglip"} or any(not torch.isfinite(torch.tensor(v)) or v<0 for v in weights.values()):
            raise ValueError("invalid teacher weights")
        self.weights=dict(weights)
        self.heads=nn.ModuleDict({name:nn.Linear(latent_dim,dim) for name,dim in dimensions.items()
                                 if self.weights.get(name,0)>0})
    def feature_loss(self,name,latent,target):
        if self.weights.get(name,0)==0:return latent.sum()*0
        if name not in self.heads or target is None:
            raise ValueError("enabled teacher requires a matching native target")
        if len(latent)==0:return latent.sum()*0
        if not torch.isfinite(target).all():raise ValueError("teacher target contains invalid values")
        return self.weights[name]*(1-F.cosine_similarity(self.heads[name](latent),target,dim=-1)).mean()


def relation_loss(latent, teacher):
    """Match native pairwise cosine relations without equating feature spaces."""
    if latent.ndim != 2 or teacher.ndim != 2 or len(latent) != len(teacher):
        raise ValueError('relation sample axes differ')
    if len(latent) < 2:
        return latent.sum() * 0
    z = F.normalize(latent, dim=-1)
    t = F.normalize(teacher.detach(), dim=-1)
    pairs = torch.triu(torch.ones(len(z), len(z), device=z.device, dtype=torch.bool), diagonal=1)
    return F.smooth_l1_loss((z @ z.T)[pairs], (t @ t.T)[pairs])


def query_neighborhood_loss(projected, native, temperature=.1):
    """Preserve relative query neighbours; exclude trivial self-similarity."""
    if projected.ndim != 2 or native.ndim != 2 or len(projected) != len(native):
        raise ValueError('query relation axes differ')
    if temperature <= 0 or not torch.isfinite(projected).all() or not torch.isfinite(native).all():
        raise ValueError('invalid query relation evidence')
    if len(projected) < 3:
        return projected.sum() * 0
    z, t = F.normalize(projected, dim=-1), F.normalize(native.detach(), dim=-1)
    keep = ~torch.eye(len(z), device=z.device, dtype=torch.bool)
    student = (z @ z.T)[keep].reshape(len(z), -1) / temperature
    teacher = (t @ t.T)[keep].reshape(len(z), -1) / temperature
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction='batchmean')


def episode_order(count, epoch, seed, shuffle=False):
    """Epoch-local ordering without changing model or auxiliary-sampling RNG."""
    if count < 1 or epoch < 0:
        raise ValueError('invalid episode schedule')
    if not shuffle:
        return list(range(count))
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(count, generator=generator).tolist()


def sam_contour_loss(latent, mask, valid, height, width):
    """Structural pseudo supervision, not an outside-mask membership label.

    Same-region neighbours preserve continuity; mask contour neighbours preserve
    a feature discontinuity. Both endpoints must have geometric evidence.
    """
    z = F.normalize(latent, dim=-1).reshape(height, width, -1)
    m = mask.reshape(height, width).bool()
    v = valid.reshape(height, width).bool()
    inside_terms, edge_terms = [], []
    for x, y, mx, my, known in (
        (z[1:], z[:-1], m[1:], m[:-1], v[1:] & v[:-1]),
        (z[:, 1:], z[:, :-1], m[:, 1:], m[:, :-1], v[:, 1:] & v[:, :-1]),
    ):
        distance = 1 - (x * y).sum(-1)
        inside, edge = known & mx & my, known & (mx ^ my)
        if inside.any():
            inside_terms.append(distance[inside].square().mean())
        if edge.any():
            edge_terms.append(F.relu(.25 - distance[edge]).square().mean())
    terms = inside_terms + edge_terms
    return sum(terms) / len(terms) if terms else latent.sum() * 0


def episode_membership_targets(episode, masks, region_probability, observed, support):
    """Text union and explicit negatives; conflicts and unknowns stay unlabelled."""
    positive = episode.get('positive_regions', [episode['target_region']])
    negative = episode['negative_regions']
    pm = masks[positive].any(0)
    nm = masks[negative].any(0) if negative else torch.zeros_like(pm)
    known = (pm ^ nm) & (support > .5)
    pr = region_probability[positive].amax(0)
    nr = region_probability[negative].amax(0) if negative else torch.zeros_like(pr)
    row_known = ((pr > .7) ^ (nr > .7)) & observed
    return pm.float(), known, (pr > .7).float(), row_known


def soft_query_relation_loss(student_scores, teacher_scores, temperature=.07):
    """Distil relative query-to-region scores, without inventing hard labels.

    Each row is one query and columns are co-observed source regions. Teacher
    uncertainty is retained in the distribution; it is not a binary membership
    probability and supplies no unknown-background negatives.
    """
    if student_scores.ndim != 2 or student_scores.shape != teacher_scores.shape:
        raise ValueError('query/region score axes differ')
    if not temperature > 0:
        raise ValueError('positive temperature required')
    if student_scores.shape[1] < 2 or student_scores.shape[0] == 0:
        return student_scores.sum() * 0
    if not torch.isfinite(student_scores).all() or not torch.isfinite(teacher_scores).all():
        raise ValueError('query relations must be finite')
    return F.kl_div(F.log_softmax(student_scores / temperature, dim=-1),
                    F.softmax(teacher_scores.detach() / temperature, dim=-1),
                    reduction='batchmean')


def source_region_schedule(proposal_views, train_views):
    """Deterministic view-round-robin coverage independent of matched episodes."""
    views = proposal_views.tolist()
    groups = [[i for i, view in enumerate(views) if view == v] for v in train_views]
    if not groups or any(not g for g in groups):
        raise ValueError('every training view requires source-region evidence')
    return [group[index] for index in range(max(map(len, groups)))
            for group in groups if index < len(group)]


def validate_episode_split(episodes, proposal_views, frames, train_frames, validation_frames):
    """Reject query-side as well as target-side leakage before optimization."""
    train, validation = set(train_frames), set(validation_frames)
    if train & validation or not train or not validation:
        raise ValueError('training and validation frames must be disjoint and nonempty')
    if not (train | validation) <= set(frames):
        raise ValueError('split references an absent source frame')
    for episode in episodes:
        split = episode['split']
        if split not in ('train', 'validation'):
            raise ValueError('unknown episode split')
        allowed = train if split == 'train' else validation
        query_region, target_region = episode['query_region'], episode['target_region']
        regions = [query_region, target_region] + episode.get('positive_regions', []) + episode['negative_regions']
        if any(region < 0 or region >= len(proposal_views) for region in regions):
            raise ValueError('episode region outside source evidence')
        query_view, target_view = int(proposal_views[query_region]), episode['target_view']
        if query_view < 0 or target_view < 0 or query_view >= len(frames) or target_view >= len(frames):
            raise ValueError('episode view outside source authority')
        if frames[query_view] not in allowed or frames[target_view] not in allowed:
            raise ValueError('query or target view leaks across split')
        for region in [target_region] + episode.get('positive_regions', []) + episode['negative_regions']:
            if int(proposal_views[region]) != target_view:
                raise ValueError('target labels belong to a different view')


def balanced_episode_coverage(counts, steps):
    """Coverage of the deterministic modality-round-robin training schedule."""
    if not counts or any(count <= 0 for count in counts.values()) or steps < 0:
        raise ValueError('nonempty episode counts and nonnegative budget required')
    modalities=list(counts);cycle=len(modalities)
    visited={m:min(counts[m],max(0,(steps+cycle-1-i)//cycle)) for i,m in enumerate(modalities)}
    minimum=max((count-1)*cycle+i+1 for i,count in enumerate(counts.values()))
    updates={m:max(0,(steps+cycle-1-i)//cycle) for i,m in enumerate(modalities)}
    repetitions={m:{'minimum':updates[m]//counts[m],
                    'maximum':(updates[m]+counts[m]-1)//counts[m],
                    'mean':updates[m]/counts[m]} for m in modalities}
    return {'available':dict(counts),'visited':visited,'minimum_steps_for_full_coverage':minimum,
            'updates':updates,'episode_repetitions':repetitions,
            'complete':all(visited[m]==counts[m] for m in counts)}


def symmetric_query_region_relation_loss(student_scores, teacher_scores, temperature=.07):
    """Preserve both query ranking per region and region ranking per query.

    These are soft native similarity relations, never binary membership labels.
    Both inputs must use the same training-only query/region axes.
    """
    return .5 * (soft_query_relation_loss(student_scores, teacher_scores, temperature)
                 + soft_query_relation_loss(student_scores.T, teacher_scores.T, temperature))
