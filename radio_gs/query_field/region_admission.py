"""Field-compatible conditional candidate admission, preserving v10 policy.

Inputs must have the original membership/visibility measure semantics. This
operator does not manufacture candidates, infer identity, or certify a mesh to
Gaussian feature transfer. It returns transient regional weights, not a second
persistent scene representation.
"""
import torch


@torch.no_grad()
def admit_regions(quality, prediction, *, seed, candidates, intersection, area, frames, view_count):
    if area.ndim != 1 or frames.shape != area.shape or frames.dtype != torch.long or candidates.dtype != torch.long or candidates.ndim != 1:
        raise ValueError('invalid regional coordinates')
    if not 0 <= seed < len(area) or view_count <= 0 or bool(((frames < 0) | (frames >= view_count)).any()):
        raise ValueError('invalid regional partition')
    if bool(((candidates < 0) | (candidates >= len(area)) | (candidates == seed)).any()) or len(candidates.unique()) != len(candidates):
        raise ValueError('invalid candidate addresses')
    if intersection.shape != candidates.shape or prediction.shape != (len(candidates), 3):
        raise ValueError('unaligned conditional candidates')
    if not bool(torch.isfinite(area).all() & torch.isfinite(intersection).all()) or bool((area < 0).any()) or bool(((intersection < 0) | (intersection > area[candidates]) | (intersection > area[seed])).any()):
        raise ValueError('invalid membership intersections')
    # Stable input order breaks merit ties, just as np.argmax in v10.
    additions = area[candidates] - intersection
    eligible = quality.eligible(prediction, additions, frames[candidates] == frames[seed])
    if not bool(eligible.any()):
        return None
    precision, recall = prediction[:, 0], prediction[:, 1]
    merit = precision * recall / (precision + recall - precision * recall).clamp_min(1e-8)
    weights = torch.zeros_like(area, dtype=torch.float32)
    confidence = area.new_zeros(view_count, dtype=torch.float32)
    weights[seed] = 1.; confidence[frames[seed]] = 1.
    chosen = []
    for frame in torch.unique(frames[candidates[eligible]], sorted=True):
        local = torch.where(eligible & (frames[candidates] == frame))[0]
        best = local[merit[local].argmax()]; region = candidates[best]
        if bool(frame == frames[seed]):
            weights[seed] = 0
        weights[region] = precision[best].float(); confidence[frame] = precision[best].float()
        chosen.append(region)
    return weights, confidence, torch.stack(chosen)
