"""Separate a positive footprint from evidence for a unique category identity.

Only for complete mutually exclusive category menus. A shared positive region
can establish scope without resolving its label. Gates precede query batching.
They are ambiguity heuristics, not calibrated probabilities of correctness.
"""
import torch
from pathlib import Path
from radio_gs.querying.region_positive_grounding import positive_signal


@torch.inference_mode()
def source_category_ambiguity(scene, texts, device):
    result = {}
    overlapping = positive = 0
    total_gate = 0.
    for view in scene['source_evidence']['views']:
        data = torch.load(view['identity'], map_location='cpu')
        signal = positive_signal(data, texts, device)
        count = (signal > 0).sum(1)
        distribution = signal / signal.sum(1, keepdim=True).clamp_min(1e-12)
        if scene['source_category_ambiguity'].startswith('union_'):
            identity = torch.load(Path(scene['source_union_identity']) / f"frame_{view['frame']:05d}.pt", map_location='cpu')
            ids = [identity['queries'].index(t) for t in texts]
            probability = identity['class_probability'][ids][:, ids]
            ratio = (probability.diag() / probability.amax(1).clamp_min(1e-12)) * identity['valid'][ids]
            result[view['frame']] = dict(conflict=(count > 1).cpu(), ratio=dict(zip(texts, ratio.tolist())))
            overlapping += int((count > 1).sum())
            positive += int((count > 0).sum())
            continue
        if scene['source_category_ambiguity'] == 'entropy':
            entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(1)
            gate = torch.where(count > 1, 1-entropy/count.float().clamp_min(2).log(), torch.ones_like(entropy)).clamp(0, 1)
        else:
            top = signal.topk(2, dim=1).values
            gate = torch.where(count > 1, (top[:,0]-top[:,1])/top[:,0].clamp_min(1e-12), torch.ones_like(top[:,0]))
        result[view['frame']] = gate.cpu()
        overlapping += int((count > 1).sum())
        positive += int((count > 0).sum())
        total_gate += float(gate[count > 1].sum())
    return result, dict(mode=scene['source_category_ambiguity'], vocabulary_size=len(texts),
        positive_pixels=positive, conflicting_category_pixels=overlapping,
        mean_conflict_gate=total_gate/max(overlapping,1) if not scene['source_category_ambiguity'].startswith('union_') else None,
        independent_native_union_identity=scene['source_category_ambiguity'].startswith('union_'),
        competition_before_query_batching=True,
        scope_without_unique_identity_is_uncertain=True, calibrated_correctness=False)
