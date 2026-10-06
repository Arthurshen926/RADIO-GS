"""Separate detected category footprints from their native semantic identity.

One-hot emitting-query labels are the matched control. The candidate uses the
complete native distribution of each category union. Neither treats uncovered
pixels as negatives. This categorical decoder cannot serve overlapping queries.
"""
from pathlib import Path
import torch
from radio_gs.querying.region_positive_grounding import positive_signal
from radio_gs.querying.typed_evidence import visible_operator


@torch.inference_mode()
def fuse_union_category_observations(prior_logits, observed, scene, texts):
    device = prior_logits.device
    numerator = torch.zeros_like(prior_logits)
    mass = torch.zeros_like(prior_logits[:, :1])
    visibility = torch.zeros_like(mass)
    covered = 0
    for view in scene['source_evidence']['views']:
        data = torch.load(view['identity'], map_location='cpu')
        signal = positive_signal(data, texts, device)
        mode = scene['source_category_fusion']
        if mode != 'union_emitting_categories':
            identity = torch.load(Path(scene['source_union_identity']) / f"frame_{view['frame']:05d}.pt", map_location='cpu')
            ids = [identity['queries'].index(t) for t in texts]
        if mode in ('union_native_categories', 'union_native_hard_categories'):
            category = identity['class_probability'][ids][:, ids].to(device)
            category = category / category.sum(1, keepdim=True).clamp_min(1e-12)
            if scene['source_category_fusion'] == 'union_native_hard_categories':
                category = torch.nn.functional.one_hot(category.argmax(1), len(texts)).float()
            strength = signal @ category
        elif mode == 'union_verified_categories':
            # A footprint only writes its emitting identity to the extent the
            # separate region reader also supports that category. Disagreement
            # reduces observed mass, rather than relabelling the entire mask.
            category = identity['class_probability'][ids][:, ids].to(device)
            category = category / category.sum(1, keepdim=True).clamp_min(1e-12)
            strength = signal * category.diag()[None]
        else:
            strength = signal
        probability = strength / strength.sum(1, keepdim=True).clamp_min(1e-12)
        reliability = (strength if mode == 'union_verified_categories' else signal).amax(1, keepdim=True)
        op, _, visible = visible_operator(torch.load(view['contributions'], map_location='cpu'), observed, len(signal), False)
        weight = reliability * visible
        numerator += torch.sparse.mm(op.T, probability * weight)
        mass += torch.sparse.mm(op.T, weight)
        visibility += torch.sparse.mm(op.T, visible.float())
        covered += int((weight > 0).sum())
    value = (numerator + .1*visibility*prior_logits.sigmoid()) / (mass + .1*visibility).clamp_min(1e-8)
    logits = torch.where(mass > 0, torch.logit(value.clamp(1e-6, 1-1e-6)), prior_logits)
    return logits, dict(method=scene['source_category_fusion'], views=len(scene['source_evidence']['views']),
        complete_vocabulary_before_projection=True, covered_pixels=covered, prior_strength=.1,
        ground_truth_read=False, uncovered_pixels='unknown', source_union_identity=scene.get('source_union_identity'),
        semantic_scope_intersection=scene['source_category_fusion']=='union_verified_categories',
        region_probabilities_are_empirically_calibrated=False,
        identity_is_emitting_query_label=scene['source_category_fusion']=='union_emitting_categories')
