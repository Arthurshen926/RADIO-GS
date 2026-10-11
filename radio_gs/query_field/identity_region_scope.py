"""Experimental scope admission conditioned on the current native text identity.

Instance purity does not establish that an extension describes the requested
part, attribute or union. Require the existing native positive identity test
before the unchanged instance-quality admission and per-view competition.
"""
import numpy as np
import torch
from .region_carrier import RegionScopeDecoder
from .region_admission import admit_regions
from .region_carrier import query_text_region_scope
from .scope_readout import ScopeReadout


class IdentityRegionScopeDecoder(RegionScopeDecoder):
    @torch.no_grad()
    def scope(self, probability):
        f = self.features; c = self.carrier
        if probability.shape != (len(f.area),) or not bool(torch.isfinite(probability).all()):
            raise ValueError('unaligned region identity')
        native = probability.detach().cpu().numpy()
        eligible = np.flatnonzero((f.area > 0) & (native > .5))
        if not len(eligible): return None, dict(applied=False, reason='no native positive region')
        seed = int(eligible[np.argmax(native[eligible])])
        ids, features, shared = f.pairs(seed)
        positive = native[ids] > .5
        rejected = int((~positive).sum()); total = len(ids)
        ids, features, shared = ids[positive], features[positive], shared[positive]
        detail = dict(seed=seed, candidate_pairs=total, native_eligible_pairs=len(ids), native_rejected_pairs=rejected,
                      identity_qualified=True, source_access=False, query_fitting=False)
        device = c.frames.device
        quality = c.quality(torch.from_numpy(features).to(device))
        result = admit_regions(c.quality, quality, seed=seed, candidates=torch.from_numpy(ids).to(device),
            intersection=torch.from_numpy(shared).to(device), area=torch.from_numpy(f.area).to(device),
            frames=c.frames, view_count=int(c.view_count))
        if result is None: return None, dict(detail, applied=False, reason='no identity-qualified quality admission')
        weights, confidence, chosen = result
        foreground = np.asarray(f.membership @ weights.cpu().numpy()).ravel()
        known = np.asarray(f.visibility @ confidence.cpu().numpy()).ravel()
        scope = np.divide(foreground, known, out=np.zeros_like(foreground), where=known > 0).clip(0, 1)
        return (torch.from_numpy(scope), torch.from_numpy(known > 0)), dict(detail, applied=True,
            admitted=chosen.cpu().tolist(), views=int((confidence > 0).sum()))


@torch.no_grad()
def query_text_identity_scope(field, query, negatives, *, region_decoder=None,
                              exclusion_decoder=None, native_reader=None):
    decoder = region_decoder or IdentityRegionScopeDecoder(field.region_carrier, field.xyz)
    if not isinstance(decoder, IdentityRegionScopeDecoder):
        raise ValueError('identity-qualified scope decoder required')
    values, detail = query_text_region_scope(field, query, negatives, region_decoder=decoder,
        exclusion_decoder=exclusion_decoder, native_reader=native_reader)
    member = torch.maximum(values['seed_addition'], values['conditional_addition'])
    accepted = torch.zeros_like(field.region_carrier.frames, dtype=torch.bool)
    accepted[detail.get('admitted', [])] = True
    if 'seed' in detail: accepted[detail['seed']] = True
    return ScopeReadout(values['semantic_initial'], member, member > .5, accepted,
        dict(detail, seed_preserved=True, experimental=True, default_promoted=False))
