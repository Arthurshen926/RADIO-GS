"""Compare complete frozen responses using query evidence on common views.

Keep the joint-teacher answer as candidate zero. Each existing teacher adds at
most its own identity winner, decoded by the same complete membership provider.
Verification selects a whole response; it does not move its anchor, alter its
membership, fuse feature coordinates, or calibrate a foreground probability.
"""
import numpy as np
from scipy.stats import rankdata


def recover_second_teacher_ranks(native, fused, available):
    """Invert the recorded equal-rank fusion, checking its half-integer lattice.

    This recovers DINO rank information, never its missing cosine magnitudes.
    Average ranks have integer or half-integer values; float32 fusion error is
    small enough to recover that lattice for the retained regional populations.
    """
    native, fused = np.asarray(native), np.asarray(fused)
    n = len(available)
    first = rankdata(native[available])
    raw = 2*n*fused[available].astype(np.float64)-first
    second = np.rint(2*raw)/2
    error = float(np.max(np.abs(raw-second)))
    if error >= .125 or not np.array_equal(rankdata(second),second):
        raise ValueError('cache is not invertible equal-average native rank fusion')
    reconstructed = (first+second)/(2*n)
    if not np.allclose(reconstructed,fused[available],rtol=0,atol=1e-7):
        raise ValueError('teacher rank recovery does not replay fused identity')
    out = np.zeros(len(native));out[available]=second
    return out,dict(max_rank_lattice_error=error,second_teacher_cosine_recovered=False)


class CompleteResponseVerifier:
    def __init__(self,membership,visibility,frames):
        self.membership=membership.tocsc();self.visibility=visibility.tocsc()
        self.frames=np.asarray(frames)
        self.area=np.asarray(self.membership.sum(0)).ravel()
        self.available=np.flatnonzero(self.area>0)
        if self.frames.shape != self.area.shape or visibility.shape[0] != membership.shape[0]:
            raise ValueError('unaligned observation domains')

    def verify(self,selections,seeds,native,second_ranks,*,permuted=False):
        if len(selections)!=len(seeds) or not len(seeds):raise ValueError('candidate zero is required')
        if any(np.asarray(s).shape!=(self.membership.shape[0],) for s in selections):
            raise ValueError('complete response selections must share the carrier')
        if not np.isfinite(native).all() or not np.isfinite(second_ranks).all():
            raise ValueError('query evidence must be finite')
        # The UNION of all proposal views is excluded for every candidate.
        excluded=set(self.frames[np.asarray(seeds)].tolist())
        overlap=[]
        for selected in selections:
            selected=np.asarray(selected,dtype=float)
            intersection=np.asarray(self.membership.T@selected).ravel()
            visible=np.asarray(self.visibility.T@selected).ravel()[self.frames]
            union=self.area+visible-intersection
            overlap.append(np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0))
        overlap=np.stack(overlap)
        per_view=[];common=[];query_budget_error=0.;rng=np.random.default_rng(20261016)
        for frame in range(self.visibility.shape[1]):
            if frame in excluded:continue
            ids=self.available[self.frames[self.available]==frame]
            if len(ids)<2:continue
            positive=overlap[:,ids];negative=1-positive
            pm=positive.sum(1);nm=negative.sum(1)
            # Compare the hypotheses on identical, informative view domains.
            if not ((pm>0)&(nm>0)).all():continue
            query=(rankdata(native[ids])+rankdata(second_ranks[ids]))/(2*len(ids))
            if permuted:
                shuffled=query[rng.permutation(len(ids))]
                query_budget_error=max(query_budget_error,float(np.abs(np.sort(query)-np.sort(shuffled)).max()))
                query=shuffled
            contrast=positive@query/pm-negative@query/nm
            per_view.append(contrast);common.append(frame)
        scores=np.mean(per_view,axis=0) if per_view else np.zeros(len(seeds))
        winner=int(np.argmax(scores))  # Stable ties, including no evidence, keep the original.
        return winner,dict(scores=scores.tolist(),common_views=common,
            excluded_proposal_views=sorted(excluded),query_budget_error=query_budget_error,
            permutation_seed=20261016 if permuted else None,selected_candidate=winner,
            baseline_retained=winner==0,no_common_evidence=not bool(per_view),
            confidence_is_calibrated_probability=False,whole_candidate_preserved=True)
