"""Query-independent alternatives for native identity and physical scope.

Every observed region remains a hypothesis. Each hypothesis can receive one
geometrically corresponding observation per other view. Whole/part alternatives
are not collapsed by connected components or by semantic similarity. The sparse
matrix has hypotheses in rows, observations in columns; it is independent of
the carrier (mesh vertices or Gaussian elements) and of the query vocabulary.
"""
import numpy as np
from scipy import sparse


def compile_hypotheses(membership, visibility, region_frames):
    membership=membership.astype(np.float32).tocsc()
    visibility=visibility.astype(np.float32).tocsc()
    frames=np.asarray(region_frames)
    area=np.asarray(membership.sum(0)).reshape(-1)
    intersection=(membership.T@membership).tocsr()
    intersection.sort_indices()
    # coverage[r,v] is how much of region r was observable in source view v.
    coverage=(membership.T@visibility).toarray()
    rows=[];columns=[];weights=[]
    for seed in np.flatnonzero(area>0):
        lo,hi=intersection.indptr[seed:seed+2]
        regions=intersection.indices[lo:hi];shared=intersection.data[lo:hi]
        other=frames[regions]!=frames[seed]
        regions,shared=regions[other],shared[other]
        candidate_shared=coverage[regions,frames[seed]]
        seed_shared=coverage[seed,frames[regions]]
        iou=shared/np.maximum(candidate_shared+seed_shared-shared,1.)
        compatible=(candidate_shared>=.5*area[regions])&(seed_shared>=.5*area[seed])&(iou>=.5)
        regions,iou=regions[compatible],iou[compatible]
        rows.append(int(seed));columns.append(int(seed));weights.append(1.)
        for view in np.unique(frames[regions]):
            local=np.flatnonzero(frames[regions]==view)
            best=local[np.argmax(iou[local])]
            rows.append(int(seed));columns.append(int(regions[best]));weights.append(float(iou[best]))
    return sparse.csr_matrix((np.asarray(weights,np.float32),(rows,columns)),
                             shape=(len(area),len(area)))


def hypothesis_identity(hypotheses, identity):
    """Expected native cosine over compatible views, without renormalizing it.

    No view-count reward, learned projection, query softmax temperature or
    assumed negative evidence is introduced. Unmatched observations contribute
    nothing; an isolated supported hypothesis retains its native identity.
    """
    mass=np.asarray(hypotheses.sum(1)).reshape(-1)
    result=np.asarray(hypotheses@identity)
    denominator=mass if identity.ndim==1 else mass[:,None]
    return np.divide(result,denominator,out=np.zeros_like(result),where=denominator>0)


def compile_tracks(hypotheses, region_frames):
    """View-exclusive association, keeping incompatible granularities apart.

    Only mutual best physical correspondences can join components. A merge is
    rejected when it would assign two regions from one view to the same track.
    Missing overlap can be bridged through an intermediate view; native semantic
    similarity never creates a connection. No component or singleton is dropped.
    """
    frames=np.asarray(region_frames);size=len(frames)
    mutual=sparse.triu(hypotheses.minimum(hypotheses.T),k=1).tocoo()
    order=np.lexsort((mutual.col,mutual.row,-mutual.data))
    parent=np.arange(size);views=[{int(frame)} for frame in frames]
    def root(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return int(index)
    accepted=0;conflicts=0
    for index in order:
        a,b=root(mutual.row[index]),root(mutual.col[index])
        if a==b:continue
        if views[a]&views[b]:conflicts+=1;continue
        if len(views[a])<len(views[b]):a,b=b,a
        parent[b]=a;views[a].update(views[b]);accepted+=1
    supported=np.flatnonzero(np.diff(hypotheses.indptr)>0)
    roots=np.asarray([root(int(r)) for r in supported])
    _,rows=np.unique(roots,return_inverse=True)
    tracks=sparse.csr_matrix((np.ones(len(rows),np.float32),(rows,supported)),shape=(int(rows.max())+1,size))
    return tracks,dict(tracks=tracks.shape[0],multiview_tracks=int((np.diff(tracks.indptr)>1).sum()),
        mutual_edges=len(order),accepted_merges=accepted,rejected_view_conflicts=conflicts,
        no_semantic_edges=True,one_region_per_view=True,all_supported_regions_retained=True)


def anchor_track(tracks, membership, visibility, region_frames, anchor):
    """Select a physical track from the independent positive geometric input.

    A uniform Beta prior resolves finite-support ties without a tuned cutoff.
    Only tracks whose strict majority includes the supplied anchor are eligible.
    This is a view-consistency heuristic, not an independently calibrated
    posterior: adjacent source views are correlated.
    """
    positive=np.asarray(tracks@membership[anchor].toarray().reshape(-1)).reshape(-1)
    observed=visibility[anchor].toarray().reshape(-1)[region_frames]
    known=np.asarray(tracks@observed).reshape(-1)
    available=np.flatnonzero(positive>.5*known)
    if not len(available):return None,dict(eligible_tracks=0)
    value=(positive[available]+1)/(known[available]+2)
    chosen=int(available[np.argmax(value)])
    return chosen,dict(eligible_tracks=len(available),anchor_positive_views=float(positive[chosen]),
        anchor_observed_views=float(known[chosen]),anchor_track_score=float((positive[chosen]+1)/(known[chosen]+2)),
        anchor_track_rule='maximum Beta(1,1) smoothed positive fraction among strict-majority anchor-compatible tracks')
