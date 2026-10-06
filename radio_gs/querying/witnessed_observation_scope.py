"""Third-view containment evidence, separate from query identity.

Inspired by MaskClustering (CVPR 2024), not a reproduction of its panoptic
predictor or iterative clustering. Our overlapping proposals keep all native
identity candidates. A pair cannot witness its own association, and several
nested witnesses in one image still supply just one vote. Neighborhoods remain
seed-conditioned: no transitive connected-component merge is imposed.
"""
import numpy as np
from scipy import sparse
import torch

from .observation_readout import positive_support_rank


@torch.inference_mode()
def compile_witnesses(membership, visibility, frames, device='cuda'):
    membership = membership.astype(np.float32).tocsc()
    visibility = visibility.astype(np.float32).tocsc()
    frames = np.asarray(frames)
    area = np.asarray(membership.sum(0)).ravel()
    overlap = (membership.T @ membership).tocsr()
    coverage = (membership.T @ visibility).toarray()
    observed = (coverage >= .3 * area[:, None]) & (area[:, None] > 0)
    containment = []
    contained = np.zeros_like(observed)
    for view in range(visibility.shape[1]):
        regions = np.flatnonzero(frames == view)
        shared = overlap[:, regions].toarray()
        local = (shared >= .8 * coverage[:, view, None]) & observed[:, view, None]
        # Empty candidates must not become witnesses through 0 >= 0.
        local &= area[regions][None] > 0
        containment.append(local)
        contained[:, view] = local.any(1)
    foreign = np.arange(visibility.shape[1])[None] != frames[:, None]
    observed_foreign = observed & foreign
    counts = observed_foreign.sum(1)
    retained = (contained & observed_foreign).sum(1)
    # A possible whole-region witness must itself be preserved in other views.
    # This is a proposal-consistency check, not verified object completeness.
    witness_ok = (retained >= .8 * counts) & (counts >= 2)
    observers = torch.as_tensor(observed_foreign, dtype=torch.float32, device=device)
    known = observers @ observers.T
    support = torch.zeros_like(known)
    for view, local in enumerate(containment):
        regions = np.flatnonzero(frames == view)
        local = local[:, witness_ok[regions]].copy()
        local[frames == view] = False
        if local.shape[1]:
            values = torch.as_tensor(local, dtype=torch.float32, device=device)
            support += (values @ values.T > 0).float()
    accepted = (known >= 2) & (support >= .9 * known)
    accepted.fill_diagonal_(False)
    row, column = torch.where(accepted)
    quality = (support[row, column] / known[row, column]).cpu().numpy()
    graph = sparse.csr_matrix((quality, (row.cpu().numpy(), column.cpu().numpy())),
                              shape=(len(area), len(area)))
    supported = np.flatnonzero(area > 0)
    graph += sparse.csr_matrix((np.ones(len(supported), np.float32), (supported, supported)), shape=graph.shape)
    report = dict(regions=len(area), supported_regions=len(supported),
        witness_regions=int(witness_ok.sum()), directed_edges=int(len(row)),
        nontrivial_seeds=int((graph.getnnz(1) > 1).sum()),
        maximum_neighbors=int(graph.getnnz(1).max()),
        visibility_fraction=.3, containment_fraction=.8, consensus_fraction=.9,
        minimum_independent_views=2, witness_retention_fraction=.8,
        generating_views_excluded=True, votes_per_view=1,
        semantic_features_used=False, transitive_merge=False,
        all_identity_candidates_retained=True)
    return graph, report


class WitnessedObservationScope:
    def __init__(self, membership, visibility, frames, graph, rule):
        self.membership = membership.tocsc()
        self.anchor_membership = membership.tocsr()
        self.visibility = visibility.tocsc()
        self.frames = np.asarray(frames)
        self.graph = graph.tocsr()
        self.rule = rule
        self.area = np.asarray(membership.sum(0)).ravel()

    def infer(self, identity, semantic_rank, anchor=None):
        available = np.flatnonzero(self.area > 0)
        if anchor is None:
            seed = int(available[np.argmax(identity[available])])
            identity_rule = 'unchanged highest native identity score'
        else:
            regions = self.anchor_membership[anchor].indices
            if not len(regions):
                empty = np.zeros(self.membership.shape[0], np.float32)
                return empty, empty.astype(bool), dict(seed_region=None, anchor_regions=0)
            count = np.bincount(self.frames[regions], minlength=self.visibility.shape[1])
            votes = np.zeros(self.membership.shape[1], np.float32)
            votes[regions] = 1 / count[self.frames[regions]]
            agreement = np.asarray(self.graph @ votes).ravel()
            # Only an actually observed anchor-containing region may seed the
            # scope. Each view's nested proposals share one unit of anchor vote.
            seed = int(regions[np.argmax(agreement[regions])])
            identity_rule = 'anchor-containing seed with most view-balanced witnessed anchor support'
        regions = self.graph[seed].indices
        positive = np.zeros(self.membership.shape[0], np.float32)
        confidence = np.zeros(self.visibility.shape[1], np.float32)
        for view in np.unique(self.frames[regions]):
            local = regions[self.frames[regions] == view]
            union = np.asarray(self.membership[:, local].sum(1)).ravel() > 0
            positive += union
            confidence[view] = 1
        known = np.asarray(self.visibility @ confidence).ravel()
        fraction = np.divide(positive, known, out=np.zeros_like(positive), where=known > 0)
        selected = positive > 0 if self.rule == 'union' else fraction > .5
        # Union is an explicit set hypothesis, not a posterior. The ranking
        # still uses graded view support, and preserves the original native field.
        rank = fraction if anchor is not None else positive_support_rank(semantic_rank, fraction)
        return rank, selected, dict(seed_region=seed, seed_elements=int(self.area[seed]),
            matched_regions=regions.tolist(), matched_views=int(confidence.sum()),
            identity_rule=identity_rule, scope_rule='independent third-view containment neighborhood',
            selection_rule=self.rule, calibrated_membership=False,
            absent_scope_policy='not a verified negative; native ranking retained for semantic queries')
