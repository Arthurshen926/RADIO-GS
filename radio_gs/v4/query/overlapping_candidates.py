"""Query-conditioned support for overlapping object hypotheses, not a partition."""
import torch


def compose_overlapping_candidates(membership, query_support, *, chunk_size=32):
    """Preserve independently supported extents without pre-query competition.

    Max-product is a bounded evidence score, not a calibrated categorical
    posterior. Duplicated hypotheses do not accumulate mass, and an unselected
    hypothesis cannot suppress another. Multiple disjoint targets are supported.
    """
    membership = torch.as_tensor(membership, dtype=torch.float32).cpu()
    query_support = torch.as_tensor(query_support, dtype=torch.float32).cpu()
    if membership.ndim != 2 or query_support.ndim != 2 or membership.shape[1] != query_support.shape[1]:
        raise ValueError("candidate and query axes must align")
    if not membership.shape[1] or chunk_size <= 0:
        raise ValueError("candidate count and chunk size must be positive")
    for value in (membership, query_support):
        if not torch.isfinite(value).all() or bool(((value < 0) | (value > 1)).any()):
            raise ValueError("candidate support must be finite in [0,1]")
    output = torch.zeros(membership.shape[0], query_support.shape[0])
    for start in range(0, membership.shape[1], chunk_size):
        stop = start + chunk_size
        joint = membership[:, start:stop, None] * query_support[:, start:stop].T[None]
        output = torch.maximum(output, joint.max(1).values)
    return output, {"composition": "maximum_joint_candidate_support", "membership_semantics": "overlapping_hypotheses",
                    "score_semantics": "bounded_evidence_not_calibrated_probability", "pre_query_top2": False}
