"""Source-only identity anchor followed by geometrically compatible extent."""
import torch


@torch.no_grad()
def anchored_fragment_extent(positive, similarity, views, *, identity_margin=0.03,
                             minimum_containment=0.5, evidence_threshold=0.05):
    positive = torch.as_tensor(positive, dtype=torch.float32).cpu()
    similarity = torch.as_tensor(similarity, dtype=torch.float32).cpu()
    views = torch.as_tensor(views, dtype=torch.long).cpu()
    if positive.ndim != 2 or similarity.ndim != 2 or similarity.shape[1] != positive.shape[0] or views.shape != (positive.shape[0],):
        raise ValueError("fragment, query and view axes differ")
    if not 0 <= minimum_containment <= 1 or identity_margin < 0 or not 0 < evidence_threshold < 1:
        raise ValueError("invalid anchor controls")
    if not torch.isfinite(positive).all() or not torch.isfinite(similarity).all() or bool(((positive < 0) | (positive > 1)).any()):
        raise ValueError("fragment evidence and similarities must be finite and valid")
    binary = positive >= evidence_threshold
    mass = binary.sum(-1)
    if not bool((mass > 0).any()):
        return torch.zeros(positive.shape[1], similarity.shape[0]), {"selected_fragments": [[] for _ in similarity]}
    ranked = similarity.masked_fill((mass == 0)[None], -torch.inf)
    anchors = ranked.argmax(-1)
    output, selections = [], []
    for query, anchor in enumerate(anchors.tolist()):
        intersection = (binary & binary[anchor]).sum(-1)
        containment = intersection / torch.minimum(mass, mass[anchor]).clamp_min(1)
        valid = (containment >= minimum_containment) & (mass > 0) & (ranked[query] >= ranked[query, anchor] - identity_margin)
        selected = [anchor]
        for view in torch.unique(views).tolist():
            if view == int(views[anchor]):
                continue
            ids = torch.where(valid & (views == view))[0]
            if ids.numel():
                selected.append(int(ids[ranked[query, ids].argmax()]))
        output.append(positive[selected].max(0).values)
        selections.append(selected)
    return torch.stack(output, -1), {"anchor_fragments": anchors.tolist(), "selected_fragments": selections,
        "semantics": "single_identity_anchor_multi_view_extent_not_calibrated_probability",
        "identity_margin": identity_margin, "minimum_containment": minimum_containment,
        "query_independent_memory_unchanged": True, "supports_multiple_distinct_instances": False}
