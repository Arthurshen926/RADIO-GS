"""Single-pass semantic reliability; confidence/agreement are not correctness.

The held-out-view mode excludes every region from the current source view when
forming corroborating evidence. It neither iterates nor reads semantic labels.
"""
import math
import torch


@torch.inference_mode()
def reliable_observation_response(transfer, response, base, field, mode):
    if mode == 'entropy_information':
        entropy = -(response * response.clamp_min(torch.finfo(response.dtype).tiny).log()).sum(1)
        reliability = (1 - entropy / math.log(response.shape[1])).clamp(0, 1)
        corroborated = None
    elif mode == 'heldout_view_support':
        offsets = field['region_offsets'].to(response.device).long()
        coo = transfer.to_sparse_coo().coalesce()
        rows, columns = coo.indices()
        values = coo.values()
        views = torch.bucketize(columns, offsets[1:], right=True)
        pairs = torch.unique(views * transfer.shape[0] + rows)
        view_count = torch.bincount(pairs % transfer.shape[0], minlength=transfer.shape[0])
        other_view = (view_count > 1)[:, None]
        order = views.argsort()
        cuts = torch.cat([torch.zeros(1, device=response.device, dtype=torch.long),
                          torch.bincount(views, minlength=len(offsets)-1).cumsum(0)]).cpu().tolist()
        rows, columns, values = rows[order], columns[order], values[order]
        reliability = torch.ones(len(response), device=response.device)
        corroborated = torch.zeros(len(response), device=response.device, dtype=torch.bool)
        bounds = offsets.cpu().tolist()
        for v, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            lo, hi = cuts[v], cuts[v+1]
            if start == end or lo == hi:
                continue
            view = torch.sparse_coo_tensor(torch.stack([rows[lo:hi], columns[lo:hi]-start]),
                values[lo:hi], (transfer.shape[0], end-start)).coalesce()
            own = torch.sparse.mm(view, response[start:end])
            own_mass = torch.sparse.mm(view, torch.ones(end-start, 1, device=response.device))
            external_mass = torch.sparse.mm(view.T, (1-own_mass).clamp_min(0) * other_view)
            external = torch.sparse.mm(view.T, (base-own).clamp_min(0) * other_view)
            # Normalize the external categorical distribution; regions with no
            # other-view mass retain neutral weight, rather than becoming negative.
            distribution = external / external.sum(1, keepdim=True).clamp_min(1e-12)
            agreement = (response[start:end] * distribution).clamp_min(0).sqrt().sum(1).clamp(0, 1)
            supported = (external_mass[:,0] > 1e-8) & (external.sum(1) > 1e-8)
            reliability[start:end] = torch.where(supported, agreement, torch.ones_like(agreement))
            corroborated[start:end] = supported
    else:
        raise ValueError(mode)
    mass = torch.sparse.mm(transfer, reliability[:,None])
    numerator = torch.sparse.mm(transfer, reliability[:,None] * response)
    probability = torch.where(mass > 0, numerator / mass.clamp_min(torch.finfo(mass.dtype).tiny), base)
    return probability, dict(mode=mode, mean_region_weight=float(reliability.mean()),
        min_region_weight=float(reliability.min()),
        externally_supported_regions=int(corroborated.sum()) if corroborated is not None else None,
        region_count=len(response), passes=1, semantic_labels_read=False,
        agreement_is_not_ground_truth=True)
