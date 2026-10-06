"""Late native-space query pooling over unmerged source-region observations."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F

@torch.no_grad()
def query_region_bank(manifest_path, queries, *, pooling='logmeanexp', inverse_temperature=20.):
    """Return Gaussian query scores/support; no query-specific map state is saved.

    Memberships are per-view alpha-normalized Gaussian supports. Normalize
    within each view before visibility-weighted integration across views.
    The bank stores descriptors and supports, not a fixed text vocabulary.
    """
    manifest=json.loads(Path(manifest_path).read_text())
    queries=F.normalize(queries,dim=1)
    numerator=queries.new_zeros((manifest['gaussians'],len(queries)))
    denominator=queries.new_zeros((manifest['gaussians'],1))
    for entry in manifest['entries']:
        row=torch.load(entry['path'],map_location='cpu',weights_only=False)
        ids=row['ids'].to(queries.device)
        descriptors=F.normalize(row['descriptors'].to(queries.device).float(),dim=1)
        operator=torch.sparse_coo_tensor(torch.stack([row['rows'].long(),row['columns'].long()]).to(queries.device),row['values'].to(queries.device),(len(ids),len(descriptors))).coalesce()
        total=torch.sparse.mm(operator,queries.new_ones((len(descriptors),1)))
        cosine=descriptors@queries.T
        if pooling=='mean':
            response=torch.sparse.mm(operator,cosine)/total.clamp_min(1e-8)
        elif pooling=='logmeanexp':
            center=cosine.max(0,keepdim=True).values
            exponential=torch.exp(inverse_temperature*(cosine-center))
            response=torch.log((torch.sparse.mm(operator,exponential)/total.clamp_min(1e-8)).clamp_min(1e-30))/inverse_temperature+center
        elif pooling=='scope_max':
            # Preserve absolute regional coverage instead of conditioning it away.
            # Zero is the no-positive-regional-evidence response.
            contribution=operator.values()[:,None]*cosine[operator.indices()[1]]
            response=queries.new_zeros((len(ids),len(queries)))
            response.scatter_reduce_(0,operator.indices()[0][:,None].expand_as(contribution),contribution,reduce='amax',include_self=True)
        else:
            raise ValueError(pooling)
        mass=row['mass'].to(queries.device)[:,None]*(total>0)
        numerator.index_add_(0,ids,response*mass)
        denominator.index_add_(0,ids,mass)
    return numerator/denominator.clamp_min(1e-8),denominator>0
