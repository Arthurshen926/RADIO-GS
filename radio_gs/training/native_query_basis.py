"""Training-input-only low-rank query basis; no semantic labels are consumed."""
import torch
from torch.nn import functional as F


def native_query_basis(queries, dimensions):
    if queries.ndim!=2 or not torch.isfinite(queries).all() or (queries.norm(dim=1)<=0).any():
        raise ValueError('invalid native query evidence')
    if not 1<=dimensions<=min(queries.shape):raise ValueError('basis exceeds available native axes')
    # Uncentred SVD preserves the encoder's cosine origin. A centred PCA would
    # require an explicit bias and a different inference contract.
    _,singular,right=torch.linalg.svd(F.normalize(queries.detach().float(),dim=-1),full_matrices=False)
    return right[:dimensions].contiguous(),float(singular[:dimensions].square().sum()/singular.square().sum())
