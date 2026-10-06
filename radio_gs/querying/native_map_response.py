"""Native semantic response only; no inherited query scores or subset inference."""
import math

import torch
from torch.nn import functional as F


@torch.inference_mode()
def native_map_response(field, encodings, device, chunk_size=8192):
    # Preserve the stored fp16 feature contract used by the original write.
    query = F.normalize(encodings.half().float().to(device), dim=1)
    response, cosine = [], []
    support = field['supported'].reshape(-1, 1)
    for start in range(0, len(field['features']), chunk_size):
        end = start + chunk_size
        value = field['features'][start:end].float().to(device) @ query.T
        known = support[start:end].to(device)
        cosine.append((value * known).cpu())
        response.append(((math.exp(field['log_scale']) * value + field['bias']).sigmoid() * known).cpu())
    return torch.cat(cosine), torch.cat(response)
