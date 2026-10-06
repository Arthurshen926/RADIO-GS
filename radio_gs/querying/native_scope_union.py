"""Native-space semantic matching with max composition of overlapping scopes."""
import torch
from torch.nn import functional as F


class NativeScopeUnion:
    def __init__(self, graph, device='cuda'):
        self.indices = graph['indices'].to(device)
        self.weights = graph['weights'].to(device)
        self.shape = graph['shape']
        self.prototypes = F.normalize(graph['prototypes'].to(device).float(), dim=1)
        self.coverage = torch.zeros(self.shape[0], device=device).scatter_reduce_(
            0, self.indices[0], self.weights, reduce='amax', include_self=True)

    def response(self, query, log_scale, threshold):
        cosine = self.prototypes @ F.normalize(query.float(), dim=-1).T
        probability = (log_scale.exp() * (cosine - threshold)).sigmoid()
        columns = []
        for start in range(0, query.shape[0], 4):
            p = probability[:, start:start+4]
            evidence = self.weights[:, None] * p[self.indices[1]]
            columns.append(torch.zeros(self.shape[0], p.shape[1], device=query.device).scatter_reduce(
                0, self.indices[0, :, None].expand_as(evidence), evidence, reduce='amax', include_self=True))
        return torch.cat(columns, 1)


def attach_native_scope_union(model, payload):
    field = NativeScopeUnion(payload['scope_hypergraph'])
    original = model.logits
    calibration = payload['native_scope_union']
    def logits(query, modality, rows=None):
        if modality not in calibration:
            return original(query, modality, rows)
        base = original(query, modality)
        parameters = calibration[modality]
        response = field.response(query, torch.tensor(parameters['log_scale'], device=query.device), parameters['threshold'])
        # Local scopes own only their represented fraction; preserve unmatched
        # field evidence rather than treating absent range support as negative.
        probability = response + (1-field.coverage[:, None])*base.sigmoid()
        result = torch.logit(probability.clamp(1e-7, 1-1e-7))
        return result if rows is None else result[rows]
    model.logits = logits
    return model
