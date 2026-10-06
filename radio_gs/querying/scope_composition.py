"""Explain a query response jointly with nonnegative, overlapping scope factors.

Unlike independent scope averaging or maximum union, coefficients compete to
explain the same evidence. Geometry and native query coordinates remain fixed.
"""
import torch


class ScopeComposition:
    def __init__(self, payload, device='cuda', steps=80, fixed_cores=False):
        indices = payload['indices'].to(device)
        weights = payload['weights'].to(device)
        self.matrix = torch.sparse_coo_tensor(indices, weights, payload['shape']).coalesce()
        self.transpose = self.matrix.transpose(0, 1).coalesce()
        self.coverage = torch.sparse.sum(self.matrix, dim=1).to_dense()
        self.steps = steps
        self.fixed_cores = fixed_cores

    @torch.no_grad()
    def refine(self, scores, query, model, modality):
        observed = model.observed_for(modality)[:, None].float()
        results = []
        for start in range(0, scores.shape[1], 8):
            unary = scores[:, start:start+8]
            fidelity = (1+8*(2*unary-1).square())*observed
            # Diagonal majorizer of M^T W M; valid for nonnegative M. Unknown
            # rows have zero evidence weight, rather than a background label.
            majorizer = torch.sparse.mm(self.transpose, fidelity*self.coverage[:, None]).clamp_min(1e-8)
            mass = torch.sparse.mm(self.transpose, fidelity)
            coefficients = torch.zeros_like(mass)
            for _ in range(self.steps):
                reconstruction = torch.sparse.mm(self.matrix, coefficients)
                gradient = torch.sparse.mm(self.transpose, fidelity*(unary-reconstruction))
                # A small per-unit-support cost discourages redundant factors.
                coefficients.add_(.95*(gradient-.01*mass)/majorizer).clamp_(min=0)
            reconstruction = torch.sparse.mm(self.matrix, coefficients).clamp(0, 1)
            value = (fidelity*unary+reconstruction)/(fidelity+1)
            if self.fixed_cores:
                value = torch.where((unary<=.25)|(unary>=.75), unary, value)
            results.append(torch.where((self.coverage[:, None]>0)&observed.bool(), value, unary))
        return torch.cat(results, 1)
