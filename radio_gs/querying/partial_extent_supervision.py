"""Source extent measurements with explicit positive, negative and unknown sets."""
import torch


from radio_gs.training.partial_scope_metrics import extent_measures


def source_extent_delta(state, kernel, device):
    residual = state['residual'].to(device)
    if state.get('interpolation', 'signed_kernel') == 'conditional_evidence':
        support = state['evidence_support'].to(device)
        return torch.cat([
            (r.float() @ kernel) / (s.float() @ kernel + state['ridge'])
            for r, s in zip(residual.split(8192), support.split(8192))
        ])
    return torch.cat([r.float() @ (state['inverse'].to(device) @ kernel)
                      for r in residual.split(8192)])
