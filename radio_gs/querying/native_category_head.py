"""Calibrated native category readout without a learned query-space adapter.

The small head is an explicit map asset. Its affine thresholds and category
competition change cross-category decisions, never the native field features.
"""
import torch
from torch.nn import functional as F


def export_head(scene):
    """Extract parameters, not predictions, from the retained source-trained head."""
    def load(path):
        return torch.load(path,map_location='cpu',weights_only=False)
    state=load(scene['base_field'])['state_dict']
    calibration=load(scene['calibration'])
    memory=load(scene['memory'])
    return dict(threshold_weight=state['native_query_thresholds.text.linear.weight'],
                threshold_bias=state['native_query_thresholds.text.linear.bias'],
                threshold_bounds=state['native_query_thresholds.text.bounds'],
                delta_weight=calibration['weight'],delta_bias=calibration['bias'],
                max_delta=calibration['max_delta'],logscale=state['readout_log_scale.text'],
                competition_direction=memory['category_competition']['direction'],
                geometry=scene['geometry'],provenance=dict(base_field=scene['base_field'],
                calibration=scene['calibration'],competition_memory=scene['memory']),
                contract='source-trained parameter transfer; no historical query predictions')


def calibrated_category_logits(cosine,query,head):
    device=cosine.device
    q=F.normalize(query.float(),dim=-1)
    lower,upper=head['threshold_bounds'].to(device)
    threshold=(q@head['threshold_weight'].to(device).T+head['threshold_bias'].to(device)).flatten().clamp(lower,upper)
    threshold+=(q@head['delta_weight'].to(device)+head['delta_bias'].to(device)).clamp(-head['max_delta'],head['max_delta'])
    competition=(q@head['competition_direction'].float().to(device)).tanh()
    return head['logscale'].to(device).clamp(-4,6).exp()*(cosine-threshold)+competition
