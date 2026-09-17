"""Equal available origin votes within each source view, then equal view votes."""
import torch


def balance_source_views_and_origins(descriptors, weights, views, origins):
    if descriptors.ndim!=2 or weights.ndim!=2 or len(descriptors)!=len(weights) or views.shape!=(len(weights),) or origins.shape!=views.shape:
        raise ValueError('source evidence axes differ')
    if not torch.isfinite(weights).all() or (weights<0).any():raise ValueError('invalid source weights')
    result=descriptors.new_zeros(weights.shape[1],descriptors.shape[1]);view_count=weights.new_zeros(weights.shape[1])
    for view in views.unique():
        local=torch.zeros_like(result);origin_count=torch.zeros_like(view_count)
        selected_view=views==view
        for origin in origins[selected_view].unique():
            selected=selected_view&(origins==origin);w=weights[selected];mass=w.sum(0)
            local+=w.T@descriptors[selected]/mass[:,None].clamp_min(1e-8)
            origin_count+=(mass>1e-8)
        result+=local/origin_count[:,None].clamp_min(1)
        view_count+=(origin_count>0)
    return result/view_count[:,None].clamp_min(1),view_count
